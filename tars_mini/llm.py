"""v0.4 lowering of Qwen3.5 (Gated DeltaNet "LA" and gated full-attention "FA" layers, the final
norm and the tied-embedding head) to TARS-mini descriptors, single core, with every state in DDR
(conv history, DeltaNet states, KV cache). Qwen3 (cfg.model_type "qwen3": every layer full
attention without the output gate, RMSNorm weights used as they are, full rotary) lowers the same way.

    low = QwenLowering(cfg, weights, max_tokens=64, head=True)
    low.input(hidden)                           # [T, H] fp32 input embeddings (the host's lookup: low.embed(ids))
    low.chunk(0, 16)                            # prefill: tokens 0..15 through every layer, the norm, the head
    low.chunk(16, 1)                            # decode: token 16
    ddr, q, n = low.prog.build()                # one queue; Machine(ddr).run_queue(q, n)
    low.read_layer(machine, layer, t0, n)       # a layer's output rows; read_final, read_logits

The layers follow cfg.layer_types. Every linear layer is W8A8: ROW_QUANT (per-token INT8 + row
scales) then GEMM with QM_ROWCOL; the head is the same on the INT8 embedding table in equal row
blocks of at most 65535 rows (N is a 16-bit field: 248320 = 4 x 62080, 151936 = 4 x 37984). Every other step is one of the v0.4 fp32 ops (ELT, ROPE, FDOT,
row ops), so the model is data on the device; the host lays out weights, queues descriptors, looks
up the input embedding and picks the next token from the logits. The recurrence runs in its
token-by-token form for prefill and decode alike. A layer's temporaries live in a scratch region
reused by the next layer (the queue runs in order), so DDR holds the weights, the states and one
layer's scratch.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import contract as C
from .compiler import Program, quantize_weight

F32 = np.float32


@dataclass
class V:
    """A matrix in DDR: offset from the program base, shape, row stride (bytes), element format."""
    off: int
    rows: int
    cols: int
    ld: int
    fmt: int = C.OUT_FP32

    def rowv(self, r0: int, n: int) -> "V":
        return replace(self, off=self.off + r0 * self.ld, rows=n)

    def colv(self, c0: int, n: int) -> "V":
        return replace(self, off=self.off + c0 * C.ESIZE[self.fmt], cols=n)

    def heads(self, hd: int) -> "V":
        """Rows of hd elements: a dense [T, H*hd] tensor viewed as [T*H, hd]."""
        assert self.ld == self.cols * C.ESIZE[self.fmt] and self.cols % hd == 0
        return V(self.off, self.rows * self.cols // hd, hd, hd * C.ESIZE[self.fmt], self.fmt)


def _ld16(ld: int, rowbytes: int) -> int:
    if ld == rowbytes:
        return 0
    assert ld % 16 == 0 and ld > rowbytes
    return ld // 16


class Dev(Program):
    """Descriptor emitters for the v0.4 ops on V views (offsets relative to the program base)."""

    def mat(self, rows: int, cols: int, fmt: int = C.OUT_FP32) -> V:
        es = C.ESIZE[fmt]
        return V(self.alloc(rows * cols * es), rows, cols, cols * es, fmt)

    def put(self, a: np.ndarray, dt: str = "<f4") -> int:
        return self._put(np.asarray(a), dt)

    def const(self, a: np.ndarray, fmt: int = C.OUT_FP32) -> V:
        a = np.asarray(a, F32).reshape(1, -1) if np.ndim(a) < 2 else np.asarray(a, F32)
        if fmt == C.OUT_FP16:
            from .numerics import to_fp16_bits
            return V(self.put(to_fp16_bits(a), "<u2"), a.shape[0], a.shape[1], 2 * a.shape[1], fmt)
        assert fmt == C.OUT_FP32
        return V(self.put(a), a.shape[0], a.shape[1], 4 * a.shape[1])

    # -- row ops
    def rowop(self, x: V, fn: int, y: V, gamma: V | None = None, eps: float = 0.0, start: int = 0,
              scale_out: int = 0) -> None:
        assert x.cols == y.cols and x.rows == y.rows
        assert x.fmt == C.OUT_FP32 and y.fmt != C.OUT_FP16
        self._emit(OPCODE=C.OP_ROW, CTRL=fn, EPI=C.epi(C.ACT_NONE, y.fmt),
                   A_OFF=x.off, Y_OFF=y.off, M=x.rows, N=x.cols, K=start,
                   SCALE_OFF=gamma.off if gamma is not None else scale_out,
                   W_OFF=C.f32_bits(eps) if gamma is not None else 0,
                   LDA16=_ld16(x.ld, 4 * x.cols), LDY16=_ld16(y.ld, y.cols * C.ESIZE[y.fmt]))

    def norm(self, x: V, gamma: V, eps: float, y: V, l2: bool = False) -> None:
        self.rowop(x, C.ROW_L2NORM if l2 else C.ROW_RMSNORM, y, gamma, eps)

    def quant(self, x: V) -> tuple[V, int]:
        """Per-token INT8 rows and their row scales (fp32, 16-byte aligned)."""
        q = self.mat(x.rows, x.cols, C.OUT_INT8)
        s = self.alloc(4 * x.rows)
        self.rowop(x, C.ROW_QUANT, q, scale_out=s)
        return q, s

    def softmax_causal(self, x: V, y: V, first_valid: int) -> None:
        """Row m keeps elements [0, first_valid + m)."""
        self.rowop(x, C.ROW_SOFTMAX, y, start=first_valid)

    # -- matrix ops
    def gemm_rc(self, xq: V, srow: int, w: "QLin", y: V, act: int = C.ACT_NONE) -> None:
        """y = act((xq @ Wq.T) * srow[m] * scol[n] + b): W8A8 with per-token and per-channel scales."""
        assert xq.fmt == C.OUT_INT8 and xq.cols == w.k and y.cols == w.n and srow % 16 == 0
        assert y.fmt != C.OUT_FP16
        self._emit(OPCODE=C.OP_GEMM, CTRL=C.ctrl(C.QM_ROWCOL, fusion=int(act != C.ACT_NONE)),
                   EPI=C.epi(act, y.fmt, int(w.b_off is not None)), A_OFF=xq.off, W_OFF=w.w_off,
                   SCALE_OFF=w.s_off, BIAS_OFF=w.b_off or 0, Y_OFF=y.off, M=xq.rows, N=w.n, K=w.k,
                   HDR_RSV=srow // 16, LDA16=_ld16(xq.ld, xq.cols), LDY16=_ld16(y.ld, y.cols * C.ESIZE[y.fmt]))

    def fdot(self, a: V, b: V, y: V, nn: bool) -> None:
        """NT: y = a @ b.T (b [N][K]); NN: y = a @ b (b [K][N])."""
        K = a.cols
        N = b.cols if nn else b.rows
        assert (b.rows if nn else b.cols) == K and y.cols == N and y.rows == a.rows
        assert a.fmt == C.OUT_FP32 and y.fmt != C.OUT_FP16
        assert b.fmt == C.OUT_FP32 or (nn and b.fmt == C.OUT_FP16)
        brow = C.ESIZE[b.fmt] * b.cols
        self._emit(OPCODE=C.OP_FDOT, CTRL=int(nn), EPI=C.epi(C.ACT_NONE, y.fmt), A_OFF=a.off, W_OFF=b.off,
                   Y_OFF=y.off, M=a.rows, N=N, K=K, GROUP_SIZE=_ld16(b.ld, brow),
                   HDR_RSV=C.STATE_FP16 if b.fmt == C.OUT_FP16 else 0,
                   LDA16=_ld16(a.ld, 4 * a.cols), LDY16=_ld16(y.ld, y.cols * C.ESIZE[y.fmt]))

    def elt(self, op: int, a: V, b: V, y: V, bmode: int = C.BM_DENSE, arow: bool = False,
            act: int = C.ACT_NONE, c: V | None = None) -> None:
        M, N = y.rows, y.cols
        assert a.fmt in (C.OUT_FP32, C.OUT_FP16) and b.fmt == C.OUT_FP32
        assert c is None or (c.fmt == y.fmt and c.ld == y.ld)
        assert op != C.ELT_MULADD or y.fmt in (C.OUT_FP32, C.OUT_FP16)
        g = _ld16(b.ld, 4 * N) if bmode == C.BM_DENSE else 0
        self._emit(OPCODE=C.OP_ELT, CTRL=op, EPI=C.epi(act, y.fmt, int(c is not None)),
                   A_OFF=a.off, W_OFF=b.off, Y_OFF=y.off, BIAS_OFF=c.off if c is not None else 0,
                   M=M, N=N, HDR_RSV=bmode | int(arow) << 2 | (C.STATE_FP16 if a.fmt == C.OUT_FP16 else 0),
                   GROUP_SIZE=g, LDA16=0 if arow else _ld16(a.ld, C.ESIZE[a.fmt] * N),
                   LDY16=_ld16(y.ld, N * C.ESIZE[y.fmt]))

    def act(self, x: V, act: int, y: V) -> None:
        assert x.fmt == C.OUT_FP32 and y.fmt != C.OUT_FP16
        self._emit(OPCODE=C.OP_RELU, CTRL=0, EPI=C.epi(act, y.fmt), A_OFF=x.off, Y_OFF=y.off,
                   M=x.rows, N=x.cols, LDA16=_ld16(x.ld, 4 * x.cols),
                   LDY16=_ld16(y.ld, y.cols * C.ESIZE[y.fmt]))

    def rope(self, x: V, tab: V, rows_per_pos: int, rot: int, y: V) -> None:
        assert x.fmt == tab.fmt == C.OUT_FP32 and y.fmt != C.OUT_FP16
        self._emit(OPCODE=C.OP_ROPE, CTRL=0, EPI=C.epi(C.ACT_NONE, y.fmt), A_OFF=x.off, W_OFF=tab.off,
                   Y_OFF=y.off, M=x.rows, N=x.cols, K=rot, GROUP_SIZE=rows_per_pos,
                   LDA16=_ld16(x.ld, 4 * x.cols), LDY16=_ld16(y.ld, y.cols * C.ESIZE[y.fmt]))


@dataclass
class QLin:
    w_off: int
    s_off: int
    b_off: int | None
    n: int
    k: int


def pad16(n: int) -> int:
    return -(-n // 16) * 16


GEMV_K = 4096                                 # the KV260 GEMV unit's longest x (a longer K is rejected)


def input_embedding(cfg, rows: np.ndarray) -> np.ndarray:
    """Embedding rows as the model consumes them (fp32). Gemma 3 scales them by sqrt(hidden), as
    transformers does with fp32 weights: row * fp32(sqrt(H)), one fp32 rounding."""
    rows = np.asarray(rows, F32)
    if getattr(cfg, "model_type", "") != "gemma3_text":
        return rows
    return (rows * F32(cfg.hidden_size ** 0.5)).astype(F32)


class QwenLowering:
    def __init__(self, cfg, weights: dict[str, np.ndarray], max_tokens: int, ddr_bytes: int = 1 << 27,
                 head: bool = False, queue_cap: int = 0, scratch_bytes: int = 1 << 22, verify_rows: int = 1,
                 mtp: bool = False, state_dtype: str = "fp32"):
        if state_dtype not in ("fp32", "fp16"):
            raise ValueError("state_dtype must be fp32 or fp16")
        self.state_dtype = state_dtype
        self.state_fmt = C.OUT_FP16 if state_dtype == "fp16" else C.OUT_FP32
        self.cfg, self.w, self.T = cfg, weights, max_tokens
        self.mtp = mtp                           # Qwen3.5's MTP head (weights["mtp.*"]): mtp_chunk drafts a token
        self.verify_rows = verify_rows           # logits rows a chunk can return (speculative decoding)
        self.p = self.prog = Dev(ddr_bytes=ddr_bytes)
        c = cfg
        self.H, self.eps = c.hidden_size, c.rms_norm_eps
        self.q3 = getattr(c, "model_type", "") == "qwen3"
        # Gemma 3: every layer full attention without the output gate (a sliding window no shorter
        # than the context changes nothing), four norms per layer, GeGLU, a RoPE base per layer type
        self.gemma = getattr(c, "model_type", "") == "gemma3_text"
        self.types = list(c.layer_types)
        assert not self.q3 or set(self.types) == {"full_attention"}
        if self.gemma:
            assert set(self.types) <= {"sliding_attention", "full_attention"}
            assert c.attn_logit_softcapping is None and c.final_logit_softcapping is None
            assert c.hidden_activation == "gelu_pytorch_tanh"
            assert max_tokens <= c.sliding_window, "the context must fit the sliding window"
            assert all(r.get("rope_type", "default") == "default" for r in c.rope_parameters.values())
        # LA dims
        self.nk, self.nv = getattr(c, "linear_num_key_heads", 0), getattr(c, "linear_num_value_heads", 0)
        self.dk, self.dv = getattr(c, "linear_key_head_dim", 0), getattr(c, "linear_value_head_dim", 0)
        assert self.nk == self.nv, "the GQA repeat of linear attention is not lowered"
        # FA dims
        self.nh, self.nkv, self.hd = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        self.rot = self.hd if self.gemma else int(self.hd * c.rope_parameters.get("partial_rotary_factor", 1.0))
        # the attention scaling, folded into q_norm's gamma (1/16 for both models: exact)
        self.qmul = c.query_pre_attn_scalar ** -0.5 if self.gemma else self.hd ** -0.5
        self.Tp = pad16(max_tokens)
        self.head = head
        self.taps = getattr(c, "linear_conv_kernel_dim", 4)
        # low DDR: the states, the queue region, one layer's scratch (a GEMM's row scales sit at
        # BASE + 16 HDR_RSV, 24 bits: below 256 MiB); the weights after them
        self.vocab = weights["embed_tokens.weight"].shape[0] if head else 0
        self._state()
        self.queue_cap = queue_cap               # running chunk by chunk (generate): one chunk's queue at a time
        self.qbase = self.p.alloc(64 * queue_cap) if queue_cap else None
        self.scratch0 = self.p.top
        self.scratch_end = self.p.alloc(scratch_bytes) + scratch_bytes
        self._weights()
        self.end = self.p.top

    # ------------------------------------------------------------------ weights (host side)
    def lin(self, w: np.ndarray, bias: np.ndarray | None = None) -> QLin:
        qw = quantize_weight(np.asarray(w, np.float64), per_channel=True)
        assert qw.n % 16 == 0 and qw.k % 16 == 0
        b = self.p.put(np.asarray(bias, F32)) if bias is not None else None
        return QLin(self.p.put(qw.q, "i1"), self.p.put(qw.scale), b, qw.n, qw.k)

    def lin_k(self, w: np.ndarray, parts: int = 0) -> list[QLin]:
        """A linear layer whose K may exceed the GEMV unit's: quantized as one (per-channel scales
        over the whole row), stored as `parts` (default: the fewest) equal K blocks sharing the
        scales; the partial products add in fp32."""
        qw = quantize_weight(np.asarray(w, np.float64), per_channel=True)
        parts = parts or -(-qw.k // GEMV_K)
        kb = qw.k // parts
        assert kb * parts == qw.k and kb % 16 == 0 and qw.n % 16 == 0
        s = self.p.put(qw.scale)
        return [QLin(self.p.put(np.ascontiguousarray(qw.q[:, c:c + kb]), "i1"), s, None, qw.n, kb)
                for c in range(0, qw.k, kb)]

    def gamma1(self, w: np.ndarray, mul: float = 1.0) -> V:
        """The RMSNorm weight as gamma: Qwen3.5's zero-centered one, fp32(1 + w) (as torch: 1.0 +
        weight.float()); Qwen3's, w itself. Times `mul`: the attention scaling folded into q_norm
        (exact for Qwen3.5's 1/16; Qwen3's 128^-0.5 rounds once, in fp32)."""
        g = np.asarray(w, F32) if self.q3 else (F32(1.0) + np.asarray(w, F32)).astype(F32)
        return self.p.const((g * F32(mul)).astype(F32))

    def _la_weights(self, L: str) -> dict:
        W, p = self.w, self.p
        la = L + "linear_attn."
        kd, vd = self.nk * self.dk, self.nv * self.dv
        qkv = W[la + "in_proj_qkv.weight"]
        d = dict(
            ln_in=self.gamma1(W[L + "input_layernorm.weight"]),
            ln_post=self.gamma1(W[L + "post_attention_layernorm.weight"]),
            q=self.lin(qkv[:kd]), k=self.lin(qkv[kd:2 * kd]), v=self.lin(qkv[2 * kd:]),
            z=self.lin(W[la + "in_proj_z.weight"]),
            b=self.lin(W[la + "in_proj_b.weight"]),
            a=self.lin(W[la + "in_proj_a.weight"], W[la + "dt_bias"]),
            negA=p.const(-np.exp(W[la + "A_log"].astype(F32))),
            out=self.lin(W[la + "out_proj.weight"]),
            gnorm=p.const(W[la + "norm.weight"]),
            qscale=p.const(np.full(self.dk, F32(self.dk ** -0.5))),
            kscale=p.const(np.ones(self.dk, F32)),
            gate=self.lin(W[L + "mlp.gate_proj.weight"]), up=self.lin(W[L + "mlp.up_proj.weight"]),
            down=self.lin(W[L + "mlp.down_proj.weight"]))
        cw = W[la + "conv1d.weight"][:, 0, :]                  # [C, 4]
        self.taps = cw.shape[1]
        d["conv"] = {n: [p.const(cw[o:o + dd, j]) for j in range(self.taps)]
                     for n, o, dd in (("q", 0, kd), ("k", kd, kd), ("v", 2 * kd, vd))}
        return d

    def _fa_weights(self, A: str) -> dict:
        W = self.w
        sa = A + "self_attn."
        if self.gemma:
            g1 = lambda n, mul=1.0: self.gamma1(W[A + n + ".weight"], mul)
            return dict(
                ln_in=g1("input_layernorm"), attn_norm=g1("post_attention_layernorm"),
                ln_post=g1("pre_feedforward_layernorm"), mlp_norm=g1("post_feedforward_layernorm"),
                q=self.lin(W[sa + "q_proj.weight"]), g=None,
                k=self.lin(W[sa + "k_proj.weight"]), v=self.lin(W[sa + "v_proj.weight"]),
                qn=g1("self_attn.q_norm", self.qmul), kn=g1("self_attn.k_norm"),
                o=self.lin(W[sa + "o_proj.weight"]),
                gate=self.lin(W[A + "mlp.gate_proj.weight"]), up=self.lin(W[A + "mlp.up_proj.weight"]),
                down=self.lin_k(W[A + "mlp.down_proj.weight"]))
        if self.q3:                                            # no output gate
            q, g = self.lin(W[sa + "q_proj.weight"]), None
        else:                                                  # per head: the query's rows, then the gate's
            qp = W[sa + "q_proj.weight"].reshape(self.nh, 2, self.hd, self.H)
            q, g = self.lin(qp[:, 0].reshape(-1, self.H)), self.lin(qp[:, 1].reshape(-1, self.H))
        return dict(
            ln_in=self.gamma1(W[A + "input_layernorm.weight"]),
            ln_post=self.gamma1(W[A + "post_attention_layernorm.weight"]),
            q=q, g=g,
            k=self.lin(W[sa + "k_proj.weight"]), v=self.lin(W[sa + "v_proj.weight"]),
            qn=self.gamma1(W[sa + "q_norm.weight"], self.qmul), kn=self.gamma1(W[sa + "k_norm.weight"]),
            o=self.lin(W[sa + "o_proj.weight"]),
            gate=self.lin(W[A + "mlp.gate_proj.weight"]), up=self.lin(W[A + "mlp.up_proj.weight"]),
            down=self.lin(W[A + "mlp.down_proj.weight"]))

    def _weights(self) -> None:
        p = self.p
        self.taps = 4
        self.lw = [self._la_weights(f"layers.{i}.") if t == "linear_attention" else self._fa_weights(f"layers.{i}.")
                   for i, t in enumerate(self.types)]
        if self.gemma:
            for w, t in zip(self.lw, self.types):
                w["rope"] = t
        self.final_norm = self.gamma1(self.w["norm.weight"])
        if self.mtp:
            W = self.w
            self.mw = dict(self._fa_weights("mtp.layers.0."), fc=self.lin(W["mtp.fc.weight"]),
                           ge=self.gamma1(W["mtp.pre_fc_norm_embedding.weight"]),
                           gh=self.gamma1(W["mtp.pre_fc_norm_hidden.weight"]),
                           norm=self.gamma1(W["mtp.norm.weight"]))
        # RoPE table: row t = cos(t * inv_freq), sin(t * inv_freq) (fp32, as torch computes them)
        def table(theta: float) -> V:
            th = F32(theta)
            inv = (1.0 / (th ** (np.arange(0, self.rot, 2, dtype=F32) / F32(self.rot)))).astype(F32)
            fr = (np.arange(self.T, dtype=F32)[:, None] * inv[None, :]).astype(F32)
            return p.const(np.concatenate([np.cos(fr), np.sin(fr)], axis=1).astype(F32))
        if self.gemma:
            self.rope_tabs = {t: table(self.cfg.rope_parameters[t]["rope_theta"]) for t in sorted(set(self.types))}
        else:
            self.rope_tab = table(self.cfg.rope_parameters["rope_theta"])
        # the head: the tied embedding table as INT8 per-channel row blocks
        self.head_lins = []
        if self.head:
            from .qwen_ref import head_rows
            e = self.w["embed_tokens.weight"]
            hr = head_rows(e.shape[0])
            self.head_lins = [self.lin(e[r:r + hr]) for r in range(0, e.shape[0], hr)]

    def embed(self, ids) -> np.ndarray:
        """The host's lookup: input embeddings [len(ids), H] fp32."""
        return input_embedding(self.cfg, self.w["embed_tokens.weight"][np.asarray(ids)])

    # ------------------------------------------------------------------ state in DDR
    def _state(self) -> None:
        p, T, H = self.p, self.T, self.H
        kd, vd = self.nk * self.dk, self.nv * self.dv
        hist = self.taps - 1
        self.xs = [p.mat(T, H) for _ in range(len(self.types) + 1)]   # the residual stream: input, each layer's output
        self.x0 = self.xs[0]
        self.y = p.mat(T, H)                       # after the final norm
        self.st = []
        for t in self.types:
            if t == "linear_attention":
                # conv inputs, rows t + hist (the first hist rows are the zero history); DeltaNet states
                # with the MTP two copies: a verify's last token updates the other one (see _la)
                self.st.append(dict(cin={n: p.mat(T + hist, d) for n, d in (("q", kd), ("k", kd), ("v", vd))},
                                    S=[[p.mat(self.dk, self.dv, self.state_fmt) for _ in range(self.nv)]
                                       for _ in range(2 if self.mtp else 1)]))
            else:                                  # KV cache (zero padded)
                self.st.append(dict(kc=p.mat(self.Tp, self.nkv * self.hd), vc=p.mat(self.Tp, self.nkv * self.hd)))
        if self.mtp:                               # the MTP layer's KV cache (row r: token r + 1, hidden r)
            self.mst = dict(kc=p.mat(self.Tp, self.nkv * self.hd), vc=p.mat(self.Tp, self.nkv * self.hd))
        self.logits = p.mat(self.verify_rows, self.vocab) if self.head else None   # the chunk's last token (or tokens)

    def input(self, x: np.ndarray, t0: int = 0) -> None:
        a = self.p.base + self.x0.off + t0 * self.x0.ld
        self.p.ddr[a:a + x.size * 4] = np.asarray(x, F32).tobytes()

    # ------------------------------------------------------------------ blocks
    def chunk(self, t0: int, n: int, all_rows: bool = False, parity: int = 0, spec: bool = False) -> None:
        """Tokens t0 .. t0 + n - 1 through every layer, the final norm and (with the head) the last
        token's logits; all_rows: every token's, for a speculative verify (logits_rows reads them).
        Each head block writes its n rows densely (LDY16 cannot span a vocabulary-wide row).
        parity: the DeltaNet state copy in use; spec: the last token writes the other copy, so a
        rejected draft leaves this one as it was after token n - 2 (accepted: the parity flips)."""
        assert not all_rows or n <= self.verify_rows
        self.parity, self.spec = parity, spec
        p = self.p
        for i, t in enumerate(self.types):
            p.top = self.scratch0                  # this layer's temporaries over the last layer's
            (self._la if t == "linear_attention" else self._fa)(i, t0, n)
            assert p.top <= self.scratch_end, "scratch_bytes too small for this chunk"
        p.top = self.scratch0
        p.norm(self.xs[-1].rowv(t0, n), self.final_norm, self.eps, self.y.rowv(t0, n))
        if self.head:
            r = n if all_rows else 1
            self._head(self.y.rowv(t0 + n - r, r))
        assert p.top <= self.scratch_end, "scratch_bytes too small for this chunk"
        p.top = self.end                           # anything allocated later (the queue) after the weights

    def swap_state(self) -> None:
        """Copy the second DeltaNet state copy to the first, so plain queues (parity 0) can follow."""
        p = self.p
        for st, w in zip(self.st, self.lw):
            if "S" in st:
                one = replace(w["kscale"], rows=1, cols=1, ld=4)     # 1.0: the product is exact
                for a, b in zip(st["S"][1], st["S"][0]):
                    p.elt(C.ELT_MUL, a, one, b, bmode=C.BM_SCALAR)

    def _head(self, y: V) -> None:
        p, r, c0 = self.p, y.rows, 0
        yq, ys = p.quant(y)
        for lin in self.head_lins:
            p.gemm_rc(yq, ys, lin, V(self.logits.off + 4 * c0 * r, r, lin.n, 4 * lin.n))
            c0 += lin.n

    def mtp_chunk(self, r0: int, n: int, head: bool = True) -> None:
        """MTP rows r0 .. r0 + n - 1 (row r: the embedding of token r + 1 and the hidden state of
        position r, predicting token r + 2): fc over the two norms, one gated full-attention layer
        with its own KV cache, the norm and (head) the last row's logits, on the tied head."""
        p, w, H = self.p, self.mw, self.H
        p.top = self.scratch0
        cat = p.mat(n, 2 * H)
        p.norm(self.x0.rowv(r0 + 1, n), w["ge"], self.eps, cat.colv(0, H))
        p.norm(self.y.rowv(r0, n), w["gh"], self.eps, cat.colv(H, H))
        cq, cs = p.quant(cat)
        x, out = p.mat(n, H), p.mat(n, H)
        p.gemm_rc(cq, cs, w["fc"], x)
        self._fa_block(w, self.mst, x, out, r0, n)
        if head:
            y = p.mat(1, H)
            p.norm(out.rowv(n - 1, 1), w["norm"], self.eps, y)
            self._head(y)
        assert p.top <= self.scratch_end, "scratch_bytes too small for this chunk"
        p.top = self.end

    def _mlp(self, w: dict, x: V, out: V) -> None:
        """out = x + down(silu(gate(h)) * up(h)), h = RMSNorm(x). Gemma 3: GELU (tanh) for SiLU,
        the down projection in K blocks, its output normalized before the residual add."""
        p, n = self.p, x.rows
        h = p.mat(n, self.H)
        p.norm(x, w["ln_post"], self.eps, h)
        hq, hs = p.quant(h)
        g, u = p.mat(n, w["gate"].n), p.mat(n, w["up"].n)
        p.gemm_rc(hq, hs, w["gate"], g, act=C.ACT_GELU if self.gemma else C.ACT_SILU)
        p.gemm_rc(hq, hs, w["up"], u)
        p.elt(C.ELT_MUL, g, u, g)                                   # in place: act(gate) * up
        gq, gs = p.quant(g)
        d = p.mat(n, self.H)
        downs = w["down"] if isinstance(w["down"], list) else [w["down"]]
        c0 = 0
        for i, lin in enumerate(downs):
            part = d if i == 0 else p.mat(n, self.H)
            p.gemm_rc(gq.colv(c0, lin.k), gs, lin, part)
            if i:
                p.elt(C.ELT_ADD, d, part, d)
            c0 += lin.k
        if w.get("mlp_norm") is not None:
            dn = p.mat(n, self.H)
            p.norm(d, w["mlp_norm"], self.eps, dn)
            d = dn
        p.elt(C.ELT_ADD, x, d, out)

    def _la(self, li: int, t0: int, n: int) -> None:
        p, w, st = self.p, self.lw[li], self.st[li]
        kd, vd, hist = self.nk * self.dk, self.nv * self.dv, self.taps - 1
        x = self.xs[li].rowv(t0, n)
        h = p.mat(n, self.H)
        p.norm(x, w["ln_in"], self.eps, h)
        hq, hs = p.quant(h)
        # projections; q, k, v straight into the conv input rows
        for nm in ("q", "k", "v"):
            p.gemm_rc(hq, hs, w[nm], st["cin"][nm].rowv(hist + t0, n))
        z = p.mat(n, vd)
        p.gemm_rc(hq, hs, w["z"], z)
        beta, sp = p.mat(n, self.nv), p.mat(n, self.nv)
        p.gemm_rc(hq, hs, w["b"], beta, act=C.ACT_SIGMOID)
        p.gemm_rc(hq, hs, w["a"], sp, act=C.ACT_SOFTPLUS)           # + dt_bias (the GEMM's bias)
        g = p.mat(n, self.nv)
        p.elt(C.ELT_MUL, sp, w["negA"], g, bmode=C.BM_ROW)          # g = -exp(A_log) * softplus
        decay = p.mat(n, self.nv)
        p.act(g, C.ACT_EXP, decay)
        # causal depthwise conv (taps over rows t-3..t), then SiLU
        conv = {}
        for nm, d in (("q", kd), ("k", kd), ("v", vd)):
            src, acc = st["cin"][nm], p.mat(n, d)
            p.elt(C.ELT_MUL, src.rowv(t0, n), w["conv"][nm][0], acc, bmode=C.BM_ROW)
            for j in range(1, self.taps):
                p.elt(C.ELT_MULADD, src.rowv(t0 + j, n), w["conv"][nm][j], acc, bmode=C.BM_ROW, c=acc)
            o = p.mat(n, d)
            p.act(acc, C.ACT_SILU, o)
            conv[nm] = o
        # q, k: L2 norm per head (q also scaled by dk^-0.5 through gamma)
        qn, kn = p.mat(n, kd), p.mat(n, kd)
        p.norm(conv["q"].heads(self.dk), w["qscale"], 1e-6, qn.heads(self.dk), l2=True)
        p.norm(conv["k"].heads(self.dk), w["kscale"], 1e-6, kn.heads(self.dk), l2=True)
        qh, kh, vh = qn.heads(self.dk), kn.heads(self.dk), conv["v"].heads(self.dv)
        # the gated delta rule, token by token, head by head
        o = p.mat(n, vd)
        oh = o.heads(self.dv)
        kvm, dl = p.mat(1, self.dv), p.mat(1, self.dv)
        for t in range(n):
            for hh in range(self.nv):
                r = t * self.nv + hh
                # the decay pass reads the state in use; a verify's last token writes the other copy
                S0 = st["S"][self.parity][hh]
                S = st["S"][1 - self.parity][hh] if self.spec and t == n - 1 else S0
                p.elt(C.ELT_MUL, S0, decay.rowv(t, 1).colv(hh, 1), S, bmode=C.BM_SCALAR)
                p.fdot(kh.rowv(r, 1), S, kvm, nn=True)                       # kv_mem = S^T k
                p.elt(C.ELT_SUB, vh.rowv(r, 1), kvm, dl)                     # v - kv_mem
                p.elt(C.ELT_MUL, dl, beta.rowv(t, 1).colv(hh, 1), dl, bmode=C.BM_SCALAR)
                p.elt(C.ELT_MULADD, dl, kh.rowv(r, 1), S, bmode=C.BM_COL, arow=True, c=S)  # S += k (x) delta
                p.fdot(qh.rowv(r, 1), S, oh.rowv(r, 1), nn=True)             # out = S^T q
        # gated RMSNorm per head, then silu(z) * it
        on, og = p.mat(n, vd), p.mat(n, vd)
        p.norm(oh, w["gnorm"], self.eps, on.heads(self.dv))
        p.elt(C.ELT_MUL, z, on, og, act=C.ACT_SILU)
        oq, os_ = p.quant(og)
        a = p.mat(n, self.H)
        p.gemm_rc(oq, os_, w["out"], a)
        x1 = p.mat(n, self.H)
        p.elt(C.ELT_ADD, x, a, x1)
        self._mlp(w, x1, self.xs[li + 1].rowv(t0, n))

    def _fa(self, li: int, t0: int, n: int) -> None:
        self._fa_block(self.lw[li], self.st[li], self.xs[li].rowv(t0, n), self.xs[li + 1].rowv(t0, n), t0, n)

    def _fa_block(self, w: dict, st: dict, x: V, out: V, t0: int, n: int) -> None:
        p = self.p
        kc, vc = st["kc"], st["vc"]
        nh, nkv, hd = self.nh, self.nkv, self.hd
        h = p.mat(n, self.H)
        p.norm(x, w["ln_in"], self.eps, h)
        hq, hs = p.quant(h)
        q = p.mat(n, nh * hd)
        k, v = p.mat(n, nkv * hd), vc.rowv(t0, n)
        p.gemm_rc(hq, hs, w["q"], q)
        if w["g"] is not None:
            g = p.mat(n, nh * hd)
            p.gemm_rc(hq, hs, w["g"], g)
        p.gemm_rc(hq, hs, w["k"], k)
        p.gemm_rc(hq, hs, w["v"], v)                                 # V straight into the cache
        # q / k norm per head (q's gamma carries hd^-0.5), RoPE; K into the cache
        qn, kn = p.mat(n, nh * hd), p.mat(n, nkv * hd)
        p.norm(q.heads(hd), w["qn"], self.eps, qn.heads(hd))
        p.norm(k.heads(hd), w["kn"], self.eps, kn.heads(hd))
        qr = p.mat(n, nh * hd)
        tab = (self.rope_tabs[w["rope"]] if "rope" in w else self.rope_tab).rowv(t0, (n * nh - 1) // nh + 1)
        p.rope(qn.heads(hd), tab, nh, self.rot, qr.heads(hd))
        p.rope(kn.heads(hd), tab, nkv, self.rot, kc.rowv(t0, n).heads(hd))
        # attention per query head over keys [0, t0 + n) (padded to 16, masked causally)
        nk = pad16(t0 + n)
        att = p.mat(n, nh * hd)
        sc, pr = p.mat(n, nk), p.mat(n, nk)
        for hh in range(nh):
            kv = hh // (nh // nkv)
            qv = V(qr.off + hh * hd * 4, n, hd, qr.ld)
            kmat = V(kc.off + kv * hd * 4, nk, hd, kc.ld)
            vmat = V(vc.off + kv * hd * 4, nk, hd, vc.ld)
            p.fdot(qv, kmat, sc, nn=False)                           # scores (scaling in q_norm)
            p.softmax_causal(sc, pr, t0 + 1)
            p.fdot(pr, vmat, V(att.off + hh * hd * 4, n, hd, att.ld), nn=True)
        if w["g"] is not None:
            ga = p.mat(n, nh * hd)
            p.elt(C.ELT_MUL, g, att, ga, act=C.ACT_SIGMOID)          # attn * sigmoid(gate)
            att = ga
        aq, as_ = p.quant(att)
        o = p.mat(n, self.H)
        p.gemm_rc(aq, as_, w["o"], o)
        if w.get("attn_norm") is not None:
            on = p.mat(n, self.H)
            p.norm(o, w["attn_norm"], self.eps, on)
            o = on
        x1 = p.mat(n, self.H)
        p.elt(C.ELT_ADD, x, o, x1)
        self._mlp(w, x1, out)

    # ------------------------------------------------------------------ running chunk by chunk
    def stage(self, ddr: bytearray, a: int, b: int) -> tuple[int, int]:
        """Descriptors a .. b - 1 into the queue region of `ddr`: (queue address, count)."""
        assert self.qbase is not None and b - a <= self.queue_cap, "queue_cap too small"
        q = self.p.base + self.qbase
        for i in range(a, b):
            ddr[q + 64 * (i - a):q + 64 * (i - a + 1)] = self.p.descs[i]
        return q, b - a

    def put_input(self, ddr: bytearray, t0: int, x: np.ndarray) -> None:
        a = self.p.base + self.x0.off + t0 * self.x0.ld
        ddr[a:a + x.size * 4] = np.asarray(x, F32).tobytes()

    def logits_of(self, ddr) -> np.ndarray:
        a = self.p.base + self.logits.off
        return np.frombuffer(bytes(ddr[a:a + 4 * self.vocab]), "<f4").copy()

    def logits_rows(self, ddr, n: int) -> np.ndarray:
        """The n rows [n, vocab] of an all_rows chunk (block by block in DDR)."""
        a = self.p.base + self.logits.off
        flat = np.frombuffer(bytes(ddr[a:a + 4 * self.vocab * n]), "<f4")
        out, c0 = np.empty((n, self.vocab), F32), 0
        for lin in self.head_lins:
            out[:, c0:c0 + lin.n] = flat[c0 * n:(c0 + lin.n) * n].reshape(n, lin.n)
            c0 += lin.n
        return out

    def generate(self, run, prompt, n_new: int) -> tuple[list[int], list[np.ndarray], list[int]]:
        """Greedy decoding with the host in the loop: the host looks up the input embedding,
        `run(ddr, q, n) -> ddr` executes one chunk's queue (the golden model or the RTL), the host
        takes the argmax of the logits. Returns (token ids, logits per step, descriptors per step)."""
        assert self.head
        ddr = self.p.ddr                         # run in place: the program image becomes the device's DDR
        ids, logits, counts = list(prompt), [], []
        t0, n = 0, len(prompt)
        for _ in range(n_new + 1):
            self.put_input(ddr, t0, self.embed(ids[t0:t0 + n]))
            a = len(self.p.descs)
            self.chunk(t0, n)
            q, cnt = self.stage(ddr, a, len(self.p.descs))
            out = run(ddr, q, cnt)
            ddr = out if isinstance(out, bytearray) else bytearray(out)
            lg = self.logits_of(ddr)
            logits.append(lg)
            counts.append(cnt)
            ids.append(int(np.argmax(lg)))
            t0, n = t0 + n, 1
        return ids, logits, counts

    def generate_mtp(self, run, prompt, n_new: int, oracle=None) -> tuple[list[int], dict[int, np.ndarray], int]:
        """Greedy speculative decoding with the MTP head: the same tokens as generate(). Each round
        verifies [current, draft] (two rows); a hit emits two tokens and flips the state copy, a miss
        emits the correction and keeps it. The MTP then takes the rows the verify made known and
        drafts the next token. oracle(t, draft) may replace a draft (tests). Returns (token ids,
        {position: its logits}, rounds)."""
        assert self.head and self.mtp and self.verify_rows >= 2
        state = dict(ddr=self.p.ddr)

        def go(emit) -> None:
            a = len(self.p.descs)
            emit()
            q, c = self.stage(state["ddr"], a, len(self.p.descs))
            out = run(state["ddr"], q, c)
            state["ddr"] = out if isinstance(out, bytearray) else bytearray(out)

        def put(t: int, toks) -> None:
            self.put_input(state["ddr"], t, self.embed(toks))
        ids, P, par, rounds = list(prompt), len(prompt), 0, 0
        put(0, ids)
        go(lambda: self.chunk(0, P))
        lg = {P - 1: self.logits_of(state["ddr"])}
        ids.append(int(np.argmax(lg[P - 1])))
        put(P, ids[P:])
        go(lambda: self.mtp_chunk(0, P))                    # rows 0 .. P - 1, the last one drafts
        t = P                                               # ids[t]: known, not yet in the cache
        while len(ids) < P + n_new + 1:
            d = int(np.argmax(self.logits_of(state["ddr"])))
            if oracle is not None:
                d = oracle(t, d)
            put(t + 1, [d])
            go(lambda: self.chunk(t, 2, all_rows=True, parity=par, spec=True))
            rows = self.logits_rows(state["ddr"], 2)
            rounds += 1
            lg[t] = rows[0]
            x = int(np.argmax(rows[0]))
            ids.append(x)
            if x == d:
                lg[t + 1] = rows[1]
                ids.append(int(np.argmax(rows[1])))
                par = 1 - par
                put(t + 2, ids[t + 2:t + 3])
                go(lambda: self.mtp_chunk(t, 2))
                t += 2
            else:
                put(t + 1, [x])
                go(lambda: self.mtp_chunk(t, 1))
                t += 1
        return ids, lg, rounds

    # ------------------------------------------------------------------ results
    def _read(self, m, v: V) -> np.ndarray:
        a = self.p.base + v.off
        return np.stack([np.frombuffer(m.read(a + r * v.ld, 4 * v.cols), "<f4") for r in range(v.rows)])

    def read_layer(self, m, layer: int, t0: int, n: int) -> np.ndarray:
        return self._read(m, self.xs[layer + 1].rowv(t0, n))

    def read_logits(self, m) -> np.ndarray:
        return self._read(m, self.logits)[0]

    def read_final(self, m, t0: int, n: int) -> np.ndarray:
        return self._read(m, self.y.rowv(t0, n))
