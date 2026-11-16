"""Qwen3.5-0.8B's vision tower: a transformers reference and a lowering to the v0.4 ops.

The tower is 12 blocks of 768 (12 heads of 64), patch 16, a 2x2 spatial merge, a learned 48x48
position table and axial 2D rotary — the input half of the model llm.py already runs. It needs
no new op:

    patch embed   a linear over the patch vector (Conv3d with kernel = stride = [2, 16, 16])
    block         LayerNorm -> qkv -> axial RoPE on q and k -> attention over the whole image
                  (FDOT NT, softmax, FDOT NN; no causal mask; the scores' padding masked)
                  -> proj -> residual; LayerNorm -> fc1 (GELU tanh) -> fc2 -> residual
    merger        LayerNorm per patch (768 wide) -> the 2x2 block's 4 x 768 values as one row
                  -> fc1 -> GELU -> fc2 -> 1024

The host keeps what it keeps for the text model: normalization and patchify (in spatial-merge
block order, as Qwen2VLImageProcessorFast emits them), the bilinear resample of the learned
position table, and the rotary table (one row per patch: cos[head_dim/2] then sin[head_dim/2]).

    cfg  = vision_config("/path/to/Qwen3.5-0.8B")
    w    = load_vision("/path/to/Qwen3.5-0.8B")
    px, grid_thw = patchify(normalized_image[3, H, W], cfg)     # host preprocessing
    ref  = run_reference(cfg, w, px, grid_thw)                  # transformers, fp32
    low  = VisionLowering(cfg, w, patches_max=len(px))
    got  = low.run(px, position_embeddings(cfg, w, grid_thw), rope_table(cfg, grid_thw))

`run_reference(..., fake_quant=True)` swaps every linear (patch embed included) for the W8A8
arithmetic the descriptors describe, and the merger's erf GELU for the contract's tanh GELU (the
blocks already take gelu_pytorch_tanh): that is what `got` must match.
"""
from __future__ import annotations

import numpy as np

from . import contract as C
from .llm import Dev, V, QLin, pad16

F32 = np.float32

# the 0.8B vision_config (Qwen/Qwen3.5-0.8B config.json); a checkpoint supplies the real numbers
VIT_08B = dict(depth=12, hidden_size=768, intermediate_size=3072, num_heads=12, in_channels=3,
               patch_size=16, spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=1024,
               num_position_embeddings=2304, hidden_act="gelu_pytorch_tanh")
VIT_EPS = 1e-6
MASK_NEG = F32(-3.0e38)          # the padded score: exp(that - max) underflows to +0


def vision_config(path: str | None = None, **over):
    """Qwen3_5VisionConfig: `path`'s config.json vision_config (else the 0.8B defaults), with any
    keyword overriding a field — the tests pass a much smaller tower."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    base = dict(VIT_08B)
    if path is not None:
        import json
        from pathlib import Path
        vc = json.loads((Path(path) / "config.json").read_text())["vision_config"]
        base = dict(base, **{k: v for k, v in vc.items() if k != "model_type"})
    base.update(over)
    cfg = Qwen3_5VisionConfig(**base)
    cfg._attn_implementation = "eager"
    return cfg


def head_dim(cfg) -> int:
    return cfg.hidden_size // cfg.num_heads


def _meta(cfg):
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    with torch.device("meta"):
        return Qwen3_5VisionModel(cfg)


def random_vision(cfg, seed: int = 0) -> dict[str, np.ndarray]:
    """Random weights with the tower's names and shapes: linears N(0, 0.02), norm weights
    1 + N(0, 0.1), biases N(0, 0.02), the position table N(0, 0.02)."""
    rng = np.random.default_rng(seed)
    w = {}
    for name, p in _meta(cfg).state_dict().items():
        shape = tuple(p.shape)
        if name.endswith(("norm.weight", "norm1.weight", "norm2.weight")):
            a = 1 + rng.normal(0, 0.1, shape)
        else:
            a = rng.normal(0, 0.02, shape)
        w[name] = a.astype(F32)
    return w


def load_vision(path: str) -> dict[str, np.ndarray]:
    """A checkpoint's vision tower as fp32 under the model's own names (patch_embed.*, blocks.*,
    merger.*, pos_embed.weight); every parameter must be present with its shape."""
    from pathlib import Path
    from safetensors import safe_open
    import torch
    shapes = {n: tuple(p.shape) for n, p in _meta(vision_config(path)).state_dict().items()}
    w = {}
    for f in sorted(Path(path).glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as st:
            for k in st.keys():
                if not k.startswith("model.visual."):
                    continue
                name = k[len("model.visual."):]
                if name not in shapes:
                    continue
                t = st.get_tensor(k)
                if tuple(t.shape) != shapes[name]:
                    raise ValueError(f"{k}: shape {tuple(t.shape)}, the tower needs {shapes[name]}")
                w[name] = t.to(torch.float32).numpy()
    missing = [n for n in shapes if n not in w]
    if missing:
        raise ValueError(f"{path}: the vision tower lacks {missing[:5]} ({len(missing)} parameters)")
    return w


# ---------------------------------------------------------------------- host preprocessing
def patchify(img, cfg) -> tuple[np.ndarray, np.ndarray]:
    """A normalized image [C, H, W] (sides multiples of patch * merge) as the processor emits it:
    patches [P, C * temporal * patch * patch] in spatial-merge-block order and grid_thw
    [[1, H / patch, W / patch]]. The flatten order is (channel, temporal, row, column) — the
    conv3d weight's — and a still image fills both temporal slots."""
    img = np.asarray(img, F32)
    c, hh, ww = img.shape
    ps, ms, tp = cfg.patch_size, cfg.spatial_merge_size, cfg.temporal_patch_size
    assert c == cfg.in_channels, f"{c} channels, the tower takes {cfg.in_channels}"
    assert hh % (ps * ms) == 0 and ww % (ps * ms) == 0, \
        f"{hh}x{ww}: the sides must be multiples of {ps * ms} (patch * merge)"
    gh, gw = hh // ps, ww // ps
    x = img.reshape(c, gh // ms, ms, ps, gw // ms, ms, ps).transpose(1, 4, 2, 5, 0, 3, 6)
    x = np.broadcast_to(x[..., None, :, :], x.shape[:5] + (tp, ps, ps))
    return np.ascontiguousarray(x, F32).reshape(gh * gw, c * tp * ps * ps), np.array([[1, gh, gw]])


def position_embeddings(cfg, w, grid_thw) -> np.ndarray:
    """The learned table resampled to the image's patch grid, [P, hidden] (bilinear,
    align_corners, as the tower's forward computes it)."""
    import torch
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights
    side = int(cfg.num_position_embeddings ** 0.5)
    idx, wt = get_vision_interpolation_indices_and_weights(
        torch.as_tensor(grid_thw), num_grid_per_side=side, mode="bilinear", align_corners=True,
        spatial_merge_size=cfg.spatial_merge_size)
    tab = np.asarray(w["pos_embed.weight"], F32)
    return (tab[idx.numpy()] * wt.numpy()[:, :, None]).sum(1).astype(F32)


def rope_table(cfg, grid_thw) -> np.ndarray:
    """One ROPE table row per patch: cos[rot/2] then sin[rot/2] of the axial angles
    [h (rot/4), w (rot/4)]. The tower duplicates (h, w) to fill the head dim and pairs element i
    with i + rot/2, so that first half is the whole rotation."""
    import torch
    from transformers.vision_utils import get_vision_position_ids
    hd = head_dim(cfg)
    sd = hd // 2                                            # the spatial dim: h and w angles each
    pos = get_vision_position_ids(torch.as_tensor(grid_thw), cfg.spatial_merge_size).float()
    inv = 1.0 / (cfg.rope_parameters["rope_theta"]
                 ** (torch.arange(0, sd, 2, dtype=torch.float) / sd))
    fr = pos[..., None].float() * inv                       # [P, 2, hd/4]
    ang = torch.cat([fr[:, 0], fr[:, 1]], dim=-1)            # [P, hd/2]: h then w
    return torch.cat([ang.cos(), ang.sin()], dim=-1).numpy().astype(F32)


# ---------------------------------------------------------------------- the reference
def _fq_patch_embed(pe) -> None:
    """The Conv3d as the linear the lowering emits, W8A8-emulated (per-token INT8 activations,
    per-channel INT8 weights, exact integer products, the two scales)."""
    import torch
    from .compiler import quantize_weight
    from .qwen_ref import quant_tokens
    W = pe.proj.weight.detach().numpy().reshape(pe.embed_dim, -1).astype(np.float64)
    bias = torch.from_numpy(pe.proj.bias.detach().numpy().astype(np.float64))[None, :]
    qw = quantize_weight(W, per_channel=True)
    wq, sc = torch.from_numpy(qw.q), torch.from_numpy(qw.scale.astype(np.float64))

    def fwd(hidden_states):
        x = hidden_states.reshape(-1, W.shape[1]).numpy()
        q, s = quant_tokens(x)
        acc = torch.from_numpy(q.astype(np.float64)) @ wq.double().T
        y = acc * torch.from_numpy(s.astype(np.float64))[:, None] * sc[None, :] + bias
        return y.to(torch.float32).reshape(-1, pe.embed_dim)

    pe.forward = fwd


def _fq_merger_gelu(merger) -> None:
    """The merger's nn.GELU() (the erf form) as ACT_GELU (the tanh form): the descriptor set's
    only GELU, and within ~1e-3 of erf. The blocks already take hidden_act gelu_pytorch_tanh."""
    import torch

    def fwd(x):
        return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))
    merger.act_fn.forward = fwd


def run_reference(cfg, w, patches: np.ndarray, grid_thw, fake_quant: bool = False,
                  blocks: bool = False, dtype: str = "fp32") -> dict:
    """The tower in transformers on the processor's patches: the merged visual tokens [P/4,
    out_hidden], and with `blocks` the patches' hidden state before the merger and every block's
    output. `fake_quant` swaps every linear (patch embed included) for the W8A8 arithmetic."""
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    td = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    m = Qwen3_5VisionModel(cfg).to(td).eval()
    m.load_state_dict({k: torch.from_numpy(v).to(td) for k, v in w.items()}, strict=True)
    if fake_quant:
        from .qwen_ref import _fake_quant_linear
        _fq_patch_embed(m.patch_embed)
        _fq_merger_gelu(m.merger)
        for mod in m.modules():
            if isinstance(mod, torch.nn.Linear):
                _fake_quant_linear(mod)
    outs: list[np.ndarray] = []
    hooks = [b.register_forward_hook(lambda _m, _i, o: outs.append(o.detach().float().numpy().copy()))
             for b in m.blocks]
    with torch.no_grad():
        out = m(hidden_states=torch.from_numpy(np.ascontiguousarray(patches)).to(td),
                grid_thw=torch.as_tensor(grid_thw))
    for h in hooks:
        h.remove()
    res = dict(merged=out.pooler_output.float().numpy().copy())
    if blocks:
        res["blocks"] = outs
        res["last"] = out.last_hidden_state.float().numpy().copy()
    return res


# ---------------------------------------------------------------------- the lowering
class VisionLowering:
    """The vision tower as v0.4 descriptors on one core.

        low = VisionLowering(cfg, w, patches_max=P)
        tokens = low.run(patches, pos, rope)      # the golden model: [P/4, out_hidden]

    The key axis is padded to a multiple of 16 (a descriptor's N): the padded key and value rows
    are zero and the padded scores are masked to MASK_NEG, so the padding leaves the sums alone
    (exp underflows to +0, and the lane folding of the sum is unchanged). The weights and the
    buffers are fixed and sized for `patches_max`; the host data (the patches, the position and
    rotary rows, the two score-mask rows) is written per image, so one program serves any image up
    to that capacity.
    """

    def __init__(self, cfg, w, patches_max: int, ddr_bytes: int = 1 << 28,
                 scratch_bytes: int | None = None, wbits: int = 8):
        """`ddr_bytes` holds the weights, the buffers and the queue; `scratch_bytes` holds what a
        host writes per image (the patches, the position and rotary tables, the quantized
        activations) — by default about twice what a full image measures (~10 MB at 448x448)."""
        if wbits != 8:
            raise ValueError("the vision tower is lowered W8A8")
        self.cfg, self.w = cfg, w
        self.H, self.I = cfg.hidden_size, cfg.intermediate_size
        self.nh, self.hd = cfg.num_heads, head_dim(cfg)
        self.K = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size ** 2
        self.ms2 = cfg.spatial_merge_size ** 2
        self.P, self.npad = patches_max, pad16(patches_max)
        assert self.nh * self.hd == self.H and self.H % 16 == 0 and self.K % 16 == 0
        assert self.hd % 32 == 0, "the rotary width must be a multiple of 32"
        self.eps = VIT_EPS
        self.p = self.prog = Dev(ddr_bytes=ddr_bytes)
        self._weights()
        self._bufs()
        self.scratch0 = self.p.top
        if scratch_bytes is None:                       # patches + pos + rope + the widest quant
            scratch_bytes = 4 * patches_max * (self.K + self.ms2 * self.H + 2 * self.H) + (1 << 20)
        self.scratch_end = self.p.alloc(scratch_bytes) + scratch_bytes
        self.end = self.p.top

    # -- weights and buffers
    def lin(self, w: np.ndarray, b: np.ndarray | None = None) -> QLin:
        from .compiler import quantize_weight
        qw = quantize_weight(np.asarray(w, np.float64), per_channel=True)
        assert qw.n % 16 == 0 and qw.k % 16 == 0, (qw.n, qw.k)
        b_off = self.p.put(np.asarray(b, F32)) if b is not None else None
        return QLin(self.p.put(qw.q, "i1"), self.p.put(qw.scale), b_off, qw.n, qw.k)

    def _weights(self) -> None:
        w, H = self.w, self.H
        self.pe = self.lin(w["patch_embed.proj.weight"].reshape(H, -1), w["patch_embed.proj.bias"])
        self.blocks = []
        for i in range(self.cfg.depth):
            b = f"blocks.{i}."
            self.blocks.append(dict(
                n1=(self.p.const(w[b + "norm1.weight"]), self.p.const(w[b + "norm1.bias"])),
                n2=(self.p.const(w[b + "norm2.weight"]), self.p.const(w[b + "norm2.bias"])),
                qkv=self.lin(w[b + "attn.qkv.weight"], w[b + "attn.qkv.bias"]),
                proj=self.lin(w[b + "attn.proj.weight"], w[b + "attn.proj.bias"]),
                fc1=self.lin(w[b + "mlp.linear_fc1.weight"], w[b + "mlp.linear_fc1.bias"]),
                fc2=self.lin(w[b + "mlp.linear_fc2.weight"], w[b + "mlp.linear_fc2.bias"])))
        # the merger normalizes each patch's H values, then folds the 2x2 block into one row
        self.m_norm = (self.p.const(w["merger.norm.weight"]), self.p.const(w["merger.norm.bias"]))
        self.m_fc1 = self.lin(w["merger.linear_fc1.weight"], w["merger.linear_fc1.bias"])
        self.m_fc2 = self.lin(w["merger.linear_fc2.weight"], w["merger.linear_fc2.bias"])
        assert self.m_fc1.k == self.ms2 * H, "the merger takes the 2x2 block's values"
        self.sc_q = self.p.const(np.array([F32(self.hd ** -0.5)], F32))    # the attention scaling

    def _bufs(self) -> None:
        p, n, npad = self.p, self.P, self.npad
        nm = n // self.ms2
        # the padded key / value rows are zero; the padded scores go to MASK_NEG in two steps (a
        # MUL by a 0/1 row, then an ADD of a row that is 0 where the key is real), because the
        # ELT.MULADD bias is an M x N matrix rather than a row
        self.kbuf = p.const(np.zeros((npad, self.H), F32))
        self.vbuf = p.const(np.zeros((npad, self.H), F32))
        self.sc = p.const(np.zeros((n, npad), F32))
        self.prow = p.const(np.zeros((n, npad), F32))      # the softmax's output, one row per patch
        self.mask = p.const(np.concatenate([np.ones((1, n), F32), np.zeros((1, npad - n), F32)], axis=1))
        self.mbias = p.const(np.concatenate([np.zeros((1, n), F32),
                                            np.full((1, npad - n), MASK_NEG, F32)], axis=1))
        self.zrow = p.const(np.zeros((1, self.H), F32))    # the zero row a copy adds
        # one INT8 activation buffer and one row-scale row, reused before every GEMM: a per-layer
        # allocation would grow the program by depth x the activation width
        self.q8 = p.mat(n, max(self.I, self.ms2 * self.H), C.OUT_INT8)
        self.qs = p.alloc(4 * n)
        self.x = p.mat(n, self.H)                 # the residual stream
        self.h = p.mat(n, self.H)                 # a norm's output
        self.qkv = p.mat(n, 3 * self.H)
        self.qd = p.mat(n, self.H)                # q / k of this block, dense (rope's input)
        self.kd = p.mat(n, self.H)
        self.qr = p.mat(n, self.H)                # q after RoPE and the 1/sqrt(d) scaling
        self.att = p.mat(n, self.H)
        self.b = p.mat(n, self.H)                 # a projection / the MLP's down output
        self.ml = p.mat(n, self.I)
        self.mm = p.mat(nm, self.m_fc1.n)
        self.mt = p.mat(nm, self.ms2 * self.H)
        self.out = p.mat(nm, self.cfg.out_hidden_size)

    # -- ops
    def _ln(self, x: V, nb: tuple[V, V], y: V) -> None:
        self.p.rowop(x, C.ROW_LAYERNORM, y, nb[0], self.eps, beta=nb[1])

    def _quant(self, x: V) -> tuple[V, int]:
        """x's INT8 rows into the shared activation buffer (and the shared row-scale row)."""
        return self.p.quant(x, V(self.q8.off, x.rows, x.cols, self.q8.ld, C.OUT_INT8), self.qs)

    def _linear(self, x: V, w: QLin, y: V, act: int = C.ACT_NONE) -> None:
        xq, xs = self._quant(x)
        self.p.gemm_rc(xq, xs, w, y, act)

    def _emit_image(self, patches: np.ndarray, pos: np.ndarray, rope: np.ndarray) -> V:
        p, n, H, hd, nh, npad = self.p, patches.shape[0], self.H, self.hd, self.nh, self.npad
        assert patches.shape[1] == self.K and n <= self.P and n % self.ms2 == 0
        assert pos.shape == (n, H) and rope.shape == (n, hd)
        p.top = self.scratch0
        # the score mask is this image's patch count, not the capacity's: the host writes both
        # rows with the patches, so one program serves any image up to patches_max
        on, ng = np.zeros(npad, F32), np.zeros(npad, F32)
        on[:n], ng[n:] = 1.0, MASK_NEG
        for buf, row in ((self.mask, on), (self.mbias, ng)):
            a = p.base + buf.off
            p.ddr[a:a + 4 * npad] = row.tobytes()
        x = self.x.rowv(0, n)
        self._linear(p.const(np.asarray(patches, F32)), self.pe, x)      # patch embed
        p.elt(C.ELT_ADD, x, p.const(np.asarray(pos, F32)), x)            # + position embedding
        ropet = p.const(np.asarray(rope, F32))
        sc = V(self.sc.off, n, npad, self.sc.ld)                          # scores (npad wide)
        for b in self.blocks:
            h = self.h.rowv(0, n)
            self._ln(x, b["n1"], h)
            qkv = self.qkv.rowv(0, n)
            p.gemm_rc(*self._quant(h), b["qkv"], qkv)
            # q, k, v into their own dense buffers: the rope and the per-head views need a
            # uniform row stride, which a column slice of the qkv output does not have
            p.elt(C.ELT_ADD, qkv.colv(0, H), self.zrow, self.qd.rowv(0, n), bmode=C.BM_ROW)
            p.elt(C.ELT_ADD, qkv.colv(H, H), self.zrow, self.kd.rowv(0, n), bmode=C.BM_ROW)
            p.elt(C.ELT_ADD, qkv.colv(2 * H, H), self.zrow, self.vbuf.rowv(0, n), bmode=C.BM_ROW)
            qr = self.qr.rowv(0, n)
            p.rope(self.qd.rowv(0, n).heads(hd), ropet, nh, hd, qr.heads(hd))
            p.elt(C.ELT_MUL, qr, self.sc_q, qr, bmode=C.BM_SCALAR)        # 1/sqrt(head_dim)
            p.rope(self.kd.rowv(0, n).heads(hd), ropet, nh, hd, self.kbuf.rowv(0, n).heads(hd))
            att = self.att.rowv(0, n)
            for hh in range(nh):
                p.fdot(V(qr.off + hh * hd * 4, n, hd, 4 * H),            # one head: n patch rows
                       V(self.kbuf.off + hh * hd * 4, npad, hd, 4 * H), sc, nn=False)
                p.elt(C.ELT_MUL, sc, self.mask, sc, bmode=C.BM_ROW)     # padded columns -> +0
                p.elt(C.ELT_ADD, sc, self.mbias, sc, bmode=C.BM_ROW)    # ... and then to MASK_NEG
                p.rowop(sc, C.ROW_SOFTMAX, V(self.prow.off, n, npad, self.prow.ld))
                p.fdot(V(self.prow.off, n, npad, self.prow.ld),
                       V(self.vbuf.off + hh * hd * 4, npad, hd, 4 * H),
                       V(att.off + hh * hd * 4, n, hd, att.ld), nn=True)
            self._linear(att, b["proj"], self.b.rowv(0, n))
            p.elt(C.ELT_ADD, x, self.b.rowv(0, n), x)
            self._ln(x, b["n2"], h)
            ml = self.ml.rowv(0, n)
            p.gemm_rc(*self._quant(h), b["fc1"], ml, C.ACT_GELU)
            self._linear(ml, b["fc2"], self.b.rowv(0, n))
            p.elt(C.ELT_ADD, x, self.b.rowv(0, n), x)
        nm = n // self.ms2
        # the merger's norm is H wide and stays per patch: mt's words are the patches' rows in
        # spatial-merge order, so normalizing the [n, H] view then reading the [n/4, 4H] blocks
        # is exactly the tower's norm(x) followed by x.view(-1, 4H)
        self._ln(x, self.m_norm, V(self.mt.off, n, H, 4 * H))
        mt = self.mt.rowv(0, nm)
        p.gemm_rc(*self._quant(mt), self.m_fc1, self.mm.rowv(0, nm), C.ACT_GELU)
        out = self.out.rowv(0, nm)
        self._linear(self.mm.rowv(0, nm), self.m_fc2, out)
        assert p.top <= self.scratch_end, "scratch_bytes too small for one image"
        p.top = self.end
        return out

    def run(self, patches: np.ndarray, pos: np.ndarray, rope: np.ndarray,
            blocks: int | None = None) -> np.ndarray:
        """Lower, build and run the golden model: the visual tokens [P/4, out_hidden]. With
        `blocks` only that many transformer blocks run, and the patches' hidden state [P, H]
        comes back instead — what the reference's per-block outputs are compared with."""
        n = patches.shape[0]
        keep = self.blocks
        if blocks is not None:
            self.blocks = keep[:blocks]
        try:
            self.p.descs.clear()          # a second image rebuilds the queue where it stood
            self.p.queue_off = None
            out = self._emit_image(patches, pos, rope)
            st, mach = self.p.run()
            assert st.done, (st.err_code, st.op_idx)
            return self._read(mach, out if blocks is None else self.x.rowv(0, n))
        finally:
            self.blocks = keep

    def _read(self, m, v: V) -> np.ndarray:
        a = self.p.base + v.off
        return np.stack([np.frombuffer(m.read(a + r * v.ld, 4 * v.cols), "<f4") for r in range(v.rows)])
