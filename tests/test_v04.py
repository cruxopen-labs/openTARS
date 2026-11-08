"""Contract v0.4 on the golden model: the new ops' numerics, encodings, error rules, and the Qwen3.5
LA + FA blocks against the transformers implementation (when torch is installed)."""
from __future__ import annotations

import numpy as np
import pytest

from tars_mini import contract as C
from tars_mini import numerics as F
from tars_mini.golden import ContractError, Machine
from tars_mini.llm import Dev, V

F32 = np.float32
QA = 0x7000


def _run(p: Dev):
    ddr, q, n = p.build()
    m = Machine(bytearray(ddr))
    return m.run_queue(q, n), m


def _rd(p: Dev, m: Machine, v: V) -> np.ndarray:
    a = p.base + v.off
    dt = {C.OUT_FP32: "<f4", C.OUT_BF16: "<u2", C.OUT_INT8: "i1"}[v.fmt]
    return np.stack([np.frombuffer(m.read(a + r * v.ld, v.cols * C.ESIZE[v.fmt]), dt) for r in range(v.rows)])


def _rand(rng, *shape, scale=2.0):
    return (rng.normal(0, scale, shape)).astype(F32)


# ---------------------------------------------------------------------- numerics
def test_sigmoid_softplus_accuracy():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.uniform(-40, 40, 200000), rng.uniform(-1, 1, 50000)]).astype(F32)
    sp = F.softplus_f(x)
    ref = np.logaddexp(0, x.astype(np.float64))
    assert (np.abs(sp - ref) / np.spacing(ref.astype(F32))).max() <= 6
    sg = F.sigmoid_f(x)
    ref = 1 / (1 + np.exp(-x.astype(np.float64)))
    assert (np.abs(sg - ref) / np.maximum(ref, 1e-30)).max() < 1e-6
    sp = F.softplus_f(np.array([np.nan, np.inf, -np.inf, 0, 21, 20], F32))
    assert np.isnan(sp[0]) and sp[1] == np.inf and sp[2] == 0 and sp[4] == 21
    assert F.act_fn(np.array([1.0], F32), C.ACT_EXP)[0] == F.exp_f(np.array([1.0], F32))[0]


def test_quant_rows():
    x = np.array([[1, -2, 0.5] + [0] * 13, [0] * 16, [127, -127.5] + [3] * 14], F32)
    q, s = F.quant_rows(x)
    assert q.dtype == np.int8 and np.abs(q.astype(int)).max() <= 127
    assert (q[1] == 0).all() and s[1] == 0                               # zero row
    assert q[0, 1] == -127 and s[0] == F.fmul(F32(2), F.INV127)
    assert np.abs(q.astype(np.float64) * s[:, None] - x).max() < 0.6     # within half a step


def test_fdot_orders_and_rope_match_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    a, b = _rand(rng, 3, 64), _rand(rng, 5, 64)
    # NT: the 16-lane order, written out independently
    lanes = np.zeros((3, 5, 16), F32)
    for k in range(64):
        lanes[:, :, k % 16] = F.fadd(lanes[:, :, k % 16], F.fmul(a[:, None, k], b[None, :, k]))
    while lanes.shape[2] > 1:
        lanes = F.fadd(lanes[:, :, 0::2], lanes[:, :, 1::2])
    assert (F.fdot_nt(a, b) == lanes[:, :, 0]).all()
    assert np.allclose(F.fdot_nt(a, b), a.astype(np.float64) @ b.T.astype(np.float64), rtol=1e-5, atol=1e-4)
    bn = _rand(rng, 64, 32)
    assert np.allclose(F.fdot_nn(a, bn), a.astype(np.float64) @ bn.astype(np.float64), rtol=1e-5, atol=1e-4)
    # RoPE: torch's rotate_half form, elementwise in fp32 (no fused multiply-add), bit for bit
    x, R, G = _rand(rng, 6, 128), 64, 2
    pos = np.arange(3, dtype=F32)
    inv = (1.0 / (1e7 ** (np.arange(0, R, 2, dtype=F32) / F32(R)))).astype(F32)
    fr = (pos[:, None] * inv[None, :]).astype(F32)
    tab = np.concatenate([np.cos(fr), np.sin(fr)], 1).astype(F32)
    tx = torch.from_numpy(x)
    cos = torch.from_numpy(np.repeat(np.cos(fr), G, 0))
    sin = torch.from_numpy(np.repeat(np.sin(fr), G, 0))
    cos, sin = torch.cat([cos, cos], -1), torch.cat([sin, sin], -1)
    xr = tx[:, :R]
    rot = torch.cat([-xr[:, R // 2:], xr[:, :R // 2]], -1)
    ref = torch.cat([xr * cos + rot * sin, tx[:, R:]], -1).numpy()
    assert (F.rope_rows(x, tab, G, R).view(np.uint32) == F.ftz(ref).view(np.uint32)).all()


def test_softmax_causal_rows():
    rng = np.random.default_rng(2)
    x = _rand(rng, 4, 32)
    y = F.softmax_rows_causal(x, [3, 4, 5, 32])
    for m, n in enumerate([3, 4, 5, 32]):
        e = np.exp(x[m, :n].astype(np.float64) - x[m, :n].max())
        assert np.allclose(y[m, :n], e / e.sum(), rtol=1e-5) and (y[m, n:] == 0).all()
    assert (F.softmax_rows_causal(x, [32] * 4) == F.softmax_rows(x)).all()


# ---------------------------------------------------------------------- ops through descriptors
@pytest.mark.parametrize("op", [C.ELT_MUL, C.ELT_ADD, C.ELT_SUB, C.ELT_MULADD])
@pytest.mark.parametrize("bmode", [C.BM_DENSE, C.BM_ROW, C.BM_COL, C.BM_SCALAR])
def test_elt_modes(op, bmode):
    rng = np.random.default_rng(op * 4 + bmode)
    M, N = 5, 32
    p = Dev(ddr_bytes=1 << 16)
    a = p.const(_rand(rng, M, N))
    bshape = {C.BM_DENSE: (M, N), C.BM_ROW: (1, N), C.BM_COL: (1, M + 3), C.BM_SCALAR: (1, 16)}[bmode]
    b = p.const(_rand(rng, *bshape))
    c = p.const(_rand(rng, M, N)) if op == C.ELT_MULADD else None
    y = p.mat(M, N)
    bv = b if bmode in (C.BM_DENSE, C.BM_ROW) else V(b.off + 4, 1, 1, 4)    # 4-byte aligned scalars
    p.elt(op, a, bv, y, bmode=bmode, act=C.ACT_SIGMOID, c=c)
    st, m = _run(p)
    assert st.done
    A = F.sigmoid_f(_rd(p, m, a))
    Bfull = _rd(p, m, b)
    B = {C.BM_DENSE: lambda: Bfull, C.BM_ROW: lambda: np.broadcast_to(Bfull, (M, N)),
         C.BM_COL: lambda: np.broadcast_to(Bfull[0, 1:M + 1, None], (M, N)),
         C.BM_SCALAR: lambda: np.broadcast_to(Bfull[0, 1], (M, N))}[bmode]()
    ref = {C.ELT_MUL: lambda: F.fmul(A, B), C.ELT_ADD: lambda: F.fadd(A, B),
           C.ELT_SUB: lambda: F.fadd(A, F._neg(B)),
           C.ELT_MULADD: lambda: F.fadd(_rd(p, m, c), F.fmul(A, B))}[op]()
    assert (_rd(p, m, y).view(np.uint32) == ref.view(np.uint32)).all()


def test_elt_in_place_and_row_broadcast_a():
    """S = S + k (x) d (the delta rule's update): A broadcast as a row, B one value per row, C = Y."""
    rng = np.random.default_rng(5)
    p = Dev(ddr_bytes=1 << 16)
    S0 = _rand(rng, 16, 32)
    S, d, k = p.const(S0), p.const(_rand(rng, 1, 32)), p.const(_rand(rng, 1, 16))
    p.elt(C.ELT_MULADD, d, k, S, bmode=C.BM_COL, arow=True, c=S)
    p.elt(C.ELT_MUL, S, V(k.off, 1, 1, 4), S, bmode=C.BM_SCALAR)            # in place through A
    st, m = _run(p)
    assert st.done
    kv, dv = _rd(p, m, k)[0], _rd(p, m, d)[0]
    ref = F.fmul(F.fadd(S0, F.fmul(dv[None, :], kv[:, None])), kv[0])
    assert (_rd(p, m, S) == ref).all()


def test_partial_overlap_is_refused():
    p = Dev(ddr_bytes=1 << 16)
    a = p.mat(4, 16)
    p.elt(C.ELT_ADD, a, a, a.rowv(1, 3).__class__(a.off + 64, 3, 16, 64))  # y = rows 1..3 of a
    ddr, q, n = p.build()
    with pytest.raises(ContractError):
        Machine(bytearray(ddr)).run_queue(q, n)


def test_gemm_rowcol_and_quant_through_descriptors():
    from tars_mini.compiler import quantize_weight
    from tars_mini.llm import QLin
    rng = np.random.default_rng(6)
    x = _rand(rng, 7, 64)
    W = rng.normal(0, 0.3, (32, 64))
    p = Dev(ddr_bytes=1 << 17)
    xv = p.const(x)
    q, s = p.quant(xv)
    qw = quantize_weight(W)
    lin = QLin(p.put(qw.q, "i1"), p.put(qw.scale), p.put(np.ones(32, F32)), 32, 64)
    y = p.mat(7, 32)
    p.gemm_rc(q, s, lin, y, act=C.ACT_SOFTPLUS)
    st, m = _run(p)
    assert st.done
    qq, ss = F.quant_rows(x)
    assert (_rd(p, m, q) == qq).all()
    acc = qq.astype(np.int64) @ qw.q.astype(np.int64).T
    ref = F.softplus_f(F.fadd(F.fmul(F.fmul(F.i2f(acc), ss[:, None]), qw.scale[None, :]), F32(1)))
    assert (_rd(p, m, y) == ref).all()


def _desc(**f):
    return C.encode({"MAGIC": C.MAGIC, "VERSION": C.VERSION, "VALID": 1, "A_OFF": 0x100, "Y_OFF": 0x800,
                     "W_OFF": 0x1000, "M": 2, "N": 16, **f})


@pytest.mark.parametrize("fields,code", [
    (dict(OPCODE=C.OP_ELT, CTRL=4), C.E_FIELD),                                   # CTRL[2]
    (dict(OPCODE=C.OP_ELT, HDR_RSV=16), C.E_FIELD),                               # beyond broadcast + FP16 A
    (dict(OPCODE=C.OP_ELT, CTRL=C.ELT_MULADD), C.E_FIELD),                        # MULADD without C
    (dict(OPCODE=C.OP_ELT, EPI=C.epi(bias=1), BIAS_OFF=0x400), C.E_FIELD),        # C without MULADD
    (dict(OPCODE=C.OP_ELT, CTRL=C.ELT_MULADD, EPI=C.epi(out=C.OUT_BF16, bias=1), BIAS_OFF=0x400), C.E_FIELD),
    (dict(OPCODE=C.OP_ELT, K=16), C.E_FIELD),
    (dict(OPCODE=C.OP_ELT, W_OFF=0x1004), C.E_ALIGN),                             # dense B
    (dict(OPCODE=C.OP_ELT, W_OFF=0x1004, HDR_RSV=C.BM_SCALAR), C.E_OK),           # scalar: 4-byte
    (dict(OPCODE=C.OP_ELT, W_OFF=0x1002, HDR_RSV=C.BM_COL), C.E_ALIGN),
    (dict(OPCODE=C.OP_ELT, GROUP_SIZE=1), C.E_ALIGN),                             # ldb < row
    (dict(OPCODE=C.OP_ROPE, K=16, GROUP_SIZE=1), C.E_ALIGN),                      # R % 32
    (dict(OPCODE=C.OP_ROPE, K=32, N=16, GROUP_SIZE=1), C.E_ALIGN),                # R > N
    (dict(OPCODE=C.OP_ROPE, K=16), C.E_FIELD),                                    # G = 0
    (dict(OPCODE=C.OP_ROPE, K=16, GROUP_SIZE=1, EPI=C.epi(C.ACT_EXP)), C.E_FIELD),
    (dict(OPCODE=C.OP_FDOT, K=16, CTRL=2), C.E_FIELD),
    (dict(OPCODE=C.OP_FDOT, K=8), C.E_ALIGN),
    (dict(OPCODE=C.OP_FDOT, K=16, EPI=C.epi(bias=1), BIAS_OFF=0x400), C.E_FIELD),
    (dict(OPCODE=C.OP_FDOT, K=16), C.E_OK),
    (dict(OPCODE=C.OP_ROW, CTRL=5, SCALE_OFF=0x400), C.E_FIELD),                  # fn 5
    (dict(OPCODE=C.OP_ROW, CTRL=C.ROW_QUANT, SCALE_OFF=0x400, W_OFF=0), C.E_FIELD),  # QUANT out fp32
    (dict(OPCODE=C.OP_ROW, CTRL=C.ROW_QUANT, SCALE_OFF=0x400, W_OFF=0, EPI=C.epi(out=C.OUT_INT8)), C.E_OK),
    (dict(OPCODE=C.OP_ROW, CTRL=C.ROW_QUANT, SCALE_OFF=0x400, EPI=C.epi(out=C.OUT_INT8)), C.E_FIELD),
    (dict(OPCODE=C.OP_ROW, CTRL=C.ROW_RMSNORM, SCALE_OFF=0x400, K=1), C.E_FIELD),  # K: softmax only
    (dict(OPCODE=C.OP_ROW, CTRL=C.ROW_SOFTMAX, W_OFF=0, K=3), C.E_OK),            # causal
    (dict(OPCODE=C.OP_ROW, CTRL=C.ROW_L2NORM, SCALE_OFF=0x400, W_OFF=C.f32_bits(1e-6)), C.E_OK),
    (dict(OPCODE=C.OP_GEMM, CTRL=C.ctrl(C.QM_ROWCOL), K=16, SCALE_OFF=0x400, HDR_RSV=0x50), C.E_OK),
    (dict(OPCODE=C.OP_CONV, CTRL=C.ctrl(C.QM_ROWCOL), K=16), C.E_FIELD),          # conv: no ROWCOL
    (dict(OPCODE=C.OP_GEMM, CTRL=C.ctrl(C.QM_CHANNEL), K=16, HDR_RSV=1), C.E_FIELD),
    (dict(OPCODE=C.OP_RELU, EPI=C.epi(C.ACT_SOFTPLUS), W_OFF=0), C.E_OK),
    (dict(OPCODE=0x09), C.E_OPCODE),
])
def test_v04_rules(fields, code):
    ddr = bytearray(1 << 16)
    ddr[QA:QA + 64] = _desc(**fields)
    st = Machine(ddr).run_queue(QA, 1)
    assert st.err_code == code, (fields, st)


# ---------------------------------------------------------------------- the Qwen3.5 blocks
@pytest.mark.parametrize("cfgname,chunks", [("small", [8, 1, 1, 1]), ("small", [1, 1, 3, 16]),
                                            ("08b", [16, 1, 1])])
def test_qwen_blocks_match_transformers(cfgname, chunks):
    """LA + FA + final norm on the golden model vs transformers with the same W8A8 linears:
    each block's update (output - input) within 2e-5 of the reference's (fp32 order only)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tars_mini import llm as L
    from tars_mini import qwen_ref as Q
    cfg = Q.make_config(Q.CFG_SMALL if cfgname == "small" else Q.CFG_08B)
    w = Q.random_weights(cfg)
    T = sum(chunks)
    x = Q.random_inputs(cfg, T)
    low = L.QwenLowering(cfg, w, max_tokens=T, ddr_bytes=1 << (26 if cfgname == "small" else 28))
    low.input(x)
    t0 = 0
    for c in chunks:
        low.chunk(t0, c)
        t0 += c
    ddr, q, n = low.prog.build()
    m = Machine(bytearray(ddr))
    assert m.run_queue(q, n).done
    rq = Q.run_reference(cfg, w, x, chunks, fake_quant=True)
    t0 = 0
    for i, c in enumerate(chunks):
        prev, prev_r = x[t0:t0 + c], x[t0:t0 + c]
        for lay in (0, 1):
            g = low.read_layer(m, lay, t0, c)
            dg, dr = g - prev, rq.layer_out[i][lay] - prev_r
            assert np.abs(dg - dr).max() <= 2e-5 * np.abs(dr).max(), (i, lay)
            prev, prev_r = g, rq.layer_out[i][lay]
        fin = low.read_final(m, t0, c)
        assert np.abs(fin - rq.final[i]).max() <= 2e-5 * np.abs(rq.final[i]).max()
        t0 += c


def test_qwen_model_with_head_greedy_matches_transformers():
    """Eight layers (the 0.8B pattern LA LA LA FA, twice; small dims), final norm, tied head
    (vocabulary 4096): greedy decoding with the host in the loop on the golden model; logits at
    every step within 1e-5 (relative to the largest) of transformers with the same W8A8 linears
    and head, and the same tokens."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tars_mini import llm as L
    from tars_mini import qwen_ref as Q
    cfg = Q.make_config(Q.CFG_SMALL, Q.LAYERS_08B[:8], vocab=4096)
    w = Q.random_weights(cfg, embed=True)
    low = L.QwenLowering(cfg, w, max_tokens=16, ddr_bytes=1 << 25, head=True, queue_cap=4096)

    def run(ddr, q, n):
        m = Machine(ddr)
        assert m.run_queue(q, n).done
        return m.ddr
    prompt = [5, 77, 901, 12, 3000]
    ids, logits, _ = low.generate(run, prompt, 3)
    ref = Q.run_reference(cfg, w, low.embed(ids[:-1]), [len(prompt), 1, 1, 1], fake_quant=True, head=True)
    for i, (g, r) in enumerate(zip(logits, ref.logits)):
        assert np.abs(g - r).max() <= 1e-5 * np.abs(r).max(), i
        assert int(np.argmax(r)) == ids[len(prompt) + i]


def test_qwen3_model_with_head_greedy_matches_transformers():
    """Qwen3 (every layer full attention, no output gate, plain RMSNorm weights, full rotary):
    four layers (small dims), final norm, tied head (vocabulary 4096 in one block, and 151936's
    split checked): greedy decoding on the golden model; logits at every step within 1e-5 of
    transformers' Qwen3Model with the same W8A8 linears and head, and the same tokens."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tars_mini import llm as L
    from tars_mini import qwen_ref as Q
    assert Q.head_rows(Q.VOCAB_Q3_06B) == 37984 and Q.head_rows(Q.VOCAB_08B) == Q.HEAD_ROWS
    cfg = Q.make_config(Q.CFG_Q3_SMALL, ["full_attention"] * 4, vocab=4096)
    w = Q.random_weights(cfg, embed=True)
    low = L.QwenLowering(cfg, w, max_tokens=16, ddr_bytes=1 << 25, head=True, queue_cap=4096)

    def run(ddr, q, n):
        m = Machine(ddr)
        assert m.run_queue(q, n).done
        return m.ddr
    prompt = [5, 77, 901, 12, 3000]
    ids, logits, _ = low.generate(run, prompt, 3)
    ref = Q.run_reference(cfg, w, low.embed(ids[:-1]), [len(prompt), 1, 1, 1], fake_quant=True, head=True)
    for i, (g, r) in enumerate(zip(logits, ref.logits)):
        assert np.abs(g - r).max() <= 1e-5 * np.abs(r).max(), i
        assert int(np.argmax(r)) == ids[len(prompt) + i]


@pytest.mark.parametrize("chunks", [[5, 1, 1, 1], [9, 1, 1]])
def test_gemma3_model_with_head_greedy_matches_transformers(chunks):
    """Gemma 3 (sliding and global layers with their own RoPE bases, four norms per layer, GeGLU,
    the scaled embedding; small dims, intermediate 4608 so the down projection runs as two K
    blocks; the KV260 build's GEMV-only rules): greedy decoding on the golden model; each layer's
    update within 2e-5 and the logits within 1e-5 of transformers' Gemma3TextModel with the same
    W8A8 linears and head, and the same tokens."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tars_mini import llm as L
    from tars_mini import qwen_ref as Q
    cfg = Q.make_config(dict(Q.CFG_G3_SMALL, intermediate_size=4608),
                        ["sliding_attention", "full_attention", "sliding_attention"], vocab=4096)
    w = Q.random_weights(cfg, embed=True)
    low = L.QwenLowering(cfg, w, max_tokens=16, ddr_bytes=1 << 26, head=True, queue_cap=4096)
    assert [d.k for d in low.lw[0]["down"]] == [2304, 2304]

    def run(ddr, q, n):
        m = Machine(ddr)
        m.gemv_only = True
        assert m.run_queue(q, n).done
        return m.ddr
    prompt = [5, 77, 901, 12, 3000, 2, 9, 40, 1][:chunks[0]]
    ids, logits, _ = low.generate(run, prompt, len(chunks) - 1)
    x = low.embed(ids[:-1])
    assert np.array_equal(x[0], (w["embed_tokens.weight"][ids[0]] * np.float32(16.0)).astype(np.float32))
    ref = Q.run_reference(cfg, w, x, [len(prompt)] + [1] * (len(chunks) - 1), fake_quant=True, head=True)
    for i, (g, r) in enumerate(zip(logits, ref.logits)):
        assert np.abs(g - r).max() <= 1e-5 * np.abs(r).max(), i
        assert int(np.argmax(r)) == ids[len(prompt) + i]


def test_gemma3_blocks_match_transformers():
    """Gemma 3's layers vs transformers with the same W8A8 linears, chunks of several sizes (the
    sliding and global RoPE tables at offsets): each layer's update within 2e-5 of the reference's."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tars_mini import llm as L
    from tars_mini import qwen_ref as Q
    cfg = Q.make_config(Q.CFG_G3_SMALL, ["sliding_attention", "full_attention"])
    w = Q.random_weights(cfg)
    chunks = [7, 1, 3, 16]
    T = sum(chunks)
    x = Q.random_inputs(cfg, T)
    low = L.QwenLowering(cfg, w, max_tokens=T, ddr_bytes=1 << 26)
    low.input(x)
    t0 = 0
    for c in chunks:
        low.chunk(t0, c)
        t0 += c
    ddr, q, n = low.prog.build()
    m = Machine(bytearray(ddr))
    assert m.run_queue(q, n).done
    rq = Q.run_reference(cfg, w, x, chunks, fake_quant=True)
    t0 = 0
    for i, c in enumerate(chunks):
        prev, prev_r = x[t0:t0 + c], x[t0:t0 + c]
        for lay in (0, 1):
            g = low.read_layer(m, lay, t0, c)
            dg, dr = g - prev, rq.layer_out[i][lay] - prev_r
            assert np.abs(dg - dr).max() <= 2e-5 * np.abs(dr).max(), (i, lay)
            prev, prev_r = g, rq.layer_out[i][lay]
        fin = low.read_final(m, t0, c)
        assert np.abs(fin - rq.final[i]).max() <= 2e-5 * np.abs(rq.final[i]).max()
        t0 += c


@pytest.mark.parametrize("cfgname", ["CFG_Q3_SMALL"])
def test_speculative_verify_bit_identical_to_plain_decode(cfgname):
    """Greedy speculative decoding on Qwen3: chunks of up to 4 tokens at any position, every row's
    logits, rejected drafts left in the KV cache and overwritten later. Every accepted position's
    logits must equal plain one-token decoding bit for bit, so the tokens are the same."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tars_mini import llm as L
    from tars_mini import qwen_ref as Q
    cfg = Q.make_config(getattr(Q, cfgname), ["full_attention"] * 3, vocab=4096)
    w = Q.random_weights(cfg, embed=True)
    mk = lambda: L.QwenLowering(cfg, w, max_tokens=48, ddr_bytes=1 << 25, head=True, queue_cap=4096, verify_rows=4)

    def run(ddr, q, n):
        m = Machine(ddr)
        assert m.run_queue(q, n).done
        return m.ddr
    prompt, new = [5, 77, 901, 12, 3000], 24
    a = mk()
    ids, plain, _ = a.generate(run, prompt, new)     # plain[i]: the logits after position len(prompt) - 1 + i
    b = mk()
    ddr = b.p.ddr
    rng = np.random.default_rng(1)

    def go(t0, toks, all_rows):
        nonlocal ddr
        b.put_input(ddr, t0, b.embed(toks))
        s = len(b.p.descs)
        b.chunk(t0, len(toks), all_rows=all_rows)
        q, c = b.stage(ddr, s, len(b.p.descs))
        ddr = run(ddr, q, c)
    go(0, prompt, False)
    out = list(prompt)
    lg = b.logits_of(ddr)
    assert lg.tobytes() == plain[0].tobytes()
    out.append(int(np.argmax(lg)))
    rejected = 0
    while len(out) < len(ids):
        t = len(out) - 1                             # the last token is not in the cache yet
        k = min(int(rng.integers(0, 4)), len(prompt) + new - 1 - t)   # drafts after it
        right = ids[t + 1:t + 1 + k]
        cut = int(rng.integers(0, len(right) + 1))   # the first `cut` drafts right, the next wrong
        draft = list(right[:cut]) + [(int(v) + 1) % 4096 for v in right[cut:]]
        toks = [out[-1]] + draft
        go(t, toks, True)
        rows = b.logits_rows(ddr, len(toks))
        for i in range(len(toks)):
            assert rows[i].tobytes() == plain[t + i + 1 - len(prompt)].tobytes(), (t, i)
            nxt = int(np.argmax(rows[i]))
            out.append(nxt)
            if i == len(draft) or draft[i] != nxt:
                rejected += i < len(draft)
                break
    assert out[:len(ids)] == ids and rejected >= 2


@pytest.mark.parametrize("state_dtype", ["fp32", "fp16"])
def test_mtp_speculative_bit_identical_to_plain_decode(state_dtype):
    """Qwen3.5 (LA and FA layers) with the MTP head: greedy speculative decoding gives the tokens of
    plain decoding and, at every position it verified, the same logits bit for bit, through hits
    and misses alike (the oracle forces both: a miss must leave the DeltaNet state as it was)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tars_mini import llm as L
    from tars_mini import qwen_ref as Q
    cfg = Q.make_config(Q.CFG_SMALL, Q.LAYERS_08B[:4], vocab=4096)
    w = {**Q.random_weights(cfg, embed=True), **Q.random_mtp(cfg)}
    mk = lambda **k: L.QwenLowering(cfg, w, max_tokens=48, ddr_bytes=1 << 25, head=True, queue_cap=8192,
                                   state_dtype=state_dtype, **k)

    def run(ddr, q, n):
        m = Machine(ddr)
        assert m.run_queue(q, n).done
        return m.ddr
    prompt, new = [5, 77, 901, 12, 3000], 20
    ids, plain, _ = mk().generate(run, prompt, new)
    rng = np.random.default_rng(2)
    hits = []

    def oracle(t, d):                                # ids[t + 1] is the right draft for row t + 1
        right = rng.random() < 0.5 and t + 1 < len(ids)
        hits.append(right)
        return ids[t + 1] if right else (ids[min(t + 1, len(ids) - 1)] + 1) % 4096
    got, lg, rounds = mk(verify_rows=2, mtp=True).generate_mtp(run, prompt, new, oracle)
    assert got[:len(ids)] == ids
    for pos, row in lg.items():
        if pos - len(prompt) + 1 < len(plain):
            assert row.tobytes() == plain[pos - len(prompt) + 1].tobytes(), pos
    assert 3 <= sum(hits) <= len(hits) - 3 and rounds < new


def test_swap_state_copies_exactly():
    """swap_state: the second DeltaNet state copy into the first, bit for bit (signed zeros, tiny and
    huge values), so plain queues can continue after a speculative run."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tars_mini import llm as L
    from tars_mini import qwen_ref as Q
    cfg = Q.make_config(Q.CFG_SMALL, ["linear_attention", "full_attention"], vocab=64)
    w = {**Q.random_weights(cfg, embed=True), **Q.random_mtp(cfg)}
    low = L.QwenLowering(cfg, w, max_tokens=8, ddr_bytes=1 << 24, head=True, queue_cap=2048, verify_rows=2, mtp=True)
    ddr = low.p.ddr
    rng = np.random.default_rng(3)
    S = low.st[0]["S"]
    vals = []
    for v in S[1]:
        a = (rng.normal(0, 1, (v.rows, v.cols)) * 10.0 ** rng.integers(-30, 30, (v.rows, v.cols))).astype(F32)
        a[0, :4] = [0.0, -0.0, 1e-37, -3e38]
        vals.append(a.tobytes())
        o = low.p.base + v.off
        ddr[o:o + len(vals[-1])] = vals[-1]
    a0 = len(low.p.descs)
    low.swap_state()
    q, n = low.stage(ddr, a0, len(low.p.descs))
    m = Machine(ddr)
    assert m.run_queue(q, n).done
    for v, b in zip(S[0], vals):
        o = low.p.base + v.off
        assert bytes(m.ddr[o:o + len(b)]) == b

