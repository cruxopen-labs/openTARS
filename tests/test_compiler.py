"""Compiler: emitted programs are legal, compute the right thing, and do not depend on how the
work is split into descriptors, whether the activation is fused, or where the program sits."""
import numpy as np
import pytest

from tars_mini import compiler as Cc
from tars_mini import contract as C
from tars_mini.golden import Machine


def _layer(seed, M, K, N, per_channel=True):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, (M, K))
    w = rng.normal(0, 0.2, (N, K))
    b = rng.normal(0, 0.5, N)
    s_x = Cc.act_scale(x)
    return x, Cc.quantize_weight(w, per_channel), b, s_x


def _run_linear(seed, M, K, N, per_channel=True, **kw):
    x, qw, b, s_x = _layer(seed, M, K, N, per_channel)
    p = Cc.Program(**{k: kw.pop(k) for k in ("base_addr",) if k in kw})
    xin = p.input_int8(Cc.quantize_act(x, s_x), s_x)
    y = p.linear(xin, qw, bias=b, **kw)
    st, m = p.run()
    assert st.done, st
    return p, m, y, (x, qw, b, s_x)


def test_every_emitted_descriptor_validates():
    p, m, y, _ = _run_linear(0, 7, 50, 10, act="relu", out="int8", out_scale=0.05)
    ddr, q, n = p.build()
    for i in range(n):
        err, _ = Machine(ddr).validate(bytes(ddr[q + 64 * i:q + 64 * (i + 1)]))
        assert err == C.E_OK


@pytest.mark.parametrize("M,K,N", [(1, 16, 16), (5, 50, 10), (33, 64, 130), (3, 200, 7)])
@pytest.mark.parametrize("per_channel", [True, False])
def test_linear_matches_float64_on_the_same_integers(M, K, N, per_channel):
    """Padding K and N is invisible; the fp32 epilogue is within a few ulp of float64."""
    p, m, y, (x, qw, b, s_x) = _run_linear(M + K + N, M, K, N, per_channel, out="fp32")
    got = p.read(m, y).astype(np.float64)
    assert got.shape == (M, N)
    acc = Cc.quantize_act(x, s_x).astype(np.int64) @ qw.q.astype(np.int64).T
    ref = acc * (s_x * qw.scale.astype(np.float64)) + b
    assert np.allclose(got, ref, rtol=1e-6, atol=1e-6 * np.abs(ref).max())


@pytest.mark.parametrize("out,act", [("fp32", "relu"), ("bf16", "relu6"), ("int8", "relu")])
@pytest.mark.parametrize("m_split", [1, 3, 16])
def test_result_independent_of_split_fusion_and_base(out, act, m_split):
    kw = dict(act=act, out=out, out_scale=0.02 if out == "int8" else None)
    p0, m0, y0, _ = _run_linear(7, 37, 48, 40, **kw)
    ref = p0.read(m0, y0).tobytes()
    for variant in (dict(m_split=m_split), dict(fused=False), dict(fused=False, m_split=m_split),
                    dict(base_addr=1 << 20)):
        p, m, y, _ = _run_linear(7, 37, 48, 40, **kw, **variant)
        assert p.read(m, y).tobytes() == ref, variant


def test_unfused_emits_gemm_then_relu():
    p, _, _, _ = _run_linear(1, 4, 16, 16, act="relu", out="bf16", fused=False)
    ops = [C.decode(r)["OPCODE"] for r in p.descs]
    assert ops == [C.OP_GEMM, C.OP_RELU]


def test_int8_chain_carries_its_scale():
    x, qw, b, s_x = _layer(3, 4, 32, 32)
    p = Cc.Program()
    h = p.linear(p.input_int8(Cc.quantize_act(x, s_x), s_x), qw, bias=b, act="relu",
                 out="int8", out_scale=0.03)
    assert h.scale == 0.03 and h.fmt == C.OUT_INT8
    p.linear(h, Cc.quantize_weight(np.ones((16, 32)) * 0.1), out="fp32")
    st, _ = p.run()
    assert st.done


def test_compiler_refuses_bad_inputs():
    p = Cc.Program()
    with pytest.raises(ValueError):
        p.linear(p.input_fp32(np.zeros((2, 16))), Cc.quantize_weight(np.ones((16, 16))))
    xin = p.input_int8(np.zeros((2, 16), np.int8), 1.0)
    with pytest.raises(ValueError):
        p.linear(xin, Cc.quantize_weight(np.ones((16, 32))))           # K mismatch
    with pytest.raises(ValueError):
        p.linear(xin, Cc.quantize_weight(np.ones((16, 16))), out="int8")  # no out_scale


# ---------------------------------------------------------------------- v0.2: N-splitting
@pytest.mark.parametrize("n_split", [16, 32, 48])
@pytest.mark.parametrize("out,act,fused", [("fp32", "none", True), ("int8", "relu", True),
                                           ("bf16", "relu6", False)])
def test_n_split_is_bit_identical(n_split, out, act, fused):
    kw = dict(act=act, out=out, out_scale=0.02 if out == "int8" else None, fused=fused)
    p0, m0, y0, _ = _run_linear(11, 9, 64, 70, **kw)
    ref = p0.read(m0, y0).tobytes()
    for variant in (dict(n_split=n_split), dict(n_split=n_split, m_split=4)):
        p, m, y, _ = _run_linear(11, 9, 64, 70, **kw, **variant)
        assert p.read(m, y).tobytes() == ref, variant
        if variant.get("n_split", 80) < 80:
            assert any(C.decode(r)["LDY16"] for r in p.descs)   # the stride is in use


def test_concat_by_writing_into_column_views():
    """Two GEMMs write the two halves of one buffer (strided outputs); a third GEMM reads the
    whole buffer. Same bytes as one GEMM with the weights stacked."""
    rng = np.random.default_rng(5)
    x = rng.uniform(-1, 1, (6, 32))
    w1, w2 = rng.normal(0, .3, (32, 32)), rng.normal(0, .3, (48, 32))
    s_x = Cc.act_scale(x)
    q1, q2 = Cc.quantize_weight(w1), Cc.quantize_weight(w2)
    qs = Cc.QWeight(np.concatenate([q1.q, q2.q]), np.concatenate([q1.scale, q2.scale]), 80, 32)

    p = Cc.Program()
    xin = p.input_int8(Cc.quantize_act(x, s_x), s_x)
    cat = Cc.Buf(p.alloc(6 * 80), 6, 80, C.OUT_INT8)
    p.linear(xin, q1, act="relu", out="int8", out_scale=0.05, into=cat.cols_view(0, 32))
    p.linear(xin, q2, act="relu", out="int8", out_scale=0.05, into=cat.cols_view(32, 80))
    st, m = p.run()
    assert st.done and all(C.decode(r)["LDY16"] == 5 for r in p.descs)    # 80-byte rows

    p2 = Cc.Program()
    y = p2.linear(p2.input_int8(Cc.quantize_act(x, s_x), s_x), qs, act="relu", out="int8",
                  out_scale=0.05)
    st2, m2 = p2.run()
    assert st2.done
    assert p.read(m, cat).tobytes() == p2.read(m2, y).tobytes()


def test_gemm_reads_a_column_view():
    """A GEMM whose input is columns [16, 48) of a wider INT8 tensor (LDA16 in use)."""
    rng = np.random.default_rng(6)
    wide = rng.integers(-127, 128, (5, 64)).astype(np.int8)
    qw = Cc.quantize_weight(rng.normal(0, .3, (16, 32)))
    p = Cc.Program()
    v = p.input_int8(wide, 0.1).cols_view(16, 48)
    y = p.linear(v, qw, out="fp32")
    st, m = p.run()
    assert st.done and C.decode(p.descs[0])["LDA16"] == 4
    p2 = Cc.Program()
    y2 = p2.linear(p2.input_int8(wide[:, 16:48], 0.1), qw, out="fp32")
    st2, m2 = p2.run()
    assert p.read(m, y).tobytes() == p2.read(m2, y2).tobytes()
