"""W4A16 GEMM on the RTL core vs the golden model (identical DDR bytes and end status): FP32 and
BF16 activations, K 128 .. 4096, M across unit passes, strided A, bias, activations (the shared
SFU's on the FPGA build), every output format, NaN / infinity groups, the malformed and out-of-range
descriptors, and a small W4A16 lowering program."""
from __future__ import annotations

import numpy as np
import pytest

from tars_mini import contract as C
from tars_mini import numerics as F
from tars_mini import rtlsim
from tars_mini.compiler import quantize_weight4
from tars_mini.llm import Dev, QLin, V

from test_rtl import MEM, both, expect_code

pytestmark = pytest.mark.skipif(not rtlsim.have_verilator(), reason="needs Verilator")
F32 = np.float32


def _lin(p: Dev, w: np.ndarray, bias: np.ndarray | None = None) -> QLin:
    qw = quantize_weight4(w)
    b = p.put(np.asarray(bias, F32)) if bias is not None else None
    return QLin(p.put(qw.q, "u1"), p.put(qw.scale, "<u2"), b, qw.n, qw.k, 4)


def _view(x: V, cols: int) -> V:
    """The first cols values of every row (a strided activation)."""
    return V(x.off, x.rows, cols, x.ld, x.fmt)


def _act(p: Dev, a: np.ndarray, half: bool) -> V:
    if half:
        return V(p.put(F.to_bf16_bits(a), "<u2"), a.shape[0], a.shape[1], 2 * a.shape[1], C.OUT_BF16)
    return p.const(a)


# The W4 unit reads RW / 4 whole groups of a channel a beat; the plain core's testbench takes the
# width as a generic, the KV260 top fixes it in RTL.
RWP: dict = {"RW": 8} if rtlsim.TOP == "core" else {}


def _both(p: Dev, **kw):
    ddr, q, n = p.build()
    kw.setdefault("params", {}).update(RWP)
    return both(bytes(ddr), q, n, **kw)


def _int8_scale(p: Dev, s: float) -> None:
    p.descs[-1] = C.encode({**C.decode(p.descs[-1]), "OUT_SCALE": C.f32_bits(s)})


def _w(rng, *s, scale=1.0):
    return rng.normal(0, scale, s).astype(F32)


@pytest.mark.parametrize("out", [C.OUT_FP32, C.OUT_BF16, C.OUT_INT8])
@pytest.mark.parametrize("half", [False, True])
def test_w4_gemm_on_rtl(half, out):
    """FP32 / BF16 activations x every output format, strided A, K = 256, M = 3, bias, ReLU."""
    rng = np.random.default_rng(1000 + int(half) * 10 + out)
    M, N, K = 3, 32, 256
    p = Dev(ddr_bytes=MEM)
    aw = _act(p, _w(rng, M, K + 8), half)                     # rows of K + 8 values
    a = _view(aw, K)
    wl = _lin(p, _w(rng, N, K), _w(rng, N, scale=0.25))
    y = p.mat(M, N, out)
    p.gemm_w4(a, wl, y, act=C.ACT_RELU)
    if out == C.OUT_INT8:
        _int8_scale(p, 5.0)
    st, _ = _both(p, stall=30, lat=5)
    assert st.done


@pytest.mark.parametrize("half", [False, True])
def test_w4_k256_on_rtl(half):
    """The smallest GEMM: two groups of K (the wide beat's two halves), one channel of N."""
    rng = np.random.default_rng(70 + int(half))
    M, N, K = 1, 16, 2 * C.W4_GROUP
    p = Dev(ddr_bytes=MEM)
    a = _act(p, _w(rng, M, K), half)
    wl = _lin(p, _w(rng, N, K))
    y = p.mat(M, N, C.OUT_FP32)
    p.gemm_w4(a, wl, y)
    st, _ = _both(p)
    assert st.done


@pytest.mark.parametrize("act", [C.ACT_GELU, C.ACT_SILU, C.ACT_SIGMOID, C.ACT_SOFTPLUS, C.ACT_EXP])
def test_w4_activations_on_rtl(act):
    """The long activations (the shared SFU on the FPGA build) and softplus, M across two channels."""
    rng = np.random.default_rng(200 + act)
    M, N, K = 5, 32, 256
    p = Dev(ddr_bytes=MEM)
    a = _act(p, _w(rng, M, K, scale=3.0), False)
    wl = _lin(p, _w(rng, N, K))
    y = p.mat(M, N, C.OUT_FP32)
    p.gemm_w4(a, wl, y, act=act)
    st, _ = _both(p, stall=17)
    assert st.done


def test_w4_k4096_passes_on_rtl():
    """K = 4096 (the GEMV profile's limit) and M = 9: three passes of the unit."""
    rng = np.random.default_rng(9)
    M, N, K = 9, 16, 4096
    p = Dev(ddr_bytes=1 << 18)
    a = p.const(_w(rng, M, K, scale=0.5))
    wl = _lin(p, _w(rng, N, K))
    y = p.mat(M, N, C.OUT_FP32)
    p.gemm_w4(a, wl, y)
    st, _ = _both(p)
    assert st.done


def test_w4_nan_inf_groups_on_rtl():
    """A group with a NaN (all mantissas 0, canonical NaN scale), one with infinities, an all-zero
    group and subnormals: the contract's alignment rules, byte for byte."""
    a = np.zeros((2, 256), F32)
    a[0, 0:128] = [np.inf, -np.inf, 1.0, -2.5, 3e-40] + [0.5] * 123
    a[0, 128:256] = [np.nan] + [1.0] * 127
    a[1, 0:128] = 0.0
    a[1, 128:256] = [5e-40, -5e-40] + [1e-39] * 126
    rng = np.random.default_rng(11)
    p = Dev(ddr_bytes=MEM)
    x = p.const(a)
    wl = _lin(p, _w(rng, 16, 256))
    y = p.mat(2, 16, C.OUT_FP32)
    p.gemm_w4(x, wl, y)
    st, _ = _both(p)
    assert st.done


def _bad(**f):
    base = dict(OPCODE=C.OP_GEMM, CTRL=C.ctrl(C.QM_GROUP, a_dtype=C.DT_FP32, w_dtype=C.DT_INT4),
                K=256, GROUP_SIZE=C.W4_GROUP, SCALE_OFF=0x400, A_OFF=0x100, Y_OFF=0x4000,
                W_OFF=0x1000, M=2, N=16)
    return C.encode({"MAGIC": C.MAGIC, "VERSION": C.VERSION, "VALID": 1, **base, **f})


@pytest.mark.parametrize("fields,code", [
    (dict(), C.E_OK),
    (dict(CTRL=C.ctrl(C.QM_GROUP, a_dtype=C.DT_BF16, w_dtype=C.DT_INT4)), C.E_OK),
    (dict(CTRL=C.ctrl(C.QM_GROUP, a_dtype=C.DT_INT8, w_dtype=C.DT_INT4)), C.E_FIELD),
    (dict(GROUP_SIZE=64), C.E_FIELD),
    (dict(GROUP_SIZE=0), C.E_FIELD),
    (dict(HDR_RSV=1), C.E_FIELD),
    (dict(K=64), C.E_ALIGN),
    (dict(K=384 - 128 + 64), C.E_ALIGN),             # K % 256 != 0 (a wide beat splits a group)
    (dict(K=128), C.E_ALIGN),
    (dict(LDA16=16), C.E_ALIGN),                     # 256 bytes < one FP32 K=256 row (1024)
    (dict(A_OFF=0x104), C.E_ALIGN),
    (dict(W_OFF=0x1010), C.E_ALIGN),                 # 16-byte aligned but not 128-byte
    (dict(K=4096 + 128), C.E_ALIGN),                 # the KV260 profile's E_OPCODE first
])
def test_w4_rules_on_rtl(fields, code):
    """One descriptor per queue: the RTL's end status is the golden model's (the GEMV-only
    profile's E_OPCODE where it rejects first)."""
    p = Dev(ddr_bytes=MEM)
    p.descs += [_bad(**fields)]
    ddr, q, n = p.build()
    st, _ = both(bytes(ddr), q, n, params=dict(RWP))
    assert st.err_code == expect_code(_bad(**fields), code), (fields, st)


def test_w4_range_on_rtl():
    """The W4A16 extents (W = N K / 2, scales 2 N (K / 128), no row scale) push E_RANGE, and an
    in-range descriptor with the same shape runs."""
    def desc(s_off: int):
        return C.encode({"MAGIC": C.MAGIC, "VERSION": C.VERSION, "VALID": 1, "OPCODE": C.OP_GEMM,
                         "CTRL": C.ctrl(C.QM_GROUP, a_dtype=C.DT_FP32, w_dtype=C.DT_INT4),
                         "K": 256, "GROUP_SIZE": C.W4_GROUP, "A_OFF": 0x100, "Y_OFF": 0x4000,
                         "W_OFF": 0x1000, "SCALE_OFF": s_off, "M": 2, "N": 16})
    for s_off, code in ((0xFFF0, C.E_RANGE), (0x2000, C.E_OK)):
        p = Dev(ddr_bytes=MEM)
        p.descs += [desc(s_off)]
        ddr, q, n = p.build()
        st, _ = both(bytes(ddr), q, n, params=dict(RWP))
        assert st.err_code == code, (hex(s_off), st)


def test_w4_lowering_on_rtl():
    """The Qwen3.5 lowering with wbits = 4: every GEMM a W4A16 descriptor, prefill then decode,
    the RTL's DDR byte for byte the golden model's."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tars_mini import llm as L
    from tars_mini import qwen_ref as Q
    cfg = Q.make_config(dict(Q.CFG_G4_SMALL, hidden_size=256, intermediate_size=512,
                             hidden_size_per_layer_input=256),
                        ["sliding_attention", "full_attention"], vocab=1024)
    w = Q.random_weights(cfg, embed=True)
    low = L.QwenLowering(cfg, w, max_tokens=4, ddr_bytes=1 << 23, head=True, queue_cap=2048,
                         scratch_bytes=1 << 20, keep_layers=False, wbits=4)
    low.input(low.embed([1, 2]), ple=low.embed_ple([1, 2]))
    low.chunk(0, 2)
    ddr, q, n = low.prog.build()
    st, r = both(bytes(ddr), q, n, stall=10, lat=4, params=dict(RWP))
    assert st.done and r.done
