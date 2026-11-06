"""Contract v0.4 on the RTL core vs the golden model (identical DDR bytes and end status): the
extension unit's ops (ELT, ROPE, FDOT NT / NN, ROW QUANT), the row engine's L2NORM and causal
softmax, the GEMM's QM_ROWCOL, every new check rule, random v0.4 queues with injected errors, and
the Qwen3.5 LA + FA blocks (small config) end to end."""
from __future__ import annotations

import numpy as np
import pytest

from tars_mini import contract as C
from tars_mini import rtlsim
from tars_mini.golden import ContractError, Machine
from tars_mini.llm import Dev, V

from test_rtl import MEM, QADDR, both, expect_code
import test_v04 as T4

pytestmark = pytest.mark.skipif(not rtlsim.have_verilator(), reason="needs Verilator")
F32 = np.float32
QA = 0x7000


def _r(rng, *s, scale=2.0):
    return rng.normal(0, scale, s).astype(F32)


def _both(p: Dev, **kw):
    ddr, q, n = p.build()
    return both(bytes(ddr), q, n, **kw)


# ---------------------------------------------------------------------- directed ops
ACTS = [C.ACT_NONE, C.ACT_RELU, C.ACT_SIGMOID, C.ACT_SOFTPLUS, C.ACT_EXP, C.ACT_SILU, C.ACT_GELU]


@pytest.mark.parametrize("op", [C.ELT_MUL, C.ELT_ADD, C.ELT_SUB, C.ELT_MULADD])
@pytest.mark.parametrize("bmode", [C.BM_DENSE, C.BM_ROW, C.BM_COL, C.BM_SCALAR])
def test_elt_on_rtl(op, bmode):
    rng = np.random.default_rng(10 * op + bmode)
    M, N = 5, 32
    out = C.OUT_FP32 if op == C.ELT_MULADD else [C.OUT_FP32, C.OUT_BF16, C.OUT_INT8][bmode % 3]
    p = Dev(ddr_bytes=MEM)
    a = p.const(_r(rng, M, N))
    b = p.const(_r(rng, *{C.BM_DENSE: (M, N), C.BM_ROW: (1, N), C.BM_COL: (1, M + 3), C.BM_SCALAR: (1, 16)}[bmode]))
    c = p.const(_r(rng, M, N)) if op == C.ELT_MULADD else None
    y = p.mat(M, N, out)
    bv = b if bmode in (C.BM_DENSE, C.BM_ROW) else V(b.off + 4 * (1 + bmode), 1, 1, 4)
    p.elt(op, a, bv, y, bmode=bmode, act=ACTS[(op + bmode) % len(ACTS)], c=c)
    if out == C.OUT_INT8:
        p.descs[-1] = C.encode({**C.decode(p.descs[-1]), "OUT_SCALE": C.f32_bits(9.0)})
    st, _ = _both(p, stall=30 * (bmode % 2), lat=1 + 2 * op)
    assert st.done


def test_elt_in_place_strided_and_row_broadcast_on_rtl():
    """The delta rule's update in place (C = Y, A a row, B per row), then a scalar MUL in place
    through A, on strided views."""
    rng = np.random.default_rng(5)
    p = Dev(ddr_bytes=MEM)
    S = p.const(_r(rng, 16, 48))
    Sv = V(S.off + 16, 16, 32, 4 * 48)                       # columns 4..35 of each row
    d, k = p.const(_r(rng, 1, 32)), p.const(_r(rng, 1, 16))
    p.elt(C.ELT_MULADD, d, k, Sv, bmode=C.BM_COL, arow=True, c=Sv)
    p.elt(C.ELT_MUL, Sv, V(k.off + 8, 1, 1, 4), Sv, bmode=C.BM_SCALAR, act=C.ACT_EXP)
    st, _ = _both(p, stall=50, lat=5)
    assert st.done


@pytest.mark.parametrize("out", [C.OUT_FP32, C.OUT_BF16])
@pytest.mark.parametrize("N,R,G", [(64, 32, 2), (96, 64, 1), (32, 32, 3)])
def test_rope_on_rtl(out, N, R, G):
    rng = np.random.default_rng(N + R + G)
    M = 7
    p = Dev(ddr_bytes=MEM)
    x = p.const(_r(rng, M, N))
    tab = p.const(_r(rng, (M - 1) // G + 1, R, scale=1.0))
    y = p.mat(M, N, out)
    p.rope(x, tab, G, R, y)
    st, _ = _both(p, stall=20, lat=3)
    assert st.done


@pytest.mark.parametrize("nn", [False, True])
@pytest.mark.parametrize("out", [C.OUT_FP32, C.OUT_BF16, C.OUT_INT8])
def test_fdot_on_rtl(nn, out):
    rng = np.random.default_rng(int(nn) * 3 + out)
    M, N, K = 3, 32, 48
    p = Dev(ddr_bytes=MEM)
    a = p.const(_r(rng, M, K + 16))
    av = V(a.off, M, K, a.ld)                                 # strided A
    b = p.const(_r(rng, K, N + 16) if nn else _r(rng, N, K + 16))
    bv = V(b.off, K, N, b.ld) if nn else V(b.off, N, K, b.ld)  # strided B (GROUP_SIZE)
    y = p.mat(M, N, out)
    p.fdot(av, bv, y, nn=nn)
    if out == C.OUT_INT8:
        p.descs[-1] = C.encode({**C.decode(p.descs[-1]), "OUT_SCALE": C.f32_bits(3.0)})
    st, _ = _both(p, stall=[0, 40][int(nn)], lat=4)
    assert st.done


@pytest.mark.parametrize("nn", [False, True])
@pytest.mark.parametrize("M,N,K,stall", [(1, 16, 16, 0), (3, 48, 16, 30), (2, 16, 32, 0), (1, 32, 1024, 0),
                                         (2, 16, 1024, 20), (2, 16, 1040, 0), (3, 64, 128, 40), (4, 16, 48, 0)])
def test_fdot_passes_overlap_on_rtl(nn, M, N, K, stall):
    """FDOT's pass ends running behind the next pass, A's row kept on chip (NT, K <= 1024; 1040: the
    plain path), an A beat every 4 k (NN): passes shorter than the fold, the kept row's limit, rows
    after the first, and two descriptors back to back (the second must not catch the first's fold)."""
    rng = np.random.default_rng(M * 1000 + N * 7 + K + int(nn))
    p = Dev(ddr_bytes=1 << 19)
    a = p.const(_r(rng, M, K))
    y = [p.mat(M, N) for _ in range(2)]
    for yi in y:
        b = p.const(_r(rng, K, N) if nn else _r(rng, N, K))
        p.fdot(a, b, yi, nn=nn)
    st, _ = _both(p, stall=stall, lat=3)
    assert st.done


@pytest.mark.skipif(rtlsim.TOP != "kv260", reason="the KV260 testbench's write-only stall")
@pytest.mark.parametrize("nn", [False, True])
def test_fdot_fold_waits_for_a_full_output_queue(nn, monkeypatch):
    """Short passes against a write channel shut for long stretches: the output queue fills, a readout waits, and the
    next pass must not overwrite the fold's input. K = 16, many outputs."""
    monkeypatch.setenv("TM_TBARGS", "+wstall=100")
    rng = np.random.default_rng(11 + int(nn))
    M, N, K = 2, 1024, 16
    p = Dev(ddr_bytes=1 << 19)
    a = p.const(_r(rng, M, K))
    b = p.const(_r(rng, K, N) if nn else _r(rng, N, K))
    p.fdot(a, b, p.mat(M, N), nn=nn)
    st, _ = _both(p, stall=0, lat=2)
    assert st.done


def test_quant_l2_causal_and_rowcol_gemm_on_rtl():
    """A W8A8 linear (QUANT + ROWCOL GEMM with softplus fused), an L2NORM and a causal softmax."""
    from tars_mini.compiler import quantize_weight
    from tars_mini.llm import QLin
    rng = np.random.default_rng(6)
    x = _r(rng, 7, 64)
    x[2] = 0                                                  # an all-zero row: q = 0, s = 0
    x[4, 5] = 40.0
    W = rng.normal(0, 0.3, (32, 64))
    p = Dev(ddr_bytes=MEM)
    xv = p.const(x)
    q, s = p.quant(xv)
    qw = quantize_weight(W)
    lin = QLin(p.put(qw.q, "i1"), p.put(qw.scale), p.put(rng.normal(0, 1, 32).astype(F32)), 32, 64)
    y = p.mat(7, 32)
    p.gemm_rc(q, s, lin, y, act=C.ACT_SOFTPLUS)
    g = p.const(_r(rng, 1, 64, scale=0.5))
    z = p.mat(7, 64)
    p.norm(xv, g, 1e-6, z, l2=True)
    sm = p.mat(7, 64, C.OUT_BF16)
    p.softmax_causal(xv, sm, 3)
    st, _ = _both(p, stall=25, lat=3)
    assert st.done


@pytest.mark.parametrize("seed", range(3))
def test_xpu_small_output_queue(seed):
    """A 2-slot output queue under heavy write backpressure: every launch (an ELT step through the
    122-cycle exp, an FDOT NN readout, QUANT's scale and rows) must wait for a slot, counting the
    launch already under way."""
    rng = np.random.default_rng(seed)
    p = Dev(ddr_bytes=MEM)
    a, b = p.const(_r(rng, 6, 64)), p.const(_r(rng, 6, 64))
    p.elt(C.ELT_MUL, a, b, p.mat(6, 64), act=C.ACT_EXP)
    p.fdot(p.const(_r(rng, 2, 32)), p.const(_r(rng, 32, 32)), p.mat(2, 32), nn=True)
    p.quant(a)
    st, _ = _both(p, R=4, C_=4, stall=85, lat=2, seed=seed + 1, params={"OQ": 2})
    assert st.done


def test_causal_softmax_ignores_the_first_masked_element_on_rtl():
    """The element just past each row's valid range is the row's largest: it must stay out of
    the max (and the sum); rows split across beats at every offset, overlapped or not."""
    rng = np.random.default_rng(9)
    for N, K in ((64, 5), (48, 1)):
        p = Dev(ddr_bytes=MEM)
        x = _r(rng, 12, N)
        for m in range(12):
            if K + m < N:
                x[m, K + m] = 50.0
        p.softmax_causal(p.const(x), p.mat(12, N), K)
        st, _ = _both(p, stall=10, lat=3)
        assert st.done


# ---------------------------------------------------------------------- GEMV (GEMM with M = 1)
def _gemv_program(rng, mem, n, k, qm, bias, act, out, lda_pad=0, ldy_pad=0):
    """One M = 1 GEMM descriptor on random operands (finite scales, some huge, a zero x chunk)."""
    ddr = bytearray(rng.integers(0, 256, mem, dtype=np.uint8).tobytes())
    off = {"S": 0x200, "B": 0x300, "R": 0x400}
    w_off = -(-(0x500 + 4 * n) // 16) * 16
    y_off = w_off + n * k
    x = rng.integers(-128, 128, k).astype(np.int8)
    x[16:32] = 0                                             # a zero chunk (not the first: a stale or
                                                             # unwritten buffer entry reads as zeros)
    a_off = y_off + 16 * n + 0x100                           # x after the output when it is long
    ddr[a_off:a_off + k] = x.tobytes()
    ddr[w_off:w_off + n * k] = rng.integers(-128, 128, n * k).astype(np.int8).tobytes()
    sc = rng.uniform(1e-4, 3e-2, n).astype("<f4")
    sc[n // 2] = 3e30
    ddr[off["S"]:off["S"] + 4 * n] = sc.tobytes()
    ddr[off["B"]:off["B"] + 4 * n] = rng.normal(0, 2, n).astype("<f4").tobytes()
    ddr[off["R"]:off["R"] + 4] = np.float32(0.37).tobytes()
    f = dict(MAGIC=C.MAGIC, VERSION=C.VERSION, VALID=1, OPCODE=C.OP_GEMM, M=1, N=n, K=k,
             CTRL=C.ctrl(qm, fusion=int(act != 0)), EPI=C.epi(act, out, bias), A_OFF=a_off, W_OFF=w_off,
             Y_OFF=y_off, SCALE_OFF=off["S"], BIAS_OFF=off["B"] if bias else 0,
             HDR_RSV=off["R"] // 16 if qm == C.QM_ROWCOL else 0, LDA16=lda_pad, LDY16=ldy_pad)
    if out == C.OUT_INT8:
        f["OUT_SCALE"] = C.f32_bits(5.0)
    q = mem - 0x100
    ddr[q:q + 64] = C.encode(f)
    return bytes(ddr), q


@pytest.mark.parametrize("qm", [C.QM_TENSOR, C.QM_CHANNEL, C.QM_ROWCOL])
@pytest.mark.parametrize("bias", [0, 1])
@pytest.mark.parametrize("act,out", [(C.ACT_NONE, C.OUT_FP32), (C.ACT_RELU, C.OUT_INT8),
                                     (C.ACT_SILU, C.OUT_BF16), (C.ACT_SOFTPLUS, C.OUT_INT8)])
def test_gemv_on_rtl(qm, bias, act, out):
    rng = np.random.default_rng(qm * 100 + bias * 10 + act)
    ddr, q = _gemv_program(rng, MEM, 48, 16 * int(rng.integers(1, 40)), qm, bias, act, out)
    st, _ = both(ddr, q, 1, stall=[0, 40][bias], lat=3)
    assert st.done


@pytest.mark.parametrize("k", [16, 4096, 4112])
def test_gemv_buffer_boundary_on_rtl(k):
    """K = 16 (one W beat a channel: a channel ends every cycle); K = 4096 fills the x buffer
    exactly; K = 4112 goes to the systolic array instead."""
    rng = np.random.default_rng(k)
    ddr, q = _gemv_program(rng, 1 << 21, 32, k, C.QM_ROWCOL, 1, C.ACT_GELU, C.OUT_FP32)
    st, _ = both(ddr, q, 1, stall=20, lat=4)
    if k > 4096 and Machine(bytearray(16)).gemv_only:      # the KV260 build (no array): not supported
        assert st.err_code == C.E_OPCODE
    else:
        assert st.done


@pytest.mark.parametrize("m,k,walign", [(1, 1024, 64), (2, 256, 64), (3, 192, 64), (4, 4096, 64), (4, 512, 64),
                                         (2, 64, 64), (4, 128, 64), (3, 320, 16), (4, 1040, 64)])
def test_gemv_batched_on_rtl(m, k, walign):
    """M = 1 .. 4 rows on the GEMV unit (the KV260 build's limit); W 64-byte aligned with K a
    multiple of 64 and K / 64 >= M: 64-byte W beats on the KV260 top (RW 4), else 16-byte ones."""
    rng = np.random.default_rng(1000 * m + k + walign)
    mem = 1 << 21
    n = 32
    ddr = bytearray(rng.integers(0, 256, mem, dtype=np.uint8).tobytes())
    s_off, b_off, r_off = 0x200, 0x300, 0x400
    w_off = -(-(0x500 + 4 * n) // 64) * 64 + (16 if walign == 16 else 0)
    y_off = w_off + n * k + 0x40
    lda = -(-k // 16) + 1                                    # x rows 16 bytes apart more than dense
    a_off = y_off + m * 16 * n + 0x100
    x = rng.integers(-128, 128, (m, k)).astype(np.int8)
    for i in range(m):
        ddr[a_off + 16 * lda * i:a_off + 16 * lda * i + k] = x[i].tobytes()
    ddr[w_off:w_off + n * k] = rng.integers(-128, 128, n * k).astype(np.int8).tobytes()
    ddr[s_off:s_off + 4 * n] = rng.uniform(1e-4, 3e-2, n).astype("<f4").tobytes()
    ddr[b_off:b_off + 4 * n] = rng.normal(0, 2, n).astype("<f4").tobytes()
    ddr[r_off:r_off + 16] = rng.uniform(0.1, 2.0, 4).astype("<f4").tobytes()
    f = dict(MAGIC=C.MAGIC, VERSION=C.VERSION, VALID=1, OPCODE=C.OP_GEMM, M=m, N=n, K=k,
             CTRL=C.ctrl(C.QM_ROWCOL, fusion=1), EPI=C.epi(C.ACT_SILU, C.OUT_FP32, 1), A_OFF=a_off,
             W_OFF=w_off, Y_OFF=y_off, SCALE_OFF=s_off, BIAS_OFF=b_off, HDR_RSV=r_off // 16, LDA16=lda,
             LDY16=n // 4 + 1)
    q = mem - 0x100
    ddr[q:q + 64] = C.encode(f)
    st, _ = both(bytes(ddr), q, 1, stall=15, lat=3)
    assert st.done


# ---------------------------------------------------------------------- the check rules
@pytest.mark.parametrize("fields,code", T4.test_v04_rules.pytestmark[0].args[1])
def test_v04_rules_on_rtl(fields, code):
    ddr = bytearray(np.random.default_rng(2).integers(0, 256, MEM, dtype=np.uint8).tobytes())
    ddr[QA:QA + 64] = T4._desc(**fields)
    st, _ = both(bytes(ddr), QA, 1)
    assert st.err_code == expect_code(bytes(ddr[QA:QA + 64]), code)


def _range_cases():
    """Each v0.4 extent ending exactly at the top of DDR (legal) and one element further."""
    top = MEM
    cases = []

    def pair(name, f, end_field, nbytes, step):
        for over in (False, True):
            cases.append(pytest.param({**f, end_field: top - nbytes + (step if over else 0)},
                                      C.E_RANGE if over else C.E_OK, id=f"{name}-{'over' if over else 'fit'}"))
    elt = dict(OPCODE=C.OP_ELT, M=3, N=32)
    pair("elt-dense-B", elt, "W_OFF", 2 * 128 + 128, 16)
    pair("elt-row-B", {**elt, "HDR_RSV": C.BM_ROW}, "W_OFF", 128, 16)
    pair("elt-col-B", {**elt, "HDR_RSV": C.BM_COL}, "W_OFF", 12, 4)
    pair("elt-scalar-B", {**elt, "HDR_RSV": C.BM_SCALAR}, "W_OFF", 4, 4)
    pair("elt-arow-A", {**elt, "HDR_RSV": 4}, "A_OFF", 128, 16)
    pair("elt-C", {**elt, "CTRL": C.ELT_MULADD, "EPI": C.epi(bias=1)}, "BIAS_OFF", 3 * 128, 16)
    pair("rope-table", dict(OPCODE=C.OP_ROPE, M=5, N=64, K=32, GROUP_SIZE=2), "W_OFF", 3 * 128, 16)
    pair("fdot-nt-B", dict(OPCODE=C.OP_FDOT, M=2, N=16, K=32, GROUP_SIZE=9), "W_OFF", 15 * 144 + 128, 16)
    pair("fdot-nn-B", dict(OPCODE=C.OP_FDOT, CTRL=1, M=2, N=16, K=32), "W_OFF", 32 * 64, 16)
    pair("quant-scales", dict(OPCODE=C.OP_ROW, CTRL=C.ROW_QUANT, EPI=C.epi(out=C.OUT_INT8), W_OFF=0, M=5),
         "SCALE_OFF", 32, 16)
    return cases


@pytest.mark.parametrize("fields,code", _range_cases())
def test_v04_ranges_on_rtl(fields, code):
    ddr = bytearray(MEM)
    ddr[QA:QA + 64] = T4._desc(**fields)
    st, _ = both(bytes(ddr), QA, 1)
    assert st.err_code == expect_code(bytes(ddr[QA:QA + 64]), code)


@pytest.mark.parametrize("over", [False, True])
def test_rowcol_row_scale_range_on_rtl(over):
    top = MEM - 16 * 4 + (16 if over else 0)
    f = dict(OPCODE=C.OP_GEMM, CTRL=C.ctrl(C.QM_ROWCOL), K=16, SCALE_OFF=0x400, M=16, HDR_RSV=top // 16)
    ddr = bytearray(MEM)
    ddr[QA:QA + 64] = T4._desc(**f)
    st, _ = both(bytes(ddr), QA, 1)
    assert st.err_code == (C.E_RANGE if over else C.E_OK)


# ---------------------------------------------------------------------- random v0.4 queues
def _random_v04(rng, inject: bool) -> bytes:
    def off(nbytes, al=16):
        if rng.random() < 0.6:
            slot = 0x800 * int(rng.integers(0, 0xE000 // 0x800))
            if slot + nbytes <= 0xE000:
                return slot + (int(rng.integers(0, 4)) * 4 if al == 4 else 0)
        return al * int(rng.integers(0, (0xE000 - nbytes) // al))

    def pad():
        return 16 * int(rng.integers(0, 3)) * int(rng.random() < 0.4)

    kind = rng.choice(["elt", "rope", "fdot", "quant", "l2", "causal", "rowcol"], p=[.36, .14, .2, .1, .07, .07, .06])
    M = int(rng.integers(1, 10))
    N = 16 * int(rng.integers(1, 5))
    out = int(rng.integers(0, 3))
    f = dict(MAGIC=C.MAGIC, VERSION=C.VERSION, VALID=1, M=M, N=N)
    if kind == "elt":
        op = int(rng.integers(0, 4))
        out = C.OUT_FP32 if op == C.ELT_MULADD else out
        bm, arow = int(rng.integers(0, 4)), int(rng.random() < 0.2)
        lda, ldy, ldb = 4 * N + pad(), 4 * N * 0 + N * C.ESIZE[out] + pad(), 4 * N + pad()
        y = off((M - 1) * ldy + N * C.ESIZE[out])
        place = rng.random()
        a = y if (place < 0.25 and not arow and out == C.OUT_FP32 and lda == ldy) else \
            off(4 * N if arow else (M - 1) * lda + 4 * N)
        nb = {C.BM_DENSE: (M - 1) * ldb + 4 * N, C.BM_ROW: 4 * N, C.BM_COL: 4 * M, C.BM_SCALAR: 4}[bm]
        f.update(OPCODE=C.OP_ELT, CTRL=op, HDR_RSV=bm | arow << 2, A_OFF=a, Y_OFF=y,
                 W_OFF=off(nb, 4 if bm >= C.BM_COL else 16),
                 GROUP_SIZE=ldb // 16 if (bm == C.BM_DENSE and ldb != 4 * N) else 0,
                 LDA16=0 if (arow or lda == 4 * N) else lda // 16,
                 LDY16=0 if ldy == N * C.ESIZE[out] else ldy // 16,
                 EPI=C.epi(int(rng.integers(0, 8)), out, int(op == C.ELT_MULADD)))
        if op == C.ELT_MULADD:
            f["BIAS_OFF"] = y if place > 0.7 else off((M - 1) * ldy + 4 * N)
    elif kind == "rope":
        N = 32 * int(rng.integers(1, 4))
        R, G = 32 * int(rng.integers(1, N // 32 + 1)), int(rng.integers(1, 4))
        lda, ldy = 4 * N + pad(), N * C.ESIZE[out] + pad()
        out = out if out != C.OUT_INT8 else C.OUT_FP32
        ldy = N * C.ESIZE[out] + pad()
        f.update(OPCODE=C.OP_ROPE, N=N, K=R, GROUP_SIZE=G, A_OFF=off((M - 1) * lda + 4 * N),
                 Y_OFF=off((M - 1) * ldy + N * C.ESIZE[out]), W_OFF=off(4 * R * ((M - 1) // G + 1)),
                 LDA16=0 if lda == 4 * N else lda // 16, LDY16=0 if ldy == N * C.ESIZE[out] else ldy // 16,
                 EPI=C.epi(0, out))
    elif kind == "fdot":
        nn = int(rng.random() < 0.5)
        K = 16 * int(rng.integers(1, 4))
        brow = 4 * (N if nn else K)
        ldb = brow + pad()
        lda, ldy = 4 * K + pad(), N * C.ESIZE[out] + pad()
        f.update(OPCODE=C.OP_FDOT, CTRL=nn, K=K, A_OFF=off((M - 1) * lda + 4 * K),
                 W_OFF=off(((K if nn else N) - 1) * ldb + brow), Y_OFF=off((M - 1) * ldy + N * C.ESIZE[out]),
                 GROUP_SIZE=0 if ldb == brow else ldb // 16, LDA16=0 if lda == 4 * K else lda // 16,
                 LDY16=0 if ldy == N * C.ESIZE[out] else ldy // 16, EPI=C.epi(0, out))
    elif kind == "quant":
        lda, ldy = 4 * N + pad(), N + pad()
        out = C.OUT_INT8
        f.update(OPCODE=C.OP_ROW, CTRL=C.ROW_QUANT, A_OFF=off((M - 1) * lda + 4 * N),
                 Y_OFF=off((M - 1) * ldy + N), SCALE_OFF=off(4 * M), EPI=C.epi(0, out),
                 LDA16=0 if lda == 4 * N else lda // 16, LDY16=0 if ldy == N else ldy // 16)
    elif kind in ("l2", "causal"):
        f.update(OPCODE=C.OP_ROW, A_OFF=off((M - 1) * 4 * N + 4 * N), Y_OFF=off(M * N * C.ESIZE[out]),
                 EPI=C.epi(0, out))
        if kind == "l2":
            f.update(CTRL=C.ROW_L2NORM, SCALE_OFF=off(4 * N), W_OFF=C.f32_bits(1e-6))
        else:
            f.update(CTRL=C.ROW_SOFTMAX, K=int(rng.integers(0, N + 2)))
    else:
        K = 16 * int(rng.integers(1, 4))
        act = int(rng.integers(0, 8))
        f.update(OPCODE=C.OP_GEMM, CTRL=C.ctrl(C.QM_ROWCOL, fusion=int(act != 0)), K=K, EPI=C.epi(act, out),
                 A_OFF=off(M * K), W_OFF=off(N * K), SCALE_OFF=off(4 * N), Y_OFF=off(M * N * C.ESIZE[out]),
                 HDR_RSV=off(4 * M) // 16)
    if out == C.OUT_INT8 and not (f["OPCODE"] == C.OP_ROW and f.get("CTRL") == C.ROW_QUANT):
        f["OUT_SCALE"] = C.f32_bits(float(rng.uniform(0.5, 60)))
    if inject:
        field = ["CTRL", "HDR_RSV", "GROUP_SIZE", "K", "W_OFF", "BIAS_OFF", "EPI", "M", "SCALE_OFF"][int(rng.integers(0, 9))]
        val = {"CTRL": f.get("CTRL", 0) ^ (1 << int(rng.integers(0, 4))), "HDR_RSV": f.get("HDR_RSV", 0) ^ 8,
               "GROUP_SIZE": int(rng.choice([0, 1, 2000])), "K": f.get("K", 0) + int(rng.choice([8, 16, 32])),
               "W_OFF": f.get("W_OFF", 0) + int(rng.choice([2, 4, 8])), "BIAS_OFF": 0x40,
               "EPI": f.get("EPI", 0) ^ (1 << int(rng.integers(0, 6))), "M": 4000, "SCALE_OFF": 0x40}[field]
        f[field] = val
    return C.encode(f)


def random_v04_program(seed: int, n: int = 8, inject_p: float = 0.15) -> bytes:
    rng = np.random.default_rng(seed)
    special = np.array([0x7F800000, 0xFF800000, 0x7FC00000, 0x00000001, 0x80400000, 0x7F7FFFFF,
                        0x80000000, 0x42000000, 0xC1F00000], dtype="<u4").view("<f4")
    while True:
        v = rng.normal(0, 3, MEM // 4).astype("<f4")
        hit = rng.random(v.size) < 0.02
        v[hit] = rng.choice(special, int(hit.sum()))
        ddr = bytearray(v.tobytes())
        for i in range(n):
            ddr[QADDR + 64 * i:QADDR + 64 * (i + 1)] = _random_v04(rng, rng.random() < inject_p)
        try:
            Machine(bytearray(ddr)).run_queue(QADDR, n)
            return bytes(ddr)
        except ContractError:
            continue


FUZZ_SIZES = [(8, 8), (4, 8), (3, 5)]


@pytest.mark.parametrize("seed", range(24), ids=lambda s: "{}x{}-{}".format(*FUZZ_SIZES[s % 3], s))
def test_fuzz_v04_queues(seed):
    R, C_ = FUZZ_SIZES[seed % 3]
    ddr = random_v04_program(7000 + seed)
    both(ddr, QADDR, 8, R, C_, stall=[0, 30, 70][seed % 4 % 3], lat=[1, 3, 6][seed % 4 % 3], seed=seed + 1)


def test_fuzz_v04_coverage():
    """The fuzz programs reach every new op and mode, completed and refused."""
    seen, errs = set(), set()
    for seed in range(24):
        ddr = random_v04_program(7000 + seed)
        st = Machine(bytearray(ddr)).run_queue(QADDR, 8)
        errs.add(st.err_code)
        for i in range(st.op_idx):
            d = C.decode(ddr[QADDR + 64 * i:QADDR + 64 * (i + 1)])
            op = d["OPCODE"]
            if op == C.OP_ELT:
                seen.add(("elt", d["CTRL"] & 3, d["HDR_RSV"] & 3))
            elif op == C.OP_FDOT:
                seen.add(("fdot", d["CTRL"] & 1))
            elif op == C.OP_ROPE:
                seen.add(("rope",))
            elif op == C.OP_ROW:
                seen.add(("row", d["CTRL"]))
    assert {("fdot", 0), ("fdot", 1), ("rope",), ("row", C.ROW_QUANT)} <= seen
    assert sum(1 for s in seen if s[0] == "elt") >= 12
    assert {C.E_OK, C.E_FIELD} <= errs and len(errs) >= 3


# ---------------------------------------------------------------------- the Qwen3.5 blocks
@pytest.mark.parametrize("chunks", [[3, 1]])
def test_qwen_blocks_on_rtl(chunks):
    """LA + FA + final norm, small config (16 linear-attention heads, FA 4 / 2 heads x 128),
    prefill then decode: the RTL writes exactly the golden model's bytes."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tars_mini import llm as L
    from tars_mini import qwen_ref as Q
    cfg = Q.make_config(Q.CFG_SMALL)
    w = Q.random_weights(cfg)
    T = sum(chunks)
    x = Q.random_inputs(cfg, T)
    low = L.QwenLowering(cfg, w, max_tokens=T, ddr_bytes=1 << 21, scratch_bytes=1 << 18)
    low.input(x)
    t0 = 0
    for c in chunks:
        low.chunk(t0, c)
        t0 += c
    ddr, q, n = low.prog.build()
    st, r = both(bytes(ddr), q, n, stall=10, lat=4)
    assert st.done and r.done


def test_qwen_model_greedy_on_rtl():
    """Four layers (LA LA LA FA, small dims), final norm, tied head (vocabulary 4096): greedy
    decoding, prefill 3 then 2 decode steps, each step's queue run on the RTL and on the golden
    model from the same DDR: identical bytes, so identical logits and tokens. The decode steps'
    M = 1 linears and head run on the GEMV path."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tars_mini import llm as L
    from tars_mini import qwen_ref as Q
    cfg = Q.make_config(Q.CFG_SMALL, Q.LAYERS_08B[:4], vocab=4096)
    w = Q.random_weights(cfg, embed=True)
    low = L.QwenLowering(cfg, w, max_tokens=8, ddr_bytes=1 << 23, head=True, queue_cap=2048, scratch_bytes=1 << 20)

    def run(ddr, q, n):
        st, r = both(bytes(ddr), q, n, stall=10, lat=3)
        assert st.done
        return r.ddr
    ids, _, _ = low.generate(run, [7, 1000, 42], 2)
    assert len(ids) == 6


def test_qwen3_model_greedy_on_rtl():
    """Qwen3 (two full-attention layers without the output gate, small dims), final norm, tied
    head (vocabulary 4096): greedy decoding, prefill 3 then 2 decode steps, each step's queue on
    the RTL and on the golden model from the same DDR: identical bytes."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tars_mini import llm as L
    from tars_mini import qwen_ref as Q
    cfg = Q.make_config(Q.CFG_Q3_SMALL, ["full_attention"] * 2, vocab=4096)
    w = Q.random_weights(cfg, embed=True)
    low = L.QwenLowering(cfg, w, max_tokens=8, ddr_bytes=1 << 23, head=True, queue_cap=2048, scratch_bytes=1 << 20)

    def run(ddr, q, n):
        st, r = both(bytes(ddr), q, n, stall=10, lat=3)
        assert st.done
        return r.ddr
    ids, _, _ = low.generate(run, [7, 1000, 42], 2)
    assert len(ids) == 6


def test_speculative_verify_on_rtl():
    """A speculative verify on the RTL: a chunk of 4 at an unaligned position with every row's logits
    (the head as 4-row GEMV passes), then a rejected draft's stale K / V rows overwritten by a chunk
    at the position after the accepted token. RTL == golden byte for byte at every step."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tars_mini import llm as L
    from tars_mini import qwen_ref as Q
    cfg = Q.make_config(Q.CFG_Q3_SMALL, ["full_attention"] * 2, vocab=4096)
    w = Q.random_weights(cfg, embed=True)
    low = L.QwenLowering(cfg, w, max_tokens=16, ddr_bytes=1 << 23, head=True, queue_cap=2048,
                         scratch_bytes=1 << 20, verify_rows=4)
    ddr = low.p.ddr
    for t0, toks, rows in (([0, [7, 1000, 42], False]), (3, [9, 11, 13, 15], True), (5, [21, 23, 25], True),
                           (7, [31], True), (8, [33, 35], True)):
        low.put_input(ddr, t0, low.embed(toks))
        s = len(low.p.descs)
        low.chunk(t0, len(toks), all_rows=rows)
        q, n = low.stage(ddr, s, len(low.p.descs))
        st, r = both(bytes(ddr), q, n, stall=10, lat=3)
        assert st.done
        ddr = bytearray(r.ddr)


def test_mtp_speculative_on_rtl():
    """Qwen3.5 (LA then FA, small dims) with the MTP head: speculative rounds on the RTL, the verify's
    ping-pong state, a miss and a hit, every queue byte-identical to the golden model."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tars_mini import llm as L
    from tars_mini import qwen_ref as Q
    cfg = Q.make_config(Q.CFG_SMALL, ["linear_attention", "full_attention"], vocab=4096)
    w = {**Q.random_weights(cfg, embed=True), **Q.random_mtp(cfg)}
    low = L.QwenLowering(cfg, w, max_tokens=16, ddr_bytes=1 << 23, head=True, queue_cap=4096,
                         scratch_bytes=1 << 20, verify_rows=2, mtp=True)

    def run(ddr, q, n):
        st, r = both(bytes(ddr), q, n, stall=10, lat=3)
        assert st.done
        return r.ddr
    seen = []

    def oracle(t, d):                                # the second round's draft is whatever the first gave
        seen.append(t)
        return d if len(seen) % 2 else (d + 1) % 4096
    ids, _, rounds = low.generate_mtp(run, [7, 1000, 42], 3, oracle)
    assert rounds >= 2 and len(ids) >= 7

