"""Golden model: numerics edge cases, error semantics, fusion equivalence, and agreement with P0."""
import struct

import numpy as np
import pytest

import tars_mini_ref as P0
from tars_mini import contract as C
from tars_mini import numerics as F
from tars_mini.golden import ContractError, Machine

F32 = np.float32


# ---------------------------------------------------------------------- numerics
def test_relu_signed_zero_and_nan():
    x = np.array([-0.0, 0.0, -1.0, np.nan, 2.5], F32)
    r = F.act_fn(x, C.ACT_RELU)
    assert r.view(np.uint32).tolist() == [0, 0, 0, 0, C.f32_bits(2.5)]   # all +0, not -0


def test_relu6_clamps():
    r = F.act_fn(np.array([5.0, 6.0, 6.5, 1e30, np.inf], F32), C.ACT_RELU6)
    assert r.tolist() == [5.0, 6.0, 6.0, 6.0, 6.0]


def test_ftz_inputs_and_outputs():
    den = np.array([1e-40], F32)                      # a denormal
    assert F.ftz(den)[0] == 0 and F.ftz(-den).view(np.uint32)[0] == 0x80000000
    assert F.fmul(np.array([1e-20], F32), np.array([1e-20], F32))[0] == 0   # result flushes


def test_bf16_round_ties_to_even():
    cases = {0x3F808000: 0x3F80,   # tie, even below -> stays
             0x3F818000: 0x3F82,   # tie, odd below -> rounds up
             0x3F808001: 0x3F81,   # above tie -> up
             0x7F7FFFFF: 0x7F80,   # largest float -> inf
             0x7FC12345: 0x7FC0}   # NaN -> canonical
    x = np.array(list(cases), np.uint32).view(F32)
    assert F.to_bf16_bits(x).tolist() == list(cases.values())


def test_q8_ties_saturation_nan():
    x = np.array([0.5, 1.5, 2.5, -0.5, 200.0, -200.0, np.nan], F32)
    assert F.q8(x, F32(1.0)).tolist() == [0, 2, 2, 0, 127, -127, 0]


def test_no_fma():
    """acc*s + b with two roundings differs from a fused multiply-add; the golden does two."""
    rng = np.random.default_rng(1)
    for _ in range(10000):
        acc = int(rng.integers(-2**20, 2**20))
        s, b = F32(rng.uniform(1e-4, 1e-2)), F32(rng.uniform(-50, 50))
        two = F32(F32(F32(acc) * s) + b)
        fma = F32(np.float64(F32(acc)) * np.float64(s) + np.float64(b))
        if two != fma:
            break
    else:
        pytest.skip("no FMA-distinguishing case found")
    y = F.fadd(F.fmul(F.i2f(acc), s), b)
    assert y == two and y != fma


def test_i2f_rounds_to_even_above_2_24():
    assert F.i2f(16777217)[()] == F32(16777216)       # 2^24 + 1 -> tie -> even
    assert F.i2f(16777219)[()] == F32(16777220)


# ---------------------------------------------------------------------- queue and errors
M, N, K = 1, 16, 16


def _base():
    ddr = bytearray(1 << 14)
    ddr[0x100:0x100 + K] = bytes(range(K))
    ddr[0x200:0x200 + N * K] = bytes([1]) * (N * K)
    ddr[0x400:0x404] = struct.pack("<f", 1.0)
    good = dict(MAGIC=C.MAGIC, VERSION=C.VERSION, OPCODE=C.OP_GEMM, CTRL=C.ctrl(C.QM_TENSOR),
                EPI=C.epi(), BASE_ADDR=0, A_OFF=0x100, W_OFF=0x200, Y_OFF=0x800, SCALE_OFF=0x400,
                M=M, N=N, K=K, VALID=1)
    return ddr, good


def _run(ddr, *descs):
    mem = bytearray(ddr)
    for i, r in enumerate(descs):
        mem[0x2000 + 64 * i:0x2000 + 64 * (i + 1)] = r
    m = Machine(mem)
    return m.run_queue(0x2000, len(descs)), m


@pytest.mark.parametrize("code,change", [
    (C.E_MAGIC, {"MAGIC": 0x55AA}),
    (C.E_VERSION, {"VERSION": 1}),
    (C.E_VALID, {"VALID": 0}),
    (C.E_OPCODE, {"OPCODE": 0x0F}),                                 # v0.4: 0x06-0x08 are ELT / ROPE / FDOT
    (C.E_FIELD, {"CORE_ID": 1}),
    (C.E_FIELD, {"HDR_RSV": 1}),
    (C.E_FIELD, {"DEP_SIG": 1}),
    (C.E_FIELD, {"GROUP_SIZE": 128}),
    (C.E_FIELD, {"CTRL": C.ctrl(C.QM_NONE)}),                       # v0.4: QM 3 is ROWCOL
    (C.E_FIELD, {"CTRL": C.ctrl(C.QM_TENSOR, a_dtype=C.DT_BF16)}),
    (C.E_FIELD, {"CTRL": C.ctrl(C.QM_TENSOR, fusion=1)}),            # fusion without activation
    (C.E_FIELD, {"EPI": C.epi(C.ACT_RELU)}),                         # activation without fusion
    (C.E_FIELD, {"OUT_SCALE": 1}),                                   # out_scale with fp32 out
    (C.E_ALIGN, {"N": 8}),
    (C.E_ALIGN, {"K": 24}),
    (C.E_ALIGN, {"M": 0}),
    (C.E_ALIGN, {"A_OFF": 0x108}),
    (C.E_RANGE, {"Y_OFF": 0x3FF0}),
    (C.E_RANGE, {"BASE_ADDR": 1 << 40}),
])
def test_error_codes(code, change):
    ddr, good = _base()
    st, m = _run(ddr, C.encode({**good, **change}))
    assert (st.done, st.err_code, st.op_idx) == (False, code, 0)
    assert m.ddr[:0x2000] == ddr[:0x2000]          # nothing written outside the queue


def test_crc_error():
    ddr, good = _base()
    st, _ = _run(ddr, C.encode(good, fix_crc=False))
    assert st.err_code == C.E_CRC


def test_field_error_has_priority_over_align():
    ddr, good = _base()
    st, _ = _run(ddr, C.encode({**good, "CORE_ID": 1, "N": 8}))
    assert st.err_code == C.E_FIELD


def test_queue_stops_and_keeps_earlier_writes():
    ddr, good = _base()
    st, m = _run(ddr, C.encode(good), C.encode({**good, "MAGIC": 0}), C.encode({**good, "Y_OFF": 0xC00}))
    assert (st.err_code, st.op_idx) == (C.E_MAGIC, 1)
    assert m.read(0x800, 64) != bytes(64) and m.read(0xC00, 64) == bytes(64)


def test_output_overlapping_input_is_refused():
    ddr, good = _base()
    with pytest.raises(ContractError):
        _run(ddr, C.encode({**good, "Y_OFF": 0x200}))


def test_sequential_raw_war_waw():
    """Each descriptor sees every earlier descriptor's writes, in queue order.

    op0 GEMM   A@0x100 -> Y@0x800 (fp32)
    op1 RELU   0x800 -> 0xA00                    RAW on op0's output
    op2 RELU   0xA00 -> 0x100 (bf16)             WAR: overwrites op0's input A
    op3 GEMM   A@0x100 (now op2's bytes) -> 0x800  WAW over op0's output, RAW on op2
    """
    ddr, good = _base()
    relu = dict(MAGIC=C.MAGIC, VERSION=C.VERSION, OPCODE=C.OP_RELU, M=1, N=16, VALID=1)
    descs = [C.encode(good),
             C.encode({**relu, "EPI": C.epi(C.ACT_RELU), "A_OFF": 0x800, "Y_OFF": 0xA00}),
             C.encode({**relu, "EPI": C.epi(C.ACT_RELU, C.OUT_BF16), "A_OFF": 0xA00, "Y_OFF": 0x100}),
             C.encode(good)]
    st, m = _run(ddr, *descs)
    assert st.done and st.op_idx == 4
    # recompute serially by hand
    a0 = np.arange(K, dtype=np.int64)
    y0 = np.full(N, a0.sum(), np.float32)                         # W all ones, scale 1.0
    assert np.array_equal(np.frombuffer(m.read(0xA00, 64), "<f4"), y0)
    bf = F.to_bf16_bits(y0).astype("<u2").tobytes()
    assert m.read(0x100, 32) == bf
    a3 = np.frombuffer(bf[:K], np.int8).astype(np.int64)          # op3 reads op2's bytes as INT8
    assert np.array_equal(np.frombuffer(m.read(0x800, 64), "<f4"), np.full(N, a3.sum(), np.float32))


@pytest.mark.parametrize("act", [C.ACT_RELU, C.ACT_RELU6, C.ACT_GELU, C.ACT_SILU])
@pytest.mark.parametrize("out", [C.OUT_FP32, C.OUT_BF16, C.OUT_INT8])
@pytest.mark.parametrize("seed", range(5))
def test_fusion_equivalence(act, out, seed):
    """GEMM(fused act, out) == GEMM(fp32) ; RELU(act, out), byte for byte."""
    rng = np.random.default_rng(seed)
    Mm, Nn, Kk = int(rng.integers(1, 9)), 16 * int(rng.integers(1, 4)), 16 * int(rng.integers(1, 5))
    ddr = bytearray(1 << 16)
    ddr[0x100:0x100 + Mm * Kk] = rng.integers(-128, 128, Mm * Kk, dtype=np.int8).tobytes()
    ddr[0x1000:0x1000 + Nn * Kk] = rng.integers(-128, 128, Nn * Kk, dtype=np.int8).tobytes()
    ddr[0x3000:0x3000 + 4 * Nn] = rng.uniform(1e-4, 1e-2, Nn).astype("<f4").tobytes()
    ddr[0x3400:0x3400 + 4 * Nn] = rng.uniform(-6, 6, Nn).astype("<f4").tobytes()
    inv = C.f32_bits(float(rng.uniform(1, 40))) if out == C.OUT_INT8 else 0
    common = dict(MAGIC=C.MAGIC, VERSION=C.VERSION, OPCODE=C.OP_GEMM, A_OFF=0x100, W_OFF=0x1000,
                  SCALE_OFF=0x3000, BIAS_OFF=0x3400, M=Mm, N=Nn, K=Kk, VALID=1)
    fused = C.encode({**common, "CTRL": C.ctrl(C.QM_CHANNEL, fusion=1), "EPI": C.epi(act, out, 1),
                      "Y_OFF": 0x4000, "OUT_SCALE": inv})
    plain = C.encode({**common, "CTRL": C.ctrl(C.QM_CHANNEL), "EPI": C.epi(C.ACT_NONE, C.OUT_FP32, 1),
                      "Y_OFF": 0x6000})
    relu = C.encode(dict(MAGIC=C.MAGIC, VERSION=C.VERSION, OPCODE=C.OP_RELU, EPI=C.epi(act, out),
                         A_OFF=0x6000, Y_OFF=0x8000, M=Mm, N=Nn, VALID=1, OUT_SCALE=inv))
    for i, r in enumerate((fused, plain, relu)):
        ddr[0xA000 + 64 * i:0xA000 + 64 * (i + 1)] = r
    m = Machine(ddr)
    assert m.run_queue(0xA000, 3).done
    nb = Mm * Nn * C.ESIZE[out]
    assert m.read(0x4000, nb) == m.read(0x8000, nb)


@pytest.mark.parametrize("seed", range(100))
def test_matches_p0_reference(seed):
    """The packaged golden model and the P0 reference script agree on random queues."""
    rng = np.random.default_rng(seed)
    ddr = bytearray(rng.integers(0, 256, 1 << 14, dtype=np.uint8).tobytes())
    ddr[0x2000:0x3000] = rng.uniform(-4, 4, 1024).astype("<f4").tobytes()
    for i in range(4):
        op = C.OP_GEMM if rng.random() < .7 else C.OP_RELU
        Mm, Nn, Kk = int(rng.integers(1, 5)), 16 * int(rng.integers(1, 3)), 16 * int(rng.integers(1, 3))
        act, out, bias = int(rng.integers(0, 3)), int(rng.integers(0, 3)), int(rng.integers(0, 2))
        if op == C.OP_RELU:
            f = dict(OPCODE=op, EPI=C.epi(max(act, 1), out), A_OFF=0x2000, Y_OFF=0x1000 + 0x400 * i, M=Mm, N=Nn)
        else:
            f = dict(OPCODE=op, CTRL=C.ctrl(int(rng.integers(1, 3)), fusion=int(act != 0)),
                     EPI=C.epi(act, out, bias), A_OFF=0x100, W_OFF=0x400, SCALE_OFF=0x2000,
                     BIAS_OFF=0x2100, Y_OFF=0x1000 + 0x400 * i, M=Mm, N=Nn, K=Kk)
        if out == C.OUT_INT8:
            f["OUT_SCALE"] = C.f32_bits(float(rng.uniform(0.5, 30)))
        raw = C.encode({**f, "MAGIC": C.MAGIC, "VERSION": C.VERSION, "VALID": 1})
        if rng.random() < .1:
            b = bytearray(raw)
            b[int(rng.integers(0, 64))] ^= 1 << int(rng.integers(0, 8))
            raw = bytes(b)
        ddr[0x3000 + 64 * i:0x3040 + 64 * i] = raw
    a, b = P0.Machine(bytearray(ddr)), Machine(bytearray(ddr))
    sa, sb = a.run_queue(0x3000, 4), b.run_queue(0x3000, 4)
    assert (sa.done, sa.err_code, sa.op_idx) == (sb.done, sb.err_code, sb.op_idx)
    assert a.ddr == b.ddr


# ---------------------------------------------------------------------- v0.2: strides, NaN
def test_canonical_nan_from_mul_and_add():
    inf, nan = np.array([np.inf], F32), np.array([np.nan], F32)
    neg_nan = np.array([0xFFC00000], np.uint32).view(F32)
    assert F.fmul(inf, np.array([0.0], F32)).view(np.uint32)[0] == C.CANON_NAN
    assert F.fadd(inf, -inf).view(np.uint32)[0] == C.CANON_NAN
    assert F.fadd(neg_nan, np.array([1.0], F32)).view(np.uint32)[0] == C.CANON_NAN
    assert F.fmul(nan, nan).view(np.uint32)[0] == C.CANON_NAN


def test_nan_reaches_fp32_output_canonical():
    """inf scale x zero accumulator = NaN; stored FP32 it must be 0x7FC00000, not x86's 0xFFC00000."""
    ddr, good = _base()
    ddr[0x100:0x100 + K] = bytes(K)                             # A = 0 -> acc = 0
    ddr[0x400:0x404] = struct.pack("<f", np.inf)
    st, m = _run(ddr, C.encode(good))
    assert st.done
    assert set(np.frombuffer(m.read(0x800, 64), "<u4").tolist()) == {C.CANON_NAN}


@pytest.mark.parametrize("lda16,ldy16,code", [
    (1, 0, C.E_OK),        # 16-byte input stride holds the K=16 row
    (0, 3, C.E_ALIGN),     # 48 bytes cannot hold a 64-byte fp32 output row
    (0, 4, C.E_OK),        # exactly one row
    (2, 5, C.E_OK)])
def test_stride_must_hold_a_row(lda16, ldy16, code):
    ddr, good = _base()
    st, _ = _run(ddr, C.encode({**good, "LDA16": lda16, "LDY16": ldy16}))
    assert st.err_code == code


@pytest.mark.parametrize("seed", range(10))
def test_strided_gemm_equals_dense(seed):
    """Strided A rows and strided Y rows give the dense result, and touch nothing in the gaps."""
    rng = np.random.default_rng(seed)
    Mm, Nn, Kk = int(rng.integers(1, 7)), 16 * int(rng.integers(1, 4)), 16 * int(rng.integers(1, 4))
    lda, ldy = Kk + 16 * int(rng.integers(1, 4)), Nn * 4 + 16 * int(rng.integers(1, 4))
    A = rng.integers(-128, 128, (Mm, Kk), dtype=np.int8)
    ddr = bytearray(rng.integers(0, 256, 1 << 16, dtype=np.uint8).tobytes())
    for mm in range(Mm):
        ddr[0x100 + mm * lda:0x100 + mm * lda + Kk] = A[mm].tobytes()
    dense = bytearray(ddr)
    dense[0x4000:0x4000 + Mm * Kk] = A.tobytes()
    W = rng.integers(-128, 128, Nn * Kk, dtype=np.int8).tobytes()
    s = rng.uniform(1e-3, 1e-2, Nn).astype("<f4").tobytes()
    for mem in (ddr, dense):
        mem[0x5000:0x5000 + Nn * Kk] = W              # the queue sits at 0x2000 (_run)
        mem[0x3800:0x3800 + 4 * Nn] = s
    common = dict(MAGIC=C.MAGIC, VERSION=C.VERSION, OPCODE=C.OP_GEMM, CTRL=C.ctrl(C.QM_CHANNEL),
                  W_OFF=0x5000, SCALE_OFF=0x3800, M=Mm, N=Nn, K=Kk, VALID=1)
    before = bytes(ddr)
    st1, m1 = _run(ddr, C.encode({**common, "A_OFF": 0x100, "LDA16": lda // 16,
                                  "Y_OFF": 0x6000, "LDY16": ldy // 16}))
    st2, m2 = _run(dense, C.encode({**common, "A_OFF": 0x4000, "Y_OFF": 0x6000}))
    assert st1.done and st2.done
    rowb = Nn * 4
    for mm in range(Mm):
        assert m1.read(0x6000 + mm * ldy, rowb) == m2.read(0x6000 + mm * rowb, rowb)
        gap = (0x6000 + mm * ldy + rowb, 0x6000 + (mm + 1) * ldy)
        assert m1.ddr[gap[0]:gap[1]] == before[gap[0]:gap[1]]          # gaps untouched


def test_input_in_the_gap_between_output_rows_is_legal():
    """Overlap is judged per row segment: an input that sits between output rows is fine."""
    ddr, good = _base()
    f = {**good, "Y_OFF": 0x0C0, "LDY16": 8}       # rows at 0x0C0..0x100, 0x140.. (M=1: one row)
    st, _ = _run(ddr, C.encode({**f, "M": 1}))
    assert st.done
    with pytest.raises(ContractError):            # but a row that covers A is refused
        _run(ddr, C.encode({**good, "Y_OFF": 0x0E0}))


# ---------------------------------------------------------------------- v0.3 primitives: accuracy
def _ulp_err(got, ref):
    sp = np.spacing(np.abs(ref).astype(F32)).astype(np.float64)
    return np.abs(got.astype(np.float64) - ref) / sp


def test_exp_recip_rsqrt_error_bounds():
    """The contract's error bounds, measured: exp <= 3 ulp, recip and rsqrt <= 1.25 ulp (normal
    results); and the exact cases a compiler relies on."""
    rng = np.random.default_rng(30)
    x = rng.uniform(-87.3, 88.7, 300000).astype(F32)
    assert _ulp_err(F.exp_f(x), np.exp(x.astype(np.float64))).max() <= 3.0
    x = (np.exp(rng.uniform(-87.3, 87.3, 300000)) * rng.choice([-1, 1], 300000)).astype(F32)
    assert _ulp_err(F.recip_f(x), 1 / x.astype(np.float64)).max() <= 1.25   # results >= 2^-126
    x = np.exp(rng.uniform(-87.3, 88.7, 300000)).astype(F32)
    assert _ulp_err(F.rsqrt_f(x), 1 / np.sqrt(x.astype(np.float64))).max() <= 1.25
    p2 = np.ldexp(1.0, np.arange(-125, 126)).astype(F32)
    assert (F.recip_f(p2) == (1 / p2.astype(np.float64)).astype(F32)).all()
    assert F.exp_f(np.array([0.0], F32))[0] == 1 and F.rsqrt_f(np.array([4.0], F32))[0] == 0.5


def test_exp_recip_rsqrt_specials():
    inf, nan = np.inf, np.nan
    e = F.exp_f(np.array([inf, -inf, nan, 89.0, -104.0], F32))
    assert e[0] == inf and e[1] == 0 and np.isnan(e[2]) and e[3] == inf and e[4] == 0
    r = F.recip_f(np.array([0.0, -0.0, inf, -inf, 2.0**126, 1.5 * 2.0**126], F32))
    assert list(r[:2]) == [inf, -inf] and r[2] == 0 and np.signbit(r[3]) and r[4] == 2.0**-126 and r[5] == 0
    q = F.rsqrt_f(np.array([0.0, -0.0, inf, -1.0, -inf], F32))
    assert list(q[:2]) == [inf, -inf] and q[2] == 0 and np.isnan(q[3:]).all()
    assert (F.canon(np.array([nan], F32)).view(np.uint32) == C.CANON_NAN).all()


def test_gelu_silu_error_and_tails():
    rng = np.random.default_rng(31)
    x = rng.uniform(-12, 12, 300000).astype(F32)
    xd = x.astype(np.float64)
    gt = 0.5 * xd * (1 + np.tanh(np.sqrt(2 / np.pi) * (xd + 0.044715 * xd ** 3)))
    assert np.abs(F.gelu_f(x) - gt).max() <= 1e-6
    st = xd / (1 + np.exp(-xd))
    assert np.abs(F.silu_f(x) - st).max() <= 3e-6
    big = np.array([1e20, -1e20, 1e5, -1e5], F32)
    g, s = F.gelu_f(big), F.silu_f(big)
    assert g[0] == big[0] and g[1] == 0 and g[2] == big[2] and g[3] == 0
    assert s[0] == big[0] and s[1] == 0 and s[2] == big[2] and s[3] == 0


def test_activation_codes_v03():
    """EPI[5] extends ACT; v0.4 uses codes 5-7 (sigmoid, softplus, exp); EPI[15:6] are E_FIELD;
    v0.2 encodings keep their meaning."""
    assert C.epi(C.ACT_GELU) == 3 and C.epi(C.ACT_SILU) == 0x20 and C.act_of(C.epi(C.ACT_SILU)) == 4
    base = dict(MAGIC=C.MAGIC, VERSION=C.VERSION, VALID=1, OPCODE=C.OP_RELU, A_OFF=0x100,
                Y_OFF=0x400, M=2, N=16)
    for e, ok in [(C.epi(C.ACT_GELU), True), (C.epi(C.ACT_SILU), True), (0x21, True),
                  (0x22, True), (0x23, True), (0x41, False)]:
        ddr = bytearray(1 << 14)
        ddr[0x3000:0x3040] = C.encode({**base, "EPI": e})
        st = Machine(ddr).run_queue(0x3000, 1)
        assert (st.err_code == C.E_OK) == ok and (ok or st.err_code == C.E_FIELD), (hex(e), st)
