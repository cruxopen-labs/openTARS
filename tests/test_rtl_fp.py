"""rtl/tm_fp_pkg.sv against tars_mini/numerics.py, bit for bit, on random and edge vectors."""
import numpy as np
import pytest

from tars_mini import numerics as F
from tars_mini import rtlsim

pytestmark = pytest.mark.skipif(not rtlsim.have_verilator(), reason="needs Verilator")

F32 = np.float32
U32 = np.uint32


def _bits(x) -> np.ndarray:
    return np.asarray(x, F32).view(U32)


def _f(b) -> np.ndarray:
    return np.asarray(b, U32).view(F32)


EDGE = np.array([0x00000000, 0x80000000, 0x00000001, 0x807FFFFF, 0x00800000, 0x80800000,
                 0x00800001, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000,
                 0xFFC00000, 0x7F800001, 0x3F800000, 0xBF800000, 0x3F7FFFFF, 0x40C00000,
                 0x40C00001, 0x3F000000, 0x3FC00000, 0x40200000, 0x42FE0000, 0x42FF0000,
                 0x43000000, 0x34000000, 0x1F800000, 0x20000000, 0x00FFFFFF], dtype=U32)


def _random_bits(rng, n):
    return rng.integers(0, 2**32, n, dtype=np.uint64).astype(U32)


def _random_floats(rng, n, lo=-40, hi=40):
    """Normal floats with exponents in [lo, hi], random signs and mantissas."""
    e = rng.integers(lo + 127, hi + 128, n).astype(U32)
    m = rng.integers(0, 2**23, n, dtype=np.uint64).astype(U32)
    s = rng.integers(0, 2, n).astype(U32)
    return (s << 31) | (e << 23) | m


def _check(op, a, b, want):
    if op in ("fadd", "fmul"):        # the three stage functions must agree as well
        _check(op + "3", a, b, want)
    if op in rtlsim.FPU_OPS:          # and the pipelined unit, fed one operation per cycle
        got = rtlsim.run_fpu(op, [(int(x), int(y)) for x, y in zip(a, b)])
        _compare(op + " (pipelined)", a, b, got, want)
    got = rtlsim.run_fp([(rtlsim.FP_OPS[op], int(x), int(y)) for x, y in zip(a, b)])
    _compare(op, a, b, got, want)


def _compare(op, a, b, got, want):
    want = np.asarray(want, dtype=np.uint64)
    assert len(got) == len(want), (op, len(got), len(want))
    bad = np.nonzero(got != want)[0]
    if len(bad):
        i = bad[0]
        raise AssertionError(f"{op}: {len(bad)}/{len(a)} differ; first a={int(a[i]):08x} "
                             f"b={int(b[i]):08x} rtl={int(got[i]):08x} golden={int(want[i]):08x}")


def test_fmul():
    rng = np.random.default_rng(10)
    a = [_random_bits(rng, 40000), _random_floats(rng, 40000), np.repeat(EDGE, len(EDGE))]
    b = [_random_bits(rng, 40000), _random_floats(rng, 40000), np.tile(EDGE, len(EDGE))]
    # products around 2^-126: the subnormal-grid rounding that may carry into the normal range
    x = _f(_random_floats(rng, 60000, -70, -50))
    t = np.ldexp(1.0, -126) * (1 + rng.integers(-40, 40, 60000) * 2.0**-24)
    y = (t / x.astype(np.float64)).astype(F32)
    a.append(_bits(x))
    b.append(_bits(y))
    # exact ties: small odd integer mantissas make 25-bit products (guard set, nothing below)
    ka = rng.integers(2**11, 2**13, 40000) | 1
    kb = rng.integers(2**11, 2**13, 40000) | 1
    sc = np.ldexp(1.0, rng.integers(-60, 60, 40000))
    a.append(_bits((ka * sc).astype(F32)))
    b.append(_bits(kb.astype(F32) * rng.choice([-1, 1], 40000).astype(F32)))
    # deep underflow and overflow
    a.append(_random_floats(rng, 20000, -80, -40))
    b.append(_random_floats(rng, 20000, -100, -60))
    a.append(_random_floats(rng, 20000, 60, 127))
    b.append(_random_floats(rng, 20000, 40, 127))
    a, b = np.concatenate(a), np.concatenate(b)
    _check("fmul", a, b, _bits(F.fmul(_f(a), _f(b))))


def test_fadd():
    rng = np.random.default_rng(11)
    x = _random_floats(rng, 50000)
    a = [_random_bits(rng, 40000), x, np.repeat(EDGE, len(EDGE))]
    b = [_random_bits(rng, 40000), _random_floats(rng, 50000), np.tile(EDGE, len(EDGE))]
    # cancellation: b = -a +- a few ulp, and close exponents
    near = (x ^ U32(0x80000000)).astype(np.int64) + rng.integers(-3, 4, 50000)
    a.append(x)
    b.append(near.astype(U32))
    xs = _random_floats(rng, 50000, -126, -120)                   # sums that fall subnormal
    a.append(xs)
    b.append((xs ^ U32(0x80000000)).astype(np.int64).__add__(rng.integers(-5000, 5000, 50000)).astype(U32))
    a.append(_random_floats(rng, 30000, 120, 127))                # overflow
    b.append(_random_floats(rng, 30000, 120, 127))
    # exact ties: a 24-bit integer plus a small integer shifted below its last bit
    big = rng.integers(2**23, 2**24, 40000).astype(np.float64)
    small = (rng.integers(1, 8, 40000) * 2 - 1) * np.ldexp(1.0, -rng.integers(1, 4, 40000))
    a.append(_bits(big.astype(F32)))
    b.append(_bits((small * rng.choice([-1, 1], 40000)).astype(F32)))
    a.append(_random_floats(rng, 30000, 0, 5))                    # alignment shifts 0..30
    b.append(_random_floats(rng, 30000, -30, 5))
    a, b = np.concatenate(a), np.concatenate(b)
    _check("fadd", a, b, _bits(F.fadd(_f(a), _f(b))))


def test_i2f():
    rng = np.random.default_rng(12)
    v = [rng.integers(-2**31, 2**31, 40000), rng.integers(-2**24, 2**24, 20000)]
    p = 2 ** rng.integers(24, 31, 20000)
    v.append(p + rng.integers(-300, 300, 20000))
    v.append(-(p + rng.integers(-300, 300, 20000)))
    v.append(np.array([0, 1, -1, 2**24 + 1, 2**24 + 3, 2**31 - 1, -2**31, 2**30 + 64, 2**30 + 192]))
    v = np.clip(np.concatenate(v), -2**31, 2**31 - 1).astype(np.int64)
    _check("i2f", v & 0xFFFFFFFF, np.zeros_like(v), _bits(F.i2f(v)))


@pytest.mark.parametrize("act", [0, 1, 2, 3, 4, 5, 6, 7])
def test_act(act):
    rng = np.random.default_rng(13 + act)
    n = 20000 if act < 3 else 6000
    a = np.concatenate([_random_bits(rng, n), _random_floats(rng, n, -5, 5),
                        _random_floats(rng, n // 2, -30, 7), EDGE,
                        np.arange(0x41A00000 - 40, 0x41A00000 + 40, dtype=U32)])   # softplus's threshold, 20
    _check("act", a, np.full(len(a), act, U32), _bits(F.act_fn(_f(a), act)))


@pytest.mark.skipif(not rtlsim.FPGA, reason="the ASIC fpu_act needs one act at a time (TM_FPGA=1)")
def test_act_mixed():
    """The FPGA fpu_act shares one exp and one recip among its functions, each at one fixed cycle:
    a different function on every cycle must not disturb any result."""
    rng = np.random.default_rng(29)
    a = np.concatenate([_random_bits(rng, 20000), _random_floats(rng, 20000, -5, 5),
                        _random_floats(rng, 10000, -30, 7), EDGE])
    acts = rng.integers(0, 8, len(a)).astype(U32)
    want = np.empty(len(a), U32)
    for k in range(8):
        sel = acts == k
        want[sel] = _bits(F.act_fn(_f(a[sel]), k))
    got = rtlsim.run_fpu("act", [(int(x), int(y)) for x, y in zip(a, acts)])
    _compare("act (mixed, pipelined)", a, acts, got, want)


def test_bf16():
    rng = np.random.default_rng(16)
    x = _random_bits(rng, 40000)
    ties = (x & U32(0xFFFF0000)) | U32(0x8000)
    a = np.concatenate([x, ties, EDGE])
    _check("bf16", a, np.zeros(len(a), U32), F.to_bf16_bits(_f(a)).astype(np.uint64))


def test_q8():
    rng = np.random.default_rng(17)
    x = np.concatenate([_random_floats(rng, 40000, -8, 8), _random_bits(rng, 20000), EDGE])
    inv = np.concatenate([_random_floats(rng, 40000, -4, 8) & U32(0x7FFFFFFF),
                          _random_floats(rng, 20000, 0, 3) & U32(0x7FFFFFFF), np.full(len(EDGE), 0x3F800000, U32)])
    # exact halves: x * 1.0 = k + 0.5
    k = rng.integers(-130, 130, 20000)
    x = np.concatenate([x, _bits((k + 0.5).astype(F32))])
    inv = np.concatenate([inv, np.full(20000, 0x3F800000, U32)])
    want = F.q8(_f(x), _f(inv)).view(np.uint8).astype(np.uint64)
    _check("q8", x, inv, want)


# ---------------------------------------------------------------------- v0.3 primitives
def _exp_ties(rng, n):
    """x with fmul(x, log2e) exactly k + 0.5: the round-to-even case of exp's rint."""
    out = []
    for k in rng.integers(-149, 128, n):
        t = F32(k + 0.5)
        x0 = (np.float64(t) / np.float64(F.LOG2E)).astype(F32)
        for d in range(-4, 5):
            x = (_bits(x0).astype(np.int64) + d).astype(U32)
            if F.fmul(_f(x), F.LOG2E) == t:
                out.append(x)
    return np.array(out, U32)


def test_exp():
    rng = np.random.default_rng(20)
    ties = _exp_ties(rng, 3000)
    assert len(ties) > 1000
    a = np.concatenate([_random_bits(rng, 30000), _bits(rng.uniform(-110, 95, 40000).astype(F32)),
                        _random_floats(rng, 20000, -30, 7), ties, EDGE,
                        _bits(np.array([88.72, 88.73, 89.0, 88.99, -87.33, -87.34, -103.9, -104.0,
                                        -0.0, 0.0], F32))])
    _check("exp", a, np.zeros(len(a), U32), _bits(F.exp_f(_f(a))))


def test_recip():
    rng = np.random.default_rng(21)
    a = np.concatenate([_random_bits(rng, 40000), _random_floats(rng, 40000, -126, 127), EDGE,
                        _bits(np.ldexp(1.0, np.arange(-126, 128)).astype(F32)),
                        _random_floats(rng, 5000, 124, 127), _random_floats(rng, 5000, -126, -123)])
    _check("recip", a, np.zeros(len(a), U32), _bits(F.recip_f(_f(a))))


def test_rsqrt():
    rng = np.random.default_rng(22)
    a = np.concatenate([_random_bits(rng, 40000), _random_floats(rng, 40000, -126, 127) & U32(0x7FFFFFFF),
                        EDGE, _bits(np.ldexp(1.0, np.arange(-126, 128)).astype(F32)),
                        _bits((np.arange(1, 2000) ** 2).astype(F32))])
    _check("rsqrt", a, np.zeros(len(a), U32), _bits(F.rsqrt_f(_f(a))))
