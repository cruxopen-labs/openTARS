"""The contract's fp32 numerics: roundTiesToEven only, flush-to-zero, no FMA, canonical NaN.

Each function is one step of the contract's numeric sequence (数值语义 1-6) and rounds once.
numpy's float32 add and multiply round to nearest even, so they are used per step and the
results are flushed on the way in and out.
"""
from __future__ import annotations

import numpy as np

F32 = np.float32


def to_fp16_bits(x) -> np.ndarray:
    """IEEE binary16 storage, RNE, gradual underflow, signed zero, canonical NaN."""
    with np.errstate(over="ignore", invalid="ignore"):
        h = np.asarray(x, F32).astype("<f2")
    return np.where(np.isnan(h), np.uint16(0x7E00), h.view(np.uint16)).astype(np.uint16)


def from_fp16_bits(x) -> np.ndarray:
    """Exact expansion to FP32, including binary16 subnormals; canonicalize NaNs."""
    return canon(np.asarray(x, dtype=np.uint16).view(np.float16).astype(F32))


def ftz(x) -> np.ndarray:
    """Denormals become a zero of the same sign."""
    x = np.asarray(x, dtype=F32)
    b = x.view(np.uint32)
    den = ((b & 0x7F800000) == 0) & ((b & 0x007FFFFF) != 0)
    return np.where(den, (b & 0x80000000).view(F32), x).astype(F32)


def canon(x) -> np.ndarray:
    """v0.2: every NaN result of a mul or add is the canonical 0x7FC00000."""
    x = np.asarray(x, dtype=F32)
    return np.where(np.isnan(x), np.uint32(0x7FC00000).view(F32), x).astype(F32)


def fmul(a, b) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return canon(ftz(ftz(a) * ftz(b)))


def fadd(a, b) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return canon(ftz(ftz(a) + ftz(b)))


def i2f(i) -> np.ndarray:
    """int32 -> fp32, round to nearest even."""
    return np.asarray(i, dtype=np.int64).astype(F32)


# ---------------------------------------------------------------------- v0.3 math primitives
# Each is a fixed sequence of the contract's fmul / fadd (each rounded once, RNE, FTZ, no FMA)
# plus exact integer operations on the bits; constants are given as fp32 bit patterns.
U32 = np.uint32


def _c(bits: int) -> np.float32:
    return np.array([bits], dtype=U32).view(F32)[0]


def _neg(x) -> np.ndarray:
    """Sign flip (exact; not an fp operation)."""
    return (np.asarray(x, F32).view(U32) ^ U32(0x80000000)).view(F32)


ONE, TWO, HALF, THREE_HALF = _c(0x3F800000), _c(0x40000000), _c(0x3F000000), _c(0x3FC00000)
LOG2E = _c(0x3FB8AA3B)                                    # 1.44269504
NEG_LN2_HI, NEG_LN2_LO = _c(0xBF317200), _c(0xB5BFBE8E)   # -0.693145751953125, -1.42860677e-06
EXP_POLY = [_c(0x3AB60B61), _c(0x3C088889), _c(0x3D2AAAAB),  # 1/720, 1/120, 1/24
            _c(0x3E2AAAAB), _c(0x3F000000), _c(0x3F800000), _c(0x3F800000)]  # 1/6, 1/2, 1, 1
EXP_HI, EXP_LO = _c(0x42B20000), _c(0xC2D00000)           # 89.0, -104.0
RECIP_MAGIC, RSQRT_MAGIC = 0x7EF311C7, 0x5F3759DF
GELU_A, GELU_C = _c(0x3D372713), _c(0x3F4C422A)           # 0.044715, sqrt(2/pi) = 0.7978846


def _set_exp(y, e) -> np.ndarray:
    """y (positive normal) with e added to its biased exponent, exactly: >= 255 -> inf, <= 0 -> +0
    (FTZ)."""
    yb = np.asarray(y, F32).view(U32).astype(np.int64)
    ne = ((yb >> 23) & 0xFF) + e
    out = np.where(ne >= 255, 0x7F800000, np.where(ne <= 0, 0, (ne << 23) | (yb & 0x7FFFFF)))
    return out.astype(U32).view(F32)


def exp_f(x) -> np.ndarray:
    """e^x. x >= 89 -> +inf; x <= -104 -> +0; NaN -> canonical NaN. Otherwise
    n = rint(x * log2e) (ties to even), r = (x + n * -ln2_hi) + n * -ln2_lo, p = Horner of the
    degree-6 polynomial EXP_POLY at r, and the result is p with n added to its exponent field
    (exact): an exponent >= 255 gives +inf, <= 0 gives +0 (FTZ)."""
    x = ftz(x)
    with np.errstate(invalid="ignore"):
        n = np.clip(np.rint(fmul(x, LOG2E)), -200, 200).astype(F32)
    r = fadd(fadd(x, fmul(n, NEG_LN2_HI)), fmul(n, NEG_LN2_LO))
    p = np.full_like(r, EXP_POLY[0])
    for c in EXP_POLY[1:]:
        p = fadd(fmul(p, r), c)
    y = _set_exp(p, np.nan_to_num(n, nan=0).astype(np.int64))
    with np.errstate(invalid="ignore"):
        y = np.where(x >= EXP_HI, F32(np.inf), np.where(x <= EXP_LO, F32(0), y))
    return canon(np.where(np.isnan(x), F32(np.nan), y).astype(F32))


def recip_f(x) -> np.ndarray:
    """1/x. +-0 -> +-inf; +-inf -> +-0; NaN -> canonical NaN. Otherwise, with m = |x| scaled to
    [1, 2) (its exponent field set to 127) and E = x's biased exponent - 127:
    y0 = bits(RECIP_MAGIC - bits(m)); three Newton steps in correction form, e = 1 - m * y,
    y = y + y * e; then E is subtracted from y's exponent (exact; <= 0 gives 0) and x's sign set."""
    x = ftz(x)
    b = x.view(U32).astype(np.int64)
    sgn, a = b & 0x80000000, b & 0x7FFFFFFF
    m = ((a & 0x7FFFFF) | (127 << 23)).astype(U32).view(F32)
    y = (np.int64(RECIP_MAGIC) - (a & 0x7FFFFF | (127 << 23))).astype(U32).view(F32)
    for _ in range(3):
        y = fadd(y, fmul(y, fadd(ONE, _neg(fmul(m, y)))))
    y = (_set_exp(y, -((a >> 23) - 127)).view(U32).astype(np.int64) | sgn).astype(U32).view(F32)
    y = np.where(a == 0, (sgn | 0x7F800000).astype(U32).view(F32), y)
    y = np.where(a == 0x7F800000, sgn.astype(U32).view(F32), y)
    return canon(np.where(np.isnan(x), F32(np.nan), y).astype(F32))


def rsqrt_f(x) -> np.ndarray:
    """1/sqrt(x). +-0 -> +-inf; +inf -> +0; x < 0 or NaN -> canonical NaN. Otherwise, with
    E = biased exponent - 127 = 2k + j (j = 0 or 1) and m = x scaled to [1, 4) (exponent field
    127 + j): y0 = bits(RSQRT_MAGIC - (bits(m) >> 1)); three Newton steps in correction form,
    e = 0.5 - 0.5 * ((m * y) * y), y = y + y * e; then k is subtracted from y's exponent."""
    x = ftz(x)
    b = x.view(U32).astype(np.int64)
    E = ((b >> 23) & 0xFF) - 127
    j, k = E & 1, E >> 1
    mb = (b & 0x7FFFFF) | ((127 + j) << 23)
    m = mb.astype(U32).view(F32)
    y = (np.int64(RSQRT_MAGIC) - (mb >> 1)).astype(U32).view(F32)
    for _ in range(3):
        y = fadd(y, fmul(y, fadd(HALF, _neg(fmul(HALF, fmul(fmul(m, y), y))))))
    y = _set_exp(y, -k)
    a = b & 0x7FFFFFFF
    y = np.where(a == 0, ((b & 0x80000000) | 0x7F800000).astype(U32).view(F32), y)
    y = np.where(b == 0x7F800000, F32(0), y)
    with np.errstate(invalid="ignore"):
        bad = np.isnan(x) | (x < F32(0))
    return canon(np.where(bad, F32(np.nan), y).astype(F32))


def gelu_f(x) -> np.ndarray:
    """0.5 x (1 + tanh(c (x + a x^3))), tanh(u) = 1 - 2 / (e^(2u) + 1), in this order:
    x3 = (x*x)*x; u = c * (x + a*x3); t = 1 - 2 * recip(exp(2u) + 1); y = (0.5*x) * (1 + t)."""
    x = ftz(x)
    x3 = fmul(fmul(x, x), x)
    u = fmul(GELU_C, fadd(x, fmul(GELU_A, x3)))
    t = fadd(ONE, _neg(fmul(TWO, recip_f(fadd(exp_f(fmul(TWO, u)), ONE)))))
    return fmul(fmul(HALF, x), fadd(ONE, t))


def silu_f(x) -> np.ndarray:
    """x * sigmoid(x) = x * recip(1 + exp(-x))."""
    x = ftz(x)
    return fmul(x, recip_f(fadd(ONE, exp_f(_neg(x)))))


def act_fn(x, act: int) -> np.ndarray:
    """0: identity; 1: relu(x) = x if x > +0 else +0 (-0 and NaN give +0); 2: relu6;
    3 (v0.3): GELU (tanh form, gelu_f); 4 (v0.3): SiLU (silu_f); 5 (v0.4): sigmoid_f;
    6 (v0.4): softplus_f; 7 (v0.4): exp_f."""
    x = ftz(x)
    if act == 0:
        return x
    if act == 3:
        return gelu_f(x)
    if act == 4:
        return silu_f(x)
    if act == 5:
        return sigmoid_f(x)
    if act == 6:
        return softplus_f(x)
    if act == 7:
        return exp_f(x)
    r = np.where(x > F32(0), x, F32(0)).astype(F32)
    if act == 1:
        return r
    return np.where(r > F32(6), F32(6), r).astype(F32)


# ---------------------------------------------------------------------- v0.4 primitives
LOG1P_POLY = [_c(0x3D70F0F1), _c(0x3D888889), _c(0x3D9D89D9), _c(0x3DBA2E8C),  # 1/17 1/15 1/13 1/11
              _c(0x3DE38E39), _c(0x3E124925), _c(0x3E4CCCCD), _c(0x3EAAAAAB),  # 1/9 1/7 1/5 1/3
              _c(0x3F800000)]                                                 # 1
SP_HI = _c(0x41A00000)                                     # 20.0
Q127, INV127 = _c(0x42FE0000), _c(0x3C010204)              # 127.0, fp32(1/127)


def sigmoid_f(x) -> np.ndarray:
    """recip(1 + exp(-x)) (the factor SiLU multiplies by)."""
    x = ftz(x)
    return recip_f(fadd(ONE, exp_f(_neg(x))))


def softplus_f(x) -> np.ndarray:
    """ln(1 + e^x), with torch's threshold: x > 20 -> x. Otherwise max(x, 0) + ln(1 + t),
    t = exp(-|x|) in [0, 1], ln(1 + t) = 2 atanh(s): s = t * recip(t + 2), z = s * s,
    p = Horner of LOG1P_POLY at z (1/17, 1/15, ..., 1/3, 1), y = relu(x) + (2 * s) * p.
    NaN -> canonical NaN."""
    x = ftz(x)
    t = exp_f(_neg(fabs(x)))
    sv = fmul(t, recip_f(fadd(t, TWO)))
    z = fmul(sv, sv)
    p = np.full_like(z, LOG1P_POLY[0])
    for c in LOG1P_POLY[1:]:
        p = fadd(fmul(p, z), c)
    y = fadd(np.where(x > F32(0), x, F32(0)).astype(F32), fmul(fmul(TWO, sv), p))
    with np.errstate(invalid="ignore"):
        y = np.where(x > SP_HI, x, y)
    return canon(np.where(np.isnan(x), F32(np.nan), y).astype(F32))


def fabs(x) -> np.ndarray:
    """Sign bit cleared (exact)."""
    return (ftz(x).view(U32) & U32(0x7FFFFFFF)).view(F32)


def to_bf16_bits(x) -> np.ndarray:
    """fp32 -> bf16 bits, round to nearest even; NaN -> 0x7FC0."""
    b = ftz(x).view(np.uint32).astype(np.uint64)
    nan = ((b & 0x7F800000) == 0x7F800000) & ((b & 0x007FFFFF) != 0)
    r = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF
    return np.where(nan, 0x7FC0, r).astype(np.uint16)


def bf16_bits_to_f32(h) -> np.ndarray:
    return (np.asarray(h, dtype=np.uint32) << 16).view(F32)


def q8(x, inv) -> np.ndarray:
    """rint(fmul(x, inv)), ties to even, saturated to [-127, 127]; NaN -> 0."""
    y = fmul(x, inv)
    y = np.where(np.isnan(y), F32(0), y)
    return np.clip(np.rint(y), -127, 127).astype(np.int8)


# ---------------------------------------------------------------------- v0.3 row reductions
LANES = 16


def fmax(a, b) -> np.ndarray:
    """max with a fixed answer for every pair, so it is commutative and associative: NaN if
    either is NaN (canonical); of two equal values (+0 and -0 included) the one whose sign bit is
    the AND of both signs (-0 only if both are -0)."""
    a, b = ftz(a), ftz(b)
    with np.errstate(invalid="ignore"):
        eq = a == b
        m = np.where(a > b, a, b)
    sgn = (a.view(U32) & b.view(U32) & U32(0x80000000)) | (a.view(U32) & U32(0x7FFFFFFF))
    m = np.where(eq, sgn.view(F32), m)
    return canon(np.where(np.isnan(a) | np.isnan(b), F32(np.nan), m).astype(F32))


def sum16(v) -> np.ndarray:
    """Row sums in the contract's order: element j of a row goes to lane j mod 16, each lane adds
    its elements in order starting from +0 (one fadd each), then the 16 lanes fold pairwise:
    (0+1), (2+3), ... then again, four levels. v: [M, N] with N a multiple of 16 -> [M]."""
    v = np.asarray(v, F32)
    lanes = np.zeros((v.shape[0], LANES), F32)
    for j in range(v.shape[1]):
        lanes[:, j % LANES] = fadd(lanes[:, j % LANES], v[:, j])
    while lanes.shape[1] > 1:
        lanes = fadd(lanes[:, 0::2], lanes[:, 1::2])
    return lanes[:, 0]


def max16(v) -> np.ndarray:
    """Row maxima with fmax (order-free, so any lane split gives the same result)."""
    v = np.asarray(v, F32)
    m = v[:, 0]
    for j in range(1, v.shape[1]):
        m = fmax(m, v[:, j])
    return m


def softmax_rows(x) -> np.ndarray:
    """mx = max16(x); e = exp(x + -mx); r = recip(sum16(e)); y = e * r."""
    x = ftz(x)
    mx = max16(x)[:, None]
    e = exp_f(fadd(x, _neg(mx)))
    return fmul(e, recip_f(sum16(e))[:, None])


def inv_n(n: int) -> np.float32:
    """1/N as the contract computes it: recip(i2f(N))."""
    return recip_f(i2f(np.array([n])))[0]


def layernorm_rows(x, gamma, beta, eps) -> np.ndarray:
    """mean = sum16(x) * 1/N; d = x + -mean; var = sum16(d * d) * 1/N;
    rs = rsqrt(var + eps); y = (d * rs) * gamma, then + beta if given."""
    x = ftz(x)
    k = inv_n(x.shape[1])
    mean = fmul(sum16(x), k)[:, None]
    d = fadd(x, _neg(mean))
    rs = rsqrt_f(fadd(fmul(sum16(fmul(d, d)), k), eps))[:, None]
    y = fmul(fmul(d, rs), gamma[None, :])
    return y if beta is None else fadd(y, beta[None, :])


def rmsnorm_rows(x, gamma, beta, eps) -> np.ndarray:
    """rs = rsqrt(sum16(x * x) * 1/N + eps); y = (x * rs) * gamma, then + beta if given."""
    x = ftz(x)
    rs = rsqrt_f(fadd(fmul(sum16(fmul(x, x)), inv_n(x.shape[1])), eps))[:, None]
    y = fmul(fmul(x, rs), gamma[None, :])
    return y if beta is None else fadd(y, beta[None, :])


# ---------------------------------------------------------------------- v0.4 row / dot ops
def softmax_rows_causal(x, valid) -> np.ndarray:
    """Softmax over the first valid[m] elements of row m (1 <= valid[m] <= N); the rest are
    outputs +0 and add +0 to their lanes: mx = max over the valid elements; e = exp(x + -mx)
    there, +0 elsewhere; r = recip(sum16(e)); y = e * r."""
    x = ftz(x)
    M, N = x.shape
    keep = np.arange(N)[None, :] < np.asarray(valid)[:, None]
    mx = np.array([max16(x[m:m + 1, :valid[m]])[0] for m in range(M)], F32)[:, None]
    e = np.where(keep, exp_f(fadd(x, _neg(mx))), F32(0)).astype(F32)
    return fmul(e, recip_f(sum16(e))[:, None])


def l2norm_rows(x, gamma, beta, eps) -> np.ndarray:
    """rs = rsqrt(sum16(x * x) + eps); y = (x * rs) * gamma, then + beta if given (RMSNorm
    without the 1/N)."""
    x = ftz(x)
    rs = rsqrt_f(fadd(sum16(fmul(x, x)), eps))[:, None]
    y = fmul(fmul(x, rs), gamma[None, :])
    return y if beta is None else fadd(y, beta[None, :])


def quant_rows(x) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (per-token) symmetric INT8: amax = max16(|x|); q = q8(x, 127 * recip(amax));
    s = amax * fp32(1/127). Returns (q int8 [M, N], s fp32 [M]). An all-zero row gives q = 0
    (0 * inf is NaN, which q8 maps to 0) and s = 0."""
    x = ftz(x)
    amax = max16(fabs(x))
    inv = fmul(Q127, recip_f(amax))
    return q8(x, inv[:, None]), fmul(amax, INV127)


def fdot_nt(a, b) -> np.ndarray:
    """y[m][n] = sum16 over k of a[m][k] * b[n][k] (lane k mod 16, then the pairwise fold)."""
    a, b = ftz(a), ftz(b)
    M, K = a.shape
    N = b.shape[0]
    p = fmul(a[:, None, :], b[None, :, :]).reshape(M * N, K)
    return sum16(p).reshape(M, N)


def fdot_nn(a, b) -> np.ndarray:
    """y[m][n] = (((+0 + a[m][0] * b[0][n]) + a[m][1] * b[1][n]) + ...) in k order."""
    a, b = ftz(a), ftz(b)
    M, K = a.shape
    y = np.zeros((M, b.shape[1]), F32)
    for k in range(K):
        y = fadd(y, fmul(a[:, k:k + 1], b[k:k + 1, :]))
    return y


def rope_rows(x, tab, rows_per_pos: int, rot: int) -> np.ndarray:
    """Rotary embedding of the first `rot` elements of each row; table row t = cos[0:rot/2] then
    sin[0:rot/2]; row m uses table row m // rows_per_pos. With h = rot/2, c = cos[i], s = sin[i]:
    y[i] = x[i] * c + -(x[i + h] * s), y[i + h] = x[i + h] * c + x[i] * s (i < h); y[n] = x[n]
    for n >= rot."""
    x = ftz(x)
    h = rot // 2
    t = ftz(tab)[np.arange(x.shape[0]) // rows_per_pos]
    c, s = t[:, :h], t[:, h:rot]
    y = x.copy()
    y[:, :h] = fadd(fmul(x[:, :h], c), _neg(fmul(x[:, h:rot], s)))
    y[:, h:rot] = fadd(fmul(x[:, h:rot], c), fmul(x[:, :h], s))
    return y
