"""TARS-mini reference: descriptor encode/decode and the bit-exact golden semantics.

This is the seed of the P1 golden model. It implements exactly what the TARS-mini contract
(TARS-mini 契约 v0.1) defines, nothing more:
  - the 512-bit descriptor layout (VERSION 0x02 field positions, mini subset)
  - CRC-32 over the descriptor with the CRC field zeroed
  - GEMM (0x02): INT8 x INT8 -> exact integer sum -> fixed fp32 epilogue
  - RELU (0x03): fp32 tensor -> activation -> output conversion
  - sequential queue semantics and error-before-write

Results never depend on MXU size, tiling, VMEM or DMA: there is no such parameter here.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------- field layout (bit hi, lo)
LAYOUT = {
    "MAGIC":      (511, 496),
    "VERSION":    (495, 480),
    "CORE_ID":    (479, 472),
    "HDR_RSV":    (471, 448),
    "OPCODE":     (447, 432),
    "CTRL":       (431, 416),
    "GROUP_SIZE": (415, 400),
    "EPI":        (399, 384),
    "BASE_ADDR":  (383, 320),
    "A_OFF":      (319, 288),
    "W_OFF":      (287, 256),
    "Y_OFF":      (255, 224),
    "SCALE_OFF":  (223, 192),
    "BIAS_OFF":   (191, 160),
    "M":          (159, 144),
    "N":          (143, 128),
    "K":          (127, 112),
    "BODY_RSV":   (111, 96),
    "OUT_SCALE":  (95, 64),
    "CRC32":      (63, 32),
    "VALID":      (31, 31),
    "TAIL_RSV":   (30, 16),
    "DEP_SIG":    (15, 0),     # V1.3.1 multi-core: DEP_MASK/DEP_SLOT/SIGNAL_MASK/SIGNAL_SLOT; 0 in mini
}

MAGIC, VERSION = 0xAA55, 0x0002
OP_GEMM, OP_RELU = 0x02, 0x03
# CTRL bits (V1.3.1 10.7 positions for the quant fields)
QM_NONE, QM_TENSOR, QM_CHANNEL, QM_GROUP = 0, 1, 2, 3
DT_BF16, DT_INT8, DT_INT4, DT_FP8 = 0, 1, 2, 3
# EPI bits
ACT_NONE, ACT_RELU, ACT_RELU6 = 0, 1, 2
OUT_FP32, OUT_BF16, OUT_INT8 = 0, 1, 2
ESIZE = {OUT_FP32: 4, OUT_BF16: 2, OUT_INT8: 1}

# error codes (CSR 0x803 ERR_CODE)
E_OK, E_MAGIC, E_VERSION, E_CRC, E_VALID, E_OPCODE, E_FIELD, E_ALIGN, E_RANGE = range(9)


def check_layout() -> None:
    """Every field is inside [0, 511], fields do not overlap, and together they cover 512 bits."""
    used = np.zeros(512, dtype=int)
    for hi, lo in LAYOUT.values():
        assert 511 >= hi >= lo >= 0
        used[lo:hi + 1] += 1
    assert (used == 1).all(), "fields overlap or leave holes"


def width(name: str) -> int:
    hi, lo = LAYOUT[name]
    return hi - lo + 1


# ---------------------------------------------------------------- encode / decode
def ctrl(qmode=QM_TENSOR, symmetric=1, a_dtype=DT_INT8, w_dtype=DT_INT8, fusion=0) -> int:
    return (qmode & 7) | (symmetric & 1) << 3 | (a_dtype & 15) << 4 | (w_dtype & 15) << 8 | (fusion & 1) << 12


def epi(act=ACT_NONE, out=OUT_FP32, bias=0) -> int:
    return (act & 3) | (out & 3) << 2 | (bias & 1) << 4


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def encode(fields: dict, *, fix_crc: bool = True) -> bytes:
    """512-bit descriptor as 64 bytes, little endian: byte i holds bits [8i+7 : 8i]."""
    v = 0
    for name, val in fields.items():
        hi, lo = LAYOUT[name]
        assert 0 <= val < (1 << (hi - lo + 1)), (name, val)
        v |= val << lo
    raw = bytearray(v.to_bytes(64, "little"))
    if fix_crc:
        raw[4:8] = b"\0\0\0\0"                       # CRC32 = bits 63:32 = bytes 4..7
        raw[4:8] = struct.pack("<I", zlib.crc32(bytes(raw)) & 0xFFFFFFFF)
    return bytes(raw)


def decode(raw: bytes) -> dict:
    v = int.from_bytes(raw, "little")
    return {n: (v >> lo) & ((1 << (hi - lo + 1)) - 1) for n, (hi, lo) in LAYOUT.items()}


def crc_ok(raw: bytes) -> bool:
    z = bytearray(raw)
    z[4:8] = b"\0\0\0\0"
    return struct.unpack("<I", raw[4:8])[0] == (zlib.crc32(bytes(z)) & 0xFFFFFFFF)


# ---------------------------------------------------------------- fp32 numerics (RNE, FTZ, no FMA)
F32 = np.float32


def ftz(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=F32)
    b = x.view(np.uint32)
    den = ((b & 0x7F800000) == 0) & ((b & 0x007FFFFF) != 0)
    return np.where(den, (b & 0x80000000).view(F32), x).astype(F32)


def fmul(a, b):
    with np.errstate(over="ignore", invalid="ignore"):
        return ftz(ftz(a) * ftz(b))       # numpy float32 multiply rounds RNE


def fadd(a, b):
    with np.errstate(over="ignore", invalid="ignore"):
        return ftz(ftz(a) + ftz(b))


def i2f(i: np.ndarray) -> np.ndarray:
    """int32 -> fp32, round to nearest even (numpy's conversion is RNE)."""
    return np.asarray(i, dtype=np.int64).astype(F32)


def act_fn(x: np.ndarray, act: int) -> np.ndarray:
    x = ftz(x)
    pos = x > F32(0)                       # -0, +0 and NaN are not > +0
    r = np.where(pos, x, F32(0)).astype(F32)
    if act == ACT_NONE:
        return x
    if act == ACT_RELU:
        return r
    return np.where(r > F32(6), F32(6), r).astype(F32)


def to_bf16_bits(x: np.ndarray) -> np.ndarray:
    b = ftz(x).view(np.uint32).astype(np.uint64)
    nan = ((b & 0x7F800000) == 0x7F800000) & ((b & 0x007FFFFF) != 0)
    r = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF
    return np.where(nan, 0x7FC0, r).astype(np.uint16)


def q8(x: np.ndarray, inv: np.float32) -> np.ndarray:
    y = fmul(x, inv)
    y = np.where(np.isnan(y), F32(0), y)
    r = np.rint(y)                          # rint is round-half-even
    return np.clip(r, -127, 127).astype(np.int8)


def convert_out(y: np.ndarray, out: int, out_scale_bits: int) -> bytes:
    if out == OUT_FP32:
        return ftz(y).astype("<f4").tobytes()
    if out == OUT_BF16:
        return to_bf16_bits(y).astype("<u2").tobytes()
    inv = np.frombuffer(struct.pack("<I", out_scale_bits), dtype="<f4")[0]
    return q8(y, inv).tobytes()


# ---------------------------------------------------------------- the machine
class ContractError(Exception):
    """An illegal program the compiler must not emit (UNDEFINED on hardware; the golden refuses)."""


@dataclass
class Status:
    done: bool = False
    err_code: int = E_OK
    op_idx: int = 0           # index of the descriptor that stopped the queue (CUR_OP_IDX)


@dataclass
class Machine:
    ddr: bytearray
    core_id: int = 0
    status: Status = field(default_factory=Status)

    # -- memory helpers
    def _rd(self, addr: int, n: int) -> bytes:
        if addr < 0 or addr + n > len(self.ddr):
            raise IndexError
        return bytes(self.ddr[addr:addr + n])

    def _wr(self, addr: int, data: bytes) -> None:
        if addr < 0 or addr + len(data) > len(self.ddr):
            raise IndexError
        self.ddr[addr:addr + len(data)] = data

    # -- validation: every check happens before any write of the descriptor
    def _validate(self, raw: bytes) -> tuple[int, dict]:
        d = decode(raw)
        if d["MAGIC"] != MAGIC:
            return E_MAGIC, d
        if d["VERSION"] != VERSION:
            return E_VERSION, d
        if not crc_ok(raw):
            return E_CRC, d
        if d["VALID"] != 1:
            return E_VALID, d
        if d["OPCODE"] not in (OP_GEMM, OP_RELU):
            return E_OPCODE, d
        if d["CORE_ID"] != self.core_id or d["HDR_RSV"] or d["BODY_RSV"] or d["TAIL_RSV"] \
                or d["DEP_SIG"] or d["GROUP_SIZE"]:
            return E_FIELD, d
        # E_FIELD: every field-value rule (all of them before any alignment rule)
        c, e = d["CTRL"], d["EPI"]
        qmode, sym, adt, wdt, fus = c & 7, c >> 3 & 1, c >> 4 & 15, c >> 8 & 15, c >> 12 & 1
        act, out, bias = e & 3, e >> 2 & 3, e >> 4 & 1
        if c >> 13 or e >> 5 or act == 3 or out == 3:
            return E_FIELD, d
        if out != OUT_INT8 and d["OUT_SCALE"]:
            return E_FIELD, d
        gemm = d["OPCODE"] == OP_GEMM
        if gemm:
            if qmode not in (QM_TENSOR, QM_CHANNEL) or sym != 1 or adt != DT_INT8 or wdt != DT_INT8:
                return E_FIELD, d
            if (act != ACT_NONE) != (fus == 1):          # fusion flag iff an activation is fused
                return E_FIELD, d
        else:  # RELU: only X (A_OFF), Y, M, N, EPI; everything GEMM-only is 0
            if c != 0 or bias or d["K"] or d["W_OFF"] or d["SCALE_OFF"] or d["BIAS_OFF"] \
                    or act == ACT_NONE:
                return E_FIELD, d
        # E_ALIGN: shapes and offsets
        M, N, K = d["M"], d["N"], d["K"]
        if M == 0 or N == 0 or N % 16 or (gemm and (K == 0 or K % 16)):
            return E_ALIGN, d
        for off in ("A_OFF", "W_OFF", "Y_OFF", "SCALE_OFF", "BIAS_OFF"):
            if d[off] % 16:
                return E_ALIGN, d
        return E_OK, d

    def _ranges(self, d: dict) -> tuple[list, tuple]:
        """Byte ranges [lo, hi) the descriptor reads, and the one it writes."""
        B, M, N, K = d["BASE_ADDR"], d["M"], d["N"], d["K"]
        out = d["EPI"] >> 2 & 3
        y = (B + d["Y_OFF"], B + d["Y_OFF"] + M * N * ESIZE[out])
        if d["OPCODE"] == OP_RELU:
            return [(B + d["A_OFF"], B + d["A_OFF"] + M * N * 4)], y
        ns = 1 if (d["CTRL"] & 7) == QM_TENSOR else N
        r = [(B + d["A_OFF"], B + d["A_OFF"] + M * K), (B + d["W_OFF"], B + d["W_OFF"] + N * K),
             (B + d["SCALE_OFF"], B + d["SCALE_OFF"] + 4 * ns)]
        if d["EPI"] >> 4 & 1:
            r.append((B + d["BIAS_OFF"], B + d["BIAS_OFF"] + 4 * N))
        return r, y

    # -- execution
    def _gemm(self, d: dict) -> bytes:
        B, M, N, K = d["BASE_ADDR"], d["M"], d["N"], d["K"]
        e = d["EPI"]
        A = np.frombuffer(self._rd(B + d["A_OFF"], M * K), np.int8).reshape(M, K).astype(np.int64)
        W = np.frombuffer(self._rd(B + d["W_OFF"], N * K), np.int8).reshape(N, K).astype(np.int64)
        acc = A @ W.T                                   # exact integers; |acc| < 2^31 for K <= 65535
        assert np.abs(acc).max(initial=0) < 2**31
        ns = 1 if (d["CTRL"] & 7) == QM_TENSOR else N
        s = np.frombuffer(self._rd(B + d["SCALE_OFF"], 4 * ns), "<f4").astype(F32)
        y = fmul(i2f(acc), np.broadcast_to(s, (M, N)))  # one rounded multiply
        if e >> 4 & 1:
            b = np.frombuffer(self._rd(B + d["BIAS_OFF"], 4 * N), "<f4").astype(F32)
            y = fadd(y, np.broadcast_to(b, (M, N)))     # then one rounded add (never an FMA)
        y = act_fn(y, e & 3)
        return convert_out(y, e >> 2 & 3, d["OUT_SCALE"])

    def _relu(self, d: dict) -> bytes:
        B, M, N = d["BASE_ADDR"], d["M"], d["N"]
        x = np.frombuffer(self._rd(B + d["A_OFF"], M * N * 4), "<f4").astype(F32).reshape(M, N)
        return convert_out(act_fn(x, d["EPI"] & 3), d["EPI"] >> 2 & 3, d["OUT_SCALE"])

    def run_queue(self, queue_addr: int, queue_len: int) -> Status:
        """npu_start: descriptors at queue_addr, 64 bytes apart, executed strictly in order."""
        self.status = Status()
        for i in range(queue_len):
            raw = self._rd(queue_addr + 64 * i, 64)
            err, d = self._validate(raw)
            if err == E_OK:
                reads, (ylo, yhi) = self._ranges(d)
                if any(hi > min(len(self.ddr), 1 << 64) for _, hi in reads + [(ylo, yhi)]):
                    err = E_RANGE
                else:
                    for lo, hi in reads:
                        if lo < yhi and ylo < hi:
                            raise ContractError(f"descriptor {i}: output overlaps an input")
                    data = self._gemm(d) if d["OPCODE"] == OP_GEMM else self._relu(d)
            if err != E_OK:
                self.status = Status(done=False, err_code=err, op_idx=i)
                return self.status
            self._wr(ylo, data)
        self.status = Status(done=True, err_code=E_OK, op_idx=queue_len)
        return self.status


# ---------------------------------------------------------------- self-check and worked example
def _example() -> None:
    check_layout()
    rng = np.random.default_rng(2026)
    M, N, K = 2, 16, 32
    A = rng.integers(-128, 128, (M, K), dtype=np.int8)
    W = rng.integers(-128, 128, (N, K), dtype=np.int8)
    S = (rng.uniform(0.5, 2.0, N) / 4096).astype(F32)
    Bv = rng.uniform(-8, 8, N).astype(F32)

    ddr = bytearray(1 << 16)
    a_off, w_off, s_off, b_off, y1, y2, y3, q = 0x0100, 0x0200, 0x0600, 0x0700, 0x1000, 0x2000, 0x3000, 0x8000
    ddr[a_off:a_off + M * K] = A.tobytes()
    ddr[w_off:w_off + N * K] = W.tobytes()
    ddr[s_off:s_off + 4 * N] = S.astype("<f4").tobytes()
    ddr[b_off:b_off + 4 * N] = Bv.astype("<f4").tobytes()

    common = dict(MAGIC=MAGIC, VERSION=VERSION, OPCODE=OP_GEMM, BASE_ADDR=0, A_OFF=a_off, W_OFF=w_off,
                  SCALE_OFF=s_off, BIAS_OFF=b_off, M=M, N=N, K=K, VALID=1)
    fused = encode({**common, "CTRL": ctrl(QM_CHANNEL, fusion=1),
                    "EPI": epi(ACT_RELU, OUT_BF16, bias=1), "Y_OFF": y1})
    plain = encode({**common, "CTRL": ctrl(QM_CHANNEL, fusion=0),
                    "EPI": epi(ACT_NONE, OUT_FP32, bias=1), "Y_OFF": y2})
    relu = encode(dict(MAGIC=MAGIC, VERSION=VERSION, OPCODE=OP_RELU, CTRL=0, EPI=epi(ACT_RELU, OUT_BF16),
                       BASE_ADDR=0, A_OFF=y2, Y_OFF=y3, M=M, N=N, VALID=1))
    for i, r in enumerate((fused, plain, relu)):
        ddr[q + 64 * i:q + 64 * (i + 1)] = r

    m = Machine(ddr)
    st = m.run_queue(q, 3)
    assert st.done and st.err_code == E_OK, st
    nb = M * N * 2
    assert m.ddr[y1:y1 + nb] == m.ddr[y3:y3 + nb], "fusion equivalence violated"

    # every element by hand (scalar fp32 ops), then show the first positive one
    shown = None
    for mm in range(M):
        for nn in range(N):
            acc = int(A[mm].astype(np.int64) @ W[nn].astype(np.int64))
            p = F32(np.float32(acc) * S[nn])
            v = F32(p + Bv[nn])
            r = v if v > 0 else F32(0)
            bf = int(to_bf16_bits(np.array([r], F32))[0])
            at = y1 + 2 * (mm * N + nn)
            assert struct.unpack("<H", bytes(m.ddr[at:at + 2]))[0] == bf
            if shown is None and r > 0:
                shown = (mm, nn, acc, p, v, r, bf)
    mm, nn, acc00, p, v, r, bf = shown

    # error-before-write: a corrupted CRC stops the queue at that descriptor, Y untouched
    bad = bytearray(fused)
    bad[20] ^= 1
    ddr2 = bytearray(1 << 16)
    ddr2[:] = ddr
    ddr2[y1:y1 + nb] = b"\xEE" * nb
    ddr2[q:q + 64] = bad
    m2 = Machine(ddr2)
    st2 = m2.run_queue(q, 1)
    assert st2.err_code == E_CRC and st2.op_idx == 0 and m2.ddr[y1:y1 + nb] == b"\xEE" * nb

    print("layout: 23 fields, 512 bits, no overlap")
    print(f"example: M={M} N={N} K={K}, per-channel scale, bias, ReLU, BF16 out")
    print(f"  element (m={mm}, n={nn}); bf16 out at Y_OFF + {2 * (mm * N + nn)}")
    print(f"  acc       = {acc00}")
    print(f"  i2f*scale = {float(p)!r}  (scale[{nn}] = {float(S[nn])!r}, bits {f32_bits(float(S[nn])):08X})")
    print(f"  +bias     = {float(v)!r}  (bias[{nn}] = {float(Bv[nn])!r}, bits {f32_bits(float(Bv[nn])):08X})")
    print(f"  fp32 bits = {f32_bits(float(r)):08X}")
    print(f"  relu      = {r!r}")
    print(f"  bf16 bits = 0x{bf:04X}")
    print("fused descriptor (64 bytes, byte 0 first):")
    for i in range(0, 64, 16):
        print("  " + " ".join(f"{b:02X}" for b in fused[i:i + 16]))
    dd = decode(fused)
    print(f"  CRC32 = 0x{dd['CRC32']:08X}, CTRL = 0x{dd['CTRL']:04X}, EPI = 0x{dd['EPI']:04X}")
    print("fusion equivalence: GEMM+ReLU(BF16) == GEMM(FP32) ; RELU(BF16)  -> bit-identical")
    print("error before write: CRC error at op 0, Y untouched")


def _error_cases() -> None:
    """Each error code, its priority, and that a stopped queue keeps the earlier descriptors' writes."""
    M, N, K = 1, 16, 16
    ddr = bytearray(1 << 14)
    ddr[0x100:0x100 + K] = bytes(range(K))
    ddr[0x200:0x200 + N * K] = bytes([1]) * (N * K)
    ddr[0x400:0x404] = struct.pack("<f", 1.0)
    good = dict(MAGIC=MAGIC, VERSION=VERSION, OPCODE=OP_GEMM, CTRL=ctrl(QM_TENSOR), EPI=epi(),
                BASE_ADDR=0, A_OFF=0x100, W_OFF=0x200, Y_OFF=0x800, SCALE_OFF=0x400, M=M, N=N, K=K, VALID=1)

    def run(*descs) -> tuple[Status, Machine]:
        mem = bytearray(ddr)
        for i, r in enumerate(descs):
            mem[0x2000 + 64 * i:0x2000 + 64 * (i + 1)] = r
        mach = Machine(mem)
        return mach.run_queue(0x2000, len(descs)), mach

    cases = {
        E_MAGIC: encode({**good, "MAGIC": 0x55AA}),
        E_VERSION: encode({**good, "VERSION": 1}),
        E_CRC: encode(good, fix_crc=False),
        E_VALID: encode({**good, "VALID": 0}),
        E_OPCODE: encode({**good, "OPCODE": 0x05}),
        E_FIELD: encode({**good, "CORE_ID": 1}),
        E_ALIGN: encode({**good, "N": 8}),
        E_RANGE: encode({**good, "Y_OFF": 0x3FF0}),
    }
    for code, raw in cases.items():
        st, _ = run(raw)
        assert st.err_code == code and st.op_idx == 0 and not st.done, (code, st)
    # priority: a field error and an alignment error together report E_FIELD
    st, _ = run(encode({**good, "CORE_ID": 1, "N": 8}))
    assert st.err_code == E_FIELD
    # fusion flag without an activation is a field error
    st, _ = run(encode({**good, "CTRL": ctrl(QM_TENSOR, fusion=1)}))
    assert st.err_code == E_FIELD
    # the queue stops at op 1; op 0's output stays written
    st, mach = run(encode(good), encode({**good, "MAGIC": 0}))
    assert st.err_code == E_MAGIC and st.op_idx == 1 and mach.ddr[0x800:0x840] != bytes(64)
    # output overlapping an input is an illegal program: the golden refuses it
    try:
        run(encode({**good, "Y_OFF": 0x200}))
        raise AssertionError("overlap not refused")
    except ContractError:
        pass
    print("errors: 8 codes, E_FIELD before E_ALIGN, queue stops at the failing op, overlap refused")


if __name__ == "__main__":
    _example()
    _error_cases()
