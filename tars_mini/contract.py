"""TARS-mini contract v0.4: the 512-bit descriptor layout, field codes and encode/decode.

Field positions follow the TARS V1.3.1 descriptor family (VERSION 0x02). The contract document
("TARS-mini 契约 v0.2") is the source of truth; this module is its executable form.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

# (hi, lo) bit positions; together they cover bits 511..0 exactly once (check_layout)
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
    "LDA16":      (111, 96),   # v0.2: input row stride in 16-byte units; 0 = dense
    "OUT_SCALE":  (95, 64),
    "CRC32":      (63, 32),
    "VALID":      (31, 31),
    "LDY16":      (30, 16),    # v0.2: output row stride in 16-byte units; 0 = dense
    "DEP_SIG":    (15, 0),     # V1.3.1 multi-core DEP_MASK/DEP_SLOT/SIGNAL_MASK/SIGNAL_SLOT; 0 in mini
}

DESC_BYTES = 64
MAGIC, VERSION = 0xAA55, 0x0002
OP_GEMM, OP_RELU, OP_ROW = 0x02, 0x03, 0x04
# v0.3 OPCODE 0x04 (row ops, V1.1 "Softmax/LayerNorm"): CTRL[1:0] selects the function; for
# LayerNorm / RMSNorm, SCALE_OFF = gamma, BIAS_OFF = beta (EPI.bias), and the W_OFF field holds
# eps as fp32 bits
ROW_SOFTMAX, ROW_LAYERNORM, ROW_RMSNORM = 0, 1, 2
# v0.3 OPCODE 0x01 Conv / 0x05 Conv+activation (V1.1): an implicit GEMM over an NHWC INT8 image.
# M = OH*OW output pixels, N = Cout, K = KH*KW*Cin ([n][ky][kx][cin] weights), LDA16 = pixel
# stride. Geometry (pending ratification against V1.3.1): GROUP_SIZE = W (input width), HDR_RSV =
# H[23:12] | KH[11:8] | KW[7:4] | (STRIDE-1)[3:2] | PAD[1:0].
OP_CONV, OP_CONV_ACT = 0x01, 0x05
CONV_OPS = (OP_CONV, OP_CONV_ACT)

# ---------------------------------------------------------------------- v0.4 (LLM blocks)
# New OPCODEs beyond V1.1's 0x01-0x05 (pending ratification against the main spec):
#   0x06 ELT  - elementwise y = A op B (op in CTRL[1:0]; MULADD adds C at BIAS_OFF), A and B may
#               broadcast (HDR_RSV), B at W_OFF with row stride GROUP_SIZE * 16 (0 = dense)
#   0x07 ROPE - rotary embedding of the first R = K elements of each row; cos / sin table rows at
#               W_OFF, table row = m // G (G = GROUP_SIZE rows per position)
#   0x08 FDOT - fp32 dot products on the row engine: y[m][n] = sum_k A[m][k] * B(k, n), B at W_OFF
#               laid out [N][K] (CTRL[0] = 0, NT) or [K][N] (CTRL[0] = 1, NN), row stride
#               GROUP_SIZE * 16 (0 = dense)
OP_ELT, OP_ROPE, OP_FDOT = 0x06, 0x07, 0x08
V04_OPS = (OP_ELT, OP_ROPE, OP_FDOT)
ELT_MUL, ELT_ADD, ELT_MULADD, ELT_SUB = 0, 1, 2, 3
# ELT broadcast modes: HDR_RSV[1:0] for B, HDR_RSV[2] for A
BM_DENSE, BM_ROW, BM_COL, BM_SCALAR = 0, 1, 2, 3   # B(m,n) = B[m][n], B[n], B[m], B[0]
AM_DENSE, AM_ROW = 0, 1                            # A(m,n) = A[m][n], A[n]
FDOT_NT, FDOT_NN = 0, 1
# Row-op functions (OPCODE 0x04): CTRL[2:0] (v0.4 extends v0.3's CTRL[1:0]; CTRL[2] was 0)
ROW_L2NORM, ROW_QUANT = 3, 4


def conv_geom(h: int, kh: int, kw: int, stride: int = 1, pad: int = 0) -> int:
    assert 0 < h < 4096 and 0 < kh < 16 and 0 < kw < 16 and 1 <= stride <= 4 and 0 <= pad <= 3
    return h << 12 | kh << 8 | kw << 4 | (stride - 1) << 2 | pad


def conv_params(d: dict) -> dict:
    """H, W, KH, KW, S, P from a conv descriptor; Cin, OH, OW derived (OH/OW may be <= 0)."""
    g = d["HDR_RSV"]
    h, kh, kw, s, p, w = g >> 12, g >> 8 & 15, g >> 4 & 15, (g >> 2 & 3) + 1, g & 3, d["GROUP_SIZE"]
    cin = d["K"] // (kh * kw) if kh and kw else 0
    return dict(H=h, W=w, KH=kh, KW=kw, S=s, P=p, Cin=cin,
                OH=(h + 2 * p - kh) // s + 1, OW=(w + 2 * p - kw) // s + 1)

# CTRL (431:416), quant field positions per V1.3.1 section 10.7
QM_NONE, QM_TENSOR, QM_CHANNEL, QM_ROWCOL = 0, 1, 2, 3
# v0.4 QM_ROWCOL (V1.3.1's code 3, "group", unused so far): per-channel weight scales at
# SCALE_OFF and per-row (per-token) activation scales at BASE_ADDR + 16 * HDR_RSV
QM_GROUP = QM_ROWCOL
DT_BF16, DT_INT8, DT_INT4, DT_FP8 = 0, 1, 2, 3

# EPI (399:384). The activation code is EPI[1:0] | EPI[5] << 2 (v0.3: EPI[5] was a must-be-0
# reserved bit, so every v0.2 descriptor keeps its meaning)
ACT_NONE, ACT_RELU, ACT_RELU6, ACT_GELU, ACT_SILU = 0, 1, 2, 3, 4
ACT_SIGMOID, ACT_SOFTPLUS, ACT_EXP = 5, 6, 7        # v0.4: the 3-bit code is now fully used
OUT_FP32, OUT_BF16, OUT_INT8 = 0, 1, 2
ESIZE = {OUT_FP32: 4, OUT_BF16: 2, OUT_INT8: 1}

# ERR_CODE (CSR 0x803), in check order: the lowest failing check wins
E_OK, E_MAGIC, E_VERSION, E_CRC, E_VALID, E_OPCODE, E_FIELD, E_ALIGN, E_RANGE = range(9)
ERR_NAMES = {E_OK: "OK", E_MAGIC: "E_MAGIC", E_VERSION: "E_VERSION", E_CRC: "E_CRC",
             E_VALID: "E_VALID", E_OPCODE: "E_OPCODE", E_FIELD: "E_FIELD", E_ALIGN: "E_ALIGN",
             E_RANGE: "E_RANGE"}

K_MAX = 65535       # |acc| <= 128 * 128 * 65535 < 2^31
CANON_NAN = 0x7FC00000   # v0.2: every NaN a mul or add produces


def check_layout() -> None:
    """Every field lies in [0, 511]; together the fields cover all 512 bits exactly once."""
    used = np.zeros(512, dtype=int)
    for hi, lo in LAYOUT.values():
        assert 511 >= hi >= lo >= 0
        used[lo:hi + 1] += 1
    assert (used == 1).all(), "fields overlap or leave holes"


def width(name: str) -> int:
    hi, lo = LAYOUT[name]
    return hi - lo + 1


def ctrl(qmode: int = QM_TENSOR, symmetric: int = 1, a_dtype: int = DT_INT8,
         w_dtype: int = DT_INT8, fusion: int = 0) -> int:
    return ((qmode & 7) | (symmetric & 1) << 3 | (a_dtype & 15) << 4 | (w_dtype & 15) << 8
            | (fusion & 1) << 12)


def epi(act: int = ACT_NONE, out: int = OUT_FP32, bias: int = 0) -> int:
    return (act & 3) | (out & 3) << 2 | (bias & 1) << 4 | (act >> 2 & 1) << 5


def act_of(e: int) -> int:
    """The activation code of an EPI field."""
    return (e & 3) | (e >> 5 & 1) << 2


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def bits_f32(b: int) -> np.float32:
    return np.frombuffer(struct.pack("<I", b), dtype="<f4")[0]


def encode(fields: dict, *, fix_crc: bool = True) -> bytes:
    """A descriptor as 64 bytes, little endian: byte i holds bits [8i+7 : 8i].

    Fields not given are 0. With fix_crc the CRC32 field is computed over all 64 bytes with the
    CRC field (bytes 4..7) zeroed (CRC-32/ISO-HDLC, as zlib.crc32).
    """
    v = 0
    for name, val in fields.items():
        hi, lo = LAYOUT[name]
        if not 0 <= val < (1 << (hi - lo + 1)):
            raise ValueError(f"{name}={val} does not fit {hi - lo + 1} bits")
        v |= val << lo
    raw = bytearray(v.to_bytes(DESC_BYTES, "little"))
    if fix_crc:
        raw[4:8] = b"\0\0\0\0"
        raw[4:8] = struct.pack("<I", zlib.crc32(bytes(raw)) & 0xFFFFFFFF)
    return bytes(raw)


def decode(raw: bytes) -> dict:
    v = int.from_bytes(raw, "little")
    return {n: (v >> lo) & ((1 << (hi - lo + 1)) - 1) for n, (hi, lo) in LAYOUT.items()}


def crc_ok(raw: bytes) -> bool:
    z = bytearray(raw)
    z[4:8] = b"\0\0\0\0"
    return struct.unpack("<I", raw[4:8])[0] == (zlib.crc32(bytes(z)) & 0xFFFFFFFF)
