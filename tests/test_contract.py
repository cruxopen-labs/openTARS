"""The descriptor layout, encoding, CRC, and the contract document's worked example."""
import struct

import numpy as np
import pytest

from tars_mini import contract as C
from tars_mini import numerics as F
from tars_mini.golden import Machine

# the worked example in "TARS-mini 契约 v0.1" (不变量与示例)
DOC_FUSED_DESC = bytes.fromhex(
    "000000807FC4843700000000000020001000020000070000000600000010000000020000000100"
    "000000000000000000150000001A11020000000000020055AA")


def test_layout_covers_512_bits_once():
    C.check_layout()
    assert sum(C.width(n) for n in C.LAYOUT) == 512
    assert len(C.LAYOUT) == 23


@pytest.mark.parametrize("seed", range(20))
def test_encode_decode_roundtrip(seed):
    rng = np.random.default_rng(seed)
    f = {n: int(rng.integers(0, 1 << min(C.width(n), 62))) for n in C.LAYOUT if n != "CRC32"}
    raw = C.encode(f)
    d = C.decode(raw)
    assert all(d[n] == v for n, v in f.items())
    assert C.crc_ok(raw)


def test_crc_detects_every_single_bit_flip():
    raw = C.encode({"MAGIC": C.MAGIC, "VERSION": C.VERSION, "OPCODE": C.OP_GEMM, "VALID": 1})
    for byte in range(64):
        for bit in range(8):
            b = bytearray(raw)
            b[byte] ^= 1 << bit
            assert not C.crc_ok(bytes(b))


def test_byte_order_magic_in_last_bytes():
    raw = C.encode({"MAGIC": C.MAGIC})
    assert raw[62:64] == b"\x55\xAA"          # bits 511:496, little endian


def _doc_example():
    rng = np.random.default_rng(2026)
    M, N, K = 2, 16, 32
    A = rng.integers(-128, 128, (M, K), dtype=np.int8)
    W = rng.integers(-128, 128, (N, K), dtype=np.int8)
    S = (rng.uniform(0.5, 2.0, N) / 4096).astype(np.float32)
    B = rng.uniform(-8, 8, N).astype(np.float32)
    ddr = bytearray(1 << 16)
    ddr[0x100:0x100 + M * K] = A.tobytes()
    ddr[0x200:0x200 + N * K] = W.tobytes()
    ddr[0x600:0x600 + 4 * N] = S.astype("<f4").tobytes()
    ddr[0x700:0x700 + 4 * N] = B.astype("<f4").tobytes()
    f = dict(MAGIC=C.MAGIC, VERSION=C.VERSION, OPCODE=C.OP_GEMM, BASE_ADDR=0, A_OFF=0x100,
             W_OFF=0x200, SCALE_OFF=0x600, BIAS_OFF=0x700, Y_OFF=0x1000, M=M, N=N, K=K, VALID=1,
             CTRL=C.ctrl(C.QM_CHANNEL, fusion=1), EPI=C.epi(C.ACT_RELU, C.OUT_BF16, bias=1))
    return ddr, C.encode(f)


def test_worked_example_matches_the_document():
    ddr, raw = _doc_example()
    assert raw == DOC_FUSED_DESC
    d = C.decode(raw)
    assert (d["CRC32"], d["CTRL"], d["EPI"]) == (0x3784C47F, 0x111A, 0x0015)
    ddr[0x8000:0x8040] = raw
    m = Machine(ddr)
    assert m.run_queue(0x8000, 1).done
    # element (m=0, n=1) at Y_OFF + 2: acc -11968 -> -5.6617074 -> 0.58563852 -> bf16 0x3F16
    assert struct.unpack("<H", m.read(0x1002, 2))[0] == 0x3F16
    # and the step values the document lists
    A = np.frombuffer(m.read(0x100, 64), np.int8).reshape(2, 32).astype(np.int64)
    W = np.frombuffer(m.read(0x200, 512), np.int8).reshape(16, 32).astype(np.int64)
    acc = int(A[0] @ W[1])
    p = F.fmul(F.i2f(acc), np.frombuffer(m.read(0x604, 4), "<f4"))
    v = F.fadd(p, np.frombuffer(m.read(0x704, 4), "<f4"))
    assert acc == -11968
    assert C.f32_bits(float(np.frombuffer(m.read(0x604, 4), "<f4")[0])) == 0x39F80671
    assert C.f32_bits(float(np.frombuffer(m.read(0x704, 4), "<f4")[0])) == 0x40C7EA42
    assert C.f32_bits(float(v[0])) == 0x3F15EC68
