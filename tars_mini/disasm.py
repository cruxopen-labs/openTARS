"""Human-readable listing of a descriptor queue (debugging aid; not part of the contract)."""
from __future__ import annotations

from . import contract as C

_OPS = {C.OP_GEMM: "GEMM", C.OP_RELU: "RELU"}
_ACT = {0: "none", 1: "relu", 2: "relu6", 3: "?"}
_OUT = {0: "fp32", 1: "bf16", 2: "int8", 3: "?"}
_QM = {0: "none", 1: "tensor", 2: "channel", 3: "group", 4: "?", 5: "?", 6: "?", 7: "?"}


def describe(raw: bytes) -> str:
    d = C.decode(raw)
    op = _OPS.get(d["OPCODE"], f"op{d['OPCODE']:#x}")
    e, c = d["EPI"], d["CTRL"]
    s = (f"{op:<4} M={d['M']} N={d['N']}" + (f" K={d['K']}" if d["K"] else "")
         + f" act={_ACT[e & 3]} out={_OUT[e >> 2 & 3]}")
    if d["OPCODE"] == C.OP_GEMM:
        s += f" scale={_QM[c & 7]}" + (" bias" if e >> 4 & 1 else "") + (" fused" if c >> 12 & 1 else "")
    s += f" base={d['BASE_ADDR']:#x} a={d['A_OFF']:#x}"
    if d["OPCODE"] == C.OP_GEMM:
        s += f" w={d['W_OFF']:#x} s={d['SCALE_OFF']:#x}" + (f" b={d['BIAS_OFF']:#x}" if e >> 4 & 1 else "")
    s += f" y={d['Y_OFF']:#x}"
    if (e >> 2 & 3) == C.OUT_INT8:
        s += f" out_scale={float(C.bits_f32(d['OUT_SCALE'])):.6g}"
    if not C.crc_ok(raw):
        s += "  [CRC MISMATCH]"
    return s


def listing(ddr: bytes, queue_addr: int, n: int) -> str:
    return "\n".join(f"{i:3d}  {describe(bytes(ddr[queue_addr + 64 * i:queue_addr + 64 * (i + 1)]))}"
                     for i in range(n))
