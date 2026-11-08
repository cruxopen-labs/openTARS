"""TARS-mini golden model (contract v0.4): the bit-exact reference every implementation is compared against.

Visible state is the DDR byte image plus the queue's end status (DONE, ERR_CODE, OP_IDX). There
is no array size, tiling, VMEM or DMA parameter here: results cannot depend on them.
"""
from __future__ import annotations

import bisect
import os
from dataclasses import dataclass, field

import numpy as np

from . import contract as C
from . import numerics as F


class ContractError(Exception):
    """An illegal program the compiler must not emit (UNDEFINED on hardware; the golden refuses)."""


@dataclass
class Status:
    done: bool = False
    err_code: int = C.E_OK
    op_idx: int = 0           # the descriptor that stopped the queue (CUR_OP_IDX); L when done

    def __str__(self) -> str:
        return f"done={int(self.done)} err={C.ERR_NAMES[self.err_code]} op_idx={self.op_idx}"


@dataclass
class OpStat:
    """Per-descriptor work, for performance models (not part of the contract's visible state)."""
    opcode: int
    macs: int
    bytes_read: int
    bytes_written: int


@dataclass
class Machine:
    ddr: bytearray
    core_id: int = 0
    # the KV260 build's profile (no systolic array: every GEMM on the GEMV unit): a Conv, or a
    # GEMM with K > 4096 (beyond the GEMV unit's x buffers), is rejected with E_OPCODE
    gemv_only: bool = field(default_factory=lambda: os.environ.get("TM_GEMV_ONLY") == "1")
    status: Status = field(default_factory=Status)
    stats: list[OpStat] = field(default_factory=list)

    # ------------------------------------------------------------------ memory
    def read(self, addr: int, n: int) -> bytes:
        return bytes(self.ddr[addr:addr + n])

    def write(self, addr: int, data: bytes) -> None:
        self.ddr[addr:addr + len(data)] = data

    # ------------------------------------------------------------------ checks, in contract order
    def validate(self, raw: bytes) -> tuple[int, dict]:
        d = C.decode(raw)
        if d["MAGIC"] != C.MAGIC:
            return C.E_MAGIC, d
        if d["VERSION"] != C.VERSION:
            return C.E_VERSION, d
        if not C.crc_ok(raw):
            return C.E_CRC, d
        if d["VALID"] != 1:
            return C.E_VALID, d
        op = d["OPCODE"]
        if op not in (C.OP_GEMM, C.OP_RELU, C.OP_ROW) + C.CONV_OPS + C.V04_OPS:
            return C.E_OPCODE, d
        if self.gemv_only and (op in C.CONV_OPS or (op == C.OP_GEMM and d["K"] > 4096)):
            return C.E_OPCODE, d
        # E_FIELD: every field-value rule, all before any alignment rule
        conv = op in C.CONV_OPS
        c, e = d["CTRL"], d["EPI"]
        qmode, sym, adt, wdt, fus = c & 7, c >> 3 & 1, c >> 4 & 15, c >> 8 & 15, c >> 12 & 1
        act, out, bias = C.act_of(e), e >> 2 & 3, e >> 4 & 1
        # HDR_RSV / GROUP_SIZE: conv geometry; v0.4: GEMM row-scale offset (QM_ROWCOL), ELT
        # broadcast modes and B stride, ROPE rows per position, FDOT B stride; 0 elsewhere
        hdr_ok = conv or (op == C.OP_GEMM and qmode == C.QM_ROWCOL) \
            or (op == C.OP_ELT and d["HDR_RSV"] < 16) \
            or (op == C.OP_FDOT and c & 1 and d["HDR_RSV"] == C.STATE_FP16)
        grp_ok = conv or op in C.V04_OPS
        if d["CORE_ID"] != self.core_id or d["DEP_SIG"] or (d["HDR_RSV"] and not hdr_ok) \
                or (d["GROUP_SIZE"] and not grp_ok):
            return C.E_FIELD, d
        if c >> 13 or e >> 6 or (out == C.OUT_FP16 and op != C.OP_ELT):
            return C.E_FIELD, d
        if out != C.OUT_INT8 and d["OUT_SCALE"]:
            return C.E_FIELD, d
        gemm = op == C.OP_GEMM or conv
        if gemm:
            if qmode not in (C.QM_TENSOR, C.QM_CHANNEL, C.QM_ROWCOL) or sym != 1 \
                    or adt != C.DT_INT8 or wdt != C.DT_INT8 or (conv and qmode == C.QM_ROWCOL):
                return C.E_FIELD, d
            if (act != C.ACT_NONE) != (fus == 1):
                return C.E_FIELD, d
            if d["OPCODE"] == C.OP_CONV_ACT and act == C.ACT_NONE:
                return C.E_FIELD, d
            if conv:
                g = C.conv_params(d)
                if not (g["H"] and g["W"] and g["KH"] and g["KW"]) or g["OH"] < 1 or g["OW"] < 1 \
                        or d["M"] != g["OH"] * g["OW"]:
                    return C.E_FIELD, d
        elif d["OPCODE"] == C.OP_RELU:
            if c != 0 or bias or d["K"] or d["W_OFF"] or d["SCALE_OFF"] or d["BIAS_OFF"] \
                    or act == C.ACT_NONE:
                return C.E_FIELD, d
        elif op == C.OP_ROW:                              # v0.3 / v0.4 row ops
            fn = c & 7
            if c >> 3 or fn > C.ROW_QUANT or act != C.ACT_NONE or (d["K"] and fn != C.ROW_SOFTMAX):
                return C.E_FIELD, d
            if fn == C.ROW_SOFTMAX and (bias or d["W_OFF"] or d["SCALE_OFF"] or d["BIAS_OFF"]):
                return C.E_FIELD, d
            if fn == C.ROW_QUANT and (bias or d["W_OFF"] or d["BIAS_OFF"] or out != C.OUT_INT8
                                      or d["OUT_SCALE"]):
                return C.E_FIELD, d
            if not bias and d["BIAS_OFF"]:
                return C.E_FIELD, d
        elif op == C.OP_ELT:                              # v0.4
            eop = c & 3
            if c >> 2 or d["K"] or d["SCALE_OFF"] or bias != (eop == C.ELT_MULADD) \
                    or (eop == C.ELT_MULADD and out not in (C.OUT_FP32, C.OUT_FP16)) or (not bias and d["BIAS_OFF"]):
                return C.E_FIELD, d
        elif op == C.OP_ROPE:
            if c or act != C.ACT_NONE or bias or d["SCALE_OFF"] or d["BIAS_OFF"] or d["GROUP_SIZE"] == 0:
                return C.E_FIELD, d
        else:                                             # FDOT
            if c >> 1 or act != C.ACT_NONE or bias or d["SCALE_OFF"] or d["BIAS_OFF"]:
                return C.E_FIELD, d
        # E_ALIGN: shapes and offsets
        M, N, K = d["M"], d["N"], d["K"]
        if M == 0 or N == 0 or N % 16 or ((gemm or op == C.OP_FDOT) and (K == 0 or K % 16)):
            return C.E_ALIGN, d
        if op == C.OP_ROPE and (K == 0 or K % 32 or K > N):
            return C.E_ALIGN, d
        if conv:
            g = C.conv_params(d)
            if K % (g["KH"] * g["KW"]) or g["Cin"] % 16:
                return C.E_ALIGN, d
        offs = ("A_OFF", "Y_OFF", "SCALE_OFF", "BIAS_OFF") + (() if op == C.OP_ROW else ("W_OFF",))
        for off in offs:
            # v0.4: an ELT B read as one value per row or one value in all (BM_COL / BM_SCALAR)
            # needs only 4-byte alignment
            al = 4 if (off == "W_OFF" and op == C.OP_ELT and d["HDR_RSV"] & 3 in (C.BM_COL, C.BM_SCALAR)) else 16
            if d[off] % al:
                return C.E_ALIGN, d
        # v0.2 strides: a non-zero stride must hold a whole row (v0.4: and the B stride)
        lda, ldy, arow, yrow = self.strides(d)
        if lda < arow or ldy < yrow:
            return C.E_ALIGN, d
        if op in (C.OP_ELT, C.OP_FDOT):
            ldb, brow = self.bstride(d)
            if ldb < brow:
                return C.E_ALIGN, d
        return C.E_OK, d

    @staticmethod
    def bstride(d: dict) -> tuple[int, int]:
        """v0.4 (ldb, B row bytes) of ELT (dense B) and FDOT; GROUP_SIZE = 0 means dense."""
        if d["OPCODE"] == C.OP_FDOT:
            brow = (2 if d["HDR_RSV"] & C.STATE_FP16 else 4) * (d["N"] if d["CTRL"] & 1 else d["K"])
        else:
            brow = 4 * d["N"] if (d["HDR_RSV"] & 3) == C.BM_DENSE else 0
        return (16 * d["GROUP_SIZE"] if d["GROUP_SIZE"] else brow), brow

    @staticmethod
    def strides(d: dict) -> tuple[int, int, int, int]:
        """(lda, ldy, input row bytes, output row bytes); a zero stride field means dense."""
        out = d["EPI"] >> 2 & 3
        op = d["OPCODE"]
        arow = d["K"] if op == C.OP_GEMM else C.conv_params(d)["Cin"] if op in C.CONV_OPS \
            else 4 * d["K"] if op == C.OP_FDOT else \
            (2 if op == C.OP_ELT and d["HDR_RSV"] & C.STATE_FP16 else 4) * d["N"]
        yrow = d["N"] * C.ESIZE.get(out, 4)
        lda = 16 * d["LDA16"] if d["LDA16"] else arow
        ldy = 16 * d["LDY16"] if d["LDY16"] else yrow
        return lda, ldy, arow, yrow

    @classmethod
    def segments(cls, d: dict) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]]]:
        """Byte segments [lo, hi) a valid descriptor reads (a list per tensor) and the segments
        it writes (one per output row, disjoint and ascending since ldy >= row bytes)."""
        B, M, N, K = d["BASE_ADDR"], d["M"], d["N"], d["K"]
        lda, ldy, arow, yrow = cls.strides(d)

        def strided(off: int, ld: int, nb: int) -> list[tuple[int, int]]:
            return [(B + off + m * ld, B + off + m * ld + nb) for m in range(M)]

        y = strided(d["Y_OFF"], ldy, yrow)
        op = d["OPCODE"]
        if op in C.V04_OPS:
            reads = [strided(d["A_OFF"], lda, arow) if not (op == C.OP_ELT and d["HDR_RSV"] >> 2 & 1)
                     else [(B + d["A_OFF"], B + d["A_OFF"] + arow)]]
            w0 = B + d["W_OFF"]
            if op == C.OP_ELT:
                bm = d["HDR_RSV"] & 3
                ldb = cls.bstride(d)[0]
                reads.append([(w0 + m * ldb, w0 + m * ldb + 4 * N) for m in range(M)] if bm == C.BM_DENSE
                             else [(w0, w0 + {C.BM_ROW: 4 * N, C.BM_COL: 4 * M, C.BM_SCALAR: 4}[bm])])
                if d["CTRL"] & 3 == C.ELT_MULADD:
                    reads.append(strided(d["BIAS_OFF"], ldy, yrow))
            elif op == C.OP_ROPE:
                reads.append([(w0, w0 + 4 * K * ((M - 1) // d["GROUP_SIZE"] + 1))])
            else:
                ldb, brow = cls.bstride(d)
                nb = K if d["CTRL"] & 1 else N
                reads.append([(w0 + i * ldb, w0 + i * ldb + brow) for i in range(nb)])
            return reads, y
        if op in C.CONV_OPS:                    # the image: H*W pixels of Cin bytes
            npx = C.conv_params(d)["H"] * C.conv_params(d)["W"]
            a0 = B + d["A_OFF"]
            reads = [[(a0, a0 + npx * arow)] if lda == arow else
                     [(a0 + p * lda, a0 + p * lda + arow) for p in range(npx)]]
        else:
            reads = [strided(d["A_OFF"], lda, arow)]
        if d["OPCODE"] in (C.OP_GEMM,) + C.CONV_OPS:
            ns = 1 if (d["CTRL"] & 7) == C.QM_TENSOR else N
            reads.append([(B + d["W_OFF"], B + d["W_OFF"] + N * K)])
            reads.append([(B + d["SCALE_OFF"], B + d["SCALE_OFF"] + 4 * ns)])
            if d["EPI"] >> 4 & 1:
                reads.append([(B + d["BIAS_OFF"], B + d["BIAS_OFF"] + 4 * N)])
            if (d["CTRL"] & 7) == C.QM_ROWCOL:
                r0 = B + 16 * d["HDR_RSV"]
                reads.append([(r0, r0 + 4 * M)])
        elif d["OPCODE"] == C.OP_ROW and d["CTRL"] & 7 in (C.ROW_LAYERNORM, C.ROW_RMSNORM, C.ROW_L2NORM):
            reads.append([(B + d["SCALE_OFF"], B + d["SCALE_OFF"] + 4 * N)])
            if d["EPI"] >> 4 & 1:
                reads.append([(B + d["BIAS_OFF"], B + d["BIAS_OFF"] + 4 * N)])
        return reads, y

    @staticmethod
    def extra_out(d: dict) -> list[tuple[int, int]]:
        """v0.4: the second output of ROW_QUANT, its row scales (4 M bytes at SCALE_OFF)."""
        if d["OPCODE"] == C.OP_ROW and d["CTRL"] & 7 == C.ROW_QUANT:
            a = d["BASE_ADDR"] + d["SCALE_OFF"]
            return [(a, a + 4 * d["M"])]
        return []

    @staticmethod
    def _overlaps(segs: list[tuple[int, int]], y: list[tuple[int, int]]) -> bool:
        """Whether any segment intersects any output row."""
        yhi = [hi for _, hi in y]
        for lo, hi in segs:
            i = bisect.bisect_right(yhi, lo)              # first output row ending after lo
            if i < len(y) and y[i][0] < hi:
                return True
        return False

    # ------------------------------------------------------------------ semantics
    def _convert(self, y: np.ndarray, out: int, out_scale_bits: int) -> np.ndarray:
        """[M, N] -> [M, row bytes] uint8."""
        return np.frombuffer(self._convert_bytes(y, out, out_scale_bits), np.uint8).reshape(y.shape[0], -1)

    def _convert_bytes(self, y: np.ndarray, out: int, out_scale_bits: int) -> bytes:
        if out == C.OUT_FP32:
            return F.ftz(y).astype("<f4").tobytes()
        if out == C.OUT_BF16:
            return F.to_bf16_bits(y).astype("<u2").tobytes()
        if out == C.OUT_FP16:
            return F.to_fp16_bits(y).astype("<u2").tobytes()
        return F.q8(y, C.bits_f32(out_scale_bits)).tobytes()

    def _conv(self, d: dict) -> np.ndarray:
        """Implicit GEMM: A row (oy, ox) = the KH x KW x Cin window at (oy*S - P, ox*S - P), in
        [ky][kx][cin] order; pixels outside the image are 0. Then exactly as GEMM."""
        g = C.conv_params(d)
        lda = self.strides(d)[0]
        B, cin = d["BASE_ADDR"] + d["A_OFF"], g["Cin"]
        img = np.zeros((g["H"] + 2 * g["P"], g["W"] + 2 * g["P"], cin), np.int64)
        for p in range(g["H"] * g["W"]):
            img[g["P"] + p // g["W"], g["P"] + p % g["W"]] = np.frombuffer(self.read(B + p * lda, cin), np.int8)
        A = np.stack([img[oy * g["S"]:oy * g["S"] + g["KH"], ox * g["S"]:ox * g["S"] + g["KW"]].reshape(-1)
                      for oy in range(g["OH"]) for ox in range(g["OW"])])
        return self._gemm(d, A)

    def _gemm(self, d: dict, A: np.ndarray | None = None) -> np.ndarray:
        B, M, N, K, e = d["BASE_ADDR"], d["M"], d["N"], d["K"], d["EPI"]
        lda = self.strides(d)[0]
        if A is None:
            A = np.stack([np.frombuffer(self.read(B + d["A_OFF"] + m * lda, K), np.int8)
                          for m in range(M)]).astype(np.int64)
        W = np.frombuffer(self.read(B + d["W_OFF"], N * K), np.int8).reshape(N, K).astype(np.int64)
        acc = A @ W.T                                         # 1. exact integer sum
        assert np.abs(acc).max(initial=0) < 2**31
        ns = 1 if (d["CTRL"] & 7) == C.QM_TENSOR else N
        s = np.frombuffer(self.read(B + d["SCALE_OFF"], 4 * ns), "<f4").astype(F.F32)
        y = F.i2f(acc)                                        # 2. i2f
        if (d["CTRL"] & 7) == C.QM_ROWCOL:                    # v0.4: the row (token) scale first
            sr = np.frombuffer(self.read(B + 16 * d["HDR_RSV"], 4 * M), "<f4").astype(F.F32)
            y = F.fmul(y, sr[:, None])
        y = F.fmul(y, np.broadcast_to(s, (M, N)))             # 3. one rounded multiply each
        if e >> 4 & 1:
            b = np.frombuffer(self.read(B + d["BIAS_OFF"], 4 * N), "<f4").astype(F.F32)
            y = F.fadd(y, np.broadcast_to(b, (M, N)))         # 4. one rounded add, never an FMA
        y = F.act_fn(y, C.act_of(e))                          # 5. activation
        return self._convert(y, e >> 2 & 3, d["OUT_SCALE"])   # 6. output conversion

    def _relu(self, d: dict) -> np.ndarray:
        B, M, N = d["BASE_ADDR"], d["M"], d["N"]
        lda = self.strides(d)[0]
        x = np.stack([np.frombuffer(self.read(B + d["A_OFF"] + m * lda, 4 * N), "<f4")
                      for m in range(M)]).astype(F.F32)
        return self._convert(F.act_fn(x, C.act_of(d["EPI"])), d["EPI"] >> 2 & 3, d["OUT_SCALE"])

    def _row(self, d: dict) -> np.ndarray:
        """v0.3 row ops on fp32 rows: Softmax, LayerNorm, RMSNorm (numerics.py gives each order)."""
        B, M, N = d["BASE_ADDR"], d["M"], d["N"]
        lda = self.strides(d)[0]
        x = np.stack([np.frombuffer(self.read(B + d["A_OFF"] + m * lda, 4 * N), "<f4")
                      for m in range(M)]).astype(F.F32)
        fn = d["CTRL"] & 7
        if fn == C.ROW_SOFTMAX and d["K"]:                   # v0.4 causal: row m sees K + m
            y = F.softmax_rows_causal(x, np.minimum(N, d["K"] + np.arange(M)))
        elif fn == C.ROW_SOFTMAX:
            y = F.softmax_rows(x)
        elif fn == C.ROW_QUANT:                             # v0.4: int8 rows + row scales
            q, sr = F.quant_rows(x)
            self._extra = (B + d["SCALE_OFF"], F.ftz(sr).astype("<f4").tobytes())
            return q.view(np.uint8)
        else:
            g = np.frombuffer(self.read(B + d["SCALE_OFF"], 4 * N), "<f4").astype(F.F32)
            b = np.frombuffer(self.read(B + d["BIAS_OFF"], 4 * N), "<f4").astype(F.F32) \
                if d["EPI"] >> 4 & 1 else None
            eps = C.bits_f32(d["W_OFF"])
            y = {C.ROW_LAYERNORM: F.layernorm_rows, C.ROW_RMSNORM: F.rmsnorm_rows,
                 C.ROW_L2NORM: F.l2norm_rows}[fn](x, g, b, eps)
        return self._convert(y, d["EPI"] >> 2 & 3, d["OUT_SCALE"])

    # ------------------------------------------------------------------ v0.4 ops
    def _rows_f32(self, addr: int, ld: int, m: int, n: int) -> np.ndarray:
        return np.stack([np.frombuffer(self.read(addr + i * ld, 4 * n), "<f4")
                         for i in range(m)]).astype(F.F32)

    def _rows_float(self, addr: int, ld: int, m: int, n: int, half: bool) -> np.ndarray:
        if not half:
            return self._rows_f32(addr, ld, m, n)
        return np.stack([F.from_fp16_bits(np.frombuffer(self.read(addr + i * ld, 2 * n), "<u2"))
                         for i in range(m)])

    def _elt(self, d: dict) -> np.ndarray:
        """y = act(A) op B (MUL, ADD, SUB = A + -B) or C + act(A) * B (MULADD), each rounded once;
        A and B broadcast per HDR_RSV."""
        B0, M, N, e = d["BASE_ADDR"], d["M"], d["N"], d["EPI"]
        lda, ldy = self.strides(d)[:2]
        a = self._rows_float(B0 + d["A_OFF"], 0 if d["HDR_RSV"] >> 2 & 1 else lda, M, N,
                             bool(d["HDR_RSV"] & C.STATE_FP16))
        bm = d["HDR_RSV"] & 3
        w0 = B0 + d["W_OFF"]
        if bm == C.BM_DENSE:
            b = self._rows_f32(w0, self.bstride(d)[0], M, N)
        else:
            v = np.frombuffer(self.read(w0, 4 * {C.BM_ROW: N, C.BM_COL: M, C.BM_SCALAR: 1}[bm]), "<f4").astype(F.F32)
            b = np.broadcast_to(v[None, :] if bm == C.BM_ROW else v[:, None] if bm == C.BM_COL else v[None, :1], (M, N))
        a = F.act_fn(a, C.act_of(e))
        eop = d["CTRL"] & 3
        if eop == C.ELT_MUL:
            y = F.fmul(a, b)
        elif eop == C.ELT_ADD:
            y = F.fadd(a, b)
        elif eop == C.ELT_SUB:
            y = F.fadd(a, F._neg(F.ftz(b)))
        else:
            c = self._rows_float(B0 + d["BIAS_OFF"], ldy, M, N, e >> 2 & 3 == C.OUT_FP16)
            y = F.fadd(c, F.fmul(a, b))
        return self._convert(y, e >> 2 & 3, d["OUT_SCALE"])

    def _rope(self, d: dict) -> np.ndarray:
        B0, M, N, R, G = d["BASE_ADDR"], d["M"], d["N"], d["K"], d["GROUP_SIZE"]
        x = self._rows_f32(B0 + d["A_OFF"], self.strides(d)[0], M, N)
        tab = self._rows_f32(B0 + d["W_OFF"], 4 * R, (M - 1) // G + 1, R)
        return self._convert(F.rope_rows(x, tab, G, R), d["EPI"] >> 2 & 3, d["OUT_SCALE"])

    def _fdot(self, d: dict) -> np.ndarray:
        B0, M, N, K = d["BASE_ADDR"], d["M"], d["N"], d["K"]
        a = self._rows_f32(B0 + d["A_OFF"], self.strides(d)[0], M, K)
        ldb = self.bstride(d)[0]
        if d["CTRL"] & 1:
            y = F.fdot_nn(a, self._rows_float(B0 + d["W_OFF"], ldb, K, N,
                                            bool(d["HDR_RSV"] & C.STATE_FP16)))
        else:
            y = F.fdot_nt(a, self._rows_f32(B0 + d["W_OFF"], ldb, N, K))
        return self._convert(y, d["EPI"] >> 2 & 3, d["OUT_SCALE"])

    @staticmethod
    def _in_place(d: dict, i: int) -> bool:
        """v0.4: an ELT input may be the output itself (same rows, fp32): read i is A (0) or C (2)."""
        out = d["EPI"] >> 2 & 3
        if d["OPCODE"] != C.OP_ELT or out not in (C.OUT_FP32, C.OUT_FP16):
            return False
        lda, ldy = Machine.strides(d)[:2]
        if i == 0:
            afmt = C.OUT_FP16 if d["HDR_RSV"] & C.STATE_FP16 else C.OUT_FP32
            return afmt == out and not (d["HDR_RSV"] >> 2 & 1) and d["A_OFF"] == d["Y_OFF"] and lda == ldy
        return i == 2 and d["BIAS_OFF"] == d["Y_OFF"]

    # ------------------------------------------------------------------ the queue
    def run_queue(self, queue_addr: int, queue_len: int) -> Status:
        """npu_start: descriptors at queue_addr, 64 bytes apart, executed strictly in order.

        Every check of a descriptor happens before it writes anything; an error stops the queue
        and keeps the earlier descriptors' writes.
        """
        self.status, self.stats = Status(), []
        for i in range(queue_len):
            qa = queue_addr + C.DESC_BYTES * i
            if qa + C.DESC_BYTES > len(self.ddr):
                self.status = Status(False, C.E_RANGE, i)
                return self.status
            raw = self.read(qa, C.DESC_BYTES)
            err, d = self.validate(raw)
            if err == C.E_OK:
                reads, y = self.segments(d)
                extra = self.extra_out(d)
                limit = min(len(self.ddr), 1 << 64)
                if any(hi > limit for seg in reads + [y, extra] for _, hi in seg):
                    err = C.E_RANGE
            if err != C.E_OK:
                self.status = Status(False, err, i)
                return self.status
            for j, seg in enumerate(reads):
                if self._overlaps(seg, y) and not self._in_place(d, j):
                    raise ContractError(f"descriptor {i}: output overlaps an input")
                if extra and self._overlaps(seg, extra):
                    raise ContractError(f"descriptor {i}: output overlaps an input")
            if extra and self._overlaps(extra, y):
                raise ContractError(f"descriptor {i}: the two outputs overlap")
            self._extra = None
            rows = {C.OP_GEMM: self._gemm, C.OP_RELU: self._relu, C.OP_ROW: self._row,
                    C.OP_CONV: self._conv, C.OP_CONV_ACT: self._conv, C.OP_ELT: self._elt,
                    C.OP_ROPE: self._rope, C.OP_FDOT: self._fdot}[d["OPCODE"]](d)
            for m in range(rows.shape[0]):
                self.write(y[m][0], rows[m].tobytes())
            if self._extra:
                self.write(*self._extra)
            macs = d["M"] * d["N"] * d["K"] if d["OPCODE"] in (C.OP_GEMM, C.OP_FDOT) + C.CONV_OPS else 0
            nread = sum(hi - lo for seg in reads for lo, hi in seg)
            self.stats.append(OpStat(d["OPCODE"], macs, nread, rows.size))
        self.status = Status(True, C.E_OK, queue_len)
        return self.status
