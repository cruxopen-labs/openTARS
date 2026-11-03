"""Minimal TARS-mini compiler (contract v0.2): DDR layout, INT8 quantization and descriptor emission.

It turns linear layers (y = act(x @ W.T + b)) into GEMM / RELU descriptors that obey the
contract: 16-byte aligned offsets, N and K padded to multiples of 16, no output overlapping an
input, per-channel scales folded with the activation scale (S[n] = s_x * s_w[n]).

    prog = Program()
    x = prog.input_int8(x_q)                              # [M, K] INT8 already quantized
    h = prog.linear(x, w1, bias=b1, act="relu", out="int8", out_scale=s_h)
    y = prog.linear(h, w2, bias=b2, out="fp32")
    status, machine = prog.run()
    logits = prog.read(machine, y)

Quantization is the host's job and is defined here, in float64 rounded to fp32 once at the end,
so the numbers the device sees are exact data, not part of the contract's arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import contract as C
from .golden import Machine, Status

ALIGN = 128         # the allocator's alignment (the contract needs 16; 128: the KV260 GEMV's wide reads)
_ACT = {"none": C.ACT_NONE, "relu": C.ACT_RELU, "relu6": C.ACT_RELU6, "gelu": C.ACT_GELU,
        "silu": C.ACT_SILU}
_OUT = {"fp32": C.OUT_FP32, "bf16": C.OUT_BF16, "int8": C.OUT_INT8}
_DT = {C.OUT_FP32: "<f4", C.OUT_BF16: "<u2", C.OUT_INT8: "i1"}


def pad16(n: int) -> int:
    return -(-n // 16) * 16


# ---------------------------------------------------------------------- host-side quantization
@dataclass
class QWeight:
    """A weight matrix [N, K] in INT8 with one fp32 scale per output channel (or one in total)."""
    q: np.ndarray            # int8 [N, K], values in [-127, 127]
    scale: np.ndarray        # float32 [N] (per-channel) or [1] (per-tensor)
    n: int                   # real (unpadded) output channels
    k: int                   # real (unpadded) input features

    @property
    def per_channel(self) -> bool:
        return self.scale.size > 1


def quantize_weight(w: np.ndarray, per_channel: bool = True) -> QWeight:
    """Symmetric INT8: s = max|w| / 127 (per output row or whole tensor), q = rint(w / s)."""
    w = np.asarray(w, dtype=np.float64)
    amax = np.abs(w).max(axis=1) if per_channel else np.array([np.abs(w).max()])
    s = np.where(amax > 0, amax / 127.0, 1.0)
    q = np.clip(np.rint(w / (s[:, None] if per_channel else s[0])), -127, 127).astype(np.int8)
    return QWeight(q=q, scale=s.astype(np.float32), n=w.shape[0], k=w.shape[1])


def act_scale(x: np.ndarray, percentile: float = 100.0) -> float:
    """Per-tensor activation scale from calibration data: amax / 127."""
    amax = float(np.percentile(np.abs(x), percentile)) if percentile < 100 else float(np.abs(x).max())
    return amax / 127.0 if amax > 0 else 1.0


def quantize_act(x: np.ndarray, s: float) -> np.ndarray:
    return np.clip(np.rint(np.asarray(x, dtype=np.float64) / s), -127, 127).astype(np.int8)


# ---------------------------------------------------------------------- program
@dataclass
class Buf:
    """A tensor in DDR: offset from the program's base, padded shape, element format, row stride.

    A Buf may be a view: a column slice of a wider tensor keeps the parent's row stride (ld).
    """
    off: int
    rows: int
    cols: int                # padded to a multiple of 16
    fmt: int                 # contract OUT_* code (FP32 / BF16 / INT8)
    scale: float | None = None   # INT8 activations: the per-tensor scale they were quantized with
    real_cols: int | None = None
    ld: int = 0              # row stride in bytes; 0 = dense (cols * element size)

    @property
    def esize(self) -> int:
        return C.ESIZE[self.fmt]

    @property
    def row_bytes(self) -> int:
        return self.cols * self.esize

    @property
    def stride(self) -> int:
        return self.ld or self.row_bytes

    def cols_view(self, c0: int, c1: int) -> "Buf":
        """Columns [c0, c1) of this tensor, without copying (c0, c1 multiples of 16)."""
        if c0 % 16 or c1 % 16 or not 0 <= c0 < c1 <= self.cols:
            raise ValueError("column views need 16-aligned bounds inside the tensor")
        real = None if self.real_cols is None else max(0, min(self.real_cols, c1) - c0)
        return Buf(self.off + c0 * self.esize, self.rows, c1 - c0, self.fmt, self.scale, real,
                   self.stride)


class Program:
    def __init__(self, ddr_bytes: int = 1 << 24, base_addr: int = 0):
        self.base = base_addr
        self.ddr = bytearray(base_addr + ddr_bytes)   # absolute addresses [0, base + size)
        self.top = ALIGN                   # keep offset 0 unused
        self.descs: list[bytes] = []
        self.queue_off: int | None = None
        self.last_output: Buf | None = None     # the most recent linear / relu result

    # -- allocation
    def alloc(self, nbytes: int) -> int:
        off = self.top
        self.top = -(-(off + nbytes) // ALIGN) * ALIGN
        if self.base + self.top > len(self.ddr):
            raise MemoryError(f"program needs {self.top} bytes of DDR, has {len(self.ddr)}")
        return off

    def _put(self, arr: np.ndarray, dtype: str) -> int:
        data = np.ascontiguousarray(arr).astype(dtype).tobytes()
        off = self.alloc(len(data))
        self.ddr[self.base + off:self.base + off + len(data)] = data
        return off

    # -- tensors
    def input_int8(self, x_q: np.ndarray, scale: float | None = None) -> Buf:
        """An INT8 activation [M, K]; K is padded with zeros to a multiple of 16."""
        M, K = x_q.shape
        xp = np.zeros((M, pad16(K)), np.int8)
        xp[:, :K] = x_q
        return Buf(self._put(xp, "i1"), M, pad16(K), C.OUT_INT8, scale, K)

    def input_fp32(self, x: np.ndarray) -> Buf:
        M, N = x.shape
        xp = np.zeros((M, pad16(N)), np.float32)
        xp[:, :N] = x
        return Buf(self._put(xp, "<f4"), M, pad16(N), C.OUT_FP32, None, N)

    def read(self, machine: Machine, b: Buf) -> np.ndarray:
        a = np.stack([np.frombuffer(machine.read(self.base + b.off + m * b.stride, b.row_bytes), _DT[b.fmt])
                      for m in range(b.rows)])
        return a[:, :b.real_cols if b.real_cols is not None else b.cols]

    @staticmethod
    def _ld16(b: Buf) -> int:
        """The descriptor's stride field for b: 0 when dense, else the stride in 16-byte units."""
        if b.stride == b.row_bytes:
            return 0
        if b.stride % 16:
            raise ValueError("row strides must be multiples of 16 bytes")
        return b.stride // 16

    # -- descriptors
    def _emit(self, **f) -> None:
        self.descs.append(C.encode({"MAGIC": C.MAGIC, "VERSION": C.VERSION, "VALID": 1,
                                    "BASE_ADDR": self.base, **f}))

    def linear(self, x: Buf, w: QWeight, bias: np.ndarray | None = None, act: str = "none",
               out: str = "fp32", out_scale: float | None = None, fused: bool = True,
               m_split: int | None = None, n_split: int | None = None,
               into: Buf | None = None) -> Buf:
        """y = act(x @ W.T * s + b) as one GEMM (fused) or GEMM(fp32) + RELU (fused=False).

        x must be INT8 with a known scale (it may be a strided view). out="int8" needs out_scale
        (the scale of y). m_split / n_split emit one descriptor per block of rows / output
        columns (n_split a multiple of 16; uses the v0.2 output stride). into= writes y into an
        existing buffer or view instead of a new one. The result must not depend on any of this.
        """
        if x.fmt != C.OUT_INT8 or x.scale is None:
            raise ValueError("GEMM input must be an INT8 activation with a scale")
        Kp = x.cols
        if w.k > Kp or pad16(w.k) != Kp:
            raise ValueError(f"weight K={w.k} does not match activation width {Kp}")
        N, Np, M = w.n, pad16(w.n), x.rows
        wq = np.zeros((Np, Kp), np.int8)
        wq[:N, :w.k] = w.q
        # folded scale S[n] = s_x * s_w[n], computed in float64 and rounded to fp32 once
        if w.per_channel:
            s = np.zeros(Np, np.float64)
            s[:N] = x.scale * w.scale.astype(np.float64)
            qm = C.QM_CHANNEL
        else:
            s = np.array([x.scale * float(w.scale[0])])
            qm = C.QM_TENSOR
        w_off = self._put(wq, "i1")
        s_off = self._put(s.astype(np.float32), "<f4")
        b_off, has_bias = 0, bias is not None
        if has_bias:
            bp = np.zeros(Np, np.float32)
            bp[:N] = bias
            b_off = self._put(bp, "<f4")

        a, o = _ACT[act], _OUT[out]
        if o == C.OUT_INT8 and not out_scale:
            raise ValueError("out='int8' needs out_scale")
        inv = C.f32_bits(float(np.float32(1.0 / out_scale))) if o == C.OUT_INT8 else 0

        direct = fused or a == C.ACT_NONE
        g_out = o if direct else C.OUT_FP32
        if direct and into is not None:
            if (into.rows, into.cols, into.fmt) != (M, Np, g_out):
                raise ValueError("into= must have the output's shape and format")
            y = into
            y.real_cols = N
        else:
            y = Buf(self.alloc(M * Np * C.ESIZE[g_out]), M, Np, g_out, None, N)
        mstep, nstep = m_split or M, n_split or Np
        if nstep % 16:
            raise ValueError("n_split must be a multiple of 16")
        for m0 in range(0, M, mstep):
            mm = min(mstep, M - m0)
            for n0 in range(0, Np, nstep):
                nn = min(nstep, Np - n0)
                self._emit(OPCODE=C.OP_GEMM,
                           CTRL=C.ctrl(qm, fusion=int(direct and a != C.ACT_NONE)),
                           EPI=C.epi(a if direct else C.ACT_NONE, g_out, int(has_bias)),
                           A_OFF=x.off + m0 * x.stride, W_OFF=w_off + n0 * Kp,
                           SCALE_OFF=s_off + (4 * n0 if w.per_channel else 0),
                           BIAS_OFF=b_off + (4 * n0 if has_bias else 0),
                           Y_OFF=y.off + m0 * y.stride + n0 * y.esize,
                           M=mm, N=nn, K=Kp, OUT_SCALE=inv if direct else 0,
                           LDA16=self._ld16(x), LDY16=self._ld16_for(y, nn))
        if direct:
            if o == C.OUT_INT8:
                y.scale = out_scale
            self.last_output = y
            return y
        return self.relu(y, act, out, out_scale, m_split=m_split, n_split=n_split, into=into)

    @staticmethod
    def _ld16_for(y: Buf, n: int) -> int:
        """Stride field for writing n columns of y: dense only if the rows are exactly n wide."""
        if y.stride == n * y.esize:
            return 0
        if y.stride % 16:
            raise ValueError("row strides must be multiples of 16 bytes")
        return y.stride // 16

    def relu(self, x: Buf, act: str = "relu", out: str = "fp32", out_scale: float | None = None,
             m_split: int | None = None, n_split: int | None = None,
             into: Buf | None = None) -> Buf:
        if x.fmt != C.OUT_FP32:
            raise ValueError("RELU input must be FP32")
        a, o = _ACT[act], _OUT[out]
        inv = C.f32_bits(float(np.float32(1.0 / out_scale))) if o == C.OUT_INT8 else 0
        if into is not None:
            if (into.rows, into.cols, into.fmt) != (x.rows, x.cols, o):
                raise ValueError("into= must have the output's shape and format")
            y = into
            y.real_cols, y.scale = x.real_cols, out_scale if o == C.OUT_INT8 else None
        else:
            y = Buf(self.alloc(x.rows * x.cols * C.ESIZE[o]), x.rows, x.cols, o,
                    out_scale if o == C.OUT_INT8 else None, x.real_cols)
        mstep, nstep = m_split or x.rows, n_split or x.cols
        if nstep % 16:
            raise ValueError("n_split must be a multiple of 16")
        for m0 in range(0, x.rows, mstep):
            mm = min(mstep, x.rows - m0)
            for n0 in range(0, x.cols, nstep):
                nn = min(nstep, x.cols - n0)
                self._emit(OPCODE=C.OP_RELU, CTRL=0, EPI=C.epi(a, o),
                           A_OFF=x.off + m0 * x.stride + 4 * n0,
                           Y_OFF=y.off + m0 * y.stride + n0 * y.esize,
                           M=mm, N=nn, OUT_SCALE=inv,
                           LDA16=self._ld16_for(x, nn), LDY16=self._ld16_for(y, nn))
        self.last_output = y
        return y

    def conv2d(self, x: Buf, h: int, w: int, weight: np.ndarray, stride: int = 1, pad: int = 0,
               bias: np.ndarray | None = None, act: str = "none", out: str = "fp32",
               out_scale: float | None = None, per_channel: bool = True,
               n_split: int | None = None) -> tuple[Buf, int, int]:
        """y = act(conv(x, weight) * s + b) as Conv (0x01) or Conv+act (0x05) descriptors.

        x: an INT8 NHWC image as a Buf of h*w pixel rows (cols = padded Cin). weight: float
        [Cout, KH, KW, Cin]; quantized here (Cin zero-padded to x's width). Returns (y, OH, OW),
        y a Buf of OH*OW pixel rows, Cout (padded) columns."""
        if x.fmt != C.OUT_INT8 or x.scale is None or x.rows != h * w:
            raise ValueError("conv input must be an INT8 image of h*w pixel rows with a scale")
        cout, kh, kw, cin = weight.shape
        cinp = x.cols
        if cin > cinp:
            raise ValueError("weight Cin exceeds the image's channels")
        wp = np.zeros((cout, kh, kw, cinp))
        wp[..., :cin] = weight
        qw = quantize_weight(wp.reshape(cout, kh * kw * cinp), per_channel)
        oh, ow = (h + 2 * pad - kh) // stride + 1, (w + 2 * pad - kw) // stride + 1
        K, N, Np = kh * kw * cinp, cout, pad16(cout)
        wq = np.zeros((Np, K), np.int8)
        wq[:N] = qw.q
        if per_channel:
            sc = np.zeros(Np, np.float64)
            sc[:N] = x.scale * qw.scale.astype(np.float64)
            qm = C.QM_CHANNEL
        else:
            sc = np.array([x.scale * float(qw.scale[0])])
            qm = C.QM_TENSOR
        w_off, s_off = self._put(wq, "i1"), self._put(sc.astype(np.float32), "<f4")
        b_off = 0
        if bias is not None:
            bp = np.zeros(Np, np.float32)
            bp[:N] = bias
            b_off = self._put(bp, "<f4")
        a, o = _ACT[act], _OUT[out]
        inv = C.f32_bits(float(np.float32(1.0 / out_scale))) if o == C.OUT_INT8 else 0
        y = Buf(self.alloc(oh * ow * Np * C.ESIZE[o]), oh * ow, Np, o,
                out_scale if o == C.OUT_INT8 else None, N)
        nstep = n_split or Np
        for n0 in range(0, Np, nstep):
            nn = min(nstep, Np - n0)
            self._emit(OPCODE=C.OP_CONV_ACT if a != C.ACT_NONE else C.OP_CONV,
                       CTRL=C.ctrl(qm, fusion=int(a != C.ACT_NONE)), EPI=C.epi(a, o, int(bias is not None)),
                       A_OFF=x.off, W_OFF=w_off + n0 * K, SCALE_OFF=s_off + (4 * n0 if per_channel else 0),
                       BIAS_OFF=b_off + (4 * n0 if bias is not None else 0), Y_OFF=y.off + n0 * y.esize,
                       M=oh * ow, N=nn, K=K, OUT_SCALE=inv, LDA16=self._ld16(x), LDY16=self._ld16_for(y, nn),
                       GROUP_SIZE=w, HDR_RSV=C.conv_geom(h, kh, kw, stride, pad))
        self.last_output = y
        return y, oh, ow

    def rowop(self, x: Buf, fn: str, gamma: np.ndarray | None = None, beta: np.ndarray | None = None,
              eps: float = 1e-5, out: str = "fp32", out_scale: float | None = None,
              m_split: int | None = None) -> Buf:
        """v0.3 row ops over whole rows: fn = "softmax", "layernorm" or "rmsnorm" (gamma required
        for the norms; beta optional). The row must be exactly its real width: padding would enter
        the sums, so the real width must be a multiple of 16."""
        if x.fmt != C.OUT_FP32:
            raise ValueError("row-op input must be FP32")
        if x.real_cols != x.cols:
            raise ValueError("row ops need rows without padding (width a multiple of 16)")
        f = {"softmax": C.ROW_SOFTMAX, "layernorm": C.ROW_LAYERNORM, "rmsnorm": C.ROW_RMSNORM}[fn]
        o = _OUT[out]
        inv = C.f32_bits(float(np.float32(1.0 / out_scale))) if o == C.OUT_INT8 else 0
        g_off = b_off = eps_bits = 0
        if f != C.ROW_SOFTMAX:
            g_off = self._put(np.asarray(gamma, np.float32), "<f4")
            if beta is not None:
                b_off = self._put(np.asarray(beta, np.float32), "<f4")
            eps_bits = C.f32_bits(eps)
        y = Buf(self.alloc(x.rows * x.cols * C.ESIZE[o]), x.rows, x.cols, o,
                out_scale if o == C.OUT_INT8 else None, x.real_cols)
        mstep = m_split or x.rows
        for m0 in range(0, x.rows, mstep):
            mm = min(mstep, x.rows - m0)
            self._emit(OPCODE=C.OP_ROW, CTRL=f, EPI=C.epi(C.ACT_NONE, o, int(beta is not None)),
                       A_OFF=x.off + m0 * x.stride, Y_OFF=y.off + m0 * y.stride,
                       SCALE_OFF=g_off, BIAS_OFF=b_off, W_OFF=eps_bits, M=mm, N=x.cols,
                       OUT_SCALE=inv, LDA16=self._ld16_for(x, x.cols), LDY16=self._ld16_for(y, x.cols))
        self.last_output = y
        return y

    # -- build and run
    def build(self) -> tuple[bytearray, int, int]:
        """Place the queue in DDR; returns (ddr image, queue address, queue length)."""
        if self.queue_off is None:
            self.queue_off = self.alloc(C.DESC_BYTES * len(self.descs))
            for i, r in enumerate(self.descs):
                a = self.base + self.queue_off + 64 * i
                self.ddr[a:a + 64] = r
        return self.ddr, self.base + self.queue_off, len(self.descs)

    def run(self) -> tuple[Status, Machine]:
        ddr, q, n = self.build()
        m = Machine(bytearray(ddr))
        return m.run_queue(q, n), m
