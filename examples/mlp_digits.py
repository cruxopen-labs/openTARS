"""P1 end to end: a trained MLP compiled to TARS-mini, run on the golden model.

Network: 64 -> 128 (ReLU) -> 10, trained with scikit-learn on its bundled handwritten-digits set
(8x8 images, 1797 samples; nothing is downloaded). Compiled as two GEMM descriptors:
  L1  INT8 x INT8, per-channel scale, bias, fused ReLU, INT8 out (re-quantized for L2)
  L2  INT8 x INT8, per-channel scale, bias, FP32 out (logits; N padded 10 -> 16)

Three references, so a wrong number has nowhere to hide:
  float     the trained float64 network (what quantization costs)
  fakequant an independent float64 re-implementation of the same INT8 program
            (same integers and scales, float64 epilogue): device vs fakequant isolates the
            fp32 rounding of the contract from the quantization itself
  golden    the TARS-mini golden model running the compiled descriptor queue

Run:  python3 examples/mlp_digits.py
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tars_mini import compiler as Cc  # noqa: E402
from tars_mini import disasm  # noqa: E402


@dataclass
class Model:
    w1: np.ndarray   # [128, 64]
    b1: np.ndarray
    w2: np.ndarray   # [10, 128]
    b2: np.ndarray
    x_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def train(seed: int = 0) -> Model:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    d = load_digits()
    x = d.data / 16.0
    xtr, xte, ytr, yte = train_test_split(x, d.target, test_size=0.2, random_state=seed,
                                          stratify=d.target)
    clf = MLPClassifier(hidden_layer_sizes=(128,), activation="relu", max_iter=600,
                        random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(xtr, ytr)
    (w1, w2), (b1, b2) = clf.coefs_, clf.intercepts_
    return Model(w1.T.copy(), b1, w2.T.copy(), b2, xtr, xte, yte)


def float_forward(m: Model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = np.maximum(x @ m.w1.T + m.b1, 0.0)
    return h, h @ m.w2.T + m.b2


@dataclass
class Quant:
    s_x: float
    s_h: float
    w1: Cc.QWeight
    w2: Cc.QWeight


def calibrate(m: Model) -> Quant:
    h, _ = float_forward(m, m.x_train)
    return Quant(Cc.act_scale(m.x_train), Cc.act_scale(h), Cc.quantize_weight(m.w1),
                 Cc.quantize_weight(m.w2))


def fakequant_forward(m: Model, q: Quant, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The same INT8 program in float64, written independently of the golden model."""
    xq = Cc.quantize_act(x, q.s_x).astype(np.float64)
    acc1 = xq @ q.w1.q.astype(np.float64).T
    h = np.maximum(acc1 * (q.s_x * q.w1.scale.astype(np.float64)) + m.b1, 0.0)
    hq = np.clip(np.rint(h / q.s_h), -127, 127)
    acc2 = hq @ q.w2.q.astype(np.float64).T
    return hq, acc2 * (q.s_h * q.w2.scale.astype(np.float64)) + m.b2


def compile_and_run(m: Model, q: Quant, x: np.ndarray, fused: bool = True,
                    m_split: int | None = None, n_split: int | None = None, base_addr: int = 0):
    p = Cc.Program(ddr_bytes=1 << 22, base_addr=base_addr)
    xin = p.input_int8(Cc.quantize_act(x, q.s_x), scale=q.s_x)
    h = p.linear(xin, q.w1, bias=m.b1, act="relu", out="int8", out_scale=q.s_h,
                 fused=fused, m_split=m_split, n_split=n_split)
    y = p.linear(h, q.w2, bias=m.b2, act="none", out="fp32", m_split=m_split, n_split=n_split)
    status, mach = p.run()
    return status, mach, p, p.read(mach, h), p.read(mach, y)


@dataclass
class Report:
    n: int
    acc_float: float
    acc_fakequant: float
    acc_golden: float
    top1_golden_vs_fakequant: float
    top1_golden_vs_float: float
    hidden_int8_mismatch: float     # fraction of hidden INT8 values that differ from fakequant
    logit_err_vs_fakequant: float   # max |golden - fakequant| / max |fakequant|
    logit_err_vs_float: float       # max |golden - float| / max |float|
    descriptors: int
    macs: int

    def exit_ok(self) -> bool:
        """P1 exit criteria (contract 验证计划 P1), all automatic."""
        return (self.hidden_int8_mismatch <= 1e-3
                and self.logit_err_vs_fakequant <= 1e-3
                and self.top1_golden_vs_fakequant >= 0.99
                and self.acc_float - self.acc_golden <= 0.01)


def evaluate(seed: int = 0, **kw) -> tuple[Report, str]:
    m = train(seed)
    q = calibrate(m)
    x, y = m.x_test, m.y_test
    _, lf = float_forward(m, x)
    hq_ref, lq = fakequant_forward(m, q, x)
    status, mach, prog, h_dev, l_dev = compile_and_run(m, q, x, **kw)
    assert status.done, status
    l_dev = l_dev.astype(np.float64)
    pd, pq, pf = l_dev.argmax(1), lq.argmax(1), lf.argmax(1)
    rep = Report(
        n=len(y),
        acc_float=float((pf == y).mean()),
        acc_fakequant=float((pq == y).mean()),
        acc_golden=float((pd == y).mean()),
        top1_golden_vs_fakequant=float((pd == pq).mean()),
        top1_golden_vs_float=float((pd == pf).mean()),
        hidden_int8_mismatch=float((h_dev.astype(np.float64) != hq_ref).mean()),
        logit_err_vs_fakequant=float(np.abs(l_dev - lq).max() / np.abs(lq).max()),
        logit_err_vs_float=float(np.abs(l_dev - lf).max() / np.abs(lf).max()),
        descriptors=len(prog.descs),
        macs=sum(s.macs for s in mach.stats),
    )
    _, qa, n = prog.build()
    return rep, disasm.listing(mach.ddr, qa, n)


def main() -> int:
    rep, listing = evaluate()
    print("compiled queue:")
    print(listing)
    print()
    print(f"test samples          {rep.n}")
    print(f"accuracy  float       {rep.acc_float:.4f}")
    print(f"accuracy  fakequant   {rep.acc_fakequant:.4f}")
    print(f"accuracy  golden      {rep.acc_golden:.4f}")
    print(f"top-1 golden=fakequant {rep.top1_golden_vs_fakequant:.4f}")
    print(f"top-1 golden=float     {rep.top1_golden_vs_float:.4f}")
    print(f"hidden INT8 mismatch  {rep.hidden_int8_mismatch:.2e}")
    print(f"logit err vs fakequant {rep.logit_err_vs_fakequant:.2e}  (fp32 rounding of the contract)")
    print(f"logit err vs float     {rep.logit_err_vs_float:.2e}  (what INT8 quantization costs)")
    print(f"descriptors {rep.descriptors}, MACs {rep.macs:,}")
    ok = rep.exit_ok()
    print("P1 exit criteria:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
