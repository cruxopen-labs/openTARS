"""Teacher-forced FP16-state vs current W8A8/FP32-state exports, through the real runtime.

This measures an incremental numerical change, not corpus perplexity or a quality guarantee.
Export both directories from the same checkpoint and context; no RTL/board timing is inferred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tars_board as B

TEXTS = {
    "english": (
        "An FPGA can execute a neural network by keeping its weights in external memory and "
        "streaming them through arithmetic units. The memory interface, the number of useful "
        "operations per clock, and the dependencies between operators all affect performance. "
        "Reducing the precision of a recurrent state saves storage, but it can also change "
        "future predictions because the state is updated repeatedly. A fair comparison feeds "
        "exactly the same input tokens to both versions and checks the logits at every position. "
        "Measurements should distinguish prefill from decoding and include long contexts. "
        "A smaller representation is not by itself evidence that the whole model runs faster."
        " To make an engineering decision, record the exact firmware version, clock frequency, "
        "model export, and numerical settings. Keep the original files so that a later test can "
        "reproduce the result. Measure traffic rather than estimating it from weight size alone. "
        "Long-running recurrences deserve additional checks because small rounding differences "
        "can propagate through many updates. Correctness tests should also exercise reset, "
        "prefill chunks, in-place writes, and speculative execution with rejected predictions. "
        "A reliable report separates observations from assumptions and states which conditions "
        "were not tested. Performance improvements should be compared against the same model "
        "and hardware, not against results from a different numeric format or an older design."
    ),
    "chinese": (
        "现场可编程门阵列可以通过流式读取外部存储器中的权重来执行神经网络。"
        "实际性能不仅取决于计算单元数量，也取决于访存带宽、数据布局以及算子之间的依赖。"
        "降低循环状态的存储精度能够节省容量，但状态会反复更新，舍入误差可能影响后续预测。"
        "公平的比较应当向两个版本输入完全相同的词元，并在每个位置检查输出分布。"
        "测试需要区分提示词处理和逐词生成，也需要覆盖较长的上下文。"
        "存储空间减半，并不意味着整个模型的运行时间也会减半。"
        "为了让工程结果能够复现，需要记录固件版本、时钟频率、模型导出和数值设置。"
        "原始文件应当保留，以便后续测试使用相同条件。实际流量应当直接测量，"
        "不能只根据权重大小推算。对于长期运行的循环计算，还需要观察误差是否会随更新次数增长。"
        "正确性测试也应覆盖复位、分块处理、原地写入，以及推测执行中预测失败后恢复状态的情况。"
        "可信的报告需要区分实测事实与假设，并明确说明尚未测试的条件。"
        "性能对比应当使用相同模型和硬件，不能混用不同数值格式或旧版设计的结果。"
    ),
}


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while data := f.read(1 << 20):
            digest.update(data)
    return digest.hexdigest()


def compare(baseline, candidate, tokens):
    """Same teacher-forced tokens and positions, FP64 metrics over FP32 logits."""
    baseline.new_conversation()
    candidate.new_conversation()
    rows = []
    for pos, token in enumerate(tokens):
        a, b = baseline.step(token), candidate.step(token)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise RuntimeError(f"nonfinite logits at position {pos}")
        a, b = a.astype(np.float64), b.astype(np.float64)
        diff = b - a
        ap, bp = np.exp(a - a.max()), np.exp(b - b.max())
        ap, bp = ap / ap.sum(), bp / bp.sum()
        row = dict(pos=pos, top1_equal=bool(a.argmax() == b.argmax()),
                   max_abs=float(np.abs(diff).max()),
                   relative_l2=float(np.linalg.norm(diff) / max(np.linalg.norm(a), 1e-30)),
                   kl_fp32_to_fp16=float(np.sum(ap * np.log(np.maximum(ap, 1e-300) /
                                                           np.maximum(bp, 1e-300)))))
        if pos + 1 < len(tokens):
            target = tokens[pos + 1]
            row["nll_fp32"] = float(-np.log(max(ap[target], 1e-300)))
            row["nll_fp16"] = float(-np.log(max(bp[target], 1e-300)))
        rows.append(row)
        print(f"position {pos}: top1 {'same' if row['top1_equal'] else 'changed'}, "
              f"relative L2 {row['relative_l2']:.6g}", flush=True)
    nll = [r for r in rows if "nll_fp32" in r]
    return dict(tokens=len(rows), top1_agreement=sum(r["top1_equal"] for r in rows) / len(rows),
                max_abs=max(r["max_abs"] for r in rows),
                max_relative_l2=max(r["relative_l2"] for r in rows),
                mean_kl=float(np.mean([r["kl_fp32_to_fp16"] for r in rows])),
                mean_nll_fp32=float(np.mean([r["nll_fp32"] for r in nll])) if nll else None,
                mean_nll_fp16=float(np.mean([r["nll_fp16"] for r in nll])) if nll else None,
                rows=rows)


def compare_on_device(dev, paths, sequences):
    """One physical buffer/core: finish all baseline runs before loading the candidate."""
    with tempfile.TemporaryDirectory(prefix="tars_state_precision_") as tmp:
        model = B.Qwen(dev, paths[0], log=lambda *a, **k: None)
        for name, ids in sequences.items():
            model.new_conversation()
            storage = np.lib.format.open_memmap(Path(tmp) / f"{name}.npy", mode="w+",
                                                dtype=np.float32, shape=(len(ids), model.vocab))
            for pos, token in enumerate(ids):
                values = model.step(token)
                if not np.isfinite(values).all():
                    raise RuntimeError(f"nonfinite baseline logits at position {pos}")
                storage[pos] = values
            storage.flush()
            del storage
            print(f"baseline {name}: {len(ids)} tokens captured", flush=True)
        del model
        model = B.Qwen(dev, paths[1], log=lambda *a, **k: None)
        reports = {}
        for name, ids in sequences.items():
            class Replay:
                def new_conversation(self):
                    self.pos = 0

                def step(self, token):
                    row = reference[self.pos]
                    self.pos += 1
                    return row
            reference = np.load(Path(tmp) / f"{name}.npy", mmap_mode="r")
            print(f"\n{name}: {len(ids)} teacher-forced tokens", flush=True)
            reports[name] = compare(Replay(), model, ids)
            del reference
        return reports


def main():
    from tokenizers import Tokenizer
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--fp16", required=True)
    ap.add_argument("--tokens", type=int, default=128, help="maximum tokens of each fixed prose sample")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fpga", action="store_true", help="use an already loaded FPGA instead of the CPU golden model")
    args = ap.parse_args()
    paths = [Path(args.fp32), Path(args.fp16)]
    manifests = [json.loads((d / "manifest.json").read_text()) for d in paths]
    if [m.get("state_dtype", "fp32") for m in manifests] != ["fp32", "fp16"]:
        ap.error("need FP32-state baseline and FP16-state candidate, in that order")
    for key in ("model", "hidden", "vocab", "max_tokens"):
        if manifests[0][key] != manifests[1][key]:
            ap.error(f"exports differ in {key}")
    if args.tokens <= 0 or args.tokens > manifests[0]["max_tokens"]:
        ap.error("--tokens must fit the exported context")
    if file_hash(paths[0] / "embed_bf16.bin") != file_hash(paths[1] / "embed_bf16.bin"):
        ap.error("exports have different embedding tables")
    tokenizer = Tokenizer.from_file(str(paths[0] / "tokenizer.json"))
    sequences = {name: tokenizer.encode(text, add_special_tokens=False).ids[:args.tokens]
                 for name, text in TEXTS.items()}
    reports = compare_on_device(B.Fpga() if args.fpga else B.Golden(), paths, sequences)
    report = dict(schema=1, baseline="W8A8 / FP32 state", candidate="W8A8 / FP16 state / FP32 arithmetic",
                  note="Fixed prose, not a corpus quality guarantee; exports must use the same checkpoint.",
                  backend="FPGA" if args.fpga else "CPU golden model",
                  manifests=[{k: m.get(k) for k in ("model", "state_dtype", "max_tokens", "image_bytes")}
                             for m in manifests], samples=reports)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    for name, result in reports.items():
        print(name, {k: v for k, v in result.items() if k != "rows"}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
