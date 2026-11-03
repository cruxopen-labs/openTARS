"""Qwen3.5-0.8B end to end on TARS-mini: all 24 layers (18 LA, 6 FA), the final norm and the tied
head (vocabulary 248320), greedy decoding with the host in the loop (embedding lookup, argmax).

    python3 tools/run_qwen_full.py golden [--prompt 8] [--new 3]   # golden model; saves the run
    python3 tools/run_qwen_full.py ref                              # transformers (W8A8 and fp32) on the same tokens
    python3 tools/run_qwen_full.py rtl [--dram]                     # one decode step on the RTL vs the golden model

The weights are random with the model's shapes (seeded), or with --ckpt a Hugging Face checkpoint
directory (Qwen/Qwen3.5-0.8B; qwen_ref.load_weights) and a text prompt through its tokenizer:

    python3 tools/run_qwen_full.py golden --ckpt ~/models/Qwen3.5-0.8B --text "..." [--chat] [--new 16]
    python3 tools/run_qwen_full.py ref --ckpt ~/models/Qwen3.5-0.8B

Each stage reloads the weights, so the stages run as separate processes (the fp32 weights alone
are 3 GB). Results go to build/qwen_full/ and docs/qwen_full.json (random weights) or
build/qwen_ckpt/ and docs/qwen_ckpt.json (a checkpoint).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tars_mini import contract as C          # noqa: E402
from tars_mini import llm as L               # noqa: E402
from tars_mini import qwen_ref as Q          # noqa: E402

OUT = ROOT / "build" / "qwen_full"
REPORT = ROOT / "docs" / "qwen_full.json"
DDR = 1 << 30
OPS = {C.OP_GEMM: "GEMM", C.OP_RELU: "ACT", C.OP_ROW: "ROW", C.OP_ELT: "ELT", C.OP_ROPE: "ROPE",
       C.OP_FDOT: "FDOT"}


def config():
    return Q.make_config(Q.CFG_08B, Q.LAYERS_08B, vocab=Q.VOCAB_08B)


def weights(args, cfg) -> dict:
    return Q.load_weights(args.ckpt, cfg, embed=True) if args.ckpt else Q.random_weights(cfg, embed=True)


def tokenizer(args):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(args.ckpt)


def report_update(key: str, value) -> None:
    r = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    r[key] = value
    REPORT.write_text(json.dumps(r, indent=1) + "\n")


def lowering(cfg, w, max_tokens: int) -> L.QwenLowering:
    return L.QwenLowering(cfg, w, max_tokens=max_tokens, ddr_bytes=DDR, head=True, queue_cap=16384)


def golden(args) -> None:
    from tars_mini.golden import Machine
    t = time.time()
    cfg = config()
    w = weights(args, cfg)
    rng = np.random.default_rng(3)
    if args.ckpt:
        tok = tokenizer(args)
        text = tok.apply_chat_template([{"role": "user", "content": args.text}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) \
            if args.chat else args.text
        prompt = [int(i) for i in tok(text)["input_ids"]]
        args.prompt = len(prompt)
    else:
        prompt = [int(i) for i in rng.integers(0, Q.VOCAB_08B, args.prompt)]
    low = lowering(cfg, w, args.prompt + args.new + 1)
    low.w = {"embed_tokens.weight": w["embed_tokens.weight"]}    # the host keeps only the embedding table
    del w
    print(f"lowered: {low.p.top / 2**20:.0f} MiB of weights and state ({time.time() - t:.0f} s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    step = [0]
    layer_rows = []

    def run(ddr, q, n):
        if step[0] == 1:                       # the first decode step: keep its input image for the RTL
            (OUT / "step1_in.bin").write_bytes(ddr)
            (OUT / "step1_q.json").write_text(json.dumps({"q": q, "n": n}))
        m = Machine(ddr)
        st = m.run_queue(q, n)
        assert st.done, st
        if step[0] == 1:
            (OUT / "step1_out.bin").write_bytes(m.ddr)
        # every layer's output rows of this step (for the layer-by-layer check against the reference)
        t0, n_ = (0, args.prompt) if step[0] == 0 else (args.prompt + step[0] - 1, 1)
        layer_rows.append(np.stack([low.read_layer(m, i, t0, n_) for i in range(len(low.types))]))
        print(f"step {step[0]}: {n} descriptors, {time.time() - t:.0f} s", flush=True)
        step[0] += 1
        return m.ddr
    ids, logits, counts = low.generate(run, prompt, args.new)
    np.save(OUT / "logits_golden.npy", np.stack(logits))
    np.save(OUT / "layers_golden.npy", np.concatenate(layer_rows, axis=1))      # [layer, token, H]
    a, b = len(low.p.descs) - counts[-1], len(low.p.descs)
    ops = Counter(OPS[C.decode(low.p.descs[i])["OPCODE"]] for i in range(a, b))
    gemv = sum(1 for i in range(a, b) if C.decode(low.p.descs[i])["OPCODE"] == C.OP_GEMM
               and C.decode(low.p.descs[i])["M"] == 1)
    report_update("golden", dict(prompt=prompt, tokens=ids[len(prompt):], descriptors_per_step=counts,
                                 decode_step_ops=dict(ops), decode_step_gemv=gemv,
                                 ddr_mib=round(low.p.top / 2**20), seconds=round(time.time() - t)))
    (OUT / "ids.json").write_text(json.dumps({"prompt": prompt, "ids": ids}))
    print("tokens", ids[len(prompt):])
    if args.ckpt:
        print("text  ", repr(tok.decode(ids[len(prompt):])))
        report_update("golden_text", dict(prompt=tok.decode(prompt), output=tok.decode(ids[len(prompt):])))


def ref(args) -> None:
    t = time.time()
    cfg = config()
    w = weights(args, cfg)
    r = json.loads((OUT / "ids.json").read_text())
    prompt, ids = r["prompt"], r["ids"]
    x = np.asarray(w["embed_tokens.weight"][np.asarray(ids[:-1])], np.float32)
    chunks = [len(prompt)] + [1] * (len(ids) - len(prompt) - 1)
    g = np.load(OUT / "logits_golden.npy")
    res = {}
    for fq in (True, False):
        rr = Q.run_reference(cfg, w, x, chunks, fake_quant=fq, head=True)
        rows = []
        for i, lr in enumerate(rr.logits):
            gi = g[i]
            rows.append(dict(step=i, max_abs_diff=float(np.abs(gi - lr).max()), ref_max=float(np.abs(lr).max()),
                             cos=float(np.dot(gi, lr) / np.linalg.norm(gi) / np.linalg.norm(lr)),
                             argmax_ref=int(np.argmax(lr)), argmax_golden=int(np.argmax(gi)),
                             top5_overlap=len(set(np.argsort(-gi)[:5]) & set(np.argsort(-lr)[:5]))))
        res["w8a8" if fq else "fp32"] = rows
        print("W8A8" if fq else "fp32", rows, flush=True)
    # layer by layer: each layer alone (with an FA layer behind it, which transformers' cache needs)
    # in the reference, fed the golden model's input rows to that layer (prefill, then decode)
    gl = np.load(OUT / "layers_golden.npy")
    lw = []
    for li, ty in enumerate(cfg.layer_types):
        fa2 = next(j for j, tt in enumerate(cfg.layer_types) if tt == "full_attention" and j != li)
        c1 = Q.make_config(Q.CFG_08B, [ty, "full_attention"])
        w1 = {k.replace(f"layers.{li}.", "layers.0."): v for k, v in w.items() if k.startswith(f"layers.{li}.")}
        w1.update({k.replace(f"layers.{fa2}.", "layers.1."): v for k, v in w.items() if k.startswith(f"layers.{fa2}.")})
        w1["norm.weight"] = w["norm.weight"]
        xin = x if li == 0 else gl[li - 1]
        r1 = Q.run_reference(c1, w1, xin, chunks, fake_quant=True)
        out = np.concatenate([o[0] for o in r1.layer_out], axis=0)
        d = float(np.abs(out - gl[li]).max())
        lw.append(dict(layer=li, type=ty[:4], max_abs_diff=d, rel=d / float(np.abs(out).max())))
        print(lw[-1], flush=True)
    res["layer_by_layer"] = lw
    res["seconds"] = round(time.time() - t)
    report_update("reference", res)


def rtl(args) -> None:
    from tars_mini import rtlsim
    t = time.time()
    qn = json.loads((OUT / "step1_q.json").read_text())
    tag = "dram" if args.dram else "ideal"
    out = OUT / f"step1_rtl_{tag}.bin"
    r = rtlsim.run_core(OUT / "step1_in.bin", qn["q"], qn["n"], R=8, C=8, dram={} if args.dram else None,
                        max_cycles=400_000_000, out_path=out, log_path=OUT / f"step1_rtl_{tag}.log")
    # compare in chunks (two 1 GB images)
    same = r.done
    with open(out, "rb") as fa, open(OUT / "step1_out.bin", "rb") as fb:
        while same:
            a, b = fa.read(1 << 24), fb.read(1 << 24)
            same = a == b
            if not a:
                break
    res = dict(cycles=r.cycles, bit_exact=same, done=r.done, err=r.err_code, sim_seconds=round(time.time() - t),
               perf=r.perf)
    report_update("rtl_decode_step" + ("_dram" if args.dram else "_ideal_port"), res)
    print({k: v for k, v in res.items() if k != "perf"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["golden", "ref", "rtl"])
    ap.add_argument("--prompt", type=int, default=8)
    ap.add_argument("--new", type=int, default=3)
    ap.add_argument("--dram", action="store_true")
    ap.add_argument("--ckpt", help="a Hugging Face checkpoint directory (default: random weights)")
    ap.add_argument("--text", default="The capital of France is", help="the prompt, with --ckpt")
    ap.add_argument("--chat", action="store_true", help="wrap --text in the chat template")
    args = ap.parse_args()
    if args.ckpt:
        global OUT, REPORT
        OUT, REPORT = ROOT / "build" / "qwen_ckpt", ROOT / "docs" / "qwen_ckpt.json"
    {"golden": golden, "ref": ref, "rtl": rtl}[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
