"""Plain greedy decoding against MTP speculative decoding on the board (an export with --mtp): the same
prompts, the tokens compared, cycles and wall time per generated token.

    sudo ./run.sh bench --data data_mtp [--new 96]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import tars_board as B  # first: it puts the bundled packages (pylib/) on the path

import numpy as np  # noqa: E402

PARA = ("The Great Wall of China is a series of fortifications built across the northern borders of ancient Chinese "
        "states to protect against nomadic groups. Construction began in the 7th century BC, and the best known "
        "sections were built by the Ming dynasty. The wall stretches over 21,000 kilometres.")
CODE = "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n\ndef mul(a, b):\n    return a * b\n"
PROMPTS = {"open_qa": "Why is the sky blue?", "story": "Write a short story about a robot who learns to paint.",
           "zh_chat": "请介绍一下长城的历史。", "advice": "Give me five tips for learning a new language.",
           "summarize": "Summarize this text in two sentences:\n" + PARA,
           "rewrite": "Rewrite this text in simpler English, keeping all the facts:\n" + PARA,
           "code_edit": "Add type hints and a docstring to every function in this code. Output only the code.\n" + CODE,
           "translate": "Translate this text into Chinese:\n" + PARA,
           "list_json": "List the planets of the solar system as a JSON array of objects with the fields name and order."}


def plain(m, ids, new, eos):
    m.new_conversation()
    lg = m.prefill(ids)
    out, cyc, t = [], 0, time.time()
    while len(out) < new and m.pos < m.T - 1:
        nxt = int(np.argmax(lg))
        if nxt in eos:
            break
        out.append(nxt)
        lg = m.step(nxt)
        cyc += m.cycles
    return out, cyc, time.time() - t


def spec(m, ids, new, eos):
    m.new_conversation()
    lg = m.prefill(ids)
    t = time.time()
    cur = m.begin_spec(lg)
    cyc, rounds, hits, pend, out, done = m.cycles, 0, 0, [cur], [], False
    while True:
        for x in pend:
            done = done or x in eos or len(out) >= new
            if not done:
                out.append(x)
        if done or len(out) >= new or not m.can_spec():
            break
        pend, _ = m.spec_round(pend[-1])
        cyc += m.cycles
        rounds, hits = rounds + 1, hits + len(pend) - 1
    m.end_spec()
    return out, cyc, time.time() - t, rounds, hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data_mtp"))
    ap.add_argument("--new", type=int, default=96)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    d = Path(a.data)
    tk = Tokenizer.from_file(str(d / "tokenizer.json"))
    dev = B.Golden() if os.environ.get("TARS_BACKEND") == "golden" else B.Fpga()
    m = B.Qwen(dev, d, log=lambda *x, **k: None)
    eos = set(m.m["eos_ids"])
    print(f"{'prompt':10s} {'tok':>4s} {'same':>5s} {'hit':>5s} {'plain ms/tok':>13s} {'mtp ms/tok':>11s} {'cycles x':>9s} "
          f"{'wall plain':>11s} {'wall mtp':>9s} {'wall x':>7s}")
    tp = ts = wp = ws = 0.0
    res = {}
    for name, text in PROMPTS.items():
        ids = tk.encode(m.m["chat_prefix"] + text + m.m["chat_suffix"], add_special_tokens=False).ids
        o1, c1, w1 = plain(m, ids, a.new, eos)
        o2, c2, w2, r, h = spec(m, ids, a.new, eos)
        n = min(len(o1), len(o2))
        same = o1[:n] == o2[:n]
        p1, p2 = c1 / len(o1) / 1e5, c2 / len(o2) / 1e5
        tp, ts, wp, ws = tp + c1 / len(o1), ts + c2 / len(o2), wp + w1 / len(o1), ws + w2 / len(o2)
        res[name] = dict(n=n, same=same, rounds=r, hits=h, c1=c1, c2=c2, w1=w1, w2=w2, n1=len(o1), n2=len(o2))
        print(f"{name:10s} {n:4d} {str(same):>5s} {h / max(r, 1):5.2f} {p1:13.1f} {p2:11.1f} {p1 / p2:9.2f} "
              f"{1e3 * w1 / len(o1):9.0f}ms {1e3 * w2 / len(o2):7.0f}ms {w1 / len(o1) / (w2 / len(o2)):7.2f}", flush=True)
    print(f"mean speedup: cycles {tp / ts:.2f}, wall {wp / ws:.2f}")
    (Path("/tmp") / "spec_bench.json").write_text(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
