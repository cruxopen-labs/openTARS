"""Chat with the exported model (Qwen3.5-0.8B, Qwen3-0.6B or Gemma 3 1B: data/manifest.json) on TARS-mini (KV260,
100 MHz), on the serial console.

    sudo python3 qwen_chat.py [--data ../data] [--max-new 128] [--raw]

Type a message and Enter; the answer streams back token by token. Commands: /reset (new
conversation), /quit. --raw sends the text as is (no chat template: plain continuation).
The host (the A53) does the tokenizer, the embedding lookup and the argmax (greedy decoding);
TARS-mini runs every layer, the final norm and the vocabulary-wide head for every token.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data"))
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--raw", action="store_true", help="no chat template")
    ap.add_argument("--per-token", action="store_true", help="the prompt a token at a time (no prefill chunks)")
    ap.add_argument("--spec", action="store_true", help="MTP speculative decoding (an export with --mtp): the same "
                    "tokens; pays off only above ~85 % draft hits, so it is off by default")
    args = ap.parse_args()
    from tokenizers import Tokenizer
    d = Path(args.data)
    tk = Tokenizer.from_file(str(d / "tokenizer.json"))
    name = json.loads((d / "manifest.json").read_text()).get("model", "the model")
    print(f"TARS-mini on KV260: loading {name} ...", flush=True)
    dev = B.Golden() if os.environ.get("TARS_BACKEND") == "golden" else B.Fpga()   # golden: a PC check
    m = B.Qwen(dev, d)
    eos = set(m.m["eos_ids"])
    print(f"ready. context {m.T} tokens. /reset = new conversation, /quit = exit\n", flush=True)
    first = True
    while True:
        try:
            line = input("you> ")
        except EOFError:
            break
        line = line.strip()
        if line in ("/quit", "/exit"):
            break
        if line == "/reset" or not line:
            m.new_conversation()
            first = True
            print("(new conversation)")
            continue
        if first:
            m.new_conversation()
        ids = m.encode(tk, line, chat=not args.raw)   # a later turn closes the model's turn first
        if m.pos + len(ids) + 1 >= m.T:
            print("(context full: new conversation)")
            m.new_conversation()
            ids = m.encode(tk, line, chat=not args.raw)
            if len(ids) + 1 >= m.T:
                print("(message too long)")
                continue
        first = False
        t0 = time.time()
        lg = m.prefill(ids, chunked=not args.per_token)   # the prompt in chunks (the weights read once per chunk)
        tp = time.time() - t0
        print("tars> ", end="", flush=True)
        out, shown, t1 = [], "", time.time()
        spec = bool(m.spec) and args.spec
        rounds = hits = 0
        if spec:                                 # the same tokens as below, up to two per round
            new, done, cut = [m.begin_spec(lg)], False, False
            while True:
                for nxt in new:
                    done = nxt in eos
                    cut = not done and len(out) >= args.max_new   # a hit's second token over the limit: dropped
                    if done or cut:
                        break
                    out.append(nxt)
                s = tk.decode(out)
                if not s.endswith("�"):
                    print(s[len(shown):], end="", flush=True)
                    shown = s
                if done or cut or len(out) >= args.max_new or not m.can_spec():
                    break
                new, _ = m.spec_round(new[-1])
                rounds, hits = rounds + 1, hits + len(new) - 1
            m.end_spec()
            print(f"\n  [draft hits {hits}/{rounds}]")
            if not (done or cut) and m.pos < m.T:  # the last token into the cache; plain steps from here
                lg = m.step(new[-1])
                spec = False
        while not spec and len(out) < args.max_new and m.pos < m.T:
            nxt = int(np.argmax(lg))
            if nxt in eos:
                break
            out.append(nxt)
            s = tk.decode(out)
            if not s.endswith("�"):          # a whole character: print what is new
                print(s[len(shown):], end="", flush=True)
                shown = s
            lg = m.step(nxt)
        dt = time.time() - t1
        print(f"\n  [{len(ids)} prompt tokens {tp:.1f} s, {len(out)} new tokens {dt:.1f} s"
              f" ({len(out) / max(dt, 1e-9):.2f} tok/s), {m.pos}/{m.T} context]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
