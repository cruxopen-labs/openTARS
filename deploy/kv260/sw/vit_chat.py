"""Image -> answer on TARS-mini (KV260): the ViT runs on the core, then Qwen3.5-0.8B reads its tokens.

    sudo python3 sw/vit_chat.py --question "这张图片里是什么？用一句话回答。"

The vision tower runs first, in its own process (`sw/vit_tower.py`), so its DDR image and hugepages
are released before the language model's; the visual tokens then take the place of the `<|image_pad|>`
rows in the prompt, exactly as the reference model does it (the host writes the vision tower's
1024-wide rows instead of looking the id up in the embedding table), and the text program runs the
prompt and greedy decoding as `sw/qwen_chat.py` does. `--tokens` skips the tower and reuses a
`vit_tokens.npy` from an earlier run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import tars_board as B  # first: it puts the bundled packages (pylib/) on the path

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent


def run_tower(vit: Path, tokens: str) -> np.ndarray:
    """The vision tower in a child process: its hugepage buffer must go away before the model's."""
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / "vit_tower.py"), "--data", str(vit), "--out", tokens],
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stdout.write(r.stderr[-2000:])
        raise RuntimeError(f"the vision tower failed (exit {r.returncode})")
    tok = np.load(tokens)
    print(f"  vision: {tok.shape[0]} tokens in {time.time() - t0:.1f} s", flush=True)
    return tok


def inject(m, dev, visual: dict[int, np.ndarray]) -> None:
    """The visual rows as the embeddings of their positions (before each step / prefill chunk)."""
    put = m._put

    def _put(t: int, tokens) -> None:
        put(t, tokens)
        for i in range(len(tokens)):
            j = t + i
            row = visual.get(j)
            if row is not None:
                a = m.m["x0_off"] + j * m.m["x0_ld"]
                dev.mem[a:a + 4 * m.H] = np.frombuffer(np.asarray(row, "<f4").tobytes(), np.uint8)
                dev.to_dev(a, 4 * m.H)

    m._put = _put


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE.parent / "data"))
    ap.add_argument("--vit", default=str(HERE.parent / "data_vit"))
    ap.add_argument("--question", default="这张图片里是什么？用一句话回答。")
    ap.add_argument("--tokens", default=None, help="reuse a vit_tokens.npy instead of running the tower")
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()

    from tokenizers import Tokenizer
    d = Path(args.data)
    tk = Tokenizer.from_file(str(d / "tokenizer.json"))
    vs = tk.token_to_id("<|vision_start|>")
    ve = tk.token_to_id("<|vision_end|>")
    ip = tk.token_to_id("<|image_pad|>")
    if None in (vs, ve, ip):
        raise RuntimeError("the tokenizer has no <|vision_start|> / <|image_pad|> / <|vision_end|>")

    tokfile = args.tokens or "/tmp/vit_tokens.npy"
    visual_tokens = np.load(tokfile) if args.tokens else run_tower(Path(args.vit), tokfile)

    mb = B.manifest_mem(d)
    dev = B.Golden(mem=mb) if os.environ.get("TARS_BACKEND") == "golden" else B.Fpga(mem_bytes=mb)
    m = B.Qwen(dev, d)
    m.new_conversation()
    prompt = ("<|vision_start|>" + "<|image_pad|>" * len(visual_tokens) + "<|vision_end|>"
              + "\n" + args.question)
    ids = m.encode(tk, prompt)
    i = ids.index(vs)
    assert ids[i] == vs and ids[i + 1:i + 1 + len(visual_tokens)] == [ip] * len(visual_tokens) \
        and ids[i + 1 + len(visual_tokens)] == ve, "the prompt's image placeholders"
    visual = {m.pos + i + 1 + k: row for k, row in enumerate(visual_tokens)}
    print(f"prompt {len(ids)} tokens ({len(visual_tokens)} image, {len(ids) - len(visual_tokens)} text), "
          f"context {m.T}", flush=True)
    inject(m, dev, visual)

    eos = set(m.m["eos_ids"])
    t0 = time.time()
    lg = m.prefill(ids, chunked=True)
    tp = time.time() - t0
    cyc = m.cycles
    print("tars> ", end="", flush=True)
    out, shown = [], ""
    t1 = time.time()
    while len(out) < args.max_new and m.pos < m.T:
        nxt = int(np.argmax(lg))
        if nxt in eos:
            break
        out.append(nxt)
        s = tk.decode(out)
        if not s.endswith("�"):
            print(s[len(shown):], end="", flush=True)
            shown = s
        lg = m.step(nxt)
    dt = time.time() - t1
    print(f"\n  [prefill {len(ids)} tokens {tp:.1f} s ({cyc / max(tp, 1e-9) / 1e6:.1f} M cycles/s), "
          f"{len(out)} new tokens {dt:.1f} s ({len(out) / max(dt, 1e-9):.2f} tok/s), {m.pos}/{m.T} context]",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
