"""Board check, in two parts (run as root after loading the bitstream):

  1. the hardware path: small random programs (the RTL regression's generator, some with an
     injected descriptor error) on the core, every byte and the end status against the golden
     model's (data/selftest/)
  2. the model (Qwen3.5-0.8B or Qwen3-0.6B) token by token on the same prompts the export ran through the golden
     model (data/expected.json): every step's logits must match bit for bit (SHA-256)

    sudo python3 selftest.py [--data ../data] [--quick]
"""
from __future__ import annotations

import argparse
import os
import hashlib
import json
import sys
import time
from pathlib import Path

import tars_board as B  # first: it puts the bundled packages (pylib/) on the path

import numpy as np  # noqa: E402


def hw_cases(dev, d: Path) -> bool:
    cases = json.loads((d / "selftest" / "cases.json").read_text())
    ok = True
    for c in cases:
        src = (d / "selftest" / f"{c['name']}_in.bin").read_bytes()
        want = (d / "selftest" / f"{c['name']}_out.bin").read_bytes()
        n = len(src)
        dev.mem[:n] = np.frombuffer(src, np.uint8)
        dev.to_dev(0, n)
        done, err, op, cyc = dev.run(c["qaddr"], c["qlen"], timeout=10)
        dev.from_dev(0, n)
        got = dev.mem[:n].tobytes()
        same = got == want
        st = (done, err, op) == (c["done"], c["err"], c["op"])
        bad = np.nonzero(np.frombuffer(got, np.uint8) != np.frombuffer(want, np.uint8))[0]
        print(f"  {c['name']:7s} status {'ok ' if st else 'BAD'} ({done}, {B.ERRORS.get(err, err)}, op {op}) "
              f"memory {'ok' if same else f'{len(bad)} bytes differ, first at {bad[0]:#x}'}  {cyc} cycles")
        ok &= same and st
    return ok


def check(c, s, lg, cycles, what) -> bool:
    good = hashlib.sha256(lg.tobytes()).hexdigest() == s["sha256"]
    print(f"  {c['name']} {what}: -> {int(np.argmax(lg)):6d} "
          f"{'bit-exact' if good else 'MISMATCH (expected argmax %d)' % s['argmax']}  "
          f"{cycles / 1e8:.2f} s at 100 MHz")
    return good


def speculative(dev, d: Path, quick: bool) -> bool:
    """MTP speculative decoding (an export with --mtp): every verified row's logits and every token
    against the golden model's plain decoding, bit for bit."""
    exp = json.loads((d / "expected.json").read_text())
    m = B.Qwen(dev, d, log=lambda *a, **k: None)
    ok = True
    for c in exp:
        m.new_conversation()
        ids, steps = c["prompt_ids"], c["steps"]
        last = min(len(steps), len(ids) + 3) if quick else len(ids) + (len(steps) - len(ids)) // 2
        new, tok = [m.begin_spec(m.prefill(ids))], []
        rounds = hits = cyc = 0
        while (m.pos < last or not m.par) and m.pos + 2 < len(steps):   # stop on the second state copy
            tok += new
            t = m.pos
            new, rows = m.spec_round(new[-1])
            cyc += m.cycles
            rounds, hits = rounds + 1, hits + len(new) - 1
            for i in range(len(new)):
                if t + i < len(steps):
                    good = hashlib.sha256(rows[i].tobytes()).hexdigest() == steps[t + i]["sha256"]
                    ok &= good
                    if not good:
                        print(f"  {c['name']} pos {t + i}: MISMATCH")
        tok += new
        want = c["generated"][:len(tok)]
        same = tok[:len(want)] == want
        ok &= same
        swapped = m.par
        m.end_spec()                                 # then plain steps from the swapped state
        while m.pos < len(steps):
            s = steps[m.pos]
            good = hashlib.sha256(m.step(s["token"]).tobytes()).hexdigest() == s["sha256"]
            ok &= good
            if not good:
                print(f"  {c['name']} pos {s['pos']} after the swap: MISMATCH")
        print(f"  {c['name']}: state copy {swapped} at the end, plain steps after it ok")
        print(f"  {c['name']}: {'tokens match' if same else 'TOKENS DIFFER'}, {rounds} rounds, draft hits {hits}, "
              f"{cyc / max(1, rounds + hits) / 1e8:.3f} s/token at 100 MHz")
    return ok


def model(dev, d: Path, quick: bool, chunked: bool) -> bool:
    """Each prompt as the chat runs it (chunked: the prompt in prefill chunks, then a token at a
    time), or every position a token at a time; the logits against the golden model's."""
    exp = json.loads((d / "expected.json").read_text())
    m = B.Qwen(dev, d)
    ok = True
    for c in exp:
        m.new_conversation()
        steps = c["steps"][:6] if quick else c["steps"]
        np_ = len(c["prompt_ids"])
        t = time.time()
        if not chunked or np_ > len(steps):
            for s in steps:
                ok &= check(c, s, m.step(s["token"]), m.cycles, f"pos {s['pos']:3d}: token {s['token']:6d}")
        else:
            lg = m.prefill([s["token"] for s in steps[:np_]], chunked=True)
            ok &= check(c, steps[np_ - 1], lg, m.cycles, f"prompt ({np_} tokens, chunked prefill)")
            for s in steps[np_:]:
                ok &= check(c, s, m.step(s["token"]), m.cycles, f"pos {s['pos']:3d}: token {s['token']:6d}")
        print(f"  {c['name']}: {len(steps)} positions in {time.time() - t:.1f} s; expected output {c['output']!r}")
        p = getattr(dev, "last_perf", None)
        if p:                                      # the last run's bridge counters
            print("  last step: " + ", ".join(f"{k} {v:.1f}" if isinstance(v, float) else f"{k} {v}"
                                               for k, v in p.items()))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data"))
    ap.add_argument("--quick", action="store_true", help="only the first 6 steps of each model check")
    ap.add_argument("--hw-only", action="store_true")
    ap.add_argument("--per-token", action="store_true", help="every position a token at a time (no prefill chunks)")
    args = ap.parse_args()
    d = Path(args.data)
    dev = B.Golden() if os.environ.get("TARS_BACKEND") == "golden" else B.Fpga()   # golden: a PC check
    print(f"TARS-mini: {dev.params}")
    print("1. hardware path (random programs vs the golden model)")
    ok1 = hw_cases(dev, d)
    print("   PASS" if ok1 else "   FAIL")
    if args.hw_only:
        return 0 if ok1 else 1
    print(f"2. {json.loads((d / 'manifest.json').read_text()).get('model', 'model')} (logits vs the golden model, bit for bit)")
    ok2 = model(dev, d, args.quick, not args.per_token)
    print("   PASS" if ok2 else "   FAIL")
    ok3 = True
    if "spec" in json.loads((d / "manifest.json").read_text()):
        print("3. MTP speculative decoding (logits and tokens vs plain decoding, bit for bit)")
        ok3 = speculative(dev, d, args.quick)
        print("   PASS" if ok3 else "   FAIL")
    ok2 = ok2 and ok3
    print("ALL PASS" if ok1 and ok2 else "FAILED")
    return 0 if ok1 and ok2 else 1


if __name__ == "__main__":
    sys.exit(main())
