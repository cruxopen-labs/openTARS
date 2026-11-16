"""Run Qwen3.5-0.8B's vision tower on TARS-mini (KV260): one image in, visual tokens out.

    sudo python3 sw/vit_tower.py [--data data_vit] [--out /tmp/vit_tokens.npy]

`data_vit/` is what `tools/prep_vision_kv260.py` writes on the PC: `program.bin` (one v0.4 program
whose queue holds the whole tower — patch embed, 24 block descriptors x 12, merger — with the image's
patches, position rows and rotary rows already in it), `meta.json` (queue address and length, the
output rows) and `tokens_golden.npy` (what the golden model computes, to compare against).

The host does nothing but load the image, start the core once and read the tokens back. The board's
image has no systolic array, so the GEMMs run on the extension unit in passes of four patch rows.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import tars_board as B  # first: it puts the bundled packages (pylib/) on the path

import numpy as np  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data_vit"))
    ap.add_argument("--out", default="/tmp/vit_tokens.npy")
    ap.add_argument("--compare", default=None, help="tokens to compare against (default: the export's own)")
    args = ap.parse_args()
    d = Path(args.data)
    m = json.loads((d / "meta.json").read_text())
    size, cnt, qaddr = int(m["ddr_bytes"]), int(m["descs"]), int(m["queue_addr"])
    print(f"vision tower: {m['image']} -> {int(m['height'])}x{int(m['width'])}, {int(m['patches'])} patches, "
          f"{int(m['tokens'])} visual tokens, {cnt} descriptors, {size >> 20} MiB", flush=True)

    dev = B.Fpga(mem_bytes=size)
    buf, mv = dev.mem, memoryview(dev.mem)
    with open(d / "program.bin", "rb") as f:
        pos = 0
        while pos < size:
            k = f.readinto(mv[pos:pos + (16 << 20)])
            if not k:
                raise IOError("short read of program.bin")
            pos += k
    print(f"  loaded {size >> 20} MiB into {len(buf) >> 20} MiB of hugepages", flush=True)
    dev.to_dev(0, size)

    t0 = time.time()
    done, err, op, cyc = dev.run(qaddr, cnt, timeout=900)
    dt = time.time() - t0
    print(f"  {'done' if done else 'STOPPED'} err={err} descriptors={op}/{cnt} cycles={cyc:,} "
          f"wall={dt:.2f} s ({cyc / max(dt, 1e-9) / 1e6:.1f} MHz effective)", flush=True)
    if not done or err:
        return 1

    rows, cols = int(m["out_rows"]), int(m["out_cols"])
    off, ld, base = int(m["out_off"]), int(m["out_ld"]), int(m.get("out_base", 0))
    dev.from_dev(base + off, ld * rows)
    tok = np.stack([np.frombuffer(bytes(buf[base + off + r * ld: base + off + r * ld + 4 * cols]), "<f4")
                    for r in range(rows)])
    np.save(args.out, tok)
    print(f"  tokens {tok.shape} -> {args.out}", flush=True)
    print(f"  row 0: {np.round(tok[0][:8], 4)} ...", flush=True)
    ref = Path(args.compare) if args.compare else d / "tokens_golden.npy"
    if ref.exists():
        want = np.load(ref)
        if want.shape == tok.shape:
            print(f"  vs {ref.name}: {'identical' if np.array_equal(tok, want) else 'DIFFERENT'}"
                  f" (max |diff| {float(np.abs(tok - want).max()):.3g}, scale {float(np.abs(want).max()):.3g})",
                  flush=True)
        else:
            print(f"  vs {ref.name}: shape {want.shape} != {tok.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
