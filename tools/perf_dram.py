"""Workloads of perf_p5.py on the LPDDR4X-4266 x32 DRAMsim3 model (sim/dram) vs the ideal port.

    python3 tools/perf_dram.py [--arrays 8,16] [--param KEY=VAL ...] [--dram KEY=VAL ...] [--json out]

Roofline under DRAM: max(MACs / R*C, (bytes read + bytes written) / 17.06 GB/s at 1 GHz): LPDDR has
one half-duplex data bus, so reads and writes share it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import perf_p5 as P  # noqa: E402
from tars_mini import rtlsim  # noqa: E402
from tars_mini.golden import Machine  # noqa: E402

DRAM_BPC = 4266e6 * 4 / 1e9      # bytes per 1 GHz cycle: 17.06


def kv(items):
    out = {}
    for it in items or []:
        k, v = it.split("=")
        out[k] = int(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", default="8,16")
    ap.add_argument("--param", nargs="*")
    ap.add_argument("--dram", nargs="*")
    ap.add_argument("--workloads", default=",".join(P.WORKLOADS))
    ap.add_argument("--json")
    a = ap.parse_args()
    params, dram = kv(a.param), kv(a.dram)
    rows = []
    for name in a.workloads.split(","):
        prog, macs, rd, wr = P.build(name, np.random.default_rng(0))
        ddr, q, n = prog.build()
        g = Machine(bytearray(ddr))
        g.run_queue(q, n)
        for s in a.arrays.split(","):
            R = int(s)
            ideal = rtlsim.run_core(bytes(ddr), q, n, R=R, C=R, lat=16, params=params)
            r = rtlsim.run_core(bytes(ddr), q, n, R=R, C=R, params=params, dram=dram)
            assert r.done and r.ddr == g.ddr and ideal.ddr == g.ddr
            roof = max(macs / (R * R), (rd + wr) / DRAM_BPC)
            p = r.perf
            rows.append(dict(workload=name, R=R, ideal=ideal.cycles, dram=r.cycles, roof=roof,
                             amp=p["dram_rd_bursts"] * p["dram_line_bytes"] / max(1, rd),
                             eff=p["rd_beats"] * 16 / max(1, p["dram_rd_bursts"] * p["dram_line_bytes"]),
                             bus=(p["dram_rd_bursts"] + p["dram_wr_bursts"]) * p["dram_line_bytes"] / DRAM_BPC,
                             lat=p["dram_rd_lat_avg"], inflight=p["max_reads_in_flight"]))
            print(f"  {name} {R}x{R}: ideal {ideal.cycles:,} dram {r.cycles:,}", file=sys.stderr)
    print("| workload | array | ideal port (lat 16) | LPDDR4X | DRAM roofline | % of roof | DRAM bus time of the actual traffic | burst use | DRAM read bytes / min | avg read lat |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for x in rows:
        print(f"| {x['workload']} | {x['R']}x{x['R']} | {x['ideal']:,} | {x['dram']:,} | {x['roof']:,.0f} | "
              f"{x['roof'] / x['dram']:.1%} | {x['bus']:,.0f} | {x['eff']:.0%} | {x['amp']:.2f}x | {x['lat']} |")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
