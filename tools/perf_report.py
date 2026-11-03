"""P3: where the RTL's cycles go, against the roofline, and what the two candidate changes buy.

    python3 tools/perf_report.py [--json out.json] [--quick] [--baseline docs/p3_perf.json]

For each workload x array size x read latency it runs one GEMM descriptor on the RTL (no
backpressure) and reports:
  - cycles, and the roofline bound max(MACs / (R*C), minimum bytes / 16 B per cycle)
  - the cycle breakdown by phase (the core's counters; they sum to the total)
  - DDR traffic actually moved vs the minimum (A, W, S, B, Y once each)
  - projections (estimates from the counters, not measurements) for:
      A   many reads in flight: each read beat costs one cycle of the 16-byte port, latency hidden
      A2  A + the next K-chunk loads while the current one feeds the array
      B   parallel epilogue: one output row (C results) per cycle, scale/bias read once per tile
      A3  A2 + continuous feed: chunks stream back to back, the skew fills once per tile
      AB  both A3 and B
and two floors no schedule change can beat: the traffic floor (bytes this tiling moves / 16) and
the schedule floor (tiles x K: an R x C output tile needs K feed cycles, full or not).
The machine model: 16 bytes per cycle on the memory port (one 128-bit beat), R*C int8 MACs per
cycle. The projections keep everything else as measured.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tars_mini import compiler as Cc  # noqa: E402
from tars_mini import contract as C  # noqa: E402
from tars_mini import rtlsim  # noqa: E402

BEAT = 16

WORKLOADS = {
    # name: (M, N, K, out, act, what it stands for)
    "mlp_l1":   (360, 128, 64, "int8", "relu", "P1 MLP layer 1: many rows, short K"),
    "mlp_l2":   (360, 16, 128, "fp32", "none", "P1 MLP layer 2: narrow output"),
    "decode":   (1, 1024, 1024, "bf16", "none", "LLM decode GEMV: one token, weight-bound"),
    "prefill":  (64, 256, 256, "bf16", "relu", "prefill chunk: 64 tokens"),
    "square":   (128, 128, 1024, "int8", "relu", "compute-heavy: long K"),
}


@dataclass
class Row:
    workload: str
    R: int
    C: int
    lat: int
    M: int
    N: int
    K: int
    macs: int
    cycles: int
    roof: float
    roof_kind: str
    min_bytes: int
    moved_bytes: int
    phases: dict
    rd_by_phase: dict
    tiles: int
    proj: dict
    traffic_floor: int = 0
    schedule_floor: int = 0
    flight: int = 0


def build(M, N, K, out, act, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (M, K))
    s_x = Cc.act_scale(x)
    p = Cc.Program(ddr_bytes=1 << 22)
    xin = p.input_int8(Cc.quantize_act(x, s_x), s_x)
    p.linear(xin, Cc.quantize_weight(rng.normal(0, .2, (N, K))), bias=rng.normal(0, .5, N),
             act=act, out=out, out_scale=0.05 if out == "int8" else None)
    return p


def min_bytes(M, N, K, out) -> int:
    es = C.ESIZE[{"fp32": C.OUT_FP32, "bf16": C.OUT_BF16, "int8": C.OUT_INT8}[out]]
    return M * K + N * K + 4 * N + 4 * N + M * N * es + 64


def project(r: rtlsim.RtlResult, R: int, Cc_: int, lat: int, tiles: int, M: int) -> dict:
    """Estimated cycles for the candidate changes, from the measured counters."""
    p, T = r.perf, r.cycles
    feed_len = 16 + max(R, Cc_) - 1
    load = p["g_load"] + p["g_load_mem"]
    load_beats = p["rd_g_load_mem"]
    epi_mem = p["g_epi_mem"]
    epi_beats = p["rd_g_epi_mem"]
    # A: a streaming reader, one beat per cycle, latency overlapped (the issue/receive states go too)
    A = T - load + load_beats - epi_mem + epi_beats
    # A2: the next chunk loads while the current one feeds; per chunk max(load, feed)
    overlap = min(load_beats, p["g_feed"])
    A2 = A - overlap
    # B: one output row per cycle; scale and bias vectors once per tile (serial reads as today)
    rows = tiles * min(R, M)
    vec_beats = tiles * math.ceil(4 * Cc_ / BEAT) * 2
    per_read = (p["g_epi_mem"] / epi_beats) if epi_beats else (lat + 3)
    B = T - p["g_epi"] - epi_mem + rows + vec_beats * per_read
    # A3: chunks back to back (16 feed cycles each, or the chunk's beats if more), skew once per tile
    K = p.get("_K", 0)
    chunks = tiles * (K // 16)
    bpc = load_beats / chunks if chunks else 0
    A3 = A - load_beats - p["g_feed"] + chunks * max(bpc, 16) + tiles * (max(R, Cc_) - 1)
    # AB: A3 and B, with the vector reads streamed too
    AB = A3 - p["g_epi"] - epi_beats + rows + vec_beats
    return {k: int(v) for k, v in dict(A=A, A2=A2, A3=A3, B=B, AB=AB).items()}


def run_one(name: str, R: int, Cc_: int, lat: int) -> Row:
    M, N, K, out, act, _ = WORKLOADS[name]
    prog = build(M, N, K, out, act)
    ddr, qa, n = prog.build()
    r = rtlsim.run_core(bytes(ddr), qa, n, R=R, C=Cc_, stall=0, lat=lat)
    assert r.done, r
    macs = M * N * K
    mb = min_bytes(M, N, K, out)
    t_comp, t_mem = macs / (R * Cc_), mb / BEAT
    tiles = math.ceil(M / R) * math.ceil(N / Cc_)
    r.perf["_K"] = K
    phases = {k: r.perf[k] for k in rtlsim.PHASES if r.perf[k]}
    rd = {k[3:]: v for k, v in r.perf.items() if k.startswith("rd_") and k[3:] in rtlsim.PHASES and v}
    return Row(name, R, Cc_, lat, M, N, K, macs, r.cycles, max(t_comp, t_mem),
               "compute" if t_comp >= t_mem else "memory", mb,
               (r.perf["rd_beats"] + r.perf["wr_beats"]) * BEAT, phases, rd, tiles,
               project(r, R, Cc_, lat, tiles, M),
               (r.perf["rd_beats"] + r.perf["wr_beats"]), tiles * K,
               r.perf.get("max_reads_in_flight", 0))


GROUPS = {"load": ("g_load", "g_load_mem"), "feed": ("g_feed",), "drain": ("g_drain",),
          "epilogue": ("g_epi", "g_epi_mem"), "write": ("write",), "other": ("fetch", "check", "other")}


def table(rows: list[Row]) -> str:
    out = ["| workload | array | lat | cycles | roofline | % of roof | load | feed | epilogue | other "
           "| traffic / min | traffic floor | schedule floor | A | A2 | A3 | B | AB |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        g = {k: sum(r.phases.get(x, 0) for x in v) / r.cycles for k, v in GROUPS.items()}
        other = g["drain"] + g["write"] + g["other"]
        sp = {k: r.cycles / v for k, v in r.proj.items()}
        out.append(f"| {r.workload} | {r.R}x{r.C} | {r.lat} | {r.cycles:,} | {r.roof:,.0f} ({r.roof_kind}) "
                   f"| {r.roof / r.cycles:.1%} | {g['load']:.0%} | {g['feed']:.0%} | {g['epilogue']:.0%} "
                   f"| {other:.0%} | {r.moved_bytes / r.min_bytes:.1f}x | {r.traffic_floor:,} "
                   f"| {r.schedule_floor:,} "
                   f"| {sp['A']:.2f}x | {sp['A2']:.2f}x | {sp['A3']:.2f}x | {sp['B']:.2f}x | {sp['AB']:.2f}x |")
    return "\n".join(out)


def compare(rows: list[Row], base_path: str, expect: str = "A3") -> str:
    """Measured cycles against a baseline run (same workloads), next to what the baseline's
    counters projected for the change just made (`expect`: A, A2, A3, B or AB)."""
    base = {(b["workload"], b["R"], b["C"], b["lat"]): b for b in json.loads(Path(base_path).read_text())}
    out = [f"| workload | array | lat | baseline cycles | cycles | measured speedup | projected {expect} "
           "| % of roof | load wait | feed | epilogue | other | traffic / min | max reads in flight |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        b = base.get((r.workload, r.R, r.C, r.lat))
        if b is None:
            continue
        g = {k: sum(r.phases.get(x, 0) for x in v) / r.cycles for k, v in GROUPS.items()}
        other = g["drain"] + g["write"] + g["other"]
        out.append(f"| {r.workload} | {r.R}x{r.C} | {r.lat} | {b['cycles']:,} | {r.cycles:,} "
                   f"| {b['cycles'] / r.cycles:.2f}x | {b['cycles'] / b['proj'][expect]:.2f}x "
                   f"| {r.roof / r.cycles:.1%} | {g['load']:.0%} | {g['feed']:.0%} "
                   f"| {g['epilogue']:.0%} | {other:.0%} | {r.moved_bytes / r.min_bytes:.1f}x | {r.flight} |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--quick", action="store_true", help="8x8 and lat 2 only")
    ap.add_argument("--baseline", help="a --json file from an earlier run to compare against")
    ap.add_argument("--expect", default="A3", help="the baseline's projection to show (A, A2, A3, B, AB)")
    args = ap.parse_args()
    sizes = [(8, 8)] if args.quick else [(8, 8), (16, 16)]
    lats = [2] if args.quick else [2, 16]
    rows = []
    for name in WORKLOADS:
        for R, Cc_ in sizes:
            for lat in lats:
                rows.append(run_one(name, R, Cc_, lat))
                print(f"  {name} {R}x{Cc_} lat {lat}: {rows[-1].cycles:,} cycles", file=sys.stderr)
    print(table(rows))
    if args.baseline:
        print()
        print(compare(rows, args.baseline, args.expect))
    if args.json:
        Path(args.json).write_text(json.dumps([asdict(r) for r in rows], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
