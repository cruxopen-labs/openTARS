"""Whole-queue profiling, without splitting descriptors or changing their dependencies.

    sudo ./run.sh profile --data data --positions 16,64,128,192,250 --out profile.json
    sudo ./run.sh profile --data data --mode prefill --check chat --out prefill.json

Old v3.3 firmware reports aggregate traffic only. Rebuild the RTL for operator counters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import tars_board as B

CLASSES = ("overhead", "gemv", "elt", "fdot_nt", "fdot_nn", "rope", "quant", "norm", "softmax", "act")
METRICS = ("cycles", "rd_wait", "wr_wait", "rd_hazard")
COUNTERS = ("rd_beats", "wr_beats", "bursts", "rd_hazard", "ar_wait", "wr_wait", "rd_wait")
CAP = 0x50524601


def workload(raw: bytes) -> dict:
    """Count descriptor work, not hardware activity. Layout is the little-endian contract."""
    if len(raw) % 64:
        raise ValueError("incomplete descriptor")
    out = {name: dict(descriptors=0, macs=0) for name in CLASSES}
    for off in range(0, len(raw), 64):
        v = int.from_bytes(raw[off:off + 64], "little")
        op, ctrl = (v >> 432) & 0xFFFF, (v >> 416) & 0xFFFF
        m, n, k = (v >> 144) & 0xFFFF, (v >> 128) & 0xFFFF, (v >> 112) & 0xFFFF
        if op == 2:
            name = "gemv"
        elif op == 4:
            name = "quant" if ctrl & 7 == 4 else "softmax" if ctrl & 3 == 0 else "norm"
        else:
            name = {3: "act", 6: "elt", 7: "rope", 8: "fdot_nn" if ctrl & 1 else "fdot_nt"}.get(op)
        if name is None:
            raise ValueError(f"unsupported profiling opcode {op}")
        out[name]["descriptors"] += 1
        if op in (2, 8):
            out[name]["macs"] += m * n * k
    return out


def summarize(calls: list[dict], mhz: float, ddr_gbs: float) -> dict:
    if not calls or not np.isfinite(mhz) or not np.isfinite(ddr_gbs) or mhz <= 0 or ddr_gbs <= 0:
        raise ValueError("need completed queues and positive clock/bandwidth")
    cycles = sum(c["cycles"] for c in calls)
    if cycles <= 0:
        raise ValueError("no measured core cycles")
    totals = {k: sum(c[k] for c in calls) for k in COUNTERS}
    seconds = cycles / (mhz * 1e6)
    rd, wr = totals["rd_beats"] * 128, totals["wr_beats"] * 32
    result = dict(cycles=cycles, core_ms=seconds * 1e3, queues=len(calls),
                  traffic=dict(read_bytes=rd, write_bus_bytes=wr,
                               read_GBs=rd / seconds / 1e9, write_GBs=wr / seconds / 1e9,
                               read_pct_of_reference=100 * rd / seconds / (ddr_gbs * 1e9)),
                  counters=totals,
                  waits_pct={k: 100 * totals[k] / cycles
                             for k in ("rd_wait", "wr_wait", "rd_hazard", "ar_wait")})
    result["mean_read_latency_cycles"] = sum(c["rd_occupancy"] for c in calls) / max(totals["rd_beats"], 1)
    result["descriptors"] = sum(sum(w["descriptors"] for w in c["work"].values()) for c in calls)
    available = ["operators" in c for c in calls]
    if any(available) and not all(available):
        raise ValueError("mixed operator-counter capabilities")
    result["operator_counters_available"] = all(available)
    if not all(available):
        return result
    # Each hardware counter is 32 bits; never silently turn wrapped data into percentages.
    if any(c["cycles"] >= 1 << 32 for c in calls):
        raise ValueError("operator counters can wrap: each queue must take < 2^32 cycles")
    operators = []
    for i, name in enumerate(CLASSES):
        row = {k: sum(c["operators"][i][j] for c in calls) for j, k in enumerate(METRICS)}
        row.update(name=name, descriptors=sum(c["work"][name]["descriptors"] for c in calls),
                   macs=sum(c["work"][name]["macs"] for c in calls))
        row["pct"] = 100 * row["cycles"] / cycles
        row["ms"] = row["cycles"] / (mhz * 1e3)
        row["waits_pct"] = {k: 100 * row[k] / max(row["cycles"], 1) for k in METRICS[1:]}
        if any(row[k] > row["cycles"] for k in METRICS[1:]):
            raise ValueError(f"invalid wait counters for {name}")
        operators.append(row)
    if sum(r["cycles"] for r in operators) != cycles:
        raise ValueError("operator cycles do not sum to core cycles (ABI mismatch or counter wrap)")
    result["operators"] = operators
    return result


class Recorder:
    """Observe the existing Qwen._run calls, including every queue in a prefill."""
    def __init__(self, dev):
        self.dev, self.run = dev, dev.run
        self.calls = []
        self.capture = False
        dev.run = self

    def __call__(self, q, n, timeout=120.0):
        self.dev.profile_enabled = self.capture
        status = self.run(q, n, timeout)
        if self.capture:
            done, err, _, cycles = status
            if not done or err:
                raise RuntimeError(f"profile queue failed: {status}")
            p = self.dev.last_perf
            if not p or p["cycles"] != cycles:
                raise RuntimeError("missing performance counters")
            p = dict(p)
            p["work"] = workload(self.dev.mem[q:q + n * 64].tobytes())
            self.calls.append(p)
        return status


def checked_hash(logits, expected: dict | None) -> tuple[str, bool | None]:
    digest = hashlib.sha256(logits.tobytes()).hexdigest()
    if expected is not None and digest != expected["sha256"]:
        raise RuntimeError(f"logits MISMATCH at position {expected['pos']}")
    return digest, True if expected is not None else None


def print_sample(sample):
    print(f"\n{sample['label']}: {sample['core_ms']:.3f} core ms, "
          f"read {sample['traffic']['read_GBs']:.3f} GB/s, "
          f"write {sample['traffic']['write_GBs']:.3f} GB/s", flush=True)
    if not sample["operator_counters_available"]:
        print("  aggregate only: rebuild/load RTL with PROFILE_CAP=0x50524601")
        return
    print("  operator       cycles       %       ms   rd_wait%   wr_wait%   hazard%")
    for r in sample["operators"]:
        w = r["waits_pct"]
        print(f"  {r['name']:10s} {r['cycles']:10d} {r['pct']:7.2f} {r['ms']:8.3f} "
              f"{w['rd_wait']:10.2f} {w['wr_wait']:10.2f} {w['rd_hazard']:9.2f}")


def collect(model, recorder, check, mode, positions, mhz, ddr_gbs):
    """Use the real runtime/default chunking; check every exported golden step reached."""
    prompt = check["prompt_ids"]
    expected = {s["pos"]: s for s in check["steps"]}
    samples = []
    model.new_conversation()

    def sample(label, lg, want, started):
        digest, verified = checked_hash(lg, want)
        s = summarize(recorder.calls, mhz, ddr_gbs)
        s.update(label=label, logits_sha256=digest, golden_verified=verified,
                 host_ms=(time.perf_counter() - started) * 1e3)
        samples.append(s)
        print_sample(s)

    if mode in ("prefill", "both"):
        recorder.calls, recorder.capture = [], True
        started = time.perf_counter()
        lg = model.prefill(prompt)
        sample(f"prefill {len(prompt)} tokens", lg, expected.get(len(prompt) - 1), started)
        recorder.capture = False
        model.new_conversation()
    if mode in ("decode", "both"):
        if not positions or min(positions) < 0 or max(positions) >= model.T:
            raise ValueError(f"positions must be inside [0, {model.T})")
        token = prompt[0]
        for pos in range(max(positions) + 1):
            recorder.calls, recorder.capture = [], pos in positions
            started = time.perf_counter()
            lg = model.step(token)
            # Warm-up is also checked; after expected.json ends this is a measured greedy
            # trajectory with a hash, not a claim of independent golden verification.
            checked_hash(lg, expected.get(pos))
            if recorder.capture:
                sample(f"decode t={pos}", lg, expected.get(pos), started)
            token = prompt[pos + 1] if pos + 1 < len(prompt) else int(np.argmax(lg))
            if pos % 32 == 0:
                print(f"  reached position {pos}", flush=True)
        recorder.capture = False
    return samples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data"))
    ap.add_argument("--mode", choices=("decode", "prefill", "both"), default="decode")
    ap.add_argument("--positions", default="16,64,128,192,250")
    ap.add_argument("--check", default="plain", help="prompt/expected sequence in expected.json")
    ap.add_argument("--mhz", type=float, default=100.0)
    ap.add_argument("--ddr-gbs", type=float, default=12.79, help="independent read-only DDR reference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-operators", action="store_true", help="fail on old firmware")
    args = ap.parse_args()
    if not np.isfinite(args.mhz) or not np.isfinite(args.ddr_gbs) or args.mhz <= 0 or args.ddr_gbs <= 0:
        ap.error("clock and DDR reference must be finite and positive")
    try:
        positions = sorted({int(p) for p in args.positions.split(",")})
    except ValueError:
        ap.error("--positions must be comma-separated integers")
    data = Path(args.data)
    checks = json.loads((data / "expected.json").read_text())
    check = next((c for c in checks if c["name"] == args.check), None)
    if check is None or not check["prompt_ids"]:
        ap.error("check must select a nonempty prompt in expected.json")
    manifest = json.loads((data / "manifest.json").read_text())
    if args.mode != "prefill" and (min(positions) < 0 or max(positions) >= manifest["max_tokens"]):
        ap.error("positions outside the exported context")
    dev = B.Fpga(profile=True)
    if args.require_operators and dev.profile_cap != CAP:
        raise RuntimeError("operator counters missing: rebuild/load profiling RTL first")
    if dev.params["version"] < 0x30300:
        raise RuntimeError("this profiler's 128-byte read accounting requires firmware v3.3 or newer")
    if dev.profile_cap != CAP:
        print("WARNING: old firmware; aggregate traffic only, no operator attribution.", flush=True)
    model = B.Qwen(dev, data)
    recorder = Recorder(dev)
    samples = collect(model, recorder, check, args.mode, positions, args.mhz, args.ddr_gbs)
    report = dict(schema=1, model=manifest.get("model"), firmware=dev.params,
                  profile_cap=dev.profile_cap, mhz=args.mhz, ddr_reference_GBs=args.ddr_gbs,
                  notes=["Queues are not split; counters are read only after halt and bridge idle.",
                         "Wait counters overlap and must not be added or treated as compute-idle time.",
                         "Traffic/core-busy time is not an instantaneous DDR or arithmetic utilization.",
                         "Write bytes count 32-byte bus beats, including masked bytes.",
                         "Host time includes counter reads, workload decoding and hashing.",
                         "golden_verified=null means no independent expected logits at that position."],
                  samples=samples)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSaved {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
