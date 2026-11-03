"""Qwen3.5-0.8B LA + FA blocks on the RTL core under LPDDR4X-4266 (DRAMsim3): the golden model
runs the prefill, then the RTL runs one decode token block by block (LA, FA, final norm), each
checked byte for byte against the golden model and timed.

    python3 tools/perf_qwen.py [--prefill 16] [--out docs/qwen_decode.json]

Needs torch + transformers (for the model's shapes and random weights) and DRAMsim3
(tools/fetch_dramsim3.sh). About an hour on 2 cores (one Verilator build, ~8 M simulated cycles).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tars_mini import contract as C          # noqa: E402
from tars_mini import llm as L               # noqa: E402
from tars_mini import qwen_ref as Q          # noqa: E402
from tars_mini import rtlsim                 # noqa: E402
from tars_mini.golden import Machine         # noqa: E402

DDR = 1 << 27
OPS = {C.OP_GEMM: "GEMM", C.OP_RELU: "ACT", C.OP_ROW: "ROW", C.OP_ELT: "ELT", C.OP_ROPE: "ROPE",
       C.OP_FDOT: "FDOT"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "docs" / "qwen_decode.json"))
    args = ap.parse_args()
    t0 = time.time()
    cfg = Q.make_config(Q.CFG_08B)
    w = Q.random_weights(cfg)
    T = args.prefill + 1
    x = Q.random_inputs(cfg, T)
    low = L.QwenLowering(cfg, w, max_tokens=T, ddr_bytes=DDR)
    low.input(x)
    marks: dict[str, list[tuple[int, int]]] = {}
    for name in ("_la", "_fa"):
        f = getattr(low, name)

        def g(*a, f=f, name=name):
            s = len(low.p.descs)
            f(*a)
            marks.setdefault(name, []).append((s, len(low.p.descs)))
        setattr(low, name, g)
    low.chunk(0, args.prefill)
    low.chunk(args.prefill, 1)
    ddr, q, n = low.prog.build()
    la, fa = marks["_la"][1], marks["_fa"][1]
    m = Machine(bytearray(ddr))
    assert m.run_queue(q, la[0]).done                 # the prefill (and its LA / FA) on the golden model
    cur = bytes(m.ddr)
    res = {"config": "Qwen3.5-0.8B dims (random weights), prefill %d tokens, then 1 decode token" % args.prefill,
           "array": "8x8", "memory": "LPDDR4X-4266 x32 (DRAMsim3), fabric 40 ns", "blocks": {}}
    for name, a, b in (("LA", *la), ("FA", *fa), ("final norm", fa[1], n)):
        g = Machine(bytearray(cur))
        sg = g.run_queue(q + 64 * a, b - a)
        ops = Counter(OPS[C.decode(ddr[q + 64 * i:q + 64 * i + 64])["OPCODE"]] for i in range(a, b))
        r = rtlsim.run_core(cur, q + 64 * a, b - a, R=8, C=8, dram={})
        same = r.ddr == g.ddr and (r.done, r.err_code, r.op_idx) == (sg.done, sg.err_code, sg.op_idx)
        p = r.perf
        res["blocks"][name] = dict(
            descriptors=b - a, ops=dict(ops), cycles=r.cycles, bit_exact=same,
            bytes_read=sum(s.bytes_read for s in g.stats), bytes_written=sum(s.bytes_written for s in g.stats),
            gemm_cycles=sum(p.get(k, 0) for k in ("g_load", "g_load_mem", "g_feed", "g_epi")),
            row_xpu_cycles=p.get("r_elem", 0) + p.get("r_mem", 0),
            other_cycles=sum(p.get(k, 0) for k in ("fetch", "check", "write", "other")), perf=p)
        print(name, {k: v for k, v in res["blocks"][name].items() if k != "perf"}, flush=True)
        if not same:
            print(f"{name}: RTL differs from the golden model")
            return 1
        cur = bytes(g.ddr)
    # the head: one row block of the tied embedding (62080 x 1024 INT8, a GEMV); the head is
    # four such blocks, identical in cost
    import numpy as np
    from tars_mini.compiler import quantize_weight
    rng = np.random.default_rng(5)
    hp = L.Dev(ddr_bytes=DDR)
    yq = hp.mat(1, cfg.hidden_size, C.OUT_INT8)
    srow = hp.put(np.array([0.01, 0, 0, 0], np.float32))
    qw = quantize_weight(rng.normal(0, 0.02, (Q.HEAD_ROWS, cfg.hidden_size)))
    lin = L.QLin(hp.put(qw.q, "i1"), hp.put(qw.scale), None, qw.n, qw.k)
    out = hp.mat(1, Q.HEAD_ROWS)
    hp.gemm_rc(yq, srow, lin, out)
    hddr, hq, hn = hp.build()
    a0 = hp.base + yq.off
    hddr[a0:a0 + cfg.hidden_size] = rng.integers(-127, 128, cfg.hidden_size).astype(np.int8).tobytes()
    from tars_mini.golden import Machine as M2
    g = M2(bytearray(hddr))
    sg = g.run_queue(hq, hn)
    r = rtlsim.run_core(bytes(hddr), hq, hn, R=8, C=8, dram={})
    same = r.ddr == g.ddr and r.done and sg.done
    hb = dict(descriptors=1, ops={"GEMM": 1}, cycles=r.cycles, bit_exact=same, bytes_read=sum(x.bytes_read for x in g.stats),
              bytes_written=sum(x.bytes_written for x in g.stats), gemm_cycles=r.perf.get("g_feed", 0),
              row_xpu_cycles=0, other_cycles=0, perf=r.perf, note="one of the head's 4 row blocks (62080 x 1024)")
    res["blocks"]["head block"] = hb
    print("head block", {k: v for k, v in hb.items() if k != "perf"}, flush=True)
    if not same:
        print("head: RTL differs from the golden model")
        return 1
    res["head_cycles"] = 4 * r.cycles
    res["decode_token_24_layers"] = dict(
        la=18, fa=6, cycles=18 * res["blocks"]["LA"]["cycles"] + 6 * res["blocks"]["FA"]["cycles"]
        + res["blocks"]["final norm"]["cycles"] + 4 * r.cycles,
        note="18 LA + 6 FA layers (each measured once, the same for every layer of its kind), the "
             "final norm, 4 head blocks; FA's attention at this context length")
    res["total_cycles"] = sum(v["cycles"] for k, v in res["blocks"].items() if k != "head block")
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    print(f"LA + FA + norm: {res['total_cycles']} cycles; 24-layer token with the head: "
          f"{res['decode_token_24_layers']['cycles']} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
