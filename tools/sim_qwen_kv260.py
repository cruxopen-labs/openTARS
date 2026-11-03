"""A reduced Qwen3.5 (the 0.8B model's dimensions, 2 layers: linear attention + full attention,
a small vocabulary) token by token on the RTL through the KV260 top (TM_TOP=kv260: AXI-Lite,
the AXI bridge, scrambled pages) with the bitstream's parameters, every step's whole DDR against
the golden model. Covers every operator shape of the full model's decode step.

    TM_FPGA=1 TM_TOP=kv260 python3 tools/sim_qwen_kv260.py [--steps 3] [--structural]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tars_mini import llm as L, qwen_ref as Q, rtlsim  # noqa: E402
from tars_mini.golden import Machine  # noqa: E402

BITSTREAM = dict(VG=1, NB=4, OT=32, HB=256, OQ=32, LG=4, VQMIN=128)   # rtl/tm_kv260_wrap.v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--structural", action="store_true")
    args = ap.parse_args()
    assert rtlsim.FPGA and rtlsim.TOP == "kv260", "run with TM_FPGA=1 TM_TOP=kv260"
    cfg = Q.make_config(Q.CFG_08B, ["linear_attention", "full_attention"], vocab=args.vocab)
    w = Q.random_weights(cfg, embed=True)
    low = L.QwenLowering(cfg, w, max_tokens=16, ddr_bytes=1 << 26, head=True, queue_cap=4096)
    ddr = low.p.ddr
    rng = np.random.default_rng(5)
    tok = int(rng.integers(0, args.vocab))
    ok = True
    for t in range(args.steps):
        low.put_input(ddr, t, low.embed([tok]))
        a = len(low.p.descs)
        low.chunk(t, 1)
        q, n = low.stage(ddr, a, len(low.p.descs))
        g = Machine(bytearray(ddr))
        st = g.run_queue(q, n)
        t0 = time.time()
        r = rtlsim.run_core(bytes(ddr), q, n, R=8, C=2, params=BITSTREAM, stall=10, lat=8, seed=t + 1,
                            max_cycles=400_000_000, structural=args.structural)
        same = r.ddr == g.ddr
        bad = 0 if same else int(np.count_nonzero(np.frombuffer(bytes(r.ddr), np.uint8) != np.frombuffer(bytes(g.ddr), np.uint8)))
        print(f"step {t}: {n} descriptors, status rtl {(r.done, r.err_code, r.op_idx)} golden "
              f"{(st.done, st.err_code, st.op_idx)}, {r.cycles} cycles ({r.cycles / 1e8:.3f} s at 100 MHz), "
              f"DDR {'bit-exact' if same else f'{bad} bytes differ'}, sim {time.time() - t0:.0f} s, perf {r.perf}",
              flush=True)
        ok &= same and (r.done, r.err_code, r.op_idx) == (st.done, st.err_code, st.op_idx)
        ddr = bytearray(g.ddr)
        tok = int(np.argmax(low.logits_of(ddr)))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
