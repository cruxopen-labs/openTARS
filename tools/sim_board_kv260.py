"""The board software (deploy/kv260/sw/tars_board.Qwen, unchanged) on the RTL: a device backend
that runs each token's queue on the KV260 top (TM_TOP=kv260: AXI-Lite, the bridge, 512 pages of
2 MB as on the board) with the bitstream's parameters, over the exported data; every step's
logits against data/expected.json (the golden model through the same runtime), bit for bit.

    TM_FPGA=1 TM_TOP=kv260 python3 tools/sim_board_kv260.py [--check plain] [--steps 2]
"""
import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deploy" / "kv260" / "sw"))
import tars_board as B  # noqa: E402
from tars_mini import rtlsim  # noqa: E402

BITSTREAM = dict(VG=1, NB=4, OT=32, HB=256, OQ=32, LG=4, VQMIN=128)   # rtl/tm_kv260_wrap.v


class RtlDev:
    params = dict(backend="RTL (tb_kv260)", version=0x30400)

    def __init__(self, work: Path):
        self.mem = np.zeros(B.MEM, np.uint8)
        self.work = work

    def to_dev(self, a, n):
        pass

    def from_dev(self, a, n):
        pass

    def run(self, q, n, timeout=None):
        src, dst = self.work / "in.bin", self.work / "out.bin"
        self.mem.tofile(src)
        t = time.time()
        r = rtlsim.run_core(src, q, n, R=8, C=2, params=BITSTREAM, stall=5, lat=6, seed=1,
                            max_cycles=600_000_000, out_path=dst, log_path=self.work / "sim.log")
        self.mem[:] = np.fromfile(dst, np.uint8)
        self.last = dict(cycles=r.cycles, sim_s=round(time.time() - t), perf=r.perf)
        return r.done, r.err_code, r.op_idx, r.cycles


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "deploy" / "kv260" / "data"))
    ap.add_argument("--check", default="plain")
    ap.add_argument("--steps", type=int, default=2)
    args = ap.parse_args()
    assert rtlsim.FPGA and rtlsim.TOP == "kv260", "run with TM_FPGA=1 TM_TOP=kv260"
    exp = next(c for c in json.loads((Path(args.data) / "expected.json").read_text()) if c["name"] == args.check)
    ok = True
    with tempfile.TemporaryDirectory(dir=ROOT / "build") as td:
        dev = RtlDev(Path(td))
        m = B.Qwen(dev, args.data, log=lambda *a, **k: None)
        m.new_conversation()
        for s in exp["steps"][:args.steps]:
            lg = m.step(s["token"])
            h = hashlib.sha256(lg.tobytes()).hexdigest()
            good = h == s["sha256"]
            ok &= good
            print(f"pos {s['pos']}: token {s['token']} -> {int(np.argmax(lg))} (golden {s['argmax']}) "
                  f"{'bit-exact' if good else 'MISMATCH'}; {dev.last['cycles']} cycles = "
                  f"{dev.last['cycles'] / 1e8:.3f} s at 100 MHz; sim {dev.last['sim_s']} s; {dev.last['perf']}",
                  flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
