"""Decode cycles against the position, measured: greedy steps from a short prompt to the end of the context.

    sudo ./run.sh hw && sudo python3 sw/ctx_bench.py data_qwen3
"""
import sys
import tars_board as B
import numpy as np
from pathlib import Path
d = Path(sys.argv[1])
m = B.Qwen(B.Fpga(), d, log=lambda *a, **k: None)
m.new_conversation()
lg = m.prefill([9707, 11, 1879, 0])
res = {}
while m.pos < m.T - 1:
    lg = m.step(int(np.argmax(lg)))
    res[m.pos - 1] = m.cycles
print(d.name, " ".join(f"t{t}={res[t] / 1e6:.2f}M" for t in (16, 64, 128, 192, 250) if t in res), flush=True)
