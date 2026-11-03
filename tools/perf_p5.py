"""P5 performance: the v0.3 operators on the RTL, against their roofline.

    python3 tools/perf_p5.py [--json out.json]

Machine model: R*C int8 MACs per cycle; a 16-byte read channel and a separate 16-byte write
channel (as the DDR model and AXI have). The memory roofline is max(minimum bytes read, minimum
bytes written) / 16. (perf_report.py, used for P3-reuse, adds reads and writes on one port: its
memory rooflines are conservative for output-heavy cases.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tars_mini import compiler as Cc  # noqa: E402
from tars_mini import rtlsim  # noqa: E402

BEAT = 16


def _x(rng, m, n):
    return rng.normal(0, 2, (m, n)).astype(np.float32)


def build(name: str, rng):
    """(program, MACs, minimum bytes read, minimum bytes written) for one workload."""
    p = Cc.Program(ddr_bytes=1 << 23)
    if name in ("softmax", "layernorm", "rmsnorm"):
        M, N = 64, 1024
        xi = p.input_fp32(_x(rng, M, N))
        kw = {} if name == "softmax" else dict(gamma=rng.uniform(.5, 1.5, N),
                                                 beta=rng.normal(0, 1, N) if name == "layernorm" else None)
        p.rowop(xi, name, out="bf16", **kw)
        extra = 0 if name == "softmax" else 4 * N * (2 if name == "layernorm" else 1)
        return p, 0, M * N * 4 + extra + 64, M * N * 2
    if name == "gelu_eltwise":
        M, N = 64, 1024
        p.relu(p.input_fp32(_x(rng, M, N)), act="gelu", out="bf16")
        return p, 0, M * N * 4 + 64, M * N * 2
    if name in ("gemm_gelu", "gemm_relu"):
        M, N, K = 64, 256, 256
        x = rng.uniform(-1, 1, (M, K))
        s_x = Cc.act_scale(x)
        p.linear(p.input_int8(Cc.quantize_act(x, s_x), s_x), Cc.quantize_weight(rng.normal(0, .2, (N, K))),
                 bias=rng.normal(0, .5, N), act=name[5:], out="bf16")
        return p, M * N * K, M * K + N * K + 8 * N + 64, M * N * 2
    if name == "conv3x3":
        H = W = 28
        cin = cout = 32
        img = rng.uniform(-1, 1, (H, W, cin))
        s_x = Cc.act_scale(img)
        xi = p.input_int8(Cc.quantize_act(img.reshape(H * W, cin), s_x), s_x)
        p.conv2d(xi, H, W, rng.normal(0, .3, (cout, 3, 3, cin)), pad=1, bias=rng.normal(0, .2, cout),
                 act="relu", out="int8", out_scale=0.05)
        return p, H * W * cout * 9 * cin, H * W * cin + cout * 9 * cin + 8 * cout + 64, H * W * cout
    raise KeyError(name)


WORKLOADS = ["softmax", "layernorm", "rmsnorm", "gelu_eltwise", "gemm_relu", "gemm_gelu", "conv3x3"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()
    rows = []
    for name in WORKLOADS:
        for R, C_ in [(8, 8), (16, 16)]:
            for lat in (2, 16):
                prog, macs, rd, wr = build(name, np.random.default_rng(0))
                mb = max(rd, wr)
                ddr, q, n = prog.build()
                r = rtlsim.run_core(bytes(ddr), q, n, R=R, C=C_, lat=lat)
                assert r.done
                roof = max(macs / (R * C_), mb / BEAT)
                rows.append(dict(workload=name, R=R, C=C_, lat=lat, cycles=r.cycles, roof=roof,
                                 kind="compute" if macs / (R * C_) >= mb / BEAT else "memory",
                                 moved=(r.perf["rd_beats"] + r.perf["wr_beats"]) * BEAT, min_bytes=rd + wr))
                print(f"  {name} {R}x{C_} lat {lat}: {r.cycles:,}", file=sys.stderr)
    print("| workload | array | lat | cycles | roofline | % of roof | traffic / min |")
    print("|---|---|---|---:|---:|---:|---:|")
    for x in rows:
        print(f"| {x['workload']} | {x['R']}x{x['C']} | {x['lat']} | {x['cycles']:,} | {x['roof']:,.0f} ({x['kind']}) "
              f"| {x['roof'] / x['cycles']:.1%} | {x['moved'] / x['min_bytes']:.1f}x |")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
