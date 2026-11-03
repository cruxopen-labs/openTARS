"""Static timing of a Yosys/ABC-mapped netlist (ASAP7 cells, flip-flops left as Yosys `always`
blocks, the fp units as black boxes): every register-to-register path, not only the worst.

    python3 tools/timing/sta.py <mapped.v> <merged.lib> [--budget 650] [--top 40]

Delay model: NLDM tables (cell_rise / cell_fall, rise / fall_transition), bilinear interpolation
in input slew and load; load = the fanout pins' capacitance (flip-flop D and black-box inputs
0.6 fF); the worse of rise and fall per arc. Start points (registers, black-box outputs, inputs)
at 0 ps with a 10 ps slew, as ABC's `stime`. Output: the worst arrival per destination register
(enables and data together), grouped by register name.
"""
from __future__ import annotations

import argparse
import bisect
import re
import sys
from collections import defaultdict, deque

ENDCAP = 0.6
# logic of a black-box unit between its input port and its first register (ps; tools/timing
# run_stages.sh / run_units.sh): a path into the port must leave room for it
STAGE1 = {"fpu_mul": 586, "fpu_add": 483, "fpu_exp": 626, "fpu_gs": 626, "fpu_recip": 300,
          "fpu_rsqrt": 300, "fpu_softplus": 626}
STAGE1_PORT0 = {"sel"}                     # ports into a delay line only
START_SLEW = 10.0
# a net loaded beyond CAPMAX (typically a register enable or a flop output driving many flops,
# which ABC cannot see past its own boundary) is given an ideal buffer tree: the driver sees
# CAPMAX, each level of fanout-4 buffers adds BUF_D
CAPMAX, BUF_D = 4.0, 20.0


def parse_lib(path):
    """cell -> {'in': {pin: cap}, 'arcs': {(in, out): (i1, i2, delay, trans)}}"""
    cells = {}
    txt = open(path).read()
    for m in re.finditer(r"\n  cell \(([A-Za-z0-9_]+)\) \{", txt):
        name = m.group(1)
        start = m.end()
        nxt = txt.find("\n  cell (", start)
        body = txt[start:nxt if nxt > 0 else len(txt)]
        cell = {"in": {}, "arcs": {}, "out": set()}
        for pm in re.finditer(r"\n    pin \(([A-Za-z0-9_]+)\) \{", body):
            pname = pm.group(1)
            pe = body.find("\n    }", pm.end())
            pb = body[pm.end():pe]
            d = re.search(r"direction : (\w+);", pb)
            if d and d.group(1) == "input":
                c = re.search(r"\n      capacitance : ([0-9.]+);", pb)
                cell["in"][pname] = float(c.group(1)) if c else 0.5
            elif d and d.group(1) == "output":
                cell["out"].add(pname)
                for tm in re.finditer(r"timing \(\) \{(.*?)\n      \}", pb, re.S):
                    tb = tm.group(1)
                    rp = re.search(r'related_pin : "(\w+)"', tb)
                    tables = {}
                    for kind in ("cell_rise", "cell_fall", "rise_transition", "fall_transition"):
                        km = re.search(kind + r" \([^)]*\) \{\s*index_1 \(\"([^\"]+)\"\);\s*index_2 \(\"([^\"]+)\"\);"
                                       r"\s*values \((.*?)\);", tb, re.S)
                        if km:
                            i1 = [float(v) for v in km.group(1).split(",")]
                            i2 = [float(v) for v in km.group(2).split(",")]
                            rows = [[float(v) for v in r.split(",")] for r in re.findall(r'"([^"]+)"', km.group(3))]
                            tables[kind] = (i1, i2, rows)
                    sense = re.search(r"timing_sense : (\w+);", tb)
                    if rp and "cell_rise" in tables and "cell_fall" in tables:
                        key = (rp.group(1), pname)
                        cell["arcs"][key] = (tables["cell_rise"], tables["cell_fall"],
                                             tables.get("rise_transition", tables["cell_rise"]),
                                             tables.get("fall_transition", tables["cell_fall"]),
                                             sense.group(1) if sense else "non_unate")
        cells[name] = cell
    return cells


def interp(tab, s, c):
    i1, i2, v = tab
    def idx(ax, x):
        k = bisect.bisect_left(ax, x)
        k = min(max(k, 1), len(ax) - 1)
        return k - 1, k
    a0, a1 = idx(i1, s)
    b0, b1 = idx(i2, c)
    ts = (s - i1[a0]) / (i1[a1] - i1[a0])
    tc = (c - i2[b0]) / (i2[b1] - i2[b0])
    v00, v01, v10, v11 = v[a0][b0], v[a0][b1], v[a1][b0], v[a1][b1]
    return (v00 * (1 - ts) * (1 - tc) + v01 * (1 - ts) * tc + v10 * ts * (1 - tc) + v11 * ts * tc)


ID = r"(?:\\\S+|[A-Za-z_][A-Za-z0-9_$]*)"


class Net:
    def __init__(self):
        self.width = {}   # name -> (msb, lsb)

    def bits(self, expr):
        """bit names (MSB first) of an expression: ids, selects, constants, concatenations"""
        expr = expr.strip()
        if expr.startswith("{"):
            assert expr.endswith("}"), expr
            out = []
            for part in split_top(expr[1:-1]):
                out += self.bits(part)
            return out
        m = re.fullmatch(r"(\d+)'([hbd])([0-9a-fA-FxXzZ_]+)", expr)
        if m:
            return [None] * int(m.group(1))
        m = re.fullmatch(r"(" + ID + r")\s*(?:\[(\d+)(?::(\d+))?\])?", expr)
        assert m, expr
        name = m.group(1)
        if m.group(2) is None:
            msb, lsb = self.width.get(name, (0, 0))
            if msb == lsb and name not in self.width:
                return [name]
            rng = range(msb, lsb - 1, -1) if msb >= lsb else range(msb, lsb + 1)
            return [f"{name}[{i}]" for i in rng]
        hi = int(m.group(2))
        lo = int(m.group(3)) if m.group(3) is not None else hi
        if name not in self.width:
            return [name] if hi == lo == 0 else [f"{name}[{i}]" for i in range(hi, lo - 1, -1)]
        return [f"{name}[{i}]" for i in range(hi, lo - 1, -1)]


def split_top(s):
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur))
    return out


def refs(text, net):
    """every net bit referenced in a statement text"""
    out = []
    for m in re.finditer(r"(" + ID + r")(\s*\[\d+(?::\d+)?\])?", text):
        name = m.group(1)
        if name in ("if", "else", "begin", "end", "posedge", "clk", "always"):
            continue
        if re.fullmatch(r"\d+'[hbd].*", text[m.start():m.start() + 6]):
            continue
        b = net.bits(name + (m.group(2) or ""))
        out += [x for x in b if x is not None]
    return out


def base(bit):
    b = bit.split("[")[0] if not bit.startswith("\\") else bit.split(" ")[0]
    b = b.lstrip("\\")
    return re.sub(r"\[\d+\]", "[]", b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("netlist")
    ap.add_argument("lib")
    ap.add_argument("--budget", type=float, default=650.0)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--path", help="print the worst path into this register (name prefix)")
    args = ap.parse_args()
    lib = parse_lib(args.lib)
    print(f"lib: {len(lib)} cells", file=sys.stderr)
    net = Net()
    lines = open(args.netlist).read().split("\n")
    # declarations
    for ln in lines:
        m = re.match(r"\s*(wire|reg|input|output)\s+(?:\[(\d+):(\d+)\]\s+)?(" + ID + r")\s*;", ln)
        if m and m.group(2) is not None:
            net.width[m.group(4)] = (int(m.group(2)), int(m.group(3)))
    drivers = {}          # bit -> list of (src bit, arc tables or None)
    fanout_cap = defaultdict(float)
    starts = set()
    ends = []             # (bit, register label)
    cellpins = []         # (cell type, {pin: bit})
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        s = ln.strip()
        if s.startswith("assign "):
            stmt = s
            while not stmt.endswith(";"):
                i += 1; stmt += lines[i].strip()
            lhs, rhs = stmt[len("assign "):-1].split("=", 1)
            lb, rb = net.bits(lhs), net.bits(rhs)
            assert len(lb) == len(rb), stmt[:120]
            for a, b in zip(lb, rb):
                if a is not None:
                    drivers[a] = [(b, None)] if b is not None else []
        elif s.startswith("always @"):
            block = []
            i += 1
            while i < n and (lines[i].startswith("    ") ):
                block.append(lines[i].strip()); i += 1
            text = " ".join(block)
            tgts = re.findall(r"(" + ID + r"(?:\s*\[\d+(?::\d+)?\])?)\s*<=", text)
            label = base(net.bits(tgts[0])[0]) if tgts else "?"
            lhs_bits = set()
            for t in tgts:
                lhs_bits.update(x for x in net.bits(t) if x)
            for t in lhs_bits:
                starts.add(t); drivers[t] = []
            body = re.sub(r"(" + ID + r"(?:\s*\[\d+(?::\d+)?\])?)\s*<=", " ", text)
            for b in refs(body, net):
                ends.append((b, label, 0)); fanout_cap[b] += ENDCAP
            continue
        else:
            m = re.match(r"\s+([A-Za-z_][A-Za-z0-9_]*) (" + ID + r") \($", ln)
            if m and m.group(1) not in ("module",):
                ctype, iname = m.group(1), m.group(2)
                pins = {}
                i += 1
                while not lines[i].strip().startswith(");"):
                    pm = re.match(r"\s*\.(\w+)\((.*)\),?$", lines[i])
                    if pm:
                        pins[pm.group(1)] = pm.group(2)
                    i += 1
                if ctype in lib:
                    c = lib[ctype]
                    pb = {p: net.bits(e)[0] for p, e in pins.items() if e.strip()}
                    for p, b in pb.items():
                        if p in c["in"] and b:
                            fanout_cap[b] += c["in"][p]
                    for o in c["out"]:
                        if o in pb and pb[o]:
                            drivers[pb[o]] = [(pb[ip], c["arcs"].get((ip, o))) for ip in c["in"] if ip in pb and pb[ip]
                                              and (ip, o) in c["arcs"]]
                    if not c["out"]:
                        pass
                else:                          # black box: outputs start, inputs end
                    for p, e in pins.items():
                        bs = [x for x in net.bits(e) if x]
                        if p in ("y", "q", "rd", "y1", "ygs", "ysig", "ylong", "ysp"):
                            for b in bs:
                                starts.add(b); drivers[b] = []
                        elif p != "clk":
                            extra = 0 if p in STAGE1_PORT0 else STAGE1.get(ctype, 0)
                            for b in bs:
                                ends.append((b, f"{ctype}.{p} <- {iname}", extra)); fanout_cap[b] += ENDCAP
        i += 1
    print(f"nets driven: {len(drivers)}  endpoints: {len(ends)}", file=sys.stderr)
    # arrival times (memoized DFS in topological order)
    arr, slew, pred = {}, {}, {}
    sys.setrecursionlimit(1000000)

    def visit(root):
        stack = [(root, False)]
        while stack:
            b, done = stack.pop()
            if b in arr:
                continue
            ds = drivers.get(b)
            if ds is None or b in starts or not ds:
                load, tree = fanout_cap[b], 0.0
                while load > CAPMAX:
                    load /= 4.0; tree += BUF_D
                arr[b], slew[b], pred[b] = (tree, tree), (START_SLEW, START_SLEW), None
                continue
            if not done:
                stack.append((b, True))
                for src, _ in ds:
                    if src not in arr:
                        stack.append((src, False))
                continue
            # rise and fall tracked apart, by the arc's unateness
            best = [0.0, 0.0]; bs = [START_SLEW, START_SLEW]; bp = None; bt = -1.0
            load = fanout_cap[b] + 0.1
            tree = 0.0
            while load > CAPMAX:
                load /= 4.0; tree += BUF_D
            for src, arcs in ds:
                (ir, if_), (sr, sf) = arr[src], slew[src]
                if arcs is None:
                    cand = [(ir, sr), (if_, sf)]
                else:
                    cr, cf, tr, tf, sense = arcs
                    def out(cell_t, tran_t, from_rise):
                        a0, s0 = (ir, sr) if from_rise else (if_, sf)
                        return a0 + interp(cell_t, s0, load), interp(tran_t, s0, load)
                    if sense == "positive_unate":
                        cand = [out(cr, tr, True), out(cf, tf, False)]
                    elif sense == "negative_unate":
                        cand = [out(cr, tr, False), out(cf, tf, True)]
                    else:
                        r1, r2 = out(cr, tr, True), out(cr, tr, False)
                        f1, f2 = out(cf, tf, True), out(cf, tf, False)
                        cand = [max(r1, r2), max(f1, f2)]
                for k in (0, 1):
                    if cand[k][0] > best[k] or bp is None:
                        if cand[k][0] > best[k]:
                            best[k], bs[k] = cand[k]
                if max(cand[0][0], cand[1][0]) > bt:
                    bt, bp = max(cand[0][0], cand[1][0]), src
            arr[b], slew[b], pred[b] = (best[0] + tree, best[1] + tree), tuple(bs), bp

    worst = defaultdict(lambda: (-1.0, None))
    for b, label, extra in ends:
        visit(b)
        if max(arr[b]) + extra > worst[label][0]:
            worst[label] = (max(arr[b]) + extra, b)
    rows = sorted(worst.items(), key=lambda kv: -kv[1][0])
    over = [r for r in rows if r[1][0] > args.budget]
    print(f"worst {rows[0][1][0]:.0f} ps; {len(over)} register groups over {args.budget:.0f} ps")
    for label, (t, b) in rows[:args.top]:
        # walk back to the start point
        p, hops = b, 0
        while pred.get(p) is not None:
            p = pred[p]; hops += 1
        print(f"{t:7.0f} ps  {label:40s} from {p}  ({hops} hops)")
    if args.path:
        for label, (t, b) in rows:
            if label.startswith(args.path):
                p = b
                while p is not None:
                    print(f"   {max(arr[p]):7.1f}  {p}")
                    p = pred.get(p)
                break


if __name__ == "__main__":
    main()
