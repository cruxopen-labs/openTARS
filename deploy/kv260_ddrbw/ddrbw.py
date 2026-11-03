#!/usr/bin/env python3
"""DDR bandwidth from the PL on the KV260 (run by run_ddrbw.sh after it loads a tgen<MHz> app).

Four traffic generators, one per S_AXI_HP0..3_FPD port (128 bits at pl_clk0). Each one streams
bursts through its own 2 MB hugepage (reads, or writes into a page of its own), so the CPU's memory
is never touched. Measures: unloaded latency; one port's bandwidth over burst length x bursts in
flight; each port alone; ports together (reads, writes, mixed). Prints tables, writes a JSON file.
    sudo python3 ddrbw.py [--lib ~/tars/sw/libtmdrv.so] [--out results.json] [--quick]
"""
import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

HPAGE = 2 << 20
NM = 4


class Tgen:
    def __init__(self, lib: str):
        L = self.L = ctypes.CDLL(lib)
        L.tm_open.restype = ctypes.c_int
        L.tm_rd.restype, L.tm_rd.argtypes = ctypes.c_uint32, [ctypes.c_uint32]
        L.tm_wr.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        L.tm_alloc.restype = ctypes.c_void_p
        L.tm_alloc.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int)]
        if L.tm_open() != 0:
            sys.exit("cannot map the registers (run as root)")
        if L.tm_rd(0x000) != 0x4E454754:
            sys.exit(f"no traffic generator at 0xA0000000 (ID {L.tm_rd(0):#x})")
        npg = 2 * NM
        ppn = (ctypes.c_uint64 * npg)()
        err = ctypes.c_int(0)
        if not L.tm_alloc(npg * HPAGE, ppn, ctypes.byref(err)):
            sys.exit(f"hugepage allocation failed ({err.value}); echo 16 > /proc/sys/vm/nr_hugepages")
        self.rd_page = [ppn[i] * HPAGE for i in range(NM)]          # master i reads page i
        self.wr_page = [ppn[NM + i] * HPAGE for i in range(NM)]     # and writes page NM + i
        self.mhz = self.clock_mhz()

    def clock_mhz(self) -> float:
        def fr():
            lo = self.L.tm_rd(0x00C)
            return (self.L.tm_rd(0x010) << 32) | lo
        c0, t0 = fr(), time.perf_counter()
        time.sleep(0.5)
        c1, t1 = fr(), time.perf_counter()
        return (c1 - c0) / (t1 - t0) / 1e6

    def run(self, jobs: dict, timeout: float = 30.0) -> dict:
        """jobs: {master: (mode 'r'|'w', beats per burst, bursts in flight, bursts)} -> results."""
        L, mask = self.L, 0
        for m, (mode, beats, ot, nb) in jobs.items():
            o = 0x100 * (m + 1)
            base = self.wr_page[m] if mode == "w" else self.rd_page[m]
            for off, v in ((0x00, base & 0xFFFFFFFF), (0x04, base >> 32), (0x08, HPAGE - 1),
                           (0x0C, beats - 1), (0x10, ot), (0x14, nb), (0x18, int(mode == "w"))):
                L.tm_wr(o + off, v)
            mask |= 1 << m
        L.tm_wr(0x004, 1 | (mask << 4))
        t0 = time.time()
        while L.tm_rd(0x008) & mask:
            if time.time() - t0 > timeout:
                sys.exit(f"TIMEOUT: generators {L.tm_rd(0x008):#x} still busy; reboot the board before using the PL again")
            time.sleep(0.001)
        res = {}
        for m, (mode, beats, ot, nb) in jobs.items():
            o = 0x100 * (m + 1)
            cyc, nbeats = L.tm_rd(o + 0x20), L.tm_rd(o + 0x24)
            lsum = L.tm_rd(o + 0x28) | (L.tm_rd(o + 0x2C) << 32)
            r = dict(mode=mode, beats=beats, ot=ot, bursts=nb, cycles=cyc, bytes=nbeats * 16,
                     lat_avg=lsum / max(nb, 1), lat_min=L.tm_rd(o + 0x30), lat_max=L.tm_rd(o + 0x34),
                     errs=L.tm_rd(o + 0x38))
            r["gbs"] = r["bytes"] / (cyc / (self.mhz * 1e6)) / 1e9 if cyc else 0.0
            if r["errs"] or r["bytes"] != nb * beats * 16:
                print(f"  WARNING master {m}: errs {r['errs']}, bytes {r['bytes']} of {nb * beats * 16}")
            res[m] = r
        cyc = max(r["cycles"] for r in res.values())
        tot = sum(r["bytes"] for r in res.values())
        res["total_gbs"] = tot / (cyc / (self.mhz * 1e6)) / 1e9
        return res


def bursts(beats: int, ot: int, mb: int = 64) -> int:
    return min((mb << 20) // (beats * 16), 4000 * ot)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=str(Path.home() / "tars/sw/libtmdrv.so"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t = Tgen(a.lib)
    peak = 19.2
    print(f"pl_clk0 {t.mhz:.1f} MHz: one HP port's ceiling {16 * t.mhz / 1e3:.2f} GB/s; DDR4-2400 x64 peak {peak} GB/s")
    out = {"mhz": t.mhz, "runs": []}

    def rec(name, jobs, res):
        out["runs"].append({"name": name, "jobs": {str(k): v for k, v in jobs.items()},
                            "res": {str(k): v for k, v in res.items()}})

    # 1. unloaded latency: one 16 B read (or 64 B), one at a time
    print("\n[1] latency, one burst at a time (HP0)")
    for beats in (1, 4, 16):
        j = {0: ("r", beats, 1, 5000)}
        r = t.run(j); rec(f"lat_r{beats}", j, r)
        x = r[0]
        print(f"  read {beats * 16:5d} B: AR->first R avg {x['lat_avg']:.0f} cyc = {x['lat_avg'] / t.mhz * 1e3:.0f} ns"
              f" (min {x['lat_min']}, max {x['lat_max']})")
    j = {0: ("w", 4, 1, 5000)}
    r = t.run(j); rec("lat_w4", j, r)
    print(f"  write   64 B: AW->B avg {r[0]['lat_avg']:.0f} cyc = {r[0]['lat_avg'] / t.mhz * 1e3:.0f} ns")

    # 2. one port: burst length x bursts in flight (reads)
    lens = (1, 4, 16, 64, 256) if a.quick else (1, 2, 4, 8, 16, 32, 64, 128, 256)
    ots = (1, 4, 16, 64) if a.quick else (1, 2, 4, 8, 16, 32, 64)
    print("\n[2] HP0 reads, GB/s: rows = burst bytes, columns = bursts in flight")
    print("  " + "burst".rjust(7) + "".join(f"{o:>7d}" for o in ots))
    for b in lens:
        row = []
        for o in ots:
            j = {0: ("r", b, o, bursts(b, o))}
            r = t.run(j); rec(f"sweep_r{b}_ot{o}", j, r)
            row.append(r[0]["gbs"])
        print("  " + f"{b * 16:6d}B" + "".join(f"{g:7.2f}" for g in row))

    best = ("r", 16, 32)                            # 256 B bursts, 32 in flight (refined below)
    # 3. each port alone
    print("\n[3] each port alone, reads (256 B bursts, 32 in flight) and writes")
    for m in range(NM):
        jr = {m: ("r", 16, 32, bursts(16, 32))}
        jw = {m: ("w", 16, 32, bursts(16, 32))}
        rr = t.run(jr); rec(f"alone_r{m}", jr, rr)
        rw = t.run(jw); rec(f"alone_w{m}", jw, rw)
        print(f"  HP{m}: read {rr[m]['gbs']:.2f} GB/s (lat {rr[m]['lat_avg'] / t.mhz * 1e3:.0f} ns), "
              f"write {rw[m]['gbs']:.2f} GB/s")

    # 4. ports together
    combos = [(0,), (0, 3), (0, 1), (1, 2), (0, 1, 3), (0, 2, 3), (0, 1, 2, 3)]
    for beats, ot in ((16, 32), (64, 16), (256, 8)):
        print(f"\n[4] ports together, {beats * 16} B bursts, {ot} in flight: reads / writes")
        for c in combos:
            jr = {m: ("r", beats, ot, bursts(beats, ot)) for m in c}
            jw = {m: ("w", beats, ot, bursts(beats, ot)) for m in c}
            rr = t.run(jr); rec(f"multi_r_{beats}_{ot}_" + "".join(map(str, c)), jr, rr)
            rw = t.run(jw); rec(f"multi_w_{beats}_{ot}_" + "".join(map(str, c)), jw, rw)
            per = " ".join(f"HP{m} {rr[m]['gbs']:.2f}" for m in c)
            print(f"  {'+'.join('HP%d' % m for m in c):18s} read {rr['total_gbs']:5.2f} GB/s ({100 * rr['total_gbs'] / peak:3.0f}%) [{per}]"
                  f"   write {rw['total_gbs']:5.2f} GB/s ({100 * rw['total_gbs'] / peak:3.0f}%)")

    # 5. mixed: reads on three ports while one port writes
    print("\n[5] mixed: HP0/HP1/HP3 read, HP2 writes (256 B bursts, 32 in flight)")
    j = {0: ("r", 16, 32, bursts(16, 32)), 1: ("r", 16, 32, bursts(16, 32)),
         3: ("r", 16, 32, bursts(16, 32)), 2: ("w", 16, 32, bursts(16, 32))}
    r = t.run(j); rec("mixed_3r1w", j, r)
    print(f"  total {r['total_gbs']:.2f} GB/s: " + " ".join(f"HP{m} {r[m]['mode']} {r[m]['gbs']:.2f}" for m in (0, 1, 3, 2)))

    fn = a.out or f"ddrbw_{round(t.mhz)}.json"
    Path(fn).write_text(json.dumps(out, indent=1))
    print(f"\nresults: {fn}")


if __name__ == "__main__":
    main()
