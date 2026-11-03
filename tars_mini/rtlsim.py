"""Build and run the TARS-mini RTL with Verilator; the harness every RTL test goes through.

Builds are cached under build/verilator/<top>_<hash>, keyed on the sources' contents, the
parameters, the flags and Verilator's version, so a test run rebuilds only what changed.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "rtl"
SIM = ROOT / "sim"
BUILD = ROOT / "build" / "verilator"

# -j: all cores for a lone build; under pytest-xdist each worker gets its share, since six
# simultaneous -j 0 builds of a large core ran the machine out of memory
_XW = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT") or 0)
_JOBS = str(max(1, (os.cpu_count() or 2) // _XW)) if _XW else "0"
VFLAGS = ["--binary", "--timing", "-j", _JOBS, "-Wno-fatal", "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC",
          "-Wno-UNUSEDSIGNAL", "-Wno-UNUSEDPARAM", "-O3", "--x-assign", "0", "--x-initial", "0"]
# TM_FPGA=1: the FPGA build of the RTL (`ifdef TM_FPGA: DSP multipliers, the shared fpu_act); the
# define is part of every build's hash, so both builds stay cached side by side
FPGA = bool(os.environ.get("TM_FPGA"))
# TM_TOP=kv260: run_core goes through the KV260 top (AXI-Lite registers, the AXI bridge with its
# page table) and an AXI memory model instead of the core's own ports (not with the DRAM model)
TOP = os.environ.get("TM_TOP", "core")
if TOP == "kv260":                     # the KV260 top has no systolic array: the golden model's profile
    os.environ.setdefault("TM_GEMV_ONLY", "1")
if FPGA:
    VFLAGS = VFLAGS + ["-DTM_FPGA"]

CORE_SOURCES = [RTL / "tm_fp_pkg.sv", RTL / "tm_fpp_pkg.sv", RTL / "tm_fpu_behav.sv", RTL / "tm_sfu.sv", RTL / "tm_mem.sv", RTL / "tm_vpu.sv", RTL / "tm_xpu.sv", RTL / "tm_vdly.sv",
                RTL / "tm_pe.sv", RTL / "tm_array.sv", RTL / "tm_core.sv"]
# the synthesizable pipelined units instead of their behavioral twins (slow to build: a cross-check)
CORE_SOURCES_STRUCT = [RTL / "tm_fpu.sv" if s.name == "tm_fpu_behav.sv" else s for s in CORE_SOURCES]


def have_verilator() -> bool:
    return shutil.which("verilator") is not None


def _version() -> str:
    try:
        return subprocess.run(["verilator", "--version"], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return ""


DRAMSIM3 = Path(os.environ.get("DRAMSIM3_DIR", ROOT / "ext_dramsim3"))
DRAM_CFG = SIM / "dram" / "LPDDR4X_4266_x32.ini"


def have_dramsim3() -> bool:
    return (DRAMSIM3 / "libdramsim3.so").exists()


def threads() -> int:
    """Verilator threads for the core testbench: TM_THREADS, else 1 under pytest-xdist (its
    workers already fill the cores) and min(8, cores / 2) for a lone long run."""
    if os.environ.get("TM_THREADS"):
        return max(1, int(os.environ["TM_THREADS"]))
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return 1
    return max(1, min(8, (os.cpu_count() or 2) // 2))


def _thread_flags() -> list[str]:
    n = threads()
    return ["--threads", str(n)] if n > 1 else []


def _dram_flags() -> list[str]:
    """Extra Verilator arguments for the DRAM-timed testbench (tools/fetch_dramsim3.sh builds it)."""
    if not have_dramsim3():
        raise RuntimeError(f"DRAMsim3 not found at {DRAMSIM3}; run tools/fetch_dramsim3.sh")
    return ["-DDRAM", str(SIM / "dram" / "tb_dram.cpp"),
            "-CFLAGS", f"-I{DRAMSIM3 / 'src'} -I{DRAMSIM3 / 'ext' / 'headers'} -std=c++17",
            "-LDFLAGS", f"-L{DRAMSIM3} -ldramsim3 -Wl,-rpath,{DRAMSIM3}"]


def build(top: str, sources: list[Path], params: dict | None = None, extra: list[str] | None = None) -> Path:
    """Verilate and compile; returns the executable. Concurrent callers share one build."""
    params = params or {}
    extra = extra or []
    h = hashlib.sha1()
    for s in sources:
        h.update(s.name.encode() + b"\0" + s.read_bytes() + b"\0")
    for e in extra:
        if e.endswith(".cpp"):
            h.update(Path(e).read_bytes())
    h.update(repr(sorted(params.items())).encode() + repr(VFLAGS[:2] + VFLAGS[4:]).encode() + repr(extra).encode()
             + _version().encode())
    out = BUILD / f"{top}_{h.hexdigest()[:12]}"
    exe = out / f"V{top}"
    if exe.exists():
        return exe
    out.mkdir(parents=True, exist_ok=True)
    with open(out / ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if exe.exists():
            return exe
        cmd = (["verilator", *VFLAGS, "--top-module", top, "-Mdir", str(out)]
               + [f"-G{k}={v}" for k, v in params.items()] + extra + [str(s) for s in sources])
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not exe.exists():
            raise RuntimeError(f"verilator failed:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}")
        for f in out.iterdir():               # keep the executable only (a build is ~200 MB)
            if f.name not in (exe.name, ".lock"):
                f.unlink()
    return exe


# ---------------------------------------------------------------------- fp unit vectors
FP_OPS = {"fmul": 0, "fadd": 1, "i2f": 2, "act": 3, "bf16": 4, "q8": 5, "exp": 6, "recip": 7,
          "rsqrt": 8, "fadd3": 9, "fmul3": 10}


def run_fp(vectors: list[tuple[int, int, int]]) -> np.ndarray:
    """Run (op, a, b) triples through rtl/tm_fp_pkg.sv; returns the 32-bit results."""
    exe = build("tb_fp", [RTL / "tm_fp_pkg.sv", RTL / "tm_fpp_pkg.sv", SIM / "tb_fp.sv"])
    with tempfile.TemporaryDirectory() as td:
        fin, fout = Path(td) / "in.txt", Path(td) / "out.txt"
        fin.write_text("".join(f"{op} {a & 0xFFFFFFFF:08x} {b & 0xFFFFFFFF:08x}\n" for op, a, b in vectors))
        r = subprocess.run([str(exe), f"+in={fin}", f"+out={fout}"], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
        return np.array([int(x, 16) for x in fout.read_text().split()], dtype=np.uint64)


FPU_OPS = {"fmul": 0, "fadd": 1, "exp": 2, "recip": 3, "rsqrt": 4, "act": 5}
FPU_SOURCES = [RTL / "tm_fp_pkg.sv", RTL / "tm_fpp_pkg.sv", RTL / "tm_fpu.sv"]


def run_fpu(op: str, vectors: list[tuple[int, int]]) -> np.ndarray:
    """Stream (a, b) pairs through the pipelined unit for `op` (rtl/tm_fpu.sv), one per cycle;
    returns the results in order. For "act" every b must carry the same act code."""
    exe = build("tb_fpu", FPU_SOURCES + [SIM / "tb_fpu.sv"])
    with tempfile.TemporaryDirectory() as td:
        fin, fout = Path(td) / "in.txt", Path(td) / "out.txt"
        fin.write_text("".join(f"{a & 0xFFFFFFFF:08x} {b & 0xFFFFFFFF:08x}\n" for a, b in vectors))
        r = subprocess.run([str(exe), f"+in={fin}", f"+out={fout}", f"+op={FPU_OPS[op]}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
        return np.array([int(x, 16) for x in fout.read_text().split()], dtype=np.uint64)


# ---------------------------------------------------------------------- the core
PHASES = ["fetch", "check", "g_load", "g_load_mem", "g_feed", "g_drain", "g_epi", "g_epi_mem",
          "write", "r_elem", "r_mem", "other"]


@dataclass
class RtlResult:
    ddr: bytearray
    done: bool
    err_code: int
    op_idx: int
    cycles: int
    perf: dict | None = None      # phase cycles (sum = cycles) and memory counters


def run_core(ddr: bytes | Path, queue_addr: int, queue_len: int, R: int = 8, C: int = 8,
             stall: int = 0, lat: int = 2, seed: int = 1, max_cycles: int = 50_000_000,
             params: dict | None = None, dram: dict | None = None, structural: bool = False,
             out_path: Path | None = None, log_path: Path | None = None) -> RtlResult:
    """Run a queue on the RTL core (R x C array; `params` sets other core parameters such as
    NB and OT). Memory timing: by default an ideal port that stalls `stall` percent of cycles on
    each side and answers each read 1..lat cycles after accepting it, in order, with any number
    in flight; with `dram` (a dict, possibly empty, of fabric / rcap / rbuf / wbuf / wflush /
    cfg) the LPDDR4X-4266 x32 DRAMsim3 model (sim/dram). The bytes are the same either way.
    `structural` builds the synthesizable pipelined fp units (rtl/tm_fpu.sv) instead of their
    cycle-identical behavioral twins (slow to build; used as a cross-check).

    For images too large to hold twice in memory: `ddr` may be the path of a raw image file, the
    final DDR then goes to `out_path` (the result's ddr is None), and the simulator's output
    (with a PROGRESS line every 10 M cycles) to `log_path` as it runs."""
    lean = isinstance(ddr, Path)
    size = ddr.stat().st_size if lean else len(ddr)
    if size % 16:
        raise ValueError("DDR image must be a whole number of 16-byte beats")
    # TM_MAXCYC caps every run (the mutation check sets it: a mutant that hangs the core then
    # fails in seconds, not after 50 M cycles)
    if os.environ.get("TM_MAXCYC"):
        max_cycles = min(max_cycles, int(os.environ["TM_MAXCYC"]))
    if TOP == "kv260" and dram is None:
        # the board's top (rtl/tm_kv260.sv) behind an AXI memory, pages scrambled (sim/tb_kv260.sv)
        # pages: small (many pages, scrambled), at most 2048 (the page table's register window)
        pb = next(p for p in [12, 11, 10, 9, 8] + list(range(13, 26))
                  if size % (1 << p) == 0 and 2 <= (size >> p) <= 2048 and (size >> p) % 5)
        exe = build("tb_kv260", (CORE_SOURCES_STRUCT if structural else CORE_SOURCES)
                    + [RTL / "tm_kv260.sv", SIM / "tb_kv260.sv"],
                    {"R": R, "C": C, "MEM_BYTES": size, "PAGE_BITS": pb, **(params or {})}, _thread_flags())
    else:
        exe = build("tb_core", (CORE_SOURCES_STRUCT if structural else CORE_SOURCES) + [SIM / "tb_core.sv"],
                    {"R": R, "C": C, "MEM_BYTES": size, **(params or {})},
                    _thread_flags() + (_dram_flags() if dram is not None else []))
    dargs = []
    if dram is not None:
        d = dict(dram)
        dargs = [f"+dramcfg={d.pop('cfg', DRAM_CFG)}"] + [f"+{k}={v}" for k, v in d.items()]
    with tempfile.TemporaryDirectory() as td:
        img = Path(td) / "img.bin"
        out = out_path if lean else Path(td) / "out.bin"
        # $fread fills each 128-bit word most significant byte first: reverse each 16-byte word
        if lean:
            src = np.memmap(ddr, dtype=np.uint8, mode="r").reshape(-1, 16)
            with open(img, "wb") as f:
                for i in range(0, src.shape[0], 1 << 20):
                    src[i:i + (1 << 20), ::-1].tofile(f)
            del src
        else:
            np.frombuffer(ddr, dtype=np.uint8).reshape(-1, 16)[:, ::-1].tofile(img)
        cmd = [str(exe), f"+img={img}", f"+out={out}", f"+qaddr={queue_addr:x}", f"+qlen={queue_len:x}",
               f"+stall={stall}", f"+lat={lat}", f"+seed={seed}", f"+maxcyc={max_cycles}", *dargs,
               *os.environ.get("TM_TBARGS", "").split()]
        if log_path is not None:
            with open(log_path, "w") as lf:
                rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT).returncode
            stdout, stderr = Path(log_path).read_text(), ""
        else:
            r = subprocess.run(cmd, capture_output=True, text=True)
            rc, stdout, stderr = r.returncode, r.stdout, r.stderr
        if os.environ.get("TM_DUMP"):
            Path(os.environ["TM_DUMP"]).write_text(stdout)
        m = re.search(r"STATUS done=(\d) err=(\d+) op=(\d+) cycles=(\d+)", stdout)
        if rc != 0 or not m:
            raise RuntimeError(f"simulation failed:\n{stdout[-3000:]}\n{stderr[-3000:]}")
        data = None if lean else bytearray(out.read_bytes())   # %u dumps each word in address order
    perf = {}
    for name, val in re.findall(r"^PERF (\w+) (\d+)$", stdout, re.M):
        if name.startswith("phase"):
            name = PHASES[int(name[5:])]
        elif name.startswith("rdph"):
            name = "rd_" + PHASES[int(name[4:])]
        perf[name] = int(val)
    return RtlResult(data, m.group(1) == "1", int(m.group(2)), int(m.group(3)), int(m.group(4)), perf)
