"""TARS-mini on the KV260: the device and the Qwen3.5-0.8B runtime.

Device backends (same interface):
  Fpga    the board: registers through libtmdrv (/dev/mem), a 1 GB hugepage buffer as the core's
          memory (the core's page table maps it), cache maintenance around each run. Root only.
  Golden  the bit-exact golden model on a PC (tars_mini.golden), to check this runtime offline.

    dev.mem            numpy uint8 view of the core's address space [0, MEM)
    dev.to_dev(a, n)   make CPU writes to [a, a + n) visible to the core
    dev.from_dev(a, n) make the core's writes to [a, a + n) visible to the CPU
    dev.run(q, n)      run n descriptors at core address q; returns (done, err, op_idx, cycles)
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if (HERE.parent / "pylib").is_dir():                 # the bundled wheels (setup.sh), ahead of the system's
    sys.path.insert(0, str(HERE.parent / "pylib"))

import numpy as np  # noqa: E402
MEM = 1 << 30                 # the core's address space (MEM_BYTES of the bitstream)
HPAGE = 2 << 20

# registers (rtl/tm_kv260.sv)
R_ID, R_VER, R_CTRL, R_STATUS, R_OPIDX = 0x000, 0x004, 0x008, 0x00C, 0x010
R_QLO, R_QHI, R_QLEN, R_CYCLO, R_CYCHI = 0x014, 0x018, 0x01C, 0x020, 0x024
R_RDB, R_WRB, R_CLEAR, R_PARAMS, R_MEMMB, R_PT = 0x028, 0x02C, 0x030, 0x034, 0x038, 0x2000
ERRORS = {0: "OK", 1: "E_MAGIC", 2: "E_VERSION", 3: "E_CRC", 4: "E_VALID", 5: "E_OPCODE", 6: "E_FIELD",
          7: "E_ALIGN", 8: "E_RANGE"}                    # tars_mini/contract.py


class Fpga:
    def __init__(self, lib: str | Path = HERE / "libtmdrv.so", hugepages: bool = True):
        if os.geteuid() != 0:
            raise PermissionError("run as root (sudo): /dev/mem and the physical addresses of the buffer")
        if not pl_loaded():                           # a read of an unprogrammed PL hangs the bus
            raise RuntimeError("the TARS-mini bitstream is not loaded (no tars_mini device, or the FPGA "
                               "manager is not 'operating'): sudo ./run.sh loads it")
        self.lib = ctypes.CDLL(str(lib))
        L = self.lib
        L.tm_open.restype = ctypes.c_int
        L.tm_rd.restype, L.tm_rd.argtypes = ctypes.c_uint32, [ctypes.c_uint32]
        L.tm_wr.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        L.tm_alloc.restype = ctypes.c_void_p
        L.tm_alloc.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int)]
        L.tm_clean.argtypes = L.tm_inval.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        e = L.tm_open()
        if e:
            raise OSError(-e, "cannot map the TARS-mini registers (is the bitstream loaded?)")
        ident = L.tm_rd(R_ID)
        if ident != 0x54415253:
            raise RuntimeError(f"no TARS-mini at 0xA0000000 (ID {ident:#x}); load the bitstream first")
        p = L.tm_rd(R_PARAMS)
        self.params = dict(R=p & 0xFF, C=(p >> 8) & 0xFF, VG=(p >> 16) & 0xF, page_bits=16 + ((p >> 24) & 0xF),
                           mem_mb=L.tm_rd(R_MEMMB), version=L.tm_rd(R_VER))
        if self.params["mem_mb"] << 20 != MEM or self.params["page_bits"] != 21:
            raise RuntimeError(f"unexpected bitstream parameters {self.params}")
        npg = MEM // HPAGE
        if hugepages:
            ensure_hugepages(npg + 16)
        ppn = (ctypes.c_uint64 * npg)()
        err = ctypes.c_int(0)
        ptr = L.tm_alloc(MEM, ppn, ctypes.byref(err))
        if not ptr:
            raise MemoryError(f"hugepage buffer: error {err.value} (need {npg} free 2 MB hugepages: "
                              f"echo {npg + 16} > /proc/sys/vm/nr_hugepages)")
        self.ptr = ptr
        self.last_perf = None
        self.mem = np.ctypeslib.as_array((ctypes.c_uint8 * MEM).from_address(ptr))
        self.ppn = list(ppn)
        self.reset()
        for i, pn in enumerate(self.ppn):          # the core's page i -> this physical page
            L.tm_wr(R_PT + 4 * i, pn)
        L.tm_wr(R_CLEAR, 1)

    def reset(self) -> None:
        self.lib.tm_wr(R_CTRL, 2)
        self.lib.tm_wr(R_CTRL, 0)

    def to_dev(self, a: int, n: int) -> None:
        self.lib.tm_clean(self.ptr + a, n)

    def from_dev(self, a: int, n: int) -> None:
        self.lib.tm_inval(self.ptr + a, n)

    def status(self) -> dict:
        s = self.lib.tm_rd(R_STATUS)
        return dict(halted=bool(s & 1), done=bool(s & 2), err=(s >> 4) & 0xF, idle=bool(s & 0x100),
                    rd_err=bool(s & 0x200), wr_err=bool(s & 0x400), in_reset=bool(s & 0x800))

    def run(self, q: int, n: int, timeout: float = 120.0) -> tuple[bool, int, int, int]:
        L = self.lib
        st = self.status()
        if not st["idle"]:
            raise RuntimeError(f"bridge busy before a run: {st}")
        self.reset()                                  # the core halts after each queue
        L.tm_wr(R_QLO, q & 0xFFFFFFFF)
        L.tm_wr(R_QHI, q >> 32)
        L.tm_wr(R_QLEN, n)
        L.tm_wr(R_CTRL, 1)
        t0 = time.time()
        while True:
            s = L.tm_rd(R_STATUS)
            if (s & 1) and (s & 0x100):              # halted, and every write acknowledged
                break
            if time.time() - t0 > timeout:
                raise TimeoutError(f"core still running after {timeout} s: {self.status()}, "
                                   f"op {L.tm_rd(R_OPIDX)}")
        st = self.status()
        if st["rd_err"] or st["wr_err"]:
            raise RuntimeError(f"AXI error responses: {st}")
        cyc = L.tm_rd(R_CYCLO)
        cyc |= L.tm_rd(R_CYCHI) << 32
        if self.params["version"] >= 0x10100:          # the bridge's counters (bitstream 1.1)
            p = {k: L.tm_rd(a) for k, a in (("rd_beats", R_RDB), ("wr_beats", R_WRB), ("bursts", 0x040),
                                             ("rd_hazard", 0x044), ("ar_wait", 0x048), ("wr_wait", 0x04C),
                                             ("rd_wait", 0x050))}
            occ = L.tm_rd(0x054)
            occ |= L.tm_rd(0x058) << 32
            p["cycles"] = cyc
            p["rd_latency"] = occ / max(p["rd_beats"], 1)      # mean, cycles (Little's law)
            self.last_perf = p
        return st["done"], st["err"], L.tm_rd(R_OPIDX), cyc


class Golden:
    """The golden model as the device (a PC check of everything above the hardware)."""
    params = dict(backend="golden model")

    def __init__(self, mem: int = MEM):
        import sys
        sys.path.insert(0, str(HERE.parents[2]))
        from tars_mini.golden import Machine
        self.Machine = Machine
        self.buf = bytearray(mem)
        self.mem = np.frombuffer(self.buf, np.uint8)

    def to_dev(self, a, n):
        pass

    def from_dev(self, a, n):
        pass

    def run(self, q, n, timeout=None):
        m = self.Machine(self.buf)
        st = m.run_queue(q, n)
        if m.ddr is not self.buf:
            self.mem[:] = np.frombuffer(m.ddr, np.uint8)
        return st.done, st.err_code, st.op_idx, 0


def pl_loaded() -> bool:
    """The overlay (tars_kv260.dtbo) is applied only after the bitstream programmed: its device node
    is the sign that 0xA0000000 answers."""
    try:
        state = Path("/sys/class/fpga_manager/fpga0/state").read_text().strip()
    except OSError:
        return False
    return state == "operating" and any(Path("/sys/bus/platform/devices").glob("*tars_mini*"))


def ensure_hugepages(n: int) -> None:
    p = Path("/proc/sys/vm/nr_hugepages")
    try:
        if int(p.read_text()) < n:
            p.write_text(str(n))
        got = int(p.read_text())
    except OSError as e:
        raise OSError(f"cannot set {p}: {e}")
    if got < n:
        raise MemoryError(f"only {got} hugepages of {n} (memory fragmented: reboot, then run this first)")


# ---------------------------------------------------------------------- Qwen3.5-0.8B
class Qwen:
    """The model's runtime: the DDR image (weights, the zeroed states), one descriptor queue per
    token position (made offline: tools/export_kv260.py), the embedding table (bf16); the host
    does the embedding lookup and the argmax, the core everything in between."""

    def __init__(self, dev, data: str | Path, log=print):
        self.dev, self.d = dev, Path(data)
        m = json.loads((self.d / "manifest.json").read_text())
        self.m = m
        self.T, self.H, self.vocab = m["max_tokens"], m["hidden"], m["vocab"]
        t = time.time()
        img = self.d / "image.bin"
        size = img.stat().st_size
        if size != m["image_bytes"]:
            raise RuntimeError(f"{img}: {size} bytes, the manifest says {m['image_bytes']}")
        with open(img, "rb") as f:                    # straight into the device buffer
            mv = memoryview(dev.mem)
            pos = 0
            while pos < size:
                k = f.readinto(mv[pos:min(size, pos + (64 << 20))])
                if not k:
                    raise IOError("short read of image.bin")
                pos += k
                log(f"\r  loading weights {pos >> 20} / {size >> 20} MiB", end="")
        log("")
        dev.mem[size:] = 0
        dev.to_dev(0, len(dev.mem))
        self.state0 = dev.mem[:m["state_end"]].copy()       # a fresh conversation's states
        self.embed = np.memmap(self.d / "embed_bf16.bin", np.uint16, "r", shape=(self.vocab, self.H))
        self.qidx = np.asarray(m["queue_index"], np.int64)  # [T, 2]: byte offset, descriptors
        self.queues = np.memmap(self.d / "queues.bin", np.uint8, "r")
        # prefill chunks: n tokens in one queue (the weights read once), at the t0 listed; only
        # chunks the GEMV unit takes in one pass (bitstream 3.0: 4 rows of 64-byte W beats; before:
        # 8 rows of 32-byte ones; a larger chunk would run on the systolic array, much slower)
        rows = 4 if getattr(dev, "params", {}).get("version", 0) >= 0x30000 else 8
        self.chunks = {}                                    # n -> (queues, {t0: (offset, descriptors)})
        for n, index in m.get("chunks", {}).items():
            if int(n) <= rows and (self.d / f"queues_c{n}.bin").exists():
                self.chunks[int(n)] = (np.memmap(self.d / f"queues_c{n}.bin", np.uint8, "r"),
                                       {int(t0): (int(o), int(k)) for t0, o, k in index})
        log(f"  ready in {time.time() - t:.0f} s ({size >> 20} MiB image, {self.T} positions, "
            f"prefill chunks {sorted(self.chunks) or 'none'})")
        self.pos = 0

    def new_conversation(self) -> None:
        self.dev.mem[:len(self.state0)] = self.state0
        self.dev.to_dev(0, len(self.state0))
        self.pos = 0

    def _put(self, t: int, tokens) -> None:
        """The embeddings of `tokens` into the input rows t, t + 1, ..."""
        m, dev = self.m, self.dev
        if t + len(tokens) > self.T:
            raise OverflowError(f"context full ({self.T} tokens)")
        for i, token in enumerate(tokens):
            x = (self.embed[token].astype(np.uint32) << 16).view(np.float32)   # bf16 -> fp32, exact
            a = m["x0_off"] + (t + i) * m["x0_ld"]
            dev.mem[a:a + 4 * self.H] = np.frombuffer(x.tobytes(), np.uint8)
            dev.to_dev(a, 4 * self.H)

    def _run(self, queues, off: int, n: int, what: str) -> int:
        """Run n descriptors from queues[off:]; a queue longer than the queue region goes in parts
        (the descriptors keep nothing on chip from one to the next). Returns the cycles."""
        q, cap, cyc = self.m["qbase"], self.m["queue_cap"], 0
        for s in range(0, n, cap):
            k = min(cap, n - s)
            self.dev.mem[q:q + 64 * k] = queues[off + 64 * s:off + 64 * (s + k)]
            self.dev.to_dev(q, 64 * k)
            done, err, op, c = self.dev.run(q, k)
            if not done or err:
                raise RuntimeError(f"{what}: the core stopped at descriptor {s + op} with error "
                                   f"{ERRORS.get(err, err)}")
            cyc += c
        return cyc

    def _logits(self) -> np.ndarray:
        lo = self.m["logits_off"]
        self.dev.from_dev(lo, 4 * self.vocab)
        return self.dev.mem[lo:lo + 4 * self.vocab].view(np.float32).copy()

    def step(self, token: int) -> np.ndarray:
        """Token `token` at the next position; returns the logits (fp32, vocab)."""
        t = self.pos
        self._put(t, [token])
        off, n = (int(v) for v in self.qidx[t])
        self.cycles = self._run(self.queues, off, n, f"position {t}")
        self.pos += 1
        return self._logits()

    def prefill(self, tokens, chunked: bool = True) -> np.ndarray:
        """`tokens` at the next positions; returns the logits after the last one. chunked: chunks of n
        tokens where the export has a queue for the position (bit-identical to one token at a time, the
        weights read once per chunk: the core's GEMV unit serves up to 8 rows a weight beat), single
        steps elsewhere; on the KV260 ~0.32 s per token in chunks of 8 against 0.73 s a step.
        chunked=False: every token a step. self.cycles: the core's total."""
        tokens = [int(v) for v in tokens]
        if not tokens:
            raise ValueError("no tokens")
        i, cyc, lg = 0, 0, None
        while i < len(tokens):
            t = self.pos
            n = next((n for n in sorted(self.chunks, reverse=True)
                      if chunked and i + n <= len(tokens) and t in self.chunks[n][1]), 1)
            if n == 1:
                lg = self.step(tokens[i])
                cyc += self.cycles
            else:
                self._put(t, tokens[i:i + n])
                queues, index = self.chunks[n]
                off, k = index[t]
                cyc += self._run(queues, off, k, f"positions {t}..{t + n - 1}")
                self.pos += n
                lg = None
            i += n
        self.cycles = cyc
        return lg if lg is not None else self._logits()
