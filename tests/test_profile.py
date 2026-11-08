"""Profiler accounting and non-invasive KV260 operator counters."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tars_mini import contract as C
from tars_mini import rtlsim
from tars_mini.golden import Machine
from tars_mini.llm import Dev, QLin

SW = Path(__file__).resolve().parents[1] / "deploy/kv260/sw"
sys.path.insert(0, str(SW))
spec = importlib.util.spec_from_file_location("tars_perf_profile", SW / "perf_profile.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def call(cycles=100, operators=True):
    p = dict(cycles=cycles, rd_beats=10, wr_beats=3, bursts=2, rd_hazard=2,
             ar_wait=3, wr_wait=4, rd_wait=5, rd_occupancy=70,
             work={n: dict(descriptors=int(n == "gemv"), macs=128 if n == "gemv" else 0)
                   for n in P.CLASSES})
    if operators:
        p["operators"] = [[0, 0, 0, 0] for _ in P.CLASSES]
        p["operators"][0] = [20, 0, 0, 0]
        p["operators"][1] = [cycles - 20, 5, 4, 2]
    return p


def test_report_units_aggregation_and_wait_overlap():
    r = P.summarize([call(), call(200)], 100, 12.79)
    assert r["cycles"] == 300 and r["core_ms"] == pytest.approx(.003)
    assert r["traffic"]["read_bytes"] == 2560
    assert r["traffic"]["write_bus_bytes"] == 192
    assert r["traffic"]["read_GBs"] == pytest.approx(2560 / 3e-6 / 1e9)
    assert r["mean_read_latency_cycles"] == 7
    assert r["descriptors"] == 2
    assert sum(x["pct"] for x in r["operators"]) == pytest.approx(100)
    assert r["operators"][1]["macs"] == 256
    # Waits are overlapping observations, not a partition of cycles.
    p = call()
    p["operators"][1][1:] = [80, 80, 80]
    assert P.summarize([p], 100, 12.79)["operators"][1]["waits_pct"]["rd_wait"] == 100


@pytest.mark.parametrize("clock,bw", [(0, 12.79), (-1, 12.79), (float("nan"), 1), (100, float("inf"))])
def test_report_rejects_invalid_reference(clock, bw):
    with pytest.raises(ValueError):
        P.summarize([call()], clock, bw)


def test_old_firmware_and_counter_validation():
    r = P.summarize([call(operators=False)], 100, 12.79)
    assert not r["operator_counters_available"] and "operators" not in r
    with pytest.raises(ValueError, match="mixed"):
        P.summarize([call(), call(operators=False)], 100, 12.79)
    p = call()
    p["operators"][0][0] += 1
    with pytest.raises(ValueError, match="sum"):
        P.summarize([p], 100, 12.79)
    with pytest.raises(ValueError, match="wrap"):
        P.summarize([call(1 << 32)], 100, 12.79)
    p = call()
    p["operators"][1][1] = 81
    with pytest.raises(ValueError, match="invalid wait"):
        P.summarize([p], 100, 12.79)


@pytest.mark.parametrize("op,ctrl,name", [
    (C.OP_GEMM, 0, "gemv"), (C.OP_ELT, 0, "elt"), (C.OP_ROPE, 0, "rope"),
    (C.OP_FDOT, 0, "fdot_nt"), (C.OP_FDOT, 1, "fdot_nn"), (C.OP_RELU, 0, "act"),
    (C.OP_ROW, C.ROW_QUANT, "quant"), (C.OP_ROW, C.ROW_SOFTMAX, "softmax"),
    (C.OP_ROW, C.ROW_LAYERNORM, "norm"), (C.OP_ROW, C.ROW_RMSNORM, "norm"),
    (C.OP_ROW, C.ROW_L2NORM, "norm"),
])
def test_descriptor_work_matches_contract(op, ctrl, name):
    raw = C.encode(dict(OPCODE=op, CTRL=ctrl, M=4, N=32, K=128))
    w = P.workload(raw * 2)
    d = C.decode(raw)
    assert w[name]["descriptors"] == 2
    assert sum(x["descriptors"] for x in w.values()) == 2
    assert w[name]["macs"] == (2 * d["M"] * d["N"] * d["K"] if op in (C.OP_GEMM, C.OP_FDOT) else 0)


def test_invalid_descriptor_work_and_hash_mismatch():
    with pytest.raises(ValueError, match="incomplete"):
        P.workload(b"\0")
    with pytest.raises(ValueError, match="unsupported"):
        P.workload(C.encode(dict(OPCODE=99)))
    lg = np.arange(16, dtype=np.float32)
    digest = hashlib.sha256(lg.tobytes()).hexdigest()
    assert P.checked_hash(lg, None) == (digest, None)
    assert P.checked_hash(lg, dict(pos=1, sha256=digest)) == (digest, True)
    with pytest.raises(RuntimeError, match="MISMATCH"):
        P.checked_hash(lg, dict(pos=1, sha256="bad"))


def test_recorder_preserves_queue_and_captures_all_calls():
    raw = C.encode(dict(OPCODE=C.OP_ELT, M=1, N=16))
    calls = []

    def run(q, n, timeout):
        calls.append((q, n, timeout))
        return True, 0, n, 100

    dev = SimpleNamespace(mem=np.frombuffer(raw * 3, np.uint8), run=run, last_perf=call())
    rec = P.Recorder(dev)
    rec.capture = True
    assert dev.run(0, 2) == (True, 0, 2, 100)
    dev.run(128, 1, 5)
    assert calls == [(0, 2, 120.0), (128, 1, 5)]
    assert [p["work"]["elt"]["descriptors"] for p in rec.calls] == [2, 1]
    rec.capture = False
    dev.run(0, 1)
    assert len(rec.calls) == 2 and not dev.profile_enabled


@pytest.mark.parametrize("cap", [0, P.CAP])
def test_fpga_reads_optional_bank_after_queue_halt(cap):
    """The actual Fpga.run path with a fake MMIO library, no /dev/mem access."""
    dev = P.B.Fpga.__new__(P.B.Fpga)
    reads, writes = [], []
    regs = {P.B.R_STATUS: 0x103, P.B.R_CYCLO: 100, P.B.R_CYCHI: 0,
            P.B.R_OPIDX: 1, P.B.R_RDB: 10, 0x054: 70, 0x058: 0}
    regs.update({0x100 + i * 16 + j * 4: call()["operators"][i][j]
                 for i in range(10) for j in range(4)})

    def rd(a):
        reads.append(a)
        return regs.get(a, 0)

    dev.lib = SimpleNamespace(tm_rd=rd, tm_wr=lambda a, v: writes.append((a, v)))
    dev.params = dict(version=0x30300)
    dev.profile_enabled, dev.profile_cap = True, cap
    assert dev.run(0x1000, 1) == (True, 0, 1, 100)
    assert dev.last_perf["rd_occupancy"] == 70
    assert ("operators" in dev.last_perf) == (cap == P.CAP)
    assert (0x100 in reads) == (cap == P.CAP)
    if cap:
        assert reads.index(0x100) > reads.index(P.B.R_CYCLO)
        assert dev.last_perf["operators"] == call()["operators"]


def test_collect_real_runtime_prefill_and_decode(tmp_path, monkeypatch):
    """Exercise Qwen's real image/queue/chunk loading and entry points on a tiny backend."""
    # Two one-descriptor queues per chunk, one per decode; a cap of one forces multiple run
    # calls. The recorder must aggregate all of them, not just dev.last_perf.
    raw = C.encode(dict(OPCODE=C.OP_ELT, M=1, N=16))
    mem = np.zeros(4096, np.uint8)
    state_end, qbase, logits = 1024, 1024, 2048
    image = tmp_path / "image.bin"
    image.write_bytes(mem.tobytes())
    (tmp_path / "embed_bf16.bin").write_bytes(np.ones((16, 16), np.float32).view(np.uint32)
                                            .__rshift__(16).astype(np.uint16).tobytes())
    (tmp_path / "queues.bin").write_bytes(raw * 8)
    (tmp_path / "queues_c4.bin").write_bytes(raw * 2)
    manifest = dict(max_tokens=8, hidden=16, vocab=16, image_bytes=4096, state_end=state_end,
                    qbase=qbase, queue_cap=1, x0_off=0, x0_ld=64, logits_off=logits,
                    queue_index=[[64 * i, 1] for i in range(8)], chunks={"4": [[0, 0, 2]]})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    def run(q, n, timeout):
        dev.mem[logits:logits + 64] = np.arange(16, dtype=np.float32).view(np.uint8)
        dev.last_perf = call()
        return True, 0, n, 100

    dev = SimpleNamespace(mem=mem, params=dict(version=0x30300), run=run,
                          to_dev=lambda a, n: None, from_dev=lambda a, n: None)
    model = P.B.Qwen(dev, tmp_path, log=lambda *a, **k: None)
    rec = P.Recorder(dev)
    digest = hashlib.sha256(np.arange(16, dtype=np.float32).tobytes()).hexdigest()
    check = dict(prompt_ids=[0, 1, 2, 3], steps=[dict(pos=i, sha256=digest) for i in range(4)])
    samples = P.collect(model, rec, check, "both", [0, 3, 6], 100, 12.79)
    assert [s["queues"] for s in samples] == [2, 1, 1, 1]
    assert [s["cycles"] for s in samples] == [200, 100, 100, 100]
    assert [s["golden_verified"] for s in samples] == [True, True, True, None]
    assert model.pos == 7
    (tmp_path / "expected.json").write_text(json.dumps([dict(check, name="plain")]))
    # The actual CLI writes a self-contained report. Test new and old capability paths.
    for cap in (P.CAP, 0):
        class Device:
            params = dict(version=0x30300)
            profile_cap = cap
            mem = np.zeros(4096, np.uint8)

            def to_dev(self, a, n):
                pass

            def from_dev(self, a, n):
                pass

            def run(self, q, n, timeout):
                self.mem[logits:logits + 64] = np.arange(16, dtype=np.float32).view(np.uint8)
                self.last_perf = call(operators=self.profile_cap == P.CAP)
                return True, 0, n, 100

        monkeypatch.setattr(P.B, "Fpga", lambda profile: Device())
        out = tmp_path / f"profile-{cap}.json"
        monkeypatch.setattr(sys, "argv", ["perf_profile.py", "--data", str(tmp_path),
                                         "--positions", "0,3,6", "--mode", "both", "--out", str(out)])
        assert P.main() == 0
        report = json.loads(out.read_text())
        assert report["profile_cap"] == cap
        assert all(s["operator_counters_available"] == (cap == P.CAP) for s in report["samples"])
        assert report["samples"][0]["queues"] == 2


def mixed_program():
    p = Dev(ddr_bytes=1 << 16)
    x = p.const(np.linspace(.1, 1, 128, dtype=np.float32).reshape(1, 128))
    y = p.mat(1, 128)
    p.elt(C.ELT_MUL, x, p.const(np.full((1, 128), .5, np.float32)), x)
    p.elt(C.ELT_ADD, x, x, y)
    q, s = p.quant(y)
    w = QLin(p.put(np.ones((16, 128), np.int8), "i1"), p.put(np.ones(16, np.float32)), None, 16, 128)
    p.gemm_rc(q, s, w, p.mat(1, 16))
    a = p.const(np.ones((1, 32), np.float32))
    p.fdot(a, p.const(np.ones((16, 32), np.float32)), p.mat(1, 16), nn=False)
    p.fdot(a, p.const(np.ones((32, 64), np.float32)), p.mat(1, 64), nn=True)
    p.norm(x, p.const(np.ones(128, np.float32)), 1e-6, y)
    p.norm(x, p.const(np.ones(128, np.float32)), 1e-6, y, l2=True)
    p.softmax_causal(x, y, 17)
    p.act(x, C.ACT_SILU, y)
    p.rope(x, p.const(np.ones((1, 32), np.float32)), 1, 32, y)
    return p.build()


@pytest.mark.skipif(rtlsim.TOP != "kv260" or not rtlsim.have_verilator(), reason="TM_TOP=kv260 + Verilator")
@pytest.mark.parametrize("stall,lat", [(0, 2), (40, 8)])
def test_operator_registers_and_bit_exact_rtl(stall, lat):
    ddr, q, n = mixed_program()
    g = Machine(bytearray(ddr), gemv_only=True)
    st = g.run_queue(q, n)
    r = rtlsim.run_core(ddr, q, n, stall=stall, lat=lat)
    assert (r.done, r.err_code, r.op_idx) == (st.done, st.err_code, st.op_idx)
    assert r.ddr == g.ddr
    assert r.perf["profile_cap"] == P.CAP
    rows = [[r.perf[f"operator_{i}_{j}"] for j in range(4)] for i in range(10)]
    assert all(row[0] > 0 for row in rows)
    assert sum(row[0] for row in rows) == r.cycles
    assert all(v <= row[0] for row in rows for v in row[1:])
    # Boundary requests can still be outstanding across descriptors. Attribution is by
    # observation cycle; waits are not subtracted from execution cycles.
    p = call()
    p.update(cycles=r.cycles, rd_beats=r.perf["rd_beats"], wr_beats=r.perf["wr_beats"],
             operators=rows, work=P.workload(bytes(ddr[q:q + n * 64])))
    assert P.summarize([p], 100, 12.79)["operator_counters_available"]


@pytest.mark.skipif(rtlsim.TOP != "kv260" or not rtlsim.have_verilator(), reason="TM_TOP=kv260 + Verilator")
def test_profiling_disabled_does_not_change_cycles_or_results():
    ddr, q, n = mixed_program()
    # No random stimulus: separate generated binaries need not schedule their global
    # $urandom calls identically. lat=1/stall=0 fixes every memory handshake/response.
    on = rtlsim.run_core(ddr, q, n, stall=0, lat=1, seed=7)
    off = rtlsim.run_core(ddr, q, n, stall=0, lat=1, seed=7, params={"PROFILE": 0})
    assert (on.ddr, on.done, on.err_code, on.cycles) == (off.ddr, off.done, off.err_code, off.cycles)
    assert off.perf["profile_cap"] == 0
    assert not any(k.startswith("operator_") for k in off.perf)
