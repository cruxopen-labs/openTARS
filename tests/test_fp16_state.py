"""Packed binary16 state, FP32 operations: independent conversion/recurrence oracles."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tars_mini import contract as C, numerics as F, rtlsim
from tars_mini.golden import ContractError, Machine
from tars_mini.llm import Dev, V

RTL = pytest.mark.skipif(not rtlsim.have_verilator(), reason="needs Verilator")


def _golden(p):
    ddr, q, n = p.build()
    g = Machine(bytearray(ddr))
    assert g.run_queue(q, n).done
    return ddr, q, n, g


def _both(p, stall=0):
    from test_rtl import both
    ddr, q, n = p.build()
    return both(bytes(ddr), q, n, stall=stall, lat=7)


def test_storage_all_half_patterns_and_numpy_rounding():
    h = np.arange(65536, dtype=np.uint16)
    x = h.view(np.float16).astype(np.float32)
    want = x.copy()
    want[np.isnan(want)] = np.uint32(0x7FC00000).view(np.float32)
    np.testing.assert_array_equal(F.from_fp16_bits(h).view(np.uint32), want.view(np.uint32))
    finite = np.isfinite(x)
    np.testing.assert_array_equal(F.to_fp16_bits(x[finite]), h[finite])
    assert F.from_fp16_bits(np.uint16(1)) == np.float32(2.0 ** -24)
    assert F.to_fp16_bits(np.float32(2.0 ** -25)) == 0
    assert F.to_fp16_bits(np.float32(-0.0)) == 0x8000
    assert F.to_fp16_bits(np.float32(65520)) == 0x7C00


@RTL
def test_rtl_fp16_exhaustive_load_and_store_midpoints():
    h = np.arange(65536, dtype=np.uint16)
    want = F.from_fp16_bits(h).view(np.uint32)
    got = rtlsim.run_fp([(12, int(v), 0) for v in h])
    np.testing.assert_array_equal(got, want)
    # Every positive finite half-way boundary, both signs, and either neighboring FP32.
    a = np.arange(0x7BFF, dtype=np.uint16).view(np.float16).astype(np.float32)
    b = np.arange(1, 0x7C00, dtype=np.uint16).view(np.float16).astype(np.float32)
    mid = (a + b) * np.float32(.5)
    edge = np.array([2.0**-25, 65520, np.inf, -np.inf, np.nan, -0.0], np.float32)
    x = np.concatenate([mid, -mid, np.nextafter(mid, np.float32(0)),
                        np.nextafter(mid, np.float32(np.inf)), edge,
                        np.random.default_rng(7).integers(0, 2**32, 65536, dtype=np.uint32).view(np.float32)])
    got = rtlsim.run_fp([(11, int(v), 0) for v in x.view(np.uint32)])
    np.testing.assert_array_equal(got, F.to_fp16_bits(x))


def recurrence_program(k=128, n=128):
    p = Dev(ddr_bytes=1 << 20)
    rng = np.random.default_rng(11)
    initial = rng.normal(0, .1, (k, n)).astype(np.float16)
    state = p.const(initial, C.OUT_FP16)
    key = rng.normal(0, .1, (1, k)).astype(np.float32)
    delta = rng.normal(0, .1, (1, n)).astype(np.float32)
    query = rng.normal(0, .1, (1, k)).astype(np.float32)
    kv, out = p.mat(1, n), p.mat(1, n)
    d = p.const(np.array([.9375], np.float32))
    p.elt(C.ELT_MUL, state, d, state, bmode=C.BM_SCALAR)
    p.fdot(p.const(key), state, kv, nn=True)
    p.elt(C.ELT_MULADD, p.const(delta), p.const(key), state,
          bmode=C.BM_COL, arow=True, c=state)
    p.fdot(p.const(query), state, out, nn=True)
    # NumPy scalar-precision operations, not the descriptor interpreter or its numeric helpers.
    decayed = (initial.astype(np.float32) * np.float32(.9375)).astype(np.float16)
    updated = (decayed.astype(np.float32) + key.T * delta).astype(np.float16)

    def dot(a, b):
        acc = np.zeros(n, np.float32)
        for i in range(k):
            product = (a[0, i] * b[i].astype(np.float32)).astype(np.float32)
            acc = (acc + product).astype(np.float32)
        return acc
    return p, state, kv, out, updated, dot(key, decayed), dot(query, updated)


def test_state_writes_round_but_dots_accumulate_fp32():
    p, s, kv, out, state, v, y = recurrence_program()
    _, _, _, g = _golden(p)
    assert g.read(s.off, state.nbytes) == state.tobytes()
    assert g.read(kv.off, v.nbytes) == v.tobytes()
    assert g.read(out.off, y.nbytes) == y.tobytes()
    # A half accumulator loses these 1024 increments; FP32 must retain all of them.
    p = Dev(ddr_bytes=1 << 18)
    a = np.full((1, 1024), 2.0**-12, np.float32)
    a[0, 0] = 1
    b = p.const(np.ones((1024, 16), np.float32), C.OUT_FP16)
    yv = p.mat(1, 16)
    p.fdot(p.const(a), b, yv, nn=True)
    _, _, _, g = _golden(p)
    assert np.frombuffer(g.read(yv.off, 64), "<f4")[0] == 1 + 1023 * 2.0**-12


@RTL
@pytest.mark.parametrize("stall", [0, 45])
def test_state_recurrence_rtl(stall):
    p, *_ = recurrence_program()
    assert _both(p, stall)[0].done


@RTL
@pytest.mark.parametrize("n,k,strided", [(16, 16, False), (48, 32, True), (64, 16, False),
                                         (128, 128, False), (192, 48, False), (128, 32, True)])
def test_fp16_fdot_wide_and_fallback(n, k, strided):
    p = Dev(ddr_bytes=1 << 20)
    rng = np.random.default_rng(n + k)
    a = p.const(rng.normal(size=(2, k)).astype(np.float32))
    width = n + 8 if strided else n
    b = p.const(rng.normal(size=(k, width)).astype(np.float32), C.OUT_FP16)
    bv = V(b.off, k, n, b.ld, b.fmt)
    for _ in range(2):
        p.fdot(a, bv, p.mat(2, n), nn=True)
    assert _both(p, 35)[0].done


@RTL
@pytest.mark.parametrize("arow", [False, True])
def test_fp16_elt_strides_subnormals_and_copy(arow):
    p = Dev(ddr_bytes=1 << 17)
    vals = np.arange(6 * 48, dtype=np.uint16).reshape(6, 48)
    vals[0, :8] = [0, 0x8000, 1, 0x8001, 0x3FF, 0x400, 0x7BFF, 0xFBFF]
    a = p.const(vals.view(np.float16), C.OUT_FP16)
    av = V(a.off + 16, 1 if arow else 6, 32, a.ld, a.fmt)
    dest = p.mat(6, 48, C.OUT_FP16)
    y = V(dest.off + 16, 6, 32, dest.ld, dest.fmt)
    one = p.const(np.ones((1, 1), np.float32))
    p.elt(C.ELT_MUL, av, one, y, bmode=C.BM_SCALAR, arow=arow)
    p.elt(C.ELT_MUL, y, one, y, bmode=C.BM_SCALAR)
    p.elt(C.ELT_MULADD, p.const(np.ones((1, 32), np.float32)),
          p.const(np.full((1, 6), 2**-24, np.float32)), y, bmode=C.BM_COL, arow=True, c=y)
    assert _both(p, 40)[0].done


def test_mixed_format_overlap_is_not_in_place():
    p = Dev(ddr_bytes=1 << 16)
    a = p.const(np.ones((2, 16), np.float32))
    p.elt(C.ELT_MUL, a, p.const(np.ones((1, 1), np.float32)),
          V(a.off, 2, 16, 32, C.OUT_FP16), bmode=C.BM_SCALAR)
    ddr, q, n = p.build()
    with pytest.raises(ContractError, match="overlaps"):
        Machine(ddr).run_queue(q, n)


def test_runtime_rejects_old_firmware_before_loading(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy/kv260/sw"))
    import tars_board as B
    (tmp_path / "manifest.json").write_text(json.dumps(dict(state_dtype="fp16")))
    dev = SimpleNamespace(params=dict(version=0x30300))
    with pytest.raises(RuntimeError, match="requires firmware"):
        B.Qwen(dev, tmp_path)


def test_precision_metrics_compare_identical_teacher_forced_inputs():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "deploy/kv260/sw/check_state_precision.py"
    spec = importlib.util.spec_from_file_location("check_state_precision", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Model:
        def __init__(self, values):
            self.values, self.seen = np.array(values, np.float32), []

        def new_conversation(self):
            self.seen = []

        def step(self, token):
            self.seen.append(token)
            return self.values

    a, b = Model([0, np.log(3)]), Model([np.log(3), 0])
    result = module.compare(a, b, [0, 1, 0])
    assert a.seen == b.seen == [0, 1, 0]
    assert result["top1_agreement"] == 0
    assert result["mean_nll_fp32"] == pytest.approx(-.5 * np.log(.75 * .25))
    assert result["mean_kl"] == pytest.approx(.5 * np.log(3))
    with pytest.raises(RuntimeError, match="nonfinite"):
        module.compare(a, Model([np.nan, 0]), [0])


def test_precision_collection_does_not_alias_one_physical_device(monkeypatch):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "deploy/kv260/sw/check_state_precision.py"
    spec = importlib.util.spec_from_file_location("precision_sequential", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dev = SimpleNamespace(active=None)
    loads, seen = [], []

    class Model:
        vocab = 2

        def __init__(self, device, data, log):
            self.name, self.dev = data, device
            device.active = data
            loads.append(data)

        def new_conversation(self):
            assert self.dev.active == self.name

        def step(self, token):
            assert self.dev.active == self.name
            seen.append((self.name, token))
            return np.array([token, -token], np.float32)

    monkeypatch.setattr(module.B, "Qwen", Model)
    report = module.compare_on_device(dev, ["fp32", "fp16"], {"a": [0, 1], "b": [1, 0]})
    assert loads == ["fp32", "fp16"]
    assert seen == [(mode, t) for mode in ("fp32", "fp16") for t in (0, 1, 1, 0)]
    assert all(r["top1_agreement"] == 1 and r["max_abs"] == 0 for r in report.values())


def test_lowered_model_matches_fp32_shadow_with_numpy_state_rounding():
    """Independent encoding oracle: unchanged FP32 descriptors, round only S writes."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tars_mini import llm as L, qwen_ref as Q
    cfg = Q.make_config(Q.CFG_SMALL)
    weights, inputs = Q.random_weights(cfg), Q.random_inputs(cfg, 4)
    results = []
    for dtype in ("fp32", "fp16"):
        low = L.QwenLowering(cfg, weights, max_tokens=4, ddr_bytes=1 << 21,
                             scratch_bytes=1 << 18, state_dtype=dtype)
        low.input(inputs)
        low.chunk(0, 3)
        low.chunk(3, 1)
        ddr, q, n = low.prog.build()
        states = [s for st in low.st if "S" in st for copy in st["S"] for s in copy]
        ranges = [(low.p.base + s.off, low.p.base + s.off + s.rows * s.ld) for s in states]

        class Shadow(Machine):
            rounded = 0

            def write(self, addr, data):
                if any(lo <= addr < hi for lo, hi in ranges):
                    self.rounded += 1
                    data = np.frombuffer(data, "<f4").astype(np.float16).astype("<f4").tobytes()
                super().write(addr, data)

        machine = Shadow(bytearray(ddr)) if dtype == "fp32" else Machine(bytearray(ddr))
        assert machine.run_queue(q, n).done
        if dtype == "fp32":
            assert machine.rounded >= 2 * 4 * len(states)
        outputs = [low.read_layer(machine, li, 0, 4) for li in range(len(low.types))]
        outputs.append(low.read_final(machine, 0, 4))
        for s in states:
            outputs.append(np.stack([
                np.frombuffer(machine.read(low.p.base + s.off + r * s.ld,
                                           s.cols * C.ESIZE[s.fmt]),
                              "<f2" if dtype == "fp16" else "<f4").astype("<f4")
                for r in range(s.rows)]))
        results.append(outputs)
    for reference, packed in zip(*results):
        np.testing.assert_array_equal(reference.view(np.uint32), packed.view(np.uint32))


@pytest.mark.parametrize("fmt", [C.OUT_FP32, C.OUT_FP16])
@pytest.mark.parametrize("mtp", [False, True])
def test_real_model_state_footprint_only_changes_s(fmt, mtp):
    from tars_mini.llm import QwenLowering
    low = QwenLowering.__new__(QwenLowering)
    low.__dict__.update(p=Dev(ddr_bytes=1 << 26), T=1, H=1024, nk=16, nv=16, dk=128, dv=128,
                        taps=4, types=(["linear_attention"] * 3 + ["full_attention"]) * 6,
                        Tp=16, mtp=mtp, state_fmt=fmt, nkv=2, hd=256, head=False)
    low._state()
    states = [s for st in low.st if "S" in st for copy in st["S"] for s in copy]
    assert sum(s.rows * s.ld for s in states) == (18 if fmt == C.OUT_FP32 else 9) * 2**20 * (2 if mtp else 1)
    assert all(s.fmt == fmt for s in states)
    assert all(s.fmt == C.OUT_FP32 for s in low.xs)
    assert all(v.fmt == C.OUT_FP32 for st in low.st for v in st.get("cin", {}).values())
    assert all(st["kc"].fmt == st["vc"].fmt == C.OUT_FP32 for st in low.st if "kc" in st)


def _half_ranges():
    elt = dict(OPCODE=C.OP_ELT, M=3, N=32)
    return [
        ({**elt, "HDR_RSV": 8}, "A_OFF", 3 * 64),
        ({**elt, "HDR_RSV": 12}, "A_OFF", 64),
        ({**elt, "EPI": C.epi(out=C.OUT_FP16)}, "Y_OFF", 3 * 64),
        ({**elt, "CTRL": C.ELT_MULADD, "EPI": C.epi(out=C.OUT_FP16, bias=1)}, "BIAS_OFF", 3 * 64),
        ({**elt, "CTRL": C.ELT_MULADD, "EPI": C.epi(out=C.OUT_FP16, bias=1), "LDY16": 6},
         "BIAS_OFF", 2 * 96 + 64),
        (dict(OPCODE=C.OP_FDOT, CTRL=1, HDR_RSV=8, M=2, N=16, K=32), "W_OFF", 32 * 32),
        (dict(OPCODE=C.OP_FDOT, CTRL=1, HDR_RSV=8, M=2, N=16, K=32, GROUP_SIZE=3),
         "W_OFF", 31 * 48 + 32),
    ]


@RTL
@pytest.mark.parametrize("fields,field,nbytes", _half_ranges())
@pytest.mark.parametrize("over", [False, True])
def test_fp16_range_ends_on_rtl(fields, field, nbytes, over):
    from test_rtl import both, MEM
    from test_v04 import _desc
    ddr = bytearray(MEM)
    fields = {**fields, field: MEM - nbytes + (16 if over else 0)}
    ddr[0x7000:0x7040] = _desc(**fields)
    st, _ = both(bytes(ddr), 0x7000, 1)
    assert st.err_code == (C.E_RANGE if over else C.E_OK)


@pytest.mark.parametrize("fields,code", [
    (dict(OPCODE=C.OP_FDOT, K=16, HDR_RSV=8), C.E_FIELD),
    (dict(OPCODE=C.OP_FDOT, K=16, CTRL=1, HDR_RSV=9), C.E_FIELD),
    (dict(OPCODE=C.OP_FDOT, K=16, CTRL=1, EPI=C.epi(out=C.OUT_FP16)), C.E_FIELD),
    (dict(OPCODE=C.OP_ROW, W_OFF=0, EPI=C.epi(out=C.OUT_FP16)), C.E_FIELD),
    (dict(OPCODE=C.OP_ELT, HDR_RSV=24), C.E_FIELD),
    (dict(OPCODE=C.OP_ELT, HDR_RSV=8, LDA16=1), C.E_ALIGN),
    (dict(OPCODE=C.OP_ELT, EPI=C.epi(out=C.OUT_FP16), LDY16=1), C.E_ALIGN),
    (dict(OPCODE=C.OP_FDOT, K=16, CTRL=1, HDR_RSV=8, GROUP_SIZE=1), C.E_ALIGN),
])
def test_fp16_fields_and_strides(fields, code):
    from test_v04 import _desc
    g = Machine(bytearray(1 << 16))
    assert g.validate(_desc(**fields))[0] == code
