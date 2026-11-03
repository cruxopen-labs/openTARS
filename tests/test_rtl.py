"""RTL (Verilator) vs the golden model: every test runs the same queue on the same DDR image
through both and requires identical DDR bytes and an identical end status.

The array size (R x C), the DDR model's backpressure and read latency vary; the golden model has
none of these parameters, so any difference is a bug in the RTL (or a hole in the contract).
"""
import numpy as np
import pytest

from tars_mini import compiler as Cc
from tars_mini import contract as C
from tars_mini import rtlsim
from tars_mini.golden import ContractError, Machine

pytestmark = pytest.mark.skipif(not rtlsim.have_verilator(), reason="needs Verilator")

SIZES = [(8, 8), (16, 16), (4, 8), (8, 4), (3, 5)]


# the KV260 top has no systolic array: Conv and GEMM with K > 4096 are rejected (E_OPCODE), the
# golden model's gemv_only profile; the DRAM-model runs go through the core's own testbench
GEMV_ONLY = rtlsim.TOP == "kv260"
no_array = pytest.mark.skipif(GEMV_ONLY, reason="the KV260 top has no systolic array")


def expect_code(raw: bytes, code: int) -> int:
    """The status code a descriptor gets on this top: the contract's, or E_OPCODE where the
    GEMV-only profile rejects it and no earlier rule applies."""
    d = C.decode(raw)
    unsup = d["OPCODE"] in C.CONV_OPS or (d["OPCODE"] == C.OP_GEMM and d["K"] > 4096)
    return C.E_OPCODE if GEMV_ONLY and unsup and (code == C.E_OK or code > C.E_OPCODE) else code


def both(ddr: bytes, qaddr: int, qlen: int, R=8, C_=8, stall=0, lat=2, seed=1, params=None, dram=None):
    g = Machine(bytearray(ddr), gemv_only=GEMV_ONLY and dram is None)
    st = g.run_queue(qaddr, qlen)
    r = rtlsim.run_core(ddr, qaddr, qlen, R=R, C=C_, stall=stall, lat=lat, seed=seed, params=params, dram=dram)
    assert (r.done, r.err_code, r.op_idx) == (st.done, st.err_code, st.op_idx), \
        f"status rtl={(r.done, r.err_code, r.op_idx)} golden={(st.done, st.err_code, st.op_idx)}"
    if r.ddr != g.ddr:
        a = np.frombuffer(bytes(r.ddr), np.uint8)
        b = np.frombuffer(bytes(g.ddr), np.uint8)
        d = np.nonzero(a != b)[0]
        raise AssertionError(f"{len(d)} DDR bytes differ; first at {d[0]:#x}: "
                             f"rtl {bytes(r.ddr[d[0]:d[0] + 8]).hex()} golden {bytes(g.ddr[d[0]:d[0] + 8]).hex()}")
    return st, r


# ---------------------------------------------------------------------- random queues
MEM = 1 << 16
QADDR = 0xF000


def _random_desc(rng, inject: bool, kmax: int = 4) -> dict:
    u = rng.random()
    gemm, row, conv = u < 0.52, 0.52 <= u < 0.72, 0.72 <= u < 0.86    # else RELU
    if conv and not GEMV_ONLY:               # (the KV260 top: a GEMM instead)
        return _random_conv(rng, inject)
    gemm = gemm or conv
    M = int(rng.integers(1, 21))
    N = 16 * int(rng.integers(1, 5))
    K = 16 * int(rng.integers(1, kmax + 1)) if gemm else 0
    out = int(rng.integers(0, 3))
    es = C.ESIZE[out]
    act = int(rng.integers(0, 5)) if gemm else 0 if row else int(rng.integers(1, 5))
    arow = K if gemm else 4 * N
    lda = arow + 16 * int(rng.integers(0, 3)) * int(rng.random() < 0.4)
    ldy = N * es + 16 * int(rng.integers(0, 3)) * int(rng.random() < 0.4)

    def off(nbytes):
        """Often a slot from a small pool, so descriptors read, overwrite and re-read each other's
        bytes (RAW / WAR / WAW, and stale-cache hazards); otherwise anywhere."""
        if rng.random() < 0.7:
            slot = 0x800 * int(rng.integers(0, 0xE000 // 0x800))
            if slot + nbytes <= 0xE000:
                return slot
        return 16 * int(rng.integers(0, (0xE000 - nbytes) // 16))

    f = dict(MAGIC=C.MAGIC, VERSION=C.VERSION, VALID=1,
             OPCODE=C.OP_GEMM if gemm else C.OP_ROW if row else C.OP_RELU,
             M=M, N=N, K=K, A_OFF=off((M - 1) * lda + arow), Y_OFF=off((M - 1) * ldy + N * es),
             LDA16=lda // 16 if lda != arow else 0, LDY16=ldy // 16 if ldy != N * es else 0)
    if gemm:
        qm = int(rng.integers(1, 3))
        bias = int(rng.integers(0, 2))
        f.update(CTRL=C.ctrl(qm, fusion=int(act != 0)), EPI=C.epi(act, out, bias),
                 W_OFF=off(N * K), SCALE_OFF=off(4 * N), BIAS_OFF=off(4 * N) if bias else 0)
    elif row:
        fn = int(rng.integers(0, 3))
        bias = int(rng.integers(0, 2)) if fn != C.ROW_SOFTMAX else 0
        f.update(CTRL=fn, EPI=C.epi(C.ACT_NONE, out, bias))
        if fn != C.ROW_SOFTMAX:
            eps = [0, C.f32_bits(1e-5), C.f32_bits(1e-6), C.f32_bits(float(rng.uniform(0, 1)))]
            f.update(SCALE_OFF=off(4 * N), BIAS_OFF=off(4 * N) if bias else 0,
                     W_OFF=eps[int(rng.integers(0, 4))])
    else:
        f.update(CTRL=0, EPI=C.epi(act, out))
    if out == C.OUT_INT8:
        f["OUT_SCALE"] = C.f32_bits(float(rng.uniform(0.5, 60)))
    raw = C.encode(f)
    if inject:
        kind = rng.integers(0, 4)
        if kind == 0:                                       # flip a bit, CRC stays stale
            b = bytearray(raw)
            b[int(rng.integers(0, 64))] ^= 1 << int(rng.integers(0, 8))
            raw = bytes(b)
        else:                                               # a bad field with a valid CRC
            field = ["CORE_ID", "GROUP_SIZE", "DEP_SIG", "VERSION", "VALID", "N", "K",
                     "A_OFF", "LDY16", "BASE_ADDR", "OPCODE"][int(rng.integers(0, 11))]
            val = {"CORE_ID": 1, "GROUP_SIZE": 64, "DEP_SIG": 1, "VERSION": 3, "VALID": 0,
                   "N": 8, "K": 8, "A_OFF": f["A_OFF"] + 4, "LDY16": 1, "BASE_ADDR": MEM,
                   "OPCODE": 0x0F}[field]
            raw = C.encode({**f, field: val})
    return raw


def _random_conv(rng, inject: bool) -> bytes:
    """A legal Conv / Conv+act descriptor: small images, 1-3 kernels, stride 1-2, pad 0-2."""
    while True:
        H, W = int(rng.integers(1, 9)), int(rng.integers(1, 9))
        KH, KW, S, P = int(rng.integers(1, 4)), int(rng.integers(1, 4)), int(rng.integers(1, 3)), int(rng.integers(0, 3))
        OH, OW = (H + 2 * P - KH) // S + 1, (W + 2 * P - KW) // S + 1
        if H + 2 * P >= KH and W + 2 * P >= KW and OH >= 1 and OW >= 1:
            break
    cin, N = 16 * int(rng.integers(1, 3)), 16 * int(rng.integers(1, 4))
    K, M = KH * KW * cin, OH * OW
    lda = cin + 16 * int(rng.integers(0, 2)) * int(rng.random() < 0.4)
    out = int(rng.integers(0, 3))
    es = C.ESIZE[out]
    act = int(rng.integers(0, 5))
    opcode = C.OP_CONV_ACT if act and rng.random() < 0.5 else C.OP_CONV
    qm, bias = int(rng.integers(1, 3)), int(rng.integers(0, 2))

    def off(nbytes):
        if rng.random() < 0.7:
            slot = 0x800 * int(rng.integers(0, 0xE000 // 0x800))
            if slot + nbytes <= 0xE000:
                return slot
        return 16 * int(rng.integers(0, (0xE000 - nbytes) // 16))

    f = dict(MAGIC=C.MAGIC, VERSION=C.VERSION, VALID=1, OPCODE=opcode, M=M, N=N, K=K,
             A_OFF=off((H * W - 1) * lda + cin), Y_OFF=off(M * N * es), W_OFF=off(N * K),
             SCALE_OFF=off(4 * N), BIAS_OFF=off(4 * N) if bias else 0,
             LDA16=lda // 16 if lda != cin else 0, CTRL=C.ctrl(qm, fusion=int(act != 0)),
             EPI=C.epi(act, out, bias), GROUP_SIZE=W, HDR_RSV=C.conv_geom(H, KH, KW, S, P))
    if out == C.OUT_INT8:
        f["OUT_SCALE"] = C.f32_bits(float(rng.uniform(0.5, 60)))
    if inject:
        bad = [dict(M=M + 1), dict(K=K + 16 * KH * KW) if False else dict(HDR_RSV=f["HDR_RSV"] & ~0xF00),
               dict(OPCODE=C.OP_CONV_ACT, CTRL=C.ctrl(qm), EPI=C.epi(0, out, bias))][int(rng.integers(0, 3))]
        f.update(bad)
    return C.encode(f)


def random_program(seed: int, n: int = 8, inject_p: float = 0.08, kmax: int = 4) -> bytes:
    """A random legal queue (outputs never overlap their own inputs; later descriptors may read,
    overwrite or re-read earlier ones' bytes), with occasional injected errors."""
    rng = np.random.default_rng(seed)

    def floats(n, lo, hi):
        """Mostly ordinary values, with ~4% inf, NaN (both signs), denormals and huge values."""
        v = rng.uniform(lo, hi, n).astype("<f4")
        special = np.array([0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7FA00001,
                            0x00000001, 0x80400000, 0x7F7FFFFF, 0x80000000], dtype="<u4").view("<f4")
        hit = rng.random(n) < 0.04
        v[hit] = rng.choice(special, int(hit.sum()))
        return v.tobytes()

    while True:
        ddr = bytearray(rng.integers(0, 256, MEM, dtype=np.uint8).tobytes())
        # sprinkle fp32 values so scales, biases and RELU inputs are often finite, sometimes not
        for _ in range(6):
            o = 16 * int(rng.integers(0, 0xE000 // 16 - 64))
            ddr[o:o + 1024] = floats(256, -3, 3)
        for o in range(0, 0xE000, 0x1000):
            ddr[o:o + 64] = floats(16, 1e-4, 3e-2)
        for i in range(n):
            ddr[QADDR + 64 * i:QADDR + 64 * (i + 1)] = _random_desc(rng, rng.random() < inject_p, kmax)
        try:
            Machine(bytearray(ddr)).run_queue(QADDR, n)
            return bytes(ddr)
        except ContractError:
            continue


@pytest.mark.parametrize("R,C_", SIZES)
@pytest.mark.parametrize("seed", range(12))
def test_fuzz_random_queues(R, C_, seed):
    ddr = random_program(1000 * R + 100 * C_ + seed)
    both(ddr, QADDR, 8, R, C_, stall=[0, 30, 70][seed % 3], lat=[1, 3, 6][seed % 3], seed=seed + 1)


# The chunk loader (A3): NB chunk buffers reused round-robin, OT reads in flight. Small NB and OT
# with long K make every slot be reused many times and the credit limit bind; long latency makes
# the loader run ahead of the feed; stalls make the feed wait mid-tile.
LOADER = [(dict(NB=2, OT=2), 8, 8), (dict(NB=2, OT=4), 16, 16), (dict(NB=2, OT=32), 3, 5),
          (dict(NB=4, OT=2), 16, 4), (dict(NB=2, OT=8), 4, 16), (dict(NB=8, OT=16), 8, 8)]


@pytest.mark.parametrize("params,R,C_", LOADER, ids=lambda v: str(v))
@pytest.mark.parametrize("seed", range(4))
def test_fuzz_loader_buffer_reuse(params, R, C_, seed):
    ddr = random_program(9000 + 10 * R + C_ + seed, kmax=12)
    both(ddr, QADDR, 8, R, C_, stall=[0, 40, 0, 80][seed], lat=[16, 2, 1, 9][seed], seed=seed + 5,
         params=params)


# On-chip reuse: a hold buffer of HB beats keeps one operand's rows across the inner tile loop
# when K / 16 <= HB / max(R, C). Small HB makes some descriptors fit and some not; the shapes
# make both loop orders (m-inner holds W, n-inner holds A) occur.
HOLD = [(dict(HB=16), 8, 8), (dict(HB=32), 16, 16), (dict(HB=10), 3, 5), (dict(HB=32, NB=2), 16, 4),
        (dict(HB=4096, OT=4), 4, 16), (dict(HB=24), 8, 4), (dict(HB=48), 3, 16)]


def _hold_plan(d: dict, R: int, C_: int, HB: int) -> tuple[bool, bool, int]:
    """(reuse, m_inner, tiles that read the held operand) for one GEMM descriptor."""
    sk = max(R, C_)
    mt, nt = -(-d["M"] // R), -(-d["N"] // C_)
    reuse = d["K"] // 16 <= HB // sk
    m_inner = mt > nt
    return reuse, m_inner, (mt * nt - (nt if m_inner else mt)) if reuse else 0


@pytest.mark.parametrize("params,R,C_", HOLD, ids=lambda v: str(v))
@pytest.mark.parametrize("seed", range(4))
def test_fuzz_hold_buffer(params, R, C_, seed):
    ddr = random_program(7000 + 10 * R + C_ + seed, kmax=6)
    both(ddr, QADDR, 8, R, C_, stall=[0, 50, 20, 0][seed], lat=[3, 1, 12, 6][seed], seed=seed + 9,
         params=params)


@no_array
def test_fuzz_hold_buffer_coverage():
    """The hold-buffer fuzz programs contain held tiles in both loop orders, and GEMMs too large
    to hold."""
    held = {True: 0, False: 0}
    too_big = 0
    for params, R, C_ in HOLD:
        for seed in range(4):
            ddr = random_program(7000 + 10 * R + C_ + seed, kmax=6)
            st = Machine(bytearray(ddr)).run_queue(QADDR, 8)
            for i in range(st.op_idx if not st.done else 8):
                d = C.decode(bytes(ddr[QADDR + 64 * i:QADDR + 64 * (i + 1)]))
                if d["OPCODE"] not in (C.OP_GEMM,) + C.CONV_OPS:      # conv uses the same hold path
                    continue
                reuse, m_inner, n = _hold_plan(d, R, C_, params["HB"])
                held[m_inner] += n
                too_big += not reuse
    assert held[True] >= 20 and held[False] >= 20 and too_big >= 10, (held, too_big)


@no_array
@pytest.mark.parametrize("OT,lat", [(2, 16), (4, 16), (32, 16), (32, 1)])
def test_loader_reads_in_flight(OT, lat):
    """The loader keeps several reads in flight (that is the point of A3) but never more than its
    credit limit OT; one GEMM with K = 256 at 8x8 (M = 16: a GEMM with M <= 8 runs on the GEMV unit)."""
    ddr = bytearray(np.random.default_rng(4).integers(0, 256, MEM, dtype=np.uint8).tobytes())
    ddr[0x400:0x404] = np.float32(0.01).tobytes()
    ddr[QADDR:QADDR + 64] = C.encode(dict(
        MAGIC=C.MAGIC, VERSION=C.VERSION, VALID=1, OPCODE=C.OP_GEMM, CTRL=C.ctrl(C.QM_TENSOR),
        EPI=C.epi(), A_OFF=0x1000, W_OFF=0x2000, SCALE_OFF=0x400, Y_OFF=0x8000, M=16, N=16, K=256))
    _, r = both(bytes(ddr), QADDR, 1, 8, 8, lat=lat, params=dict(NB=4, OT=OT))
    flight = r.perf["max_reads_in_flight"]
    assert flight <= OT and flight >= min(OT, lat), flight


@pytest.mark.parametrize("seed", range(6))
def test_fuzz_long_queue_heavy_stalls(seed):
    ddr = random_program(77 + seed, n=24, inject_p=0.0)
    both(ddr, QADDR, 24, 8, 8, stall=85, lat=8, seed=seed + 3)


def test_fuzz_hits_errors_nans_and_hazards():
    """The fuzz programs exercise error stops (a set made with more injected errors), and the very
    programs test_fuzz_random_queues runs hold NaN/inf data and descriptors that read bytes an
    earlier descriptor of the same queue wrote."""
    errs, nan_outputs, raw_hazards = set(), 0, 0
    for seed in range(60):
        st = Machine(bytearray(random_program(5000 + seed, inject_p=0.15))).run_queue(QADDR, 8)
        errs.add(st.err_code)
    for (R, C_), seed in [(rc, s) for rc in SIZES for s in range(12)]:
        ddr = random_program(1000 * R + 100 * C_ + seed)
        g = Machine(bytearray(ddr))
        st = g.run_queue(QADDR, 8)
        words = np.frombuffer(bytes(g.ddr[:0xE000]), "<u4")
        nan_outputs += int((words == C.CANON_NAN).sum())
        written: list[tuple[int, int]] = []
        for i in range(st.op_idx if not st.done else 8):
            d = C.decode(bytes(ddr[QADDR + 64 * i:QADDR + 64 * (i + 1)]))
            reads, y = Machine.segments(d)
            if any(lo < whi and wlo < hi for seg in reads for lo, hi in seg for wlo, whi in written):
                raw_hazards += 1
            written += y
    assert len(errs - {C.E_OK}) >= 4 and nan_outputs > 0 and raw_hazards >= 30, \
        (errs, nan_outputs, raw_hazards)


# ---------------------------------------------------------------------- errors, one by one
def _single(change: dict, mem=MEM):
    rng = np.random.default_rng(3)
    ddr = bytearray(rng.integers(0, 256, mem, dtype=np.uint8).tobytes())
    ddr[0x400:0x404] = np.float32(0.01).tobytes()
    good = dict(MAGIC=C.MAGIC, VERSION=C.VERSION, OPCODE=C.OP_GEMM, CTRL=C.ctrl(C.QM_TENSOR),
                EPI=C.epi(), A_OFF=0x100, W_OFF=0x1000, Y_OFF=0x3000, SCALE_OFF=0x400,
                M=5, N=32, K=32, VALID=1)
    stale_crc = "CRC" in change
    change = {k: v for k, v in change.items() if k != "CRC"}
    raw = C.encode({**good, **change}, fix_crc=not stale_crc)
    ddr[QADDR:QADDR + 64] = raw
    return bytes(ddr)


@pytest.mark.parametrize("change,code", [
    ({}, C.E_OK), ({"MAGIC": 0x55AA}, C.E_MAGIC), ({"VERSION": 1}, C.E_VERSION),
    ({"CRC": 1}, C.E_CRC), ({"VALID": 0}, C.E_VALID), ({"OPCODE": 0x0E}, C.E_OPCODE),
    ({"CORE_ID": 1}, C.E_FIELD), ({"CTRL": C.ctrl(C.QM_TENSOR, fusion=1)}, C.E_FIELD),
    ({"OUT_SCALE": 7}, C.E_FIELD), ({"N": 24}, C.E_ALIGN), ({"LDA16": 1}, C.E_ALIGN),
    ({"CORE_ID": 1, "N": 24}, C.E_FIELD), ({"Y_OFF": 0xFFF0}, C.E_RANGE),
    ({"EPI": 0x21}, C.E_FIELD), ({"EPI": 0x40}, C.E_FIELD),
    # v0.4 activation codes 5-7 (sigmoid, softplus, exp) are legal with fusion on
    ({"EPI": 0x21, "CTRL": C.ctrl(C.QM_TENSOR, fusion=1)}, C.E_OK),
    ({"EPI": 0x23, "CTRL": C.ctrl(C.QM_TENSOR, fusion=1)}, C.E_OK),
    # v0.3 row ops (the base GEMM fields become those of a LayerNorm over A rows)
    ({"OPCODE": C.OP_ROW, "CTRL": C.ROW_LAYERNORM, "K": 0, "W_OFF": 0x3727C5AC}, C.E_OK),
    ({"OPCODE": C.OP_ROW, "CTRL": C.ROW_SOFTMAX, "K": 0, "W_OFF": 0, "SCALE_OFF": 0}, C.E_OK),
    ({"OPCODE": C.OP_ROW, "CTRL": 3, "K": 0}, C.E_OK),                                 # v0.4 L2NORM
    ({"OPCODE": C.OP_ROW, "CTRL": 5, "K": 0}, C.E_FIELD),
    ({"OPCODE": C.OP_ROW, "CTRL": C.ROW_SOFTMAX, "K": 0, "W_OFF": 0}, C.E_FIELD),     # gamma on softmax
    ({"OPCODE": C.OP_ROW, "CTRL": C.ROW_RMSNORM, "K": 16}, C.E_FIELD),
    ({"OPCODE": C.OP_ROW, "CTRL": C.ROW_RMSNORM | 4, "K": 0}, C.E_FIELD),
    ({"OPCODE": C.OP_ROW, "CTRL": C.ROW_RMSNORM, "K": 0, "BIAS_OFF": 0x500}, C.E_FIELD),
    ({"OPCODE": C.OP_ROW, "CTRL": C.ROW_RMSNORM, "K": 0, "EPI": C.epi(C.ACT_RELU)}, C.E_FIELD),
    ({"OPCODE": C.OP_ROW, "CTRL": C.ROW_RMSNORM, "K": 0, "SCALE_OFF": 0xFFC0}, C.E_RANGE),
    ({"OPCODE": C.OP_ROW, "CTRL": C.ROW_RMSNORM, "K": 0, "SCALE_OFF": 0x404}, C.E_ALIGN),
    ({"OPCODE": 0x0F}, C.E_OPCODE),
    # v0.3 conv: a 5x5x16 image (80 B pixels from A_OFF), 3x3 kernel, pad 1 -> M = 25, K = 144
    ({"OPCODE": C.OP_CONV, "M": 25, "K": 144, "GROUP_SIZE": 5, "HDR_RSV": C.conv_geom(5, 3, 3, 1, 1)}, C.E_OK),
    ({"OPCODE": C.OP_CONV, "M": 24, "K": 144, "GROUP_SIZE": 5, "HDR_RSV": C.conv_geom(5, 3, 3, 1, 1)}, C.E_FIELD),
    ({"OPCODE": C.OP_CONV_ACT, "M": 25, "K": 144, "GROUP_SIZE": 5, "HDR_RSV": C.conv_geom(5, 3, 3, 1, 1)}, C.E_FIELD),
    ({"OPCODE": C.OP_CONV, "M": 25, "K": 144, "GROUP_SIZE": 0, "HDR_RSV": C.conv_geom(5, 3, 3, 1, 1)}, C.E_FIELD),
    ({"OPCODE": C.OP_CONV, "M": 25, "K": 96, "GROUP_SIZE": 5, "HDR_RSV": C.conv_geom(5, 2, 3, 1, 1)}, C.E_FIELD),
    ({"OPCODE": C.OP_CONV, "M": 36, "K": 96, "GROUP_SIZE": 5, "HDR_RSV": C.conv_geom(5, 2, 2, 1, 1)}, C.E_ALIGN),
    ({"GROUP_SIZE": 5}, C.E_FIELD), ({"HDR_RSV": 1}, C.E_FIELD),
    ({"OPCODE": C.OP_CONV, "M": 25, "K": 144, "GROUP_SIZE": 5, "HDR_RSV": C.conv_geom(5, 3, 3, 1, 1),
      "A_OFF": 0x10000 - 400}, C.E_OK),                                    # the image ends exactly at the top
    ({"OPCODE": C.OP_CONV, "M": 25, "K": 144, "GROUP_SIZE": 5, "HDR_RSV": C.conv_geom(5, 3, 3, 1, 1),
      "A_OFF": 0x10000 - 384}, C.E_RANGE),                                 # one beat over
    ({"EPI": C.epi(C.ACT_SILU), "CTRL": C.ctrl(C.QM_TENSOR, fusion=1)}, C.E_OK),
    ({"BASE_ADDR": (1 << 64) - 0x100}, C.E_RANGE)])
def test_errors_on_rtl(change, code):
    raw = _single(change)
    st, r = both(raw, QADDR, 1)
    assert st.err_code == expect_code(raw[QADDR:QADDR + 64], code)


@pytest.mark.parametrize("tensor,fits", [("Y", True), ("Y", False), ("A", True), ("A", False),
                                         ("W", True), ("W", False)])
def test_boundary_exact_fit_and_one_beat_over(tensor, fits):
    """A tensor ending exactly at the end of DDR is legal; one beat further is E_RANGE.
    (_single: M=5, N=32, K=32 -> A 160 bytes, W 1024 bytes, Y 640 bytes of fp32.)"""
    nbytes = {"Y": 5 * 32 * 4, "A": 5 * 32, "W": 32 * 32}[tensor]
    st, _ = both(_single({f"{tensor}_OFF": MEM - nbytes + (0 if fits else 16)}), QADDR, 1)
    assert st.err_code == (C.E_OK if fits else C.E_RANGE)


@pytest.mark.parametrize("qm", [C.QM_TENSOR, C.QM_CHANNEL])
def test_cache_coherence_across_descriptors(qm):
    """A later descriptor must see bytes an earlier one wrote, even where the core may still hold
    them in a read cache. The caches hold one 16-byte beat, so the hazards are built on the beat
    the core read last: the per-tensor scale beat, and the last beat of a RELU input.

    op0 GEMM   reads scale beat 0x400            op1 RELU  reads 0x6000..0x603F (last beat 0x6030)
    op2 RELU   overwrites 0x400..0x43F            op3 RELU  overwrites 0x6000..0x603F
    op4 GEMM   reads the scale beat again        op5 RELU  reads 0x6030.. (first beat = op1's last)
    """
    rng = np.random.default_rng(9)
    ddr = bytearray(rng.integers(0, 256, MEM, dtype=np.uint8).tobytes())
    ddr[0x400:0x440] = np.full(16, 0.01, "<f4").tobytes()                 # scales
    ddr[0x6000:0x6080] = rng.uniform(-2, 2, 32).astype("<f4").tobytes()    # RELU input
    ddr[0x7000:0x7040] = np.full(16, 3.5, "<f4").tobytes()                 # new values
    base = dict(MAGIC=C.MAGIC, VERSION=C.VERSION, VALID=1)
    gemm = dict(base, OPCODE=C.OP_GEMM, CTRL=C.ctrl(qm), EPI=C.epi(), A_OFF=0x100,
                W_OFF=0x1000, SCALE_OFF=0x400, M=2, N=16, K=32)
    relu = dict(base, OPCODE=C.OP_RELU, EPI=C.epi(C.ACT_RELU), M=1, N=16)
    descs = [C.encode({**gemm, "Y_OFF": 0x3000}),
             C.encode({**relu, "A_OFF": 0x6000, "Y_OFF": 0x3800}),
             C.encode({**relu, "A_OFF": 0x7000, "Y_OFF": 0x400}),
             C.encode({**relu, "A_OFF": 0x7000, "Y_OFF": 0x6000}),
             C.encode({**gemm, "Y_OFF": 0x4000}),
             C.encode({**relu, "A_OFF": 0x6030, "Y_OFF": 0x4800})]
    for i, r in enumerate(descs):
        ddr[QADDR + 64 * i:QADDR + 64 * (i + 1)] = r
    for R, C_ in [(8, 8), (3, 5)]:
        st, r = both(bytes(ddr), QADDR, len(descs), R, C_)
        assert st.done and r.ddr[0x3000:0x3080] != r.ddr[0x4000:0x4080]   # the scales changed


def test_queue_out_of_memory_is_e_range():
    both(_single({}), MEM - 32, 1)


# ---------------------------------------------------------------------- compiled programs
@pytest.mark.parametrize("R,C_", [(8, 8), (16, 16), (5, 3)])
@pytest.mark.parametrize("variant", [dict(), dict(fused=False), dict(n_split=16), dict(m_split=3),
                                     dict(n_split=32, m_split=5, fused=False)])
def test_compiled_linear_layers(R, C_, variant):
    rng = np.random.default_rng(R * 10 + C_)
    x = rng.uniform(-2, 2, (13, 40))
    s_x = Cc.act_scale(x)
    p = Cc.Program(ddr_bytes=1 << 16)
    h = p.linear(p.input_int8(Cc.quantize_act(x, s_x), s_x), Cc.quantize_weight(rng.normal(0, .3, (50, 40))),
                 bias=rng.normal(0, .5, 50), act="relu", out="int8", out_scale=0.04, **variant)
    p.linear(h, Cc.quantize_weight(rng.normal(0, .3, (20, 64)), per_channel=False),
             bias=rng.normal(0, .5, 20), act="relu6", out="bf16", **variant)
    ddr, q, n = p.build()
    st, _ = both(bytes(ddr), q, n, R, C_, stall=25, lat=3)
    assert st.done


# ---------------------------------------------------------------------- the P1 network on RTL
@pytest.mark.parametrize("R,C_,variant", [(8, 8, dict()), (16, 16, dict(n_split=64)),
                                          (8, 4, dict(fused=False, m_split=128))])
def test_mlp_digits_on_rtl(R, C_, variant):
    """The trained MLP's whole queue: RTL DDR == golden DDR, and the RTL's own logits classify
    as well as the golden model's."""
    import mlp_digits
    m = mlp_digits.train(0)
    q = mlp_digits.calibrate(m)
    status, mach, prog, h, logits = mlp_digits.compile_and_run(m, q, m.x_test, **variant)
    ddr, qa, n = prog.build()
    st, r = both(bytes(ddr), qa, n, R, C_, stall=20, lat=3)
    rtl_logits = prog.read(Machine(r.ddr), prog.last_output)
    acc_rtl = float((rtl_logits.argmax(1) == m.y_test).mean())
    assert acc_rtl == float((logits.argmax(1) == m.y_test).mean()) >= 0.96


@no_array
def test_worst_case_accumulator_k_max():
    """I5: K = 65520 with every A and W = -128 gives acc = 16384 * 65520 = 1,073,479,680, which
    must stay exact in the INT32 accumulators (contract: |acc| < 2^31 for K <= 65535)."""
    K, N = 65520, 16
    mem = 1 << 21
    ddr = bytearray(mem)
    ddr[0x1000:0x1000 + K] = bytes([0x80]) * K
    ddr[0x20000:0x20000 + N * K] = bytes([0x80]) * (N * K)
    ddr[0x400:0x404] = np.float32(1.0).tobytes()
    ddr[0x1F0000:0x1F0040] = C.encode(dict(
        MAGIC=C.MAGIC, VERSION=C.VERSION, VALID=1, OPCODE=C.OP_GEMM, CTRL=C.ctrl(C.QM_TENSOR),
        EPI=C.epi(), A_OFF=0x1000, W_OFF=0x20000, SCALE_OFF=0x400, Y_OFF=0x800, M=1, N=N, K=K))
    st, r = both(bytes(ddr), 0x1F0000, 1, 4, 4)
    y = np.frombuffer(bytes(r.ddr[0x800:0x840]), "<f4")
    assert st.done and np.all(y == np.float32(16384 * K))


def test_fuzz_covers_row_ops():
    """The random queues contain every row-op function, with and without beta, and rows that do
    and do not fit the default hold buffer at the smallest array."""
    seen = set()
    for seed in range(40):
        ddr = random_program(6000 + seed, inject_p=0.0)
        for i in range(8):
            d = C.decode(bytes(ddr[QADDR + 64 * i:QADDR + 64 * (i + 1)]))
            if d["OPCODE"] == C.OP_ROW:
                seen.add((d["CTRL"] & 3, d["EPI"] >> 4 & 1))
    assert seen >= {(0, 0), (1, 0), (1, 1), (2, 0), (2, 1)}, seen


@no_array
def test_fuzz_covers_conv():
    """The random queues contain convolutions with padding, stride 2, 1x1 kernels, both opcodes and
    a strided image."""
    seen = set()
    for seed in range(60):
        ddr = random_program(6100 + seed, inject_p=0.0)
        for i in range(8):
            d = C.decode(bytes(ddr[QADDR + 64 * i:QADDR + 64 * (i + 1)]))
            if d["OPCODE"] in C.CONV_OPS:
                g = C.conv_params(d)
                seen |= {("pad", g["P"] > 0), ("stride2", g["S"] == 2), ("1x1", g["KH"] * g["KW"] == 1),
                         ("op", d["OPCODE"]), ("lda", d["LDA16"] != 0)}
    assert {("pad", True), ("stride2", True), ("1x1", True), ("op", C.OP_CONV), ("op", C.OP_CONV_ACT),
            ("lda", True)} <= seen, seen


@no_array
@pytest.mark.parametrize("R,C_", [(8, 8), (16, 16), (3, 5)])
@pytest.mark.parametrize("geo", [(9, 7, 3, 20, 3, 3, 1, 1), (8, 8, 16, 16, 3, 3, 2, 1), (6, 5, 20, 8, 1, 1, 1, 0),
                                 (5, 6, 8, 24, 2, 3, 1, 2), (12, 12, 32, 32, 3, 3, 1, 1)])
def test_compiled_conv(R, C_, geo):
    """Compiled convolutions (padding, stride 2, 1x1, Cin padded to 16) chained into a GELU conv."""
    H, W, Cin, Cout, KH, KW, S, P = geo
    rng = np.random.default_rng(sum(geo))
    img = rng.uniform(-1, 1, (H, W, Cin))
    s_x = Cc.act_scale(img)
    p = Cc.Program(ddr_bytes=1 << 17)
    xi = p.input_int8(Cc.quantize_act(img.reshape(H * W, Cin), s_x), s_x)
    y, OH, OW = p.conv2d(xi, H, W, rng.normal(0, .3, (Cout, KH, KW, Cin)), stride=S, pad=P,
                         bias=rng.normal(0, .2, Cout), act="relu", out="int8", out_scale=0.05)
    p.conv2d(y, OH, OW, rng.normal(0, .3, (16, 3, 3, Cout)), pad=1, act="gelu", out="bf16",
             per_channel=False, n_split=16)
    ddr, q, n = p.build()
    st, _ = both(bytes(ddr), q, n, R, C_, stall=25, lat=4, params=dict(HB=64))
    assert st.done


@pytest.mark.parametrize("R,C_,HB,VG", [(8, 8, 1024, 4), (8, 8, 16, 4), (3, 5, 10, 4), (8, 8, 40, 4),
                                        (8, 8, 1024, 1), (8, 8, 1024, 2), (8, 8, 40, 2)])
@pytest.mark.parametrize("N", [48, 96, 256])
def test_compiled_rowops(R, C_, HB, VG, N):
    """Row ops on well-conditioned data (normal, fp32 out), where the summation order shows in
    the bits: random DDR bytes span 1e-38..1e38, so their sums hardly depend on the order and the
    fuzz alone cannot pin the 16-lane order down. The HB values put rows in all three row-engine
    modes: two rows (+ gamma, beta) fit (overlapped rows), one fits (sequential passes from the
    hold buffer), none (DDR re-reads); VG 1, 2, 4 beats per hold-buffer step."""
    rng = np.random.default_rng(N + HB)
    x = rng.normal(0, 3, (16, N)).astype(np.float32)
    p = Cc.Program(ddr_bytes=1 << 17)
    xi = p.input_fp32(x)
    p.rowop(xi, "softmax")
    p.rowop(xi, "layernorm", gamma=rng.uniform(.5, 1.5, N), beta=rng.normal(0, 1, N))
    p.rowop(xi, "rmsnorm", gamma=rng.uniform(.5, 1.5, N), eps=1e-6)
    p.relu(xi, act="silu", out="bf16")
    p.rowop(p.input_fp32(x - 40), "softmax")          # all-negative rows (masked attention scores)
    ddr, q, n = p.build()
    st, _ = both(bytes(ddr), q, n, R, C_, stall=30, lat=5, params=dict(HB=HB, VG=VG))
    assert st.done


@pytest.mark.parametrize("seed", range(6))
def test_queue_end_waits_for_the_last_write(seed):
    """A queue that ends with a row op (or an activation) under heavy write backpressure: the
    core may report done only after its last beat is written."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 3, (3, 32)).astype(np.float32)
    p = Cc.Program(ddr_bytes=1 << 16)
    xi = p.input_fp32(x)
    p.rowop(xi, ["softmax", "layernorm", "rmsnorm"][seed % 3],
            **({} if seed % 3 == 0 else dict(gamma=rng.uniform(.5, 1.5, 32))))
    if seed >= 3:
        p.relu(xi, act="gelu", out="int8", out_scale=0.05)
    ddr, q, n = p.build()
    st, _ = both(bytes(ddr), q, n, 8, 8, stall=90, lat=2, seed=seed + 1)
    assert st.done


@pytest.mark.parametrize("seed", range(3))
def test_epilogue_busy_when_the_next_tile_is_fed(seed):
    """Short K (the feed of a tile takes ~20 cycles), a 2-slot output queue and heavy write
    backpressure: the epilogue engine is still launching the previous tile's rows when the next
    tile's feed ends, so the handoff must wait for it."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, (24, 16))
    s_x = Cc.act_scale(x)
    p = Cc.Program(ddr_bytes=1 << 16)
    p.linear(p.input_int8(Cc.quantize_act(x, s_x), s_x), Cc.quantize_weight(rng.normal(0, .3, (32, 16))),
             bias=rng.normal(0, .5, 32), act="relu", out="fp32")
    ddr, q, n = p.build()
    st, _ = both(bytes(ddr), q, n, 4, 4, stall=85, lat=2, seed=seed + 1, params={"OQ": 2})
    assert st.done


@pytest.mark.parametrize("seed", range(4))
def test_hold_filling_tile_waits_for_its_rows(seed):
    """R > C, n-inner reuse: the first tile of each row block reads A from DDR while copying it
    into the hold buffer. Waiting for the W columns does not imply the A rows are there (R > C:
    row R-1's chunk is issued after the last column's), so under heavy read backpressure and
    latency a feed that read the hold buffer on that tile would see rows not yet filled."""
    rng = np.random.default_rng(seed)
    M, N, K = 32, 64, 64
    x = rng.uniform(-1, 1, (M, K))
    s_x = Cc.act_scale(x)
    p = Cc.Program(ddr_bytes=1 << 16)
    p.linear(p.input_int8(Cc.quantize_act(x, s_x), s_x), Cc.quantize_weight(rng.normal(0, .2, (N, K))),
             bias=rng.normal(0, .5, N), out="fp32")
    ddr, q, n = p.build()
    st, _ = both(bytes(ddr), q, n, 16, 4, stall=85, lat=16, seed=seed + 7, params=dict(HB=64))
    assert st.done


@pytest.mark.parametrize("R,C_", [(8, 8), (16, 8)])
def test_hold_reuse_reads_each_held_row_once(R, C_):
    """Traffic, not bits: with hold-buffer reuse (m-inner here: W held), W is read once per column
    block and A once per tile, plus the descriptor and the scale/bias beats. Extra W reads (the
    loader fetching a held operand it does not need, e.g. while prefetching) show up here only."""
    rng = np.random.default_rng(R)
    M, N, K = 64, 16, 256                      # M/R row tiles > N/C column tiles: m-inner
    x = rng.uniform(-1, 1, (M, K))
    s_x = Cc.act_scale(x)
    p = Cc.Program(ddr_bytes=1 << 17)
    p.linear(p.input_int8(Cc.quantize_act(x, s_x), s_x), Cc.quantize_weight(rng.normal(0, .2, (N, K))),
             bias=rng.normal(0, .5, N), out="fp32")
    ddr, q, n = p.build()
    st, r = both(bytes(ddr), q, n, R, C_, lat=8, params=dict(HB=4096))
    assert st.done
    kc, nblk = K // 16, -(-N // C_)
    a_reads = M * kc * nblk                    # every tile streams its A rows
    w_reads = N * kc                           # each W row once (then held)
    vec_max = nblk * 2 * 2 * (C_ * 4 // 16 + 1)   # scale + bias beats, at most twice per block
    assert r.perf["rd_beats"] <= 4 * n + a_reads + w_reads + vec_max, \
        (r.perf["rd_beats"], 4 * n + a_reads + w_reads)


needs_dram = pytest.mark.skipif(not rtlsim.have_dramsim3(), reason="DRAMsim3 not built (tools/fetch_dramsim3.sh)")


@needs_dram
@pytest.mark.parametrize("seed", range(10))
def test_fuzz_on_dram_model(seed):
    """Random queues on the LPDDR4X DRAMsim3 timing model: long, variable, merged read latencies
    and posted, combined writes change no bit."""
    ddr = random_program(1000 * 8 + 100 * 8 + seed)
    both(bytes(ddr), QADDR, 8, 8, 8, dram=dict(rbuf=seed % 3 * 4, wbuf=1 + seed % 8, fabric=10 + 20 * (seed % 4)))


@needs_dram
@pytest.mark.parametrize("R,C_", [(8, 8), (16, 16)])
def test_compiled_on_dram_model(R, C_):
    """A compiled conv -> GEMM -> softmax chain on the DRAM model."""
    rng = np.random.default_rng(R)
    img = rng.uniform(-1, 1, (10, 10, 16))
    s_x = Cc.act_scale(img)
    p = Cc.Program(ddr_bytes=1 << 18)
    xi = p.input_int8(Cc.quantize_act(img.reshape(100, 16), s_x), s_x)
    y, oh, ow = p.conv2d(xi, 10, 10, rng.normal(0, .3, (32, 3, 3, 16)), pad=1, act="relu", out="int8", out_scale=0.05)
    z = p.linear(y, Cc.quantize_weight(rng.normal(0, .2, (64, 32))), out="fp32")
    p.rowop(z, "softmax")
    ddr, q, n = p.build()
    st, _ = both(bytes(ddr), q, n, R, C_, dram={})
    assert st.done


def test_structural_fp_units_in_the_core():
    """The core built with the synthesizable pipelined fp units (rtl/tm_fpu.sv) instead of their
    behavioral twins: a GELU GEMM, a SiLU activation and the three row ops, bit-exact and in the
    same number of cycles as the behavioral build (slow to build: one small configuration)."""
    rng = np.random.default_rng(77)
    x = rng.uniform(-1, 1, (6, 32))
    s_x = Cc.act_scale(x)
    p = Cc.Program(ddr_bytes=1 << 16)
    xi = p.input_int8(Cc.quantize_act(x, s_x), s_x)
    y = p.linear(xi, Cc.quantize_weight(rng.normal(0, .3, (32, 32))), bias=rng.normal(0, .2, 32),
                 act="gelu", out="fp32")
    p.relu(y, act="silu", out="bf16")
    p.rowop(y, "softmax")
    p.rowop(y, "layernorm", gamma=rng.uniform(.5, 1.5, 32), beta=rng.normal(0, 1, 32))
    p.rowop(y, "rmsnorm", gamma=rng.uniform(.5, 1.5, 32))
    ddr, q, n = p.build()
    st, rb = both(bytes(ddr), q, n, 4, 4, params=dict(HB=64))
    rs = rtlsim.run_core(bytes(ddr), q, n, R=4, C=4, params=dict(HB=64), structural=True)
    assert st.done and rs.ddr == rb.ddr and rs.cycles == rb.cycles, (rs.cycles, rb.cycles)
