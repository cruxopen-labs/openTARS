"""W4A16 contract, quantization, activation alignment, memory extents and Golden arithmetic."""
from __future__ import annotations

import numpy as np
import pytest

from tars_mini import contract as C
from tars_mini import numerics as F
from tars_mini.compiler import quantize_weight4
from tars_mini.golden import Machine
from tars_mini.llm import Dev, QLin, V

F32 = np.float32
QA = 0x7000


def _desc(**f):
    return C.encode({"MAGIC": C.MAGIC, "VERSION": C.VERSION, "VALID": 1, "A_OFF": 0x100,
                     "Y_OFF": 0x4000, "W_OFF": 0x1000, "M": 2, "N": 16, **f})


def _w4(**f):
    base = dict(OPCODE=C.OP_GEMM, CTRL=C.ctrl(C.QM_GROUP, a_dtype=C.DT_FP32, w_dtype=C.DT_INT4),
                K=256, GROUP_SIZE=C.W4_GROUP, SCALE_OFF=0x400)
    base.update(f)
    return base


def test_weight4_group_quantization_and_nibble_order():
    w = np.zeros((16, 256), F32)
    w[0, :8] = [-7, -3.5, -0.5, 0, 0.5, 3.5, 7, 1]
    w[1, 128:130] = [2, -2]
    qw = quantize_weight4(w)
    assert qw.q.shape == (16, 128) and qw.scale.shape == (16, 2)
    assert qw.q.dtype == np.uint8 and qw.scale.dtype == np.uint16
    assert qw.scale[0, 0] == np.float16(1).view(np.uint16)
    np.testing.assert_array_equal(qw.q[0, :4], [0xC9, 0x00, 0x40, 0x17])
    assert qw.scale[1, 1] == np.float16(2 / 7).view(np.uint16)
    assert (qw.q[1, 64] & 15) == 7 and (qw.q[1, 64] >> 4) == 9
    # Quantization uses the rounded binary16 scale, then saturates to the symmetric [-7, 7] range.
    s = qw.scale[1, 1].view(np.float16).astype(F32)
    assert np.array_equal((qw.q[1, 64] & 15), np.uint8(int(np.rint(2 / s)) & 15))
    with pytest.raises(ValueError):
        quantize_weight4(np.zeros((16, 64), F32))


def test_alignment_reference_and_extreme_exponents():
    # Values around the 15-bit boundary, exact halfway cases, zeros and subnormals.
    x = np.zeros((4, 128), F32)
    x[0, :10] = [1.0, -1.0, 0.5, -0.5, 2 ** -20, -(2 ** -20), 2 ** -30, -(2 ** -30),
                  2 ** -149, -(2 ** -149)]
    x[1, :8] = [2 ** 100, -(2 ** 100), 2 ** 90, -(2 ** 90), 1.0, -1.0, 2 ** 80, -(2 ** 80)]
    x[2, :2] = [2 ** -120, -(2 ** -125)]
    x[3, :2] = [3.0, -3.0]
    q, s = F.align_w4_groups(x)
    assert q.shape == x.shape and q.dtype == np.int16 and s.shape == (4, 1)
    assert s[0, 0] == np.float32(2 ** -13)
    assert q[0, 0] == 8192 and q[0, 1] == -8192
    assert q[0, 2] == 4096 and q[0, 3] == -4096
    assert (q[0, 4:10] == 0).all()
    assert s[3, 0] == np.float32(2 ** -12) and q[3, 0] == 12288 and q[3, 1] == -12288
    assert s[1, 0] == np.float32(2 ** 87)
    assert q[1, 0] == 8192 and q[1, 2] == 8 and q[1, 4] == 0
    assert s[2, 0] == np.float32(2 ** -126)  # emax_eff = 14 -> biased exponent 1
    assert q[2, 0] == 64 and q[2, 1] == -2
    zq, zs = F.align_w4_groups(np.zeros((1, 128), F32))
    assert (zq == 0).all() and zs[0, 0] == 0


def test_alignment_rne_independently():
    rng = np.random.default_rng(10)
    bits = rng.integers(1, 254, (3, 256), dtype=np.uint32) << 23
    bits |= rng.integers(0, 1 << 23, bits.shape, dtype=np.uint32)
    bits |= rng.integers(0, 2, bits.shape, dtype=np.uint32) << 31
    x = bits.view(F32)
    q, s = F.align_w4_groups(x)
    # Independent scalar implementation of the specified bit sequence, including RNE ties.
    for m in range(3):
        for g in range(2):
            bb = bits[m, g * 128:(g + 1) * 128]
            ee = (bb >> 23) & 255
            emax = max(14, int(ee.max()))
            assert s[m, g].view(np.uint32) == np.uint32((emax - 13) << 23)
            for j, b in enumerate(bb):
                e = int((b >> 23) & 255)
                sh = 10 + emax - e
                u = (1 << 23) | int(b & 0x7FFFFF)
                if sh >= 25:
                    want = 0
                elif sh == 0:
                    want = u
                else:
                    base = u >> sh
                    rem = u & ((1 << sh) - 1)
                    want = base + int(rem > (1 << (sh - 1)) or
                                      (rem == (1 << (sh - 1)) and (base & 1)))
                if b >> 31:
                    want = -want
                assert int(q[m, g * 128 + j]) == want, (m, g, j)


def test_alignment_nan_and_infinity_rule():
    x = np.zeros((3, 128), F32)
    x[0, 0] = np.nan
    x[0, 1] = np.inf
    x[1, 0] = np.inf
    x[1, 1] = -np.inf
    x[1, 2] = 3
    x[2, 0] = -np.inf
    q, s = F.align_w4_groups(x)
    assert s[0, 0].view(np.uint32) == C.CANON_NAN and (q[0] == 0).all()
    assert np.isposinf(s[1, 0]) and q[1, 0] == 16384 and q[1, 1] == -16384
    assert (q[1, 2:] == 0).all()
    assert np.isposinf(s[2, 0]) and q[2, 0] == -16384


@pytest.mark.parametrize("adt", [C.DT_FP32, C.DT_BF16])
@pytest.mark.parametrize("K", [256, 512, 4096])
def test_golden_w4_fp32_bf16_strides_bias_activation(adt, K):
    rng = np.random.default_rng(100 + adt + K)
    M, N = 3, 32
    p = Dev(ddr_bytes=1 << 22)
    if adt == C.DT_FP32:
        ax = rng.normal(0, 2, (M, K + 128)).astype(F32)
        av0 = p.const(ax)
        av = V(av0.off + 4 * 128, M, K, av0.ld)
    else:
        ax = rng.normal(0, 2, (M, K + 128)).astype(F32)
        ab = F.to_bf16_bits(ax).astype(np.uint16)
        av0 = V(p.put(ab, "<u2"), M, K + 128, 2 * (K + 128), C.OUT_BF16)
        av = V(av0.off + 2 * 128, M, K, av0.ld, C.OUT_BF16)
        ax = F.bf16_bits_to_f32(ab)
    w = rng.normal(0, 0.3, (N, K)).astype(F32)
    qw = quantize_weight4(w)
    lin = QLin(p.put(qw.q, "u1"), p.put(qw.scale, "<u2"), p.put(np.ones(N, F32)), N, K, 4)
    y = p.mat(M, N)
    p.gemm_w4(av, lin, y, act=C.ACT_SOFTPLUS)
    ddr, qaddr, n = p.build()
    mach = Machine(bytearray(ddr), gemv_only=True)
    st = mach.run_queue(qaddr, n)
    assert st.done
    got = np.frombuffer(mach.read(y.off, 4 * N * M), "<f4").reshape(M, N)
    aq, asc = F.align_w4_groups(ax[:, 128:])
    lo = (qw.q & 15).astype(np.int16)
    hi = (qw.q >> 4).astype(np.int16)
    lo, hi = np.where(lo >= 8, lo - 16, lo), np.where(hi >= 8, hi - 16, hi)
    wq = np.stack((lo, hi), 2).reshape(N, K)
    ws = qw.scale.view(np.float16).astype(F32)
    acc = np.zeros((M, N, 4), F32)
    for g in range(K // 128):
        dot = aq[:, g * 128:(g + 1) * 128].astype(np.int64) @ wq[:, g * 128:(g + 1) * 128].astype(np.int64).T
        term = F.fmul(F.fmul(F.i2f(dot), ws[None, :, g]), asc[:, None, g])
        acc[:, :, g % 4] = F.fadd(acc[:, :, g % 4], term)
    ref = F.softplus_f(F.fadd(F.fadd(F.fadd(acc[:, :, 0], acc[:, :, 1]),
                                     F.fadd(acc[:, :, 2], acc[:, :, 3])), F32(1)))
    assert got.tobytes() == ref.tobytes()


def test_golden_w4_group_accumulation_and_special_values():
    M, N, K = 1, 16, 512
    p = Dev(ddr_bytes=1 << 18)
    x = np.zeros((M, K), F32)
    x[0, 0] = np.inf
    av = p.const(x)
    qw = quantize_weight4(np.ones((N, K), F32))
    lin = QLin(p.put(qw.q, "u1"), p.put(qw.scale, "<u2"), None, N, K, 4)
    y = p.mat(M, N)
    p.gemm_w4(av, lin, y)
    ddr, qaddr, n = p.build()
    mach = Machine(bytearray(ddr))
    assert mach.run_queue(qaddr, n).done
    got = np.frombuffer(mach.read(y.off, 4 * N), "<f4")
    assert np.isposinf(got).all()


@pytest.mark.parametrize("fields,code", [
    (_w4(), C.E_OK),
    (_w4(CTRL=C.ctrl(C.QM_GROUP, a_dtype=C.DT_BF16, w_dtype=C.DT_INT4)), C.E_OK),
    (_w4(CTRL=C.ctrl(C.QM_GROUP, a_dtype=C.DT_INT8, w_dtype=C.DT_INT4)), C.E_FIELD),
    (_w4(CTRL=C.ctrl(C.QM_CHANNEL, a_dtype=C.DT_FP32, w_dtype=C.DT_INT4)), C.E_FIELD),
    (_w4(CTRL=C.ctrl(C.QM_GROUP, a_dtype=C.DT_FP32, w_dtype=C.DT_INT4, symmetric=0)), C.E_FIELD),
    (_w4(HDR_RSV=1), C.E_FIELD),
    (_w4(GROUP_SIZE=0), C.E_FIELD),
    (_w4(GROUP_SIZE=64), C.E_FIELD),
    (_w4(K=64), C.E_ALIGN),
    (_w4(K=384 - 128 + 64), C.E_ALIGN),
    (_w4(LDA16=16), C.E_ALIGN),  # 256 bytes < one FP32 K=256 row (1024)
    (_w4(A_OFF=0x104), C.E_ALIGN),
    (_w4(CTRL=C.ctrl(C.QM_GROUP), GROUP_SIZE=128), C.E_FIELD),  # legacy QM3 rejects GROUP_SIZE
])
def test_w4_rules(fields, code):
    ddr = bytearray(1 << 16)
    ddr[QA:QA + 64] = _desc(**fields)
    st = Machine(ddr, gemv_only=True).run_queue(QA, 1)
    assert st.err_code == code, (fields, st)


def test_w4_ranges_and_segments():
    f = _w4(M=3, N=32, K=256)
    d = C.decode(_desc(**f))
    reads, y = Machine.segments(d)
    assert reads[1] == [(d["W_OFF"], d["W_OFF"] + 32 * 128)]
    assert reads[2] == [(d["SCALE_OFF"], d["SCALE_OFF"] + 32 * 2 * 2)]
    assert len(reads) == 3  # no legacy row-scale read
    assert Machine.strides(d)[2] == 4 * 256
    for field, size in (("W_OFF", 32 * 128), ("SCALE_OFF", 32 * 2 * 2)):
        for over in (False, True):
            ff = _w4(M=3, N=32, K=256)
            # W starts stay 128-byte aligned (the wide W beats), scale rows need 16
            ff[field] = (1 << 16) - size + ((128 if field == "W_OFF" else 16) if over else 0)
            ddr = bytearray(1 << 16)
            ddr[QA:QA + 64] = _desc(**ff)
            st = Machine(ddr).run_queue(QA, 1)
            assert st.err_code == (C.E_RANGE if over else C.E_OK), (field, over, st)


def test_w4_lowering_removes_row_quant_and_slices_scales():
    from tars_mini import qwen_ref as Q
    from tars_mini.llm import QwenLowering
    cfg = Q.make_config(dict(Q.CFG_G4_SMALL, hidden_size=256, intermediate_size=512,
                             hidden_size_per_layer_input=256),
                        ["sliding_attention", "full_attention"], vocab=1024)
    w = Q.random_weights(cfg, embed=True)
    low = QwenLowering(cfg, w, max_tokens=4, ddr_bytes=1 << 23, head=True, queue_cap=2048,
                       scratch_bytes=1 << 20, keep_layers=False, wbits=4)
    low.input(low.embed([1, 2]), ple=low.embed_ple([1, 2]))
    low.chunk(0, 2)
    ops = [C.decode(r) for r in low.p.descs]
    gemm = [d for d in ops if d["OPCODE"] == C.OP_GEMM]
    assert gemm and not any(d["OPCODE"] == C.OP_ROW and d["CTRL"] & 7 == C.ROW_QUANT for d in ops)
    assert all(d["CTRL"] >> 8 & 15 == C.DT_INT4 and d["GROUP_SIZE"] == 128 and d["HDR_RSV"] == 0
               for d in gemm)
    assert all(d["K"] >= 2 * C.W4_GROUP and d["K"] % (2 * C.W4_GROUP) == 0 for d in gemm)


@pytest.mark.parametrize("chunks", [[4, 1, 1], [6, 1]])
def test_gemma4_w4_model_greedy_matches_reference(chunks):
    """Whole Gemma 4 lowering with W4A16 linears and head: greedy tokens match the exact FP64 W4
    reference; logits allow the coarser 4-bit weights to amplify the usual FP32 model differences."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tars_mini import qwen_ref as Q
    from tars_mini.llm import QwenLowering
    cfg = Q.make_config(dict(Q.CFG_G4_SMALL, hidden_size_per_layer_input=256),
                        ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
                        vocab=4096)
    w = Q.random_weights(cfg, embed=True)
    low = QwenLowering(cfg, w, max_tokens=16, ddr_bytes=1 << 26, head=True, queue_cap=4096,
                       keep_layers=False, wbits=4)

    def run(ddr, q, n):
        m = Machine(ddr, gemv_only=True)
        assert m.run_queue(q, n).done
        return m.ddr

    prompt = [5, 77, 901, 12, 3000, 2][:chunks[0]]
    ids, logits, _ = low.generate(run, prompt, len(chunks) - 1)
    x = low.embed(ids[:-1])
    ple = low.embed_ple(ids[:-1])
    ref = Q.run_reference(cfg, w, x, [len(prompt)] + [1] * (len(chunks) - 1),
                          fake_quant=True, head=True, ple=ple, wbits=4)
    assert len(ref.logits) == len(logits)
    for i, (g, r) in enumerate(zip(logits, ref.logits)):
        assert np.abs(g - r).max() <= 1e-3 * np.abs(r).max(), i
        assert int(np.argmax(r)) == ids[len(prompt) + i]
