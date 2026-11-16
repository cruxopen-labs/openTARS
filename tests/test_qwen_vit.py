"""The Qwen3.5-0.8B vision tower: the transformers reference and the lowering to v0.4 ops.

The tower is the same structure at every size, so the tests lower a small one (patch 4, 64 wide,
2 blocks) for speed, run that program on the verilated core, and — when the checkpoint is on this
machine — check the real 0.8B tower.
"""
from pathlib import Path

import numpy as np
import pytest

from tars_mini import qwen_vit as V
from tars_mini import contract as C
from tars_mini import rtlsim
from tars_mini.golden import Machine

VIT_SMALL = dict(depth=2, hidden_size=64, intermediate_size=256, num_heads=2, patch_size=4,
                 spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=48,
                 num_position_embeddings=36)
CKPT = "/home/jiang/models/Qwen3.5-0.8B"
needs_ckpt = pytest.mark.skipif(not Path(CKPT).exists(), reason="needs the 0.8B checkpoint")


@pytest.fixture(scope="module")
def small():
    cfg = V.vision_config(**VIT_SMALL)
    return cfg, V.random_vision(cfg, seed=0)


def _image(size: int, seed: int = 3):
    """A normalized (0.5 / 0.5) image with structure: smooth gradients plus noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    base = (np.sin(xx / 3.0) + np.cos(yy / 5.0))[:, :, None] / 2.0 + rng.normal(0, 0.3, (size, size, 3))
    return np.ascontiguousarray(base.transpose(2, 0, 1), np.float32).astype(np.float32)


def test_patchify_is_the_processors_order(small):
    """The patches and their order are Qwen2VLImageProcessorFast's: spatial-merge blocks major,
    then (in-row, in-column) inside a block, flattened as (channel, temporal, row, column).
    Restated here with explicit loops; the processor's own patchify is used when torchvision is
    installed."""
    cfg, _ = small
    img = _image(8)
    got, grid = V.patchify(img, cfg)
    ps, ms, tp, c = cfg.patch_size, cfg.spatial_merge_size, cfg.temporal_patch_size, cfg.in_channels
    gh, gw = 8 // ps, 8 // ps
    want = []
    for br in range(gh // ms):
        for bc in range(gw // ms):
            for ir in range(ms):
                for ic in range(ms):
                    row, col = br * ms + ir, bc * ms + ic
                    for ch in range(c):
                        for _ in range(tp):
                            for r in range(ps):
                                for cc in range(ps):
                                    want.append(img[ch, row * ps + r, col * ps + cc])
    assert grid.tolist() == [[1, gh, gw]]
    np.testing.assert_array_equal(got.reshape(-1), np.array(want, np.float32))
    try:
        import torch
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    except Exception:                      # no torchvision: the loop restatement stands alone
        return
    ref = Qwen2VLImageProcessor.patchify(torch.from_numpy(img[None]), ps, ms, tp)[0].numpy()
    np.testing.assert_array_equal(got, ref.reshape(*got.shape))


def test_rope_table_is_the_axial_rotation(small):
    """The table row per patch is cos[rot/2] then sin[rot/2] of [h (rot/4), w (rot/4)]: applying
    the contract's rope (numerics.rope_rows) reproduces the tower's rotate_half form."""
    from tars_mini import numerics as F
    import torch
    from transformers.vision_utils import get_vision_position_ids
    cfg, _ = small
    grid = np.array([[1, 4, 4]])
    hd = V.head_dim(cfg)
    tab = V.rope_table(cfg, grid)
    assert tab.shape == (16, hd)
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (16 * cfg.num_heads, hd)).astype(np.float32)
    y = F.rope_rows(x, tab, cfg.num_heads, hd)      # one table row per patch (num_heads rows each)
    sd = hd // 2
    pos = get_vision_position_ids(torch.as_tensor(grid), cfg.spatial_merge_size).float()
    inv = 1.0 / (cfg.rope_parameters["rope_theta"]
                 ** (torch.arange(0, sd, 2, dtype=torch.float) / sd))
    ang = torch.cat([pos[:, 0:1] * inv, pos[:, 1:2] * inv], dim=-1)       # [P, hd/2]: h then w
    cos = ang.cos().numpy().repeat(cfg.num_heads, axis=0)
    sin = ang.sin().numpy().repeat(cfg.num_heads, axis=0)
    want = np.empty_like(x)                                               # rotate_half form
    want[:, :sd] = x[:, :sd] * cos - x[:, sd:] * sin
    want[:, sd:] = x[:, sd:] * cos + x[:, :sd] * sin
    np.testing.assert_allclose(y, want, rtol=1e-6, atol=1e-6)


def test_lowering_matches_the_w8a8_reference(small):
    """The golden model's visual tokens against the transformers tower with every linear
    emulated as W8A8 (the arithmetic the descriptors describe)."""
    cfg, w = small
    img = _image(8)
    px, grid = V.patchify(img, cfg)
    pos, rope = V.position_embeddings(cfg, w, grid), V.rope_table(cfg, grid)
    ref = V.run_reference(cfg, w, px, grid, fake_quant=True, blocks=True)
    low = V.VisionLowering(cfg, w, patches_max=len(px))
    got = low.run(px, pos, rope)
    assert got.shape == ref["merged"].shape == (len(px) // 4, cfg.out_hidden_size)
    scale = max(np.abs(ref["merged"]).max(), np.abs(ref["last"]).max())
    for k, want in enumerate(ref["blocks"], 1):                    # every block, then the merger
        np.testing.assert_allclose(low.run(px, pos, rope, blocks=k), want,
                                   rtol=1e-4, atol=1e-5 * max(scale, 1.0))
    np.testing.assert_allclose(got, ref["merged"], rtol=1e-4, atol=1e-5 * max(scale, 1.0))


@needs_ckpt
def test_real_tower_matches_the_w8a8_reference():
    """The real 0.8B tower, 64x64 (16 patches -> 4 visual tokens): the first block at 768 wide and
    the merged tokens against the W8A8 emulation.

    The end-to-end tolerance is the reference's own sensitivity: its activations reach the
    thousands, so 12 quantized blocks turn a 1e-7 input change into a ~3% change of the merged
    tokens. The lowering has to sit at that floor, and track the fp32 tower no worse than the
    emulation does.
    """
    cfg = V.vision_config(CKPT)
    w = V.load_vision(CKPT)
    img = _image(64, seed=5)
    px, grid = V.patchify(img, cfg)
    pos, rope = V.position_embeddings(cfg, w, grid), V.rope_table(cfg, grid)
    ref = V.run_reference(cfg, w, px, grid, fake_quant=True, blocks=True)
    fp32 = V.run_reference(cfg, w, px, grid)
    px2 = (px + np.random.default_rng(0).normal(0, 1e-7, px.shape).astype(np.float32)).astype(np.float32)
    floor = np.abs(V.run_reference(cfg, w, px2, grid, fake_quant=True)["merged"] - ref["merged"]).max()
    low = V.VisionLowering(cfg, w, patches_max=len(px))
    got = low.run(px, pos, rope)
    assert got.shape == ref["merged"].shape == (4, cfg.out_hidden_size)
    first = low.run(px, pos, rope, blocks=1)
    assert np.abs(first - ref["blocks"][0]).max() <= 5e-3 * np.abs(ref["blocks"][0]).max()
    rms = lambda a: float(np.sqrt((np.asarray(a) ** 2).mean()))
    assert rms(got - fp32["merged"]) <= 1.5 * rms(ref["merged"] - fp32["merged"]), \
        "the lowering must track the fp32 tower as closely as the W8A8 emulation does"
    assert np.abs(got - ref["merged"]).max() <= 4 * max(floor, 1e-9) + 1e-5 * np.abs(ref["merged"]).max()


def test_one_program_serves_a_smaller_image(small):
    """The score-mask rows follow the image, not the program's capacity: a program sized for 16
    patches gives an 8x8 (4-patch) image the same tokens as one built for 4 patches."""
    cfg, w = small
    px, grid = V.patchify(_image(8), cfg)
    pos, rope = V.position_embeddings(cfg, w, grid), V.rope_table(cfg, grid)
    got = V.VisionLowering(cfg, w, patches_max=4).run(px, pos, rope)
    np.testing.assert_array_equal(V.VisionLowering(cfg, w, patches_max=16).run(px, pos, rope), got)
    ref = V.run_reference(cfg, w, px, grid, fake_quant=True)["merged"]
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5 * max(float(np.abs(ref).max()), 1.0))


def test_the_gemv_only_profile_gives_the_same_tokens(small):
    """The board image has no array: a GEMM keeps any M and K <= 4096 and the core runs it in passes
    of GB = 4 rows. Same descriptors, same tokens — the cost is the weight stream, not the result."""
    cfg, w = small
    px, grid = V.patchify(_image(16), cfg)                  # 16 patches: M = 16, four passes a GEMM
    pos, rope = V.position_embeddings(cfg, w, grid), V.rope_table(cfg, grid)
    low = V.VisionLowering(cfg, w, patches_max=len(px), ddr_bytes=1 << 23)
    got = low.run(px, pos, rope)                            # the full-array profile
    ddr, q, cnt = low.p.build()
    assert 16 in [C.decode(raw)["M"] for raw in low.p.descs if C.decode(raw)["OPCODE"] == C.OP_GEMM]
    a = bytearray(ddr)
    st = Machine(a, gemv_only=True).run_queue(q, cnt)
    assert st.done and st.err_code == 0, (st.err_code, st.op_idx)
    out = low.out.rowv(0, len(px) // 4)
    base = low.p.base + out.off
    prof = np.stack([np.frombuffer(bytes(a[base + r * out.ld: base + r * out.ld + 4 * out.cols]), "<f4")
                     for r in range(out.rows)])
    np.testing.assert_array_equal(prof, got)


@pytest.mark.skipif(not rtlsim.have_verilator(), reason="needs Verilator")
def test_small_tower_on_the_rtl_core(small):
    """The lowered program on the verilated v0.4 core: the same end status and the same DDR bytes
    as the golden model — the tower's own combination of ROPE, FDOT NT and NN, ROW_SOFTMAX,
    ROW_QUANT and in-place ELT."""
    cfg, w = small
    img = _image(8)
    px, grid = V.patchify(img, cfg)
    pos, rope = V.position_embeddings(cfg, w, grid), V.rope_table(cfg, grid)
    low = V.VisionLowering(cfg, w, patches_max=len(px), ddr_bytes=1 << 23)
    low.run(px, pos, rope)
    ddr, q, cnt = low.p.build()
    want = bytearray(ddr)                                    # the golden model's own DDR
    st = Machine(want).run_queue(q, cnt)
    r = rtlsim.run_core(bytes(ddr), q, cnt, R=8, C=8)
    assert (r.done, r.err_code, r.op_idx) == (st.done, st.err_code, st.op_idx), \
        f"rtl {(r.done, r.err_code, r.op_idx)} golden {(st.done, st.err_code, st.op_idx)}"
    assert r.ddr == bytes(want), "the core's DDR differs from the golden model's"
