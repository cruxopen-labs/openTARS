# Qwen3.5-0.8B's vision tower: reference model and lowering

The 0.8B checkpoint carries a ViT that the text lowering never touched: `model.visual` is 12 blocks
of 768 (12 heads of 64), patch 16 with temporal 2, a 2x2 spatial merge, a learned 48x48 position
table, axial 2-D RoPE (theta 1e4) and a merger that folds the 2x2 block's four 768-vectors into one
1024-wide visual token. This document covers the transformers reference for it, the lowering to the
v0.4 descriptor set, and what the two measure.

**No RTL change.** The tower runs on the ops v0.4 already has: GEMM-RC (W8A8), FDOT (NT for the
scores, NN for the value weighting), ROW_LAYERNORM, ROW_SOFTMAX, ROW_QUANT (per-token INT8), ELT
(MUL / ADD / MULADD) and ROPE.

## The interface (`tars_mini/qwen_vit.py`)

| Function | What it does |
|---|---|
| `vision_config(path=None, **over)` | the tower's config: the checkpoint's `vision_config`, else the 0.8B defaults; keyword overrides let the tests lower a much smaller tower |
| `load_vision(path)` | the checkpoint's fp32 weights under the model's own names (`patch_embed.*`, `blocks.*`, `merger.*`, `pos_embed.weight`) from the `model.visual.` keys |
| `run_reference(cfg, w, patches, grid_thw, fake_quant=…)` | the tower in transformers: the merged visual tokens, and with `blocks=True` every block's output; `fake_quant` swaps every Linear (patch embed included) for the W8A8 arithmetic the descriptors describe |
| `patchify(img, cfg)` | a normalized `[3, H, W]` image as the processor's patches `[P, K]`, spatial-merge block major, each block's patches in row-major order, flattened `(channel, temporal, row, column)` — plus its `grid_thw` |
| `position_embeddings(cfg, w, grid_thw)` | the learned table resampled bilinearly to the image's patch grid (`align_corners`), one row per patch |
| `rope_table(cfg, grid_thw)` | one row per patch: `cos[head_dim/2]` then `sin[head_dim/2]` of the axial angles `[h, w]` |
| `VisionLowering(cfg, w, patches_max=P)` | the program; `run(patches, pos, rope)` returns `[P/4, out_hidden]` after the golden model executes it, and `run(..., blocks=k)` returns the patches' hidden state after `k` blocks instead |

The split is the one the text model uses: the host normalizes the image, patchifies it, resamples
the position table and builds the rotary table; the core runs patch embed, the blocks and the merger.
One image is one program — the weights and the buffers are sized for `patches_max`, and the host's
data (patches, position rows, rotary rows, and the two score-mask rows, which follow the image's
patch count rather than the capacity) is written at build time, so one program serves any image up to
that capacity.

## What the lowering does

```
patch embed   one GEMM-RC over the patch vector (the Conv3d of kernel = stride = [2, 16, 16])
              + the resampled position rows
per block     LayerNorm -> qkv GEMM-RC -> q, k, v copied into their own rows
              -> ROPE on q and k (axial table, per head) -> q scaled by 1/sqrt(head_dim)
              -> FDOT NT over the (padded) keys -> padded scores masked to -3e38 -> ROW_SOFTMAX
              -> FDOT NN over the values -> proj GEMM-RC -> + residual
              -> LayerNorm -> fc1 GEMM-RC (tanh GELU) -> fc2 GEMM-RC -> + residual
merger        LayerNorm per patch (H wide) -> the 2x2 block's 4 x H values as one row
              -> fc1 GEMM-RC (tanh GELU) -> fc2 GEMM-RC -> [P/4, 1024]
```

Four details are worth recording, because each one is a trap in this contract:

- **Attention has no causal mask and no padding.** The patch count is padded to a multiple of 16 (a
  descriptor's `N`), the padded key and value rows are zero, and the padded scores are forced to
  `-3e38` so that `exp` underflows to `+0` and the softmax's lane folding is unchanged. The
  `ELT.MULADD` bias in this contract is an `M x N` *matrix* (read with the output's row stride), not
  a row, so the mask costs two descriptors: `MUL` by the 0/1 row, then `ADD` the `-3e38` row.
- **q, k and v are copied into their own dense rows** before the rope and the per-head views. A
  column slice of the qkv output has a uniform row stride only within a patch, so it cannot be
  viewed as `[patches x heads, head_dim]`; a copy costs one ELT and makes every later view dense.
- **One shared INT8 activation buffer** (and one row-scale row) serves every GEMM in the image.
  Allocating one per layer would grow the program by `depth x width` — 50 MB at 448x448.
- The merger's LayerNorm is **per patch**, not over the 4·H folded row: the tower applies an
  H-wide norm and then reshapes (`use_postshuffle_norm=False`). The merger's activation is
  `nn.GELU()` (the erf form) while the blocks use `gelu_pytorch_tanh`; the descriptor set has only
  the tanh form, so `run_reference(fake_quant=True)` also swaps the merger's GELU (it differs from
  erf by up to ~1e-3).

## Cost (measured, golden model)

One image, `ddr_bytes` = the whole program (weights included), 12 blocks, per-head attention:

| Image | Patches | Visual tokens | Program DDR | Descriptors | MACs | DDR read | DDR written |
|---|---|---|---|---|---|---|---|
| 64x64 | 16 | 4 | 102 MB | 944 | 1.43 G | 115 MB | 14 MB |
| 224x224 | 196 | 49 | 118 MB | 944 | 18.3 G | 372 MB | 254 MB |
| 448x448 | 784 | 196 | 171 MB | 944 | 81.3 G | 2.23 GB | 2.06 GB |

The descriptor count does not grow with the image: attention is emitted per head, not per patch, and
only the `N`/`K` fields change. The traffic does: the scores are `patches x padded_patches` per head,
so a 448x448 image writes and reads ~2 GB through the score and softmax buffers, and the reads grow
from 115 MB to 2.2 GB while the MACs grow from 1.4 G to 81 G. The design is a per-head materialized
score matrix on a core with no flash-attention path, so the score traffic is the part to attack
first if a real image ever has to be fast.

## The KV260 profile (no array): same tokens, the weights re-streamed

The board image is GEMV-only (`ARR=0`): a GEMM with any `M` and `K <= 4096` runs on the extension
unit, which holds `GB = 4` rows of A per pass and streams W again for the next pass. The tower's `K`s
are all `<= 3072` and the patch embed is emitted as a GEMM rather than a Conv, so **every descriptor
of the program is legal on that image** and the core splits `M` (the patch count) into `ceil(P/4)`
passes by itself — the descriptor count stays 944. Measured: the 0.8B program at 64x64 and at 224x224
runs to completion under the GEMV-only profile with tokens bit-identical to the full-array profile,
and the small tower at 16 patches (`M = 16`, four passes a GEMM) runs on the verilated KV260 top
(`tb_kv260`, AXI memory, scrambled pages) giving back the golden model's DDR byte for byte.

So the difference is throughput, not function — and it is all in the weight stream, because W is read
once per pass: 99 MB of weights become 357 MB at 64x64 (x3.6; the merger's `M = 4` is one pass),
4.4 GB at 224x224 (x44) and 17.5 GB at 448x448 (x177), against the full-array program's 115 MB /
372 MB / 2.23 GB of total traffic. That is why an image on the board wants the array, or a GEMV pass
deeper than four rows, before it wants anything else.

## Verification

`python -m pytest tests/test_qwen_vit.py -q` (7 tests, ~25 s with the core already built, the 0.8B one
needs the checkpoint):

- `patchify` against a loop restatement of the processor's order (and the processor itself when
  torchvision is installed);
- the rotary table against the tower's own `cos`/`sin` (recomposed `[h_w, h_w]` layout) for a 4x4 grid;
- a small tower (64 wide, 2 blocks, 8x8 image) against the W8A8 reference, **block by block** and then
  the merged tokens, at `rtol=1e-4`;
- one program serving a smaller image than its capacity (the score-mask rows are host data, written
  per image): a program built for 16 patches must return the same tokens as one built for 4;
- the real 0.8B tower at 64x64: the first block at 768 wide to 0.5% and the merged tokens at the
  reference's own noise floor. That floor is measured, not assumed: perturbing the input patches by
  1e-7 moves the W8A8 emulation's tokens by ~3% of their scale (the tower's activations reach the
  thousands, and 12 quantized blocks amplify every rounding difference). The lowering stays inside
  `4 x` that sensitivity and tracks the fp32 tower no worse than the emulation itself does;
- the small tower's program on the verilated v0.4 core: the same end status and the same DDR bytes as
  the golden model (skipped without Verilator).

A bug found by this work, in the shared reference helper: `qwen_ref._fake_quant_linear` dropped the
module's bias. No text linear in the models lowered so far has one, but every vision linear does, so
the fake-quant tower was wrong by more than the quantization error until the bias was added back.

## Cross-check on the RTL core

`tars_mini/rtlsim.run_core` runs the same program on the verilated v0.4 core (`R=8, C=8`) and its DDR
comes back byte for byte what the golden model computed: the small tower on every run of the test
above, and the real 0.8B tower at 32x32 — 944 descriptors, 4 patches, 1 visual token, 7,527,624 cycles
(37.6 ms at 200 MHz) with the tokens identical to the golden model's (`max |difference| = 0.0`). The
descriptor count is the tower's, not the image's (944 at 32x32 and at 224x224), so the cycles scale
with the patches through the `N`/`K` fields alone.

The KV260 top is GEMV-only (`ARR=0`) and runs the same program unchanged — bit-identical to the
golden model, at the cost of re-streaming the weights once per four patch rows (the section above).
