"""Export Qwen3.5-0.8B's vision tower for the KV260 (PC side).

    python3 tools/prep_vision_kv260.py --ckpt ~/models/Qwen3.5-0.8B --image catsvsdogs.jpg \
        --out build/data_vit

Writes `program.bin` (one v0.4 program image: the W8A8 weights, the buffers and the descriptor
queue, the image's patches / position rows / rotary rows already in it), `meta.json` (the queue's
address and length, the output rows) and `tokens_golden.npy` (what the golden model computes, for
the board to compare against). The image is preprocessed the way Qwen3VLProcessor does it: the
`smart_resize` of `Qwen2VLImageProcessorFast` (sides on the 32-pixel grid, 65536..16777216 pixels),
bicubic resize, and `(x/255 - 0.5) / 0.5`.

The program is legal on the board's GEMV-only image: every K is <= 4096 and the patch embed is a
GEMM, so the core runs it in passes of GB rows; see docs/qwen_vit/README.md.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tars_mini import qwen_vit as V  # noqa: E402

HPAGE = 2 << 20


def smart_resize(h: int, w: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Qwen2VLImageProcessorFast.smart_resize: the sides on the patch*merge grid."""
    if h < factor or w < factor:
        raise ValueError(f"the image's sides must be at least {factor} pixels, got {h}x{w}")
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return h_bar, w_bar


def load_image(path: str | Path, cfg, pre: dict) -> tuple[np.ndarray, tuple[int, int]]:
    """[3, H, W] float32, normalized, resized as the processor does; plus the (height, width) used."""
    from PIL import Image
    factor = cfg.patch_size * cfg.spatial_merge_size
    im = Image.open(path).convert("RGB")
    w0, h0 = im.size
    h, w = smart_resize(h0, w0, factor, pre["shortest_edge"], pre["longest_edge"])
    x = torch.from_numpy(np.asarray(im, np.uint8).astype(np.float32)).permute(2, 0, 1)[None]
    x = torch.nn.functional.interpolate(x, size=(h, w), mode="bicubic", align_corners=False, antialias=True)
    x = (x[0] / 255.0 - torch.tensor(pre["mean"])[:, None, None]) / torch.tensor(pre["std"])[:, None, None]
    return np.ascontiguousarray(x.numpy(), np.float32), (h, w)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/jiang/models/Qwen3.5-0.8B")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="build/data_vit")
    args = ap.parse_args()

    ck = Path(args.ckpt)
    pre = json.loads((ck / "preprocessor_config.json").read_text())
    images = {"mean": pre.get("image_mean", [0.5, 0.5, 0.5]), "std": pre.get("image_std", [0.5, 0.5, 0.5]),
              "shortest_edge": pre["size"]["shortest_edge"], "longest_edge": pre["size"]["longest_edge"]}
    cfg = V.vision_config(ck)
    w = V.load_vision(ck)
    img, (h, wd) = load_image(args.image, cfg, images)
    px, grid = V.patchify(img, cfg)
    pos, rope = V.position_embeddings(cfg, w, grid), V.rope_table(cfg, grid)
    n = len(px)
    print(f"{args.image}: {wd}x{h} -> {n} patches ({grid.tolist()}), {n // cfg.spatial_merge_size ** 2} tokens")

    low = V.VisionLowering(cfg, w, patches_max=n, ddr_bytes=1 << 28)
    tokens = low.run(px, pos, rope)                               # the golden model
    ddr, q, cnt = low.p.build()
    size = -(-(q + 64 * cnt) // HPAGE) * HPAGE                    # the queue sits above the buffers
    assert size <= len(ddr), (size, len(ddr))
    out = low.out.rowv(0, n // cfg.spatial_merge_size ** 2)

    ref = V.run_reference(cfg, w, px, grid, fake_quant=True)["merged"]
    rel = float(np.abs(tokens - ref).max() / max(np.abs(ref).max(), 1e-30))
    print(f"program {size >> 20} MiB, {cnt} descriptors, queue at {q:#x}; "
          f"golden vs the W8A8 reference: {100 * rel:.2f}% of scale")

    outd = Path(args.out)
    outd.mkdir(parents=True, exist_ok=True)
    (outd / "program.bin").write_bytes(bytes(ddr[:size]))
    np.save(outd / "tokens_golden.npy", tokens)
    meta = dict(image=str(args.image), height=h, width=wd, patches=n, grid=grid.tolist(),
                tokens=int(tokens.shape[0]), hidden=int(tokens.shape[1]), descs=cnt, queue_addr=q,
                ddr_bytes=size, out_off=out.off, out_ld=out.ld, out_rows=out.rows, out_cols=out.cols,
                out_base=low.p.base)
    (outd / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {outd}/program.bin, meta.json, tokens_golden.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
