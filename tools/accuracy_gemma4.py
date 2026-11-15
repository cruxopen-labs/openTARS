"""Gemma 4 E2B quantization accuracy: the BF16 checkpoint against the NPU's W8A8 and W4A16.

    python3 tools/accuracy_gemma4.py --ckpt ~/models/gemma-4-E2B-it \
        [--prompt "..."] [--tokens 32] [--out /tmp/gemma4_accuracy.json]

One teacher-forced pass over the prompt: the reference model computes each position's logits with
the checkpoint's BF16 weights (Torch bfloat16); the quantized runs use the NPU's W8A8 / W4A16
linears and head on the same activations. Metrics over positions 1..T-1: top-1 agreement with the
BF16 logits, mean KL(BF16 || quantized), NLL / perplexity on the prompt's own next tokens, and the
max / mean absolute logit error. W4A16 uses the hardware's exact group alignment (FP32 activations
aligned to 16-bit block-floating-point mantissas), not a rounded simulation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tars_mini import contract as C                     # noqa: E402
from tars_mini import qwen_ref as Q                     # noqa: E402
from tools.export_kv260 import model_config             # noqa: E402


def head_all(final: np.ndarray, hw, fake_quant: bool, wbits: int, softcap) -> np.ndarray:
    """Every row's logits [T, vocab] through the tied head as the device computes it."""
    return np.stack([Q.head_logits(row, hw, fake_quant, softcap=softcap, wbits=wbits)
                     for row in final])


def quant_head(e: np.ndarray, wbits: int):
    hr = Q.head_rows(e.shape[0])
    if wbits == 4:
        from tars_mini.compiler import quantize_weight4
        return [quantize_weight4(e[r:r + hr].astype(np.float64)) for r in range(0, e.shape[0], hr)]
    return [Q.quant_weight(e[r:r + hr].astype(np.float64)) for r in range(0, e.shape[0], hr)]


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def metrics(ref: np.ndarray, got: np.ndarray, targets: np.ndarray) -> dict:
    p = softmax(ref)
    q = softmax(got)
    kl = np.sum(p * (np.log(np.maximum(p, 1e-45)) - np.log(np.maximum(q, 1e-45))), axis=-1)
    nll_q = -np.log(np.maximum(q[np.arange(len(targets)), targets], 1e-45))
    nll_p = -np.log(np.maximum(p[np.arange(len(targets)), targets], 1e-45))
    err = got.astype(np.float64) - ref.astype(np.float64)
    return dict(positions=int(len(targets)),
                top1_agreement=float((np.argmax(got, -1) == np.argmax(ref, -1)).mean()),
                target_agreement=float((np.argmax(got, -1) == targets).mean()),
                kl_mean=float(kl.mean()), kl_max=float(kl.max()),
                nll_mean=float(nll_q.mean()), perplexity=float(math.exp(min(nll_q.mean(), 700))),
                nll_delta_vs_bf16=float(nll_q.mean() - nll_p.mean()),
                logit_mae=float(np.abs(err).mean()), logit_max_abs=float(np.abs(err).max()))


def ple_rows(ckpt: str, cfg, ids: list[int]) -> np.ndarray:
    """The selected PLE embedding rows without holding the whole [vocab, L*ple_dim] table."""
    from safetensors import safe_open
    import torch
    want = (cfg.vocab_size_per_layer_input, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input)
    prefixes = (Q.CKPT_PREFIX, Q.CKPT_PREFIX_Q3)
    for f in sorted(Path(ckpt).glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as st:
            for k in st.keys():
                if not any(k == pre + "embed_tokens_per_layer.weight" for pre in prefixes):
                    continue
                sl = st.get_slice(k)
                assert tuple(sl.get_shape()) == want
                blk = sl[ids]
                if blk.dtype == torch.bfloat16:
                    bits = np.ascontiguousarray(blk.view(torch.uint16).numpy()).astype(np.uint32) << 16
                else:
                    bits = np.ascontiguousarray(blk.to(torch.float32).numpy()).view(np.uint32)
                return bits.view(np.float32)
    raise ValueError(f"{ckpt}: missing embed_tokens_per_layer.weight")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(Path.home() / "models" / "gemma-4-E2B-it"))
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tokens", type=int, default=32, help="prompt tokens (2 or more)")
    ap.add_argument("--out", default=None, help="write the metrics JSON here")
    args = ap.parse_args()
    t0 = time.time()
    cfg = model_config(args.ckpt)
    w = Q.load_weights(args.ckpt, cfg, embed=True, ple_embed=False)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    ids = list(tok(args.prompt, add_special_tokens=True).input_ids)[:args.tokens]
    assert len(ids) >= 2, "need at least two tokens"
    from tars_mini import llm as L
    x = L.input_embedding(cfg, w["embed_tokens.weight"][np.asarray(ids)])
    ple = L.ple_embedding(cfg, ple_rows(args.ckpt, cfg, ids)) if Q.is_gemma4(cfg) else None
    softcap = getattr(cfg, "final_logit_softcapping", None)
    print(f"{len(ids)} tokens ({time.time() - t0:.0f} s)", flush=True)
    ref = Q.run_reference(cfg, w, x, [len(ids)], fake_quant=False, ple=ple, dtype="bfloat16")
    ref_lg = head_all(ref.final[0], w["embed_tokens.weight"], False, 8, softcap)
    print(f"bf16 reference ({time.time() - t0:.0f} s)", flush=True)
    res = dict(model=Path(args.ckpt).name, prompt=args.prompt, tokens=len(ids), variants={})
    for name, wbits in (("w8a8", 8), ("w4a16", 4)):
        run = Q.run_reference(cfg, w, x, [len(ids)], fake_quant=True, ple=ple, wbits=wbits)
        lg = head_all(run.final[0], quant_head(w["embed_tokens.weight"], wbits), True, wbits, softcap)
        res["variants"][name] = metrics(ref_lg[1:], lg[1:], np.asarray(ids[1:]))
        print(f"{name}: {json.dumps(res['variants'][name], indent=1)} ({time.time() - t0:.0f} s)",
              flush=True)
        del run, lg
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
