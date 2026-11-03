"""Reference for the v0.4 end-to-end target: one Qwen3.5 linear-attention (Gated DeltaNet) block and
one full-attention block, run by the transformers implementation (Qwen3_5TextModel) in fp32.

The weights are random with the 0.8B model's dimensions, or a real checkpoint's (load_weights:
Qwen/Qwen3.5-0.8B from Hugging Face has exactly these names and shapes). Two
references are produced for the same weights and inputs:

  float   - the model as written, fp32 throughout
  fq      - the same model with every linear layer replaced by what TARS-mini computes:
            per-token dynamic INT8 activations (amax / 127), per-channel INT8 weights, exact
            integer products, then the row and column scales (W8A8)

`fq` checks the lowering (the device must match it up to fp32 rounding order); `float` measures
what the quantization costs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# The 0.8B text config (Qwen3.5-0.8B config.json, text_config): 24 layers, 3 LA : 1 FA, tied
# embeddings, vocabulary 248320. make_config picks the layers (default: one LA then one FA) and
# the vocabulary (default 16: the block tests need no head)
VOCAB_08B = 248320
LAYERS_08B = (["linear_attention"] * 3 + ["full_attention"]) * 6
CFG_08B = dict(
    hidden_size=1024, intermediate_size=3584, num_attention_heads=8, num_key_value_heads=2,
    head_dim=256, linear_conv_kernel_dim=4, linear_key_head_dim=128, linear_value_head_dim=128,
    linear_num_key_heads=16, linear_num_value_heads=16, rms_norm_eps=1e-6, vocab_size=16,
    max_position_embeddings=262144, attn_output_gate=True,
    rope_parameters=dict(rope_type="default", rope_theta=10000000.0, partial_rotary_factor=0.25,
                         mrope_section=[11, 11, 10], mrope_interleaved=True),
)
# A small config with the same structure, for RTL runs
# (16 linear-attention heads as the real model; head dims kept multiples of 16 and the rotary
# width a multiple of 32, as the contract needs)
CFG_SMALL = dict(CFG_08B, hidden_size=256, intermediate_size=512, num_attention_heads=4,
                 num_key_value_heads=2, head_dim=128, linear_key_head_dim=16, linear_value_head_dim=16,
                 linear_num_key_heads=16, linear_num_value_heads=16)


def _torch():
    import torch                                     # noqa: F401  (optional dependency)
    return torch


def make_config(base: dict, layer_types: list[str] | None = None, vocab: int | None = None):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    lt = list(layer_types or ["linear_attention", "full_attention"])
    b = dict(base, vocab_size=vocab or base["vocab_size"], tie_word_embeddings=True)
    cfg = Qwen3_5TextConfig(**b, num_hidden_layers=len(lt), layer_types=lt)
    cfg._attn_implementation = "eager"
    return cfg


def _meta_model(cfg):
    """The model's structure without memory (parameter names and shapes)."""
    torch = _torch()
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    with torch.device("meta"):
        return Qwen3_5TextModel(cfg)


def random_weights(cfg, seed: int = 0, embed: bool = False) -> dict[str, np.ndarray]:
    """Random weights with the model's shapes: linears N(0, 0.02) (the config's initializer
    range) with a few heavier output channels, zero-centered norm weights N(0, 0.1), the gated
    norm's weight 1 + N(0, 0.1), conv taps N(0, 0.3), A_log = log U(1, 16), dt_bias U(0.5, 1.5);
    with `embed`, the (tied) embedding table N(0, 0.02) too. Built from the model's structure
    alone, row block by row block, so a full-size model costs its fp32 weights and no more."""
    m = _meta_model(cfg)
    rng = np.random.default_rng(seed)
    w = {}
    for name, p in m.named_parameters():
        shape = tuple(p.shape)
        if name.startswith("embed_tokens"):
            if embed:
                e = np.empty(shape, np.float32)
                for r in range(0, shape[0], 8192):
                    e[r:r + 8192] = rng.normal(0, 0.02, (min(8192, shape[0] - r), shape[1]))
                w[name] = e
            continue
        if len(shape) == 2:
            a = rng.normal(0, 0.02, shape) * np.where(rng.random(shape[0]) < 0.02, 4.0, 1.0)[:, None]
        elif "conv1d" in name:
            a = rng.normal(0, 0.3, shape)
        elif name.endswith("A_log"):
            a = np.log(rng.uniform(1, 16, shape))
        elif name.endswith("dt_bias"):
            a = rng.uniform(0.5, 1.5, shape)
        elif name.endswith("linear_attn.norm.weight"):
            a = 1 + rng.normal(0, 0.1, shape)
        else:                                        # zero-centered RMSNorm weights
            a = rng.normal(0, 0.1, shape)
        w[name] = a.astype(np.float32)
    return w


CKPT_PREFIX = "model.language_model."     # Qwen3_5ForConditionalGeneration's text model


def load_weights(path, cfg, embed: bool = False) -> dict[str, np.ndarray]:
    """A Hugging Face checkpoint's text-model weights (a directory of safetensors, any float
    dtype) as fp32, under the names random_weights uses; bf16 to fp32 is exact. The vision tower
    and the MTP head are not loaded. Every parameter of the model's structure must be present
    with its shape."""
    from pathlib import Path
    from safetensors import safe_open
    torch = _torch()
    shapes = {n: tuple(p.shape) for n, p in _meta_model(cfg).named_parameters()}
    w = {}
    for f in sorted(Path(path).glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as st:
            for k in st.keys():
                name = k[len(CKPT_PREFIX):] if k.startswith(CKPT_PREFIX) else None
                if name not in shapes or (name.startswith("embed_tokens") and not embed):
                    continue
                t = st.get_tensor(k)
                if tuple(t.shape) != shapes[name]:
                    raise ValueError(f"{k}: shape {tuple(t.shape)}, the model needs {shapes[name]}")
                w[name] = t.to(torch.float32).numpy()
    missing = [n for n in shapes if n not in w and (embed or not n.startswith("embed_tokens"))]
    if missing:
        raise ValueError(f"checkpoint {path} lacks {missing[:5]} ({len(missing)} parameters)")
    return w


def random_inputs(cfg, n_tokens: int, seed: int = 1) -> np.ndarray:
    """Hidden states [T, H]: N(0, 1) with 1 % of the channels 20x larger (LLM-style outliers)."""
    rng = np.random.default_rng(seed)
    h = cfg.hidden_size
    x = rng.normal(0, 1, (n_tokens, h)) * np.where(rng.random(h) < 0.01, 20.0, 1.0)[None, :]
    return x.astype(np.float32)


# ---------------------------------------------------------------------- the W8A8 linear
def quant_tokens(x):
    """Per-token symmetric INT8 as TARS-mini's ROW_QUANT (fp32 steps, see numerics.quant_rows)."""
    from . import numerics as F
    return F.quant_rows(np.asarray(x, np.float32))


def quant_weight(w: np.ndarray):
    from .compiler import quantize_weight
    return quantize_weight(w, per_channel=True)


@dataclass
class Run:
    layer_out: list[np.ndarray] = field(default_factory=list)    # per step: each layer's output
    final: list[np.ndarray] = field(default_factory=list)        # per step: after the final norm
    logits: list[np.ndarray] = field(default_factory=list)       # per step: the last token's (with the head)


def run_reference(cfg, weights: dict, x: np.ndarray, chunks: list[int], fake_quant: bool,
                  head: bool = False) -> Run:
    """Feed x (input embeddings) in chunks (the first is the prefill, then e.g. 1-token decode
    steps) through the layers with a cache; collect each layer's output and the final-norm
    output per chunk, and with `head` the last token's logits (the tied embedding; W8A8 as the
    linears when fake_quant). The model shares the weights' memory."""
    torch = _torch()
    from transformers.cache_utils import DynamicCache
    m = _meta_model(cfg)
    m.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items() if not k.startswith("embed_tokens")},
                      strict=False, assign=True)
    m.rotary_emb = type(m.rotary_emb)(config=cfg)              # its buffers, off the meta device
    m.eval()
    if fake_quant:
        for mod in m.modules():
            if isinstance(mod, torch.nn.Linear):
                _fake_quant_linear(mod)
    hw = None
    if head:
        e = weights["embed_tokens.weight"]
        hw = [quant_weight(e[r:r + HEAD_ROWS].astype(np.float64)) for r in range(0, e.shape[0], HEAD_ROWS)] \
            if fake_quant else e
    outs: list[np.ndarray] = []
    hooks = [lay.register_forward_hook(lambda _m, _i, o: outs.append(o.detach().numpy().copy()))
             for lay in m.layers]
    cache = DynamicCache(config=cfg)
    res, t0 = Run(), 0
    with torch.no_grad():
        for n in chunks:
            outs.clear()
            y = m(inputs_embeds=torch.from_numpy(x[None, t0:t0 + n]), past_key_values=cache,
                  use_cache=True).last_hidden_state
            res.layer_out.append([o[0] for o in outs])
            res.final.append(y[0].numpy().copy())
            if head:
                res.logits.append(head_logits(y[0, -1].numpy(), hw, fake_quant))
            t0 += n
    for h in hooks:
        h.remove()
    return res


HEAD_ROWS = 62080          # the head's rows per GEMM (N is a 16-bit field): 248320 = 4 x 62080


def head_logits(y: np.ndarray, hw, fake_quant: bool) -> np.ndarray:
    """The last token's logits: y (the final norm's output) times the tied embedding, W8A8 per
    row block as the device computes it, or fp32."""
    if not fake_quant:
        return (hw.astype(np.float64) @ y.astype(np.float64)).astype(np.float32)
    q, s = quant_tokens(y[None, :])
    out = []
    for qw in hw:
        acc = qw.q.astype(np.float64) @ q[0].astype(np.float64)              # exact integers
        out.append((acc * np.float64(s[0]) * qw.scale.astype(np.float64)).astype(np.float32))
    return np.concatenate(out)


def _fake_quant_linear(mod) -> None:
    torch = _torch()
    qw = quant_weight(mod.weight.detach().numpy().astype(np.float64))
    wq = torch.from_numpy(qw.q)                                  # int8: widened per call
    sc = torch.from_numpy(qw.scale.astype(np.float64))

    def fwd(x):
        shp = x.shape
        q, s = quant_tokens(x.reshape(-1, shp[-1]).numpy())
        acc = torch.from_numpy(q.astype(np.float64)) @ wq.double().T        # exact integers
        y = acc * torch.from_numpy(s.astype(np.float64))[:, None] * sc[None, :]
        return y.to(torch.float32).reshape(*shp[:-1], -1)
    mod.forward = fwd
