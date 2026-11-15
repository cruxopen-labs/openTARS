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
# Qwen3-0.6B (Qwen/Qwen3-0.6B config.json): 28 full-attention layers (no output gate, QK-norm with
# plain weights, full rotary), tied embeddings, vocabulary 151936
VOCAB_Q3_06B = 151936
CFG_Q3_06B = dict(
    model_type="qwen3", hidden_size=1024, intermediate_size=3072, num_attention_heads=16,
    num_key_value_heads=8, head_dim=128, rms_norm_eps=1e-6, vocab_size=16, max_position_embeddings=40960,
    num_hidden_layers=28, rope_parameters=dict(rope_type="default", rope_theta=1000000.0),
)
# its small config, for RTL runs (the rotary width = the head dim: a multiple of 32)
CFG_Q3_SMALL = dict(CFG_Q3_06B, hidden_size=256, intermediate_size=512, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=64)
# Gemma 3 1B (google/gemma-3-1b-it config.json): 26 layers, five sliding-window (512) layers to one
# global, 4 query heads / 1 KV head x 256, QK-norm, four RMSNorms (1 + w) per layer, GeGLU (tanh),
# local / global RoPE bases 1e4 / 1e6, tied embeddings scaled by sqrt(hidden), vocabulary 262144
VOCAB_G3_1B = 262144
LAYERS_G3_1B = (["sliding_attention"] * 5 + ["full_attention"]) * 4 + ["sliding_attention"] * 2
CFG_G3_1B = dict(
    model_type="gemma3_text", hidden_size=1152, intermediate_size=6912, num_attention_heads=4,
    num_key_value_heads=1, head_dim=256, query_pre_attn_scalar=256, rms_norm_eps=1e-6, vocab_size=16,
    sliding_window=512, hidden_activation="gelu_pytorch_tanh", max_position_embeddings=32768,
    attn_logit_softcapping=None, final_logit_softcapping=None,
    rope_parameters={"full_attention": dict(rope_type="default", rope_theta=1000000.0),
                     "sliding_attention": dict(rope_type="default", rope_theta=10000.0)},
)
CFG_G3_SMALL = dict(CFG_G3_1B, hidden_size=256, intermediate_size=512, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=64, query_pre_attn_scalar=64, sliding_window=64)
# Gemma 4 E2B (google/gemma-4-E2B-it config.json, text_config): 35 layers, four sliding-window (512)
# layers to one full-attention layer, 8 query heads / 1 KV head, head_dim 256 (sliding) / 512 (full),
# proportional 25 % RoPE on full layers, KV sharing across the last 20 layers (double-wide MLP 12288),
# per-layer embeddings (ple_dim 256), V-norm without scale, layer_scalar, softcapped tied head (30.0)
VOCAB_G4_E2B = 262144
LAYERS_G4_E2B = (["sliding_attention"] * 4 + ["full_attention"]) * 7
CFG_G4_E2B = dict(
    model_type="gemma4_text", hidden_size=1536, intermediate_size=6144, num_attention_heads=8,
    num_key_value_heads=1, head_dim=256, global_head_dim=512, hidden_size_per_layer_input=256,
    vocab_size_per_layer_input=262144, num_kv_shared_layers=20, use_double_wide_mlp=True,
    enable_moe_block=False, attention_k_eq_v=False, rms_norm_eps=1e-6, vocab_size=16,
    sliding_window=512, hidden_activation="gelu_pytorch_tanh", max_position_embeddings=131072,
    final_logit_softcapping=30.0,
    rope_parameters={
        "full_attention": dict(rope_type="proportional", rope_theta=1000000.0, partial_rotary_factor=0.25),
        "sliding_attention": dict(rope_type="default", rope_theta=10000.0),
    },
)
CFG_G4_SMALL = dict(CFG_G4_E2B, hidden_size=256, intermediate_size=512, num_attention_heads=4,
                    num_key_value_heads=1, head_dim=64, global_head_dim=128,
                    hidden_size_per_layer_input=128, vocab_size_per_layer_input=16,
                    num_kv_shared_layers=2, sliding_window=64)
# A small config with the same structure, for RTL runs
# (16 linear-attention heads as the real model; head dims kept multiples of 16 and the rotary
# width a multiple of 32, as the contract needs)
CFG_SMALL = dict(CFG_08B, hidden_size=256, intermediate_size=512, num_attention_heads=4,
                 num_key_value_heads=2, head_dim=128, linear_key_head_dim=16, linear_value_head_dim=16,
                 linear_num_key_heads=16, linear_num_value_heads=16)


def _torch():
    import torch                                     # noqa: F401  (optional dependency)
    return torch


def is_qwen3(cfg) -> bool:
    """Qwen3 (every layer full attention, plain RMSNorm weights) rather than Qwen3.5."""
    return getattr(cfg, "model_type", "") == "qwen3"


def is_gemma3(cfg) -> bool:
    return getattr(cfg, "model_type", "") == "gemma3_text"


def is_gemma4(cfg) -> bool:
    return getattr(cfg, "model_type", "") == "gemma4_text"


def make_config(base: dict, layer_types: list[str] | None = None, vocab: int | None = None):
    if base.get("model_type") == "gemma4_text":
        from transformers.models.gemma4 import Gemma4TextConfig
        b = {k: v for k, v in base.items() if k != "model_type"}
        lt = list(layer_types or ["sliding_attention", "full_attention", "sliding_attention", "full_attention"])
        vs = vocab or b["vocab_size"]
        vs_ple = vs if vocab is not None or b.get("vocab_size_per_layer_input") == b["vocab_size"] \
            else b["vocab_size_per_layer_input"]
        n_shared = min(b.get("num_kv_shared_layers", 0), max(0, len(lt) - 2))
        cfg = Gemma4TextConfig(
            **dict(b, vocab_size=vs, vocab_size_per_layer_input=vs_ple, num_kv_shared_layers=n_shared,
                   tie_word_embeddings=True),
            num_hidden_layers=len(lt), layer_types=lt)
        cfg._attn_implementation = "eager"
        return cfg
    if base.get("model_type") == "gemma3_text":
        from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
        b = {k: v for k, v in base.items() if k != "model_type"}
        lt = list(layer_types or ["sliding_attention", "full_attention"])
        cfg = Gemma3TextConfig(**dict(b, vocab_size=vocab or b["vocab_size"], tie_word_embeddings=True),
                               num_hidden_layers=len(lt), layer_types=lt)
        cfg._attn_implementation = "eager"
        return cfg
    if base.get("model_type") == "qwen3":
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        b = {k: v for k, v in base.items() if k != "model_type"}
        n = len(layer_types) if layer_types else b.pop("num_hidden_layers")
        b.pop("num_hidden_layers", None)
        cfg = Qwen3Config(**dict(b, vocab_size=vocab or b["vocab_size"], tie_word_embeddings=True),
                          num_hidden_layers=n, layer_types=["full_attention"] * n)
        cfg._attn_implementation = "eager"
        return cfg
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    lt = list(layer_types or ["linear_attention", "full_attention"])
    b = dict(base, vocab_size=vocab or base["vocab_size"], tie_word_embeddings=True)
    cfg = Qwen3_5TextConfig(**b, num_hidden_layers=len(lt), layer_types=lt)
    cfg._attn_implementation = "eager"
    return cfg


def _meta_model(cfg, dtype: str = "fp32"):
    """The model's structure without memory (parameter names and shapes); `dtype` selects the
    reference computation's float type ("fp32" or "bfloat16")."""
    torch = _torch()
    if is_qwen3(cfg):
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Model as M
    elif is_gemma3(cfg):
        from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel as M
    elif is_gemma4(cfg):
        from transformers.models.gemma4 import Gemma4TextModel as M
    else:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel as M
    with torch.device("meta"):
        return M(cfg)


def random_weights(cfg, seed: int = 0, embed: bool = False) -> dict[str, np.ndarray]:
    """Random weights with the model's shapes: linears N(0, 0.02) (the config's initializer
    range) with a few heavier output channels, zero-centered norm weights N(0, 0.1), the gated
    norm's weight 1 + N(0, 0.1), conv taps N(0, 0.3), A_log = log U(1, 16), dt_bias U(0.5, 1.5);
    with `embed`, the (tied) embedding table N(0, 0.02) too. Built from the model's structure
    alone, row block by row block, so a full-size model costs its fp32 weights and no more."""
    m = _meta_model(cfg)
    rng = np.random.default_rng(seed)
    w = {}
    for name, p in m.state_dict().items():
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
        elif name.endswith("dt_bias") or name.endswith(".layer_scalar"):
            a = rng.uniform(0.5, 1.5, shape)
        elif name.endswith("linear_attn.norm.weight"):
            a = 1 + rng.normal(0, 0.1, shape)
        elif is_qwen3(cfg) or is_gemma4(cfg):        # Qwen3 / Gemma 4: RMSNorm weights around 1
            a = 1 + rng.normal(0, 0.1, shape)
        else:                                        # zero-centered RMSNorm weights
            a = rng.normal(0, 0.1, shape)
        w[name] = a.astype(np.float32)
    return w


CKPT_PREFIX = "model.language_model."     # Qwen3_5 / Gemma4 conditional generation text model
CKPT_PREFIX_Q3 = "model."                 # Qwen3ForCausalLM's (and Gemma3ForCausalLM's)


def load_weights(path, cfg, embed: bool = False, ple_embed: bool | None = None) -> dict[str, np.ndarray]:
    """A Hugging Face checkpoint's text-model weights (a directory of safetensors, any float
    dtype) as fp32, under the names random_weights uses; bf16 to fp32 is exact. The vision tower
    and the MTP head are not loaded. Every parameter of the model's structure must be present
    with its shape."""
    from pathlib import Path
    from safetensors import safe_open
    torch = _torch()
    if ple_embed is None:
        ple_embed = embed
    shapes = {n: tuple(p.shape) for n, p in _meta_model(cfg).state_dict().items()}
    prefixes = (CKPT_PREFIX_Q3,) if is_qwen3(cfg) or is_gemma3(cfg) else (CKPT_PREFIX, CKPT_PREFIX_Q3)

    def wanted(n: str) -> bool:
        if n == "embed_tokens.weight":
            return embed
        if n == "embed_tokens_per_layer.weight":
            return ple_embed
        return True

    w = {}
    for f in sorted(Path(path).glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as st:
            for k in st.keys():
                name = next((k[len(pre):] for pre in prefixes if k.startswith(pre)), None)
                if name not in shapes or not wanted(name):
                    continue
                t = st.get_tensor(k)
                if tuple(t.shape) != shapes[name]:
                    raise ValueError(f"{k}: shape {tuple(t.shape)}, the model needs {shapes[name]}")
                w[name] = t.to(torch.float32).numpy()
    missing = [n for n in shapes if n not in w and wanted(n)]
    if missing:
        raise ValueError(f"checkpoint {path} lacks {missing[:5]} ({len(missing)} parameters)")
    return w


def load_mtp(path) -> dict[str, np.ndarray]:
    """A checkpoint's MTP head as fp32, under its own names (mtp.*)."""
    from pathlib import Path
    from safetensors import safe_open
    torch = _torch()
    w = {}
    for f in sorted(Path(path).glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as st:
            for k in st.keys():
                if k.startswith("mtp."):
                    w[k] = st.get_tensor(k).to(torch.float32).numpy()
    if not w:
        raise ValueError(f"checkpoint {path} has no MTP head")
    return w


def random_mtp(cfg, seed: int = 5) -> dict[str, np.ndarray]:
    """Random MTP weights with cfg's dimensions (the names and shapes of load_mtp's)."""
    rng = np.random.default_rng(seed)
    H, I, hd = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim
    nq, nkv = cfg.num_attention_heads * hd, cfg.num_key_value_heads * hd
    lin = {"fc": (H, 2 * H), "layers.0.self_attn.q_proj": (2 * nq, H), "layers.0.self_attn.k_proj": (nkv, H),
           "layers.0.self_attn.v_proj": (nkv, H), "layers.0.self_attn.o_proj": (H, nq),
           "layers.0.mlp.gate_proj": (I, H), "layers.0.mlp.up_proj": (I, H), "layers.0.mlp.down_proj": (H, I)}
    nrm = {"norm": H, "pre_fc_norm_embedding": H, "pre_fc_norm_hidden": H, "layers.0.input_layernorm": H,
           "layers.0.post_attention_layernorm": H, "layers.0.self_attn.q_norm": hd, "layers.0.self_attn.k_norm": hd}
    w = {f"mtp.{k}.weight": rng.normal(0, 0.02, s).astype(np.float32) for k, s in lin.items()}
    w.update({f"mtp.{k}.weight": rng.normal(0, 0.1, n).astype(np.float32) for k, n in nrm.items()})
    return w


def random_inputs(cfg, n_tokens: int, seed: int = 1) -> np.ndarray:
    """Hidden states [T, H]: N(0, 1) with 1 % of the channels 20x larger (LLM-style outliers)."""
    rng = np.random.default_rng(seed)
    h = cfg.hidden_size
    x = rng.normal(0, 1, (n_tokens, h)) * np.where(rng.random(h) < 0.01, 20.0, 1.0)[None, :]
    return x.astype(np.float32)


def random_ple_inputs(cfg, n_tokens: int, seed: int = 2) -> np.ndarray:
    """Gemma 4 per-layer token embeddings [T, L * ple_dim] for block tests without embed_tokens."""
    rng = np.random.default_rng(seed)
    cols = cfg.num_hidden_layers * cfg.hidden_size_per_layer_input
    return rng.normal(0, 0.3, (n_tokens, cols)).astype(np.float32)


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
                  head: bool = False, ple: np.ndarray | None = None, wbits: int = 8,
                  dtype: str = "fp32") -> Run:
    """Feed x (input embeddings) in chunks (the first is the prefill, then e.g. 1-token decode
    steps) through the layers with a cache; collect each layer's output and the final-norm
    output per chunk, and with `head` the last token's logits (the tied embedding; W8A8 as the
    linears when fake_quant). The model shares the weights' memory."""
    torch = _torch()
    from transformers.cache_utils import DynamicCache
    m = _meta_model(cfg, dtype)
    td = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    m.load_state_dict({k: torch.from_numpy(v).to(td) for k, v in weights.items()
                       if not k.startswith("embed_tokens")}, strict=False, assign=True)
    if is_gemma4(cfg):
        m.per_layer_model_projection_scale = torch.tensor(float(cfg.hidden_size) ** -0.5, dtype=td)
        m.per_layer_input_scale = torch.tensor(2.0 ** -0.5, dtype=td)
    m.rotary_emb = type(m.rotary_emb)(config=cfg)              # its buffers, off the meta device
    m.eval()
    if fake_quant:
        for mod in m.modules():
            if isinstance(mod, torch.nn.Linear):
                _fake_quant_linear(mod, wbits)
    hw = None
    softcap = getattr(cfg, "final_logit_softcapping", None)
    if head:
        e = weights["embed_tokens.weight"]
        hr = head_rows(e.shape[0])
        if not fake_quant:
            hw = e
        elif wbits == 4:
            from .compiler import quantize_weight4
            hw = [quantize_weight4(e[r:r + hr].astype(np.float64)) for r in range(0, e.shape[0], hr)]
        else:
            hw = [quant_weight(e[r:r + hr].astype(np.float64)) for r in range(0, e.shape[0], hr)]
    outs: list[np.ndarray] = []
    hooks = [lay.register_forward_hook(lambda _m, _i, o: outs.append(o.detach().float().numpy().copy()))
             for lay in m.layers]
    cache = DynamicCache(config=cfg)
    res, t0 = Run(), 0
    with torch.no_grad():
        for n in chunks:
            outs.clear()
            kw = {}
            if is_gemma4(cfg):
                assert ple is not None, "Gemma 4 reference needs ple=[T, L*ple_dim]"
                kw["per_layer_inputs"] = torch.from_numpy(
                    ple[t0:t0 + n].reshape(1, n, cfg.num_hidden_layers,
                                           cfg.hidden_size_per_layer_input)).to(td)
            y = m(inputs_embeds=torch.from_numpy(x[None, t0:t0 + n]).to(td), past_key_values=cache,
                  use_cache=True, **kw).last_hidden_state
            res.layer_out.append([o[0] for o in outs])
            res.final.append(y[0].float().numpy().copy())
            if head:
                res.logits.append(head_logits(y[0, -1].float().numpy(), hw, fake_quant,
                                              softcap=softcap, wbits=wbits))
            t0 += n
    for h in hooks:
        h.remove()
    return res


HEAD_ROWS = 62080          # the head's rows per GEMM (N is a 16-bit field): 248320 = 4 x 62080


def head_rows(vocab: int) -> int:
    """The head's rows per GEMM: the vocabulary in the fewest equal blocks of at most 65535 rows,
    each a multiple of 16 (248320: 62080; 151936: 37984)."""
    for b in range(1, vocab // 16 + 1):
        r = vocab // b
        if vocab % b == 0 and r <= 65535 and r % 16 == 0:
            return r
    raise ValueError(f"vocabulary {vocab}: no block split")


def head_logits(y: np.ndarray, hw, fake_quant: bool, softcap: float | None = None,
                wbits: int = 8) -> np.ndarray:
    """The last token's logits: y (the final norm's output) times the tied embedding, W8A8 / W4A16
    per row block as the device computes it, or fp32."""
    if not fake_quant:
        lg = (hw.astype(np.float64) @ y.astype(np.float64)).astype(np.float32)
    elif wbits == 4:
        out = []
        for qw in hw:
            out.append(_w4_matmul(y[None, :], qw)[0])
        lg = np.concatenate(out)
    else:
        q, s = quant_tokens(y[None, :])
        out = []
        for qw in hw:
            acc = qw.q.astype(np.float64) @ q[0].astype(np.float64)              # exact integers
            out.append((acc * np.float64(s[0]) * qw.scale.astype(np.float64)).astype(np.float32))
        lg = np.concatenate(out)
    if softcap is not None:
        sc = np.float32(softcap)
        lg = (np.tanh(lg / sc) * sc).astype(np.float32)
    return lg


def _w4_matmul(x: np.ndarray, qw) -> np.ndarray:
    """The W4A16 hardware's linear on x [M, K] (the reference's FP64 mirror of the FP32 epilogue)."""
    from . import numerics as F
    from . import contract as C
    q, xs = F.align_w4_groups(np.asarray(x, np.float32))
    M, K = q.shape
    G = K // C.W4_GROUP
    lo, hi = (qw.q & 15).astype(np.int16), (qw.q >> 4).astype(np.int16)
    lo, hi = np.where(lo >= 8, lo - 16, lo), np.where(hi >= 8, hi - 16, hi)
    wq = np.stack((lo, hi), axis=2).reshape(qw.n, G, C.W4_GROUP).astype(np.int64)
    ws = qw.scale.view(np.float16).astype(np.float32)
    acc = np.zeros((M, qw.n, 4), np.float32)
    qg = q.reshape(M, G, C.W4_GROUP).astype(np.int64)
    for g in range(G):
        dot = np.einsum("mk,nk->mn", qg[:, g], wq[:, g])
        term = F.fmul(F.fmul(F.i2f(dot), ws[None, :, g]), xs[:, None, g])
        acc[:, :, g % 4] = F.fadd(acc[:, :, g % 4], term)
    y = F.fadd(F.fadd(acc[:, :, 0], acc[:, :, 1]), F.fadd(acc[:, :, 2], acc[:, :, 3]))
    return y


def _fake_quant_linear(mod, wbits: int = 8) -> None:
    torch = _torch()
    if wbits == 4:
        from .compiler import quantize_weight4
        qw = quantize_weight4(mod.weight.detach().numpy().astype(np.float64))

        def fwd(x):
            shp = x.shape
            y = _w4_matmul(x.reshape(-1, shp[-1]).numpy(), qw)
            return torch.from_numpy(y).reshape(*shp[:-1], -1)
    else:
        qw = quant_weight(mod.weight.detach().numpy().astype(np.float64))
        wq = torch.from_numpy(qw.q)                              # int8: widened per call
        sc = torch.from_numpy(qw.scale.astype(np.float64))

        def fwd(x):
            shp = x.shape
            q, s = quant_tokens(x.reshape(-1, shp[-1]).numpy())
            acc = torch.from_numpy(q.astype(np.float64)) @ wq.double().T  # exact integers
            y = acc * torch.from_numpy(s.astype(np.float64))[:, None] * sc[None, :]
            return y.to(torch.float32).reshape(*shp[:-1], -1)
    mod.forward = fwd
