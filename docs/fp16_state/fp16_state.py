"""How much does Qwen3.5-0.8B's output change when the Gated DeltaNet recurrence (the state and the
vector ops around it) runs in FP16 instead of FP32? transformers (fp32 weights, CPU) with the
recurrence replaced by a token-by-token loop whose precision is a parameter:

  fp32     every op fp32 (the reference; what TARS-mini computes)
  s16      state stored in FP16 after each pass over it (decay; update), ops in fp32
  bf16s    state stored in BF16, ops in fp32
  f16      FP16 lanes: inputs, every product and sum, and the state in FP16; dot products summed
           in the contract's order (16 lanes, then a pairwise fold), each add rounded to FP16
  f16a32   FP16 lanes, but dot products accumulate in fp32 (rounded to FP16 at the end)
  w8a8     yardstick: fp32 recurrence, every linear W8A8 (per-channel INT8 weights, per-token
           INT8 activations: TARS-mini's quantization) -- the deviation the hardware already has
  w8a8+f16 both

Teacher-forced over long texts; per 256-token window: logits cosine (mean / min), top-1 agreement
with fp32, KL(fp32 || variant), and the text's NLL under each variant.
    python fp16_state.py [--T 2048] [--modes fp32,s16,...]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/jiang/01_Github/01_HW/00_NPU/tars_mini_fpga")
sys.path.insert(0, str(ROOT))
from tars_mini import qwen_ref as Q  # noqa: E402
import transformers.models.qwen3_5.modeling_qwen3_5 as M  # noqa: E402

CKPT = Path.home() / "models/Qwen3.5-0.8B"
torch.set_num_threads(16)
MODES = {
    "fp32":  dict(st=None, op=None, acc="fp32"),
    "s16":   dict(st=torch.float16, op=None, acc="fp32"),
    "bf16s": dict(st=torch.bfloat16, op=None, acc="fp32"),
    "f16":   dict(st=torch.float16, op=torch.float16, acc="lanes16"),
    "f16a32": dict(st=torch.float16, op=torch.float16, acc="fp32"),
}
CUR = dict(MODES["fp32"])


def r(x, dt):
    return x if dt is None else x.to(dt).float()


def dot(S, x):
    """sum over k of S[..., k, v] * x[..., k]: the state times a vector (FDOT NT)."""
    op, acc = CUR["op"], CUR["acc"]
    p = S * x.unsqueeze(-1)
    if acc == "lanes16":                         # FP16 products, 16 lanes in order, pairwise fold
        p = r(p, op)
        sh = p.shape
        p = p.view(*sh[:-2], sh[-2] // 16, 16, sh[-1])
        a = torch.zeros_like(p[..., 0, :, :])
        for i in range(p.shape[-3]):
            a = r(a + p[..., i, :, :], op)
        while a.shape[-2] > 1:
            a = r(a[..., 0::2, :] + a[..., 1::2, :], op)
        return a[..., 0, :]
    return r(p.sum(-2), op)                      # exact products (fp16 x fp16 fits fp32), fp32 sum


def rec(query, key, value, g, beta, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=False, **kw):
    st, op = CUR["st"], CUR["op"]
    dt0 = query.dtype
    q, k, v, b, dec = [x.transpose(1, 2).to(torch.float32) for x in (query, key, value, beta, g)]
    if use_qk_l2norm_in_kernel:
        q = M.l2norm(q, dim=-1, eps=1e-6)
        k = M.l2norm(k, dim=-1, eps=1e-6)
    q = q / (q.shape[-1] ** 0.5)
    dec = dec.exp()                              # the gate: fp32 (exp / softplus stay fp32)
    q, k, v, b, dec = [r(x, op) for x in (q, k, v, b, dec)]
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    S = torch.zeros(B, H, Dk, Dv) if initial_state is None else initial_state.float()
    S = r(S, st)
    out = torch.zeros(B, H, T, Dv)
    for t in range(T):
        S = r(r(S * dec[:, :, t, None, None], op), st)                       # decay pass
        kv = dot(S, k[:, :, t])
        delta = r(r(v[:, :, t] - kv, op) * b[:, :, t, None], op)
        S = r(r(S + r(k[:, :, t, :, None] * delta[:, :, None, :], op), op), st)  # update pass
        out[:, :, t] = dot(S, q[:, :, t])
    return out.transpose(1, 2).contiguous().to(dt0), (S if output_final_state else None)


M.torch_chunk_gated_delta_rule = rec
M.torch_recurrent_gated_delta_rule = rec


def build():
    c = json.loads((CKPT / "config.json").read_text())
    lt = (c.get("text_config") or c)["layer_types"]
    cfg = Q.make_config(Q.CFG_08B, lt, vocab=248320)
    w = Q.load_weights(CKPT, cfg, embed=True)
    m = M.Qwen3_5TextModel(cfg)
    m.load_state_dict({k: torch.from_numpy(v) for k, v in w.items()}, strict=True)
    m.eval()
    return m


QW = {}                                          # W8A8: original weights, hooks


def w8a8(m, on: bool):
    lins = [mod for mod in m.modules() if isinstance(mod, torch.nn.Linear)]
    if on and not QW:
        for mod in lins:
            W = mod.weight.data
            QW[mod] = W.clone()
            s = W.abs().amax(1, keepdim=True).clamp_min(1e-30) / 127
            mod.weight.data = (W / s).round().clamp(-127, 127) * s

            def pre(module, args):
                x = args[0]
                s = x.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127
                return ((x / s).round().clamp(-127, 127) * s,) + tuple(args[1:])
            QW[(mod, "h")] = mod.register_forward_pre_hook(pre)
    if not on and QW:
        for k in list(QW):
            if isinstance(k, tuple):
                QW[k].remove()
            elif not isinstance(k, str):
                k.weight.data = QW[k]
        QW.clear()


def head(m, h, quant: bool):
    E = m.embed_tokens.weight
    if quant:
        s = h.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127
        h = (h / s).round().clamp(-127, 127) * s
        if "E" not in QW:
            se = E.abs().amax(1, keepdim=True).clamp_min(1e-30) / 127
            QW["E"] = ((E / se).round().clamp(-127, 127) * se)
        E = QW["E"]
    return h @ E.T


@torch.no_grad()
def hidden(m, ids):
    return m(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state[0]


@torch.no_grad()
def generate(m, prompt_ids, n):
    from transformers.cache_utils import DynamicCache
    ids = list(prompt_ids)
    out = m(input_ids=torch.tensor([ids]), use_cache=True)
    cache = out.past_key_values
    for _ in range(n):
        tok = int(head(m, out.last_hidden_state[0, -1], False).argmax())
        ids.append(tok)
        out = m(input_ids=torch.tensor([[tok]]), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--gen", type=int, default=1600)
    ap.add_argument("--modes", default="s16,bf16s,f16,f16a32,w8a8,w8a8+f16")
    ap.add_argument("--out", default="fp16_state.json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(CKPT))
    m = build()
    texts = {}
    en = tok(Path("/usr/share/doc/dosfstools/COPYING").read_text())["input_ids"][:a.T]
    texts["en_gpl3"] = en
    zcache = Path("zh_text.json")
    if zcache.exists():
        texts["zh_gen"] = json.loads(zcache.read_text())[:a.T]
    else:
        msgs = [{"role": "user", "content": "请写一篇详细的长文，介绍中国长城的历史、建筑结构、各朝代的修建过程、"
                                            "著名关隘、文化意义以及今天的保护工作。"}]
        p = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        p = p["input_ids"] if hasattr(p, "keys") else p
        p = list(p[0]) if len(p) and isinstance(p[0], (list, tuple)) else list(p)
        t0 = time.time()
        z = generate(m, p, a.gen)
        print(f"generated {a.gen} tokens in {time.time() - t0:.0f} s", flush=True)
        zcache.write_text(json.dumps(z))
        texts["zh_gen"] = z[:a.T]
        print(tok.decode(z[len(p):len(p) + 120]).replace("\n", " "), "...", flush=True)
    res = {}
    for tname, ids in texts.items():
        CUR.update(MODES["fp32"]); w8a8(m, False)
        t0 = time.time()
        href = hidden(m, ids)
        print(f"\n== {tname}: {len(ids)} tokens (fp32 reference {time.time() - t0:.0f} s)", flush=True)
        y = torch.tensor(ids[1:])
        for mode in a.modes.split(","):
            quant = mode.startswith("w8a8")
            CUR.update(MODES["f16" if mode.endswith("f16") and quant else ("fp32" if quant else mode)])
            w8a8(m, quant)
            t0 = time.time()
            hv = hidden(m, ids)
            el = time.time() - t0
            rows = []
            for s0 in range(0, len(ids), 256):
                s1 = min(s0 + 256, len(ids))
                lr = head(m, href[s0:s1], False)
                lv = head(m, hv[s0:s1], quant)
                cos = torch.nn.functional.cosine_similarity(lr, lv, dim=-1)
                pr, pv = lr.log_softmax(-1), lv.log_softmax(-1)
                kl = (pr.exp() * (pr - pv)).sum(-1)
                top1 = (lr.argmax(-1) == lv.argmax(-1)).float()
                n1 = min(s1, len(ids) - 1) - s0
                nll_r = -pr[:n1].gather(1, y[s0:s0 + n1, None]).mean()
                nll_v = -pv[:n1].gather(1, y[s0:s0 + n1, None]).mean()
                rows.append(dict(win=f"{s0}-{s1}", cos_mean=float(cos.mean()), cos_min=float(cos.min()),
                                 top1=float(top1.mean()), kl=float(kl.mean()), kl_max=float(kl.max()),
                                 nll_ref=float(nll_r), nll=float(nll_v)))
            res.setdefault(tname, {})[mode] = rows
            print(f"-- {mode} ({el:.0f} s)", flush=True)
            print(f"   {'window':>10s} {'cos mean':>9s} {'cos min':>8s} {'top1':>6s} {'KL mean':>9s} {'KL max':>8s} {'NLL ref':>8s} {'NLL':>7s}")
            for q_ in rows:
                print(f"   {q_['win']:>10s} {q_['cos_mean']:9.5f} {q_['cos_min']:8.4f} {100 * q_['top1']:5.1f}% "
                      f"{q_['kl']:9.2e} {q_['kl_max']:8.2e} {q_['nll_ref']:8.3f} {q_['nll']:7.3f}", flush=True)
        w8a8(m, False)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
