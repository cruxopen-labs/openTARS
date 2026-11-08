"""Export Qwen3.5-0.8B, Qwen3-0.6B or Gemma 3 1B (the checkpoint's config.json model_type picks it) for
the KV260 board runtime (deploy/kv260/sw/tars_board.py).

    python3 tools/export_kv260.py --ckpt ~/models/Qwen3.5-0.8B [--T 256] [--out deploy/kv260/data]
    python3 tools/export_kv260.py --ckpt ~/models/Qwen3-0.6B --out deploy/kv260/data_qwen3
    python3 tools/export_kv260.py --ckpt ~/models/gemma-3-1b-it --out deploy/kv260/data_gemma3  # T <= 512 (its window)
    python3 tools/export_kv260.py --mtp --out deploy/kv260/data_mtp  # Qwen3.5 with its MTP head: speculative decoding
    python3 tools/export_kv260.py --out deploy/kv260/data --check-only      # redo the golden check
    python3 tools/export_kv260.py --chunks-only    # add the prefill chunk queues to an export

Writes into --out:
  image.bin        the DDR image from address 0: the zeroed states, the queue region, scratch, the
                   quantized weights (the core's address space is the board's 1 GB buffer)
  embed_bf16.bin   the embedding table [vocab, H] as bf16 (the checkpoint's own values: exact; Gemma's
                   input embedding is a row times the manifest's embed_scale, in fp32)
  queues.bin       one descriptor queue per token position t (t0 = t, one token), back to back
  queues_c<n>.bin  prefill: one queue per chunk of n tokens at t0 (a multiple of ALIGN): the weights
                   are read once for n tokens; bit-identical to n one-token steps (the same
                   operations on each row, the states the same)
  queues_s_*.bin   with --mtp: v2p0 / v2p1 verify two tokens at t on either DeltaNet state copy, m1 / m2
                   MTP rows with the draft head, mp1 / mp4 MTP rows of a prompt (no head), swap the state copy
  manifest.json    addresses and sizes the runtime needs, the queue index, tokenizer settings
  tokenizer.json   (and the chat template's prefix / suffix in the manifest)
  expected.json    the golden model run through the same runtime (tars_board.Qwen on the Golden
                   backend), token by token: each step's argmax and the SHA-256 of its logits.
                   The board must reproduce these bits exactly.
  selftest/        small random programs (the RTL regression's generator) with the golden
                   model's results, for a quick check of the hardware path
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deploy" / "kv260" / "sw"))

DDR = 1 << 30
CHUNKS, ALIGN = (8, 4), 4            # prefill chunk sizes (the runtime uses those <= the GEMV rows: 4 from bitstream 3.0, 8 before); chunks start at multiples of ALIGN
CHECKS = [dict(name="plain", text="The capital of France is", chat=False, new=8),
          dict(name="chat", text="Write one sentence about the Great Wall.", chat=True, new=12)]


def model_config(ckpt):
    """The lowering's config for a checkpoint: Qwen3 (model_type "qwen3") or Qwen3.5."""
    from tars_mini import qwen_ref as Q
    mt = json.loads((Path(ckpt) / "config.json").read_text()).get("model_type", "")
    if mt == "qwen3":
        return Q.make_config(Q.CFG_Q3_06B, vocab=Q.VOCAB_Q3_06B)
    if mt in ("gemma3_text", "gemma3"):
        cfg = Q.make_config(Q.CFG_G3_1B, Q.LAYERS_G3_1B, vocab=Q.VOCAB_G3_1B)
        from transformers import AutoConfig
        have = AutoConfig.from_pretrained(ckpt)
        have = getattr(have, "text_config", have)
        keys = ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim",
                "query_pre_attn_scalar", "rms_norm_eps", "vocab_size", "layer_types", "hidden_activation",
                "attn_logit_softcapping", "final_logit_softcapping", "rope_parameters")
        diff = {k: (getattr(have, k, None), getattr(cfg, k)) for k in keys if getattr(have, k, None) != getattr(cfg, k)}
        if diff:
            raise ValueError(f"{ckpt}: not the Gemma 3 1B this export supports: {diff}")
        return cfg
    return Q.make_config(Q.CFG_08B, Q.LAYERS_08B, vocab=Q.VOCAB_08B)


def export(args) -> None:
    from tars_mini import llm as L
    from tars_mini import qwen_ref as Q
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    cfg = model_config(args.ckpt)
    w = Q.load_weights(args.ckpt, cfg, embed=True)
    if args.mtp:
        w.update(Q.load_mtp(args.ckpt))
    emb = np.ascontiguousarray(w["embed_tokens.weight"], np.float32)
    bits = emb.view(np.uint32)
    assert not (bits & 0xFFFF).any(), "the embedding is not bf16-exact"
    (bits >> 16).astype(np.uint16).tofile(out / "embed_bf16.bin")
    print(f"embedding {emb.shape} ({time.time() - t:.0f} s)", flush=True)
    low = L.QwenLowering(cfg, w, max_tokens=args.T, ddr_bytes=DDR, head=True, queue_cap=16384,
                         verify_rows=2 if args.mtp else 1, mtp=args.mtp, state_dtype=args.state_dtype)
    del w
    assert low.p.base == 0
    img = bytes(low.p.ddr[:low.end])
    pad = (-len(img)) % (1 << 20)
    (out / "image.bin").write_bytes(img + bytes(pad))
    print(f"image {len(img) / 2**20:.0f} MiB ({time.time() - t:.0f} s)", flush=True)
    index, off = [], 0
    with open(out / "queues.bin", "wb") as f:
        for t0 in range(args.T):
            a = len(low.p.descs)
            low.chunk(t0, 1)
            b = len(low.p.descs)
            assert b - a <= low.queue_cap
            f.write(b"".join(low.p.descs[a:b]))
            index.append([off, b - a])
            off += 64 * (b - a)
            del low.p.descs[a:b]
    print(f"queues: {args.T} positions, {off / 2**20:.1f} MiB ({time.time() - t:.0f} s)", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    mark = "⁣MSG⁣"
    tmpl = tok.apply_chat_template([{"role": "user", "content": mark}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    pre, post = tmpl.split(mark)
    bos = []
    if tok.bos_token and pre.startswith(tok.bos_token):      # Gemma: <bos> once, at position 0
        pre, bos = pre[len(tok.bos_token):], [int(tok.bos_token_id)]
    assert bos or not low.gemma, "Gemma's chat template starts with <bos>"
    turn_end = "<end_of_turn>\n" if low.gemma else "<|im_end|>\n"
    gen = json.loads((Path(args.ckpt) / "generation_config.json").read_text()) \
        if (Path(args.ckpt) / "generation_config.json").exists() else {}
    eos = gen.get("eos_token_id", [])
    vocab = tok.get_vocab()
    eos = sorted({*(eos if isinstance(eos, list) else [eos]),
                  *(vocab[s] for s in ("<|im_end|>", "<|endoftext|>", "<end_of_turn>", "<eos>") if s in vocab)}
                 - {None})
    for fn in ("tokenizer.json", "tokenizer_config.json"):
        if (Path(args.ckpt) / fn).exists():
            shutil.copy(Path(args.ckpt) / fn, out / fn)
    la = not (low.q3 or low.gemma)                           # DeltaNet states: Qwen3.5 only
    m = dict(model=Path(args.ckpt).resolve().name, max_tokens=args.T, hidden=low.H, vocab=low.vocab,
             state_dtype=args.state_dtype if la else "fp32",
             min_firmware=0x30400 if args.state_dtype == "fp16" and la else 0,
             bos_ids=bos, turn_end=turn_end,
             embed_scale=float(L.input_embedding(cfg, np.ones((1, 1), np.float32))[0, 0]) if low.gemma else None,
             image_bytes=len(img) + pad, state_end=low.scratch_end, qbase=low.qbase,
             queue_cap=low.queue_cap, x0_off=low.x0.off, x0_ld=low.x0.ld, logits_off=low.logits.off,
             mem_bytes=DDR, chat_prefix=pre, chat_suffix=post, eos_ids=[int(e) for e in eos],
             queue_index=index, head_blocks=[lin.n for lin in low.head_lins])
    (out / "manifest.json").write_text(json.dumps(m))
    print(f"manifest: {({k: v for k, v in m.items() if k != 'queue_index'})}", flush=True)
    export_chunks(low, out, args.T)
    if args.mtp:
        export_spec(low, out, args.T)


def export_spec(low, out: Path, T: int) -> None:
    """The speculative queues, one per position (see the docstring); their index into the manifest
    (spec: {name: [[t, offset, count], ...]})."""
    t = time.time()
    m = json.loads((out / "manifest.json").read_text())
    kinds = dict(
        v2p0=(range(T - 1), lambda r: low.chunk(r, 2, all_rows=True, parity=0, spec=True)),
        v2p1=(range(T - 1), lambda r: low.chunk(r, 2, all_rows=True, parity=1, spec=True)),
        m1=(range(T - 1), lambda r: low.mtp_chunk(r, 1)),
        m2=(range(T - 2), lambda r: low.mtp_chunk(r, 2)),
        mp1=(range(T - 1), lambda r: low.mtp_chunk(r, 1, head=False)),
        mp4=(range(0, T - 4, 4), lambda r: low.mtp_chunk(r, 4, head=False)),
        swap=(range(1), lambda r: low.swap_state()))
    m["spec"] = {}
    for name, (rows, emit) in kinds.items():
        index, off = [], 0
        with open(out / f"queues_s_{name}.bin", "wb") as f:
            for r in rows:
                a = len(low.p.descs)
                emit(r)
                b = len(low.p.descs)
                f.write(b"".join(low.p.descs[a:b]))
                index.append([r, off, b - a])
                off += 64 * (b - a)
                del low.p.descs[a:b]
        m["spec"][name] = index
        print(f"spec {name}: {len(index)} queues, {off / 2**20:.1f} MiB ({time.time() - t:.0f} s)", flush=True)
    (out / "manifest.json").write_text(json.dumps(m))


def export_chunks(low, out: Path, T: int) -> None:
    """The prefill queues: for each chunk size n, one queue per t0 (a multiple of ALIGN, t0 + n <= T),
    into queues_c<n>.bin; their index into the manifest (chunks: {n: [[t0, offset, count], ...]})."""
    t = time.time()
    m = json.loads((out / "manifest.json").read_text())
    m["chunks"] = {}
    for n in CHUNKS:
        index, off = [], 0
        with open(out / f"queues_c{n}.bin", "wb") as f:
            for t0 in range(0, T - n + 1, ALIGN):
                a = len(low.p.descs)
                low.chunk(t0, n)
                b = len(low.p.descs)
                f.write(b"".join(low.p.descs[a:b]))
                index.append([t0, off, b - a])
                off += 64 * (b - a)
                del low.p.descs[a:b]
        m["chunks"][str(n)] = index
        print(f"chunk {n}: {len(index)} queues, {off / 2**20:.1f} MiB ({time.time() - t:.0f} s)", flush=True)
    (out / "manifest.json").write_text(json.dumps(m))


def chunks_only(args) -> None:
    """Rebuild the lowering (the same layout: checked against image.bin) and add the chunk queues."""
    from tars_mini import llm as L
    from tars_mini import qwen_ref as Q
    out = Path(args.out)
    m = json.loads((out / "manifest.json").read_text())
    cfg = model_config(args.ckpt)
    w = Q.load_weights(args.ckpt, cfg, embed=True)          # as export(): the head is tied to it
    low = L.QwenLowering(cfg, w, max_tokens=m["max_tokens"], ddr_bytes=DDR, head=True, queue_cap=m["queue_cap"],
                         state_dtype=m.get("state_dtype", "fp32"))
    del w
    img = bytes(low.p.ddr[:low.end])
    have = (out / "image.bin").read_bytes()[:len(img)]
    assert img == have and low.qbase == m["qbase"], "the lowering does not match image.bin: export again"
    print("lowering matches image.bin", flush=True)
    export_chunks(low, out, m["max_tokens"])


def selftest(out: Path, n: int = 24) -> None:
    sys.path.insert(0, str(ROOT / "tests"))
    import test_rtl
    from test_rtl import QADDR, random_program
    from tars_mini.golden import Machine
    test_rtl.GEMV_ONLY = True                # the board's core has no systolic array: no conv in the queues
    d = out / "selftest"
    d.mkdir(parents=True, exist_ok=True)
    from tars_mini import contract as C
    cases, seed = [], 777000
    while len(cases) < n:
        ddr = random_program(seed)
        seed += 1
        g = Machine(bytearray(ddr), gemv_only=True)
        st = g.run_queue(QADDR, 8)
        if st.err_code == C.E_RANGE:          # the board's address space is 1 GB, not the image's size
            continue
        i = len(cases)
        (d / f"case{i}_in.bin").write_bytes(ddr)
        (d / f"case{i}_out.bin").write_bytes(bytes(g.ddr))
        cases.append(dict(name=f"case{i}", qaddr=QADDR, qlen=8, size=len(ddr), done=bool(st.done),
                          err=int(st.err_code), op=int(st.op_idx)))
    (d / "cases.json").write_text(json.dumps(cases, indent=1))
    print(f"selftest: {n} cases, {sum(c['err'] != 0 for c in cases)} with an injected error", flush=True)


def check(out: Path) -> None:
    import tars_board as B
    from tokenizers import Tokenizer
    m = json.loads((out / "manifest.json").read_text())
    tk = Tokenizer.from_file(str(out / "tokenizer.json"))
    dev = B.Golden()
    model = B.Qwen(dev, out, log=lambda *a, **k: None)
    res = []
    for c in CHECKS:
        model.new_conversation()
        ids = model.encode(tk, c["text"], chat=c["chat"])
        steps, t = [], time.time()
        cur = ids[0]
        gen = []
        for i in range(len(ids) + c["new"] - 1):
            lg = model.step(cur)
            steps.append(dict(pos=i, token=int(cur), argmax=int(np.argmax(lg)),
                              sha256=hashlib.sha256(lg.tobytes()).hexdigest()))
            if i + 1 < len(ids):
                cur = ids[i + 1]
            else:
                cur = int(np.argmax(lg))
                gen.append(cur)
            print(f"  {c['name']} step {i}: token {steps[-1]['token']} -> {steps[-1]['argmax']} "
                  f"({time.time() - t:.0f} s)", flush=True)
        res.append(dict(c, prompt_ids=ids, generated=gen, output=tk.decode(gen), steps=steps))
        print(f"{c['name']}: {c['text']!r} -> {tk.decode(gen)!r}", flush=True)
    (out / "expected.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(Path.home() / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--out", default=str(ROOT / "deploy" / "kv260" / "data"))
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--chunks-only", action="store_true")
    ap.add_argument("--mtp", action="store_true", help="Qwen3.5: the MTP head and the speculative queues")
    ap.add_argument("--state-dtype", choices=("fp16", "fp32"), default="fp32",
                    help="DeltaNet S storage only; all arithmetic remains FP32. fp16 needs firmware v3.4, "
                         "changes outputs and measured slower on the KV260")
    args = ap.parse_args()
    out = Path(args.out)
    if args.chunks_only:
        chunks_only(args)
        return 0
    if not args.check_only:
        export(args)
        selftest(out)
    check(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
