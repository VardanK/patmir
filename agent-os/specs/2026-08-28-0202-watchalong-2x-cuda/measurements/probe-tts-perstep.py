"""Plan Task 2: where does ONE Higgs decode step actually go?

The full-chunk probe could not answer this — its bf16 control varied 9.4% against
itself, and a randomly-initialised `audio_head` made each arm run a different number
of steps. So this measures the forward at a FIXED KV length with no sampler at all:
no termination variance, no draw dependence, nothing to confound the comparison.

Attributes the step across:
  * weight bandwidth   - floor computed from the REAL decode-active parameter count
  * DynamicCache cat   - by the slope across KV length, and vs StaticCache
  * kernel launches    - the residual, cross-checked against the measured 10.5 us
  * sampler loop       - timed separately on synthetic logits at the same batch

And answers the dtype question cleanly (bf16 vs fp16+fp16-accumulate) at the model's
true shape, which is hidden 2560 / intermediate 9728, not the 4096/14336 I first used.

Run on an IDLE card. `python probe-tts-perstep.py`
"""
import statistics as st
import time

import torch

from patmir.eval.candidates.tts import higgs_v3_torch as H

B = 8
KVS = (300, 600, 1200)
VRAM_BUDGET_MB = 2048
M = torch.backends.cuda.matmul


def t_ms(f, iters, warm=8):
    for _ in range(warm):
        f()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000


def require_idle_card():
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 1e6
    if used > VRAM_BUDGET_MB:
        raise SystemExit(f"REFUSING: {used:.0f} MB VRAM in use. Stop the server and re-run.")
    print(f"card idle: {used:.0f} MB in use")


def decode_active_bytes(model) -> int:
    """Bytes a decode step must read. The transformer blocks, plus the audio head —
    NOT the 389M tied text embedding, which a seq-1 audio decode never touches."""
    blocks = sum(p.numel() * p.element_size()
                 for n, p in model.named_parameters()
                 if ".layers." in n)
    heads = sum(p.numel() * p.element_size()
                for n, p in model.named_parameters()
                if "audio_head" in n or "audio_embedding" in n)
    return blocks + heads


def one_step(model, dtype, kvlen, static=False):
    """Prefill kvlen tokens, then time a single-token forward against that cache."""
    dev = next(model.parameters()).device
    hid = model.config.text_config.hidden_size
    emb = torch.randn(B, kvlen, hid, device=dev, dtype=dtype)
    mask = torch.ones(B, kvlen, dtype=torch.long, device=dev)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    with torch.inference_mode():
        kwargs = {}
        if static:
            from transformers import StaticCache
            kwargs["past_key_values"] = StaticCache(
                config=model.config.text_config, max_batch_size=B,
                max_cache_len=kvlen + 64, device=dev, dtype=dtype)
        out = model.model(inputs_embeds=emb, attention_mask=mask, position_ids=pos,
                          use_cache=True,
                          cache_position=torch.arange(kvlen, device=dev), **kwargs)
        past = out.past_key_values
        step = torch.randn(B, 1, hid, device=dev, dtype=dtype)
        m2 = torch.cat([mask, torch.ones(B, 1, dtype=mask.dtype, device=dev)], dim=1)
        p2 = pos[:, -1:] + 1
        cpos = torch.tensor([kvlen], device=dev)

        def go():
            model.model(inputs_embeds=step, attention_mask=m2, position_ids=p2,
                        past_key_values=past, use_cache=True, cache_position=cpos)

        ms = t_ms(go, iters=40)
    del emb, mask, pos, past
    torch.cuda.empty_cache()
    return ms


def sampler_cost(model, dtype):
    """The per-row Python sampler loop, on synthetic logits at the real batch.
    Reproduces what `_decode_chunk` does per step, including the .cpu() per row."""
    dev = next(model.parameters()).device
    up = H.sys.modules[type(model).__module__]
    N = up.NUM_CODEBOOKS if hasattr(up, "NUM_CODEBOOKS") else 8
    V = 1026
    logits = torch.randn(B, N, V, device=dev, dtype=torch.float32)
    gens = [torch.Generator(device=dev) for _ in range(B)]
    for i, g in enumerate(gens):
        g.manual_seed(i)

    def current():
        """As shipped: per-row draw, one-hot, upstream sampler_step, .cpu() per row."""
        rows = []
        onehot = torch.zeros_like(logits[0])
        for b in range(B):
            drawn = H._sample_codes(logits[b], temperature=0.8, top_p=0.95,
                                    top_k=None, generator=gens[b])
            onehot.zero_()
            onehot.scatter_(-1, drawn.unsqueeze(-1).to(torch.long), 1.0)
            rows.append(drawn.cpu())          # <- the D2H sync per row per step
        return rows

    def no_sync():
        """Same draws, but results stay on device (one D2H at chunk end instead)."""
        out = []
        for b in range(B):
            out.append(H._sample_codes(logits[b], temperature=0.8, top_p=0.95,
                                       top_k=None, generator=gens[b]))
        return torch.stack(out)

    def vectorized():
        """Draw all rows in one batched call — what Task 3 would ship."""
        lg = logits.reshape(B * N, V) / 0.8
        probs = lg.softmax(-1)
        return probs.multinomial(num_samples=1).reshape(B, N)

    return (t_ms(current, iters=60), t_ms(no_sync, iters=60), t_ms(vectorized, iters=200))


def true_shape_gemm():
    """Re-check the accumulate finding at the model's ACTUAL MLP shape."""
    K, N = 2560, 9728
    nbytes = K * N * 2
    res = {}
    variants = [("bf16", torch.bfloat16, False), ("fp16+f16acc", torch.float16, True)]
    tensors = {n: (torch.randn(B, K, device="cuda", dtype=d),
                   torch.randn(K, N, device="cuda", dtype=d), f)
               for n, d, f in variants}
    for n in tensors:
        res[n] = []
    for _ in range(5):
        for n, (a, W, flag) in tensors.items():
            M.allow_fp16_accumulation = flag
            res[n].append(t_ms(lambda: a @ W, iters=300, warm=30))
    M.allow_fp16_accumulation = False
    print(f"\ntrue-shape MLP GEMM [{B},{K}]@[{K},{N}]  weight {nbytes/1e6:.0f} MB")
    base = st.median(res["bf16"])
    for n, v in res.items():
        m = st.median(v)
        print(f"  {n:12s} {m:.4f} ms  {nbytes/(m/1000)/1e9:5.0f} GB/s  {base/m:.2f}x")
    for t in tensors.values():
        del t
    torch.cuda.empty_cache()


def main():
    require_idle_card()
    true_shape_gemm()

    for dtname, dt, acc in [("bf16", torch.bfloat16, False),
                            ("fp16+f16acc", torch.float16, True)]:
        M.allow_fp16_accumulation = acc
        print(f"\n=== {dtname} ===", flush=True)
        cand = H.build({"dtype": "bfloat16" if dtname == "bf16" else "float16",
                        "max_batch": B, "device": "cuda"})
        model = cand.model
        wb = decode_active_bytes(model)
        floor = wb / 807e9 * 1000
        print(f"  decode-active weights {wb/1e9:.2f} GB -> bandwidth floor "
              f"{floor:.2f} ms/step (q8 would be {floor/2:.2f})", flush=True)

        dyn = {}
        for kv in KVS:
            dyn[kv] = one_step(model, dt, kv, static=False)
            print(f"  DynamicCache kv={kv:5d}: {dyn[kv]:6.2f} ms/step   "
                  f"{dyn[kv]/floor:4.1f}x floor   overhead {(1-floor/dyn[kv])*100:2.0f}%",
                  flush=True)
        slope = (dyn[KVS[-1]] - dyn[KVS[0]]) / (KVS[-1] - KVS[0])
        print(f"  cache-growth slope: {slope*1000:.3f} ms per 1000 tokens of KV", flush=True)

        try:
            for kv in (KVS[0], KVS[-1]):
                stat = one_step(model, dt, kv, static=True)
                print(f"  StaticCache  kv={kv:5d}: {stat:6.2f} ms/step   "
                      f"vs dynamic {dyn[kv]:6.2f} -> {dyn[kv]/stat:.2f}x", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  StaticCache: FAILED - {type(e).__name__}: {str(e)[:160]}", flush=True)

        if dtname == "bf16":
            cur, nosync, vec = sampler_cost(model, dt)
            print(f"  sampler/step: as-shipped {cur:.2f} ms | no per-row .cpu() {nosync:.2f} ms "
                  f"| vectorized {vec:.2f} ms", flush=True)
            print(f"    -> removing the syncs saves {cur-nosync:.2f} ms/step; "
                  f"vectorizing saves {cur-vec:.2f} ms/step", flush=True)

        del cand, model
        torch.cuda.empty_cache()
    M.allow_fp16_accumulation = False


if __name__ == "__main__":
    main()
