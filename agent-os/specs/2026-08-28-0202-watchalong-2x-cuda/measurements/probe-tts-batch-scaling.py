"""Plan Task 3, revised: is the Higgs decode step actually launch-bound, or is it
running its kernels at a fraction of bandwidth roof?

Task 2 attributed a 43 ms step (B=8, kv=300) as 9.06 ms of weight bandwidth, 4.73 ms
of sampler, and ~29 ms of "kernel launch / dispatch". Task 3(b) then tested that
attribution the only way that settles it -- by removing launches entirely with a CUDA
graph -- and got **1.18x**, not the ~4x the attribution predicts. So the residual is
NOT dispatch overhead. With launches gone the step still needs ~42 ms to move ~9 GB,
i.e. ~214 GB/s against a measured roof of 807 GB/s: about 26%.

That reframes the whole stage. If the kernels are inefficient because M=8 gives them
too little work per byte of weight read, then the fix is not to make the step faster
but to put MORE SENTENCES IN IT -- time per step stays flat while throughput scales,
which is exactly the shape of a memory-bound decode below its ridge point.

This probe measures that directly: step time vs batch size, at the model's real
shapes. Three things fall out of one sweep --

  * per-step time vs B          (flat => batching is nearly free)
  * per-sentence time vs B      (the number that sets TTS wall)
  * achieved bandwidth vs B     (how far from the 807 GB/s roof each B gets)

plus the standalone GEMM at the same M, so a gap between "one big GEMM" and "the
whole model" is attributable to the model rather than to the hardware.

Run on an IDLE card. `python probe-tts-batch-scaling.py`
"""
import json
import subprocess
import sys
import time

BATCHES = (1, 2, 4, 8, 16, 24, 32)
KV = 300
VRAM_BUDGET_MB = 2048
ROOF_GBS = 807.0        # measured on this card, idle (Task 2 hardware probe)


def t_ms(f, iters=30, warm=6):
    import torch
    for _ in range(warm):
        f()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000


def require_idle_card():
    import torch
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 1e6
    if used > VRAM_BUDGET_MB:
        raise SystemExit(f"REFUSING: {used:.0f} MB VRAM in use. Stop the server and re-run.")
    return used


def decode_active_bytes(model) -> int:
    """Bytes a decode step must read: the transformer blocks plus the audio head.
    NOT the tied text embedding, which a seq-1 audio decode never touches."""
    blocks = sum(p.numel() * p.element_size()
                 for n, p in model.named_parameters() if ".layers." in n)
    heads = sum(p.numel() * p.element_size()
                for n, p in model.named_parameters()
                if "audio_head" in n or "audio_embedding" in n)
    return blocks + heads


def kv_bytes(model, B: int, kv: int) -> int:
    tc = model.config.text_config
    n_kv = getattr(tc, "num_key_value_heads", tc.num_attention_heads)
    head_dim = getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads)
    return tc.num_hidden_layers * 2 * B * kv * n_kv * head_dim * 2


def child(B: int) -> dict:
    import torch

    from patmir.eval.candidates.tts.higgs_v3_torch import REPO_DEFAULT, HiggsV3Torch

    require_idle_card()
    model = HiggsV3Torch(repo_id=REPO_DEFAULT, device="cuda", dtype="bfloat16").model
    dev = next(model.parameters()).device
    dtype = torch.bfloat16
    hid = model.config.text_config.hidden_size
    inter = model.config.text_config.intermediate_size

    res = {"B": B, "weight_bytes": decode_active_bytes(model),
           "kv_bytes": kv_bytes(model, B, KV)}

    # Control: one standalone MLP-shaped GEMM at this M. If the model is far from
    # this, the gap is the model's, not the card's.
    a = torch.randn(B, hid, device=dev, dtype=dtype)
    w = torch.randn(inter, hid, device=dev, dtype=dtype)     # nn.Linear layout
    res["gemm_ms"] = t_ms(lambda: torch.nn.functional.linear(a, w), iters=200, warm=50)
    res["gemm_gbs"] = (w.numel() * 2) / (res["gemm_ms"] / 1000) / 1e9
    del a, w
    torch.cuda.empty_cache()

    with torch.inference_mode():
        emb = torch.randn(B, KV, hid, device=dev, dtype=dtype)
        mask = torch.ones(B, KV, dtype=torch.long, device=dev)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        out = model.model(inputs_embeds=emb, attention_mask=mask, position_ids=pos,
                          use_cache=True, cache_position=torch.arange(KV, device=dev))
        past = out.past_key_values
        s_emb = torch.randn(B, 1, hid, device=dev, dtype=dtype)
        s_mask = torch.cat([mask, torch.ones(B, 1, dtype=mask.dtype, device=dev)], dim=1)
        s_pos = pos[:, -1:] + 1
        s_cpos = torch.tensor([KV], device=dev)

        def go():
            model.model(inputs_embeds=s_emb, attention_mask=s_mask, position_ids=s_pos,
                        past_key_values=past, use_cache=True, cache_position=s_cpos)

        res["step_ms"] = t_ms(go)
    res["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return res


def main():
    print(f"card idle: {require_idle_card():.0f} MB in use")
    print(f"\nHiggs decode step vs batch size, kv={KV}, bf16, bandwidth roof {ROOF_GBS:.0f} GB/s")
    print("per-sentence is what sets TTS wall; flat step time means batching is free\n")
    print(f"{'B':>4}{'step ms':>10}{'per-sent':>10}{'GB moved':>10}{'GB/s':>9}"
          f"{'% roof':>8}{'gemm GB/s':>11}{'peak GB':>9}")
    print("-" * 71)

    rows = []
    for B in BATCHES:
        proc = subprocess.run([sys.executable, "-u", __file__, "--child", str(B)],
                              capture_output=True, text=True)
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
        if not line:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-2:]
            print(f"{B:>4}   FAILED rc={proc.returncode}: {' | '.join(tail)[:100]}")
            continue
        r = json.loads(line[-1])
        rows.append(r)
        gb = (r["weight_bytes"] + r["kv_bytes"]) / 1e9
        gbs = gb / (r["step_ms"] / 1000)
        print(f"{B:>4}{r['step_ms']:>10.2f}{r['step_ms'] / B:>10.2f}{gb:>10.2f}"
              f"{gbs:>9.0f}{100 * gbs / ROOF_GBS:>7.0f}%{r['gemm_gbs']:>11.0f}"
              f"{r['peak_gb']:>9.2f}")

    if rows:
        base = next((r for r in rows if r["B"] == 8), rows[0])
        print(f"\nrelative to B={base['B']} (what ships today), per sentence:")
        for r in rows:
            rel = (base["step_ms"] / base["B"]) / (r["step_ms"] / r["B"])
            print(f"  B={r['B']:>3}: {rel:>5.2f}x")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        print(json.dumps(child(int(sys.argv[2]))), flush=True)
    else:
        main()
