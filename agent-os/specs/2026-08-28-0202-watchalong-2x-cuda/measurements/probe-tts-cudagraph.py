"""Plan Task 3(b): can the Higgs decode step be CUDA-graph captured, and what does it buy?

Task 2 attributed a 43.13 ms/step (B=8, kv=300) as:
    weight bandwidth floor    9.06 ms  (21%)
    sampler loop              4.73 ms  (11%)
    residual launch/dispatch  29.34 ms (68%)   <- this probe targets the 68%

A graph replay collapses the whole block stack into one launch, so if the residual
really is dispatch overhead, replay should move a long way toward the bandwidth floor.

The catch this probe exists to price: capture needs STATIC shapes, so the KV cache
must be a `StaticCache` sized to its maximum, and every step then reads the whole
padded cache instead of just the live prefix. Task 2 measured that at a 9.416 ms
slope per 1000 KV tokens -- so a graph sized for 1400 tokens pays ~13 ms of KV read
from step 1, against ~2.8 ms for a dynamic cache at kv=300. Bucketing (capture
several lengths, replay the smallest that fits) is what makes it pay, so the buckets
are measured rather than assumed: `cap=384` is the small bucket, `cap=1408` the
large one, and the kv=300/cap=1408 row is what a SINGLE full-size graph would cost.

**One config per process, by construction.** The first version of this probe ran all
configs in one process and SEGFAULTED inside `g.replay()`. Capture and replay were
then verified sound in isolation at four levels (plain mm, sdpa+mask, full
`model.model`, and after a dynamic-cache run with and without `empty_cache()`) -- all
fine. So the crash is contamination between configs, not a property of graphs on this
box, and the fix is process isolation rather than a workaround whose failure would be
silently mismeasured. `main()` runs the child processes; a child does exactly one.

Run on an IDLE card. `python probe-tts-cudagraph.py`
"""
import json
import subprocess
import sys
import time

B = 8
VRAM_BUDGET_MB = 2048
FLOOR_MS = 9.06          # Task 2: 7.31 GB decode-active weights / 807 GB/s

# (kvlen, cap | None). cap None = DynamicCache baseline, i.e. what ships today.
CONFIGS = [
    (300, None),
    (1200, None),
    (300, 384),
    (300, 1408),
    (1200, 1408),
]


def t_ms(f, iters=40, warm=8):
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


def child(kvlen: int, cap: int | None) -> dict:
    """Measure ONE config. Emits a JSON line the parent collects."""
    import torch

    from patmir.eval.candidates.tts.higgs_v3_torch import REPO_DEFAULT, HiggsV3Torch

    require_idle_card()
    torch.backends.cuda.matmul.allow_tf32 = True
    model = HiggsV3Torch(repo_id=REPO_DEFAULT, device="cuda", dtype="bfloat16").model
    dev = next(model.parameters()).device
    dtype = torch.bfloat16
    hid = model.config.text_config.hidden_size
    res: dict = {"kv": kvlen, "cap": cap}

    with torch.inference_mode():
        emb = torch.randn(B, kvlen, hid, device=dev, dtype=dtype)
        if cap is None:
            mask = torch.ones(B, kvlen, dtype=torch.long, device=dev)
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            out = model.model(inputs_embeds=emb, attention_mask=mask, position_ids=pos,
                              use_cache=True,
                              cache_position=torch.arange(kvlen, device=dev))
            past = out.past_key_values
            step_mask = torch.cat(
                [mask, torch.ones(B, 1, dtype=mask.dtype, device=dev)], dim=1)
        else:
            from transformers import StaticCache
            past = StaticCache(config=model.config.text_config, max_batch_size=B,
                               max_cache_len=cap, device=dev, dtype=dtype)
            full_mask = torch.zeros(B, cap, dtype=torch.long, device=dev)
            full_mask[:, :kvlen] = 1
            pos = torch.arange(kvlen, device=dev).unsqueeze(0).expand(B, -1)
            model.model(inputs_embeds=emb, attention_mask=full_mask[:, :kvlen],
                        position_ids=pos, past_key_values=past, use_cache=True,
                        cache_position=torch.arange(kvlen, device=dev))
            step_mask = full_mask.clone()
            step_mask[:, kvlen] = 1

        # Static step buffers. The graph reads these addresses, so a real
        # implementation would write the next token's values INTO them.
        s_emb = torch.randn(B, 1, hid, device=dev, dtype=dtype)
        s_pos = torch.full((B, 1), kvlen, dtype=torch.long, device=dev)
        s_cpos = torch.tensor([kvlen], device=dev)

        def go():
            return model.model(inputs_embeds=s_emb, attention_mask=step_mask,
                               position_ids=s_pos, past_key_values=past,
                               use_cache=True, cache_position=s_cpos)

        res["eager_ms"] = t_ms(go)

        if cap is not None:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    go()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                go()
            torch.cuda.synchronize()
            res["graph_ms"] = t_ms(g.replay)
            res["capture_vram_gb"] = round(
                (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1e9, 2)
    return res


def main():
    import torch
    print(f"card idle: {require_idle_card():.0f} MB in use")
    print(f"\nB={B}   bandwidth floor {FLOOR_MS:.2f} ms/step (Task 2)")
    print("one process per config -- see module docstring\n")
    print(f"{'config':<30}{'eager':>9}{'graph':>9}{'x floor':>9}{'speedup':>9}")
    print("-" * 66)

    rows = []
    for kv, cap in CONFIGS:
        proc = subprocess.run(
            [sys.executable, "-u", __file__, "--child", str(kv), str(cap)],
            capture_output=True, text=True)
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
        if not line:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
            print(f"{f'kv={kv} cap={cap}':<30}  FAILED rc={proc.returncode}: {' | '.join(tail)[:120]}")
            continue
        r = json.loads(line[-1])
        rows.append(r)
        tag = f"kv={kv} " + ("DynamicCache" if cap is None else f"static cap={cap}")
        eager, graph = r["eager_ms"], r.get("graph_ms")
        best = graph if graph else eager
        speed = f"{eager / graph:.2f}x" if graph else "--"
        print(f"{tag:<30}{eager:>9.2f}{(f'{graph:.2f}' if graph else '--'):>9}"
              f"{best / FLOOR_MS:>9.1f}{speed:>9}")

    base = {r["kv"]: r["eager_ms"] for r in rows if r["cap"] is None}
    print("\nvs the DynamicCache baseline that ships today:")
    for r in rows:
        if r["cap"] is None or r["kv"] not in base:
            continue
        g = r.get("graph_ms")
        if g:
            print(f"  kv={r['kv']:>4} cap={r['cap']:>5}: {base[r['kv']]:.2f} -> {g:.2f} ms"
                  f"  = {base[r['kv']] / g:.2f}x on the forward")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        kv = int(sys.argv[2])
        cap = None if sys.argv[3] == "None" else int(sys.argv[3])
        print(json.dumps(child(kv, cap)), flush=True)
    else:
        main()
