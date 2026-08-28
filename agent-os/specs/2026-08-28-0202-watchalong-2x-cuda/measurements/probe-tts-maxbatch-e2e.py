"""Plan Task 3, revised: what does raising `max_batch` buy on a REAL watch-along section?

`probe-tts-batch-scaling.py` measured the forward alone: step time is flat from B=1 to
B=24 (45-49 ms) while per-sentence cost falls linearly, so batching is nearly free up
to 24. That is a microbenchmark, and a microbenchmark is not a plan -- three things
could eat the win in production:

  * a chunk costs `max(steps)` over its rows, so a bigger chunk mixes more length
    variance and its max grows;
  * a watch-along section only has ~10-12 sentences, so above ~12 there is nothing
    left to batch and the curve must flatten;
  * VRAM has to hold the bigger KV cache alongside everything else the pipeline has
    resident (the accelerator lock means one pipeline, but that pipeline is ~16 GB).

So this drives the real `HiggsV3Torch.synthesize_batch` over a real section's real
translated sentences and real pinned refs, at each `max_batch`, and reports wall,
chunk count, peak VRAM and per-sentence audio md5.

**The md5s are expected to CHANGE between arms, and that is not a bug.** Per ADR
2026-08-24-tts-sampling-reproducibility the decode is reproducible but NOT
batch-invariant: batched matmuls and the left-padded prefill are not bitwise-stable
across shapes, and at temperature 0.8 a one-bit logit difference flips a multinomial
draw. Changing `max_batch` changes which sentences share a chunk, so lines get
redrawn. The md5 column is here to show HOW MANY lines changed, which is the size of
the ear-test this knob needs -- not as a pass/fail.

One process per arm: the CUDA-graph probe segfaulted from cross-arm contamination in
a single process, and the same discipline applies here.

Run on an IDLE card, with a section dir that has translated.json + diarize/refs.json.
`python probe-tts-maxbatch-e2e.py [section_dir]`
"""
import hashlib
import json
import os
import subprocess
import sys
import time

SECTION = "/home/vardan/patmir/runs/20260825-182659-section"
ARMS = (8, 12, 16, 24)
VRAM_BUDGET_MB = 2048


def require_idle_card():
    import torch
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 1e6
    if used > VRAM_BUDGET_MB:
        raise SystemExit(f"REFUSING: {used:.0f} MB VRAM in use. Stop the server and re-run.")
    return used


def load_requests(section: str) -> list[dict]:
    """Real translated sentences + the ref each would actually clone from.

    Ref lookup mirrors the TTS adapter: `(speaker, emotion)` first, then that
    speaker's neutral cell. A sentence whose speaker has no ref at all is kept with
    `ref_audio=None` rather than dropped -- that is a real case in production and it
    changes the prompt length, so removing it would flatter the measurement.
    """
    d = json.load(open(os.path.join(section, "translated.json")))
    sents = d["sentences"] if isinstance(d, dict) else d
    refs = json.load(open(os.path.join(section, "diarize", "refs.json")))
    out = []
    for s in sents:
        spk, emo = s.get("speaker_id"), s.get("emotion") or "neutral"
        cell = refs.get(f"{spk}__{emo}") or refs.get(f"{spk}__neutral")
        out.append({
            "text": s["target_text"],
            "ref_audio": cell["ref_audio"] if cell else None,
            "ref_text": cell.get("ref_text") if cell else None,
            "emotion": emo,
        })
    return out


def child(section: str, max_batch: int) -> dict:
    import torch

    from patmir.eval.candidates.tts.higgs_v3_torch import REPO_DEFAULT, HiggsV3Torch

    require_idle_card()
    reqs = load_requests(section)
    cand = HiggsV3Torch(repo_id=REPO_DEFAULT, device="cuda", dtype="bfloat16",
                        max_batch=max_batch, max_retries=2)

    chunks: list[dict] = []

    def on_chunk(indices, wall_s):
        chunks.append({"n": len(indices), "wall_s": round(wall_s, 2)})

    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    wavs = cand.synthesize_batch(reqs, on_chunk=on_chunk)
    wall = time.perf_counter() - t

    audio_s = 0.0
    md5s = []
    for w in wavs:
        md5s.append(hashlib.md5(w).hexdigest()[:8])
        # WAV: 44-byte canonical header, 16-bit mono at the codec's 24 kHz
        audio_s += max(0, len(w) - 44) / 2 / 24000
    return {
        "max_batch": max_batch, "n_sentences": len(reqs),
        "wall_s": round(wall, 2), "audio_s": round(audio_s, 2),
        "rtf": round(wall / audio_s, 3) if audio_s else None,
        "chunks": chunks, "md5s": md5s,
        "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "overflows": cand._summary.get("overflows"),
    }


def main():
    section = sys.argv[1] if len(sys.argv) > 1 else SECTION
    print(f"card idle: {require_idle_card():.0f} MB in use")
    print(f"section: {section}")
    print(f"sentences: {len(load_requests(section))}\n")
    print(f"{'max_batch':>10}{'wall s':>9}{'rtf':>8}{'chunks':>9}{'peak GB':>9}"
          f"{'overflow':>10}{'lines changed':>15}")
    print("-" * 70)

    rows = []
    for mb in ARMS:
        proc = subprocess.run(
            [sys.executable, "-u", __file__, "--child", section, str(mb)],
            capture_output=True, text=True)
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
        if not line:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-2:]
            print(f"{mb:>10}   FAILED rc={proc.returncode}: {' | '.join(tail)[:90]}")
            continue
        r = json.loads(line[-1])
        rows.append(r)
        shape = "+".join(str(c["n"]) for c in r["chunks"])
        changed = ("--" if not rows[:-1]
                   else f"{sum(a != b for a, b in zip(rows[0]['md5s'], r['md5s']))}"
                        f"/{len(r['md5s'])}")
        print(f"{mb:>10}{r['wall_s']:>9.2f}{r['rtf']:>8.3f}{shape:>9}{r['peak_gb']:>9.2f}"
              f"{str(r['overflows']):>10}{changed:>15}")

    if rows:
        base = rows[0]
        print(f"\nvs max_batch={base['max_batch']} (what ships today):")
        for r in rows[1:]:
            print(f"  max_batch={r['max_batch']:>3}: {base['wall_s']:.2f} -> {r['wall_s']:.2f} s"
                  f"  = {base['wall_s'] / r['wall_s']:.2f}x   (+{r['peak_gb'] - base['peak_gb']:.2f} GB peak)")
        print("\nmd5 changes are EXPECTED -- the decode is reproducible but not")
        print("batch-invariant (ADR 2026-08-24). The count is the size of the ear test.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        print(json.dumps(child(sys.argv[2], int(sys.argv[3]))), flush=True)
    else:
        main()
