"""D3 throughput spike — TTS sweep (THROWAWAY, not a pipeline candidate).

Question: can OmniVoice MLX TTS get under ~0.8x realtime on this Mac so a
watch-along buffer doesn't drain? TTS is 74% of pipeline wall-clock and the
single feasibility blocker (~2.9x video length at num_steps=32, sequential).

Levers measured:
  1. num_steps sweep (32 / 16 / 8) at concurrency 1 — diffusion-loop steps are
     a near-linear, no-architecture-change speed knob the original plan missed.
  2. thread concurrency sweep (1 / 2 / 4) at num_steps=32 — MLX serializes on a
     single GPU stream, so this likely won't scale; we measure to confirm.

Metrics per config: total wall-clock, mean wall per sentence, total output
audio duration, and RTF = wall / output_audio_seconds (compute-seconds per
second-of-dub — the buffer-drain-relevant number). Also projects against the
127.3s source clip for comparability with the manifest baseline (369s).

Run: .venv/bin/python spike/d3_throughput/tts_sweep.py
"""
from __future__ import annotations

import io
import json
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "runs/20260528-010843-source"
VIDEO_DURATION = 127.321  # audio.meta.json
SUBSET = 10               # sentences per config (keeps the sweep tractable)

# --- candidate config, mirrors patmir.toml [[eval.tts.candidates]] omnivoice_mlx_q8
CFG = {
    "repo_id": "lightsofapollo/omnivoice-mlx-q8-g64",
    "ref_audio": str(RUN / "diarize/refs/SPEAKER_00/neutral.wav"),  # existing fallback
    "ref_text": "sure?",
    "num_steps": 32,
    "guidance_scale": 2.0,
    "language": "None",
    "lang_code": "None",
}


def load_sentences():
    d = json.loads((RUN / "translated.json").read_text())
    return d["sentences"] if isinstance(d, dict) else d


def load_refs():
    d = json.loads((RUN / "diarize/refs.json").read_text())
    # (speaker, emotion) -> (ref_audio, ref_text)
    refs = {}
    for v in d.values():
        refs[(v["speaker_id"], v["emotion"])] = (v["ref_audio"], v["ref_text"])
    return refs


def lookup_ref(refs, speaker_id, emotion):
    """Mirror of adapter._lookup_ref fallback chain (speaker first, emotion second)."""
    emo = emotion or "neutral"
    if (speaker_id, emo) in refs:
        return refs[(speaker_id, emo)]
    if (speaker_id, "neutral") in refs:
        return refs[(speaker_id, "neutral")]
    same = [v for (spk, _e), v in refs.items() if spk == speaker_id]
    if same:
        return same[0]
    # global fallback: SPEAKER_00 neutral (always exists in this run)
    return refs.get(("SPEAKER_00", "neutral"))


def wav_duration(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def run_config(model, sentences, refs, *, num_steps, concurrency):
    model.num_steps = num_steps  # candidate reads self.num_steps per call

    def synth(s):
        ref = lookup_ref(refs, s.get("speaker_id"), s.get("emotion"))
        ra, rt = (ref if ref else (CFG["ref_audio"], CFG["ref_text"]))
        t0 = time.perf_counter()
        wb = model.synthesize(s["target_text"], ref_audio=ra, ref_text=rt)
        return time.perf_counter() - t0, wav_duration(wb)

    items = [s for s in sentences if s["target_text"].strip()][:SUBSET]
    t_start = time.perf_counter()
    if concurrency == 1:
        results = [synth(s) for s in items]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            results = list(ex.map(synth, items))
    wall = time.perf_counter() - t_start

    per_sentence = [r[0] for r in results]
    out_audio = sum(r[1] for r in results)
    return {
        "num_steps": num_steps,
        "concurrency": concurrency,
        "n": len(items),
        "wall_s": round(wall, 1),
        "mean_per_sentence_s": round(sum(per_sentence) / len(per_sentence), 2),
        "output_audio_s": round(out_audio, 1),
        "rtf_compute_per_dub_s": round(wall / out_audio, 2),
    }


def main():
    from patmir.eval.candidates.tts.omnivoice_mlx import build

    print("loading OmniVoice MLX q8 ...", flush=True)
    model = build(CFG)
    sentences = load_sentences()
    refs = load_refs()
    print(f"{len([s for s in sentences if s['target_text'].strip()])} non-empty sentences; "
          f"sweeping subset of {SUBSET}", flush=True)

    # warm-up (first call pays lazy compile / model warm-up — exclude from timing)
    print("warm-up ...", flush=True)
    s0 = next(s for s in sentences if s["target_text"].strip())
    ref = lookup_ref(refs, s0.get("speaker_id"), s0.get("emotion"))
    model.num_steps = 8
    model.synthesize(s0["target_text"], ref_audio=ref[0], ref_text=ref[1])

    configs = [
        dict(num_steps=32, concurrency=1),   # re-baseline (apples-to-apples on subset)
        dict(num_steps=16, concurrency=1),
        dict(num_steps=8, concurrency=1),
        dict(num_steps=32, concurrency=2),
        dict(num_steps=32, concurrency=4),
    ]
    rows = []
    for c in configs:
        print(f"\n>>> num_steps={c['num_steps']} concurrency={c['concurrency']}", flush=True)
        r = run_config(model, sentences, refs, **c)
        rows.append(r)
        print(json.dumps(r), flush=True)

    out = REPO / "spike/d3_throughput/tts_sweep_results.json"
    out.write_text(json.dumps({"video_duration_s": VIDEO_DURATION, "rows": rows}, indent=2))
    print(f"\nwrote {out}")
    print("\n=== SUMMARY (RTF = compute-seconds per second-of-dubbed-audio; <1.0 = faster than realtime) ===")
    for r in rows:
        print(f"steps={r['num_steps']:>2} conc={r['concurrency']}  "
              f"wall={r['wall_s']:>6}s  per_sent={r['mean_per_sentence_s']:>5}s  "
              f"RTF={r['rtf_compute_per_dub_s']}")


if __name__ == "__main__":
    main()
