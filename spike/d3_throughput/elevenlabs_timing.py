"""D3 spike — ElevenLabs eleven_v3 TTS timing under parallelism (THROWAWAY).

Cloud TTS parallelizes (httpx releases GIL on network I/O), unlike MLX.
Question: does parallel ElevenLabs clear realtime AT FULL QUALITY, making it
the watch-along TTS instead of low-step OmniVoice?

Pre-warms IVC voices serially (so upload latency doesn't pollute synth timing),
then times an 8-sentence subset at concurrency 1 / 4 / 8.

NOTE: spends ElevenLabs credits. Auto-deletes uploaded clone voices on exit.
The Fallout test clip is real actors -> IVC may trip captcha 403 and fall back
to the default voice; timing is still valid (quality of cloning is not the
point here, latency is).

Run: .venv/bin/python spike/d3_throughput/elevenlabs_timing.py
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "runs/20260528-010843-source"
VIDEO_DURATION = 127.321
SUBSET = 8


def load_env():
    env = REPO / ".env"
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def main():
    load_env()
    from patmir.eval.candidates.tts.elevenlabs import build

    refs_raw = json.loads((RUN / "diarize/refs.json").read_text())
    refs = {(v["speaker_id"], v["emotion"]): (v["ref_audio"], v["ref_text"]) for v in refs_raw.values()}
    sents = [s for s in json.loads((RUN / "translated.json").read_text())["sentences"]
             if s["target_text"].strip()][:SUBSET]

    def ref_for(s):
        return (refs.get((s["speaker_id"], s.get("emotion") or "neutral"))
                or refs.get((s["speaker_id"], "neutral"))
                or refs.get(("SPEAKER_00", "neutral")))

    cfg = {"model_id": "eleven_v3", "language_code": "hy", "cleanup_voices": True}
    model = build(cfg)

    # output audio durations of these sentences are unknown until synth; we
    # measure wall vs VIDEO duration share for comparability with other probes.
    out_audio_total = {}

    def synth(s):
        ra, rt = ref_for(s)
        t0 = time.perf_counter()
        wb = model.synthesize(s["target_text"], ref_audio=ra, ref_text=rt, emotion=s.get("emotion"))
        dt = time.perf_counter() - t0
        import io, wave
        with wave.open(io.BytesIO(wb), "rb") as wf:
            dur = wf.getnframes() / wf.getframerate()
        return dt, dur

    # Pre-warm: one call per unique speaker (serial) to do IVC uploads up front.
    print("pre-warming IVC voices (serial) ...", flush=True)
    seen = set()
    for s in sents:
        if s["speaker_id"] not in seen:
            seen.add(s["speaker_id"])
            dt, _ = synth(s)
            print(f"  warm {s['speaker_id']}: {dt:.2f}s", flush=True)

    rows = []
    for conc in (1, 4, 8):
        print(f"\n>>> concurrency={conc}", flush=True)
        t_start = time.perf_counter()
        if conc == 1:
            results = [synth(s) for s in sents]
        else:
            with ThreadPoolExecutor(max_workers=conc) as ex:
                results = list(ex.map(synth, sents))
        wall = time.perf_counter() - t_start
        per = [r[0] for r in results]
        out_audio = sum(r[1] for r in results)
        row = {
            "concurrency": conc,
            "n": len(sents),
            "wall_s": round(wall, 1),
            "mean_per_sentence_s": round(sum(per) / len(per), 2),
            "output_audio_s": round(out_audio, 1),
            "rtf_compute_per_dub_s": round(wall / out_audio, 2),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    summary = model.summarize() if hasattr(model, "summarize") else {}
    out = {"video_duration_s": VIDEO_DURATION, "rows": rows,
           "silent_fallback_lines": summary.get("silent_fallback_lines"),
           "speaker_status": {k: v.get("status") for k, v in (summary.get("speakers") or {}).items()}}
    (REPO / "spike/d3_throughput/elevenlabs_timing_results.json").write_text(json.dumps(out, indent=2))
    print("\n=== SUMMARY (RTF = compute-s per dub-s; <1.0 = faster than realtime) ===")
    for r in rows:
        print(f"conc={r['concurrency']}  wall={r['wall_s']:>5}s  "
              f"per_sent={r['mean_per_sentence_s']:>5}s  RTF={r['rtf_compute_per_dub_s']}")
    print(f"silent_fallback_lines={out['silent_fallback_lines']}  speakers={out['speaker_status']}")


if __name__ == "__main__":
    main()
