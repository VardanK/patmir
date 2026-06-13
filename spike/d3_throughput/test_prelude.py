"""Integration: run the #8 first-chunk prelude end-to-end on the real source.

Slices the opening 60s, runs ingest→separate→diarize→emotion→asr, then builds
the populated SpeakerRegistry (profile + voice + ECAPA embedding per speaker).

.venv/bin/python spike/d3_throughput/test_prelude.py
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE = Path("/Users/vardan.kurkchiyan/.patmir/downloads/183f3cc80610ea04/source.webm")

NAMES = {
    "JBFqnCBsd6RMkjVDRZzb": "George", "EXAVITQu4vr4xnSDxMaL": "Sarah",
    "N2lVS1w4EtoT3dr4eOWO": "Callum", "cgSgspJ2msm6clMCkdW9": "Jessica",
    "IKne3meq5aSn9XLyUdCD": "Charlie", "FGY2WhTYpPnrIDTdsKH5": "Laura",
    "SOYHLrjzK2X1ezoPC6cr": "Harry", "pqHfZKP75CvOlQylNhV4": "Bill",
}


def load_env():
    for line in (REPO / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main():
    load_env()
    from patmir.pipeline.config import load_pipeline_config
    from patmir.pipeline.streaming.prelude import run_prelude

    cfg = load_pipeline_config(REPO / "patmir.toml")
    toml = tomllib.load(open(REPO / "patmir.toml", "rb"))
    pool = next(c for c in toml["eval"]["tts"]["candidates"] if c["id"] == "elevenlabs_pool")["voice_pool"]

    print(f"running prelude on {SOURCE.name} (first 60s) ...", flush=True)
    result = run_prelude(
        SOURCE, cfg, window_seconds=60.0,
        runs_root=REPO / "runs", voice_pool=pool,
    )

    print(f"\nwindow_end (section 1 starts): {result.window_end:.2f}s")
    print(f"analysis job dir: {result.job_dir.name}")
    print(f"cast established ({len(result.speakers)} speakers):")
    for spk, info in sorted(result.speakers.items()):
        print(f"  {spk}: {NAMES.get(info['voice_id'], info['voice_id'])} "
              f"(gender={info['gender']}, age={info['age']}, embedding={info['has_embedding']})")

    # sanity: every speaker got a voice + embedding
    assert result.speakers, "prelude found no speakers"
    assert all(i["has_embedding"] for i in result.speakers.values()), "missing embeddings"
    assert all(i["voice_id"] for i in result.speakers.values()), "missing voice"
    # sanity: resolve a known speaker's own embedding back to itself
    reg = result.registry
    any_spk = next(iter(result.speakers))
    emb = reg._embeddings[any_spk]
    assert reg.resolve(emb) == any_spk, "self-resolve failed"
    print("\nPASS — registry populated, self-resolve OK")


if __name__ == "__main__":
    main()
