"""End-to-end #9 proof: prelude -> section 1, verifying cross-section identity.

The crux: a character established in the prelude who also speaks in section 1
must resolve to the SAME global voice (not spawn a new one). Prints the prelude
cast + the section's local→global map + any new speakers + the dub chunk.

.venv/bin/python spike/d3_throughput/test_streaming_e2e.py
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


def vid(v):
    return NAMES.get(v, v)


def main():
    load_env()
    from patmir.pipeline.streaming.prelude import run_prelude
    from patmir.pipeline.streaming.runner import streaming_config
    from patmir.pipeline.streaming.section import process_section

    cfg = streaming_config(REPO / "patmir.toml")
    toml = tomllib.load(open(REPO / "patmir.toml", "rb"))
    pool = next(c for c in toml["eval"]["tts"]["candidates"] if c["id"] == "elevenlabs_pool")["voice_pool"]
    runs_root = REPO / "runs"

    print("=== PRELUDE (first 60s) ===", flush=True)
    prelude = run_prelude(SOURCE, cfg, window_seconds=60.0, runs_root=runs_root, voice_pool=pool)
    print(f"window_end={prelude.window_end:.1f}s, cast:")
    for spk, info in sorted(prelude.speakers.items()):
        print(f"  {spk} -> {vid(info['voice_id'])} ({info['gender']}/{info['age']})")

    print(f"\n=== SECTION 1 (from {prelude.window_end:.1f}s) ===", flush=True)
    sec = process_section(
        SOURCE, cfg, prelude.registry,
        index=1, start=prelude.window_end, target_seconds=30.0, runs_root=runs_root,
    )
    print(f"section [{sec.start:.1f}-{sec.end:.1f}]s  dub={sec.dub_wav}")
    print(f"local→global identity map:")
    for local, glob in sec.local_to_global.items():
        tag = "NEW" if glob in sec.new_speakers else "matched-to-prelude"
        print(f"  {local} -> {glob} ({vid(prelude.registry.assignments().get(glob, glob))}) [{tag}]")
    print(f"new speakers this section: {sec.new_speakers or 'none'}")

    # Proof: at least one section speaker matched an existing prelude character.
    matched = [g for g in sec.local_to_global.values() if g not in sec.new_speakers]
    if matched:
        print(f"\nPASS: cross-section identity works — {len(matched)} speaker(s) "
              f"re-identified from the prelude (same voice kept).")
    else:
        print("\nNOTE: no section speaker matched the prelude cast "
              "(all new) — check threshold / whether the same characters speak here.")
    assert sec.dub_wav is not None and sec.dub_wav.exists(), "no section dub produced"
    print(f"section dub chunk: {sec.dub_wav}")


if __name__ == "__main__":
    main()
