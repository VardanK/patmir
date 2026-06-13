"""End-to-end #7 check: profile real speakers + assign matched pool voices.
.venv/bin/python spike/d3_throughput/test_profile_assign.py
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "runs/20260528-010843-source"

NAMES = {
    "JBFqnCBsd6RMkjVDRZzb": "George", "EXAVITQu4vr4xnSDxMaL": "Sarah",
    "N2lVS1w4EtoT3dr4eOWO": "Callum", "cgSgspJ2msm6clMCkdW9": "Jessica",
    "IKne3meq5aSn9XLyUdCD": "Charlie", "FGY2WhTYpPnrIDTdsKH5": "Laura",
    "SOYHLrjzK2X1ezoPC6cr": "Harry", "pqHfZKP75CvOlQylNhV4": "Bill",
}


def main():
    from patmir.pipeline.streaming.registry import SpeakerRegistry, build_pool
    from patmir.pipeline.streaming.speaker_profile import profile_speaker

    cfg = tomllib.load(open(REPO / "patmir.toml", "rb"))
    pool_cfg = next(c for c in cfg["eval"]["tts"]["candidates"] if c["id"] == "elevenlabs_pool")
    registry = SpeakerRegistry(build_pool(pool_cfg["voice_pool"]))

    refs = json.loads((RUN / "diarize/refs.json").read_text())
    # one ref per speaker (neutral if present)
    by_spk = {}
    for v in refs.values():
        by_spk.setdefault(v["speaker_id"], v)

    for spk, v in sorted(by_spk.items()):
        prof = profile_speaker(v["ref_audio"])
        voice = registry.assign_voice(spk, prof)
        print(f"{spk}: profile(gender={prof.gender}, age={prof.age}) "
              f"-> {NAMES.get(voice, voice)}")
    print("\nassignments:", {k: NAMES.get(v, v) for k, v in registry.assignments().items()})


if __name__ == "__main__":
    main()
