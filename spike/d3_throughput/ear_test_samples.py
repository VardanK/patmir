"""D3 spike — render ear-test samples at num_steps 8 / 16 / 32 (THROWAWAY).

The go/no-go for watch-along hinges on whether num_steps=8 Armenian is good
enough. This renders the same sentences at 8/16/32 steps so the author can
A/B listen. Picks sentences from distinct speakers/emotions.

Output: spike/d3_throughput/ear_test/<speaker>_<emotion>_<idx>__steps<N>.wav
Run: .venv/bin/python spike/d3_throughput/ear_test_samples.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "runs/20260528-010843-source"
OUT = REPO / "spike/d3_throughput/ear_test"

CFG = {
    "repo_id": "lightsofapollo/omnivoice-mlx-q8-g64",
    "ref_audio": str(RUN / "diarize/refs/SPEAKER_00/neutral.wav"),
    "ref_text": "sure?",
    "num_steps": 32,
    "guidance_scale": 2.0,
    "language": "None",
    "lang_code": "None",
}


def main():
    from patmir.eval.candidates.tts.omnivoice_mlx import build

    OUT.mkdir(parents=True, exist_ok=True)
    refs_raw = json.loads((RUN / "diarize/refs.json").read_text())
    refs = {(v["speaker_id"], v["emotion"]): (v["ref_audio"], v["ref_text"]) for v in refs_raw.values()}
    sents = json.loads((RUN / "translated.json").read_text())["sentences"]

    # pick a few longer, distinct sentences (longer = more audible quality diff)
    picks = []
    seen_spk = set()
    for s in sents:
        if not s["target_text"].strip() or len(s["target_text"]) < 18:
            continue
        key = s["speaker_id"]
        if key in seen_spk and len(picks) >= 3:
            continue
        seen_spk.add(key)
        picks.append(s)
        if len(picks) >= 5:
            break

    model = build(CFG)
    for s in picks:
        ref = refs.get((s["speaker_id"], s.get("emotion") or "neutral")) \
            or refs.get((s["speaker_id"], "neutral")) \
            or refs.get(("SPEAKER_00", "neutral"))
        for steps in (8, 16, 32):
            model.num_steps = steps
            wb = model.synthesize(s["target_text"], ref_audio=ref[0], ref_text=ref[1])
            name = f"{s['speaker_id']}_{s.get('emotion','neutral')}_{s['idx']:04d}__steps{steps:02d}.wav"
            (OUT / name).write_bytes(wb)
            print(f"wrote {name}  | {s['target_text'][:50]}", flush=True)

    print(f"\n{len(picks)} sentences x 3 step-counts -> {OUT}")
    print("Listen to steps08 vs steps32 per sentence and judge if 8-step is watchable.")


if __name__ == "__main__":
    main()
