"""Verify ElevenLabs no-clone preset-pool mode (#6).

Checks:
  - clone=False does NO IVC upload (no voices/add call) and assigns a stable
    pool voice per speaker, distinct across speakers.
  - same speaker_id always returns the same voice (consistency).
  - synthesis returns real Armenian WAV bytes.

Renders one sentence per speaker into spike/d3_throughput/pool_test/ for ear check.
Run: .venv/bin/python spike/d3_throughput/test_pool_mode.py
"""
from __future__ import annotations

import io
import json
import os
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "runs/20260528-010843-source"
OUT = REPO / "spike/d3_throughput/pool_test"


def load_env():
    for line in (REPO / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main():
    load_env()
    from patmir.eval.candidates.tts.elevenlabs import build, _DEFAULT_VOICE_POOL

    OUT.mkdir(parents=True, exist_ok=True)
    model = build({"model_id": "eleven_v3", "language_code": "hy", "clone": False})

    # Fail loudly if any IVC upload is attempted in pool mode.
    def _no_upload(*a, **k):
        raise AssertionError("pool mode must NOT call IVC upload (_voice_id_for_ref)")
    model._voice_id_for_ref = _no_upload  # type: ignore

    # 1) consistency + distinctness on the speaker→voice mapping (no network).
    v00a = model._pool_voice_for_speaker("SPEAKER_00")
    v00b = model._pool_voice_for_speaker("SPEAKER_00")
    v01 = model._pool_voice_for_speaker("SPEAKER_01")
    v03 = model._pool_voice_for_speaker("SPEAKER_03")
    assert v00a == v00b, "same speaker must map to the same voice"
    assert len({v00a, v01, v03}) == 3, "distinct speakers must get distinct voices"
    assert v00a == _DEFAULT_VOICE_POOL[0], "first speaker takes first pool voice"
    print(f"mapping OK: SPEAKER_00={v00a[:8]} SPEAKER_01={v01[:8]} SPEAKER_03={v03[:8]}")

    # 2) real synthesis, one line per speaker (passes speaker_id like the adapter does).
    sents = json.loads((RUN / "translated.json").read_text())["sentences"]
    seen = set()
    for s in sents:
        spk = s["speaker_id"]
        if spk in seen or not s["target_text"].strip():
            continue
        seen.add(spk)
        wb = model.synthesize(s["target_text"], speaker_id=spk, emotion=s.get("emotion"))
        with wave.open(io.BytesIO(wb), "rb") as wf:
            dur = wf.getnframes() / wf.getframerate()
        assert dur > 0.2, f"suspiciously short/empty WAV for {spk}: {dur}s"
        name = f"{spk}__{model._speaker_voice[spk][:8]}.wav"
        (OUT / name).write_bytes(wb)
        print(f"  {spk} -> voice {model._speaker_voice[spk][:8]}  {dur:.2f}s  | {s['target_text'][:40]}")

    summary = model.summarize()
    assert all(v["status"] == "pool" for v in summary["speakers"].values()), summary
    print(f"\nsummary statuses all 'pool' OK. WAVs -> {OUT}")
    print("PASS")


if __name__ == "__main__":
    main()
