"""JobContext — per-run state passed to every stage.

Holds the working directory, source path, detected source language (set by
ASR), pipeline options, and a `progress(...)` callback that appends to
`<job_dir>/progress.jsonl` for the CLI tail + FastAPI SSE stream.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_DEFAULT_EMOTION_GUIDANCE: dict[str, float] = {
    # Anchored at OmniVoice's constructor default (2.0); pushes up for
    # intense emotions to amplify ref-driven expression. See ADR
    # 2026-05-26-emotion-routed-through-ref-clip-binning.
    "neutral": 1.8,
    "sad":     2.0,
    "happy":   2.8,
    "angry":   3.0,
    "fearful": 2.6,
}


@dataclass
class PipelineOptions:
    stretch_cap: float = 0.20
    asr_int8: bool = True
    keep_intermediate: bool = True
    # Linear gain applied to the accompaniment bed when mixing the AR voice
    # on top. 1.0 = unchanged source mix volume; 0.6 = -4 dB (a touch quieter
    # so the dialogue sits above the music). The voice itself is normalized
    # to roughly match the original dialogue loudness.
    music_bed_gain: float = 0.6
    # Global trim (dB) on top of the mux per-sentence RMS loudness match.
    # Negative = quieter. -3.0 came out of the 2026-05-27 masha-2-5min ear-test.
    voice_gain_db: float = -3.0
    # Minimum duration (seconds) a turn must be — after 100 ms inward trim
    # on each side — to qualify as a reference clip for OmniVoice cloning.
    # Used by ref_picker, which runs in the runner post-emotion stage.
    min_ref_seconds: float = 1.5
    # Per-sentence afade duration (seconds) applied to the synthesized track
    # at sentence boundaries in mux. Suppresses the original-voice bleed at
    # boundaries reported as quality issue #4.
    sentence_afade_seconds: float = 0.05
    # No-overlap voice placement in mux. When True, a synthesized line never
    # starts before the previous line's audio has finished — the mux "ripples"
    # a line later if its translated audio (often longer than the source slot,
    # e.g. ElevenLabs lines over fast cartoon dialogue) would otherwise pile on
    # top of the previous one. Drift self-resets at every speech gap (a pause
    # in the source puts the next line's start past the cursor), so it stays
    # local to dense bursts. Set False for the old place-at-source-time-with-
    # overrun behavior. Gap below is the minimum silence inserted between lines.
    voice_no_overlap: bool = True
    voice_overlap_gap_seconds: float = 0.06
    # OmniVoice guidance_scale per emotion bin. The TTS adapter looks this
    # up per sentence and passes it to the inner candidate as a per-call
    # override of the constructor default.
    emotion_guidance: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_EMOTION_GUIDANCE)
    )


@dataclass
class JobContext:
    job_dir: Path
    source_path: Path
    options: PipelineOptions
    source_lang: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def progress(self, stage: str, pct: float, msg: str = "") -> None:
        record = {
            "ts": time.time(),
            "stage": stage,
            "pct": round(pct, 4),
            "msg": msg,
        }
        path = self.job_dir / "progress.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
