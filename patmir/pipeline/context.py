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
    # Linear gain on the ORIGINAL VOCALS stem where it survives the duck (i.e.
    # in gaps between ASR sentences — un-transcribed dialogue, laughs, breaths).
    # Inside ASR windows the duck zeroes it so the AR dub plays alone; this floor
    # only governs the gaps. The old behavior folded vocals into the bed at
    # music_bed_gain (0.6), so any dialogue ASR missed leaked at full music
    # level (quality issue #9 / #10). 0.12 keeps non-verbal texture faint;
    # set 0.0 to fully remove the original voice.
    vocals_bed_gain: float = 0.12
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
    # Pin ONE reference clip per speaker (the longest/cleanest, used for every
    # line regardless of emotion) instead of routing emotion through a
    # per-(speaker, emotion) ref clip. Per-emotion refs made a character's
    # cloned voice audibly swap whenever the detected emotion changed — badly so
    # on chaotic cartoon audio where the per-emotion clips are short, noisy
    # snippets (screams/laughs). Pinning one ref keeps each character's voice
    # consistent. Trades away per-line emotion routing (emotion still varies via
    # guidance_scale). Set False to restore the Phase-2.x emotion-ref binning.
    tts_pin_speaker_ref: bool = True
    # Duck the original vocals stem this many seconds BEFORE each line's source
    # start (and after its end via duck_tail_seconds). ASR word-onset timestamps
    # land ~0.24s LATE vs the true speech onset, so without a lead the original
    # speaker is audible at the start of every line before the duck engages
    # (quality issue #9). The pre-roll silences the bed-vocals before the onset.
    duck_lead_seconds: float = 0.25
    duck_tail_seconds: float = 0.10
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
    # Max pitch-preserving speed-up applied to a line to help it fit before the
    # next line, so dense dialogue (translated TTS often ~1.4× longer than the
    # source) is absorbed by a mix of mild compression AND shifting rather than
    # pure shifting (which packs lines back-to-back and truncates the tail).
    # 1.0 disables extra compression (pure ripple). 1.5 ≈ up to 50% faster.
    voice_max_compress: float = 1.5
    # Curated fallback ref-clip pool for clone TTS ([[tts_ref_pool]] tables in
    # patmir.toml → {path, gender?, text?}). A speaker with no usable ref used
    # to synthesize with NO reference — an anonymous default voice that also
    # varies run to run. With a pool configured, such speakers get a stable
    # stand-in clip from here (gender-matched when the speaker's profile is
    # known); with no pool configured the adapter falls back to reusing
    # another character's pinned ref from the same job before ever going
    # ref-less. Quality issue #12.
    tts_ref_pool: list[dict] = field(default_factory=list)
    # Minimum mean level (dBFS RMS) for a sliced turn to qualify as a ref clip.
    # Quiet slices are usually far-field / bleed-heavy and clone badly. Windows
    # below the floor lose to any louder eligible window; if ALL of a cell's
    # windows fail the floor, the longest still wins (never drop a speaker's
    # only ref). Quality issue #12.
    ref_min_rms_db: float = -33.0
    # Watch-along only: seconds of the opening analyzed in the prelude to
    # establish the cast before playback starts. Smaller = faster time-to-first
    # -audio, but fewer characters are seen up front (later ones register on the
    # fly mid-stream). 40s balances first-audio latency against cast coverage.
    prelude_window_seconds: float = 40.0
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
