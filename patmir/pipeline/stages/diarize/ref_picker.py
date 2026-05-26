"""Per-speaker reference clip picker.

Given the pyannote turn list, the overlap intervals, and the VAD-masked
`speech_vocals` stem, pick one reference window per speaker — the longest
turn that

  (a) does not intersect any overlap region,
  (b) is at least `min_ref_seconds` long after trimming 100 ms inward each side.

The slice is resampled to 16 kHz mono and written as `<refs_dir>/<spk>.wav`.

`ref_text` is filled in later by `runner._backfill_ref_text` — we don't have
ASR words yet at diarize time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from patmir.pipeline.stage import OverlapInterval, SpeakerRef, SpeakerTurn

REF_SR = 16_000
EDGE_TRIM_SEC = 0.100


def _intersects_any(start: float, end: float, overlaps: list[OverlapInterval]) -> bool:
    for ov in overlaps:
        if ov.start < end and ov.end > start:
            return True
    return False


def _resample_linear(mono: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return mono.astype("float32", copy=False)
    n_target = int(len(mono) * dst_sr / src_sr)
    if n_target <= 0:
        return np.zeros(0, dtype="float32")
    x_old = np.linspace(0, 1, len(mono), endpoint=False)
    x_new = np.linspace(0, 1, n_target, endpoint=False)
    return np.interp(x_new, x_old, mono).astype("float32")


def pick_refs(
    *,
    turns: list[SpeakerTurn],
    overlaps: list[OverlapInterval],
    speech_vocals: Path,
    refs_dir: Path,
    min_ref_seconds: float,
) -> dict[str, SpeakerRef]:
    """Pick + slice one reference clip per speaker.

    Speakers with no qualifying turn are omitted from the result; the TTS
    adapter falls back to the global ref for them.
    """
    refs_dir.mkdir(parents=True, exist_ok=True)

    audio, src_sr = sf.read(str(speech_vocals), dtype="float32", always_2d=True)
    # Downmix to mono — cloning refs are mono and pyannote+OmniVoice both prefer it.
    mono_full = audio.mean(axis=1)

    by_speaker: dict[str, list[SpeakerTurn]] = {}
    for t in turns:
        by_speaker.setdefault(t.speaker_id, []).append(t)

    out: dict[str, SpeakerRef] = {}
    for spk, spk_turns in by_speaker.items():
        # Trim each turn by EDGE_TRIM_SEC, drop ones intersecting overlaps,
        # drop ones too short after trimming.
        eligible: list[tuple[float, float]] = []
        for t in spk_turns:
            s = t.start + EDGE_TRIM_SEC
            e = t.end - EDGE_TRIM_SEC
            if e - s < min_ref_seconds:
                continue
            if _intersects_any(s, e, overlaps):
                continue
            eligible.append((s, e))
        if not eligible:
            continue
        # Pick the longest.
        s, e = max(eligible, key=lambda se: se[1] - se[0])
        i0 = max(0, int(s * src_sr))
        i1 = min(len(mono_full), int(e * src_sr))
        if i1 <= i0:
            continue
        slice_mono = mono_full[i0:i1]
        slice_16k = _resample_linear(slice_mono, src_sr, REF_SR)
        ref_path = refs_dir / f"{spk}.wav"
        sf.write(str(ref_path), slice_16k, REF_SR, subtype="PCM_16")
        out[spk] = SpeakerRef(
            speaker_id=spk,
            ref_audio=ref_path,
            ref_text="",  # backfilled after ASR
            source_start=s,
            source_end=e,
        )
    return out
