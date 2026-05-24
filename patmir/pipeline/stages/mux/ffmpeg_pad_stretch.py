"""Mux stage — per-sentence pad + bounded time-stretch.

For each synthesized sentence we:
  1. Compute the source window from ASR timing.
  2. Compute stretch ratio = wav_duration / window, clamped to ±stretch_cap.
  3. Apply pyrubberband.time_stretch (pitch-preserving) if outside [0.98, 1.02].
  4. Pad the result with trailing silence if still shorter than the window.
  5. If still longer, allow overrun — the next sentence's window starts when
     the prior one ends (drift accepted; lip sync is not a goal).

Output is a single concatenated WAV at the synthesized audio's sample rate.

System requirement: `rubberband` CLI must be installed (`brew install rubberband`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import AudioTrack, MuxedAudio, SynthesizedAudio


_NO_STRETCH_BAND = (0.98, 1.02)


class FFmpegPadStretch:
    id = "ffmpeg_pad_stretch"

    def mux(
        self,
        synth: SynthesizedAudio,
        source: AudioTrack,
        ctx: JobContext,
    ) -> MuxedAudio:
        import soundfile as sf  # lazy import

        cap = ctx.options.stretch_cap
        sr = synth.sample_rate or 22050
        total = len(synth.sentences) or 1

        chunks: list[np.ndarray] = []
        cursor = 0.0  # current position in seconds along the output track

        for i, s in enumerate(synth.sentences):
            audio, file_sr = sf.read(str(s.wav_path), dtype="float32", always_2d=False)
            if file_sr != sr:
                audio = _simple_resample(audio, file_sr, sr)
            window = max(0.0, s.source_end - s.source_start)

            # Lead silence if the source window starts after the cursor.
            if s.source_start > cursor:
                gap = s.source_start - cursor
                chunks.append(np.zeros(int(gap * sr), dtype="float32"))
                cursor += gap

            audio = _stretch_to_window(audio, sr, window, cap)
            duration = len(audio) / sr

            if window > 0 and duration < window:
                pad = int((window - duration) * sr)
                audio = np.concatenate([audio, np.zeros(pad, dtype="float32")])
                duration = len(audio) / sr

            chunks.append(audio)
            cursor += duration

            ctx.progress("mux", (i + 1) / total, f"sentence {i + 1}/{total}")

        if not chunks:
            chunks = [np.zeros(int(source.duration * sr), dtype="float32")]

        out_audio = np.concatenate(chunks)
        # Trim or pad to roughly the source duration.
        target_samples = int(source.duration * sr)
        if len(out_audio) < target_samples:
            out_audio = np.concatenate([
                out_audio,
                np.zeros(target_samples - len(out_audio), dtype="float32"),
            ])
        # Overrun is fine — we don't truncate; we just stop matching the source picture.

        out_path = ctx.job_dir / "dub.wav"
        sf.write(str(out_path), out_audio, sr, subtype="PCM_16")
        _write_meta(ctx.job_dir, out_path, sr)
        return MuxedAudio(path=out_path, sample_rate=sr)


def _stretch_to_window(
    audio: np.ndarray,
    sr: int,
    window: float,
    cap: float,
) -> np.ndarray:
    """Clamp the duration into the window using a pitch-preserving stretch."""
    if window <= 0:
        return audio
    duration = len(audio) / sr
    if duration <= 0:
        return audio
    desired_ratio = duration / window  # >1 = need to compress (speed up)
    clamped = max(1 - cap, min(1 + cap, desired_ratio))
    if _NO_STRETCH_BAND[0] <= clamped <= _NO_STRETCH_BAND[1]:
        return audio
    try:
        import pyrubberband as pyrb  # lazy import
        return pyrb.time_stretch(audio, sr, clamped)
    except Exception:
        # rubberband binary missing or call failed — fall back to no stretch.
        return audio


def _simple_resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Linear-interp resample. Adequate for low-stakes voice tracks."""
    if from_sr == to_sr:
        return audio
    ratio = to_sr / from_sr
    n = int(len(audio) * ratio)
    if n == 0:
        return np.zeros(0, dtype="float32")
    x_old = np.linspace(0, 1, len(audio), endpoint=False)
    x_new = np.linspace(0, 1, n, endpoint=False)
    return np.interp(x_new, x_old, audio).astype("float32")


def _write_meta(job_dir: Path, path: Path, sample_rate: int) -> None:
    meta = {"path": str(path), "sample_rate": sample_rate}
    (job_dir / "dub.meta.json").write_text(json.dumps(meta, indent=2))


def build(config: dict[str, Any]) -> FFmpegPadStretch:
    return FFmpegPadStretch()
