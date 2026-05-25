"""Mux stage — layer AR voice over the accompaniment music bed.

For each synthesized sentence:
  1. Compute the source window from ASR timing.
  2. Stretch ratio = wav_duration / window, clamped to ±stretch_cap.
  3. Apply pyrubberband.time_stretch (pitch-preserving) if outside [0.98, 1.02].
  4. Position the AR voice at the sentence's source start.
  5. Allow overrun if it doesn't fit — drift accepted; lip sync is not a goal.

Then mix:
  out = accompaniment * music_bed_gain + voice * voice_gain
with a peak normalize to prevent clipping. Voice tracks live at the same
sample rate and channel count as the music bed (stereo, typically 44.1 kHz)
so the bed isn't downgraded by the voice path.

System requirement: `rubberband` CLI installed (`brew install rubberband`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import (
    AudioTrack,
    MuxedAudio,
    SeparatedAudio,
    SynthesizedAudio,
)


_NO_STRETCH_BAND = (0.98, 1.02)


class FFmpegPadStretch:
    id = "ffmpeg_pad_stretch"

    def mux(
        self,
        synth: SynthesizedAudio,
        source: AudioTrack,
        separated: SeparatedAudio,
        ctx: JobContext,
    ) -> MuxedAudio:
        import soundfile as sf

        cap = ctx.options.stretch_cap
        voice_lin = 10.0 ** (ctx.options.voice_gain_db / 20.0)
        music_lin = ctx.options.music_bed_gain

        # Read the music bed. If the separate stage produced a non-speech-vocal
        # stem (laughs / screams / grunts), mix it into the bed so those
        # atmospheric sounds survive into the final dub.
        bed, sr = sf.read(str(separated.accompaniment), dtype="float32", always_2d=True)
        if separated.non_speech_vocals is not None and separated.non_speech_vocals.exists():
            nsv, nsv_sr = sf.read(str(separated.non_speech_vocals), dtype="float32", always_2d=True)
            if nsv_sr == sr and nsv.shape == bed.shape:
                bed = bed + nsv
        n_samples = bed.shape[0]
        n_channels = bed.shape[1]

        voice_track = np.zeros_like(bed)
        total = len(synth.sentences) or 1

        for i, s in enumerate(synth.sentences):
            audio, file_sr = sf.read(str(s.wav_path), dtype="float32", always_2d=False)
            if file_sr != sr:
                audio = _simple_resample(audio, file_sr, sr)
            window = max(0.0, s.source_end - s.source_start)
            audio = _stretch_to_window(audio, sr, window, cap)

            start_sample = int(s.source_start * sr)
            end_sample = start_sample + len(audio)
            if start_sample >= n_samples:
                ctx.progress("mux", (i + 1) / total, f"sentence {i+1}/{total} (past end)")
                continue
            if end_sample > n_samples:
                audio = audio[: n_samples - start_sample]
                end_sample = n_samples

            mono = audio  # voice is mono
            # Spread mono voice across all bed channels.
            voice_track[start_sample:end_sample, :] += mono[:, None] * voice_lin

            ctx.progress("mux", (i + 1) / total, f"sentence {i + 1}/{total}")

        out_audio = bed * music_lin + voice_track

        # Peak normalize to -1 dBFS to prevent clipping after the sum.
        peak = float(np.max(np.abs(out_audio)))
        if peak > 0:
            target = 10.0 ** (-1.0 / 20.0)
            if peak > target:
                out_audio = out_audio * (target / peak)

        out_path = ctx.job_dir / "dub.wav"
        sf.write(str(out_path), out_audio, sr, subtype="PCM_16")
        _write_meta(ctx.job_dir, out_path, sr, n_channels)
        return MuxedAudio(path=out_path, sample_rate=sr)


def _stretch_to_window(
    audio: np.ndarray,
    sr: int,
    window: float,
    cap: float,
) -> np.ndarray:
    if window <= 0:
        return audio
    duration = len(audio) / sr
    if duration <= 0:
        return audio
    desired_ratio = duration / window
    clamped = max(1 - cap, min(1 + cap, desired_ratio))
    if _NO_STRETCH_BAND[0] <= clamped <= _NO_STRETCH_BAND[1]:
        return audio
    try:
        import pyrubberband as pyrb  # lazy import
        return pyrb.time_stretch(audio, sr, clamped)
    except Exception:
        return audio


def _simple_resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return audio
    ratio = to_sr / from_sr
    n = int(len(audio) * ratio)
    if n == 0:
        return np.zeros(0, dtype="float32")
    x_old = np.linspace(0, 1, len(audio), endpoint=False)
    x_new = np.linspace(0, 1, n, endpoint=False)
    return np.interp(x_new, x_old, audio).astype("float32")


def _write_meta(job_dir: Path, path: Path, sample_rate: int, channels: int) -> None:
    meta = {"path": str(path), "sample_rate": sample_rate, "channels": channels}
    (job_dir / "dub.meta.json").write_text(json.dumps(meta, indent=2))


def build(config: dict[str, Any]) -> FFmpegPadStretch:
    return FFmpegPadStretch()
