"""Ingest stage — ffmpeg audio extraction.

Extracts a 16 kHz mono PCM WAV from the source video, suitable for Whisper.
ffprobe reads the source duration for progress reporting downstream.

Requires `ffmpeg` and `ffprobe` on PATH (`brew install ffmpeg`).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import AudioTrack


class FFmpegIngest:
    id = "ffmpeg"

    def extract(self, video_path: Path, ctx: JobContext) -> AudioTrack:
        audio_path = ctx.job_dir / "audio.wav"
        duration = _probe_duration(video_path)
        ctx.progress("ingest", 0.1, f"source duration={duration:.1f}s")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(audio_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        track = AudioTrack(path=audio_path, sample_rate=16000, duration=duration)
        _write_meta(ctx.job_dir, track)
        return track


def _probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(out.stdout.strip())


def _write_meta(job_dir: Path, track: AudioTrack) -> None:
    meta = {
        "path": str(track.path),
        "sample_rate": track.sample_rate,
        "duration": track.duration,
    }
    (job_dir / "audio.meta.json").write_text(json.dumps(meta, indent=2))


def build(config: dict[str, Any]) -> FFmpegIngest:
    return FFmpegIngest()
