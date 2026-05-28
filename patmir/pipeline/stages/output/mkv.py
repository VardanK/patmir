"""Output stage — MKV remux with Armenian as default audio.

Builds the final container:
  - Video stream (when the source has one): copied from source (no re-encode).
  - Audio track 0: Armenian (AAC-encoded from `dub.wav`), language=hye, default.
  - Audio track 1: original audio from the source, language=<detected>, copied.
  - Subtitle track 0: Armenian SRT generated from the translated transcript.

When the source has no video stream (e.g. the local server pulls
audio-only YouTube downloads), the video map + codec flags are omitted and
the output is an audio + subtitle MKV.

Designed so MP4 output is a sibling module later — only the ffmpeg flags
change (and MP4 sub-track handling differs).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import (
    MuxedAudio,
    OutputArtifacts,
    TranslatedSentence,
    TranslatedTranscript,
)


class MKVOutput:
    id = "mkv_default"

    def write(
        self,
        muxed: MuxedAudio,
        source_video: Path,
        translated: TranslatedTranscript,
        ctx: JobContext,
    ) -> OutputArtifacts:
        out_video = ctx.job_dir / "out.mkv"
        srt_path = ctx.job_dir / "out.hy.srt"
        _write_srt(srt_path, translated.sentences)
        ctx.progress("output", 0.2, "wrote SRT")

        has_video = _has_video_stream(source_video)
        src_lang_tag = translated.source_lang or "und"
        cmd: list[str] = [
            "ffmpeg", "-y",
            "-i", str(source_video),
            "-i", str(muxed.path),
            "-i", str(srt_path),
        ]
        if has_video:
            cmd += ["-map", "0:v:0"]
        cmd += [
            "-map", "1:a:0",
            "-map", "0:a:0?",
            "-map", "2:s:0",
        ]
        if has_video:
            cmd += ["-c:v", "copy"]
        cmd += [
            "-c:a:0", "aac",
            "-b:a:0", "192k",
            "-c:a:1", "copy",
            "-c:s", "srt",
            "-metadata:s:a:0", "language=hye",
            "-metadata:s:a:0", "title=Armenian (Patmir)",
            "-metadata:s:a:1", f"language={src_lang_tag}",
            "-metadata:s:a:1", "title=Original",
            "-metadata:s:s:0", "language=hye",
            "-disposition:a:0", "default",
            "-disposition:a:1", "0",
            "-disposition:s:0", "default",
            str(out_video),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        ctx.progress("output", 1.0, str(out_video))
        return OutputArtifacts(video=out_video, subtitles=srt_path)


def _has_video_stream(path: Path) -> bool:
    """True when ffprobe reports at least one video stream on `path`.

    Audio-only sources (yt-dlp's `bestaudio/best`) return `False`, in which
    case the output stage emits an audio + subtitle MKV.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v",
                "-show_entries", "stream=codec_type",
                "-of", "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        data = json.loads(result.stdout or "{}")
        return bool(data.get("streams"))
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        return False


def _write_srt(path: Path, sentences: list[TranslatedSentence]) -> None:
    lines: list[str] = []
    for i, s in enumerate(sentences, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_time(s.start)} --> {_srt_time(s.end)}")
        lines.append(s.target_text)
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _srt_time(t: float) -> str:
    if t < 0:
        t = 0.0
    h, rem = divmod(int(t), 3600)
    m, sec = divmod(rem, 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def build(config: dict[str, Any]) -> MKVOutput:
    return MKVOutput()
