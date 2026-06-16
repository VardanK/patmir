"""Source separation via Demucs on MLX (Apple Silicon native).

Same model + output contract as the PyTorch `htdemucs.py` candidate, but runs
the separation through `demucs-mlx` (`demucs_mlx.Separator`) so it executes on
the Mac GPU + unified memory instead of the PyTorch MPS path. `htdemucs_ft`
(fine-tuned 4-model bag) is the batch default; the base `htdemucs` single model
is the faster watch-along variant.

Everything around the separation engine is shared with the PyTorch candidate:
ffmpeg full-quality extraction, the Silero-VAD speech/non-speech split
(`_split_vocals_with_vad`), and the `SeparatedAudio` artifact writing. Only the
stem-producing call differs.

Install once: `uv pip install demucs-mlx`.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import SeparatedAudio
from patmir.pipeline.stages.separate.htdemucs import _split_vocals_with_vad


class HTDemucsMLX:
    id = "htdemucs_mlx"

    def __init__(self, model_name: str = "htdemucs_ft", shifts: int = 1, overlap: float = 0.25) -> None:
        from demucs_mlx import Separator  # lazy import (heavy)

        self.model_name = model_name
        self.separator = Separator(model=model_name, shifts=shifts, overlap=overlap, split=True)
        self.sample_rate: int = int(self.separator.samplerate)
        self.channels: int = int(self.separator.audio_channels)
        self.device = "mps"  # MLX runs on Apple Metal; recorded for the meta sidecar

    def separate(self, source_video: Path, ctx: JobContext) -> SeparatedAudio:
        import numpy as np  # lazy
        import soundfile as sf  # lazy

        # Demucs needs full-quality stereo at its native sample rate.
        full_audio = ctx.job_dir / "full_audio.wav"
        if not full_audio.exists():
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(source_video),
                "-vn",
                "-ac", str(self.channels),
                "-ar", str(self.sample_rate),
                "-c:a", "pcm_s16le",
                str(full_audio),
            ], check=True, capture_output=True)
        ctx.progress("separate", 0.05, "extracted full-quality audio")

        ctx.progress("separate", 0.10, f"running demucs-mlx ({self.model_name})")
        out = self.separator.separate_audio_file(str(full_audio))
        # separate_audio_file returns (sample_rate_or_info, stems_dict); stems are
        # keyed by name with (channels, samples) float32 arrays.
        stems_raw = out[1] if isinstance(out, tuple) else out
        stems = {name: np.asarray(arr, dtype="float32") for name, arr in stems_raw.items()}
        ctx.progress("separate", 0.85, "stems ready")

        vocals = stems.get("vocals")
        accompaniment = None
        for name, arr in stems.items():
            if name == "vocals":
                continue
            accompaniment = arr if accompaniment is None else accompaniment + arr
        if accompaniment is None:  # degenerate: model emitted only vocals
            accompaniment = np.zeros_like(vocals)

        accomp_path = ctx.job_dir / "accompaniment.wav"
        vocals_path = ctx.job_dir / "vocals.wav"
        # stems are (channels, samples); soundfile wants (samples, channels).
        sf.write(str(accomp_path), accompaniment.T, self.sample_rate, subtype="PCM_16")
        if vocals is not None:
            sf.write(str(vocals_path), vocals.T, self.sample_rate, subtype="PCM_16")

        if not ctx.options.keep_intermediate and full_audio.exists():
            full_audio.unlink()

        speech_path: Path | None = None
        non_speech_path: Path | None = None
        vad_intervals: list[tuple[float, float]] | None = None
        if vocals is not None:
            ctx.progress("separate", 0.9, "running Silero VAD on vocals")
            speech_path, non_speech_path, vad_intervals = _split_vocals_with_vad(
                vocals_path=vocals_path,
                accompaniment_shape=accompaniment.shape,
                sample_rate=self.sample_rate,
                job_dir=ctx.job_dir,
            )

        meta = {
            "accompaniment": str(accomp_path),
            "vocals": str(vocals_path) if vocals is not None else None,
            "speech_vocals": str(speech_path) if speech_path else None,
            "non_speech_vocals": str(non_speech_path) if non_speech_path else None,
            "vad_intervals": vad_intervals,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "model": self.model_name,
            "device": self.device,
            "engine": "demucs-mlx",
        }
        (ctx.job_dir / "separated.meta.json").write_text(json.dumps(meta, indent=2))
        ctx.progress("separate", 1.0, str(accomp_path))
        return SeparatedAudio(
            accompaniment=accomp_path,
            vocals=vocals_path,
            sample_rate=self.sample_rate,
            channels=self.channels,
            speech_vocals=speech_path,
            non_speech_vocals=non_speech_path,
            vad_intervals=vad_intervals,
        )


def build(config: dict[str, Any]) -> HTDemucsMLX:
    return HTDemucsMLX(
        model_name=config.get("model", "htdemucs_ft"),
        shifts=int(config.get("shifts", 1)),
        overlap=float(config.get("overlap", 0.25)),
    )
