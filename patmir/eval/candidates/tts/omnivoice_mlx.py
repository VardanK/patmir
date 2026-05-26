"""OmniVoice TTS via mlx-audio (native MLX, Apple Silicon).

Mirrors `omnivoice.py` (PyTorch loader) but loads a MLX-quantized OmniVoice
weights set through `mlx-audio`'s `load_model`. The q8 variant
`lightsofapollo/omnivoice-mlx-q8-g64` is ~1 GB total (vs ~3 GB for the
PyTorch fp16) and ~25% faster RTF.

Voice cloning is the same shape: a reference clip + its transcription
seeds the speaker; each call synthesizes the target text in that voice.

Install once: `uv pip install mlx-audio`.
"""
from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Any

import numpy as np


class OmniVoiceMLXTTS:
    id = "omnivoice_mlx"

    def __init__(
        self,
        repo_id: str,
        ref_audio: Path,
        ref_text: str,
        num_steps: int = 32,
        guidance_scale: float = 2.0,
        language: str = "None",
        lang_code: str = "None",
    ) -> None:
        from huggingface_hub import snapshot_download  # lazy
        from mlx_audio.tts.utils import load_model  # lazy

        self.model_path = Path(snapshot_download(repo_id))
        self.model = load_model(self.model_path)
        self.ref_audio = str(ref_audio)
        self.ref_text = ref_text
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.language = language
        self.lang_code = lang_code

    def synthesize(
        self,
        text: str,
        *,
        ref_audio: Path | str | None = None,
        ref_text: str | None = None,
        guidance_scale: float | None = None,
    ) -> bytes:
        import mlx.core as mx  # lazy

        # Per-sentence overrides — fall back to constructor defaults when not
        # supplied. OmniVoice's `model.generate` already accepts these per
        # call, so per-emotion ref clips + per-emotion guidance is just a
        # different kwarg per call.
        eff_ref_audio = str(ref_audio) if ref_audio is not None else self.ref_audio
        eff_ref_text = ref_text if ref_text is not None else self.ref_text
        eff_guidance = guidance_scale if guidance_scale is not None else self.guidance_scale

        # `Model.generate` is a generator yielding GenerationResult chunks.
        chunks = list(
            self.model.generate(
                text=text,
                ref_audio=eff_ref_audio,
                ref_text=eff_ref_text,
                num_steps=self.num_steps,
                guidance_scale=eff_guidance,
                language=self.language,
                lang_code=self.lang_code,
            )
        )
        if not chunks:
            raise RuntimeError("omnivoice_mlx: model.generate yielded no chunks")
        sr = int(chunks[0].sample_rate)
        audio = mx.concatenate([c.audio.reshape(-1) for c in chunks])
        audio_np = np.asarray(audio, dtype=np.float32).squeeze()
        pcm = np.clip(audio_np * 32767.0, -32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()


def build(config: dict[str, Any]) -> OmniVoiceMLXTTS:
    return OmniVoiceMLXTTS(
        repo_id=config["repo_id"],
        ref_audio=Path(config["ref_audio"]),
        ref_text=config["ref_text"],
        num_steps=int(config.get("num_steps", 32)),
        guidance_scale=float(config.get("guidance_scale", 2.0)),
        language=str(config.get("language", "None")),
        lang_code=str(config.get("lang_code", "None")),
    )
