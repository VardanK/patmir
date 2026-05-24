"""Coqui XTTS-v2 — multilingual TTS with voice cloning (deferred, GPU pass).

Disabled by default in patmir.toml. XTTS-v2 lists Armenian among its
languages but quality is unverified; this candidate is for the GPU pass.

Requires the `TTS` package: `pip install TTS`. Voice cloning needs a short
reference audio clip (configurable below).
"""
from __future__ import annotations

from typing import Any
import io
import wave


class XTTSv2:
    id = "xtts_v2"

    def __init__(self, device: str = "cuda", speaker_wav: str | None = None) -> None:
        from TTS.api import TTS  # lazy import — heavy dep tree.

        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        self.speaker_wav = speaker_wav  # optional reference audio for voice cloning
        self.sample_rate = 24000

    def synthesize(self, text: str) -> bytes:
        kwargs: dict[str, Any] = {"text": text, "language": "hy"}
        if self.speaker_wav:
            kwargs["speaker_wav"] = self.speaker_wav
        wav = self.tts.tts(**kwargs)  # numpy float array

        import numpy as np

        samples = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
        int16 = (samples * 32767.0).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(int16.tobytes())
        return buf.getvalue()


def build(config: dict[str, Any]) -> XTTSv2:
    return XTTSv2(device=config.get("device", "cuda"), speaker_wav=config.get("speaker_wav"))
