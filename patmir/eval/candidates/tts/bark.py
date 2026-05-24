"""Bark — multilingual generative TTS (deferred, GPU pass).

Disabled by default. Armenian quality is unverified; roadmap notes
"prosody can be uneven." Use only as part of the GPU-pass sweep.
"""
from __future__ import annotations

from typing import Any
import io
import wave


class BarkTTS:
    id = "bark"

    def __init__(self, device: str = "cuda") -> None:
        from transformers import AutoProcessor, BarkModel  # lazy import

        self.processor = AutoProcessor.from_pretrained("suno/bark")
        self.model = BarkModel.from_pretrained("suno/bark").to(device)
        self.device = device
        self.sample_rate = self.model.generation_config.sample_rate

    def synthesize(self, text: str) -> bytes:
        inputs = self.processor(text, return_tensors="pt").to(self.device)
        audio = self.model.generate(**inputs)
        samples = audio.squeeze(0).cpu().numpy()

        import numpy as np

        samples = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
        int16 = (samples * 32767.0).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(int16.tobytes())
        return buf.getvalue()


def build(config: dict[str, Any]) -> BarkTTS:
    return BarkTTS(device=config.get("device", "cuda"))
