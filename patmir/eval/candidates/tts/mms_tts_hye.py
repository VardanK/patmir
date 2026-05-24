"""Meta MMS-TTS Armenian (`facebook/mms-tts-hye`).

Tech-stack calls this "likely the strongest pure-local starting point" for
Armenian TTS. Runs on CPU; slow but usable for ~20-utterance evals.
"""
from __future__ import annotations

import io
import wave
from typing import Any


class MMSTTS:
    id = "mms_tts_hye"

    def __init__(self, model_id: str, device: str = "cpu") -> None:
        from transformers import AutoTokenizer, VitsModel  # lazy import
        import torch

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = VitsModel.from_pretrained(model_id).to(device)
        self.sampling_rate = self.model.config.sampling_rate  # typically 16 kHz for MMS-TTS

    def synthesize(self, text: str) -> bytes:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            waveform = self.model(**inputs).waveform
        # waveform: [1, n_samples] float32 in roughly [-1, 1].
        samples = waveform.squeeze(0).cpu().numpy()
        return _to_wav_bytes(samples, self.sampling_rate)


def _to_wav_bytes(samples, sample_rate: int) -> bytes:
    import numpy as np

    samples = np.asarray(samples, dtype=np.float32)
    # Clip and convert to int16 PCM.
    samples = np.clip(samples, -1.0, 1.0)
    int16 = (samples * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16.tobytes())
    return buf.getvalue()


def build(config: dict[str, Any]) -> MMSTTS:
    return MMSTTS(model_id=config["model_id"], device=config.get("device", "cpu"))
