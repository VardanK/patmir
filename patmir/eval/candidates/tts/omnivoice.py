"""OmniVoice TTS — zero-shot multilingual voice cloning.

`k2-fsa/OmniVoice` — Apache-2.0 diffusion language model TTS that lists 646
languages, Armenian (`hy` / `hye`) included with ~42 hours of training data.
Dialect is the ISO macrolanguage code, so Eastern vs Western has to be
ear-judged from output.

Architecture-wise, OmniVoice is voice-clone-only: every synthesis needs a
reference audio clip + its transcription. The pipeline keeps a single
"default" reference clip per Patmir voice (configured by `ref_audio` /
`ref_text` in the candidate entry); each call clones that voice for the
given target text.

For the v1 reference we use one of the Phase 0 Piper outputs
(`hy_AM-gor-medium` saying "Արկածի գնանք") — a known-Eastern Armenian clip,
locally available, no extra download. Once the user records a native
Eastern-Armenian voice memo, swap the `ref_audio`/`ref_text` in patmir.toml.
"""
from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Any

import numpy as np


class OmniVoiceTTS:
    id = "omnivoice"

    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: str,
        ref_audio: Path,
        ref_text: str,
        num_step: int = 32,
        speed: float = 1.0,
    ) -> None:
        import torch  # lazy
        from omnivoice import OmniVoice  # lazy import

        self.device = device
        self.ref_audio = str(ref_audio)
        self.ref_text = ref_text
        self.num_step = num_step
        self.speed = speed
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype, torch.float32)
        self.model = OmniVoice.from_pretrained(
            model_id,
            device_map=device,
            dtype=torch_dtype,
        )

    def synthesize(self, text: str) -> bytes:
        out = self.model.generate(
            text=text,
            ref_audio=self.ref_audio,
            ref_text=self.ref_text,
            num_step=self.num_step,
            speed=self.speed,
        )
        # OmniVoice returns a list of float32 numpy arrays at 24 kHz mono.
        audio = out[0] if isinstance(out, (list, tuple)) else out
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        # Convert float32 [-1, 1] to int16 PCM WAV bytes.
        pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()


def _resolve_device(requested: str) -> str:
    if requested in ("cpu", "cuda", "mps"):
        return requested
    try:
        import torch  # type: ignore[import-not-found]
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def build(config: dict[str, Any]) -> OmniVoiceTTS:
    ref_audio = Path(config["ref_audio"])
    if not ref_audio.is_absolute():
        ref_audio = (Path.cwd() / ref_audio).resolve()
    return OmniVoiceTTS(
        model_id=config.get("model_id", "k2-fsa/OmniVoice"),
        device=_resolve_device(config.get("device", "auto")),
        dtype=config.get("dtype", "float32"),
        ref_audio=ref_audio,
        ref_text=config["ref_text"],
        num_step=int(config.get("num_step", 32)),
        speed=float(config.get("speed", 1.0)),
    )
