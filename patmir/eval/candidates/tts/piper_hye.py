"""Piper TTS — community Eastern-Armenian voice.

Uses `davit312/piper-TTS-Armenian` from HuggingFace (`hy_AM-gor-medium`).
This is currently the only known local Eastern-Armenian TTS; see
`agent-os/product/tech-stack.md` Stage 4.

Files are downloaded once via huggingface_hub and cached.
"""
from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Any

HF_REPO = "davit312/piper-TTS-Armenian"
DEFAULT_ONNX = "v3/hy_AM-gor-medium.onnx"
DEFAULT_CONFIG = "v3/hy_AM-gor-medium.onnx.json"


class PiperHye:
    id = "piper_hye"

    def __init__(self, onnx_path: Path, config_path: Path) -> None:
        from piper import PiperVoice  # lazy import

        self.voice = PiperVoice.load(str(onnx_path), config_path=str(config_path))

    def synthesize(self, text: str) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self.voice.synthesize_wav(text, wf)
        return buf.getvalue()


def build(config: dict[str, Any]) -> PiperHye:
    from huggingface_hub import hf_hub_download  # lazy import

    repo = config.get("repo_id", HF_REPO)
    onnx_file = config.get("onnx_file", DEFAULT_ONNX)
    config_file = config.get("config_file", DEFAULT_CONFIG)

    onnx_path = Path(hf_hub_download(repo_id=repo, filename=onnx_file))
    config_path = Path(hf_hub_download(repo_id=repo, filename=config_file))
    return PiperHye(onnx_path=onnx_path, config_path=config_path)
