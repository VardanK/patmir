"""MADLAD-400 translator (deferred — GPU pass only).

Disabled by default in patmir.toml; flip `enabled = true` when running on
the workstation GPU. Uses a T5-style prompt prefix to select the target.
"""
from __future__ import annotations

from typing import Any

MADLAD_LANG = {"en": "en", "ru": "ru", "hy": "hy"}


class MADLADTranslator:
    id = "madlad"

    def __init__(self, model_id: str, device: str = "cuda") -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # lazy import

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)

    def translate(self, text: str, source_lang: str) -> str:
        # MADLAD prepends a target-language tag.
        prompt = f"<2{MADLAD_LANG['hy']}> {text}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=256, num_beams=4)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]


def build(config: dict[str, Any]) -> MADLADTranslator:
    return MADLADTranslator(model_id=config["model_id"], device=config.get("device", "cuda"))
