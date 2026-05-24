"""Helsinki-NLP OPUS-MT translator candidate.

Pair-specific MarianMT models (e.g. `Helsinki-NLP/opus-mt-en-hy`,
`opus-mt-ru-hy`) — small (~80 MB), Eastern Armenian output (hye), MPS-friendly.

One model = one source language. The `source_lang` config field declares
which source pair this entry is for; `translate(text, source_lang)` rejects
mismatches loudly so a misconfigured pair can't silently swap engines.
"""
from __future__ import annotations

from typing import Any


class OpusMTTranslator:
    id = "opus_mt"

    def __init__(self, model_id: str, source_lang: str, device: str = "cpu") -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # lazy import

        self.device = device
        self.source_lang = source_lang
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)

    def translate(self, text: str, source_lang: str) -> str:
        if source_lang != self.source_lang:
            raise ValueError(
                f"OpusMT model is configured for {self.source_lang!r}, got {source_lang!r}. "
                f"Register a separate opus-mt entry per source language."
            )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=256, num_beams=4)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]


def build(config: dict[str, Any]) -> OpusMTTranslator:
    return OpusMTTranslator(
        model_id=config["model_id"],
        source_lang=config["source_lang"],
        device=config.get("device", "cpu"),
    )
