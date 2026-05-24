"""NLLB-200 translator candidate.

Used for both `nllb_200_distilled_600m` (CPU/Mac default) and `nllb_200_3.3b`
(GPU pass) — the build() factory reads `model_id` and `device` from config.
"""
from __future__ import annotations

from typing import Any

# NLLB uses FLORES-200 BCP-47-style codes.
NLLB_LANG = {
    "en": "eng_Latn",
    "ru": "rus_Cyrl",
    "hy": "hye_Armn",
}


class NLLBTranslator:
    id = "nllb"

    def __init__(self, model_id: str, device: str = "cpu") -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # lazy import

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
        self.target_lang = NLLB_LANG["hy"]

    def translate(self, text: str, source_lang: str) -> str:
        src = NLLB_LANG[source_lang]
        # NLLB expects the source-language tag set on the tokenizer.
        self.tokenizer.src_lang = src
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        target_token_id = self.tokenizer.convert_tokens_to_ids(self.target_lang)
        out = self.model.generate(
            **inputs,
            forced_bos_token_id=target_token_id,
            max_new_tokens=256,
            num_beams=4,
        )
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]


def build(config: dict[str, Any]) -> NLLBTranslator:
    return NLLBTranslator(model_id=config["model_id"], device=config.get("device", "cpu"))
