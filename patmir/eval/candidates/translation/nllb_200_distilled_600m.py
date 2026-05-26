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

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        dtype: str = "float32",
        max_new_tokens: int = 512,
        num_beams: int = 4,
        no_repeat_ngram_size: int = 3,
    ) -> None:
        import torch  # lazy
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # lazy import

        self.device = device
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype, torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, torch_dtype=torch_dtype
        ).to(device)
        # NLLB ships `generation_config.max_length=200` which collides with
        # our explicit `max_new_tokens` and triggers a noisy warning every
        # generate() call. Clearing it leaves `max_new_tokens` as the sole
        # length control.
        self.model.generation_config.max_length = None
        self.target_lang = NLLB_LANG["hy"]
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.no_repeat_ngram_size = no_repeat_ngram_size

    def translate(self, text: str, source_lang: str) -> str:
        src = NLLB_LANG[source_lang]
        # NLLB expects the source-language tag set on the tokenizer.
        self.tokenizer.src_lang = src
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        target_token_id = self.tokenizer.convert_tokens_to_ids(self.target_lang)
        out = self.model.generate(
            **inputs,
            forced_bos_token_id=target_token_id,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
        )
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]


def build(config: dict[str, Any]) -> NLLBTranslator:
    return NLLBTranslator(
        model_id=config["model_id"],
        device=config.get("device", "cpu"),
        dtype=config.get("dtype", "float32"),
        max_new_tokens=int(config.get("max_new_tokens", 512)),
        num_beams=int(config.get("num_beams", 4)),
        no_repeat_ngram_size=int(config.get("no_repeat_ngram_size", 3)),
    )
