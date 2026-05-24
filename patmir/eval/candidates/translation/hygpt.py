"""HyGPT translator candidate (registered for the GPU re-eval, stub in Phase 1).

`Gen2B/HyGPT-10b-it` — Eastern-Armenian-centric LLM based on Gemma-2-9b,
trained on 10B Armenian tokens (July 2025). Supports EN/RU/hy translation
via prompting. ~20 GB VRAM at fp16. Requires the GPU workstation.

The build() factory here loads the model lazily and exposes the standard
`translate(text, source_lang) -> str` interface so the existing eval harness
and Phase 1 pipeline can consume it without special casing.
"""
from __future__ import annotations

from typing import Any

_LANG_NAME = {"en": "English", "ru": "Russian"}


class HyGPTTranslator:
    id = "hygpt"

    def __init__(self, model_id: str, device: str = "cuda") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy import

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    def translate(self, text: str, source_lang: str) -> str:
        src_name = _LANG_NAME.get(source_lang, source_lang)
        prompt = (
            f"Translate the following {src_name} text to Eastern Armenian (hy_AM). "
            f"Reply with only the Armenian translation, no explanation.\n\n"
            f"{src_name}: {text}\nArmenian:"
        )
        chat = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
        out = self.model.generate(inputs, max_new_tokens=256, do_sample=False)
        decoded = self.tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
        return decoded.strip()


def build(config: dict[str, Any]) -> HyGPTTranslator:
    return HyGPTTranslator(
        model_id=config["model_id"],
        device=config.get("device", "cuda"),
    )
