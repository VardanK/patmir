"""OpenAI GPT translator (cloud ceiling reference)."""
from __future__ import annotations

import os
from typing import Any

SOURCE_NAME = {"en": "English", "ru": "Russian"}

PROMPT = (
    "Translate the following {source_name} text into Eastern Armenian. "
    "Preserve register and tone. Output only the Armenian translation, "
    "with no commentary or quotation marks.\n\n{text}"
)


class GPTTranslator:
    id = "gpt"

    def __init__(self, model: str) -> None:
        from openai import OpenAI  # lazy import

        self.client = OpenAI()  # picks up OPENAI_API_KEY
        self.model = model

    def translate(self, text: str, source_lang: str) -> str:
        prompt = PROMPT.format(source_name=SOURCE_NAME[source_lang], text=text)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()


def build(config: dict[str, Any]) -> GPTTranslator:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set; cannot run gpt_cloud candidate.")
    return GPTTranslator(model=config["model"])
