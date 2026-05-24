"""Anthropic Claude translator (cloud ceiling reference)."""
from __future__ import annotations

import os
from typing import Any

SOURCE_NAME = {"en": "English", "ru": "Russian"}

PROMPT = (
    "Translate the following {source_name} text into Eastern Armenian. "
    "Preserve register and tone. Output only the Armenian translation, "
    "with no commentary or quotation marks.\n\n{text}"
)


class ClaudeTranslator:
    id = "claude"

    def __init__(self, model: str) -> None:
        from anthropic import Anthropic  # lazy import

        self.client = Anthropic()  # picks up ANTHROPIC_API_KEY
        self.model = model

    def translate(self, text: str, source_lang: str) -> str:
        prompt = PROMPT.format(source_name=SOURCE_NAME[source_lang], text=text)
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        # `content` is a list of blocks; the text block carries the translation.
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()


def build(config: dict[str, Any]) -> ClaudeTranslator:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set; cannot run claude_cloud candidate.")
    return ClaudeTranslator(model=config["model"])
