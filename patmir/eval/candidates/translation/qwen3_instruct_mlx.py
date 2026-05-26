"""Qwen3-Instruct translator via mlx-lm (Apple Silicon GPU).

Prompts a Qwen3 chat model to translate the input sentence into Eastern
Armenian. The Qwen3 lineup uses a "hybrid" thinking mode — the same weights
can chain-of-thought reason or respond directly; for translation we suppress
thinking via the `/no_think` chat marker to keep latency low.

Lazy import keeps `mlx-lm` out of the import path when other ASR/translation
candidates are active.

Install once: `uv pip install mlx-lm`
"""
from __future__ import annotations

from typing import Any


_LANG_NAME = {"en": "English", "ru": "Russian", "hy": "Armenian"}


_SYSTEM = (
    "You are a professional translator specializing in Eastern Armenian (hy_AM). "
    "You translate text from English or Russian into natural, idiomatic Eastern "
    "Armenian as spoken in Armenia. You preserve tone, register, and emotion. "
    "You never produce Western Armenian (hyw). You reply with ONLY the Armenian "
    "translation — no quotes, no explanations, no source-language echoing."
)


class Qwen3InstructTranslator:
    id = "qwen3_instruct_mlx"

    def __init__(
        self,
        model_id: str,
        max_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        no_think: bool = True,
    ) -> None:
        from mlx_lm import load  # lazy

        self.model, self.tokenizer = load(model_id)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_think = no_think

    def translate(self, text: str, source_lang: str, emotion: str | None = None) -> str:
        from mlx_lm import generate  # lazy
        from mlx_lm.sample_utils import make_logits_processors, make_sampler  # lazy

        src_name = _LANG_NAME.get(source_lang, source_lang)
        if emotion and emotion != "neutral":
            # The system prompt already directs Qwen3 to "preserve tone,
            # register, and emotion"; the per-sentence tag tells it *which*
            # emotion the source sentence carries, so the Armenian phrasing
            # can match. Skip for neutral — adding "with emotion: neutral"
            # tends to flatten the translation rather than help.
            user_msg = (
                f"Translate this {src_name} sentence into Eastern Armenian. "
                f"The speaker's emotion is {emotion}; reflect that tone in the Armenian phrasing.\n\n"
                f"{text}"
            )
        else:
            user_msg = (
                f"Translate this {src_name} sentence into Eastern Armenian:\n\n"
                f"{text}"
            )
        if self.no_think:
            user_msg += " /no_think"

        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=not self.no_think,
        )
        # A tiny temperature + top-p + repetition penalty avoids the under-
        # trained-on-Armenian LLM falling into "որտեղ ես գնում, որտեղ ես
        # գնում, …" loops on RU source. Greedy decoding (temp=0) loops; a
        # small stochastic floor breaks the cycle without hurting quality
        # noticeably.
        sampler = make_sampler(temp=self.temperature, top_p=self.top_p)
        logits_processors = make_logits_processors(
            repetition_penalty=self.repetition_penalty,
            repetition_context_size=40,
        )
        out = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            verbose=False,
        )
        return _strip_thinking(out).strip()


def _strip_thinking(text: str) -> str:
    """Qwen3's thinking mode wraps reasoning in <think>...</think> tags; even
    with `/no_think` an empty tag pair can appear at the start of output."""
    if "<think>" in text and "</think>" in text:
        end = text.index("</think>") + len("</think>")
        return text[end:].strip()
    return text


def build(config: dict[str, Any]) -> Qwen3InstructTranslator:
    return Qwen3InstructTranslator(
        model_id=config["model_id"],
        max_tokens=int(config.get("max_tokens", 256)),
        temperature=float(config.get("temperature", 0.1)),
        top_p=float(config.get("top_p", 0.9)),
        repetition_penalty=float(config.get("repetition_penalty", 1.15)),
        no_think=bool(config.get("no_think", True)),
    )
