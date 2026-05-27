"""Translator via OpenRouter — gateway to frontier LLMs.

OpenRouter (openrouter.ai) exposes an OpenAI-compatible chat-completions
endpoint that routes to GPT-5, Claude Opus 4.7, Gemini 2.5 Flash, etc.
behind a single API key. We use it as the cloud-translator escalation
path per the local-first policy:

> Cloud is opt-in per-stage only, brought in after the local set for
> that stage has been judged inadequate.

See ADR `2026-05-26-cloud-translate-via-openrouter.md` for the local
exhaustion that triggered this.

Same `translate(text, source_lang, emotion=None) -> str` contract as the
local translator candidates (Qwen3-Instruct-MLX in particular). The
SentenceTranslator adapter probes the signature and only forwards
`emotion=` when accepted — this candidate accepts it.

Reads the API key from the env var named by `api_key_env` (default
`OPENROUTER_API_KEY`). Raises a clear error if missing.

No new Python deps — uses `httpx` which is already in the venv via
huggingface_hub / fastapi / etc.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

_LANG_NAME = {"en": "English", "ru": "Russian", "hy": "Armenian"}

_SYSTEM = (
    "You are a professional translator specializing in Eastern Armenian (hy_AM). "
    "You translate text from English or Russian into natural, idiomatic Eastern "
    "Armenian as spoken in Armenia. You preserve tone, register, and emotion. "
    "You never produce Western Armenian (hyw). You reply with ONLY the Armenian "
    "translation — no quotes, no explanations, no source-language echoing."
)

_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterTranslator:
    id = "openrouter"

    def __init__(
        self,
        model: str,
        api_key_env: str = "OPENROUTER_API_KEY",
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        timeout_seconds: float = 60.0,
        base_url: str = _BASE_URL,
        # Optional OpenRouter attribution headers (recommended by their docs).
        referer: str = "https://github.com/patmir-dub",
        title: str = "Patmir Armenian Dubbing",
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"openrouter translator: env var {api_key_env!r} is not set. "
                f"Sign up at openrouter.ai, create a key, and "
                f"`export {api_key_env}=sk-or-...`."
            )
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": referer,
            "X-Title": title,
        }
        self._client = httpx.Client(timeout=timeout_seconds)

    def translate(self, text: str, source_lang: str, emotion: str | None = None) -> str:
        src_name = _LANG_NAME.get(source_lang, source_lang)
        if emotion and emotion != "neutral":
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

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"openrouter translator: HTTP {resp.status_code} from {self.model}: "
                f"{resp.text[:500]}"
            )
        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"openrouter translator: unexpected response shape from {self.model}: "
                f"{str(data)[:500]}"
            ) from exc
        return _strip_thinking(content).strip()


def _strip_thinking(text: str) -> str:
    """Some frontier models (Gemini's reasoning variants, Claude with extended
    thinking enabled) wrap reasoning in <think>...</think> tags. Strip the
    block but keep the final answer that follows."""
    if "<think>" in text and "</think>" in text:
        end = text.index("</think>") + len("</think>")
        return text[end:].strip()
    return text


def build(config: dict[str, Any]) -> OpenRouterTranslator:
    return OpenRouterTranslator(
        model=config["model"],
        api_key_env=config.get("api_key_env", "OPENROUTER_API_KEY"),
        max_tokens=int(config.get("max_tokens", 512)),
        temperature=float(config.get("temperature", 0.2)),
        top_p=float(config.get("top_p", 0.9)),
        timeout_seconds=float(config.get("timeout_seconds", 60.0)),
        base_url=config.get("base_url", _BASE_URL),
        referer=config.get("referer", "https://github.com/patmir-dub"),
        title=config.get("title", "Patmir Armenian Dubbing"),
    )
