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
    "You translate dialogue into spoken, casual Eastern Armenian (hy_AM) — "
    "the way families and kids actually talk in Yerevan, NOT formal book "
    "Armenian and NOT news-anchor register. This is for a cartoon dub, so "
    "keep it playful, natural, and conversational. "
    "Concrete guidance: "
    "(1) prefer everyday colloquial vocabulary over literary equivalents "
    "(e.g. 'լավ' over 'բարի', 'ուզում եմ' over 'ցանկանում եմ'); "
    "(2) use spoken contractions and elisions where a native speaker would; "
    "(3) keep sentences short — match the source's rhythm, don't pad with "
    "subordinate clauses; "
    "(4) match the source's register — exclamations stay exclamatory, "
    "questions stay direct, slang/baby-talk in the source stays slang in "
    "the target; "
    "(5) preserve emotion and tone (anger sounds angry, joy sounds bouncy); "
    "(6) for sound-effect-ish words ('Ого!', 'Ой!', 'Фу!') use the natural "
    "Armenian equivalent ('Վա՜յ', 'Օ՜', 'Ֆու՜'), not a literary substitution. "
    "Never produce Western Armenian (hyw). "
    "Reply with ONLY the Armenian translation — no quotes, no explanations, "
    "no source-language echoing, no translator notes."
)

_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterTranslator:
    id = "openrouter"
    # Network-bound + TTFT-dominated (~1s/round-trip) → the translate adapter
    # fires sentences concurrently. httpx.Client is thread-safe. 6 stays well
    # under OpenRouter rate limits while cutting the serial 40-63s stage to
    # seconds (performance.md roadmap #1).
    translate_concurrency = 6

    def __init__(
        self,
        model: str,
        api_key_env: str = "OPENROUTER_API_KEY",
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        timeout_seconds: float = 60.0,
        base_url: str = _BASE_URL,
        # Reasoning controls. Some OpenRouter models (gemini-3.5-flash,
        # gemini-3.1-pro-preview, gpt-5, claude-opus-4.7 with extended
        # thinking) burn 400-1500 reasoning tokens per call by default —
        # makes translate stages crawl serially and burn cost. We default
        # to `effort=minimal` which forces 0 reasoning tokens on models
        # that mandate thinking (Gemini 3.5+ refuses `enabled=false`).
        # `thinking`: DeepSeek-native control (`{type: "disabled"}` turns off
        # chain-of-thought). OpenRouter passes provider-specific fields through.
        # Use this instead of `reasoning.effort` for DeepSeek models — they
        # ignore OpenRouter's reasoning abstraction and stay in thinking mode
        # by default, which on V4 Pro adds ~12 s per call. Verified 2026-05-27.
        # Set `reasoning=None` to opt out of sending the parameter (some
        # legacy models don't accept it).
        reasoning: dict | None = None,
        # DeepSeek-native control passed straight through OpenRouter. Set
        # `{"type": "disabled"}` to skip chain-of-thought.
        thinking: dict | None = None,
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
        self.reasoning = reasoning
        self.thinking = thinking
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
                f"Translate this {src_name} cartoon line into casual spoken "
                f"Eastern Armenian. The speaker's emotion is {emotion}; the "
                f"Armenian should sound {emotion} too.\n\n"
                f"{text}"
            )
        else:
            user_msg = (
                f"Translate this {src_name} cartoon line into casual spoken "
                f"Eastern Armenian:\n\n"
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
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        if self.thinking is not None:
            payload["thinking"] = self.thinking
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
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"openrouter translator: unexpected response shape from {self.model}: "
                f"{str(data)[:500]}"
            ) from exc
        # Some reasoning-enabled models (DeepSeek V4 Pro, others) occasionally
        # return `content: null` while the answer lives in `reasoning` or
        # `reasoning_content`. Fall through in order so the contract still
        # returns a usable string instead of crashing the translate loop.
        content = message.get("content")
        if not content:
            content = message.get("reasoning_content") or message.get("reasoning") or ""
        if not isinstance(content, str):
            content = ""
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
        reasoning=config.get("reasoning"),
        thinking=config.get("thinking"),
        referer=config.get("referer", "https://github.com/patmir-dub"),
        title=config.get("title", "Patmir Armenian Dubbing"),
    )
