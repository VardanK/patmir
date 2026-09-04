"""Translator via OpenRouter — gateway to frontier LLMs.

OpenRouter (openrouter.ai) exposes an OpenAI-compatible chat-completions
endpoint that routes to GPT-5, Claude Opus 4.7, Gemini 2.5 Flash, etc.
behind a single API key. We use it as the cloud-translator escalation
path per the local-first policy:

> Cloud is opt-in per-stage only, brought in after the local set for
> that stage has been judged inadequate.

See ADR `2026-05-26-cloud-translate-via-openrouter.md` for the local
exhaustion that triggered this.

Same `translate(text, source_lang, emotion=None, audience=None) -> str`
contract as the local translator candidates (Qwen3-Instruct-MLX in
particular). The SentenceTranslator adapter probes the signature and only
forwards `emotion=` / `audience=` when accepted — this candidate accepts
both. `audience` is the age-register guardrail (see `_audience_clause`);
candidates that take no prompt at all (NLLB, MADLAD, OPUS-MT) simply never
receive it, and the adapter warns rather than failing.

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

# ── Audience age guardrail ────────────────────────────────────────────────────
#
# An additive clause appended to `_SYSTEM`. "adult" contributes the empty string,
# so the adult prompt is byte-for-byte what it was before this feature existed
# and nothing regresses — a property pinned by test.
#
# The rule these clauses enforce is PROMPT-ONLY SOFTENING: choose gentler
# wording, never remove content. This dub is watched by a hearing-aid-wearing
# child who is reading the dub the way another viewer reads subtitles — a missing
# line is a hole in the story, strictly worse than a softened word. Hence the
# HARD RULE paragraph in both clauses, and hence no redaction / bleep / "[...]"
# vocabulary is offered to the model anywhere. Giving it that vocabulary is how
# you get a safety-shaped refusal back as a silent gap.
#
# Shared with `openrouter_batch.py`, which imports `_audience_clause` next to
# `_SYSTEM` so register guidance cannot drift between the per-sentence and the
# batched candidate.

_AUDIENCE_CHILD = (
    "\n\n"
    "AUDIENCE — YOUNG CHILD (about 5-10 years old). The Armenian must be safe to "
    "hear at that age AND must still say everything the source says. Soften the "
    "WORDS, never the content. Specifically: "
    "(1) profanity and crude insults become the mild things an Armenian child "
    "actually hears — 'վա՜յ', 'ա՛յ քեզ բան', 'ի՜նչ հիմար բան', 'տո՛ բալես' — never "
    "obscene, never scatological, never an anatomical or sexual curse; "
    "(2) never reproduce a slur of any kind (ethnic, religious, disability, "
    "sexual): translate its FORCE, not its content — the speaker still sounds "
    "hostile and rude toward that person, in plain non-degrading words; "
    "(3) sexual content stays non-explicit: attraction, kissing, flirting, "
    "'սիրած' are fine; anything graphic becomes a short everyday euphemism an "
    "adult recognizes and a child hears as grown-up business; "
    "(4) violence keeps the fact and drops the gore — someone is hit, hurt or "
    "killed, without describing wounds, blood, or how a body looks; threats stay "
    "threatening but never relish the detail; "
    "(5) alcohol, drugs, smoking and gambling are named as plainly as the plot "
    "needs ('խմել ա', 'ծխում ա') and never made to sound appealing. "
    "Do not moralize, do not add a warning, do not lecture, do not explain that "
    "you softened something. Do not change who says what, the plot facts, the "
    "names, or the numbers. "
    "HARD RULE: every input line gets one real, speakable Armenian sentence. "
    "Never delete a line, never return an empty or whitespace-only line, never "
    "shorten a line to nothing, never bleep it, star it out, or mark it '[...]', "
    "and never replace it with a refusal or a note about its content. If a line "
    "is hard, write the gentlest TRUE version of it and move on — a missing line "
    "is worse for this listener than a softened one."
)

_AUDIENCE_TEENAGER = (
    "\n\n"
    "AUDIENCE — TEENAGER (about 11-15 years old). Keep the edge, lose the "
    "obscenity. Specifically: "
    "(1) strong language lands as the sharp-but-clean Armenian a teenager would "
    "actually use — 'գժվե՛լ ա', 'էշ բան ա', 'տո՛ լավ էլի', 'դժոխք' — insults may "
    "bite, but no obscenities, no scatology, no sexual or anatomical curses; "
    "(2) never reproduce a slur of any kind (ethnic, religious, disability, "
    "sexual): keep the hostility, drop the epithet; "
    "(3) sex may be acknowledged directly — attraction, a relationship, someone "
    "sleeping with someone — but not described; keep it a fact of the plot, not "
    "an image; "
    "(4) violence may be tense and frightening, death included, without "
    "lingering on gore or on how a body looks; "
    "(5) alcohol, drugs and smoking may be named and discussed, consequences "
    "included, with no instructions and no glamour. "
    "Register stays teen-natural — neither childish nor sanitized. Do not "
    "moralize, do not add a warning, do not explain that you softened something. "
    "Do not change who says what, the plot facts, the names, or the numbers. "
    "HARD RULE: every input line gets one real, speakable Armenian sentence. "
    "Never delete a line, never return an empty or whitespace-only line, never "
    "bleep it, star it out, or mark it '[...]', and never replace it with a "
    "refusal — a missing line is worse for this listener than a softened one."
)

# "adult" is the historical behaviour: no clause at all.
_AUDIENCE_CLAUSES: dict[str, str] = {
    "child": _AUDIENCE_CHILD,
    "teenager": _AUDIENCE_TEENAGER,
    "adult": "",
}


def _audience_clause(level: str | None) -> str:
    """Register clause for an audience level, or "" for adult/None/unknown.

    Deliberately tolerant of an unknown value rather than raising: this module is
    a Phase-0 candidate and stays isolated from the pipeline's data contracts.
    Validation happens at the edges — the config loader, `JobSpec.__post_init__`,
    and the HTTP handlers.
    """
    return _AUDIENCE_CLAUSES.get((level or "").strip().lower(), "")


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
        # How many sentence round-trips are in flight at once. Overrides the
        # class default so the knob that decides ~13% of batch wall is tunable
        # from `patmir.toml` instead of only from code.
        concurrency: int | None = None,
        # Bounded retry for transient failures. Before this, ANY non-200 —
        # a 429, a provider 5xx, a dropped connection — raised straight out of
        # the thread pool and failed the whole translate stage, discarding
        # every sentence already translated in that job.
        max_attempts: int = 3,
        retry_backoff_s: float = 0.5,
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
        if concurrency is not None:
            self.translate_concurrency = max(1, int(concurrency))
        self.max_attempts = max(1, int(max_attempts))
        self.retry_backoff_s = float(retry_backoff_s)
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": referer,
            "X-Title": title,
        }
        self._client = httpx.Client(timeout=timeout_seconds)

    # Statuses worth trying again: rate limiting, and the provider's own
    # transient failures. A 400 or a 401 will fail identically next time, so
    # retrying them only delays a real error.
    _RETRYABLE = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

    def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """One chat completion, retried on transient failures.

        Before this, any non-200 raised straight out of the adapter's thread
        pool and failed the whole translate stage — throwing away every
        sentence already translated in that job for one 429. Cloud output is
        already non-deterministic at temperature 0.2, so a retry cannot change
        the guarantees; it only decides whether one bad round-trip costs a
        sentence or the run.
        """
        import time as _time

        last: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                resp = self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                )
            except Exception as exc:  # noqa: BLE001 — httpx transport errors
                last = exc
            else:
                if resp.status_code == 200:
                    return resp.json()
                last = RuntimeError(
                    f"openrouter translator: HTTP {resp.status_code} from "
                    f"{self.model}: {resp.text[:500]}"
                )
                if resp.status_code not in self._RETRYABLE:
                    raise last
            if attempt + 1 < self.max_attempts:
                # 0.5, 1.0, 2.0 ... — short enough not to stall a watch-along
                # section, long enough for a rate limiter to drain.
                _time.sleep(self.retry_backoff_s * (2 ** attempt))
        assert last is not None
        raise last

    def translate(
        self,
        text: str,
        source_lang: str,
        emotion: str | None = None,
        audience: str | None = None,
    ) -> str:
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
                {"role": "system", "content": _SYSTEM + _audience_clause(audience)},
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
        data = self._post_with_retry(payload)
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
        concurrency=config.get("concurrency"),
        max_attempts=int(config.get("max_attempts", 3)),
        retry_backoff_s=float(config.get("retry_backoff_s", 0.5)),
        referer=config.get("referer", "https://github.com/patmir-dub"),
        title=config.get("title", "Patmir Armenian Dubbing"),
    )
