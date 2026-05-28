"""Batched OpenRouter translator with structured JSON output.

Translates N sentences per call instead of one-at-a-time. The LLM sees:

  - the chunk's source sentences (with `speaker_id` + `emotion` as conditioning
    metadata),
  - a small overlap of already-translated prior sentences as discourse context
    (target_text only — we never re-ask the model for them),

and returns a JSON array of `{idx, text}`. The schema is enforced via
OpenRouter's `response_format: {type: "json_schema", strict: true}`. We
validate that the returned `idx` set matches what we asked for; on any
mismatch the candidate falls back to per-sentence translation for that
chunk so one bad LLM response doesn't kill the run.

Quality goals over the per-sentence path:

  - Character voice coherence: with `speaker_id` in the input metadata + a
    couple of that character's earlier translated lines visible as context,
    each character's vocabulary register stays stable across the chunk.
  - Anaphora resolution: pronouns ("he"/"she"/"that") can resolve against
    the surrounding sentences instead of being guessed.
  - Discourse connectives: "потом", "так что", "наконец" translate stronger
    with surrounding lines visible.
  - Latency: ~10× fewer API round-trips on a 200-sentence episode.

Same OpenRouter endpoint / auth / reasoning controls as the per-sentence
candidate. The two share `_SYSTEM` so register guidance stays consistent.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import httpx

from patmir.eval.candidates.translation.openrouter import (
    OpenRouterTranslator,
    _LANG_NAME,
    _SYSTEM,
    _strip_thinking,
)

_BASE_URL = "https://openrouter.ai/api/v1"

# JSON schema we send via `response_format`. OpenRouter forwards this to the
# upstream provider; Gemini 3.5+ honors `strict=true` and refuses to emit
# additional keys. Even when an older provider ignores the schema, the
# instructions in `_user_prompt_for_chunk` ask for the same shape and the
# post-parse validation catches drift.
_RESPONSE_SCHEMA: dict = {
    "name": "patmir_translations",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "translations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "idx":  {"type": "integer"},
                        "text": {"type": "string"},
                    },
                    "required": ["idx", "text"],
                },
            },
        },
        "required": ["translations"],
    },
}


@dataclass
class _Item:
    """Inputs the candidate accepts. Decoupled from `pipeline.stage.Sentence`
    so candidates stay isolated from pipeline data contracts."""
    idx: int
    text: str
    speaker_id: str | None = None
    emotion: str | None = None


def to_items(sentences: Iterable[Any]) -> list[_Item]:
    """Coerce an iterable of `Sentence`-like / dict-like inputs into `_Item`s."""
    out: list[_Item] = []
    for s in sentences:
        if isinstance(s, _Item):
            out.append(s)
        elif isinstance(s, dict):
            out.append(_Item(
                idx=int(s["idx"]),
                text=str(s["text"]),
                speaker_id=s.get("speaker_id"),
                emotion=s.get("emotion"),
            ))
        else:
            out.append(_Item(
                idx=int(getattr(s, "idx")),
                text=str(getattr(s, "text")),
                speaker_id=getattr(s, "speaker_id", None),
                emotion=getattr(s, "emotion", None),
            ))
    return out


class OpenRouterBatchTranslator:
    id = "openrouter_batch"

    def __init__(
        self,
        model: str,
        api_key_env: str = "OPENROUTER_API_KEY",
        max_tokens: int = 4096,
        temperature: float = 0.2,
        top_p: float = 0.9,
        timeout_seconds: float = 120.0,
        base_url: str = _BASE_URL,
        chunk_size: int = 20,
        overlap: int = 2,
        reasoning: dict | None = None,
        thinking: dict | None = None,
        referer: str = "https://github.com/patmir-dub",
        title: str = "Patmir Armenian Dubbing",
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"openrouter batch translator: env var {api_key_env!r} is not set."
            )
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                f"overlap must be in [0, chunk_size); got {overlap} with chunk_size={chunk_size}"
            )
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.chunk_size = chunk_size
        self.overlap = overlap
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
        # Per-sentence fallback shares the OpenRouter client + endpoint so a
        # batch failure can degrade to one-call-per-sentence without spinning
        # up another candidate.
        self._fallback = OpenRouterTranslator(
            model=model,
            api_key_env=api_key_env,
            max_tokens=512,
            temperature=temperature,
            top_p=top_p,
            timeout_seconds=timeout_seconds,
            base_url=base_url,
            reasoning=reasoning,
            thinking=thinking,
            referer=referer,
            title=title,
        )

    # --- public API ------------------------------------------------------

    def translate(self, text: str, source_lang: str, emotion: str | None = None) -> str:
        """Single-sentence path (uses the fallback candidate). Lets this
        candidate be used by the adapter the same way as the per-sentence
        one when only one sentence is in scope."""
        return self._fallback.translate(text, source_lang=source_lang, emotion=emotion)

    def translate_batch(
        self,
        sentences: Iterable[Any],
        source_lang: str,
        *,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> dict[int, str]:
        """Translate a list of sentences in chunks. Returns `{idx: target_text}`.

        Chunks of `chunk_size`. The `overlap` last sentences of each chunk's
        committed translations are included as read-only discourse context
        in the next chunk's prompt (translated text only — we never re-ask
        the model for them).

        On per-chunk failure (transport error, malformed JSON, mismatched
        idx set, empty translations), the candidate falls back to
        per-sentence translation for that chunk's sentences. The run keeps
        going.
        """
        items = to_items(sentences)
        results: dict[int, str] = {}
        if not items:
            return results

        chunks = _split_into_chunks(items, self.chunk_size)
        total = len(items)
        translated_so_far = 0
        prior: list[tuple[_Item, str]] = []  # (source item, target_text) for overlap context

        for ci, chunk in enumerate(chunks):
            try:
                chunk_out = self._translate_chunk(
                    chunk, source_lang=source_lang, prior=prior[-self.overlap:] if self.overlap else [],
                )
            except _BatchError as exc:
                # Per-chunk fallback — keep going.
                if progress:
                    progress(translated_so_far, total, f"chunk {ci + 1}/{len(chunks)} fell back to per-sentence ({exc})")
                chunk_out = self._fallback_chunk(chunk, source_lang=source_lang)

            for it in chunk:
                target = chunk_out.get(it.idx, "")
                results[it.idx] = target
                if target:
                    prior.append((it, target))
                translated_so_far += 1
                if progress:
                    progress(translated_so_far, total, f"sentence {translated_so_far}/{total}")

        return results

    # --- internals -------------------------------------------------------

    def _translate_chunk(
        self,
        chunk: list[_Item],
        *,
        source_lang: str,
        prior: list[tuple[_Item, str]],
    ) -> dict[int, str]:
        user_msg = _user_prompt_for_chunk(chunk, source_lang=source_lang, prior=prior)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM + "\n\n" + _BATCH_SYSTEM_SUFFIX},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "response_format": {"type": "json_schema", "json_schema": _RESPONSE_SCHEMA},
        }
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning
        if self.thinking is not None:
            payload["thinking"] = self.thinking

        try:
            resp = self._client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise _BatchError(f"transport: {type(exc).__name__}: {exc}") from exc
        if resp.status_code != 200:
            raise _BatchError(f"HTTP {resp.status_code}: {resp.text[:300]}")

        try:
            content = _extract_message_content(resp.json())
        except _BatchError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _BatchError(f"unexpected response: {exc}") from exc

        parsed = _parse_translations(content)
        # Validate set match.
        expected = {it.idx for it in chunk}
        got = {t["idx"] for t in parsed}
        if got != expected:
            raise _BatchError(
                f"idx mismatch: expected {sorted(expected)}, got {sorted(got)}"
            )
        out: dict[int, str] = {}
        for t in parsed:
            text = (t.get("text") or "").strip()
            if not text:
                raise _BatchError(f"empty translation for idx={t['idx']}")
            out[int(t["idx"])] = text
        return out

    def _fallback_chunk(self, chunk: list[_Item], *, source_lang: str) -> dict[int, str]:
        out: dict[int, str] = {}
        for it in chunk:
            try:
                out[it.idx] = self._fallback.translate(
                    it.text, source_lang=source_lang, emotion=it.emotion,
                )
            except Exception:  # noqa: BLE001
                out[it.idx] = ""
        return out


class _BatchError(RuntimeError):
    pass


_BATCH_SYSTEM_SUFFIX = (
    "You will receive several sentences in JSON. For each sentence, return one "
    "Armenian translation. Respond ONLY with the JSON object described by the "
    "schema, never with surrounding prose. Preserve the `idx` of every input "
    "sentence — never merge or split sentences. The `speaker_id` and `emotion` "
    "fields are read-only conditioning hints: use them to keep each character's "
    "vocabulary consistent across the chunk and to color the wording with the "
    "right emotion."
)


def _user_prompt_for_chunk(
    chunk: list[_Item], *, source_lang: str, prior: list[tuple[_Item, str]],
) -> str:
    src_name = _LANG_NAME.get(source_lang, source_lang)
    payload: dict[str, Any] = {
        "source_language": src_name,
        "target_language": "Eastern Armenian (hy_AM)",
        "sentences": [
            {
                "idx": it.idx,
                "speaker_id": it.speaker_id or "",
                "emotion": it.emotion or "neutral",
                "text": it.text,
            }
            for it in chunk
        ],
    }
    if prior:
        payload["context"] = [
            {
                "idx": p_item.idx,
                "speaker_id": p_item.speaker_id or "",
                "text_source": p_item.text,
                "text_target": p_target,
            }
            for p_item, p_target in prior
        ]
    return (
        "Translate every entry in `sentences` to casual spoken Eastern Armenian. "
        "Return JSON matching the response schema: a `translations` array with "
        "one object `{idx, text}` per input sentence, idx preserved.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _extract_message_content(data: dict) -> str:
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise _BatchError(f"unexpected response shape: {str(data)[:300]}") from exc
    content = message.get("content")
    if not content:
        content = message.get("reasoning_content") or message.get("reasoning") or ""
    if not isinstance(content, str):
        raise _BatchError(f"non-string content: {type(content).__name__}")
    return _strip_thinking(content).strip()


def _parse_translations(content: str) -> list[dict]:
    """Find the JSON object in `content` (LLMs sometimes wrap it in prose
    even with strict schemas) and return its `translations` array."""
    text = content.strip()
    # Try direct parse first.
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = None
    if obj is None:
        # Last-resort: find the first top-level `{` … `}` slice.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise _BatchError(f"no JSON object found in content: {text[:200]!r}")
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise _BatchError(f"JSON parse failed: {exc}") from exc
    if not isinstance(obj, dict):
        raise _BatchError(f"top-level JSON is not an object: {type(obj).__name__}")
    arr = obj.get("translations")
    if not isinstance(arr, list):
        raise _BatchError("missing `translations` array")
    for t in arr:
        if not isinstance(t, dict) or "idx" not in t or "text" not in t:
            raise _BatchError(f"bad translation entry: {t!r}")
    return arr


def _split_into_chunks(items: list[_Item], chunk_size: int) -> list[list[_Item]]:
    chunks: list[list[_Item]] = []
    for i in range(0, len(items), chunk_size):
        chunks.append(items[i : i + chunk_size])
    return chunks


def build(config: dict[str, Any]) -> OpenRouterBatchTranslator:
    return OpenRouterBatchTranslator(
        model=config["model"],
        api_key_env=config.get("api_key_env", "OPENROUTER_API_KEY"),
        max_tokens=int(config.get("max_tokens", 4096)),
        temperature=float(config.get("temperature", 0.2)),
        top_p=float(config.get("top_p", 0.9)),
        timeout_seconds=float(config.get("timeout_seconds", 120.0)),
        base_url=config.get("base_url", _BASE_URL),
        chunk_size=int(config.get("chunk_size", 20)),
        overlap=int(config.get("overlap", 2)),
        reasoning=config.get("reasoning"),
        thinking=config.get("thinking"),
        referer=config.get("referer", "https://github.com/patmir-dub"),
        title=config.get("title", "Patmir Armenian Dubbing"),
    )
