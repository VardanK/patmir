"""ASR stage — OpenRouter multimodal chat completions (cloud).

OpenRouter does not expose a dedicated `/v1/audio/transcriptions` endpoint
(verified 2026-05-26). Audio-in is supported only via `/v1/chat/completions`
with `input_audio` content blocks. Word-level timestamps are therefore not
emitted by the model — every call is followed by a forced-alignment pass
(`_aligner.align_words`) on the same audio to recover precise word timings.

Active model: `openai/gpt-audio`. Alternates registered in patmir.toml:
`mistralai/voxtral-small-24b-2507` (cheaper, weaker Russian),
`openai/gpt-audio-mini` (cheap but tends to refuse short cartoon clips —
shipped as disabled).

Reads the API key from the env var named by `api_key_env` (default
`OPENROUTER_API_KEY`). Raises a clear error if missing.

Audio is base64-encoded WAV in the JSON body. Long inputs are chunked
into ≤`chunk_seconds` segments; transcripts are concatenated and the
forced-aligner restores per-word timing across the full vocals file.

See ADR `2026-05-26-cloud-asr-via-openrouter.md` for the local
exhaustion + pivot history that triggered this.
"""
from __future__ import annotations

import base64
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import httpx

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import AudioTrack, Sentence, Transcript, Word
from patmir.pipeline.stages.asr._aligner import align_words
from patmir.pipeline.stages.asr._sentence import (
    group_into_sentences,
    is_non_lexical,
    write_transcript,
)

_BASE_URL = "https://openrouter.ai/api/v1"

_SYSTEM_PROMPT = (
    "You are a verbatim speech transcriber. Listen to the audio and output "
    "ONLY the spoken dialogue (lexical words) exactly as heard, in the source "
    "language's native script. Preserve punctuation (.,!?) where the prosody "
    "suggests it. Put each spoken sentence on its own line. "
    "DO NOT transcribe non-lexical vocalizations: laughter ('ха-ха'), "
    "screams ('а-а-а'), sobs, sighs, gasps, animal sounds, whistles, "
    "or onomatopoeia. Skip them entirely — leave their time silent rather "
    "than filling with repeated tokens. "
    "No translation, no commentary, no language tags, no quotation marks, "
    "no speaker labels, no bracketed sound descriptions like '[laughs]'. "
    "If the audio contains no lexical speech, output an empty response."
)

_LANG_NAME = {
    "ru": "Russian",
    "en": "English",
    "hy": "Armenian",
    "zh": "Chinese",
    "uk": "Ukrainian",
}


class OpenRouterSTT:
    id = "openrouter_stt"

    def __init__(
        self,
        model: str,
        api_key_env: str = "OPENROUTER_API_KEY",
        base_url: str = _BASE_URL,
        chunk_seconds: float = 240.0,
        timeout_seconds: float = 240.0,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        language: str | None = None,
        referer: str = "https://github.com/patmir-dub",
        title: str = "Patmir Armenian Dubbing",
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"openrouter STT: env var {api_key_env!r} is not set. "
                f"Sign up at openrouter.ai, create a key, and "
                f"`export {api_key_env}=sk-or-...` (or add it to .env)."
            )
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.chunk_seconds = chunk_seconds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.language = language
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": referer,
            "X-Title": title,
        }
        self._client = httpx.Client(timeout=timeout_seconds)

    def transcribe(self, audio: AudioTrack, ctx: JobContext) -> Transcript:
        ctx.progress("asr", 0.02, f"openrouter STT ({self.model})")
        language = ctx.source_lang or self.language or "ru"
        ctx.source_lang = language

        chunks = self._plan_chunks(audio, ctx)
        ctx.progress("asr", 0.05, f"transcribing {len(chunks)} chunk(s) of ≤{self.chunk_seconds:.0f}s")

        chunk_texts: list[str] = []
        with tempfile.TemporaryDirectory(prefix="patmir-or-stt-") as tmpd:
            tmp_root = Path(tmpd)
            for i, (start_s, end_s) in enumerate(chunks):
                chunk_path = tmp_root / f"chunk_{i:03d}.wav"
                _slice_wav(audio.path, chunk_path, start_s, end_s - start_s)
                text = self._transcribe_chunk(chunk_path, language)
                chunk_texts.append(text.strip())
                done = (i + 1) / max(1, len(chunks))
                ctx.progress("asr", 0.05 + 0.55 * done, f"chunk {i + 1}/{len(chunks)}")

        full_text = "\n".join(t for t in chunk_texts if t).strip()
        if not full_text:
            ctx.progress("asr", 1.0, "no speech")
            transcript = Transcript(language=language, words=[], sentences=[])
            write_transcript(ctx.job_dir, transcript)
            return transcript

        align_text = _strip_nonlexical_lines(full_text)
        if not align_text:
            ctx.progress("asr", 1.0, "no lexical speech after filter")
            transcript = Transcript(language=language, words=[], sentences=[])
            write_transcript(ctx.job_dir, transcript)
            return transcript

        ctx.progress(
            "asr", 0.65,
            f"forced-aligning {len(align_text)} chars (filtered from {len(full_text)})",
        )
        words = align_words(audio.path, align_text, language)
        if not words:
            # Aligner returned nothing (model load failed, language unsupported,
            # etc.) — fall back to distributing words proportionally across the
            # full audio duration. Worse for stretch-cap math but keeps the
            # pipeline moving.
            ctx.progress("asr", 0.85, "aligner returned empty; using proportional fallback")
            words = _proportional_words(full_text, audio.duration)

        sentences = _build_sentences(words)
        transcript = Transcript(language=language, words=words, sentences=sentences)
        write_transcript(ctx.job_dir, transcript)
        ctx.progress("asr", 1.0, f"done ({len(sentences)} sentences)")
        return transcript

    def _plan_chunks(self, audio: AudioTrack, ctx: JobContext) -> list[tuple[float, float]]:
        duration = float(audio.duration)
        if duration <= self.chunk_seconds:
            return [(0.0, duration)]
        vad: list[tuple[float, float]] | None = ctx.extra.get("vad_intervals")
        starts = [0.0]
        while starts[-1] + self.chunk_seconds < duration:
            tentative = starts[-1] + self.chunk_seconds
            split = _snap_to_silence(tentative, vad, window=8.0)
            if split <= starts[-1] + 1.0:
                split = tentative
            starts.append(split)
        chunks: list[tuple[float, float]] = []
        for i, s in enumerate(starts):
            e = starts[i + 1] if i + 1 < len(starts) else duration
            chunks.append((s, e))
        return chunks

    def _transcribe_chunk(self, wav_path: Path, language: str) -> str:
        data = base64.b64encode(wav_path.read_bytes()).decode("ascii")
        lang_name = _LANG_NAME.get(language, language)
        user_msg = (
            f"Transcribe the following audio verbatim in {lang_name}. "
            "Output only the transcript text."
        )
        payload = {
            "model": self.model,
            "modalities": ["text"],
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_msg},
                        {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}},
                    ],
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"openrouter STT: HTTP {resp.status_code} from {self.model}: "
                f"{resp.text[:500]}"
            )
        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"openrouter STT: unexpected response shape from {self.model}: "
                f"{str(data)[:500]}"
            ) from exc
        return content or ""


def _strip_nonlexical_lines(text: str) -> str:
    """Remove non-lexical lines from a multi-line transcript before alignment.

    MMS-FA's CTC alignment can be dominated by long runs of identical tokens
    (e.g. 200 repetitions of `А-а-а!` during cartoon screaming) — they consume
    most of the audio span and crush real dialogue into tiny windows. Drop
    them here so the aligner only places lexical content.
    """
    import re
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", text) if ln.strip()]
    kept: list[str] = []
    for ln in lines:
        # Also break on sentence-final punctuation to catch lines where the
        # cloud model packed multiple sentences without newlines.
        for chunk in re.split(r"(?<=[.!?…])\s+", ln):
            chunk = chunk.strip()
            if chunk and not is_non_lexical(chunk):
                kept.append(chunk)
    return " ".join(kept).strip()


def _slice_wav(src: Path, dst: Path, start_s: float, dur_s: float) -> None:
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-ss", f"{start_s:.3f}",
            "-t", f"{dur_s:.3f}",
            "-i", str(src),
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(dst),
        ],
        check=True,
    )


def _snap_to_silence(
    t: float,
    vad: list[tuple[float, float]] | None,
    window: float,
) -> float:
    """Move `t` to the nearest VAD silence within ±`window` seconds. Falls
    back to `t` if no VAD or no silence found in the window."""
    if not vad:
        return t
    lo, hi = t - window, t + window
    best = t
    best_score = float("inf")
    prev_end = 0.0
    for vs, ve in vad:
        # silence gap between prev_end and vs
        gap_lo, gap_hi = prev_end, vs
        if gap_hi >= lo and gap_lo <= hi:
            cand = (max(gap_lo, lo) + min(gap_hi, hi)) / 2.0
            score = abs(cand - t)
            if score < best_score:
                best = cand
                best_score = score
        prev_end = ve
    return best


def _build_sentences(words: list[Word]) -> list[Sentence]:
    sentences = group_into_sentences(words, max_gap_s=0.6)
    # Drop pure-onomatopoeic sentences — same policy as the local Whisper
    # candidate, so downstream stages see the same shape.
    kept = [s for s in sentences if s.text.strip() and not is_non_lexical(s.text)]
    for i, s in enumerate(kept):
        s.idx = i
    return kept


def _proportional_words(text: str, duration: float) -> list[Word]:
    """Last-resort fallback when the forced-aligner returns nothing.
    Distributes whitespace-split tokens proportionally to character count
    across the full audio duration."""
    tokens = [t for t in text.split() if t]
    if not tokens or duration <= 0:
        return []
    total_chars = sum(len(t) for t in tokens)
    if total_chars == 0:
        return []
    out: list[Word] = []
    cursor = 0.0
    for tok in tokens:
        share = (len(tok) / total_chars) * duration
        out.append(Word(text=tok, start=cursor, end=cursor + share))
        cursor += share
    return out


def build(config: dict[str, Any]) -> OpenRouterSTT:
    return OpenRouterSTT(
        model=config["model"],
        api_key_env=config.get("api_key_env", "OPENROUTER_API_KEY"),
        base_url=config.get("base_url", _BASE_URL),
        chunk_seconds=float(config.get("chunk_seconds", 240.0)),
        timeout_seconds=float(config.get("timeout_seconds", 240.0)),
        max_tokens=int(config.get("max_tokens", 4096)),
        temperature=float(config.get("temperature", 0.0)),
        language=config.get("language"),
        referer=config.get("referer", "https://github.com/patmir-dub"),
        title=config.get("title", "Patmir Armenian Dubbing"),
    )
