"""Qwen3-ASR backend.

`Qwen/Qwen3-ASR-0.6B` and `Qwen/Qwen3-ASR-1.7B` — Apache-2.0 multilingual ASR
from Alibaba (Jan 2026). 52 supported languages including EN and RU. On
LibriSpeech "other" (noisy) the 1.7B variant beats Whisper-large-v3 by ~15%
relative WER; English clean is roughly a wash; the multilingual average is
meaningfully better.

Word-level timestamps require the companion `Qwen3-ForcedAligner-0.6B` —
loaded alongside the ASR model via `forced_aligner=...` at init time. Without
the aligner, only segment-level timestamps are available.

Install: `pip install qwen-asr` (the official package from the Qwen team).
Heavy deps; the module imports lazily so adding the candidate to patmir.toml
doesn't pull anything in until the user flips it on.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import AudioTrack, Transcript, Word
from patmir.pipeline.stages.asr._sentence import group_into_sentences, write_transcript


# Qwen3-ASR returns the language as the full English name (e.g. "English",
# "Russian"). Downstream stages expect ISO 639-1 codes. Limited to the source
# languages the rest of the pipeline supports; anything else falls back to
# lowercased first token.
_QWEN_LANG_TO_ISO: dict[str, str] = {
    "english": "en",
    "russian": "ru",
    "armenian": "hy",
}


def _to_iso(language: str | None) -> str | None:
    if not language:
        return None
    key = language.strip().lower()
    if key in _QWEN_LANG_TO_ISO:
        return _QWEN_LANG_TO_ISO[key]
    # Accept already-ISO inputs like "en" / "ru" unchanged.
    if len(key) == 2 and key.isalpha():
        return key
    return key.split()[0]


def _resolve_device(requested: str) -> str:
    if requested in ("cpu", "cuda", "mps"):
        return requested
    try:
        import torch  # type: ignore[import-not-found]
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _device_map_arg(device: str) -> str:
    # qwen-asr's device_map expects "cuda:0" / "cpu" / "mps".
    return "cuda:0" if device == "cuda" else device


def _dtype(name: str):
    import torch  # type: ignore[import-not-found]
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(name, torch.float32)


class Qwen3ASR:
    id = "qwen3_asr"

    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: str = "bfloat16",
        forced_aligner: str | None = "Qwen/Qwen3-ForcedAligner-0.6B",
        aligner_dtype: str = "bfloat16",
        max_new_tokens: int = 256,
    ) -> None:
        from qwen_asr import Qwen3ASRModel  # lazy import (heavy)

        self.device = _resolve_device(device)
        self.model_id = model_id
        self.forced_aligner = forced_aligner
        kwargs: dict[str, Any] = dict(
            dtype=_dtype(dtype),
            device_map=_device_map_arg(self.device),
            max_new_tokens=max_new_tokens,
        )
        if forced_aligner:
            kwargs["forced_aligner"] = forced_aligner
            kwargs["forced_aligner_kwargs"] = dict(
                dtype=_dtype(aligner_dtype),
                device_map=_device_map_arg(self.device),
            )
        self.model = Qwen3ASRModel.from_pretrained(model_id, **kwargs)

    def transcribe(self, audio: AudioTrack, ctx: JobContext) -> Transcript:
        ctx.progress("asr", 0.05, f"qwen3-asr ({self.model_id}) on {self.device}")
        results = self.model.transcribe(
            audio=str(audio.path),
            return_time_stamps=self.forced_aligner is not None,
        )
        ctx.progress("asr", 0.9, "decoded")

        if not results:
            return Transcript(language=ctx.source_lang or "en", words=[], sentences=[])

        first = results[0]
        raw_lang = getattr(first, "language", None)
        language = _to_iso(raw_lang) or ctx.source_lang or "en"
        ctx.source_lang = language

        # Word timestamps are present when the forced aligner was loaded.
        # The qwen_asr API exposes them as `time_stamps` — a list of entries
        # with `word`/`text` + `start` + `end` keys (or tuple).
        words: list[Word] = []
        ts = getattr(first, "time_stamps", None) or getattr(first, "word_timestamps", None)
        if ts:
            for entry in ts:
                word_text, start, end = _parse_ts_entry(entry)
                if word_text is None:
                    continue
                words.append(Word(text=word_text, start=float(start), end=float(end)))
        else:
            # No aligner — synthesize one "word" per character-free segment so
            # downstream sentence-grouping still gets something to chew on.
            # In practice users will run with the aligner enabled.
            text = first.text.strip()
            if text:
                words.append(Word(text=text, start=0.0, end=audio.duration))

        sentences = group_into_sentences(words)
        transcript = Transcript(language=language, words=words, sentences=sentences)
        write_transcript(ctx.job_dir, transcript)
        return transcript


def _parse_ts_entry(entry: Any) -> tuple[str | None, float, float]:
    if isinstance(entry, dict):
        text = entry.get("word") or entry.get("text") or entry.get("token")
        start = entry.get("start") or entry.get("start_time") or entry.get("begin") or 0.0
        end = entry.get("end") or entry.get("end_time") or entry.get("stop") or 0.0
        return (str(text).strip() if text else None, float(start), float(end))
    if isinstance(entry, (list, tuple)) and len(entry) >= 3:
        text, start, end = entry[0], entry[1], entry[2]
        return (str(text).strip() if text else None, float(start), float(end))
    # Object with attributes
    text = getattr(entry, "word", None) or getattr(entry, "text", None)
    start = getattr(entry, "start", 0.0)
    end = getattr(entry, "end", 0.0)
    return (str(text).strip() if text else None, float(start), float(end))


def build(config: dict[str, Any]) -> Qwen3ASR:
    return Qwen3ASR(
        model_id=config["model_id"],
        device=config.get("device", "auto"),
        dtype=config.get("dtype", "bfloat16"),
        forced_aligner=config.get("forced_aligner", "Qwen/Qwen3-ForcedAligner-0.6B"),
        aligner_dtype=config.get("aligner_dtype", "bfloat16"),
        max_new_tokens=int(config.get("max_new_tokens", 256)),
    )
