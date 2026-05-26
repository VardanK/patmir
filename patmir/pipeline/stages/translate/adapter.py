"""Translate stage adapter.

Wraps a Phase 0 translator candidate (loaded from `[[eval.translation.candidates]]`)
with the pipeline's typed `TranslateStage` interface. The inner candidate
exposes `translate(text, source_lang) -> str`; the adapter iterates sentences
from the ASR transcript, calls it per sentence, attaches Armenian text + source
timing to each entry, and writes `translated.json`.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import (
    TranslatedSentence,
    TranslatedTranscript,
    Transcript,
)


class SentenceTranslator:
    id = "translate_adapter"

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def translate(self, transcript: Transcript, ctx: JobContext) -> TranslatedTranscript:
        source_lang = ctx.source_lang or transcript.language
        sentences: list[TranslatedSentence] = []
        total = len(transcript.sentences) or 1
        # Some inner translators (e.g. Qwen3-Instruct-MLX) accept an `emotion`
        # kwarg and inject it into the per-sentence prompt; NLLB ignores it.
        # Probe once via the function signature.
        import inspect
        try:
            inner_sig = inspect.signature(self.inner.translate)
            inner_accepts_emotion = "emotion" in inner_sig.parameters
        except (TypeError, ValueError):
            inner_accepts_emotion = False

        for i, s in enumerate(transcript.sentences):
            if inner_accepts_emotion:
                target_text = self.inner.translate(
                    s.text, source_lang=source_lang, emotion=s.emotion,
                )
            else:
                target_text = self.inner.translate(s.text, source_lang=source_lang)
            sentences.append(TranslatedSentence(
                idx=s.idx,
                source_text=s.text,
                target_text=target_text,
                start=s.start,
                end=s.end,
                speaker_id=s.speaker_id,
                emotion=s.emotion,
            ))
            ctx.progress("translate", (i + 1) / total, f"sentence {i + 1}/{total}")

        out = TranslatedTranscript(
            source_lang=source_lang,
            target_lang="hy",
            sentences=sentences,
        )
        _write_translated(ctx.job_dir, out)
        return out


def _write_translated(job_dir: Path, t: TranslatedTranscript) -> None:
    payload = {
        "source_lang": t.source_lang,
        "target_lang": t.target_lang,
        "sentences": [asdict(s) for s in t.sentences],
    }
    (job_dir / "translated.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def build(config: dict[str, Any]) -> SentenceTranslator:
    """Build the adapter directly from an inner candidate config.

    The pipeline config-loader normally instantiates the inner Phase-0
    candidate and wraps it via `SentenceTranslator(inner=...)` itself; this
    `build()` exists so the adapter is round-trippable through the standard
    candidate-loader pattern.
    """
    module = importlib.import_module(config["module"])
    inner = module.build(config)
    if isinstance(inner, SentenceTranslator):
        return inner
    return SentenceTranslator(inner=inner)
