"""Forced-alignment helper for cloud ASR candidates.

Cloud STT routes (OpenRouter chat-completions multimodal audio, ElevenLabs
Scribe pre-API, etc.) don't return word-level timestamps. Patmir's
downstream stages need them — apply_speaker_tags depends on overlap
between word windows and diarization turns; mux's stretch-cap math
depends on per-sentence start/end. This module fills that gap.

Backed by Meta's MMS-FA (Multilingual Massively-Speech Forced Aligner) via
the `ctc-forced-aligner` package. Single shared ONNX session per process
(lazy-loaded on first call) — model is ~1.2 GB and worth caching across
all chunks in a job.

Russian was the trigger language (Patmir's primary source). MMS-FA
supports 1000+ languages by uroman-romanizing the script before
alignment, so this works for any source language we care about.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

from patmir.pipeline.stage import Word

# ISO 639-1 → MMS-FA 3-letter language code (best-effort).
# Map is conservative — when unsure, MMS-FA falls back to its general
# multilingual pipeline since `language` is only used by `uroman`.
_LANG_MAP = {
    "ru": "rus",
    "en": "eng",
    "uk": "ukr",
    "be": "bel",
    "hy": "hye",
    "ka": "kat",
    "tr": "tur",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "pl": "pol",
    "ja": "jpn",
    "zh": "chi",
    "ko": "kor",
    "ar": "ara",
}

_lock = threading.Lock()
_aligner = None  # cached Alignment instance


def _get_aligner():
    global _aligner
    with _lock:
        if _aligner is not None:
            return _aligner
        from ctc_forced_aligner import Alignment, MODEL_URL, ensure_onnx_model
        cache_dir = os.path.expanduser("~/.cache/ctc_forced_aligner")
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, "mms_fa.onnx")
        ensure_onnx_model(model_path, MODEL_URL)
        _aligner = Alignment(model_path)
        return _aligner


def align_words(audio_path: Path, text: str, language: str) -> list[Word]:
    """Return word-level timings for `text` against `audio_path`.

    `language` is an ISO 639-1 code (e.g. "ru"); unrecognized codes are
    passed through as-is and MMS-FA tries its best.

    Returns [] if anything fails — caller should fall back gracefully.
    """
    if not text.strip():
        return []

    try:
        from ctc_forced_aligner import (
            generate_emissions,
            get_alignments,
            get_spans,
            load_audio,
            postprocess_results,
            preprocess_text,
        )
    except ImportError:
        return []

    aligner = _get_aligner()
    lang_code = _LANG_MAP.get(language, language[:3] if language else "eng")

    try:
        waveform = load_audio(str(audio_path))
        emissions, stride = generate_emissions(aligner.model, waveform, batch_size=4)
        tokens_starred, text_starred = preprocess_text(
            text.strip(), romanize=True, language=lang_code,
        )
        segments, scores, blank_token = get_alignments(
            emissions, tokens_starred, aligner.tokenizer,
        )
        spans = get_spans(tokens_starred, segments, blank_token)
        word_ts = postprocess_results(text_starred, spans, stride, scores)
    except Exception:
        return []

    return [
        Word(
            text=w["text"],
            start=float(w["start"]),
            end=float(w["end"]),
        )
        for w in word_ts
        if w.get("text") and w["text"] != "<star>"
    ]
