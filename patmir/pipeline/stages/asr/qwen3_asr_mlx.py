"""Qwen3-ASR via MLX (native Apple Silicon GPU).

`mlx-qwen3-asr` (PyPI: `mlx-qwen3-asr`, ~v0.3.x) — a ground-up MLX
reimplementation of the official PyTorch backend. Runs on the Apple Silicon
GPU + unified memory pool. Bundles a custom MLX `ForcedAligner` that produces
real word-level timestamps on Mac (the official CUDA-bound aligner does not).

This is the right Mac path for Qwen3-ASR. CUDA workstations should use the
sibling `qwen3_asr` module via the `qwen-asr` package instead.

Models loaded from `mlx-community/Qwen3-ASR-{0.6B,1.7B}-{bf16,4bit,5bit,6bit,8bit}`.

Install once: `uv pip install mlx-qwen3-asr`
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import AudioTrack, Sentence, Transcript, Word
from patmir.pipeline.stages.asr._sentence import filter_words_by_vad


# Qwen3-ASR labels languages with full English names ("English", "Russian", …).
# Downstream pipeline stages expect ISO 639-1.
_FULL_TO_ISO: dict[str, str] = {
    "english": "en",
    "russian": "ru",
    "armenian": "hy",
}


def _to_iso(language: str | None) -> str | None:
    if not language:
        return None
    key = language.strip().lower()
    if key in _FULL_TO_ISO:
        return _FULL_TO_ISO[key]
    if len(key) == 2 and key.isalpha():
        return key
    return key.split()[0]


def _mx_dtype(name: str):
    import mlx.core as mx  # type: ignore[import-not-found]
    return {
        "bfloat16": mx.bfloat16,
        "float16": mx.float16,
        "float32": mx.float32,
    }.get(name, mx.bfloat16)


class Qwen3ASRMLX:
    id = "qwen3_asr_mlx"

    def __init__(
        self,
        model_id: str,
        dtype: str,
        forced_aligner: str | None,
        aligner_dtype: str,
        max_new_tokens: int | None,
    ) -> None:
        from mlx_qwen3_asr import ForcedAligner, load_model  # lazy

        self.dtype_name = dtype
        self.max_new_tokens = max_new_tokens
        self.model, _ = load_model(model_id, dtype=_mx_dtype(dtype))
        self.aligner = (
            ForcedAligner(forced_aligner, dtype=_mx_dtype(aligner_dtype))
            if forced_aligner
            else None
        )

    def transcribe(self, audio: AudioTrack, ctx: JobContext) -> Transcript:
        from mlx_qwen3_asr import transcribe as mlx_transcribe

        ctx.progress("asr", 0.05, f"mlx-qwen3-asr ({self.dtype_name})")
        kwargs: dict[str, Any] = dict(
            model=self.model,
            forced_aligner=self.aligner,
            return_timestamps=self.aligner is not None,
            return_chunks=True,
            dtype=_mx_dtype(self.dtype_name),
        )
        if self.max_new_tokens is not None:
            kwargs["max_new_tokens"] = self.max_new_tokens
        result = mlx_transcribe(str(audio.path), **kwargs)

        # Pick the dominant ISO language across chunks weighted by chunk
        # duration. The MLX transcriber labels each chunk independently and a
        # cartoon's intro music can get mis-labeled (e.g. "Chinese" for
        # "嗯。啊。" filler), so a per-first-chunk read is unreliable. Always
        # normalize through _to_iso, never pass a full-name language ("English")
        # downstream where NLLB will choke on it.
        chunks = list(result.chunks or [])
        weighted: dict[str, float] = {}
        for c in chunks:
            iso = _to_iso(c.get("language"))
            if not iso:
                continue
            span = max(0.0, float(c.get("end", 0.0)) - float(c.get("start", 0.0)))
            weighted[iso] = weighted.get(iso, 0.0) + span
        if weighted:
            language = max(weighted.items(), key=lambda kv: kv[1])[0]
        else:
            language = _to_iso(ctx.source_lang) or "en"
        ctx.source_lang = language

        # Word-level entries come back as `segments`; sentence-ish groupings as `chunks`.
        words: list[Word] = []
        for seg in (result.segments or []):
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            words.append(Word(
                text=text,
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", seg.get("start", 0.0))),
            ))
        # VAD post-filter: drop words that don't overlap any speech interval
        # from the `separate` stage. This catches ASR hallucinations on pure
        # scream/laugh regions without clipping soft-onset real speech.
        words = filter_words_by_vad(words, ctx.extra.get("vad_intervals"))

        # mlx-qwen3-asr emits sentence-shaped chunks with punctuation; use
        # them as sentence boundaries but narrow each one's [start, end] to
        # the *word range* of words that survived VAD filtering. Chunks with
        # no surviving words (e.g. a chunk that was entirely scream-region
        # hallucination) are dropped wholesale.
        sentences: list[Sentence] = []
        idx = 0
        for c in chunks:
            txt = (c.get("text") or "").strip()
            if not txt:
                continue
            chunk_lang = _to_iso(c.get("language"))
            if chunk_lang and chunk_lang != language:
                continue
            cstart = float(c.get("start", 0.0))
            cend = float(c.get("end", cstart))
            matching = [w for w in words if cstart <= w.start <= cend]
            if not matching:
                # Whole chunk got VAD-filtered out — skip.
                continue
            s_start = min(w.start for w in matching)
            s_end = max(w.end for w in matching)
            sentences.append(Sentence(
                idx=idx,
                text=txt,
                start=s_start,
                end=s_end,
                words=matching,
            ))
            idx += 1

        transcript = Transcript(language=language, words=words, sentences=sentences)
        _write_transcript(ctx.job_dir, transcript)
        return transcript


def _write_transcript(job_dir: Path, transcript: Transcript) -> None:
    payload = {
        "language": transcript.language,
        "words": [asdict(w) for w in transcript.words],
        "sentences": [
            {
                "idx": s.idx,
                "text": s.text,
                "start": s.start,
                "end": s.end,
                "words": [asdict(w) for w in s.words],
            }
            for s in transcript.sentences
        ],
    }
    (job_dir / "transcript.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    sentences_only = [
        {"idx": s.idx, "text": s.text, "start": s.start, "end": s.end}
        for s in transcript.sentences
    ]
    (job_dir / "sentences.json").write_text(json.dumps(sentences_only, indent=2, ensure_ascii=False))


def build(config: dict[str, Any]) -> Qwen3ASRMLX:
    return Qwen3ASRMLX(
        model_id=config["model_id"],
        dtype=config.get("dtype", "bfloat16"),
        forced_aligner=config.get("forced_aligner", "Qwen/Qwen3-ForcedAligner-0.6B"),
        aligner_dtype=config.get("aligner_dtype", "bfloat16"),
        max_new_tokens=config.get("max_new_tokens"),
    )
