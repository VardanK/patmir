"""Qwen3-ASR via MLX (native Apple Silicon GPU).

`mlx-qwen3-asr` (PyPI: `mlx-qwen3-asr`, ~v0.3.x) — a ground-up MLX
reimplementation of the official PyTorch backend. Runs on the Apple Silicon
GPU + unified memory pool. Bundles a custom MLX `ForcedAligner` that produces
real word-level timestamps on Mac (the official CUDA-bound aligner does not).

This is the right Mac path for Qwen3-ASR. CUDA workstations should use the
sibling `qwen3_asr` module, which drives the same models through transformers'
native `qwen3_asr` support (5.13+). Note the aligner repo differs between the
two: this module wants the original `Qwen/Qwen3-ForcedAligner-0.6B` layout,
the transformers one wants the `-hf` conversion.

Models loaded from `mlx-community/Qwen3-ASR-{0.6B,1.7B}-{bf16,4bit,5bit,6bit,8bit}`.

Install once: `uv pip install mlx-qwen3-asr`
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from typing import Any

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import AudioTrack, Sentence, Transcript, Word
from patmir.pipeline.stages.asr._sentence import (
    filter_words_by_vad,
    group_into_sentences,
    word_in_target_script,
    write_transcript,
)


# Language-tag normalization is shared with the other Qwen3-ASR candidate
# (`_language.py`) so the two runtimes cannot disagree about what a tag
# means. `_to_iso` returns None for anything it does not recognise —
# the callers rely on that to fall back to a full probe.
from patmir.pipeline.stages.asr._language import FULL_TO_ISO as _FULL_TO_ISO, to_iso as _to_iso


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
        sentence_gap_s: float = 0.35,
        max_sentence_s: float = 8.0,
    ) -> None:
        from mlx_qwen3_asr import ForcedAligner, load_model  # lazy

        self.dtype_name = dtype
        self.max_new_tokens = max_new_tokens
        self.sentence_gap_s = sentence_gap_s
        self.max_sentence_s = max_sentence_s
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
        # Build sentences from Qwen3-ASR chunks first, then fall back to
        # gap-based grouping for any aligned words that weren't covered by
        # an accepted chunk. Qwen3-ASR occasionally tags chunks with the
        # wrong language (e.g. "und"/"en" on Russian interjections like
        # "Ням ням ням"), and those chunks get filtered out — but the
        # forced aligner still emits the words, so we'd lose real dialogue.
        sentences: list[Sentence] = []
        idx = 0
        covered_word_ids: set[int] = set()
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
            for w in matching:
                covered_word_ids.add(id(w))
            if matching:
                # A chunk is not a sentence. Qwen3-ASR chunks span tens of
                # seconds, and emitting one Sentence per chunk (which this did
                # until 2026-08-24) handed the whole span a single DOMINANT
                # speaker in `apply_speaker_tags` — so a chunk covering several
                # diarization turns was synthesized in ONE character's voice,
                # and the TTS decoded the whole span as one unit. Group on
                # punctuation / word gaps like the other candidates; the gap
                # branch is what carries this one, since Qwen3-ASR emits no
                # terminal punctuation. Narrowing to the aligned word range
                # (what s_start/s_end did) falls out of grouping for free.
                for sent in group_into_sentences(
                    matching,
                    max_gap_s=self.sentence_gap_s,
                    max_sentence_s=self.max_sentence_s,
                ):
                    sentences.append(replace(sent, idx=idx))
                    idx += 1
            else:
                # No aligned words in this chunk's window: keep the model's own
                # text at chunk-level timings rather than dropping the line.
                sentences.append(Sentence(
                    idx=idx,
                    text=txt,
                    start=cstart,
                    end=cend,
                    words=[],
                ))
                idx += 1

        # Orphan-word recovery: any aligned word that wasn't covered by an
        # accepted chunk gets folded into a synthetic sentence, grouped by
        # gap so continuous speech becomes one utterance. We additionally
        # require the word to contain at least one character in the target
        # language's script — Qwen3-ASR routinely hallucinates CJK
        # characters for cartoon sound effects ("嗯", "滋 巴"), and those
        # would translate to nonsense if let through.
        orphan_words = [
            w for w in words
            if id(w) not in covered_word_ids and word_in_target_script(w.text, language)
        ]
        if orphan_words:
            orphan_words.sort(key=lambda w: w.start)
            gap_threshold = 1.5  # seconds — split groups at >1.5s of silence
            group: list[Word] = []
            groups: list[list[Word]] = []
            for w in orphan_words:
                if group and (w.start - group[-1].end) > gap_threshold:
                    groups.append(group)
                    group = []
                group.append(w)
            if group:
                groups.append(group)
            for g in groups:
                text = " ".join(w.text for w in g).strip()
                if not text:
                    continue
                sentences.append(Sentence(
                    idx=idx,
                    text=text,
                    start=min(w.start for w in g),
                    end=max(w.end for w in g),
                    words=g,
                ))
                idx += 1
            sentences.sort(key=lambda s: s.start)
            for i, s in enumerate(sentences):
                s.idx = i

        transcript = Transcript(language=language, words=words, sentences=sentences)
        write_transcript(ctx.job_dir, transcript)
        return transcript


def build(config: dict[str, Any]) -> Qwen3ASRMLX:
    return Qwen3ASRMLX(
        model_id=config["model_id"],
        dtype=config.get("dtype", "bfloat16"),
        forced_aligner=config.get("forced_aligner", "Qwen/Qwen3-ForcedAligner-0.6B"),
        aligner_dtype=config.get("aligner_dtype", "bfloat16"),
        max_new_tokens=config.get("max_new_tokens"),
        sentence_gap_s=float(config.get("sentence_gap_s", 0.35)),
        max_sentence_s=float(config.get("max_sentence_s", 8.0)),
    )
