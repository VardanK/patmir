"""Qwen3-ASR on PyTorch, via transformers' native `qwen3_asr` support.

`Qwen/Qwen3-ASR-{0.6B,1.7B}-hf` — Apache-2.0 multilingual ASR from Alibaba
(Jan 2026), 30 supported languages including EN and RU. On LibriSpeech "other"
(noisy) the 1.7B variant beats Whisper-large-v3 by ~15% relative WER.

**This candidate deliberately does NOT use the official `qwen-asr` package.**
That package pins `transformers==4.57.6` exactly, and the Higgs Audio v3 TTS
checkpoint needs `transformers>=5.5` — the two cannot coexist in one
environment. Transformers grew native `qwen3_asr` support in **5.13.0**
(`Qwen3ASRForConditionalGeneration` + `Qwen3ASRForTokenClassification` for the
forced aligner), so driving the models straight through transformers removes
the conflict entirely and drops four exact pins (`transformers`, `accelerate`,
`nagisa`, `soynlp`) plus `vllm`, `gradio`, and `flask` from the dependency set.

Requires **transformers >= 5.13**. On an older transformers the checkpoint's
`model_type: qwen3_asr` is unknown and `from_pretrained` fails with
`KeyError: 'qwen3_asr'`; `__init__` raises with that instruction instead.

Note the `-hf` checkpoint suffix: `Qwen/Qwen3-ASR-1.7B-hf` is the
transformers-native conversion. The unsuffixed `Qwen/Qwen3-ASR-1.7B` is the
original layout the `qwen-asr` package reads and will NOT load here.

## Long-form audio

Upstream's package hid long-form segmentation inside `transcribe()`; the
transformers API is single-shot, so segmentation lives here. The encoder does
handle arbitrary length (windowed, ~13 audio tokens/second), but decoding a
whole 5-minute transcript in one autoregressive pass invites repetition loops
and needs a huge `max_new_tokens`. Instead we cut the audio into chunks at
Silero VAD silences (`ctx.extra["vad_intervals"]`, set by the separate stage),
capped at `chunk_seconds`, and run them **batched** — the ASR and the aligner
both support batch inference natively.

Cutting on VAD boundaries means no decode budget is spent on silence.

It does **not** by itself give finer sentences — that claim was in this docstring
until 2026-08-24 and was wrong. A chunk is up to `chunk_seconds` of audio, and
until then each chunk became one `Sentence`, so a 5-minute clip yielded 23
sentences with spans up to 26.3 s. Sentences now come from
`group_into_sentences` over the aligned words (`sentence_gap_s`), like every
other ASR candidate. See `stages/asr/qwen3-asr-transformers.md`.

## Word timestamps

`Qwen/Qwen3-ForcedAligner-0.6B-hf` is a single forward pass (no AR decode) with
a classification head predicting each word's length. It covers a **subset** of
the ASR languages (`_ALIGNER_LANGS`); outside that set, or if an optional
tokenizer dependency is missing, we degrade to chunk-level sentence timings
rather than failing the stage.

Install: `uv pip install "transformers>=5.13" torch torchaudio soundfile`
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import AudioTrack, Sentence, Transcript, Word
from patmir.pipeline.stages.asr._sentence import (
    filter_words_by_vad,
    group_into_sentences,
    is_non_lexical,
    word_in_target_script,
    write_transcript,
)

SAMPLE_RATE = 16_000

# Minimum transformers release carrying `models/qwen3_asr` (added 5.13.0).
MIN_TRANSFORMERS = (5, 13)

# Qwen3-ASR reports the language as a full English name ("English", "Russian").
# Downstream pipeline stages expect ISO 639-1.
_FULL_TO_ISO: dict[str, str] = {
    "english": "en",
    "russian": "ru",
    "armenian": "hy",
}

# The forced aligner covers a subset of the ASR languages (ISO codes for the
# set documented on the model card). Japanese additionally needs `nagisa` and
# Korean `soynlp`; a missing one surfaces as ImportError and is caught.
_ALIGNER_LANGS = frozenset({"zh", "en", "yue", "fr", "de", "it", "ja", "ko", "pt", "ru", "es"})


def _to_iso(language: str | None) -> str | None:
    if not language:
        return None
    key = language.strip().lower()
    if key in _FULL_TO_ISO:
        return _FULL_TO_ISO[key]
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
    except Exception:
        pass
    return "cpu"


def _dtype(name: str):
    import torch  # type: ignore[import-not-found]
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(name, torch.bfloat16)


def _check_transformers() -> None:
    """Fail with the fix rather than `KeyError: 'qwen3_asr'`."""
    import transformers  # type: ignore[import-not-found]

    version = getattr(transformers, "__version__", "0")
    parts: list[int] = []
    for piece in version.split(".")[:2]:
        digits = "".join(c for c in piece if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 2:
        parts.append(0)

    if tuple(parts) < MIN_TRANSFORMERS:
        want = ".".join(str(p) for p in MIN_TRANSFORMERS)
        raise RuntimeError(
            f"Qwen3-ASR needs transformers >= {want} for native `qwen3_asr` support; "
            f"found {version}. Older releases fail with KeyError: 'qwen3_asr'. "
            f"Fix: uv pip install --upgrade 'transformers>={want}'. "
            "NOTE: do NOT install the `qwen-asr` package — it pins "
            "transformers==4.57.6 and would break the Higgs TTS stage."
        )


def build_chunks(
    duration: float,
    vad_intervals: list[tuple[float, float]] | None,
    max_seconds: float,
    pad_seconds: float = 0.2,
) -> list[tuple[float, float]]:
    """Cut `[0, duration]` into ASR windows no longer than `max_seconds`.

    With VAD intervals the cuts land in silence, so no chunk boundary splits a
    word and no decode budget is spent on silence. Without them, fixed windows.
    A single speech interval longer than `max_seconds` is hard-split.
    """
    if max_seconds <= 0:
        raise ValueError("max_seconds must be > 0")

    if not vad_intervals:
        chunks: list[tuple[float, float]] = []
        start = 0.0
        while start < duration:
            chunks.append((start, min(duration, start + max_seconds)))
            start += max_seconds
        return chunks

    # Split over-long intervals first, so the greedy pass below only ever has
    # to decide where to *group*, never where to cut inside speech.
    pieces: list[tuple[float, float]] = []
    for raw_start, raw_end in sorted(vad_intervals):
        start, end = max(0.0, float(raw_start)), min(duration, float(raw_end))
        if end <= start:
            continue
        while end - start > max_seconds:
            pieces.append((start, start + max_seconds))
            start += max_seconds
        pieces.append((start, end))

    grouped: list[tuple[float, float]] = []
    for start, end in pieces:
        if grouped and end - grouped[-1][0] <= max_seconds:
            grouped[-1] = (grouped[-1][0], end)
        else:
            grouped.append((start, end))

    # Pad outward for word onsets, but never past a neighbour's midpoint.
    padded: list[tuple[float, float]] = []
    for i, (start, end) in enumerate(grouped):
        lo = 0.0 if i == 0 else (grouped[i - 1][1] + start) / 2.0
        hi = duration if i == len(grouped) - 1 else (end + grouped[i + 1][0]) / 2.0
        padded.append((max(lo, start - pad_seconds), min(hi, end + pad_seconds)))
    return padded


class Qwen3ASRTorch:
    id = "qwen3_asr"

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        dtype: str = "bfloat16",
        forced_aligner: str | None = "Qwen/Qwen3-ForcedAligner-0.6B-hf",
        aligner_dtype: str = "bfloat16",
        max_new_tokens: int = 256,
        chunk_seconds: float = 28.0,
        sentence_gap_s: float = 0.35,
        max_sentence_s: float = 8.0,
        batch_size: int = 8,
        attn_implementation: str = "sdpa",
    ) -> None:
        _check_transformers()
        from transformers import (  # type: ignore[import-not-found]
            AutoProcessor,
            Qwen3ASRForConditionalGeneration,
            Qwen3ASRForTokenClassification,
        )

        self.device = _resolve_device(device)
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.chunk_seconds = chunk_seconds
        self.sentence_gap_s = sentence_gap_s
        self.max_sentence_s = max_sentence_s
        self.batch_size = max(1, batch_size)

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(
            model_id,
            dtype=_dtype(dtype),
            attn_implementation=attn_implementation,
        ).to(self.device).eval()

        self.aligner = None
        self.aligner_processor = None
        if forced_aligner:
            self.aligner_processor = AutoProcessor.from_pretrained(forced_aligner)
            self.aligner = Qwen3ASRForTokenClassification.from_pretrained(
                forced_aligner,
                dtype=_dtype(aligner_dtype),
                attn_implementation=attn_implementation,
            ).to(self.device).eval()

    # ---- audio -----------------------------------------------------------

    def _load_audio(self, audio: AudioTrack) -> np.ndarray:
        import soundfile as sf  # type: ignore[import-not-found]

        samples, sr = sf.read(str(audio.path), dtype="float32", always_2d=False)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if sr != SAMPLE_RATE:
            import torch  # type: ignore[import-not-found]
            import torchaudio  # type: ignore[import-not-found]
            samples = torchaudio.functional.resample(
                torch.from_numpy(samples), sr, SAMPLE_RATE
            ).numpy()
        return np.ascontiguousarray(samples, dtype=np.float32)

    # ---- ASR -------------------------------------------------------------

    def _transcribe_batch(
        self, clips: list[np.ndarray], language: str | None
    ) -> list[dict[str, Any]]:
        import torch  # type: ignore[import-not-found]

        inputs = self.processor.apply_transcription_request(
            audio=clips,
            language=[language] * len(clips) if language else None,
        ).to(self.model.device, self.model.dtype)

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        parsed = self.processor.decode(generated, return_format="parsed")
        return list(parsed)

    def _align_batch(
        self, clips: list[np.ndarray], transcripts: list[str], language: str
    ) -> list[list[dict[str, Any]]]:
        """Word timestamps per clip, relative to that clip's start.

        Returns `[]` for a clip whose alignment produced nothing. Any failure
        degrades the whole batch to empty rather than failing the stage —
        chunk-level timings still place the line, just less precisely.
        """
        import torch  # type: ignore[import-not-found]

        assert self.aligner is not None and self.aligner_processor is not None
        aligner_inputs, word_lists = self.aligner_processor.prepare_forced_aligner_inputs(
            audio=clips, transcript=transcripts, language=language,
        )
        aligner_inputs = aligner_inputs.to(self.aligner.device, self.aligner.dtype)

        with torch.inference_mode():
            outputs = self.aligner(**aligner_inputs)

        return list(self.aligner_processor.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=aligner_inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=self.aligner.config.timestamp_token_id,
        ))

    def transcribe(self, audio: AudioTrack, ctx: JobContext) -> Transcript:
        ctx.progress("asr", 0.02, f"qwen3-asr ({self.model_id}) on {self.device}")
        samples = self._load_audio(audio)

        vad: list[tuple[float, float]] | None = ctx.extra.get("vad_intervals")
        chunks = build_chunks(audio.duration, vad, self.chunk_seconds)
        ctx.probe.bump("n_chunks", len(chunks))
        if not chunks:
            return Transcript(language=ctx.source_lang or "en", words=[], sentences=[])

        clips = [
            samples[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
            for start, end in chunks
        ]
        ctx.progress("asr", 0.05, f"{len(chunks)} chunks (VAD-snapped)" if vad else f"{len(chunks)} chunks")

        # Language: honour an explicit source language, else auto-detect on the
        # longest chunk (most speech = most reliable) and then FORCE it for
        # every chunk. Per-chunk auto-detect mislabels short interjections,
        # which is a documented source of foreign-script hallucination.
        language = ctx.source_lang
        if not language:
            longest = max(range(len(clips)), key=lambda i: len(clips[i]))
            with ctx.probe.section("probe_s"):
                probe = self._transcribe_batch([clips[longest]], None)
            language = _to_iso(probe[0].get("language") if probe else None) or "en"
            ctx.progress("asr", 0.08, f"detected {language}")
        ctx.source_lang = language

        results: list[dict[str, Any]] = []
        for offset in range(0, len(clips), self.batch_size):
            batch = clips[offset:offset + self.batch_size]
            with ctx.probe.section("generate_s"):
                results.extend(self._transcribe_batch(batch, language))
            ctx.probe.bump("n_generate_batches")
            done = min(len(clips), offset + self.batch_size)
            ctx.progress("asr", 0.08 + 0.62 * done / len(clips), f"decoded {done}/{len(clips)}")

        texts = [str(r.get("transcription") or "").strip() for r in results]

        # Align only the chunks that produced text; keep the mapping back.
        timings: dict[int, list[dict[str, Any]]] = {}
        can_align = self.aligner is not None and language in _ALIGNER_LANGS
        if self.aligner is not None and not can_align:
            ctx.progress("asr", 0.72, f"aligner has no {language} — chunk-level timings")
        if can_align:
            indices = [i for i, t in enumerate(texts) if t]
            for offset in range(0, len(indices), self.batch_size):
                group = indices[offset:offset + self.batch_size]
                try:
                    with ctx.probe.section("align_s"):
                        aligned = self._align_batch(
                            [clips[i] for i in group], [texts[i] for i in group], language,
                        )
                    ctx.probe.bump("n_align_batches")
                except ImportError as exc:
                    # ja/ko word splitting needs an optional tokenizer package.
                    ctx.progress("asr", 0.8, f"aligner unavailable ({exc}) — chunk-level timings")
                    break
                for i, entries in zip(group, aligned):
                    timings[i] = entries
                done = min(len(indices), offset + self.batch_size)
                ctx.progress("asr", 0.7 + 0.25 * done / max(1, len(indices)), f"aligned {done}/{len(indices)}")

        words: list[Word] = []
        sentences: list[Sentence] = []
        for i, ((chunk_start, chunk_end), text) in enumerate(zip(chunks, texts)):
            if not text or is_non_lexical(text):
                continue

            chunk_words: list[Word] = []
            for entry in timings.get(i, []):
                word_text = str(entry.get("text") or "").strip()
                if not word_text or not word_in_target_script(word_text, language):
                    continue
                chunk_words.append(Word(
                    text=word_text,
                    start=chunk_start + float(entry.get("start_time", 0.0)),
                    end=chunk_start + float(entry.get("end_time", 0.0)),
                ))
            chunk_words = filter_words_by_vad(chunk_words, vad)

            words.extend(chunk_words)

            if chunk_words:
                # A chunk is up to `chunk_seconds` of audio, so it is NOT a
                # sentence. Emitting one Sentence per chunk (which this did
                # until 2026-08-24) broke everything downstream that assumes a
                # sentence is one utterance by one speaker: `apply_speaker_tags`
                # gave the whole span a single DOMINANT speaker, so a chunk
                # covering several diarization turns was synthesized in one
                # character's voice, and the TTS decoded 13-26 s of speech as one
                # unit. Measured on a 5-minute clip: 23 sentences against
                # diarization's 108 turns, spans up to 26.3 s.
                #
                # Group like every other ASR candidate does. The helper splits on
                # terminal punctuation OR a word gap — and the gap branch is what
                # carries this candidate, because Qwen3-ASR emits no punctuation
                # at all (0 of 398 words on that clip), so punctuation alone would
                # never split anything.
                #
                # Sentence start/end come from the grouped words, which also
                # narrows them to where the speaker actually talks: the chunk
                # carries VAD padding, and the mux places TTS at these times.
                for sent in group_into_sentences(
                    chunk_words,
                    max_gap_s=self.sentence_gap_s,
                    max_sentence_s=self.max_sentence_s,
                ):
                    sentences.append(replace(sent, idx=len(sentences)))
            else:
                # No aligner coverage for this language (or the aligner is
                # disabled): there are no word timings to group on, so the chunk
                # stays one sentence with the model's own text and chunk-level
                # timings. This is the documented degradation path, not the
                # normal one.
                sentences.append(Sentence(
                    idx=len(sentences),
                    text=text,
                    start=chunk_start,
                    end=chunk_end,
                    words=[],
                ))

        ctx.progress("asr", 1.0, f"{len(sentences)} sentences, {len(words)} words")
        transcript = Transcript(language=language, words=words, sentences=sentences)
        write_transcript(ctx.job_dir, transcript)
        return transcript


def build(config: dict[str, Any]) -> Qwen3ASRTorch:
    return Qwen3ASRTorch(
        model_id=config["model_id"],
        device=config.get("device", "auto"),
        dtype=config.get("dtype", "bfloat16"),
        forced_aligner=config.get("forced_aligner", "Qwen/Qwen3-ForcedAligner-0.6B-hf"),
        aligner_dtype=config.get("aligner_dtype", "bfloat16"),
        max_new_tokens=int(config.get("max_new_tokens", 256)),
        chunk_seconds=float(config.get("chunk_seconds", 28.0)),
        sentence_gap_s=float(config.get("sentence_gap_s", 0.35)),
        max_sentence_s=float(config.get("max_sentence_s", 8.0)),
        batch_size=int(config.get("batch_size", 8)),
        attn_implementation=config.get("attn_implementation", "sdpa"),
    )
