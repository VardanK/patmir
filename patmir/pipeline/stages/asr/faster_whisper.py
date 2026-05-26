"""ASR stage — faster-whisper (large-v3-turbo by default).

Word-level timestamps + auto language detection. Groups words into sentences
on terminal punctuation (.!?) or gaps > max_gap_s.

Note on `device`: faster-whisper's CTranslate2 backend supports `cpu` and
`cuda` natively. MPS is **not** supported by CT2 today — on Apple Silicon we
run on `cpu` with `int8`, which is still fast for the model size. We resolve
`device = "auto"` accordingly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import AudioTrack, Sentence, Transcript, Word
from patmir.pipeline.stages.asr._sentence import (
    filter_words_by_vad,
    group_into_sentences,
    write_transcript,
)


def _resolve_device(requested: str) -> str:
    if requested in ("cpu", "cuda"):
        return requested
    # "auto" / "mps" / anything else — CT2 does not support MPS; fall back to CPU.
    try:
        import torch  # type: ignore[import-not-found]
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class FasterWhisperASR:
    id = "faster_whisper"

    def __init__(self, model_id: str, device: str, compute_type: str, vad_filter: bool = False) -> None:
        from faster_whisper import WhisperModel  # lazy import

        self.device = _resolve_device(device)
        self.compute_type = compute_type
        self.vad_filter = vad_filter
        self.model = WhisperModel(model_id, device=self.device, compute_type=compute_type)

    def transcribe(self, audio: AudioTrack, ctx: JobContext) -> Transcript:
        segments_iter, info = self.model.transcribe(
            str(audio.path),
            word_timestamps=True,
            language=None,
            vad_filter=self.vad_filter,
        )
        ctx.source_lang = info.language
        ctx.progress("asr", 0.05, f"detected language={info.language}")

        words: list[Word] = []
        for seg in segments_iter:
            if not seg.words:
                continue
            for w in seg.words:
                words.append(Word(text=w.word.strip(), start=float(w.start), end=float(w.end)))
            if audio.duration > 0:
                ctx.progress("asr", min(0.95, seg.end / audio.duration), f"t={seg.end:.1f}s")

        words = filter_words_by_vad(words, ctx.extra.get("vad_intervals"))
        sentences = group_into_sentences(words)
        transcript = Transcript(language=info.language, words=words, sentences=sentences)
        write_transcript(ctx.job_dir, transcript)
        return transcript


def build(config: dict[str, Any]) -> FasterWhisperASR:
    return FasterWhisperASR(
        model_id=config["model_id"],
        device=config.get("device", "auto"),
        compute_type=config.get("compute_type", "int8"),
        vad_filter=config.get("vad_filter", False),
    )
