"""ASR stage — faster-whisper (large-v3-turbo by default).

Word-level timestamps + auto language detection. Groups words into sentences
on terminal punctuation (.!?) or gaps > max_gap_s.

Note on `device`: faster-whisper's CTranslate2 backend supports `cpu` and
`cuda` natively. MPS is **not** supported by CT2 today — on Apple Silicon we
run on `cpu` with `int8`, which is still fast for the model size. We resolve
`device = "auto"` accordingly.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import AudioTrack, Sentence, Transcript, Word


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

    def __init__(self, model_id: str, device: str, compute_type: str) -> None:
        from faster_whisper import WhisperModel  # lazy import

        self.device = _resolve_device(device)
        self.compute_type = compute_type
        self.model = WhisperModel(model_id, device=self.device, compute_type=compute_type)

    def transcribe(self, audio: AudioTrack, ctx: JobContext) -> Transcript:
        segments_iter, info = self.model.transcribe(
            str(audio.path),
            word_timestamps=True,
            language=None,
            vad_filter=True,
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

        sentences = _group_into_sentences(words)
        transcript = Transcript(language=info.language, words=words, sentences=sentences)
        _write_transcript(ctx.job_dir, transcript)
        return transcript


def _group_into_sentences(words: list[Word], max_gap_s: float = 0.6) -> list[Sentence]:
    sentences: list[Sentence] = []
    bucket: list[Word] = []
    idx = 0

    def flush() -> None:
        nonlocal idx, bucket
        if not bucket:
            return
        text = " ".join(w.text for w in bucket).strip()
        sentences.append(Sentence(
            idx=idx,
            text=text,
            start=bucket[0].start,
            end=bucket[-1].end,
            words=list(bucket),
        ))
        idx += 1
        bucket = []

    for i, w in enumerate(words):
        bucket.append(w)
        ends_sentence = w.text.endswith((".", "!", "?", "。", "！", "？"))
        gap_next = words[i + 1].start - w.end if i + 1 < len(words) else 0.0
        if ends_sentence or gap_next > max_gap_s:
            flush()
    flush()
    return sentences


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


def build(config: dict[str, Any]) -> FasterWhisperASR:
    return FasterWhisperASR(
        model_id=config["model_id"],
        device=config.get("device", "auto"),
        compute_type=config.get("compute_type", "int8"),
    )
