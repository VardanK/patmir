"""TTS stage adapter.

Wraps a Phase 0 TTS candidate (`synthesize(text) -> bytes`) with the typed
`TTSStage` protocol. Writes per-sentence WAV files to `<job_dir>/tts/<idx>.wav`
and a `synthesized.json` index for resume + UI.

Reads each WAV's sample rate and duration from the file header rather than
decoding the audio — Piper emits PCM WAV that wave.open can parse directly.
"""
from __future__ import annotations

import importlib
import io
import json
import wave
from pathlib import Path
from typing import Any

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import (
    SynthesizedAudio,
    SynthesizedSentence,
    TranslatedTranscript,
)


class SentenceTTS:
    id = "tts_adapter"

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def synthesize(self, translated: TranslatedTranscript, ctx: JobContext) -> SynthesizedAudio:
        out_dir = ctx.job_dir / "tts"
        out_dir.mkdir(exist_ok=True)
        total = len(translated.sentences) or 1
        sentences: list[SynthesizedSentence] = []
        sample_rate = 0

        for i, s in enumerate(translated.sentences):
            if not s.target_text.strip():
                continue
            wav_bytes = self.inner.synthesize(s.target_text)
            wav_path = out_dir / f"{s.idx:04d}.wav"
            wav_path.write_bytes(wav_bytes)
            sr, duration = _wav_header_meta(wav_bytes)
            sample_rate = sample_rate or sr
            sentences.append(SynthesizedSentence(
                idx=s.idx,
                wav_path=wav_path,
                wav_duration=duration,
                source_start=s.start,
                source_end=s.end,
                target_text=s.target_text,
            ))
            ctx.progress("tts", (i + 1) / total, f"sentence {i + 1}/{total}")

        out = SynthesizedAudio(sentences=sentences, sample_rate=sample_rate)
        _write_index(ctx.job_dir, out)
        return out


def _wav_header_meta(wav_bytes: bytes) -> tuple[int, float]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        return sr, n / sr


def _write_index(job_dir: Path, synth: SynthesizedAudio) -> None:
    payload = {
        "sample_rate": synth.sample_rate,
        "sentences": [
            {
                "idx": s.idx,
                "wav_path": str(s.wav_path),
                "wav_duration": s.wav_duration,
                "source_start": s.source_start,
                "source_end": s.source_end,
                "target_text": s.target_text,
            }
            for s in synth.sentences
        ],
    }
    (job_dir / "synthesized.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def build(config: dict[str, Any]) -> SentenceTTS:
    module = importlib.import_module(config["module"])
    inner = module.build(config)
    if isinstance(inner, SentenceTTS):
        return inner
    return SentenceTTS(inner=inner)
