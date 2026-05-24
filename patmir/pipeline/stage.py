"""Typed stage protocols and inter-stage data contracts.

One `Protocol` per stage; one dataclass per stage's output. The runner only
talks to stages through these — concrete stages live under `pipeline.stages.*`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from patmir.pipeline.context import JobContext


# ─── inter-stage data contracts ───────────────────────────────────────────────

@dataclass
class AudioTrack:
    path: Path
    sample_rate: int
    duration: float  # seconds


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Sentence:
    idx: int
    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    language: str  # ISO 639-1, e.g. "en", "ru"
    words: list[Word]
    sentences: list[Sentence]


@dataclass
class TranslatedSentence:
    idx: int
    source_text: str
    target_text: str
    start: float
    end: float


@dataclass
class TranslatedTranscript:
    source_lang: str
    target_lang: str  # always "hy" in Phase 1
    sentences: list[TranslatedSentence]


@dataclass
class SynthesizedSentence:
    idx: int
    wav_path: Path
    wav_duration: float
    source_start: float
    source_end: float
    target_text: str


@dataclass
class SynthesizedAudio:
    sentences: list[SynthesizedSentence]
    sample_rate: int


@dataclass
class MuxedAudio:
    path: Path  # single concatenated AR track
    sample_rate: int


@dataclass
class OutputArtifacts:
    video: Path
    subtitles: Path | None


# ─── stage protocols ──────────────────────────────────────────────────────────

@runtime_checkable
class IngestStage(Protocol):
    def extract(self, video_path: Path, ctx: "JobContext") -> AudioTrack: ...


@runtime_checkable
class ASRStage(Protocol):
    def transcribe(self, audio: AudioTrack, ctx: "JobContext") -> Transcript: ...


@runtime_checkable
class TranslateStage(Protocol):
    def translate(self, transcript: Transcript, ctx: "JobContext") -> TranslatedTranscript: ...


@runtime_checkable
class TTSStage(Protocol):
    def synthesize(self, translated: TranslatedTranscript, ctx: "JobContext") -> SynthesizedAudio: ...


@runtime_checkable
class MuxStage(Protocol):
    def mux(self, synth: SynthesizedAudio, source: AudioTrack, ctx: "JobContext") -> MuxedAudio: ...


@runtime_checkable
class OutputStage(Protocol):
    def write(
        self,
        muxed: MuxedAudio,
        source_video: Path,
        translated: TranslatedTranscript,
        ctx: "JobContext",
    ) -> OutputArtifacts: ...


STAGE_ORDER: tuple[str, ...] = (
    "ingest",
    "asr",
    "translate",
    "tts",
    "mux",
    "output",
)
