"""Typed stage protocols and inter-stage data contracts.

One `Protocol` per stage; one dataclass per stage's output. The runner only
talks to stages through these — concrete stages live under `pipeline.stages.*`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from patmir.pipeline.context import JobContext


# ─── emotion ─────────────────────────────────────────────────────────────────

Emotion = Literal["neutral", "happy", "sad", "angry", "fearful"]
"""Coarse emotion bins used for per-(speaker, emotion) ref-clip selection and
per-emotion TTS guidance. The full 9-class output from emotion2vec collapses
to these 5 bins via the [emotion_bins] table in patmir.toml.
"""


# ─── inter-stage data contracts ───────────────────────────────────────────────

@dataclass
class AudioTrack:
    path: Path
    sample_rate: int
    duration: float  # seconds


@dataclass
class SeparatedAudio:
    """Result of vocal / accompaniment separation + speech filtering.

    `accompaniment` is the music + sound-effects bed from Demucs (drums +
    bass + other stems summed). `vocals` is the full Demucs vocal stem —
    contains spoken dialogue AND non-speech vocalizations (laughs, screams,
    grunts, etc.) — kept as the upstream artifact and the audio ASR sees.

    `speech_vocals` and `non_speech_vocals` are precomputed VAD-masked stems
    of `vocals` — `non_speech_vocals` is mixed into the bed alongside
    `accompaniment` so the cartoon's atmospheric vocal sounds survive into
    the final dub. `speech_vocals` is kept as an inspection artifact.

    `vad_intervals` is the list of [start, end] (seconds) speech windows
    from Silero VAD. ASR uses these as a *post-filter* on its word
    timestamps — words that don't overlap any interval get dropped — which
    keeps soft-onset speech (that the hard-mask would have clipped) while
    still discarding ASR hallucinations on pure scream/laugh regions.
    """
    accompaniment: Path
    vocals: Path
    sample_rate: int
    channels: int
    speech_vocals: Path | None = None
    non_speech_vocals: Path | None = None
    vad_intervals: list[tuple[float, float]] | None = None


@dataclass
class SpeakerTurn:
    """A single contiguous span attributed to one speaker by the diarizer.

    `emotion` is filled by the runner after the `emotion` stage runs — it's
    the dominant predicted emotion bin over this turn's window.
    """
    start: float
    end: float
    speaker_id: str
    emotion: Emotion | None = None


@dataclass
class EmotionPrediction:
    """Per-turn (or per-window) speech-emotion-recognition output.

    `emotion` is one of the 5 collapsed bins (post `[emotion_bins]` mapping).
    `confidence` is the SER model's softmax probability for the winning class
    *before* any confidence floor is applied — predictions below the
    `min_confidence` config threshold are reassigned to `"neutral"` at the
    candidate level (not here), so this dataclass always carries the
    post-floor effective label.
    """
    start: float
    end: float
    emotion: Emotion
    confidence: float


@dataclass
class OverlapInterval:
    """A time window pyannote flagged as overlapped speech (>1 speaker active)."""
    start: float
    end: float


@dataclass
class SpeakerRef:
    """Reference clip + transcript for cloning one speaker.

    `ref_text` is empty at diarize time and backfilled after ASR runs.
    """
    speaker_id: str
    ref_audio: Path
    ref_text: str
    source_start: float
    source_end: float


@dataclass
class Diarization:
    """Output of the diarize stage + the runner's post-emotion ref-picker.

    `refs` is keyed by `(speaker_id, emotion_bin)` in Phase 2.x. It's empty
    when the diarize stage returns — the runner populates it after the
    emotion stage runs, since picking the right ref per (speaker, emotion)
    requires knowing each turn's dominant emotion.

    JSON serialization keys these tuples as `"speaker__emotion"` strings.
    """
    turns: list[SpeakerTurn]
    overlaps: list[OverlapInterval]
    refs: dict[tuple[str, Emotion], SpeakerRef]
    sample_rate: int


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
    speaker_id: str | None = None
    emotion: Emotion | None = None


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
    speaker_id: str | None = None
    emotion: Emotion | None = None


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
class SeparateStage(Protocol):
    def separate(self, source_video: Path, ctx: "JobContext") -> SeparatedAudio: ...


@runtime_checkable
class DiarizeStage(Protocol):
    def diarize(self, separated: SeparatedAudio, ctx: "JobContext") -> Diarization: ...


@runtime_checkable
class EmotionStage(Protocol):
    def detect(
        self,
        separated: SeparatedAudio,
        diarization: Diarization,
        ctx: "JobContext",
    ) -> list[EmotionPrediction]: ...


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
    def mux(
        self,
        synth: SynthesizedAudio,
        source: AudioTrack,
        separated: SeparatedAudio,
        ctx: "JobContext",
    ) -> MuxedAudio: ...


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
    "separate",
    "diarize",
    "emotion",
    "asr",
    "translate",
    "tts",
    "mux",
    "output",
)
