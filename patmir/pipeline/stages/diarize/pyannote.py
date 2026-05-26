"""Speaker diarization via pyannote-audio 3.1.

Runs `pyannote/speaker-diarization-3.1` on the Demucs vocal stem (full
`vocals.wav`, not the VAD-masked stem — same reason ASR consumes the full
stem: pyannote needs soft onsets to place turn boundaries accurately).

Post-filters the pyannote turns against the Silero VAD intervals that the
separate stage already produced: any turn that has zero intersection with
any VAD interval is dropped (catches ghost-speakers pyannote hallucinates
in scream/laugh regions).

Overlap regions come straight from `Annotation.get_overlap()` — pyannote
detects overlapped speech as part of segmentation.

Reference clip slicing is **not** done here (Phase 2.x change). The
`emotion` stage runs between `diarize` and `asr`, and ref clips need the
per-turn emotion bin to be picked correctly (one ref per `(speaker,
emotion)` cell). So this stage writes `turns.json` + overlaps and returns
a `Diarization` with `refs = {}`; the runner calls `ref_picker.pick_refs`
after the emotion stage returns. `ref_text` is then backfilled by the
runner after ASR runs.

HF token: read from the env var named by `hf_token_env` (default `HF_TOKEN`).
Falls back to the `hf auth login` cached token if the env var is unset.
All three gated repos must be license-accepted on huggingface.co:
  pyannote/segmentation-3.0, pyannote/speaker-diarization-3.1,
  pyannote/speaker-diarization-community-1.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import (
    Diarization,
    OverlapInterval,
    SeparatedAudio,
    SpeakerTurn,
)


def _resolve_device(requested: str) -> str:
    if requested in ("cpu", "cuda", "mps"):
        return requested
    try:
        import torch  # type: ignore[import-not-found]
        # pyannote 3.1 has incomplete MPS coverage; prefer CUDA, fall back to CPU.
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _passes_vad(start: float, end: float, vad_intervals: list[tuple[float, float]] | None) -> bool:
    if not vad_intervals:
        return True
    for vs, ve in vad_intervals:
        if vs < end and ve > start:
            return True
    return False


def _extract_overlaps(diar_annotation) -> list[OverlapInterval]:
    """Pull overlap regions directly from pyannote's Annotation.

    `Annotation.get_overlap()` returns a Timeline of regions where >=2
    different labels are simultaneously active.
    """
    timeline = diar_annotation.get_overlap()
    return [OverlapInterval(start=float(seg.start), end=float(seg.end)) for seg in timeline]


class PyannoteDiarize:
    id = "pyannote_3_1"

    def __init__(
        self,
        model_id: str = "pyannote/speaker-diarization-3.1",
        device: str = "auto",
        min_speakers: int = 0,
        max_speakers: int = 8,
        min_ref_seconds: float = 3.0,
        hf_token_env: str = "HF_TOKEN",
    ) -> None:
        from pyannote.audio import Pipeline  # lazy (heavy)
        import torch  # type: ignore[import-not-found]

        # Token resolution order: explicit env var, then HF CLI cached token
        # (`hf auth login` / `huggingface-cli login`). The cached token covers
        # the common case of "I'm already logged in to HF on this machine."
        token = os.environ.get(hf_token_env)
        if not token:
            try:
                from huggingface_hub import get_token
                token = get_token()
            except Exception:
                token = None
        if not token:
            raise RuntimeError(
                f"diarize: no HF token found. Either `hf auth login` (recommended) "
                f"or set env var {hf_token_env!r}. Also accept the license on all "
                "three gated repos pyannote 3.1 pulls in:\n"
                "  - huggingface.co/pyannote/segmentation-3.0\n"
                "  - huggingface.co/pyannote/speaker-diarization-3.1\n"
                "  - huggingface.co/pyannote/speaker-diarization-community-1"
            )
        self.pipeline = Pipeline.from_pretrained(model_id, token=token)
        self.device = _resolve_device(device)
        self.pipeline.to(torch.device(self.device))
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.min_ref_seconds = min_ref_seconds

    def diarize(self, separated: SeparatedAudio, ctx: JobContext) -> Diarization:
        ctx.progress("diarize", 0.05, f"loaded pyannote on {self.device}")

        kwargs: dict[str, Any] = {}
        if self.min_speakers and self.min_speakers > 0:
            kwargs["min_speakers"] = self.min_speakers
        if self.max_speakers and self.max_speakers > 0:
            kwargs["max_speakers"] = self.max_speakers

        ctx.progress("diarize", 0.10, "running diarization")
        result = self.pipeline(str(separated.vocals), **kwargs)
        ctx.progress("diarize", 0.85, "pyannote finished")

        # pyannote 3.1 returns a `DiarizeOutput` (with `.speaker_diarization`
        # and `.exclusive_speaker_diarization` Annotations) by default; older
        # builds + `legacy=True` return an Annotation directly. Handle both.
        annotation = getattr(result, "speaker_diarization", result)

        # Collect turns. annotation.itertracks yields (Segment, track_name, label).
        raw_turns: list[SpeakerTurn] = []
        for segment, _track, label in annotation.itertracks(yield_label=True):
            raw_turns.append(SpeakerTurn(
                start=float(segment.start),
                end=float(segment.end),
                speaker_id=str(label),
            ))
        raw_turns.sort(key=lambda t: (t.start, t.end))

        # Drop turns that have zero intersection with any VAD interval.
        kept: list[SpeakerTurn] = [
            t for t in raw_turns if _passes_vad(t.start, t.end, separated.vad_intervals)
        ]

        overlaps = _extract_overlaps(annotation)

        # Persist turns.json.
        diarize_dir = ctx.job_dir / "diarize"
        diarize_dir.mkdir(exist_ok=True)
        speakers = sorted({t.speaker_id for t in kept})
        (diarize_dir / "turns.json").write_text(json.dumps({
            "model": self.pipeline.__class__.__name__,
            "device": self.device,
            "speakers": speakers,
            "turns": [{"start": t.start, "end": t.end, "speaker_id": t.speaker_id} for t in kept],
            "overlaps": [{"start": ov.start, "end": ov.end} for ov in overlaps],
        }, indent=2))

        ctx.progress("diarize", 1.0, f"{len(speakers)} speakers, {len(kept)} turns")

        # `refs` is intentionally empty — the runner populates it after the
        # `emotion` stage runs, since per-(speaker, emotion) ref clips need
        # the per-turn emotion bin to be picked.
        return Diarization(
            turns=kept,
            overlaps=overlaps,
            refs={},
            sample_rate=separated.sample_rate,
        )


def build(config: dict[str, Any]) -> PyannoteDiarize:
    return PyannoteDiarize(
        model_id=config.get("model_id", "pyannote/speaker-diarization-3.1"),
        device=config.get("device", "auto"),
        min_speakers=int(config.get("min_speakers", 0)),
        max_speakers=int(config.get("max_speakers", 8)),
        min_ref_seconds=float(config.get("min_ref_seconds", 3.0)),
        hf_token_env=config.get("hf_token_env", "HF_TOKEN"),
    )
