"""Speaker diarization via NVIDIA Sortformer 4-speaker streaming on MLX.

Runs through `mlx-audio` (`mlx_audio.vad.load`). `model.generate_stream(path,
chunk_duration=...)` yields chunk results, each with `.segments`
(`DiarizationSegment(.start, .end, .speaker:int)`). End-to-end neural
diarization (no clustering hyperparameters), which resolves short cartoon turns
far better than pyannote — the failure mode that pushed us to cloud Scribe.

Sortformer emits fragmented same-speaker segments, so we merge adjacent
same-speaker pieces (gap ≤ `merge_gap_seconds`) into turns. Because it's
end-to-end it also emits concurrent speakers, so we re-derive overlap intervals
(≥2 distinct speakers active) — restoring the overlap output Scribe dropped.

Output contract mirrors `pyannote.py`: a `Diarization` with `turns` +
`overlaps`, `refs={}` (the runner's `pick_refs` fills refs after the emotion
stage), normalized `SPEAKER_NN` ids, and `diarize/turns.json` on disk.

The post-processing (label normalization, same-speaker merge, overlap
derivation, VAD filter) lives in `_postprocess.py`, shared with the NeMo/CUDA
runtime of the same model (`sortformer_nemo.py`) so the two cannot drift.

Capped at 4 speakers (model architecture). VAD post-filter drops turns with
zero intersection with the Silero VAD intervals from the separate stage.

Install once: `uv pip install "git+https://github.com/Blaizzy/mlx-audio.git"`.
"""
from __future__ import annotations

import json
from typing import Any

from patmir.pipeline.context import JobContext
from patmir.pipeline.stage import Diarization, SeparatedAudio, SpeakerTurn
from patmir.pipeline.stages.diarize._postprocess import (
    derive_overlaps,
    merge_duplicate_speakers,
    merge_same_speaker,
    norm_speaker,
    passes_vad,
)


class SortformerMLX:
    id = "sortformer_mlx"

    def __init__(
        self,
        repo_id: str,
        chunk_duration: float = 5.0,
        merge_gap_seconds: float = 0.5,
        duplicate_containment: float = 0.4,
        duplicate_min_seconds: float = 1.0,
        duplicate_min_speaker_seconds: float = 10.0,
    ) -> None:
        from mlx_audio.vad import load  # lazy import (heavy)
        self._model = load(repo_id)
        self.repo_id = repo_id
        self.chunk_duration = chunk_duration
        self.merge_gap_seconds = merge_gap_seconds
        self.duplicate_containment = float(duplicate_containment)
        self.duplicate_min_seconds = float(duplicate_min_seconds)
        self.duplicate_min_speaker_seconds = float(duplicate_min_speaker_seconds)

    def diarize(self, separated: SeparatedAudio, ctx: JobContext) -> Diarization:
        ctx.progress("diarize", 0.05, f"sortformer-mlx ({self.repo_id})")

        raw: list[SpeakerTurn] = []
        for res in self._model.generate_stream(str(separated.vocals), chunk_duration=self.chunk_duration):
            for seg in getattr(res, "segments", []) or []:
                raw.append(SpeakerTurn(
                    start=float(getattr(seg, "start", 0.0)),
                    end=float(getattr(seg, "end", 0.0)),
                    speaker_id=norm_speaker(getattr(seg, "speaker", getattr(seg, "speaker_id", "0"))),
                ))
        ctx.progress("diarize", 0.8, f"{len(raw)} raw segments")

        merged = merge_same_speaker(raw, self.merge_gap_seconds)
        merged, aliases = merge_duplicate_speakers(
            merged, self.merge_gap_seconds,
            min_containment=self.duplicate_containment,
            min_shared_seconds=self.duplicate_min_seconds,
            min_speaker_seconds=self.duplicate_min_speaker_seconds,
        )
        if aliases:
            ctx.progress("diarize", 0.85, "one speaker on two slots: " + ", ".join(
                f"{k}->{v}" for k, v in sorted(aliases.items())))
        kept = [t for t in merged if passes_vad(t.start, t.end, separated.vad_intervals)]
        overlaps = derive_overlaps(kept)
        speakers = sorted({t.speaker_id for t in kept})

        diarize_dir = ctx.job_dir / "diarize"
        diarize_dir.mkdir(exist_ok=True)
        (diarize_dir / "turns.json").write_text(json.dumps({
            "model": self.repo_id,
            "device": "mps",
            "speakers": speakers,
            "merged_speakers": aliases or None,
            "turns": [{"start": t.start, "end": t.end, "speaker_id": t.speaker_id} for t in kept],
            "overlaps": [{"start": ov.start, "end": ov.end} for ov in overlaps],
        }, indent=2))

        ctx.progress("diarize", 1.0, f"{len(speakers)} speakers, {len(kept)} turns, {len(overlaps)} overlaps")

        # refs intentionally empty — runner's pick_refs fills it after the
        # emotion stage (per-(speaker, emotion) clips need the emotion bin).
        return Diarization(
            turns=kept,
            overlaps=overlaps,
            refs={},
            sample_rate=separated.sample_rate,
        )


def build(config: dict[str, Any]) -> SortformerMLX:
    return SortformerMLX(
        repo_id=config["repo_id"],
        chunk_duration=float(config.get("chunk_duration", 5.0)),
        merge_gap_seconds=float(config.get("merge_gap_seconds", 0.5)),
        duplicate_containment=float(config.get("duplicate_containment", 0.4)),
        duplicate_min_seconds=float(config.get("duplicate_min_seconds", 1.0)),
        duplicate_min_speaker_seconds=float(
            config.get("duplicate_min_speaker_seconds", 10.0)),
    )
