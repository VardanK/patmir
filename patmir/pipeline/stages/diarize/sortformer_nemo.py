"""Speaker diarization via NVIDIA Sortformer on NeMo — the CUDA path.

Same model as `sortformer_mlx.py`, different runtime. Sortformer is NVIDIA's
own, so NeMo is its reference implementation; `mlx-audio` is the third-party
Apple-Silicon port. This candidate exists so the CUDA box is not forced back to
`pyannote_3_1`, which merges short turns from distinct characters into one
speaker — the failure that drove diarize off pyannote in the first place
(ADR 2026-06-15, quality issue #13).

End-to-end neural diarization: no clustering hyperparameters, and because it
resolves concurrent speakers we can re-derive overlap intervals. Capped at 4
speakers by the model architecture. Post-processing (label normalization,
same-speaker merge, overlap derivation, VAD filter) is shared with the MLX
runtime in `_postprocess.py` so the two cannot drift and be mistaken for a model
difference.

Two things this handles that the MLX path does not have to:

  * **Audio format.** The model card specifies mono 16 kHz, and the vocals stem
    arrives at Demucs's native rate in stereo. `mlx-audio` and pyannote both
    resample internally; rather than rely on NeMo's dataloader doing the same
    (documented nowhere), the stem is converted with ffmpeg into an inspectable
    job artifact first.
  * **Return shape.** `diarize()` is documented across all three Sortformer
    model cards as returning segments "formatted as 'begin_seconds,
    end_seconds, speaker_index'", but none of them shows a literal example, so
    whether an element is a string, a tuple or an object was unspecified when
    this was written. `parse_segment` accepts all three and raises on anything
    else rather than guessing — a silently mis-parsed diarization is worse than
    a crash, since it would surface much later as characters sharing a voice.

    **Observed on NeMo 2026-08-23** (RTX 3090), and pinned in
    `test_diarize_sortformer.ObservedNeMoShapeTest`:

        [['0.000 1.840 speaker_0', '2.000 3.680 speaker_1', ...]]

    A list per input file, of **whitespace**-separated strings — not the
    comma-separated form the prose implies — with `speaker_N` labels rather than
    bare integers. That is the live branch; the tuple and object branches are
    now insurance against a NeMo change rather than genuine unknowns. Note the
    first real call landed exactly on the ambiguous case the parser was written
    for: a nested list of *three* string segments, which is structurally also a
    valid 3-element segment.

Install (Linux + CUDA):
    uv pip install -e ".[cuda]"      # nemo_toolkit[asr]>=2.0 from PyPI

Verified 2026-08-24: `nemo_toolkit 3.0.0` installs cleanly from **PyPI**, no git
checkout and no Cython prerequisite. This docstring previously said to build from
git with Cython first; that was written before the dep was declared and is wrong.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
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

# The model's required input format (model card: "Mono, 16kHz").
TARGET_SR = 16000

# Streaming knobs, in 80 ms frames. Left unset by default so the checkpoint's own
# configuration stands; see the candidate entry in patmir.toml for the presets
# NVIDIA publishes. Setting any of these applies all that are set and then asks
# NeMo to validate the combination.
_STREAMING_KNOBS = (
    "chunk_len",
    "chunk_right_context",
    "fifo_len",
    "spkcache_update_period",
    "spkcache_len",
)


def _resolve_device(requested: str) -> str:
    if requested in ("cpu", "cuda", "mps"):
        return requested
    try:
        import torch  # type: ignore[import-not-found]
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def parse_segment(seg: Any) -> tuple[float, float, Any]:
    """One Sortformer segment -> `(start, end, speaker)`.

    Tolerates the three plausible shapes for an undocumented return type:
    an object with `.start`/`.end`/`.speaker`, a 3-element sequence, or a
    string of three fields separated by commas or whitespace.

    Raises `ValueError` on anything else. This is deliberate — see the module
    docstring: guessing would turn a format surprise into two characters
    quietly sharing one voice several stages downstream.
    """
    # Object form (what mlx-audio yields, and cheap to accept here too).
    start = getattr(seg, "start", None)
    end = getattr(seg, "end", None)
    if start is not None and end is not None:
        speaker = getattr(seg, "speaker", getattr(seg, "speaker_id", 0))
        return float(start), float(end), speaker

    # String form: "0.5, 2.3, 1" or "0.5 2.3 speaker_1".
    if isinstance(seg, str):
        text = seg.strip()
        fields = [f for f in (text.split(",") if "," in text else text.split()) if f.strip()]
        if len(fields) == 3:
            try:
                return float(fields[0]), float(fields[1]), fields[2].strip()
            except ValueError as exc:
                raise ValueError(
                    f"Sortformer segment {seg!r} has 3 fields but the first two are not "
                    f"times: {exc}"
                ) from exc
        raise ValueError(
            f"Sortformer segment {seg!r} split into {len(fields)} fields, expected 3 "
            "('begin_seconds, end_seconds, speaker_index')."
        )

    # Sequence form: (0.5, 2.3, 1).
    if isinstance(seg, (list, tuple)) and len(seg) == 3:
        return float(seg[0]), float(seg[1]), seg[2]

    raise ValueError(
        f"Unrecognized Sortformer segment {seg!r} (type {type(seg).__name__}). Expected an "
        "object with .start/.end/.speaker, a 3-element sequence, or a 3-field string."
    )


def _is_segment(item: Any) -> bool:
    """Is this one segment, or a group of them?

    The only genuinely ambiguous case is a 3-element sequence, which is both a
    valid segment `(start, end, speaker)` and a valid group of three segments.
    A segment's first two elements are times; a group's are not.
    """
    if isinstance(item, str):
        return True
    if hasattr(item, "start") and hasattr(item, "end"):
        return True
    if isinstance(item, (list, tuple)) and len(item) == 3:
        try:
            float(item[0])
            float(item[1])
        except (TypeError, ValueError):
            return False
        return True
    return False


def iter_segments(predicted: Any) -> list[Any]:
    """Flatten `diarize()`'s output to a flat list of segments.

    The model card's example indexes per audio file (`predicted_segments[0]`),
    but we always pass exactly one file, so accept either the nested or the flat
    shape rather than depending on which it is.
    """
    if not isinstance(predicted, (list, tuple)) or not predicted:
        return []
    if _is_segment(predicted[0]):
        return list(predicted)
    return [seg for group in predicted for seg in (group or [])]


class SortformerNeMo:
    id = "sortformer_nemo"

    def __init__(
        self,
        model_id: str = "nvidia/diar_streaming_sortformer_4spk-v2.1",
        device: str = "auto",
        merge_gap_seconds: float = 0.5,
        duplicate_containment: float = 0.4,
        duplicate_min_seconds: float = 1.0,
        batch_size: int = 1,
        streaming: dict[str, Any] | None = None,
    ) -> None:
        import torch  # lazy (heavy)
        from nemo.collections.asr.models import SortformerEncLabelModel  # lazy (very heavy)

        self.model_id = model_id
        self.device = _resolve_device(device)
        self.merge_gap_seconds = float(merge_gap_seconds)
        self.duplicate_containment = float(duplicate_containment)
        self.duplicate_min_seconds = float(duplicate_min_seconds)
        self.batch_size = int(batch_size)

        # A local `.nemo` checkpoint is supported so a box can run fully offline
        # (the model card documents both entry points).
        local = Path(model_id)
        if local.suffix == ".nemo" and local.exists():
            self.model = SortformerEncLabelModel.restore_from(
                restore_path=str(local), map_location=self.device, strict=False,
            )
        else:
            self.model = SortformerEncLabelModel.from_pretrained(model_id)
        self.model.to(torch.device(self.device))
        self.model.eval()

        self.streaming = {k: int(v) for k, v in (streaming or {}).items()
                          if k in _STREAMING_KNOBS and v is not None}
        if self.streaming:
            modules = self.model.sortformer_modules
            for key, value in self.streaming.items():
                setattr(modules, key, value)
            # NeMo validates the combination — an inconsistent set of frame
            # counts is a config error we want at build time, not mid-run.
            check = getattr(modules, "_check_streaming_parameters", None)
            if callable(check):
                check()

    def _to_mono_16k(self, vocals: Path, out_dir: Path, ctx: JobContext) -> Path:
        """Convert the vocals stem to the mono 16 kHz WAV the model expects."""
        target = out_dir / "vocals_16k_mono.wav"
        if target.exists():
            return target
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(vocals),
            "-vn",
            "-ac", "1",
            "-ar", str(TARGET_SR),
            "-c:a", "pcm_s16le",
            str(target),
        ], check=True, capture_output=True)
        ctx.progress("diarize", 0.10, f"vocals -> mono {TARGET_SR} Hz")
        return target

    def diarize(self, separated: SeparatedAudio, ctx: JobContext) -> Diarization:
        ctx.progress("diarize", 0.05, f"sortformer-nemo ({self.model_id}) on {self.device}")

        diarize_dir = ctx.job_dir / "diarize"
        diarize_dir.mkdir(exist_ok=True)
        audio = self._to_mono_16k(separated.vocals, diarize_dir, ctx)

        ctx.progress("diarize", 0.15, "running diarization")
        predicted = self.model.diarize(audio=[str(audio)], batch_size=self.batch_size)

        raw: list[SpeakerTurn] = []
        for seg in iter_segments(predicted):
            start, end, speaker = parse_segment(seg)
            raw.append(SpeakerTurn(start=start, end=end, speaker_id=norm_speaker(speaker)))
        ctx.progress("diarize", 0.8, f"{len(raw)} raw segments")

        merged = merge_same_speaker(raw, self.merge_gap_seconds)
        merged, aliases = merge_duplicate_speakers(
            merged, self.merge_gap_seconds,
            min_containment=self.duplicate_containment,
            min_shared_seconds=self.duplicate_min_seconds,
        )
        if aliases:
            ctx.progress("diarize", 0.85, "one speaker on two slots: " + ", ".join(
                f"{k}->{v}" for k, v in sorted(aliases.items())))
        kept = [t for t in merged if passes_vad(t.start, t.end, separated.vad_intervals)]
        overlaps = derive_overlaps(kept)
        speakers = sorted({t.speaker_id for t in kept})

        (diarize_dir / "turns.json").write_text(json.dumps({
            "model": self.model_id,
            "device": self.device,
            "streaming": self.streaming or None,
            "speakers": speakers,
            "merged_speakers": aliases or None,
            "turns": [{"start": t.start, "end": t.end, "speaker_id": t.speaker_id} for t in kept],
            "overlaps": [{"start": ov.start, "end": ov.end} for ov in overlaps],
        }, indent=2))

        ctx.progress("diarize", 1.0,
                     f"{len(speakers)} speakers, {len(kept)} turns, {len(overlaps)} overlaps")

        # refs intentionally empty — the runner's pick_refs fills it after the
        # emotion stage (per-(speaker, emotion) clips need the emotion bin).
        return Diarization(
            turns=kept,
            overlaps=overlaps,
            refs={},
            sample_rate=separated.sample_rate,
        )


def build(config: dict[str, Any]) -> SortformerNeMo:
    streaming = {k: config[k] for k in _STREAMING_KNOBS if config.get(k) is not None}
    return SortformerNeMo(
        model_id=str(config.get("model_id", "nvidia/diar_streaming_sortformer_4spk-v2.1")),
        device=str(config.get("device", "auto")),
        merge_gap_seconds=float(config.get("merge_gap_seconds", 0.5)),
        duplicate_containment=float(config.get("duplicate_containment", 0.4)),
        duplicate_min_seconds=float(config.get("duplicate_min_seconds", 1.0)),
        batch_size=int(config.get("batch_size", 1)),
        streaming=streaming or None,
    )
