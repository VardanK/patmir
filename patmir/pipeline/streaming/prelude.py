"""First-chunk prelude (Slice 2 #8).

On play, analyze the opening window of the video to establish the cast before
streaming starts. Reuses the existing batch stages (ingest→separate→diarize→
emotion→asr, incl. the runner's post-stage pick_refs + speaker/emotion tagging)
via `run_job(stop_after="asr")` on a sliced copy of the source, then builds a
populated `SpeakerRegistry`:

  for each speaker found in the window:
    - profile (gender + age)              → speaker_profile.profile_speaker
    - assign a gender/age-matched voice   → registry.assign_voice
    - store an ECAPA embedding            → registry.set_embedding   (for #9 matching)
    - lock the clone ref (OmniVoice path) → registry.lock_ref

Returns the registry + the analysis job dir + the VAD-snapped window end (where
the streaming runner's section 1 begins).
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

from patmir.pipeline.config import PipelineConfig
from patmir.pipeline.runner import run_job
from patmir.pipeline.streaming.embeddings import embed
from patmir.pipeline.streaming.registry import SpeakerRegistry, VoiceEntry, build_pool
from patmir.pipeline.streaming.speaker_profile import profile_speaker

DEFAULT_WINDOW_SECONDS = 60.0


@dataclass
class PreludeResult:
    registry: SpeakerRegistry
    job_dir: Path
    window_end: float                       # section 1 starts here (seconds)
    speakers: dict[str, dict] = field(default_factory=dict)  # speaker_id → {voice, gender, age, ref}
    # Language detected once, on the opening window. Sections reuse it instead
    # of re-probing: Qwen3-ASR's detection pass cost 3.5-4.6 s on EVERY section
    # of the 2026-08-25 session (12-15% of section wall) to re-answer a question
    # the prelude had already answered about the same soundtrack.
    source_lang: str | None = None


def _default_pool() -> list[VoiceEntry]:
    from patmir.eval.candidates.tts.elevenlabs import _DEFAULT_VOICE_POOL
    return build_pool(_DEFAULT_VOICE_POOL)


def _slice_audio(source: Path, seconds: float, dest: Path) -> None:
    """Extract the first `seconds` of `source` to a high-quality stereo WAV the
    existing ingest/separate stages can read (video or audio source)."""
    subprocess.run(
        ["ffmpeg", "-y", "-ss", "0", "-t", str(seconds), "-i", str(source),
         "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(dest)],
        check=True, capture_output=True,
    )


# A VAD interval end only marks a real silence if a gap follows it. Silero runs
# with `speech_pad_ms=500` (separate/htdemucs.py), which pads every speech run by
# 500 ms per side, so two runs separated by less than ~1 s of silence come back
# TOUCHING — `end == next_start`. Such a boundary sits mid-speech. Require a gap
# comfortably above float noise before trusting one.
_MIN_SILENCE_GAP_S = 0.05

# A snap this far below the requested target wastes most of the section: the
# whole padded slice is analyzed either way, so cutting at 8 s of 38 s throws
# away 79% of the work AND leaves the buffer short. Below this fraction of the
# target, cut at the target instead and accept a rougher boundary.
_MIN_SNAP_FRACTION = 0.5


def _snap_window_end(job_dir: Path, target: float,
                     *, min_fraction: float = _MIN_SNAP_FRACTION) -> float:
    """VAD-snapped window end: the end of the last complete speech utterance at
    or before `target`, so the next section begins on a silence boundary
    (sentence-safe). Falls back to `target` if VAD data is absent.

    Two rules, both learned from `runs/20260825-182332-section`, which analyzed
    38 s and emitted **8.1 s** (RTF 4.0) with 8 of its 10 ASR sentences thrown
    away:

    1. **A candidate must be followed by real silence.** That section's
       intervals were `[[0.0, 6.3], [6.3, 8.1], [8.1, 38.0]]` — contiguous,
       because Silero's 500 ms padding makes runs separated by short silences
       abut. Taking `max(end <= target)` picked 8.1, which is not a silence at
       all but the seam between two adjacent speech runs. Cutting there is
       exactly the mid-utterance cut the snap exists to avoid.

    2. **A candidate must not collapse the section.** Even a genuine silence
       early in the slice is the wrong cut if the only later boundary sits past
       `target`: the section still pays full analysis cost for a fraction of the
       output, and the buffer it was meant to extend shrinks instead.

    When nothing qualifies, cut at `target`. That can land mid-utterance — the
    transcript trim drops the partial sentence and the next section re-analyzes
    from the boundary — which is the honest trade: on continuous speech there is
    no silence to find, and a rough cut beats emitting a fifth of a section.
    """
    meta = job_dir / "separated.meta.json"
    if not meta.exists():
        return target
    try:
        raw = json.loads(meta.read_text()).get("vad_intervals") or []
        intervals = sorted((float(s), float(e)) for s, e in raw)
    except (ValueError, TypeError, json.JSONDecodeError):
        return target

    floor = target * min_fraction
    ends = []
    for i, (_s, e) in enumerate(intervals):
        nxt = intervals[i + 1][0] if i + 1 < len(intervals) else None
        # Rule 1: the last interval is always followed by silence (or the slice
        # edge); an interior one only counts if a real gap follows.
        if nxt is not None and nxt - e < _MIN_SILENCE_GAP_S:
            continue
        # Rule 2.
        if floor <= e <= target:
            ends.append(e)
    return max(ends) if ends else target


def _refs_by_speaker(job_dir: Path) -> dict[str, list[dict]]:
    """Group all of diarize/refs.json by speaker (every emotion clip)."""
    refs_path = job_dir / "diarize" / "refs.json"
    if not refs_path.exists():
        return {}
    grouped: dict[str, list[dict]] = {}
    for r in json.loads(refs_path.read_text()).values():
        grouped.setdefault(r["speaker_id"], []).append(r)
    return grouped


def _primary_ref(refs: list[dict]) -> dict:
    """Pick the representative ref for a speaker: neutral if present, else the
    longest source window (used for voice assignment + profiling)."""
    neutral = [r for r in refs if r["emotion"] == "neutral"]
    pool = neutral or refs
    return max(pool, key=lambda r: r["source_end"] - r["source_start"])


def _one_ref_per_speaker(job_dir: Path) -> dict[str, dict]:
    """One representative ref per speaker (back-compat helper)."""
    return {spk: _primary_ref(refs) for spk, refs in _refs_by_speaker(job_dir).items()}


def run_prelude(
    source: Path,
    cfg: PipelineConfig,
    *,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    runs_root: Path | None = None,
    voice_pool: list | None = None,
    stage_cache: dict | None = None,
    cancel: threading.Event | None = None,
    bench: bool = False,
) -> PreludeResult:
    pool = build_pool(voice_pool) if voice_pool else _default_pool()
    registry = SpeakerRegistry(pool)

    with tempfile.TemporaryDirectory(prefix="patmir-prelude-") as tmp:
        slice_wav = Path(tmp) / "prelude.wav"
        _slice_audio(source, window_seconds, slice_wav)
        manifest = run_job(slice_wav, cfg, runs_root=runs_root, stop_after="asr",
                           stage_cache=stage_cache, cancel=cancel,
                           extra={"bench": True} if bench else None)
        job_dir = (runs_root or cfg.storage.runs_root) / manifest.job_id
        detected_lang = manifest.source_lang

    window_end = _snap_window_end(job_dir, window_seconds)

    speakers: dict[str, dict] = {}
    for spk, refs in _refs_by_speaker(job_dir).items():
        primary = _primary_ref(refs)
        profile = profile_speaker(primary["ref_audio"])
        voice_id = registry.assign_voice(spk, profile)
        registry.lock_ref(spk, primary["ref_audio"], primary.get("ref_text", ""))
        # Seed one embedding per emotion clip so a speaker seen in several
        # states matches robustly later (resolve scores against the best).
        for r in refs:
            registry.set_embedding(spk, embed(r["ref_audio"]))
        speakers[spk] = {
            "voice_id": voice_id,
            "gender": profile.gender,
            "age": profile.age,
            "ref_audio": primary["ref_audio"],
            "embeddings": len(registry._embeddings.get(spk, [])),
        }

    return PreludeResult(
        registry=registry,
        job_dir=job_dir,
        window_end=window_end,
        speakers=speakers,
        source_lang=detected_lang,
    )
