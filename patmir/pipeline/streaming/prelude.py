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


def _snap_window_end(job_dir: Path, target: float) -> float:
    """VAD-snapped window end: the end of the last complete speech utterance at
    or before `target`, so section 1 begins on a silence boundary (sentence-safe).
    Falls back to `target` if VAD data is absent."""
    meta = job_dir / "separated.meta.json"
    if not meta.exists():
        return target
    intervals = json.loads(meta.read_text()).get("vad_intervals") or []
    ends = [float(e) for _s, e in intervals if float(e) <= target]
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
) -> PreludeResult:
    pool = build_pool(voice_pool) if voice_pool else _default_pool()
    registry = SpeakerRegistry(pool)

    with tempfile.TemporaryDirectory(prefix="patmir-prelude-") as tmp:
        slice_wav = Path(tmp) / "prelude.wav"
        _slice_audio(source, window_seconds, slice_wav)
        manifest = run_job(slice_wav, cfg, runs_root=runs_root, stop_after="asr",
                           stage_cache=stage_cache)
        job_dir = (runs_root or cfg.storage.runs_root) / manifest.job_id

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
    )
