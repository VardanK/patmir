"""Per-section processing for the streaming runner (Slice 2 #9b).

Given the global SpeakerRegistry from the prelude, process one VAD-snapped
section of the video:

  1. slice [start, start+target+pad] of the source
  2. analyze it (ingest→separate→diarize→emotion→asr) via run_job(stop_after=asr)
  3. VAD-snap the section end (so the next section begins on a silence boundary)
  4. resolve each section-local speaker to a global character (ECAPA match
     against the registry, or register a new one) → local→global id map
  5. remap + trim the transcript to the section boundary
  6. translate→tts→mux via run_job(resume_from=translate) with the registry's
     voice assignments → the section's dubbed audio chunk

The chunk's audio timeline is slice-relative ([0, section_len]); the result
carries the global `start`/`end` so the client plays it at the right offset.

NOTE (perf): this calls run_job twice per section, which rebuilds stage models
each call. Functionally correct; a warm-model loop (build stages once, process
sections in place) is a follow-up optimization.
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
from patmir.pipeline.streaming.prelude import _primary_ref, _refs_by_speaker, _snap_window_end
from patmir.pipeline.streaming.registry import SpeakerRegistry
from patmir.pipeline.streaming.speaker_profile import profile_speaker

DEFAULT_SECTION_SECONDS = 30.0
DEFAULT_PAD_SECONDS = 15.0


@dataclass
class SectionResult:
    index: int
    start: float                 # global timeline start (s)
    end: float                   # global timeline end, VAD-snapped (s)
    dub_wav: Path | None         # section dub chunk (slice-relative)
    job_dir: Path
    local_to_global: dict[str, str] = field(default_factory=dict)
    new_speakers: list[str] = field(default_factory=list)


def _slice_audio(source: Path, start: float, duration: float, dest: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-t", str(duration), "-i", str(source),
         "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(dest)],
        check=True, capture_output=True,
    )


def _trim_wav(src: Path, seconds: float, dest: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-t", str(seconds), "-i", str(src), "-c", "copy", str(dest)],
        check=True, capture_output=True,
    )


def _resolve_identities(registry: SpeakerRegistry, job_dir: Path) -> tuple[dict[str, str], list[str]]:
    """Map this section's local speaker labels to global characters, matching
    against the registry's stored embeddings (resolve-or-register)."""
    local_to_global: dict[str, str] = {}
    new_speakers: list[str] = []
    for local_id, refs in _refs_by_speaker(job_dir).items():
        primary = _primary_ref(refs)
        emb = embed(primary["ref_audio"])
        profile = profile_speaker(primary["ref_audio"])
        gid, is_new = registry.resolve_or_register(emb, profile=profile, ref_audio=primary["ref_audio"])
        local_to_global[local_id] = gid
        if is_new:
            new_speakers.append(gid)
            # seed the rest of this speaker's emotion clips for robust future matching
            for r in refs:
                if r is not primary:
                    registry.set_embedding(gid, embed(r["ref_audio"]))
        else:
            # online enrichment: add this section's clips to the matched character
            for r in refs:
                registry.set_embedding(gid, embed(r["ref_audio"]))
    return local_to_global, new_speakers


def _remap_and_trim_transcript(job_dir: Path, mapping: dict[str, str], section_end_rel: float) -> None:
    """Rewrite transcript.json: map sentence speaker_ids local→global and keep
    only sentences within the section boundary (the rest are re-analyzed by the
    next section, whose start is this silence boundary)."""
    tpath = job_dir / "transcript.json"
    data = json.loads(tpath.read_text())
    kept = []
    for s in data.get("sentences", []):
        if s.get("start", 0.0) >= section_end_rel:
            continue
        if s.get("speaker_id") in mapping:
            s["speaker_id"] = mapping[s["speaker_id"]]
        kept.append(s)
    data["sentences"] = kept
    tpath.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def dub_prelude_window(
    cfg: PipelineConfig,
    prelude,
    *,
    runs_root: Path | None = None,
) -> SectionResult:
    """Produce the opening dub for the prelude window, reusing the prelude's
    already-completed analysis (no second Demucs/Scribe pass).

    The prelude analyzed `[0, window_end]` and assigned each Scribe-local
    speaker label a registry voice keyed by that same label — so the transcript
    needs no local→global remap. We just trim it to the VAD-snapped window end,
    then resume translate→tts→mux with the registry voices. This is section 0;
    the streaming runner picks up at `window_end`.
    """
    runs_root = runs_root or Path.cwd() / "runs"
    job_dir = prelude.job_dir
    end_rel = prelude.window_end

    # Identity remap is a no-op (prelude labels == registry keys); trim only.
    _remap_and_trim_transcript(job_dir, {}, end_rel)

    run_job(
        job_dir, cfg, runs_root=runs_root,
        resume_job_dir=job_dir, resume_from="translate", stop_after="mux",
        extra={"voice_assignments": prelude.registry.assignments()},
    )

    dub = job_dir / "dub.wav"
    out_chunk = job_dir / "section_dub.wav"
    if dub.exists():
        _trim_wav(dub, end_rel, out_chunk)
    else:
        out_chunk = None  # type: ignore[assignment]

    return SectionResult(
        index=0, start=0.0, end=end_rel, dub_wav=out_chunk, job_dir=job_dir,
        local_to_global={}, new_speakers=list(prelude.registry.known_speakers()),
    )


def process_section(
    source: Path,
    cfg: PipelineConfig,
    registry: SpeakerRegistry,
    *,
    index: int,
    start: float,
    target_seconds: float = DEFAULT_SECTION_SECONDS,
    pad_seconds: float = DEFAULT_PAD_SECONDS,
    runs_root: Path | None = None,
) -> SectionResult:
    runs_root = runs_root or Path.cwd() / "runs"
    slice_len = target_seconds + pad_seconds

    with tempfile.TemporaryDirectory(prefix=f"patmir-sec{index}-") as tmp:
        slice_wav = Path(tmp) / "section.wav"
        _slice_audio(source, start, slice_len, slice_wav)
        manifest = run_job(slice_wav, cfg, runs_root=runs_root, stop_after="asr")
        job_dir = runs_root / manifest.job_id

    # VAD-snap the section end within the target window (slice-relative).
    end_rel = _snap_window_end(job_dir, target_seconds)

    mapping, new_speakers = _resolve_identities(registry, job_dir)
    _remap_and_trim_transcript(job_dir, mapping, end_rel)

    # translate → tts → mux on the (remapped) transcript, with registry voices.
    # On resume the `source` arg is unused (earlier stages load from disk).
    run_job(
        job_dir,
        cfg,
        runs_root=runs_root,
        resume_job_dir=job_dir,
        resume_from="translate",
        stop_after="mux",
        extra={"voice_assignments": registry.assignments()},
    )

    dub = job_dir / "dub.wav"
    out_chunk = job_dir / "section_dub.wav"
    if dub.exists():
        _trim_wav(dub, end_rel, out_chunk)
    else:
        out_chunk = None  # type: ignore[assignment]

    return SectionResult(
        index=index,
        start=start,
        end=start + end_rel,
        dub_wav=out_chunk,
        job_dir=job_dir,
        local_to_global=mapping,
        new_speakers=new_speakers,
    )
