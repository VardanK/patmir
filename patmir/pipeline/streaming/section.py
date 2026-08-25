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

Perf: this calls run_job twice per section (analysis, then dub). Callers pass a
shared `stage_cache` so stage models are built once for the whole stream and
kept warm across sections + across the two calls — the per-section rebuild tax
(measured 11–13% of watch-along wall) only applies when no cache is given.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import threading
from time import perf_counter
from dataclasses import dataclass, field
from pathlib import Path

from patmir.pipeline.config import PipelineConfig
from patmir.pipeline.runner import run_job
from patmir.pipeline.streaming.embeddings import embed
from patmir.pipeline.streaming.prelude import _primary_ref, _refs_by_speaker, _snap_window_end
from patmir.pipeline.streaming.registry import SpeakerRegistry
from patmir.pipeline.streaming.speaker_profile import profile_speaker

DEFAULT_SECTION_SECONDS = 30.0
# Lookahead past the target so VAD can find the next silence to cut on. The
# padded audio is fully analyzed, and the dub now extends INTO the pad up to
# the last safe VAD silence (see END_GUARD_SECONDS) instead of being discarded
# at the target — longer sections, fewer boundaries, less re-analyzed overlap
# (performance.md roadmap #5). 8s comfortably covers one trailing utterance.
DEFAULT_PAD_SECONDS = 8.0
# VAD ends within this many seconds of the hard slice end are suspect — an
# utterance cut mid-speech by the slice looks like a VAD end there. The
# section-end snap only trusts silences at least this far from the cut.
END_GUARD_SECONDS = 1.5


@dataclass
class SectionResult:
    index: int
    start: float                 # global timeline start (s)
    end: float                   # global timeline end, VAD-snapped (s)
    dub_wav: Path | None         # section dub chunk (slice-relative)
    job_dir: Path
    local_to_global: dict[str, str] = field(default_factory=dict)
    new_speakers: list[str] = field(default_factory=list)
    # Wall seconds for the whole section, INCLUDING the work no stage owns
    # (ffmpeg slice/trim, ECAPA embed, age-gender profile). Set by the two
    # producers below so the live path and bench read one number, not two.
    wall_s: float | None = None


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
        gid, is_new = registry.resolve_or_register(
            emb, profile=profile,
            ref_audio=primary["ref_audio"], ref_text=primary.get("ref_text", ""),
        )
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


def _remap_refs(job_dir: Path, mapping: dict[str, str]) -> None:
    """Rewrite diarize/refs.json speaker ids local→global, in lockstep with the
    transcript remap. Without this the dub pass's TTS ref lookup — keyed by the
    transcript's now-global speaker_id — finds nothing and every line silently
    falls back to the no-ref default voice (the watch-along voice-swap bug).
    Two local speakers can resolve to the same global character; on a
    (speaker, emotion) key collision the longer clip wins."""
    rpath = job_dir / "diarize" / "refs.json"
    if not rpath.exists() or not mapping:
        return

    def _dur(r: dict) -> float:
        return float(r.get("source_end", 0.0)) - float(r.get("source_start", 0.0))

    remapped: dict[str, dict] = {}
    for r in json.loads(rpath.read_text()).values():
        spk = mapping.get(r.get("speaker_id"), r.get("speaker_id"))
        emo = r.get("emotion") or "neutral"
        entry = {**r, "speaker_id": spk}
        key = f"{spk}__{emo}"
        if key in remapped and _dur(remapped[key]) >= _dur(entry):
            continue
        remapped[key] = entry
    rpath.write_text(json.dumps(remapped, indent=2, ensure_ascii=False))


def dub_prelude_window(
    cfg: PipelineConfig,
    prelude,
    *,
    runs_root: Path | None = None,
    stage_cache: dict | None = None,
    cancel: threading.Event | None = None,
    bench: bool = False,
) -> SectionResult:
    """Produce the opening dub for the prelude window, reusing the prelude's
    already-completed analysis (no second Demucs/Scribe pass).

    The prelude analyzed `[0, window_end]` and assigned each Scribe-local
    speaker label a registry voice keyed by that same label — so the transcript
    needs no local→global remap. We just trim it to the VAD-snapped window end,
    then resume translate→tts→mux with the registry voices. This is section 0;
    the streaming runner picks up at `window_end`.
    """
    runs_root = runs_root or cfg.storage.runs_root
    t0 = perf_counter()
    job_dir = prelude.job_dir
    end_rel = prelude.window_end

    # Identity remap is a no-op (prelude labels == registry keys); trim only.
    _remap_and_trim_transcript(job_dir, {}, end_rel)

    run_job(
        job_dir, cfg, runs_root=runs_root,
        resume_job_dir=job_dir, resume_from="translate", stop_after="mux",
        extra={
            "voice_assignments": prelude.registry.assignments(),
            "locked_refs": prelude.registry.locked_refs(),
            "speaker_profiles": prelude.registry.profiles(),
            # Makes the tts/translate adapters write timings.jsonl. Watch-along
            # never set it, which is why BenchResult.tts/.translate were
            # hard-coded None for this mode.
            **({"bench": True} if bench else {}),
        },
        stage_cache=stage_cache,
        cancel=cancel,
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
        wall_s=perf_counter() - t0,
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
    stage_cache: dict | None = None,
    cancel: threading.Event | None = None,
    bench: bool = False,
    source_lang: str | None = None,
) -> SectionResult:
    runs_root = runs_root or cfg.storage.runs_root
    t0 = perf_counter()
    slice_len = target_seconds + pad_seconds

    with tempfile.TemporaryDirectory(prefix=f"patmir-sec{index}-") as tmp:
        slice_wav = Path(tmp) / "section.wav"
        _slice_audio(source, start, slice_len, slice_wav)
        manifest = run_job(slice_wav, cfg, runs_root=runs_root, stop_after="asr",
                           stage_cache=stage_cache, cancel=cancel,
                           extra={"bench": True} if bench else None,
                           # Skip ASR's per-section language probe (3.5-4.6 s
                           # measured, 12-15% of section wall). The prelude
                           # already detected it on the same soundtrack; ASR
                           # short-circuits when ctx.source_lang is set.
                           source_lang=source_lang)
        job_dir = runs_root / manifest.job_id

    # VAD-snap the section end (slice-relative). Snap as far into the padded
    # slice as VAD allows rather than stopping at the target: the pad is
    # analyzed either way, so dubbing it too means fewer sections per video
    # and proportionally less pad re-analysis. Never snap into the last
    # END_GUARD_SECONDS, where a mid-speech slice cut masquerades as silence.
    end_rel = _snap_window_end(job_dir, slice_len - END_GUARD_SECONDS)

    mapping, new_speakers = _resolve_identities(registry, job_dir)
    _remap_and_trim_transcript(job_dir, mapping, end_rel)
    _remap_refs(job_dir, mapping)

    # translate → tts → mux on the (remapped) transcript, with registry voices.
    # On resume the `source` arg is unused (earlier stages load from disk).
    run_job(
        job_dir,
        cfg,
        runs_root=runs_root,
        resume_job_dir=job_dir,
        resume_from="translate",
        stop_after="mux",
        extra={
            "voice_assignments": registry.assignments(),
            "locked_refs": registry.locked_refs(),
            "speaker_profiles": registry.profiles(),
            **({"bench": True} if bench else {}),
        },
        stage_cache=stage_cache,
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
        wall_s=perf_counter() - t0,
    )
