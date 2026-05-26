"""Pipeline runner — sequences the six stages and persists the manifest.

`run_job(source, cfg, resume_from)` creates `runs/<ts>-<stem>/`, instantiates
each stage from the pipeline config, runs them in order, and writes
`manifest.json` after every transition.
"""
from __future__ import annotations

import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from patmir.pipeline.config import PipelineConfig
from patmir.pipeline.context import JobContext
from patmir.pipeline.manifest import Manifest
from patmir.pipeline.stage import STAGE_ORDER


def _make_job_dir(source: Path, runs_root: Path) -> tuple[Path, str]:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = f"{ts}-{source.stem}"
    job_dir = runs_root / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    return job_dir, job_id


def run_job(
    source: Path,
    cfg: PipelineConfig,
    *,
    runs_root: Path | None = None,
    resume_job_dir: Path | None = None,
    resume_from: str | None = None,
    source_lang: str | None = None,
) -> Manifest:
    runs_root = runs_root or Path.cwd() / "runs"

    if resume_job_dir is not None:
        job_dir = resume_job_dir
        manifest = Manifest.from_path(job_dir / "manifest.json")
        job_id = manifest.job_id
    else:
        job_dir, job_id = _make_job_dir(source, runs_root)
        manifest = Manifest(job_id=job_id, source_path=str(source))

    if source_lang:
        manifest.source_lang = source_lang

    manifest_path = job_dir / "manifest.json"
    manifest.to_path(manifest_path)

    ctx = JobContext(
        job_dir=job_dir,
        source_path=source,
        options=cfg.options,
        source_lang=manifest.source_lang,
    )

    start_idx = STAGE_ORDER.index(resume_from) if resume_from else 0

    # Stage execution closures — each takes the prior stage's output and returns the next.
    state: dict[str, object] = {}

    for i, name in enumerate(STAGE_ORDER):
        st = manifest.stage(name)
        if i < start_idx:
            # Stage skipped on resume — load its artifact paths from manifest into state.
            state[name] = _load_resume_artifact(name, job_dir, ctx)
            continue
        st.state = "running"
        st.started_at = time.time()
        manifest.to_path(manifest_path)
        ctx.progress(name, 0.0, "start")
        # Build this stage's instance JIT and release it after running so MPS
        # / CUDA only holds one stage's models in memory at a time.
        stage_obj = cfg.build_stage(name)
        try:
            output = _run_stage(name, stage_obj, state, ctx)
            state[name] = output
            _record_artifacts(name, output, st, job_dir)
            st.state = "done"
            st.finished_at = time.time()
            ctx.progress(name, 1.0, "done")
            if name == "asr" and output is not None:
                manifest.source_lang = ctx.source_lang
            manifest.to_path(manifest_path)
        except Exception as exc:
            st.state = "failed"
            st.finished_at = time.time()
            st.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            manifest.to_path(manifest_path)
            ctx.progress(name, 1.0, f"failed: {exc}")
            raise
        finally:
            stage_obj = None  # noqa — drop reference
            _free_accelerator_cache()

    return manifest


def _free_accelerator_cache() -> None:
    """Drop the just-finished stage's compute graph + free MPS/CUDA cache."""
    import gc

    gc.collect()
    try:
        import torch  # type: ignore[import-not-found]
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    try:
        import mlx.core as mx  # type: ignore[import-not-found]
        # MLX has no explicit cache to clear in 0.x; the gc above + the next
        # allocator pass usually reclaims everything via reference counting.
        # Calling metal.clear_cache() is the closest equivalent and is no-op
        # if unsupported on the current MLX build.
        try:
            mx.metal.clear_cache()  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError):
            pass
    except ImportError:
        pass


def _run_stage(name: str, stage_obj: Any, state: dict[str, object], ctx: JobContext):
    from patmir.pipeline.stage import AudioTrack, Diarization, SeparatedAudio

    if name == "ingest":
        return stage_obj.extract(ctx.source_path, ctx)
    if name == "separate":
        return stage_obj.separate(ctx.source_path, ctx)
    if name == "diarize":
        sep = state["separate"]
        return stage_obj.diarize(sep, ctx)  # type: ignore[arg-type]
    if name == "asr":
        # Transcribe the FULL vocal stem (not the VAD-masked speech stem) —
        # the hard mask was clipping soft-onset speech. Instead, we expose
        # the VAD intervals on the JobContext as `ctx.extra["vad_intervals"]`,
        # and the ASR module filters its own word timestamps against those
        # intervals after decoding. Hallucinated words from pure
        # scream/laugh zones get dropped; real speech with quiet onsets is
        # kept because ASR sees the full audio context.
        sep = state.get("separate")
        ingest = state["ingest"]  # type: ignore[assignment]
        audio_for_asr = ingest  # type: ignore[assignment]
        if isinstance(sep, SeparatedAudio):
            if sep.vocals is not None and sep.vocals.exists():
                audio_for_asr = AudioTrack(
                    path=sep.vocals,
                    sample_rate=sep.sample_rate,
                    duration=ingest.duration,  # type: ignore[attr-defined]
                )
            if sep.vad_intervals:
                ctx.extra["vad_intervals"] = sep.vad_intervals
        diar = state.get("diarize")
        if isinstance(diar, Diarization):
            ctx.extra["diarization"] = diar
        transcript = stage_obj.transcribe(audio_for_asr, ctx)
        if isinstance(diar, Diarization):
            from patmir.pipeline.stages.asr._sentence import write_transcript
            from patmir.pipeline.stages.asr.gap_filler import fill_gaps
            from patmir.pipeline.stages.asr.speaker_tag import (
                apply_speaker_tags,
                backfill_ref_text,
            )
            transcript = apply_speaker_tags(transcript, diar)
            # Second-pass ASR over diarization turns the first pass missed
            # (e.g. dialogue lost inside a long sustained-noise region).
            # Only run if the active ASR exposes a slice-transcription helper.
            transcribe_slice = getattr(stage_obj, "transcribe_slice", None)
            if callable(transcribe_slice) and isinstance(sep, SeparatedAudio):
                before = len(transcript.sentences)
                transcript = fill_gaps(
                    transcript,
                    diar,
                    sep,
                    asr_fn=transcribe_slice,
                    progress=lambda msg: ctx.progress("asr", 1.0, f"gap-fill {msg}"),
                )
                added = len(transcript.sentences) - before
                if added > 0:
                    transcript = apply_speaker_tags(transcript, diar)
            backfill_ref_text(diar, transcript)
            _rewrite_refs_json(ctx, diar)
            # Persist the speaker-tagged + re-indexed transcript (overrides the
            # candidate's first write that lacked speaker_id and didn't have splits).
            write_transcript(ctx.job_dir, transcript)
        return transcript
    if name == "translate":
        return stage_obj.translate(state["asr"], ctx)  # type: ignore[arg-type]
    if name == "tts":
        return stage_obj.synthesize(state["translate"], ctx)  # type: ignore[arg-type]
    if name == "mux":
        return stage_obj.mux(
            state["tts"],          # type: ignore[arg-type]
            state["ingest"],       # type: ignore[arg-type]
            state["separate"],     # type: ignore[arg-type]
            ctx,
        )
    if name == "output":
        return stage_obj.write(
            state["mux"],  # type: ignore[arg-type]
            ctx.source_path,
            state["translate"],  # type: ignore[arg-type]
            ctx,
        )
    raise ValueError(f"unknown stage: {name}")


def _record_artifacts(name: str, output: object, st, job_dir: Path) -> None:
    # Each stage's output dataclass exposes paths; record them in the manifest.
    if output is None:
        return
    for attr in ("path", "video", "subtitles"):
        v = getattr(output, attr, None)
        if isinstance(v, Path):
            st.artifacts[attr] = str(v.relative_to(job_dir) if v.is_relative_to(job_dir) else v)
    # Stages with list outputs (transcript, translated, synth) drop sidecar JSONs already.


def _rewrite_refs_json(ctx: JobContext, diar) -> None:
    """Persist the now-backfilled SpeakerRef.ref_text values to refs.json."""
    refs_json = ctx.job_dir / "diarize" / "refs.json"
    if not refs_json.parent.exists():
        return
    refs_json.write_text(json.dumps({
        spk: {
            "ref_audio": str(r.ref_audio),
            "ref_text": r.ref_text,
            "source_start": r.source_start,
            "source_end": r.source_end,
        }
        for spk, r in diar.refs.items()
    }, indent=2, ensure_ascii=False))


def _load_resume_artifact(name: str, job_dir: Path, ctx: JobContext):
    """Materialize a prior stage's output from disk so downstream stages can run."""
    from patmir.pipeline.stage import (
        AudioTrack,
        Diarization,
        MuxedAudio,
        OverlapInterval,
        SeparatedAudio,
        Sentence,
        SpeakerRef,
        SpeakerTurn,
        SynthesizedAudio,
        SynthesizedSentence,
        TranslatedSentence,
        TranslatedTranscript,
        Transcript,
        Word,
    )

    if name == "ingest":
        data = json.loads((job_dir / "audio.meta.json").read_text())
        return AudioTrack(path=Path(data["path"]), sample_rate=data["sample_rate"], duration=data["duration"])
    if name == "separate":
        data = json.loads((job_dir / "separated.meta.json").read_text())
        speech_v = data.get("speech_vocals")
        nonspeech_v = data.get("non_speech_vocals")
        intervals = data.get("vad_intervals")
        return SeparatedAudio(
            accompaniment=Path(data["accompaniment"]),
            vocals=Path(data["vocals"]),
            sample_rate=data["sample_rate"],
            channels=data["channels"],
            speech_vocals=Path(speech_v) if speech_v else None,
            non_speech_vocals=Path(nonspeech_v) if nonspeech_v else None,
            vad_intervals=[(float(s), float(e)) for s, e in intervals] if intervals else None,
        )
    if name == "diarize":
        turns_path = job_dir / "diarize" / "turns.json"
        refs_path = job_dir / "diarize" / "refs.json"
        if not turns_path.exists():
            return None
        td = json.loads(turns_path.read_text())
        turns = [SpeakerTurn(start=t["start"], end=t["end"], speaker_id=t["speaker_id"]) for t in td["turns"]]
        overlaps = [OverlapInterval(start=o["start"], end=o["end"]) for o in td.get("overlaps", [])]
        refs: dict[str, SpeakerRef] = {}
        if refs_path.exists():
            rd = json.loads(refs_path.read_text())
            for spk, r in rd.items():
                refs[spk] = SpeakerRef(
                    speaker_id=spk,
                    ref_audio=Path(r["ref_audio"]),
                    ref_text=r.get("ref_text", ""),
                    source_start=float(r["source_start"]),
                    source_end=float(r["source_end"]),
                )
        # Sample rate is recorded indirectly via separated.meta.json; default 16k if missing.
        sample_rate = 16000
        sep_meta_path = job_dir / "separated.meta.json"
        if sep_meta_path.exists():
            sample_rate = json.loads(sep_meta_path.read_text()).get("sample_rate", 16000)
        diar = Diarization(turns=turns, overlaps=overlaps, refs=refs, sample_rate=sample_rate)
        ctx.extra["diarization"] = diar
        return diar
    if name == "asr":
        data = json.loads((job_dir / "transcript.json").read_text())
        ctx.source_lang = data["language"]
        words = [Word(**w) for w in data["words"]]
        sentences = [
            Sentence(
                idx=s["idx"],
                text=s["text"],
                start=s["start"],
                end=s["end"],
                words=[Word(**w) for w in s["words"]],
                speaker_id=s.get("speaker_id"),
            )
            for s in data["sentences"]
        ]
        return Transcript(language=data["language"], words=words, sentences=sentences)
    if name == "translate":
        data = json.loads((job_dir / "translated.json").read_text())
        return TranslatedTranscript(
            source_lang=data["source_lang"],
            target_lang=data["target_lang"],
            sentences=[TranslatedSentence(**s) for s in data["sentences"]],
        )
    if name == "tts":
        data = json.loads((job_dir / "synthesized.json").read_text())
        return SynthesizedAudio(
            sample_rate=data["sample_rate"],
            sentences=[
                SynthesizedSentence(
                    idx=s["idx"],
                    wav_path=Path(s["wav_path"]),
                    wav_duration=s["wav_duration"],
                    source_start=s["source_start"],
                    source_end=s["source_end"],
                    target_text=s["target_text"],
                )
                for s in data["sentences"]
            ],
        )
    if name == "mux":
        data = json.loads((job_dir / "dub.meta.json").read_text())
        return MuxedAudio(path=Path(data["path"]), sample_rate=data["sample_rate"])
    return None
