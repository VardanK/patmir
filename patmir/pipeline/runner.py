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
    """Create a fresh `runs_root/<timestamp>-<stem>` dir.

    Job ids are second-granularity and the watch-along loop calls `run_job`
    once per section, so two runs can land in the same second — suffix on
    collision instead of failing the pipeline.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_id = f"{ts}-{source.stem}"
    for attempt in range(1, 100):
        job_id = base_id if attempt == 1 else f"{base_id}-{attempt}"
        job_dir = runs_root / job_id
        try:
            job_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return job_dir, job_id
    raise RuntimeError(f"Could not create a unique job dir under {runs_root} for {base_id}.")


def stage_cache_key(cfg: PipelineConfig, name: str) -> tuple[str, str]:
    """Warm-model cache key: `(stage, candidate_id)`.

    Keying on the stage name alone is not safe once one cache is shared between
    configs. `streaming_config()` overrides some stages (e.g. `separate` →
    `htdemucs_mlx_fast`), so batch and watch-along can select different
    candidates for the same stage — under a name-only key one mode would
    silently receive the other's model.
    """
    entry = cfg.entries.get(name) or {}
    return (name, str(entry.get("id", name)))


def prewarm_stage_cache(
    cfg: PipelineConfig,
    stage_cache: dict[Any, Any],
    *,
    stages: list[str] | None = None,
) -> dict[str, float]:
    """Build every active stage into `stage_cache` up front.

    Called from the server's startup hook so the first real submission doesn't
    pay model load — plus, on MLX, the one-time ~240s Metal-kernel JIT that
    otherwise lands inside the first job's TTS stage.

    Best-effort per stage: a stage that fails to build (missing API key, absent
    model) is skipped with its error recorded rather than taking the server down
    — the normal JIT build path will surface the same failure on first use.

    Returns `{stage: build_seconds}` for the stages it built (a stage already in
    the cache is skipped and omitted).
    """
    built: dict[str, float] = {}
    for name in (stages if stages is not None else list(STAGE_ORDER)):
        if name not in cfg.entries:
            continue
        key = stage_cache_key(cfg, name)
        if key in stage_cache:
            continue
        t0 = time.time()
        try:
            stage_cache[key] = cfg.build_stage(name)
        except Exception as exc:  # noqa: BLE001 — startup must not die here
            print(f"[prewarm] {name} skipped: {type(exc).__name__}: {exc}", flush=True)
            continue
        built[name] = time.time() - t0
    _free_accelerator_cache()
    return built


def run_job(
    source: Path,
    cfg: PipelineConfig,
    *,
    runs_root: Path | None = None,
    resume_job_dir: Path | None = None,
    resume_from: str | None = None,
    stop_after: str | None = None,
    source_lang: str | None = None,
    extra: dict[str, Any] | None = None,
    on_job_dir_created: Any = None,
    stage_cache: dict[str, Any] | None = None,
) -> Manifest:
    """Run the full pipeline.

    `on_job_dir_created`, when given, is invoked exactly once with
    `(job_dir, job_id)` immediately after the per-job directory exists —
    lets a wrapper (e.g. the FastAPI server) bind a JobRecord to the
    run_job_id without waiting for the whole pipeline to finish.

    `stop_after`, when given (a stage name in STAGE_ORDER), runs through that
    stage and then returns — used by the streaming prelude to run only the
    analysis half (ingest→…→asr, which includes the post-ASR speaker/emotion
    tagging + ref backfill) without translate/tts/mux/output.

    `stage_cache`, when given (a mutable dict owned by the caller), keeps each
    built stage instance warm across `run_job` calls instead of rebuilding and
    freeing it every time — the watch-along loop passes one cache for the whole
    stream, killing the measured 11–13% per-section model-rebuild tax
    (performance.md roadmap #2). Batch runs leave it None and keep the default
    one-model-resident-at-a-time behavior; with a cache, ALL stage models stay
    resident (~9 GB with the current active set — fine on 32 GB, watch this if
    models grow). Entries are keyed on `(stage, candidate_id)`, not stage name
    alone, so a cache shared between configs that select different candidates
    for the same stage (batch vs the watch-along overrides) can never hand back
    the wrong model. `_free_accelerator_cache` runs only when a stage was
    actually built — on a hit there is no build scratch to reclaim."""
    runs_root = runs_root or cfg.storage.runs_root

    if resume_job_dir is not None:
        job_dir = resume_job_dir
        manifest = Manifest.from_path(job_dir / "manifest.json")
        job_id = manifest.job_id
    else:
        job_dir, job_id = _make_job_dir(source, runs_root)
        manifest = Manifest(job_id=job_id, source_path=str(source))

    if on_job_dir_created is not None:
        try:
            on_job_dir_created(job_dir, job_id)
        except Exception:  # noqa: BLE001
            # The callback is informational; never let it crash the pipeline.
            pass

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
    if extra:
        # Seed cross-stage hand-offs (e.g. streaming voice_assignments) before
        # any stage runs. See JobContext.extra.
        ctx.extra.update(extra)

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
        # / CUDA only holds one stage's models in memory at a time. Time the
        # build separately (model load / quantization) so the benchmark can
        # attribute it — it's the per-section rebuild cost in watch-along.
        # With a caller-owned `stage_cache` the instance is kept warm instead
        # (build_s = 0 on hits), trading memory for the rebuild tax.
        key = stage_cache_key(cfg, name)
        cache_hit = stage_cache is not None and key in stage_cache
        if cache_hit:
            stage_obj = stage_cache[key]
            st.build_s = 0.0
        else:
            _build_t0 = time.time()
            stage_obj = cfg.build_stage(name)
            st.build_s = time.time() - _build_t0
            if stage_cache is not None:
                stage_cache[key] = stage_obj
        try:
            output = _run_stage(name, stage_obj, state, ctx)
            state[name] = output
            _record_artifacts(name, output, st, job_dir)
            st.state = "done"
            st.finished_at = time.time()
            ctx.progress(name, 1.0, "done")
            if name == "asr" and output is not None:
                manifest.source_lang = ctx.source_lang
            if name == "output" and cfg.storage.library_root is not None:
                manifest.published_dir = _publish(job_dir, job_id, cfg, ctx)
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
            # ALWAYS reclaim, including on a cache hit. Skipping it on hits was
            # tried 2026-08-23 on the theory that a hit has no build scratch to
            # drop; watch-along regressed from 1.71 to 3.02 agg RTF with every
            # stage ~65-72% slower — the signature of a uniform system-level
            # stall, and vm_stat showed heavy swapping. MLX's Metal buffers live
            # in unified memory but sit largely outside RSS (peak RSS read only
            # 10.1 GB, which is why RSS was a misleading instrument here), so the
            # accumulating scratch is invisible to process accounting while still
            # driving the machine into swap. Do not re-introduce the gate without
            # a memory instrument that actually sees Metal allocations.
            _free_accelerator_cache()

        if stop_after and name == stop_after:
            break

    return manifest


def _publish(job_dir: Path, job_id: str, cfg: PipelineConfig, ctx: JobContext) -> str | None:
    """Copy the deliverables to `[storage].library_root/<job_id>/`.

    Never fails the run: the dub is already complete in the job dir at this
    point, so a publish error (share not mounted, out of space) is reported and
    swallowed rather than throwing away a finished pipeline.
    """
    from patmir.pipeline.publish import publish_job

    library_root = cfg.storage.library_root
    assert library_root is not None
    try:
        dest = publish_job(job_dir, job_id, library_root)
    except (OSError, FileNotFoundError) as exc:
        ctx.progress("output", 1.0, f"publish failed: {exc}")
        return None
    ctx.progress("output", 1.0, f"published to {dest}")
    return str(dest)


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
        # MLX cache-clear API moved from `mx.metal.clear_cache` (deprecated
        # in newer MLX) to `mx.clear_cache`. Try the new one first, fall
        # back to the old one, no-op if neither is available.
        clear_cache = getattr(mx, "clear_cache", None)
        if clear_cache is None:
            clear_cache = getattr(getattr(mx, "metal", None), "clear_cache", None)
        if callable(clear_cache):
            try:
                clear_cache()
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
    if name == "emotion":
        # Detect per-turn emotion on the source vocal stem, then populate
        # `Diarization.refs` here in the runner via `ref_picker.pick_refs` —
        # the diarize stage couldn't do this earlier because it didn't know
        # each turn's emotion bin yet. See ref_picker.pick_refs docstring.
        sep = state["separate"]
        diar = state["diarize"]
        if not isinstance(sep, SeparatedAudio) or not isinstance(diar, Diarization):
            return []
        predictions = stage_obj.detect(sep, diar, ctx)  # type: ignore[arg-type]
        ctx.extra["emotion_predictions"] = predictions

        # Now slice per-(speaker, emotion) reference clips.
        from patmir.pipeline.stages.diarize.ref_picker import pick_refs
        refs_dir = ctx.job_dir / "diarize" / "refs"
        speech_for_refs = sep.speech_vocals or sep.vocals
        diar.refs = pick_refs(
            turns=diar.turns,
            overlaps=diar.overlaps,
            predictions=predictions,
            speech_vocals=speech_for_refs,
            refs_dir=refs_dir,
            min_ref_seconds=ctx.options.min_ref_seconds,
            min_rms_db=getattr(ctx.options, "ref_min_rms_db", -33.0),
        )
        # Persist the now-filled refs (text empty for now — backfilled post-ASR).
        _write_refs_json(ctx, diar)
        # Rewrite turns.json so each turn carries its assigned emotion bin.
        _write_turns_json(ctx, diar)
        return predictions
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
            from patmir.pipeline.stages.asr.emotion_tag import apply_emotion_tags
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
            # Tag each sentence with its dominant emotion bin from the
            # per-turn predictions emitted by the emotion stage. No-op when
            # the emotion stage didn't run (e.g. resume from a pre-Phase-2.x
            # job).
            predictions = ctx.extra.get("emotion_predictions")
            if predictions:
                transcript = apply_emotion_tags(transcript, predictions)
            backfill_ref_text(diar, transcript)
            _write_refs_json(ctx, diar)
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


_REFS_KEY_SEP = "__"


def _refs_key(speaker_id: str, emotion: str) -> str:
    """Encode a (speaker, emotion) tuple as a JSON-portable string key.

    pyannote's speaker labels are `SPEAKER_00`-style and don't contain `__`,
    so the separator is safe.
    """
    return f"{speaker_id}{_REFS_KEY_SEP}{emotion}"


def _parse_refs_key(key: str) -> tuple[str, str]:
    """Inverse of `_refs_key`. Falls back to a `(key, "neutral")` tuple for
    legacy single-speaker keys produced by pre-Phase-2.x runs."""
    if _REFS_KEY_SEP in key:
        spk, emo = key.rsplit(_REFS_KEY_SEP, 1)
        return spk, emo
    return key, "neutral"


def _write_turns_json(ctx: JobContext, diar) -> None:
    """Rewrite `diarize/turns.json` so each turn carries its emotion bin.

    The diarize stage writes a first version without emotion; the runner
    overwrites it after the emotion stage tags every turn.
    """
    turns_json = ctx.job_dir / "diarize" / "turns.json"
    if not turns_json.parent.exists():
        return
    existing = {}
    if turns_json.exists():
        try:
            existing = json.loads(turns_json.read_text())
        except json.JSONDecodeError:
            existing = {}
    existing["turns"] = [
        {"start": t.start, "end": t.end, "speaker_id": t.speaker_id, "emotion": t.emotion}
        for t in diar.turns
    ]
    existing.setdefault("overlaps", [
        {"start": o.start, "end": o.end} for o in diar.overlaps
    ])
    turns_json.write_text(json.dumps(existing, indent=2, ensure_ascii=False))


def _write_refs_json(ctx: JobContext, diar) -> None:
    """Persist `Diarization.refs` to `diarize/refs.json`.

    Keys are encoded as `"speaker__emotion"`. Called twice per job:
    once from the `emotion` stage (right after `pick_refs` populates the
    refs map; `ref_text` is empty), and once from the `asr` branch (after
    `backfill_ref_text` fills in the text and prunes non-lexical refs).
    """
    refs_json = ctx.job_dir / "diarize" / "refs.json"
    if not refs_json.parent.exists():
        return
    refs_json.write_text(json.dumps({
        _refs_key(spk, emo): {
            "speaker_id": spk,
            "emotion": emo,
            "ref_audio": str(r.ref_audio),
            "ref_text": r.ref_text,
            "source_start": r.source_start,
            "source_end": r.source_end,
        }
        for (spk, emo), r in diar.refs.items()
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
        turns = [
            SpeakerTurn(
                start=t["start"], end=t["end"],
                speaker_id=t["speaker_id"],
                emotion=t.get("emotion"),
            )
            for t in td["turns"]
        ]
        overlaps = [OverlapInterval(start=o["start"], end=o["end"]) for o in td.get("overlaps", [])]
        refs: dict[tuple[str, str], SpeakerRef] = {}
        if refs_path.exists():
            rd = json.loads(refs_path.read_text())
            for key, r in rd.items():
                spk = r.get("speaker_id") or _parse_refs_key(key)[0]
                emo = r.get("emotion") or _parse_refs_key(key)[1]
                refs[(spk, emo)] = SpeakerRef(  # type: ignore[index]
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
    if name == "emotion":
        seg_path = job_dir / "emotion" / "segments.json"
        if not seg_path.exists():
            return []
        from patmir.pipeline.stage import EmotionPrediction
        data = json.loads(seg_path.read_text())
        predictions = [
            EmotionPrediction(
                start=float(p["start"]),
                end=float(p["end"]),
                emotion=p["emotion"],
                confidence=float(p.get("confidence", 0.0)),
            )
            for p in data.get("predictions", [])
        ]
        ctx.extra["emotion_predictions"] = predictions
        return predictions
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
                emotion=s.get("emotion"),
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
