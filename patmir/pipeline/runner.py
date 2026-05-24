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
) -> Manifest:
    runs_root = runs_root or Path.cwd() / "runs"

    if resume_job_dir is not None:
        job_dir = resume_job_dir
        manifest = Manifest.from_path(job_dir / "manifest.json")
        job_id = manifest.job_id
    else:
        job_dir, job_id = _make_job_dir(source, runs_root)
        manifest = Manifest(job_id=job_id, source_path=str(source))

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
        try:
            output = _run_stage(name, cfg, state, ctx)
            state[name] = output
            _record_artifacts(name, output, st, job_dir)
            st.state = "done"
            st.finished_at = time.time()
            ctx.progress(name, 1.0, "done")
            if name == "asr" and output is not None:
                # ASR sets language
                manifest.source_lang = ctx.source_lang
            manifest.to_path(manifest_path)
        except Exception as exc:
            st.state = "failed"
            st.finished_at = time.time()
            st.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            manifest.to_path(manifest_path)
            ctx.progress(name, 1.0, f"failed: {exc}")
            raise

    return manifest


def _run_stage(name: str, cfg: PipelineConfig, state: dict[str, object], ctx: JobContext):
    if name == "ingest":
        return cfg.ingest.extract(ctx.source_path, ctx)
    if name == "asr":
        out = cfg.asr.transcribe(state["ingest"], ctx)  # type: ignore[arg-type]
        # ASR stage sets ctx.source_lang internally
        return out
    if name == "translate":
        return cfg.translate.translate(state["asr"], ctx)  # type: ignore[arg-type]
    if name == "tts":
        return cfg.tts.synthesize(state["translate"], ctx)  # type: ignore[arg-type]
    if name == "mux":
        return cfg.mux.mux(state["tts"], state["ingest"], ctx)  # type: ignore[arg-type]
    if name == "output":
        return cfg.output.write(
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


def _load_resume_artifact(name: str, job_dir: Path, ctx: JobContext):
    """Materialize a prior stage's output from disk so downstream stages can run."""
    from patmir.pipeline.stage import (
        AudioTrack,
        MuxedAudio,
        Sentence,
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
    if name == "asr":
        data = json.loads((job_dir / "transcript.json").read_text())
        ctx.source_lang = data["language"]
        words = [Word(**w) for w in data["words"]]
        sentences = [
            Sentence(idx=s["idx"], text=s["text"], start=s["start"], end=s["end"], words=[Word(**w) for w in s["words"]])
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
