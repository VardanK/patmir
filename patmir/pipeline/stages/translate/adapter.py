"""Translate stage adapter.

Wraps a Phase 0 translator candidate (loaded from `[[eval.translation.candidates]]`)
with the pipeline's typed `TranslateStage` interface. The inner candidate
exposes `translate(text, source_lang) -> str`; the adapter iterates sentences
from the ASR transcript, calls it per sentence, attaches Armenian text + source
timing to each entry, and writes `translated.json`.
"""
from __future__ import annotations

import importlib
import inspect
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from patmir.pipeline.context import (
    NEUTRAL_AUDIENCE,
    JobContext,
    resolve_audience,
)

log = logging.getLogger(__name__)


def _accepts(fn: Any, name: str) -> bool:
    """Does `fn` take a keyword named `name`?

    The pipeline's way of discovering optional candidate capabilities: the
    candidates are Phase-0 objects with no shared base class, so the adapter
    asks the signature rather than a registry.
    """
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _warn_unsteerable(ctx: JobContext, inner: Any, audience: str) -> None:
    """A non-adult level was asked for and the candidate has no prompt surface.

    Warned, never silent, and never fatal. Not fatal because refusing would take
    out the offline fallback entirely — NLLB / MADLAD / OPUS-MT are seq2seq and
    TranslateGemma documents having no prompt surface at all, so on those the
    guardrail genuinely cannot apply. Not silent because this repo has a scar
    here: `emotion_guidance` sat INERT for two months for exactly this reason,
    and the NOTE at pipeline/context.py explicitly says a restored version must
    warn when the active candidate cannot accept it.
    """
    cid = getattr(inner, "id", type(inner).__name__)
    msg = (
        f"audience={audience} requested, but translate candidate {cid!r} takes "
        f"no prompt — register is NOT steered for this run"
    )
    log.warning("%s", msg)
    ctx.progress("translate", 0.0, msg)
from patmir.pipeline.stage import (
    TranslatedSentence,
    TranslatedTranscript,
    Transcript,
)


class SentenceTranslator:
    id = "translate_adapter"

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def translate(self, transcript: Transcript, ctx: JobContext) -> TranslatedTranscript:
        source_lang = ctx.source_lang or transcript.language
        total = len(transcript.sentences) or 1
        # Resolved once, here, from the config the runner was handed. The server
        # derives a per-job config carrying the submission's level (see
        # JobManager._cfg_for), so by this point `ctx.options.audience` IS the
        # level this job's cache key was computed with.
        audience = resolve_audience(None, ctx.options)
        steered = False

        # If the inner candidate exposes a batch API, prefer it: chunked
        # calls produce more coherent character voice + better discourse
        # flow, and trade O(N) round-trips for O(N/chunk_size). Otherwise
        # fall back to the per-sentence loop with optional emotion kwarg.
        translate_batch = getattr(self.inner, "translate_batch", None)
        if callable(translate_batch) and transcript.sentences:
            def _on_progress(done: int, total_: int, msg: str) -> None:
                ctx.progress("translate", done / max(1, total_), msg)
            batch_kwargs: dict[str, Any] = {}
            if _accepts(translate_batch, "audience"):
                batch_kwargs["audience"] = audience
                steered = True
            elif audience != NEUTRAL_AUDIENCE:
                _warn_unsteerable(ctx, self.inner, audience)
            import time as _time
            _b0 = _time.perf_counter()
            results = translate_batch(
                transcript.sentences,
                source_lang=source_lang,
                progress=_on_progress,
                **batch_kwargs,
            )
            _batch_wall = _time.perf_counter() - _b0
            # The batch path wrote NO timings at all, so watch-along translate —
            # which forces `openrouter_gemini_flash_batch` — was completely
            # un-instrumented and `BenchResult.translate` was hard-coded None
            # for that whole mode. One `kind: "chunk"` row, mirroring what the
            # TTS adapter does for batched decode: a per-sentence latency does
            # not exist once a round-trip is shared.
            if ctx.extra.get("bench") and transcript.sentences:
                tdir = ctx.job_dir / "translate"
                tdir.mkdir(exist_ok=True)
                with (tdir / "timings.jsonl").open("w") as f:
                    f.write(json.dumps({
                        "kind": "chunk",
                        "sentences": len(transcript.sentences),
                        "elapsed_s": round(_batch_wall, 4),
                        "latency_ms": round(_batch_wall * 1000, 1),
                    }, ensure_ascii=False) + "\n")
            sentences = [
                TranslatedSentence(
                    idx=s.idx,
                    source_text=s.text,
                    target_text=(results.get(s.idx) or "").strip(),
                    start=s.start,
                    end=s.end,
                    speaker_id=s.speaker_id,
                    emotion=s.emotion,
                )
                for s in transcript.sentences
            ]
        else:
            # Per-sentence path. Sentences are independent (each call carries
            # only its own text + emotion), so cloud candidates that set
            # `translate_concurrency` > 1 fire them concurrently — the output
            # is byte-identical to the serial loop, only the wall time changes
            # (each round-trip is ~1s TTFT-dominated; serial translate was
            # 12-15% of batch wall — performance.md roadmap #1). Local
            # candidates (NLLB on GPU) leave the attribute unset → serial.
            import time
            inner_accepts_emotion = _accepts(self.inner.translate, "emotion")
            inner_accepts_audience = _accepts(self.inner.translate, "audience")
            steered = inner_accepts_audience
            if audience != NEUTRAL_AUDIENCE and not inner_accepts_audience:
                _warn_unsteerable(ctx, self.inner, audience)

            bench = bool(ctx.extra.get("bench"))
            src = transcript.sentences
            results: list[str] = [""] * len(src)
            elapsed: list[float] = [0.0] * len(src)

            def _one(i: int) -> None:
                s = src[i]
                kwargs: dict[str, Any] = {"source_lang": source_lang}
                if inner_accepts_emotion:
                    kwargs["emotion"] = s.emotion
                if inner_accepts_audience:
                    kwargs["audience"] = audience
                t0 = time.perf_counter()
                results[i] = self.inner.translate(s.text, **kwargs)
                elapsed[i] = time.perf_counter() - t0

            concurrency = max(1, int(getattr(self.inner, "translate_concurrency", 1)))
            if concurrency > 1 and len(src) > 1:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=concurrency) as ex:
                    futs = {ex.submit(_one, i): i for i in range(len(src))}
                    done = 0
                    for fut in as_completed(futs):
                        fut.result()  # re-raise any translate error
                        done += 1
                        ctx.progress("translate", done / total, f"sentence {done}/{total}")
            else:
                for i in range(len(src)):
                    _one(i)
                    ctx.progress("translate", (i + 1) / total, f"sentence {i + 1}/{total}")

            sentences = [
                TranslatedSentence(
                    idx=s.idx,
                    source_text=s.text,
                    target_text=results[i],
                    start=s.start,
                    end=s.end,
                    speaker_id=s.speaker_id,
                    emotion=s.emotion,
                )
                for i, s in enumerate(src)
            ]
            # Per-sentence latency trace — only under `patmir bench`.
            if bench and src:
                tdir = ctx.job_dir / "translate"
                tdir.mkdir(exist_ok=True)
                with (tdir / "timings.jsonl").open("w") as f:
                    for i, s in enumerate(src):
                        f.write(json.dumps(
                            {"idx": s.idx, "latency_ms": round(elapsed[i] * 1000, 1)},
                            ensure_ascii=False,
                        ) + "\n")

        out = TranslatedTranscript(
            source_lang=source_lang,
            target_lang="hy",
            sentences=sentences,
        )

        # A blank line is the failure mode this guardrail exists to prevent, and
        # it is the one a safety filter produces: a provider that declines a
        # violent line under `audience="child"` returns `content: null`, which
        # the candidate turns into "" and the TTS adapter then skips — a silent
        # hole in the dub. The batch path already catches this per chunk
        # ("empty translation for idx=" -> per-sentence fallback); the
        # per-sentence path had nothing. Count and report; never fail the run,
        # because the rest of the dub is good and the operator needs to know
        # which lines to look at.
        blank = sum(1 for s in sentences if not (s.target_text or "").strip())
        if blank:
            msg = (
                f"{blank} of {len(sentences)} line(s) came back empty from the "
                f"translator (audience={audience}) — those lines will be silent"
            )
            log.warning("%s", msg)
            ctx.progress("translate", 1.0, msg)

        _write_translated(ctx.job_dir, out, audience=audience, honored=steered)
        return out


def _write_translated(
    job_dir: Path,
    t: TranslatedTranscript,
    *,
    audience: str = NEUTRAL_AUDIENCE,
    honored: bool = False,
) -> None:
    # `audience` / `audience_honored` are additive and both absent from files
    # written before the guardrail existed — readers must treat them as
    # optional. This file is copied into the cache as `transcript.json`, so a
    # cached dub is self-describing about the register it was made at.
    payload = {
        "source_lang": t.source_lang,
        "target_lang": t.target_lang,
        "audience": audience,
        "audience_honored": honored,
        "sentences": [asdict(s) for s in t.sentences],
    }
    (job_dir / "translated.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def build(config: dict[str, Any]) -> SentenceTranslator:
    """Build the adapter directly from an inner candidate config.

    The pipeline config-loader normally instantiates the inner Phase-0
    candidate and wraps it via `SentenceTranslator(inner=...)` itself; this
    `build()` exists so the adapter is round-trippable through the standard
    candidate-loader pattern.
    """
    module = importlib.import_module(config["module"])
    inner = module.build(config)
    if isinstance(inner, SentenceTranslator):
        return inner
    return SentenceTranslator(inner=inner)
