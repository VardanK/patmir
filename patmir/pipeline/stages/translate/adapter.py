"""Translate stage adapter.

Wraps a Phase 0 translator candidate (loaded from `[[eval.translation.candidates]]`)
with the pipeline's typed `TranslateStage` interface. The inner candidate
exposes `translate(text, source_lang) -> str`; the adapter iterates sentences
from the ASR transcript, calls it per sentence, attaches Armenian text + source
timing to each entry, and writes `translated.json`.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from patmir.pipeline.context import JobContext
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

        # If the inner candidate exposes a batch API, prefer it: chunked
        # calls produce more coherent character voice + better discourse
        # flow, and trade O(N) round-trips for O(N/chunk_size). Otherwise
        # fall back to the per-sentence loop with optional emotion kwarg.
        translate_batch = getattr(self.inner, "translate_batch", None)
        if callable(translate_batch) and transcript.sentences:
            def _on_progress(done: int, total_: int, msg: str) -> None:
                ctx.progress("translate", done / max(1, total_), msg)
            results = translate_batch(
                transcript.sentences,
                source_lang=source_lang,
                progress=_on_progress,
            )
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
            import inspect
            import time
            try:
                inner_sig = inspect.signature(self.inner.translate)
                inner_accepts_emotion = "emotion" in inner_sig.parameters
            except (TypeError, ValueError):
                inner_accepts_emotion = False

            bench = bool(ctx.extra.get("bench"))
            src = transcript.sentences
            results: list[str] = [""] * len(src)
            elapsed: list[float] = [0.0] * len(src)

            def _one(i: int) -> None:
                s = src[i]
                t0 = time.perf_counter()
                if inner_accepts_emotion:
                    results[i] = self.inner.translate(
                        s.text, source_lang=source_lang, emotion=s.emotion,
                    )
                else:
                    results[i] = self.inner.translate(s.text, source_lang=source_lang)
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
        _write_translated(ctx.job_dir, out)
        return out


def _write_translated(job_dir: Path, t: TranslatedTranscript) -> None:
    payload = {
        "source_lang": t.source_lang,
        "target_lang": t.target_lang,
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
