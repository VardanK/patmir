"""Prewarm the watch-along models that `prewarm_stage_cache` cannot reach.

`prewarm_stage_cache` walks `cfg.entries`, so it only ever builds the nine
pipeline STAGES. Watch-along additionally uses two models that are not stages
at all — the speaker embedder and the age/gender profiler — and they are
`lru_cache`d module functions loaded lazily on first use, which means **inside
the prelude**, on the critical path to first audio.

Measured on the author's 2026-08-25 session: the first prelude showed a
**14.3 s gap between ASR finishing and translate starting** that no stage owned,
no progress event covered, and no manifest recorded. It did not recur on the
second prelude of the same server, which is what identified it as a one-time
lazy load rather than per-run work. It was ~20% of time-to-first-audio and
completely invisible to per-stage telemetry.

Both loaders swallow their own failures and return None (the callers degrade to
no cross-section matching / a pitch heuristic), so calling them early cannot
turn a working server into a broken one — the worst case is that the same
failure happens at boot instead of mid-prelude, which is strictly better.
"""
from __future__ import annotations

import time
from pathlib import Path

from patmir.pipeline.config import PipelineConfig


def prewarm_streaming_models() -> dict[str, float]:
    """Load the non-stage watch-along models. Returns {name: seconds}.

    Best-effort and never raises: a prewarm failure must not stop the server
    from starting, exactly as `prewarm_stage_cache` does not.
    """
    from patmir.pipeline.streaming import embeddings, speaker_profile

    built: dict[str, float] = {}

    def _timed(name: str, fn) -> None:
        t0 = time.time()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — startup must not die here
            print(f"[prewarm] {name} skipped: {type(exc).__name__}: {exc}", flush=True)
            return
        built[name] = time.time() - t0

    # Whichever embedder is actually selected — prewarming the other would load
    # a model this process will never use.
    if embeddings.backend() == "redimnet":
        _timed("embed:redimnet", embeddings._redimnet)
    else:
        _timed("embed:ecapa", embeddings._encoder)
    _timed("speaker_profile", speaker_profile._load_model)
    return built


def streaming_prewarm_config(config_path: Path) -> PipelineConfig | None:
    """The watch-along stage config, for prewarming the candidates that differ
    from batch.

    The stage cache is keyed `(stage, candidate_id)`, and `streaming_config()`
    overrides two of them — `separate` (`htdemucs_fast` vs batch's
    `htdemucs_vocals`) and `translate` (the batched OpenRouter candidate). So
    prewarming with the batch config alone leaves both as MISSES, and the first
    stream of every server lifetime builds a Demucs model inside its prelude.

    Returns None if the config cannot be built (e.g. an unknown profile), since
    a prewarm is never worth failing startup over.
    """
    try:
        from patmir.pipeline.streaming.runner import streaming_config

        return streaming_config(config_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[prewarm] streaming config unavailable: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return None
