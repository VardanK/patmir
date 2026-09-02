"""Intra-stage timing probes, active only under `patmir bench`.

Per-stage walls come from the manifest and say nothing about where a stage's
own time goes. The intra-TTS numbers that showed the decode step running 4-5x
off its memory-bandwidth bound came from throwaway instrumentation that was
then reverted — leaving `runs/*/tts/bench_probe.json` on disk with no code left
in the tree to regenerate it. This is that instrumentation, kept.

Three constraints shape the design:

* **Zero cost when disabled.** Candidates call `probe.section(...)`
  unconditionally on the hot path; a disabled probe hands back a shared no-op
  context manager and never imports or touches torch.
* **Sync inside a span, and only when enabled.** A GPU span timed without
  `torch.cuda.synchronize()` measures kernel *enqueue*, not kernel time — the
  reason `_decode_chunk`'s stage wall looked honest (a per-row `.cpu()` forced
  a sync every step) while everything after Demucs did not. Syncing serializes
  CPU and GPU, so it is exactly why this must stay off outside a benchmark.
* **Additive output.** `dump()` writes whatever spans and counters the stage
  happened to fill in, so adding a span needs no reader change: `bench.py`
  merges the file verbatim under `notes["probes"][<stage>]`.

Span names carry their own unit suffix (`model_s`, `probe_s`) so the JSON keys
are self-describing and the six keys the old TTS probe wrote
(`prefill_s, forward_s, sample_s, codec_s, steps, chunks`) are reproduced
exactly, keeping the 2026-09-01 files comparable.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from time import perf_counter
from typing import Any

_NOOP = nullcontext()


def _cuda_sync() -> None:
    """Drain the device queue so the enclosing span measures kernel time.

    Guarded on `is_initialized()` rather than `is_available()`: the latter
    initializes a context as a side effect, which on a CPU-only run would make
    an instrumentation call allocate a GPU.
    """
    try:
        import torch  # type: ignore[import-not-found]
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


class Probe:
    """Accumulates named spans and counters for one stage of one job.

    A stage candidate owns one of these for its lifetime (it lives in the warm
    `stage_cache`), so `start()` resets it per job and `dump()` writes it out.
    """

    __slots__ = ("enabled", "level", "trace_path", "spans", "counters",
                 "values", "details")

    def __init__(self) -> None:
        self.enabled = False
        self.level = "chunk"
        self.trace_path: Path | None = None
        self.spans: dict[str, float] = {}
        self.counters: dict[str, int] = {}
        self.values: dict[str, Any] = {}
        self.details: list[dict[str, Any]] = []

    # -- lifecycle -- #
    def start(self, *, level: str = "chunk", trace_path: Path | None = None) -> None:
        self.enabled = True
        self.level = level
        self.trace_path = trace_path
        self.reset()

    def reset(self) -> None:
        self.spans = {}
        self.counters = {}
        self.values = {}
        self.details = []

    def stop(self) -> None:
        self.enabled = False
        self.trace_path = None

    # -- recording -- #
    def section(self, name: str):
        """Time a span, syncing the device on both edges. No-op when disabled."""
        if not self.enabled:
            return _NOOP
        return self._timed(name)

    def step_section(self, name: str):
        """A span fine-grained enough that timing it changes the measurement.

        Only active at `level="step"`: per-step syncs serialize CPU and GPU and
        inflate the very total they are meant to explain, so the split between
        (say) forward and sample is opt-in and the chunk-level total stays the
        number to trust.
        """
        if not self.enabled or self.level != "step":
            return _NOOP
        return self._timed(name)

    @contextmanager
    def _timed(self, name: str) -> Iterator[None]:
        _cuda_sync()
        t0 = perf_counter()
        try:
            yield
        finally:
            _cuda_sync()
            self.spans[name] = self.spans.get(name, 0.0) + (perf_counter() - t0)

    def bump(self, name: str, n: int = 1) -> None:
        if self.enabled:
            self.counters[name] = self.counters.get(name, 0) + n

    def set(self, name: str, value: Any) -> None:
        if self.enabled:
            self.values[name] = value

    def detail(self, row: dict[str, Any]) -> None:
        """One row per unit of work (a decode chunk, an ASR batch)."""
        if self.enabled:
            self.details.append(row)

    # -- output -- #
    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {k: round(v, 4) for k, v in self.spans.items()}
        out.update(self.counters)
        out.update(self.values)
        if self.details:
            out["details"] = self.details
        return out

    def dump(self, path: Path) -> None:
        """Write `<stage>/bench_probe.json`, MERGING with what is already there.

        Merge rather than overwrite because a stage can have two instruments
        pointed at it: the candidate's own probe (TTS owns one, because its
        decode loop never sees a `JobContext`) and the job probe the runner
        resets and dumps per stage, which is also where the runner's own
        post-stage steps are timed. Their key sets are disjoint, so an update
        is the whole merge.

        Never raises — an instrumentation write must not fail a run that has
        otherwise succeeded.
        """
        if not self.enabled:
            return
        payload = self.summary()
        if not payload:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                try:
                    prior = json.loads(path.read_text())
                    if isinstance(prior, dict):
                        prior.update(payload)
                        payload = prior
                except (OSError, json.JSONDecodeError):
                    pass
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        except OSError:
            pass


def probe_from_ctx(ctx: Any, probe: Probe, *, stage: str) -> None:
    """Arm `probe` if this job is a benchmark. Reads `ctx.extra["bench"]`, the
    same flag that already gates the tts/translate JSONL traces, plus optional
    `<stage>_probe` (level) and `<stage>_trace` keys."""
    extra = getattr(ctx, "extra", None) or {}
    if not extra.get("bench"):
        probe.stop()
        return
    level = str(extra.get(f"{stage}_probe") or "chunk")
    trace = None
    if extra.get(f"{stage}_trace"):
        trace = Path(getattr(ctx, "job_dir", ".")) / stage / f"trace_{stage}.json"
    probe.start(level=level, trace_path=trace)
