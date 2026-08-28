"""Tests for the warm-model stage cache (`stage_cache_key` + `prewarm_stage_cache`).

Three contracts:
  * The cache key is `(stage, candidate_id)`, not the stage name alone. One
    cache is shared process-wide by batch jobs and streams, and
    `streaming_config()` overrides some stages — under a name-only key a mode
    could be handed the other's model.
  * Configs that select the SAME candidate for a stage share one key, so a
    single instance serves both modes. That is what makes the 2026-08-23 TTS
    unification worth doing on a 24 GB GPU.
  * `prewarm_stage_cache` fills the cache up front, is idempotent, and is
    best-effort: a stage that cannot be built is skipped rather than taking
    server startup down with it.
  * The audience age level, which is a per-job value on `options`, does NOT
    enter the key — the design depends on it.
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass, field

from patmir.pipeline.config import PipelineConfig
from patmir.pipeline.context import PipelineOptions
from patmir.pipeline.runner import prewarm_stage_cache, stage_cache_key
from patmir.pipeline.stage import STAGE_ORDER


@dataclass
class _StubConfig(PipelineConfig):
    """PipelineConfig that fabricates stage instances instead of importing."""

    fail: tuple[str, ...] = ()
    builds: list[str] = field(default_factory=list)

    def build_stage(self, name: str):  # type: ignore[override]
        self.builds.append(name)
        if name in self.fail:
            raise RuntimeError(f"cannot build {name}")
        return f"<{name}:{self.entries[name]['id']}>"


def _cfg(overrides: dict[str, str] | None = None, **kw) -> _StubConfig:
    entries = {s: {"id": f"stub_{s}", "module": f"mod.{s}"} for s in STAGE_ORDER}
    for stage, cid in (overrides or {}).items():
        entries[stage] = {"id": cid, "module": f"mod.{stage}"}
    return _StubConfig(entries=entries, options=PipelineOptions(), **kw)


class CacheKey(unittest.TestCase):
    def test_key_is_stage_and_candidate_id(self):
        cfg = _cfg()
        self.assertEqual(stage_cache_key(cfg, "tts"), ("tts", "stub_tts"))

    def test_differing_candidates_do_not_collide(self):
        batch = _cfg()
        stream = _cfg({"separate": "htdemucs_mlx_fast"})
        self.assertNotEqual(
            stage_cache_key(batch, "separate"), stage_cache_key(stream, "separate")
        )

    def test_same_candidate_shares_one_key(self):
        # The TTS unification: both modes select higgs_v3_mlx_q8, so one 4B
        # instance serves both instead of two.
        batch = _cfg({"tts": "higgs_v3_mlx_q8"})
        stream = _cfg({"tts": "higgs_v3_mlx_q8", "separate": "htdemucs_mlx_fast"})
        self.assertEqual(stage_cache_key(batch, "tts"), stage_cache_key(stream, "tts"))

    def test_key_is_hashable_and_usable_as_dict_key(self):
        cfg = _cfg()
        cache = {stage_cache_key(cfg, "tts"): object()}
        self.assertIn(stage_cache_key(cfg, "tts"), cache)

    def test_missing_entry_falls_back_to_stage_name(self):
        cfg = _cfg()
        del cfg.entries["tts"]
        self.assertEqual(stage_cache_key(cfg, "tts"), ("tts", "tts"))


class Prewarm(unittest.TestCase):
    def test_builds_every_stage_into_the_cache(self):
        cfg, cache = _cfg(), {}
        built = prewarm_stage_cache(cfg, cache)
        self.assertEqual(set(built), set(STAGE_ORDER))
        self.assertEqual(len(cache), len(STAGE_ORDER))
        self.assertEqual(cache[stage_cache_key(cfg, "tts")], "<tts:stub_tts>")

    def test_is_idempotent(self):
        cfg, cache = _cfg(), {}
        prewarm_stage_cache(cfg, cache)
        cfg.builds.clear()
        self.assertEqual(prewarm_stage_cache(cfg, cache), {})
        self.assertEqual(cfg.builds, [], "already-cached stages must not rebuild")

    def test_a_failing_stage_is_skipped_not_raised(self):
        cfg, cache = _cfg(fail=("translate",)), {}
        built = prewarm_stage_cache(cfg, cache)
        self.assertNotIn("translate", built)
        self.assertNotIn(stage_cache_key(cfg, "translate"), cache)
        # every other stage still made it
        self.assertEqual(len(cache), len(STAGE_ORDER) - 1)

    def test_respects_an_explicit_stage_subset(self):
        cfg, cache = _cfg(), {}
        built = prewarm_stage_cache(cfg, cache, stages=["tts", "asr"])
        self.assertEqual(set(built), {"tts", "asr"})
        self.assertEqual(len(cache), 2)

    def test_shared_cache_serves_two_configs_without_crosstalk(self):
        batch = _cfg()
        stream = _cfg({"separate": "htdemucs_mlx_fast"})
        cache: dict = {}
        prewarm_stage_cache(batch, cache)
        prewarm_stage_cache(stream, cache)
        # only `separate` diverges, so exactly one extra entry appears
        self.assertEqual(len(cache), len(STAGE_ORDER) + 1)
        self.assertEqual(cache[stage_cache_key(batch, "separate")],
                         "<separate:stub_separate>")
        self.assertEqual(cache[stage_cache_key(stream, "separate")],
                         "<separate:htdemucs_mlx_fast>")


if __name__ == "__main__":
    unittest.main()


class AudienceDoesNotFragmentTheCache(unittest.TestCase):
    """The audience level rides `cfg.options`, never `cfg.entries`, precisely so
    the warm cache is untouched by it.

    This is the test that would have caught the rejected design — injecting the
    level into the translate candidate's ENTRY dict. That gets the cache key for
    free via `config_hash`, but leaves `stage_cache_key` unchanged (the
    candidate id does not move), so the first job's warm translator would be
    handed to every later job regardless of level. And `prewarm_stage_cache`
    runs at server boot, before any submission exists, so there would be nothing
    to bake in. ADR: 2026-08-28-audience-age-guardrail.
    """

    def test_same_key_for_every_level(self) -> None:
        keys = set()
        for level in ("child", "teenager", "adult"):
            cfg = _cfg()
            cfg.options.audience = level
            keys.add(stage_cache_key(cfg, "translate"))
        self.assertEqual(len(keys), 1, keys)

    def test_a_warm_translate_stage_is_reused_across_levels(self) -> None:
        """Concretely: one built instance serves all three levels, which is only
        safe because the level reaches the candidate as a per-CALL kwarg."""
        from patmir.pipeline.runner import prewarm_stage_cache
        cache: dict = {}
        child = _cfg(); child.options.audience = "child"
        prewarm_stage_cache(child, cache, stages=["translate"])
        built_once = list(child.builds)

        adult = _cfg(); adult.options.audience = "adult"
        prewarm_stage_cache(adult, cache, stages=["translate"])
        self.assertEqual(child.builds, built_once)
        self.assertEqual(adult.builds, [], "second config must not rebuild")

