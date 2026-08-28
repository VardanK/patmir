"""Tests for `patmir.ui.cache`.

The cache key is the contract between the Chrome extension's `/cache/lookup`
hits and the disk layout under `~/.patmir/cache/`. We verify:

* URL canonicalization collapses all YouTube surface variants to a single
  `yt:<videoId>` identifier.
* `cache_key` is stable across irrelevant `[pipeline.options]` changes
  (audio-tuning tweaks must not bust the cache).
* `cache_key` inverts when a stage's candidate id or any candidate-config
  field changes (model swaps must invalidate).
* `materialize` + `lookup` + `lookup_by_url` round-trip.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from patmir.pipeline.config import PipelineConfig
from patmir.pipeline.context import PipelineOptions
from patmir.pipeline.stage import STAGE_ORDER
from patmir.ui import cache as cache_mod


def _make_cfg(**entry_overrides) -> PipelineConfig:
    """Build a PipelineConfig with one stub entry per stage. Caller can
    override any stage's entry dict via `entry_overrides[stage]`."""
    entries: dict[str, dict] = {}
    for stage in STAGE_ORDER:
        entries[stage] = entry_overrides.get(stage, {"id": f"stub_{stage}", "module": f"mod.{stage}"})
    return PipelineConfig(entries=entries, options=PipelineOptions())


class CanonicalizeUrl(unittest.TestCase):
    def test_youtube_watch(self) -> None:
        self.assertEqual(
            cache_mod.canonicalize_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=1s"),
            "yt:dQw4w9WgXcQ",
        )

    def test_youtube_short_link(self) -> None:
        self.assertEqual(
            cache_mod.canonicalize_url("https://youtu.be/dQw4w9WgXcQ?si=abc"),
            "yt:dQw4w9WgXcQ",
        )

    def test_youtube_shorts(self) -> None:
        self.assertEqual(
            cache_mod.canonicalize_url("https://www.youtube.com/shorts/abcDEF12345"),
            "yt:abcDEF12345",
        )

    def test_youtube_embed(self) -> None:
        self.assertEqual(
            cache_mod.canonicalize_url("https://www.youtube.com/embed/zZzZzZzZzZ?autoplay=1"),
            "yt:zZzZzZzZzZ",
        )

    def test_generic_strips_query_and_fragment(self) -> None:
        self.assertEqual(
            cache_mod.canonicalize_url("https://vimeo.com/123456789?embed=1#t=10"),
            "https://vimeo.com/123456789",
        )

    def test_generic_lowercases_host(self) -> None:
        self.assertEqual(
            cache_mod.canonicalize_url("HTTPS://ExAmPlE.CoM/Path"),
            "https://example.com/Path",
        )


class CacheKeyInvariants(unittest.TestCase):
    def test_options_changes_do_not_bust_cache(self) -> None:
        """Audio-tuning options live on `PipelineOptions` and must NOT
        contribute to the cache key — otherwise ear-test loops would
        invalidate everything."""
        cfg_a = _make_cfg()
        cfg_b = _make_cfg()
        cfg_b.options.music_bed_gain = 0.9
        cfg_b.options.voice_gain_db = -6.0
        cfg_b.options.sentence_afade_seconds = 0.08
        src = "url:yt:abcDEF12345"
        self.assertEqual(cache_mod.cache_key(src, cfg_a), cache_mod.cache_key(src, cfg_b))

    def test_candidate_id_swap_inverts(self) -> None:
        cfg_a = _make_cfg()
        cfg_b = _make_cfg(tts={"id": "different_tts", "module": "mod.tts"})
        src = "url:yt:abcDEF12345"
        self.assertNotEqual(cache_mod.cache_key(src, cfg_a), cache_mod.cache_key(src, cfg_b))

    def test_candidate_config_field_inverts(self) -> None:
        """Changing a model_id / voice / language_code inside a candidate
        entry must invalidate the cache even when the candidate id is the same."""
        cfg_a = _make_cfg(tts={"id": "elevenlabs", "module": "m", "voice": "Rachel"})
        cfg_b = _make_cfg(tts={"id": "elevenlabs", "module": "m", "voice": "Adam"})
        src = "url:yt:abcDEF12345"
        self.assertNotEqual(cache_mod.cache_key(src, cfg_a), cache_mod.cache_key(src, cfg_b))

    def test_source_id_changes_key(self) -> None:
        cfg = _make_cfg()
        self.assertNotEqual(
            cache_mod.cache_key("url:yt:aaa", cfg),
            cache_mod.cache_key("url:yt:bbb", cfg),
        )

    def test_key_is_16_hex(self) -> None:
        cfg = _make_cfg()
        key = cache_mod.cache_key("url:yt:abcDEF12345", cfg)
        self.assertEqual(len(key), 16)
        self.assertTrue(all(c in "0123456789abcdef" for c in key))


class AudienceCacheKey(unittest.TestCase):
    """The audience age level is the ONE `[pipeline.options]` field that
    participates in the key — it changes the Armenian text, not its mix.

    It participates via an explicit `cache_key(...)` argument rather than
    `config_hash`, so the warm stage cache (keyed `(stage, candidate_id)`) stays
    un-fragmented and the startup prewarm still hits.
    """

    SRC = "url:yt:abcDEF12345"

    def test_adult_key_is_the_legacy_key(self) -> None:
        """`adult` composes a byte-identical prompt to the pre-guardrail one, so
        it must append nothing — every dub cached before this feature existed
        was produced by exactly that prompt and must keep its key.

        The literal payload is pinned here on purpose: this is the assertion
        that fails loudly if someone "tidies" the suffix logic and silently
        busts every cached dub on the machine.
        """
        cfg = _make_cfg()
        expected = hashlib.sha256(
            f"{self.SRC}|{cache_mod.config_hash(cfg)}".encode()
        ).hexdigest()[:16]
        self.assertEqual(cache_mod.cache_key(self.SRC, cfg), expected)
        self.assertEqual(cache_mod.cache_key(self.SRC, cfg, audience="adult"), expected)

    def test_each_level_gets_a_distinct_key(self) -> None:
        cfg = _make_cfg()
        keys = {
            lvl: cache_mod.cache_key(self.SRC, cfg, audience=lvl)
            for lvl in ("child", "teenager", "adult")
        }
        self.assertEqual(len(set(keys.values())), 3, keys)

    def test_audience_composes_with_mode(self) -> None:
        cfg = _make_cfg()
        wa_child = cache_mod.cache_key(self.SRC, cfg, "watch-along", "child")
        wa_adult = cache_mod.cache_key(self.SRC, cfg, "watch-along", "adult")
        q_child = cache_mod.cache_key(self.SRC, cfg, "quality", "child")
        self.assertNotEqual(wa_child, wa_adult)
        self.assertNotEqual(wa_child, q_child)

    def test_options_audience_alone_does_not_change_the_key(self) -> None:
        """`config_hash` hashes ENTRIES only, so setting the field on the config
        is not what carries the level — the resolved *argument* is.

        This is a real trap: a caller that derives a per-job config but forgets
        to pass the level would write child-register audio under the adult key.
        `resolve_audience` exists so there is one place that cannot be skipped.
        """
        cfg_a = _make_cfg()
        cfg_b = _make_cfg()
        cfg_b.options.audience = "child"
        self.assertEqual(
            cache_mod.cache_key(self.SRC, cfg_a),
            cache_mod.cache_key(self.SRC, cfg_b),
        )

    def test_lookup_by_url_respects_audience(self) -> None:
        cfg = _make_cfg()
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        src_id = cache_mod.source_id_for_url(url)
        child_key = cache_mod.cache_key(src_id, cfg, audience="child")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            entry_dir = root / child_key
            entry_dir.mkdir(parents=True)
            (entry_dir / "dub.wav").write_bytes(b"RIFF")
            (entry_dir / "metadata.json").write_text(json.dumps({
                "cache_key": child_key,
                "source_url": url,
                "duration": 1.0,
                "created_at": 0.0,
            }))
            self.assertIsNotNone(
                cache_mod.lookup_by_url(root, url, cfg, audience="child"))
            self.assertIsNone(
                cache_mod.lookup_by_url(root, url, cfg, audience="adult"))
            # The default argument is the NEUTRAL level, not the product
            # default — a caller that passes nothing looks up the adult key.
            self.assertIsNone(cache_mod.lookup_by_url(root, url, cfg))


class CacheRoundtrip(unittest.TestCase):
    def test_materialize_then_lookup_by_url(self) -> None:
        cfg = _make_cfg()
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        with tempfile.TemporaryDirectory() as cache_dir_str, tempfile.TemporaryDirectory() as job_dir_str:
            cache_root = Path(cache_dir_str)
            job_dir = Path(job_dir_str)
            # Pretend the pipeline produced a dub.wav.
            (job_dir / "dub.wav").write_bytes(b"RIFF....WAVEfake")
            (job_dir / "translated.json").write_text(json.dumps({"sentences": []}))

            src = cache_mod.source_id_for_url(url)
            key = cache_mod.cache_key(src, cfg)
            cache_mod.materialize(
                cache_root,
                key,
                job_dir=job_dir,
                source_id=src,
                source_url=url,
                job_id="fake-job",
                duration=300.0,
                pipeline_summary=cache_mod.pipeline_summary(cfg),
            )
            entry = cache_mod.lookup_by_url(cache_root, url, cfg)
            self.assertIsNotNone(entry)
            assert entry is not None  # narrow for mypy
            self.assertEqual(entry.cache_key, key)
            self.assertEqual(entry.source_url, url)
            self.assertAlmostEqual(entry.duration or 0.0, 300.0)
            self.assertTrue(entry.dub_wav().is_file())
            self.assertTrue((entry.cache_dir / "transcript.json").is_file())

    def test_lookup_misses_when_config_changes(self) -> None:
        cfg_a = _make_cfg()
        cfg_b = _make_cfg(tts={"id": "different_tts", "module": "mod.tts"})
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        with tempfile.TemporaryDirectory() as cache_dir_str, tempfile.TemporaryDirectory() as job_dir_str:
            cache_root = Path(cache_dir_str)
            job_dir = Path(job_dir_str)
            (job_dir / "dub.wav").write_bytes(b"x")
            src = cache_mod.source_id_for_url(url)
            key = cache_mod.cache_key(src, cfg_a)
            cache_mod.materialize(
                cache_root, key,
                job_dir=job_dir, source_id=src, source_url=url,
                job_id="j", duration=None,
                pipeline_summary=cache_mod.pipeline_summary(cfg_a),
            )
            self.assertIsNotNone(cache_mod.lookup_by_url(cache_root, url, cfg_a))
            self.assertIsNone(cache_mod.lookup_by_url(cache_root, url, cfg_b))


class SourceIdForFile(unittest.TestCase):
    def test_small_file_hash_is_stable(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"hello world")
            tmp_path = Path(tmp.name)
        try:
            a = cache_mod.source_id_for_file(tmp_path)
            b = cache_mod.source_id_for_file(tmp_path)
            self.assertEqual(a, b)
            self.assertTrue(a.startswith("file:11:"))
        finally:
            tmp_path.unlink()

    def test_different_files_differ(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as a, tempfile.NamedTemporaryFile(delete=False) as b:
            a.write(b"aaa")
            b.write(b"bbb")
            a_path, b_path = Path(a.name), Path(b.name)
        try:
            self.assertNotEqual(
                cache_mod.source_id_for_file(a_path),
                cache_mod.source_id_for_file(b_path),
            )
        finally:
            a_path.unlink()
            b_path.unlink()


if __name__ == "__main__":
    unittest.main()
