"""Tests for watch-along cache keying + per-section stream cache (Slice 3 #10).

Two contracts:
  * `mode` is folded into the cache key so quality and watch-along never
    collide — AND the default (quality) key is byte-for-byte unchanged, so
    existing cached dubs aren't busted.
  * The per-section stream cache (init → append → read → finalize) round-trips,
    replaces an out-of-order section at the same start instead of duplicating,
    and keeps sections sorted by global start.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from patmir.pipeline.config import PipelineConfig
from patmir.pipeline.context import PipelineOptions
from patmir.pipeline.stage import STAGE_ORDER
from patmir.ui import cache as cache_mod


def _make_cfg() -> PipelineConfig:
    entries = {s: {"id": f"stub_{s}", "module": f"mod.{s}"} for s in STAGE_ORDER}
    return PipelineConfig(entries=entries, options=PipelineOptions())


class ModeKeying(unittest.TestCase):
    def test_quality_key_unchanged_by_mode_arg(self):
        cfg = _make_cfg()
        src = "url:yt:abcDEF12345"
        self.assertEqual(
            cache_mod.cache_key(src, cfg),
            cache_mod.cache_key(src, cfg, "quality"),
        )

    def test_watch_along_key_differs(self):
        cfg = _make_cfg()
        src = "url:yt:abcDEF12345"
        self.assertNotEqual(
            cache_mod.cache_key(src, cfg, "quality"),
            cache_mod.cache_key(src, cfg, "watch-along"),
        )

    def test_watch_along_key_is_16_hex(self):
        key = cache_mod.cache_key("url:yt:abcDEF12345", _make_cfg(), "watch-along")
        self.assertEqual(len(key), 16)
        self.assertTrue(all(c in "0123456789abcdef" for c in key))


class StreamCacheRoundtrip(unittest.TestCase):
    def _wav(self, root: Path, name: str) -> Path:
        p = root / name
        p.write_bytes(b"RIFF....WAVEfake")
        return p

    def test_init_append_read_finalize(self):
        cfg = _make_cfg()
        with tempfile.TemporaryDirectory() as cd, tempfile.TemporaryDirectory() as sd:
            cache_root, src_dir = Path(cd), Path(sd)
            key = cache_mod.cache_key("url:yt:vid", cfg, "watch-along")

            cache_mod.init_stream_manifest(
                cache_root, key, source_id="url:yt:vid", source_url="http://x/v",
                duration=120.0, pipeline_summary=cache_mod.pipeline_summary(cfg),
            )
            m = cache_mod.read_stream_manifest(cache_root, key)
            self.assertFalse(m["complete"])
            self.assertEqual(m["sections"], [])

            cache_mod.append_section(cache_root, key, index=0, start=0.0, end=58.0,
                                     src_wav=self._wav(src_dir, "a.wav"))
            cache_mod.append_section(cache_root, key, index=1, start=58.0, end=90.0,
                                     src_wav=self._wav(src_dir, "b.wav"))
            m = cache_mod.read_stream_manifest(cache_root, key)
            self.assertEqual(len(m["sections"]), 2)
            self.assertEqual(m["sections"][0]["start"], 0.0)
            self.assertTrue((cache_root / key / m["sections"][0]["file"]).is_file())

            cache_mod.finalize_stream(cache_root, key)
            self.assertTrue(cache_mod.read_stream_manifest(cache_root, key)["complete"])

    def test_append_replaces_same_start(self):
        cfg = _make_cfg()
        with tempfile.TemporaryDirectory() as cd, tempfile.TemporaryDirectory() as sd:
            cache_root, src_dir = Path(cd), Path(sd)
            key = cache_mod.cache_key("url:yt:vid", cfg, "watch-along")
            cache_mod.init_stream_manifest(
                cache_root, key, source_id="s", source_url=None, duration=None,
                pipeline_summary={},
            )
            cache_mod.append_section(cache_root, key, index=5, start=60.0, end=90.0,
                                     src_wav=self._wav(src_dir, "a.wav"))
            # Re-processed (e.g. after a seek) at the same start → replace, not dup.
            cache_mod.append_section(cache_root, key, index=9, start=60.0, end=95.0,
                                     src_wav=self._wav(src_dir, "b.wav"))
            m = cache_mod.read_stream_manifest(cache_root, key)
            self.assertEqual(len(m["sections"]), 1)
            self.assertEqual(m["sections"][0]["index"], 9)
            self.assertEqual(m["sections"][0]["end"], 95.0)

    def test_sections_sorted_by_start(self):
        cfg = _make_cfg()
        with tempfile.TemporaryDirectory() as cd, tempfile.TemporaryDirectory() as sd:
            cache_root, src_dir = Path(cd), Path(sd)
            key = cache_mod.cache_key("url:yt:vid", cfg, "watch-along")
            cache_mod.init_stream_manifest(cache_root, key, source_id="s",
                                           source_url=None, duration=None, pipeline_summary={})
            # Append out of order (seek produced a later section first).
            cache_mod.append_section(cache_root, key, index=3, start=120.0, end=150.0,
                                     src_wav=self._wav(src_dir, "c.wav"))
            cache_mod.append_section(cache_root, key, index=1, start=0.0, end=30.0,
                                     src_wav=self._wav(src_dir, "a.wav"))
            m = cache_mod.read_stream_manifest(cache_root, key)
            self.assertEqual([s["start"] for s in m["sections"]], [0.0, 120.0])


if __name__ == "__main__":
    unittest.main()
