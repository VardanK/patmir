"""Tests for stream restart + pause survival (spec 2026-08-25-1931, A3 + C1).

Two defects, both observed in the author's 2026-08-25 session:

* **A3** — the scheduler ended a stream after `abandon_seconds` without a
  playhead update, but the browser client stops POSTing the playhead while
  paused. A pause longer than 45 s therefore killed the stream, and `submit()`
  handed the viewer back the finished record instead of restarting it, so the
  video was frozen at the buffer edge with no recovery short of a server bounce.

* **C1** — a restarted stream re-seeded *coverage* from the cached sections but
  restarted its section *index* at 0. `append_section` names the file from the
  index, so it overwrote `section_001.wav` while deduping entries by START and
  keeping both — two windows pointing at one file. On cache key
  `d5af5653d745ed66` indices 1 and 2 each appeared twice, so the wrong audio
  played at 154.9 s and 190.3 s, live and on every rewatch.
"""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from patmir.pipeline.streaming.scheduler import Playhead, StreamScheduler
from patmir.ui import cache as cache_mod
from patmir.ui.stream import _index_seed

# The real damage, as read off ~/.patmir/cache/d5af5653d745ed66/sections.json.
CORRUPTED = [
    {"index": 1, "start": 154.946638, "end": 190.346638, "file": "section_001.wav"},
    {"index": 2, "start": 190.346638, "end": 198.446638, "file": "section_002.wav"},
    {"index": 1, "start": 198.547435, "end": 229.847435, "file": "section_001.wav"},
    {"index": 2, "start": 229.847435, "end": 243.447435, "file": "section_002.wav"},
]


class IndexSeedTest(unittest.TestCase):
    def test_fresh_stream_starts_at_the_opening_index(self):
        self.assertEqual(_index_seed(None, 0), 0)

    def test_resumed_stream_numbers_past_the_cached_sections(self):
        """The C1 fix: the next section must not reuse index 1."""
        prior = {"sections": [{"index": 0}, {"index": 1}, {"index": 2}]}
        self.assertEqual(_index_seed(prior, 0), 2)

    def test_uses_the_highest_index_not_the_count(self):
        """Sections can be sparse after a seek, so len() is the wrong answer."""
        prior = {"sections": [{"index": 0}, {"index": 7}]}
        self.assertEqual(_index_seed(prior, 0), 7)

    def test_seeding_from_the_real_corrupted_manifest(self):
        self.assertEqual(_index_seed({"sections": CORRUPTED}, 0), 2)

    def test_malformed_entries_are_ignored_not_fatal(self):
        prior = {"sections": [{"index": "oops"}, {}, {"index": 4}, "junk"]}
        self.assertEqual(_index_seed(prior, 0), 4)

    def test_empty_sections_list(self):
        self.assertEqual(_index_seed({"sections": []}, 0), 0)


class IndexReconciliationTest(unittest.TestCase):
    """`append_section` must make the duplicate-index shape unrepresentable."""

    def _setup(self, td: str):
        root, key = Path(td), "k"
        cache_mod.stream_cache_dir(root, key)
        wav = root / "chunk.wav"
        wav.write_bytes(b"RIFF....WAVEfmt ")
        return root, key, wav

    def test_reusing_an_index_drops_the_stale_entry(self):
        with tempfile.TemporaryDirectory() as td:
            root, key, wav = self._setup(td)
            cache_mod.append_section(root, key, index=1, start=154.9, end=190.3,
                                     src_wav=wav)
            m = cache_mod.append_section(root, key, index=1, start=198.5, end=229.8,
                                         src_wav=wav)
        # One file, so one entry — the stale window is gone.
        self.assertEqual(len(m["sections"]), 1)
        self.assertEqual(m["sections"][0]["start"], 198.5)

    def test_no_two_entries_ever_share_an_index(self):
        with tempfile.TemporaryDirectory() as td:
            root, key, wav = self._setup(td)
            for idx, start in [(1, 154.9), (2, 190.3), (1, 198.5), (2, 229.8)]:
                m = cache_mod.append_section(root, key, index=idx, start=start,
                                             end=start + 20, src_wav=wav)
        indices = [s["index"] for s in m["sections"]]
        self.assertEqual(len(indices), len(set(indices)))

    def test_no_two_entries_ever_share_a_file(self):
        with tempfile.TemporaryDirectory() as td:
            root, key, wav = self._setup(td)
            for idx, start in [(1, 154.9), (2, 190.3), (1, 198.5), (2, 229.8)]:
                m = cache_mod.append_section(root, key, index=idx, start=start,
                                             end=start + 20, src_wav=wav)
        files = [s["file"] for s in m["sections"]]
        self.assertEqual(len(files), len(set(files)))

    def test_an_already_corrupted_manifest_self_heals_on_next_append(self):
        """No migration needed: the next write repairs the damaged shape."""
        with tempfile.TemporaryDirectory() as td:
            root, key, wav = self._setup(td)
            path = cache_mod.stream_cache_dir(root, key) / "sections.json"
            path.write_text(json.dumps(
                {"cache_key": key, "complete": False, "sections": list(CORRUPTED)}))
            m = cache_mod.append_section(root, key, index=1, start=300.0, end=330.0,
                                         src_wav=wav)
        idx1 = [s for s in m["sections"] if s["index"] == 1]
        self.assertEqual(len(idx1), 1)
        self.assertEqual(idx1[0]["start"], 300.0)

    def test_distinct_indices_coexist_normally(self):
        with tempfile.TemporaryDirectory() as td:
            root, key, wav = self._setup(td)
            cache_mod.append_section(root, key, index=1, start=40.0, end=70.0,
                                     src_wav=wav)
            m = cache_mod.append_section(root, key, index=2, start=70.0, end=100.0,
                                         src_wav=wav)
        self.assertEqual(len(m["sections"]), 2)


class PauseSurvivalTest(unittest.TestCase):
    """A3: a paused viewer is present; only a departed one ends the stream."""

    def _sched(self, alive_fn=None, playhead=None) -> StreamScheduler:
        return StreamScheduler(
            duration=600.0, prelude_end=600.0,          # fully covered → always idles
            process_fn=lambda _s: None,
            playhead_fn=(lambda: playhead) if playhead else Playhead,
            abandon_seconds=45.0,
            alive_fn=alive_fn,
        )

    def test_paused_viewer_that_is_still_polling_keeps_the_stream_alive(self):
        """The measured killer: playhead frozen for 10 minutes, tab still open.

        Asserted by BEHAVIOUR, not by return value alone: an abandoned viewer
        returns False immediately, while a live one falls through to the 2 s
        interruptible wait and returns True. Checking only the return value
        with `stop()` set would pass either way, since both paths end False.
        """
        stale = Playhead(t=100.0, paused=True, updated_at=time.time() - 600)
        sched = self._sched(alive_fn=lambda: time.time(), playhead=stale)
        # Precondition: the OLD rule would have abandoned this stream.
        self.assertGreater(time.time() - stale.updated_at, sched.abandon_seconds)

        t0 = time.perf_counter()
        kept_waiting = sched._default_idle()
        elapsed = time.perf_counter() - t0

        self.assertTrue(kept_waiting, "paused viewer was treated as departed")
        self.assertGreater(elapsed, 1.5, "returned early — it took the abandon branch")

    def test_a_departed_viewer_ends_the_stream(self):
        sched = self._sched(alive_fn=lambda: time.time() - 600)
        t0 = time.perf_counter()
        self.assertFalse(sched._default_idle())
        # Immediate, i.e. it really was the abandon branch and not the wait.
        self.assertLess(time.perf_counter() - t0, 0.5)

    def test_stop_ends_the_stream_even_when_the_client_is_live(self):
        sched = self._sched(alive_fn=lambda: time.time())
        sched.stop()
        t0 = time.perf_counter()
        self.assertFalse(sched._default_idle())
        self.assertLess(time.perf_counter() - t0, 0.5)

    def test_without_alive_fn_the_old_playhead_behaviour_is_kept(self):
        """bench and the existing tests construct schedulers without it."""
        stale = Playhead(t=0.0, paused=False, updated_at=time.time() - 600)
        sched = self._sched(playhead=stale)
        t0 = time.perf_counter()
        self.assertFalse(sched._default_idle())
        self.assertLess(time.perf_counter() - t0, 0.5)


if __name__ == "__main__":
    unittest.main()
