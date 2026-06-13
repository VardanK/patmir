"""Tests for ElevenLabs no-clone preset-pool voice assignment (`clone=False`).

Pure mapping logic — no network. Covers:
  1. A speaker maps to a stable voice (same id across calls).
  2. Distinct speakers take distinct pool voices, in order.
  3. More speakers than pool voices wraps round-robin.
  4. `summarize()` reports every speaker with status "pool".
  5. An explicit `voice_id` kwarg overrides pool assignment.
"""
from __future__ import annotations

import os
import unittest

from patmir.eval.candidates.tts.elevenlabs import ElevenLabsTTS, _DEFAULT_VOICE_POOL

_POOL_IDS = [e["id"] for e in _DEFAULT_VOICE_POOL]


def _candidate(pool=None):
    # __init__ only reads the API key; no network until synthesize().
    os.environ.setdefault("ELEVENLABS_API_KEY", "test-key")
    return ElevenLabsTTS(clone=False, voice_pool=pool)


class PoolVoiceAssignmentTests(unittest.TestCase):
    def test_speaker_voice_is_stable(self):
        c = _candidate()
        first = c._pool_voice_for_speaker("SPEAKER_00")
        again = c._pool_voice_for_speaker("SPEAKER_00")
        self.assertEqual(first, again)

    def test_distinct_speakers_take_pool_in_order(self):
        c = _candidate()
        v0 = c._pool_voice_for_speaker("SPEAKER_00")
        v1 = c._pool_voice_for_speaker("SPEAKER_01")
        v2 = c._pool_voice_for_speaker("SPEAKER_02")
        self.assertEqual([v0, v1, v2], _POOL_IDS[:3])
        self.assertEqual(len({v0, v1, v2}), 3)

    def test_round_robin_wraps(self):
        c = _candidate(pool=["a", "b"])
        self.assertEqual(c._pool_voice_for_speaker("S0"), "a")
        self.assertEqual(c._pool_voice_for_speaker("S1"), "b")
        self.assertEqual(c._pool_voice_for_speaker("S2"), "a")  # wraps

    def test_summarize_reports_pool_status(self):
        c = _candidate()
        # Simulate line counting + assignment as synthesize() would.
        for spk in ("SPEAKER_00", "SPEAKER_01"):
            c._speaker_lines[spk] = 1
            c._pool_voice_for_speaker(spk)
        summary = c.summarize()
        self.assertEqual(
            {s["status"] for s in summary["speakers"].values()}, {"pool"}
        )
        self.assertEqual(
            summary["speakers"]["SPEAKER_00"]["voice_id"], _POOL_IDS[0]
        )


if __name__ == "__main__":
    unittest.main()
