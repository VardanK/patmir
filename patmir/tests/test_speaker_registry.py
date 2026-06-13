"""Tests for SpeakerRegistry voice assignment + build_pool (Slice 2 #7).

Pure matching logic — no model. Covers gender match, age-range coverage,
adjacent-age fallback, stable per-speaker assignment, variety spread, and
config normalization (string ids, single age, age list).
"""
from __future__ import annotations

import unittest

from patmir.pipeline.streaming.registry import (
    SpeakerProfile,
    SpeakerRegistry,
    VoiceEntry,
    build_pool,
)


def _pool():
    return [
        VoiceEntry("m_storyteller", "male", ("young", "adult", "elder")),
        VoiceEntry("f_young", "female", ("young", "adult")),
        VoiceEntry("m_boy", "male", ("child", "young")),
        VoiceEntry("f_kid", "female", ("child", "young")),
        VoiceEntry("m_elder", "male", ("adult", "elder")),
    ]


class BuildPoolTests(unittest.TestCase):
    def test_normalizes_mixed_config(self):
        pool = build_pool([
            "bareId",
            {"id": "a", "gender": "male", "age": "adult"},
            {"id": "b", "gender": "female", "age": ["child", "young"]},
        ])
        self.assertEqual(pool[0], VoiceEntry("bareId", "any", ("any",)))
        self.assertEqual(pool[1].ages, ("adult",))
        self.assertEqual(pool[2].ages, ("child", "young"))


class AssignmentTests(unittest.TestCase):
    def test_stable_per_speaker(self):
        r = SpeakerRegistry(_pool())
        a = r.assign_voice("S0", SpeakerProfile("male", "adult"))
        b = r.assign_voice("S0", SpeakerProfile("male", "adult"))
        self.assertEqual(a, b)

    def test_gender_is_respected(self):
        r = SpeakerRegistry(_pool())
        v = r.assign_voice("S0", SpeakerProfile("female", "young"))
        self.assertIn(v, {"f_young", "f_kid"})

    def test_age_coverage_match(self):
        r = SpeakerRegistry(_pool())
        # child male → only m_boy covers child
        self.assertEqual(r.assign_voice("S0", SpeakerProfile("male", "child")), "m_boy")
        # elder male → m_storyteller or m_elder both cover elder; least-used wins
        self.assertIn(r.assign_voice("S1", SpeakerProfile("male", "elder")),
                      {"m_storyteller", "m_elder"})

    def test_adjacent_age_fallback_when_no_exact(self):
        # female elder: no female voice covers elder → nearest (adult) female.
        r = SpeakerRegistry(_pool())
        self.assertIn(r.assign_voice("S0", SpeakerProfile("female", "elder")),
                      {"f_young"})

    def test_variety_spread_for_same_profile(self):
        # Three adult males should not all collapse onto one voice.
        r = SpeakerRegistry(_pool())
        got = {r.assign_voice(f"S{i}", SpeakerProfile("male", "adult")) for i in range(3)}
        self.assertGreaterEqual(len(got), 2)

    def test_unknown_profile_still_assigns(self):
        r = SpeakerRegistry(_pool())
        self.assertTrue(r.assign_voice("S0", SpeakerProfile()))  # any/any

    def test_lock_ref_is_stable(self):
        r = SpeakerRegistry(_pool())
        first = r.lock_ref("S0", "/clip/a.wav")
        second = r.lock_ref("S0", "/clip/b.wav")  # later clip ignored
        self.assertEqual(first, "/clip/a.wav")
        self.assertEqual(second, "/clip/a.wav")
        self.assertEqual(r.locked_ref("S0"), "/clip/a.wav")


if __name__ == "__main__":
    unittest.main()
