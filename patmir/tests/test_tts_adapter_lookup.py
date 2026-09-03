"""Tests for `SentenceTTS._lookup_ref` fallback chain and the
`ref_text_optional` per-candidate gate.

Covers:
  1. exact (speaker, emotion) match
  2. (speaker, "neutral") fallback when exact misses
  3. any (speaker, *) fallback when no neutral; longest window wins
  4. None when speaker has no refs at all
  5. The `ref_text_optional = True` branch — refs with empty `ref_text` are
     usable for ElevenLabs-style IVC candidates.
  6. The default branch (`ref_text_optional` absent / False) — empty
     `ref_text` falls through, mirroring OmniVoice's needs.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from patmir.pipeline.stage import Diarization, SpeakerRef
from patmir.pipeline.stages.tts.adapter import SentenceTTS


def _ref(speaker: str, emotion: str, *, text: str = "hello", start: float = 0.0, end: float = 2.0) -> SpeakerRef:
    return SpeakerRef(
        speaker_id=speaker,
        ref_audio=Path(f"/tmp/{speaker}/{emotion}.wav"),
        ref_text=text,
        source_start=start,
        source_end=end,
    )


def _diar(refs: dict) -> Diarization:
    return Diarization(turns=[], overlaps=[], refs=refs, sample_rate=16000)


class _NeedsRefText:
    """OmniVoice-like stand-in: needs the reference transcript."""
    def synthesize(self, text: str, *, ref_audio=None, ref_text=None, guidance_scale=None):  # noqa: D401
        return b""


class _RefTextOptional:
    """ElevenLabs-like stand-in: ignores ref_text."""
    ref_text_optional = True

    def synthesize(self, text: str, *, ref_audio=None, ref_text=None, emotion=None):  # noqa: D401
        return b""


class LookupRefTests(unittest.TestCase):
    def setUp(self):
        self.adapter_needs_text = SentenceTTS(inner=_NeedsRefText())
        self.adapter_text_optional = SentenceTTS(inner=_RefTextOptional())

    def test_exact_match(self):
        diar = _diar({
            ("SPK", "happy"): _ref("SPK", "happy"),
            ("SPK", "neutral"): _ref("SPK", "neutral"),
        })
        ref = self.adapter_needs_text._lookup_ref(diar, "SPK", "happy")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.ref_audio.name, "happy.wav")

    def test_speaker_neutral_fallback(self):
        diar = _diar({
            ("SPK", "neutral"): _ref("SPK", "neutral"),
        })
        ref = self.adapter_needs_text._lookup_ref(diar, "SPK", "angry")
        self.assertEqual(ref.ref_audio.name, "neutral.wav")

    def test_any_speaker_fallback_picks_longest(self):
        diar = _diar({
            ("SPK", "happy"): _ref("SPK", "happy", start=0.0, end=2.0),
            ("SPK", "sad"): _ref("SPK", "sad", start=0.0, end=5.0),
        })
        ref = self.adapter_needs_text._lookup_ref(diar, "SPK", "angry")
        self.assertEqual(ref.ref_audio.name, "sad.wav")

    def test_no_refs_for_speaker(self):
        diar = _diar({("OTHER", "happy"): _ref("OTHER", "happy")})
        ref = self.adapter_needs_text._lookup_ref(diar, "SPK", "happy")
        self.assertIsNone(ref)

    def test_empty_text_blocks_when_candidate_needs_text(self):
        diar = _diar({("SPK", "happy"): _ref("SPK", "happy", text="")})
        ref = self.adapter_needs_text._lookup_ref(diar, "SPK", "happy")
        self.assertIsNone(ref)

    def test_empty_text_ok_when_ref_text_optional(self):
        """The bug fix: ElevenLabs-style candidates accept refs whose
        backfilled text came back empty."""
        diar = _diar({("SPK", "happy"): _ref("SPK", "happy", text="")})
        ref = self.adapter_text_optional._lookup_ref(diar, "SPK", "happy")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.ref_audio.name, "happy.wav")

    def test_no_speaker_id_returns_none(self):
        diar = _diar({("SPK", "happy"): _ref("SPK", "happy")})
        self.assertIsNone(self.adapter_text_optional._lookup_ref(diar, None, "happy"))

    def test_no_diarization_returns_none(self):
        self.assertIsNone(self.adapter_text_optional._lookup_ref(None, "SPK", "happy"))


class LockedRefTests(unittest.TestCase):
    """Streaming: `ctx.extra["locked_refs"]` (SpeakerRegistry) takes precedence
    over per-section refs so a character clones from one clip in every section."""

    def setUp(self):
        self.adapter_needs_text = SentenceTTS(inner=_NeedsRefText())
        self.adapter_text_optional = SentenceTTS(inner=_RefTextOptional())

    def test_locked_ref_used(self):
        locked = {"spk0": {"ref_audio": "/tmp/spk0/locked.wav", "ref_text": "hi"}}
        ref = self.adapter_text_optional._locked_ref(locked, "spk0")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.ref_audio, Path("/tmp/spk0/locked.wav"))
        self.assertEqual(ref.ref_text, "hi")

    def test_unknown_speaker_falls_through(self):
        self.assertIsNone(self.adapter_text_optional._locked_ref({}, "spk0"))
        self.assertIsNone(self.adapter_text_optional._locked_ref(
            {"spk1": {"ref_audio": "/tmp/x.wav"}}, "spk0"))

    def test_empty_text_blocks_when_candidate_needs_text(self):
        locked = {"spk0": {"ref_audio": "/tmp/spk0/locked.wav", "ref_text": ""}}
        self.assertIsNone(self.adapter_needs_text._locked_ref(locked, "spk0"))
        # ...but is fine for IVC-style candidates (Higgs, ElevenLabs).
        self.assertIsNotNone(self.adapter_text_optional._locked_ref(locked, "spk0"))

    def test_no_speaker_id_returns_none(self):
        locked = {"spk0": {"ref_audio": "/tmp/spk0/locked.wav", "ref_text": "hi"}}
        self.assertIsNone(self.adapter_text_optional._locked_ref(locked, None))


class PoolRefTests(unittest.TestCase):
    """Stand-in pool for ref-less speakers (quality issue #12): configured
    [[tts_ref_pool]] clips first, else another character's pinned ref — never
    the anonymous no-ref default while material exists."""

    def setUp(self):
        self.adapter = SentenceTTS(inner=_RefTextOptional())

    def test_configured_pool_gender_match(self):
        pool = [
            {"path": "/pool/m.wav", "gender": "male", "text": "hi"},
            {"path": "/pool/f.wav", "gender": "female", "text": "hi"},
        ]
        profiles = {"spk0": {"gender": "female", "age": "adult"}}
        ref = self.adapter._pool_ref(None, "spk0", pool, profiles)
        self.assertEqual(ref.ref_audio, Path("/pool/f.wav"))

    def test_assignment_stable_and_spread(self):
        pool = [{"path": "/pool/a.wav"}, {"path": "/pool/b.wav"}]
        r0 = self.adapter._pool_ref(None, "spk0", pool, {})
        r0_again = self.adapter._pool_ref(None, "spk0", pool, {})
        r1 = self.adapter._pool_ref(None, "spk1", pool, {})
        self.assertEqual(r0.ref_audio, r0_again.ref_audio)      # stable per speaker
        self.assertNotEqual(r0.ref_audio, r1.ref_audio)         # spread least-used

    def test_reuses_other_speakers_ref_without_pool(self):
        diar = _diar({
            ("OTHER", "neutral"): _ref("OTHER", "neutral", start=0.0, end=4.0),
            ("OTHER", "happy"): _ref("OTHER", "happy", start=0.0, end=2.0),
        })
        ref = self.adapter._pool_ref(diar, "SPK", [], {})
        self.assertIsNotNone(ref)
        self.assertEqual(ref.ref_audio.name, "neutral.wav")     # OTHER's longest clip

    def test_nothing_available_returns_none(self):
        self.assertIsNone(self.adapter._pool_ref(None, "SPK", [], {}))
        self.assertIsNone(self.adapter._pool_ref(_diar({}), "SPK", [], {}))

    def test_text_required_gates_pool(self):
        adapter = SentenceTTS(inner=_NeedsRefText())
        pool = [{"path": "/pool/a.wav"}]                        # no transcript
        self.assertIsNone(adapter._pool_ref(None, "SPK", pool, {}))
        pool_with_text = [{"path": "/pool/a.wav", "text": "hello"}]
        self.assertIsNotNone(adapter._pool_ref(None, "SPK", pool_with_text, {}))


if __name__ == "__main__":
    unittest.main()


class PinPrefersNeutralRefTest(unittest.TestCase):
    """Which clip the per-speaker pin chooses is not emotion-neutral.

    Higgs clones prosody from the reference, so pinning an angry clip makes a
    character deliver every line angrily — calm ones included. Plain
    `max(duration)` chose a non-neutral clip in 26 of 34 speaker-runs measured
    on real data (angry 10, happy 13, sad 3, neutral 8), which is a larger
    contributor to quality issue #15 than the emotion tag, then reaching ~9%
    of sentences.

    The rule: prefer neutral, but only when it is within `neutral_fraction` of
    the longest clip — a much shorter reference clones worse (issue #24's
    timbre drift), so neutrality is not worth any price.
    """

    def _refs(self, spec):
        # spec: {bin: seconds}
        return _diar({
            ("SPEAKER_00", bin_): _ref("SPEAKER_00", bin_, start=0.0, end=float(secs))
            for bin_, secs in spec.items()
        })

    def _pick(self, spec, fraction=0.5):
        tts = SentenceTTS(inner=_RefTextOptional())
        return tts._lookup_ref(
            self._refs(spec), "SPEAKER_00", "angry",
            pin_speaker_ref=True, neutral_fraction=fraction,
        )

    def test_neutral_wins_over_a_longer_angry_clip(self):
        # The measured case: SPEAKER_01 had angry 13.7s and neutral 8.2s, and
        # the old rule pinned angry, so the character was angry throughout.
        ref = self._pick({"angry": 13.72, "neutral": 8.20})
        self.assertEqual(ref.source_end, 8.20)

    def test_a_much_shorter_neutral_clip_does_not_win(self):
        # 1.6s against 13.7s is 0.12 of the length — below the floor, so the
        # longer clip keeps it. Trading that much reference for neutrality
        # buys emotion at the cost of the voice itself.
        ref = self._pick({"angry": 13.72, "neutral": 1.60})
        self.assertEqual(ref.source_end, 13.72)

    def test_exactly_at_the_fraction_qualifies(self):
        ref = self._pick({"angry": 10.0, "neutral": 5.0})
        self.assertEqual(ref.source_end, 5.0)

    def test_no_neutral_clip_falls_back_to_the_longest(self):
        # A speaker seen only while angry still gets their own voice rather
        # than another character's — the point of step 3 in the chain.
        ref = self._pick({"angry": 7.7, "happy": 3.0})
        self.assertEqual(ref.source_end, 7.7)

    def test_neutral_is_used_when_it_is_also_the_longest(self):
        ref = self._pick({"angry": 3.0, "neutral": 9.0})
        self.assertEqual(ref.source_end, 9.0)

    def test_a_fraction_above_one_disables_the_preference(self):
        # Same convention as `duplicate_containment`: out of range turns it
        # off, because a neutral clip cannot exceed the longest one.
        ref = self._pick({"angry": 13.72, "neutral": 8.20}, fraction=1.5)
        self.assertEqual(ref.source_end, 13.72)

    def test_pin_still_returns_none_with_no_usable_ref(self):
        # A candidate that requires `ref_text` plus a ref with none: the pin
        # must decline rather than hand back an unusable clip.
        tts = SentenceTTS(inner=_NeedsRefText())
        refs = _diar({("SPEAKER_00", "angry"):
                      _ref("SPEAKER_00", "angry", text="", end=5.0)})
        ref = tts._lookup_ref(refs, "SPEAKER_00", "angry",
                              pin_speaker_ref=True, neutral_fraction=0.5)
        self.assertIsNone(ref)

    def test_the_unpinned_chain_is_untouched(self):
        # `tts_pin_speaker_ref=false` restores Phase-2.x per-emotion binning,
        # and this change must not leak into it.
        tts = SentenceTTS(inner=_RefTextOptional())
        ref = tts._lookup_ref(
            self._refs({"angry": 13.72, "neutral": 8.20}),
            "SPEAKER_00", "angry",
            pin_speaker_ref=False, neutral_fraction=0.5,
        )
        self.assertEqual(ref.source_end, 13.72)   # exact (speaker, angry) match
