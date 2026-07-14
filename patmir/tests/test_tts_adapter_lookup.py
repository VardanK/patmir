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
