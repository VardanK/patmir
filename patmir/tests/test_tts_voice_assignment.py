"""Test that the TTS adapter passes registry voice assignments through
`ctx.extra['voice_assignments']` to candidates that accept a `voice_id`
(ElevenLabs pool mode), and skips it for candidates that don't (#9a).
"""
from __future__ import annotations

import io
import tempfile
import unittest
import wave
from pathlib import Path

from patmir.pipeline.context import JobContext, PipelineOptions
from patmir.pipeline.stage import TranslatedSentence, TranslatedTranscript
from patmir.pipeline.stages.tts.adapter import SentenceTTS


def _wav_bytes(seconds: float = 0.1, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * int(seconds * sr))
    return buf.getvalue()


class _PoolInner:
    """Candidate that accepts voice_id (like ElevenLabs pool mode)."""
    def __init__(self):
        self.calls = []

    def synthesize(self, text, *, speaker_id=None, voice_id=None, emotion=None):
        self.calls.append((speaker_id, voice_id))
        return _wav_bytes()


class _PlainInner:
    """Candidate with no voice_id kwarg (like Piper)."""
    def __init__(self):
        self.calls = []

    def synthesize(self, text):
        self.calls.append(text)
        return _wav_bytes()


def _transcript():
    return TranslatedTranscript(
        source_lang="en", target_lang="hy",
        sentences=[
            TranslatedSentence(idx=0, source_text="a", target_text="Ա", start=0.0, end=1.0, speaker_id="SPEAKER_00"),
            TranslatedSentence(idx=1, source_text="b", target_text="Բ", start=1.0, end=2.0, speaker_id="SPEAKER_01"),
        ],
    )


def _ctx(tmp, assignments):
    return JobContext(
        job_dir=Path(tmp),
        source_path=Path(tmp) / "src.wav",
        options=PipelineOptions(),
        extra={"voice_assignments": assignments},
    )


class VoiceAssignmentTests(unittest.TestCase):
    def test_voice_id_passed_per_speaker(self):
        inner = _PoolInner()
        tts = SentenceTTS(inner)
        self.assertTrue(tts._supports_voice_id)
        with tempfile.TemporaryDirectory() as tmp:
            tts.synthesize(_transcript(), _ctx(tmp, {"SPEAKER_00": "vA", "SPEAKER_01": "vB"}))
        self.assertEqual(inner.calls, [("SPEAKER_00", "vA"), ("SPEAKER_01", "vB")])

    def test_missing_assignment_omits_voice_id(self):
        inner = _PoolInner()
        tts = SentenceTTS(inner)
        with tempfile.TemporaryDirectory() as tmp:
            tts.synthesize(_transcript(), _ctx(tmp, {"SPEAKER_00": "vA"}))  # no S01
        # S01 had no assignment → voice_id defaults to None (candidate falls back)
        self.assertEqual(inner.calls, [("SPEAKER_00", "vA"), ("SPEAKER_01", None)])

    def test_plain_candidate_ignores_voice_id(self):
        inner = _PlainInner()
        tts = SentenceTTS(inner)
        self.assertFalse(tts._supports_voice_id)
        with tempfile.TemporaryDirectory() as tmp:
            tts.synthesize(_transcript(), _ctx(tmp, {"SPEAKER_00": "vA"}))
        self.assertEqual(inner.calls, ["Ա", "Բ"])  # synthesized without error


if __name__ == "__main__":
    unittest.main()
