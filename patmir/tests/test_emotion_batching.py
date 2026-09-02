"""Emotion batching (2026-09-02).

The stage made one funasr `generate` call per diarization turn — ~108 on a
5-minute clip — paying per-call setup every time. It now hands funasr the list.

Measured on 24 real turns: 0.65 s -> 0.22 s (2.96x) with **24/24 labels and
confidences identical**, because funasr still runs each clip at batch_size 1
internally. That is what makes this a scheduling change and not a quality one:
if it padded a batch to its longest clip, the padding would enter the pooled
utterance embedding and shift scores, which feed the TTS emotion tag and so the
audio.

These tests pin the parts that would silently misalign emotions with turns.
"""
from __future__ import annotations

import unittest

import numpy as np

from patmir.pipeline.probe import Probe
from patmir.pipeline.stage import Diarization, SpeakerTurn
from patmir.pipeline.stages.emotion import emotion2vec


class _Ctx:
    def __init__(self):
        self.probe = Probe()
        self.extra = {}
        self.msgs = []

    def progress(self, *a, **kw):
        self.msgs.append(a)


class _FakeModel:
    """Stands in for funasr's AutoModel. Records call shapes."""

    def __init__(self, per_clip_label=None, break_shape=False):
        self.calls = []
        self.break_shape = break_shape
        self._label = per_clip_label or (lambda arr: "angry")

    def generate(self, audio, granularity=None, extract_embedding=None):
        batched = isinstance(audio, list)
        clips = audio if batched else [audio]
        self.calls.append(len(clips))
        rows = [{"labels": [self._label(c), "neutral"], "scores": [0.9, 0.1]}
                for c in clips]
        if batched and self.break_shape:
            return rows[:-1]          # wrong length on purpose
        return rows


def _cand(model):
    c = emotion2vec.Emotion2VecCandidate.__new__(emotion2vec.Emotion2VecCandidate)
    c.model = model
    c.device = "cpu"
    c.min_confidence = 0.5
    c.bin_map = {"angry": "angry", "happy": "happy", "neutral": "neutral"}
    return c


def _diar(spans):
    return Diarization(
        turns=[SpeakerTurn(start=s, end=e, speaker_id="S0") for s, e in spans],
        overlaps=[], refs={}, sample_rate=16_000,
    )


class _Sep:
    def __init__(self, path):
        self.vocals = path


class EmotionBatchingTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        import soundfile as sf
        self.tmp = tempfile.TemporaryDirectory()
        self.wav = f"{self.tmp.name}/vocals.wav"
        # Long enough that every turn in these tests slices a real clip;
        # at 30 s the 70-turn case silently lost 40 of them to empty slices.
        sf.write(self.wav, np.zeros((16_000 * 90, 1), dtype="float32"), 16_000)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, spans, model):
        import pathlib
        ctx = _Ctx()
        ctx.job_dir = pathlib.Path(self.tmp.name)
        return _cand(model).detect(_Sep(self.wav), _diar(spans), ctx), ctx

    def test_all_turns_go_in_one_call(self):
        m = _FakeModel()
        spans = [(i, i + 1.0) for i in range(10)]
        preds, _ = self._run(spans, m)
        self.assertEqual(m.calls, [10])          # one batched call, not ten
        self.assertEqual(len(preds), 10)

    def test_batches_are_capped(self):
        m = _FakeModel()
        spans = [(i, i + 1.0) for i in range(70)]
        self._run(spans, m)
        self.assertEqual(m.calls, [32, 32, 6])

    def test_short_turns_stay_neutral_and_are_not_scored(self):
        # A turn under 200 ms is too short to score. It must still produce a
        # prediction, in the right position, without consuming a batch slot.
        m = _FakeModel()
        spans = [(0.0, 1.0), (1.0, 1.05), (2.0, 3.0)]
        preds, _ = self._run(spans, m)
        self.assertEqual(m.calls, [2])
        self.assertEqual([p.emotion for p in preds], ["angry", "neutral", "angry"])
        self.assertEqual(preds[1].confidence, 0.0)
        self.assertEqual([(p.start, p.end) for p in preds], spans)

    def test_labels_stay_aligned_with_their_turns(self):
        # The failure this guards: a misaligned batch would give every turn
        # someone else's emotion, and nothing downstream would notice.
        def label(arr):
            return "happy" if len(arr) > 16_000 else "angry"

        m = _FakeModel(per_clip_label=label)
        spans = [(0.0, 2.0), (3.0, 3.5), (5.0, 7.0), (8.0, 8.5)]
        preds, _ = self._run(spans, m)
        self.assertEqual([p.emotion for p in preds],
                         ["happy", "angry", "happy", "angry"])

    def test_a_wrong_length_reply_falls_back_per_clip(self):
        # Trusting the length would misalign every emotion after the gap, so a
        # shape mismatch must degrade to per-clip calls instead.
        m = _FakeModel(break_shape=True)
        spans = [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]
        preds, _ = self._run(spans, m)
        self.assertEqual(len(preds), 3)
        self.assertTrue(all(p.emotion == "angry" for p in preds))
        self.assertEqual(m.calls[0], 3)              # the batched attempt
        self.assertEqual(m.calls[1:], [1, 1, 1])     # then one per clip

    def test_confidence_floor_still_applies(self):
        class _Weak(_FakeModel):
            def generate(self, audio, granularity=None, extract_embedding=None):
                clips = audio if isinstance(audio, list) else [audio]
                self.calls.append(len(clips))
                return [{"labels": ["angry", "neutral"], "scores": [0.3, 0.7]}
                        for _ in clips]

        preds, _ = self._run([(0.0, 1.0), (2.0, 3.0)], _Weak())
        self.assertTrue(all(p.emotion == "neutral" for p in preds))

    def test_no_turns_is_not_an_error(self):
        m = _FakeModel()
        preds, _ = self._run([], m)
        self.assertEqual(preds, [])
        self.assertEqual(m.calls, [])


if __name__ == "__main__":
    unittest.main()
