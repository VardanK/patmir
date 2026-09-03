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


class LongClipIsolationTest(unittest.TestCase):
    """A long clip is scored alone, because batch cost is quadratic in it.

    The 2026-09-02 batching capped by COUNT, but emotion2vec's ALiBi bias is a
    `[batch * heads, T, T]` tensor (`base.py:716`, `.repeat(bn, 1, 1)`), so a
    batch's memory scales with the square of its longest member. Measured on a
    58-minute video: a 1.15 GiB single allocation and the emotion stage died of
    CUDA OOM. The speedup was measured on 24 turns of 1-8 s and is about
    amortizing per-call setup over SHORT clips; a long clip never needed it.
    """

    def _clips(self, seconds):
        import numpy as np
        return [np.zeros(int(s * 16_000), dtype="float32") for s in seconds]

    def test_short_clips_batch_together(self):
        from patmir.pipeline.stages.emotion.emotion2vec import _batches
        groups = list(_batches(self._clips([1, 1, 1, 1]), 4, 12 * 16_000))
        self.assertEqual([len(g) for _, g in groups], [4])

    def test_a_long_clip_is_alone(self):
        from patmir.pipeline.stages.emotion.emotion2vec import _batches
        groups = list(_batches(self._clips([1, 1, 30, 1]), 8, 12 * 16_000))
        self.assertEqual([len(g) for _, g in groups], [2, 1, 1])

    def test_offsets_stay_aligned_with_the_clip_list(self):
        """The offset indexes `scored_at`, so a wrong one mislabels every turn
        after it — the failure this cannot afford."""
        from patmir.pipeline.stages.emotion.emotion2vec import _batches
        clips = self._clips([1, 1, 30, 1, 1, 1])
        seen = []
        for offset, group in _batches(clips, 2, 12 * 16_000):
            for k, clip in enumerate(group):
                seen.append((offset + k, len(clip)))
        self.assertEqual([i for i, _ in seen], list(range(len(clips))))
        for i, n in seen:
            self.assertEqual(n, len(clips[i]))

    def test_every_clip_is_scored_exactly_once(self):
        from patmir.pipeline.stages.emotion.emotion2vec import _batches
        clips = self._clips([1, 20, 1, 1, 15, 1, 1, 1, 1])
        total = sum(len(g) for _, g in _batches(clips, 4, 12 * 16_000))
        self.assertEqual(total, len(clips))

    def test_a_run_of_long_clips_is_all_singletons(self):
        from patmir.pipeline.stages.emotion.emotion2vec import _batches
        groups = list(_batches(self._clips([20, 30, 40]), 8, 12 * 16_000))
        self.assertEqual([len(g) for _, g in groups], [1, 1, 1])

    def test_no_clips_yields_nothing(self):
        from patmir.pipeline.stages.emotion.emotion2vec import _batches
        self.assertEqual(list(_batches([], 4, 12 * 16_000)), [])


class OomFallbackTest(unittest.TestCase):
    """On CUDA OOM the batch halves and retries, down to one clip.

    The size cap is a heuristic and cannot know the real ceiling — it depends
    on the clip lengths, what else is resident on the card, and fragmentation.
    This does, by construction.
    """

    def _cand(self, fail_above):
        import numpy as np

        from patmir.pipeline.stages.emotion.emotion2vec import Emotion2VecCandidate
        cand = object.__new__(Emotion2VecCandidate)
        calls = []

        class _Model:
            def generate(self, group, **kw):
                calls.append(len(group))
                if len(group) > fail_above:
                    raise RuntimeError("CUDA out of memory. Tried to allocate 1.15 GiB")
                return [{"labels": ["neutral"], "scores": [0.9]} for _ in group]

        cand.model = _Model()
        return cand, calls, [np.zeros(16_000, dtype="float32") for _ in range(8)]

    def test_it_halves_until_it_fits(self):
        cand, calls, clips = self._cand(fail_above=2)
        out = cand._generate_group(clips)
        self.assertEqual(len(out), 8, "every clip must still be scored")
        self.assertIn(8, calls)
        self.assertIn(2, calls)

    def test_a_batch_that_fits_is_not_split(self):
        cand, calls, clips = self._cand(fail_above=8)
        cand._generate_group(clips)
        self.assertEqual(calls, [8])

    def test_a_single_clip_that_ooms_raises(self):
        """A genuinely exhausted card must not be silently swallowed."""
        cand, _calls, clips = self._cand(fail_above=0)
        with self.assertRaises(RuntimeError):
            cand._generate_group(clips[:1])

    def test_a_non_oom_error_is_re_raised_immediately(self):
        import numpy as np

        from patmir.pipeline.stages.emotion.emotion2vec import Emotion2VecCandidate
        cand = object.__new__(Emotion2VecCandidate)
        calls = []

        class _Model:
            def generate(self, group, **kw):
                calls.append(len(group))
                raise ValueError("something else entirely")

        cand.model = _Model()
        with self.assertRaises(ValueError):
            cand._generate_group([np.zeros(16_000, dtype="float32")] * 4)
        self.assertEqual(calls, [4], "a non-OOM failure must not be retried")

    def test_oom_is_recognised_in_both_spellings(self):
        from patmir.pipeline.stages.emotion.emotion2vec import _is_oom
        self.assertTrue(_is_oom(RuntimeError("CUDA out of memory. Tried...")))
        self.assertTrue(_is_oom(RuntimeError("cuda OUT OF MEMORY")))
        self.assertFalse(_is_oom(RuntimeError("shape mismatch")))
        self.assertFalse(_is_oom(ValueError("out of memory")))


class LongTurnWindowTest(unittest.TestCase):
    """A very long turn is scored on a bounded centred window.

    The ALiBi bias is quadratic in clip length, so a long turn is expensive on
    its own — batch or no batch. Measured: a **101.5 s** turn needed a
    **1.15 GiB** single allocation and the stage died, with the batch already
    down to one clip so no batching change could help. 16 of that job's 41
    turns were over 12 s and 3 over 60 s.

    Not a regression from the 2026-09-02 batching: the per-call code before it
    passed whole turns too. The batching blew up first on short clips and hid
    this.
    """

    def _clip(self, seconds):
        import numpy as np
        return np.arange(int(seconds * 16_000), dtype="float32")

    def test_a_clip_that_fits_is_untouched(self):
        from patmir.pipeline.stages.emotion.emotion2vec import _centre_window
        clip = self._clip(5)
        out = _centre_window(clip, 20 * 16_000)
        self.assertIs(out, clip)

    def test_a_clip_exactly_at_the_limit_is_untouched(self):
        from patmir.pipeline.stages.emotion.emotion2vec import _centre_window
        clip = self._clip(20)
        self.assertEqual(len(_centre_window(clip, 20 * 16_000)), len(clip))

    def test_the_measured_101_second_turn_is_bounded(self):
        from patmir.pipeline.stages.emotion.emotion2vec import _centre_window
        out = _centre_window(self._clip(101.5), 20 * 16_000)
        self.assertEqual(len(out), 20 * 16_000)

    def test_the_window_is_centred_not_the_head(self):
        """The start of a long turn is often a lead-in and the end a
        trail-off, so the middle is the most representative of sustained
        delivery. `_clip` is a ramp, so its values identify the position."""
        from patmir.pipeline.stages.emotion.emotion2vec import _centre_window
        clip = self._clip(60)
        out = _centre_window(clip, 20 * 16_000)
        expected_start = (len(clip) - 20 * 16_000) // 2
        self.assertEqual(float(out[0]), float(expected_start))
        self.assertGreater(float(out[0]), 0.0, "not the head of the clip")

    def test_a_short_clip_is_never_padded(self):
        # A 200 ms clip must stay 200 ms: padding would put silence into the
        # pooled utterance embedding, which is the thing that would make this
        # a quality change rather than a memory one.
        from patmir.pipeline.stages.emotion.emotion2vec import _centre_window
        clip = self._clip(0.25)
        self.assertEqual(len(_centre_window(clip, 20 * 16_000)), len(clip))

    def test_the_cap_keeps_the_allocation_small(self):
        """The point of the cap, expressed as the ALiBi bias size it implies.

        `[batch * heads, T, T]` at 4 bytes, ~50 frames/s, 12 heads: 101.5 s
        needs ~1.15 GiB (what failed), 20 s needs ~48 MB.
        """
        from patmir.pipeline.stages.emotion.emotion2vec import (
            EMOTION_MAX_CLIP_SECONDS,
        )
        frames = EMOTION_MAX_CLIP_SECONDS * 50
        gib = (12 * frames * frames * 4) / (1024 ** 3)
        self.assertLess(gib, 0.2, "the capped clip must be nowhere near 1.15 GiB")
