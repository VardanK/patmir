"""Tests for `apply_speaker_tags` — per-word speaker attribution and splitting.

Through 2026-08-24 this pass asked "does one speaker cover >= 85% of the
sentence?" and, failing that, split ONCE at the first turn whose speaker
differed from the sentence's opening speaker. Both halves broke on real
Sortformer output, which emits CONCURRENT turns: the boundary came from a turn
START, so a turn running concurrently with the opening speaker put the boundary a
few hundred ms into the sentence, the left half was one function word, the
non-lexical guard rejected the split, and the sentence fell back to its dominant
speaker — with another character's words in it.

Measured on `runs/20260824-163841-sample_1` (Reacher S02E01, 5 min): 13 of 76
sentences carried another speaker's words, 25 words of 393 in the wrong voice.
Two of those the author identified by ear. The shapes are reproduced here with
their real timings so a regression is recognisable as the thing it was.
"""
from __future__ import annotations

import unittest

from patmir.pipeline.stage import (
    Diarization,
    OverlapInterval,
    Sentence,
    SpeakerTurn,
    Transcript,
    Word,
)
from patmir.pipeline.stages.asr.speaker_tag import apply_speaker_tags


def _w(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end)


def _sent(words: list[Word]) -> Sentence:
    return Sentence(
        idx=0,
        text=" ".join(w.text for w in words),
        start=words[0].start,
        end=words[-1].end,
        words=words,
    )


def _tag(words: list[Word], turns: list[SpeakerTurn],
         overlaps: list[OverlapInterval] | None = None) -> list[Sentence]:
    diar = Diarization(turns=turns, overlaps=overlaps or [], refs={},
                       sample_rate=16_000)
    out = apply_speaker_tags(
        Transcript(language="ru", words=words, sentences=[_sent(words)]), diar)
    return out.sentences


def _pairs(sentences: list[Sentence]) -> list[tuple[str | None, str]]:
    return [(s.speaker_id, s.text) for s in sentences]


class MeasuredFailuresTest(unittest.TestCase):
    """The two cases the author picked out by ear, at their real timings."""

    def test_a_mans_line_folded_into_a_womans_sentence_is_split_out(self):
        """Output 1:01.5 — "На ней кровь" is SPEAKER_01 per word but was
        synthesized in SPEAKER_02's female voice. The old boundary landed at
        60.32 (0.06 s into the sentence) because a concurrent turn started
        there, leaving a one-word left half that the non-lexical guard threw
        away."""
        words = [
            _w("А", 60.26, 60.50), _w("с", 60.50, 60.58), _w("той", 60.58, 60.74),
            _w("что", 60.74, 60.90), _w("не", 60.90, 61.06), _w("так", 61.06, 61.46),
            _w("На", 61.62, 62.02), _w("ней", 62.02, 62.18), _w("кровь", 62.18, 62.42),
        ]
        turns = [
            SpeakerTurn(start=60.24, end=61.44, speaker_id="SPEAKER_03"),
            SpeakerTurn(start=60.32, end=61.52, speaker_id="SPEAKER_02"),
            SpeakerTurn(start=61.68, end=62.56, speaker_id="SPEAKER_01"),
        ]
        out = _tag(words, turns)
        self.assertEqual(out[-1].speaker_id, "SPEAKER_01")
        self.assertEqual(out[-1].text, "На ней кровь")
        self.assertNotIn("кровь", out[0].text)

    def test_a_third_characters_single_word_survives_as_its_own_line(self):
        """Output 1:39.5 — "Зачем" is SPEAKER_00 with 0.36 s of overlap and
        ZERO overlap with the sentence's own speaker, yet the 0.85 dominance
        gate (this sentence measured 0.811) swallowed it. A one-word run the
        diarizer places squarely on someone else must stand alone."""
        words = [
            _w("И", 97.08, 97.32), _w("деньги", 97.32, 97.64), _w("на", 97.64, 97.80),
            _w("билет", 97.80, 98.28), _w("Срочно", 98.60, 99.24),
            _w("лети", 99.24, 99.56), _w("НьюЙорк", 99.56, 100.12),
            _w("Зачем", 100.28, 100.68),
        ]
        turns = [
            SpeakerTurn(start=97.04, end=98.48, speaker_id="SPEAKER_02"),
            SpeakerTurn(start=98.72, end=100.24, speaker_id="SPEAKER_02"),
            SpeakerTurn(start=100.32, end=100.80, speaker_id="SPEAKER_00"),
        ]
        out = _tag(words, turns)
        self.assertEqual(_pairs(out)[-1], ("SPEAKER_00", "Зачем"))
        self.assertEqual(out[0].speaker_id, "SPEAKER_02")


class WobbleTest(unittest.TestCase):
    """A short run with the SAME speaker on both sides is a diarization wobble
    inside one person's utterance, not a turn."""

    def test_one_utterance_is_not_shredded_by_an_interior_wobble(self):
        """"он еще долго не очнется" came back as
        SPEAKER_01 | SPEAKER_00 x2 | SPEAKER_01 with the interior overlap an
        EXACT tie (0.32 s each way), so no overlap comparison can resolve it —
        only the A-B-A shape can."""
        words = [
            _w("еще", 21.76, 22.08), _w("долго", 22.08, 22.24),
            _w("не", 22.24, 22.40), _w("очнется", 22.40, 22.96),
        ]
        turns = [
            SpeakerTurn(start=21.52, end=22.08, speaker_id="SPEAKER_01"),
            SpeakerTurn(start=22.08, end=22.40, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=22.40, end=23.04, speaker_id="SPEAKER_01"),
        ]
        self.assertEqual(_pairs(_tag(words, turns)),
                         [("SPEAKER_01", "еще долго не очнется")])

    def test_a_genuine_three_way_exchange_keeps_its_middle_speaker(self):
        """The word cap is what separates a wobble from a real reply. Without it
        any A-B-A pattern would lose B's line."""
        words = [
            _w("a1", 0.0, 0.4), _w("a2", 0.4, 0.8),
            _w("b1", 1.0, 1.4), _w("b2", 1.4, 1.8), _w("b3", 1.8, 2.2),
            _w("a3", 2.4, 2.8), _w("a4", 2.8, 3.2),
        ]
        turns = [
            SpeakerTurn(start=0.0, end=0.9, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=0.95, end=2.3, speaker_id="SPEAKER_01"),
            SpeakerTurn(start=2.35, end=3.3, speaker_id="SPEAKER_00"),
        ]
        self.assertEqual(_pairs(_tag(words, turns)), [
            ("SPEAKER_00", "a1 a2"),
            ("SPEAKER_01", "b1 b2 b3"),
            ("SPEAKER_00", "a3 a4"),
        ])


class EdgeRunTest(unittest.TestCase):
    """A short run at a sentence edge has only one neighbour, so the decision is
    whether it fits that neighbour at least as well as its own label."""

    def test_a_leading_word_that_fits_its_own_label_better_is_kept(self):
        """"знать" scored 0.32 s against its own label and 0.02 against the
        neighbour. Scoring only the neighbour absorbed any short run with a
        non-zero neighbour overlap, which on a 1|2|1 pattern handed the whole
        sentence to the two-word middle run."""
        words = [_w("solo", 0.0, 0.5), _w("b1", 0.6, 1.0), _w("b2", 1.0, 1.4)]
        turns = [
            SpeakerTurn(start=0.0, end=0.5, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=0.55, end=1.5, speaker_id="SPEAKER_01"),
        ]
        self.assertEqual(_pairs(_tag(words, turns)),
                         [("SPEAKER_00", "solo"), ("SPEAKER_01", "b1 b2")])

    def test_a_leading_word_the_neighbour_covers_better_is_absorbed(self):
        words = [_w("lead", 0.0, 0.5), _w("b1", 0.6, 1.0), _w("b2", 1.0, 1.4)]
        turns = [
            SpeakerTurn(start=0.0, end=0.10, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=0.05, end=1.5, speaker_id="SPEAKER_01"),
        ]
        self.assertEqual(_pairs(_tag(words, turns)),
                         [("SPEAKER_01", "lead b1 b2")])

    def test_a_non_lexical_fragment_is_absorbed_whatever_it_overlaps(self):
        """An orphaned interjection must never become its own sentence for
        translate and TTS, even when the diarizer is confident about it."""
        words = [_w("Ха", 0.0, 0.4), _w("настоящие", 0.6, 1.2), _w("слова", 1.2, 1.8)]
        turns = [
            SpeakerTurn(start=0.0, end=0.45, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=0.55, end=2.0, speaker_id="SPEAKER_01"),
        ]
        self.assertEqual(_pairs(_tag(words, turns)),
                         [("SPEAKER_01", "Ха настоящие слова")])

    def test_a_zero_span_fragment_is_absorbed(self):
        """Real aligner output contains zero-length words — 13 of them on the
        measured clip. Emitted alone one becomes a zero-duration sentence for
        the mux to place."""
        words = [_w("real", 0.0, 0.6), _w("сэр", 0.62, 0.62)]
        turns = [
            SpeakerTurn(start=0.0, end=0.60, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=0.61, end=0.63, speaker_id="SPEAKER_01"),
        ]
        out = _tag(words, turns)
        self.assertEqual(_pairs(out), [("SPEAKER_00", "real сэр")])
        self.assertGreater(out[0].end - out[0].start, 0.0)


class GeneralBehaviourTest(unittest.TestCase):
    def test_a_single_speaker_sentence_is_untouched(self):
        words = [_w("все", 0.0, 0.3), _w("хорошо", 0.3, 0.8)]
        turns = [SpeakerTurn(start=0.0, end=1.0, speaker_id="SPEAKER_02")]
        out = _tag(words, turns)
        self.assertEqual(_pairs(out), [("SPEAKER_02", "все хорошо")])
        self.assertEqual(out[0].words, words)

    def test_three_speakers_in_one_sentence_all_split(self):
        """The split-once version could emit at most two fragments however many
        speakers a sentence covered."""
        words = [
            _w("a1", 0.0, 0.4), _w("a2", 0.4, 0.8),
            _w("b1", 1.0, 1.4), _w("b2", 1.4, 1.8),
            _w("c1", 2.0, 2.4), _w("c2", 2.4, 2.8),
        ]
        turns = [
            SpeakerTurn(start=0.0, end=0.9, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=0.95, end=1.9, speaker_id="SPEAKER_01"),
            SpeakerTurn(start=1.95, end=2.9, speaker_id="SPEAKER_02"),
        ]
        self.assertEqual(_pairs(_tag(words, turns)), [
            ("SPEAKER_00", "a1 a2"),
            ("SPEAKER_01", "b1 b2"),
            ("SPEAKER_02", "c1 c2"),
        ])

    def test_indices_are_contiguous_across_splits(self):
        words_a = [_w("a1", 0.0, 0.4), _w("a2", 0.4, 0.8),
                   _w("b1", 1.0, 1.4), _w("b2", 1.4, 1.8)]
        words_b = [_w("c1", 3.0, 3.4), _w("c2", 3.4, 3.8)]
        turns = [
            SpeakerTurn(start=0.0, end=0.9, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=0.95, end=1.9, speaker_id="SPEAKER_01"),
            SpeakerTurn(start=2.9, end=4.0, speaker_id="SPEAKER_00"),
        ]
        diar = Diarization(turns=turns, overlaps=[], refs={}, sample_rate=16_000)
        out = apply_speaker_tags(
            Transcript(language="ru", words=words_a + words_b,
                       sentences=[_sent(words_a), _sent(words_b)]), diar)
        self.assertEqual([s.idx for s in out.sentences], [0, 1, 2])

    def test_a_word_in_a_gap_between_turns_inherits_rather_than_splitting(self):
        """A hole in the diarization is not a speaker change."""
        words = [_w("a1", 0.0, 0.4), _w("gap", 1.0, 1.4), _w("a2", 2.0, 2.4)]
        turns = [
            SpeakerTurn(start=0.0, end=0.5, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=1.9, end=2.5, speaker_id="SPEAKER_00"),
        ]
        self.assertEqual(_pairs(_tag(words, turns)),
                         [("SPEAKER_00", "a1 gap a2")])

    def test_a_leading_word_before_any_turn_inherits_backwards(self):
        words = [_w("lead", 0.0, 0.4), _w("a1", 1.0, 1.4), _w("a2", 1.4, 1.8)]
        turns = [SpeakerTurn(start=0.9, end=2.0, speaker_id="SPEAKER_01")]
        self.assertEqual(_pairs(_tag(words, turns)),
                         [("SPEAKER_01", "lead a1 a2")])

    def test_a_sentence_no_turn_covers_gets_no_speaker(self):
        words = [_w("a1", 50.0, 50.4), _w("a2", 50.4, 50.8)]
        turns = [SpeakerTurn(start=0.0, end=1.0, speaker_id="SPEAKER_00")]
        out = _tag(words, turns)
        self.assertEqual(len(out), 1)
        self.assertIsNone(out[0].speaker_id)

    def test_a_wholly_non_lexical_sentence_is_dropped(self):
        words = [_w("Ха-ха-ха", 0.0, 0.8)]
        turns = [SpeakerTurn(start=0.0, end=1.0, speaker_id="SPEAKER_00")]
        self.assertEqual(_tag(words, turns), [])

    def test_a_sentence_with_no_word_timings_falls_back_to_dominant(self):
        """The Qwen3 candidates emit a chunk-level Sentence with no words when
        the forced aligner returns nothing. There is nothing to split on."""
        sent = Sentence(idx=0, text="без слов", start=0.0, end=4.0, words=[])
        turns = [
            SpeakerTurn(start=0.0, end=3.0, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=3.0, end=4.0, speaker_id="SPEAKER_01"),
        ]
        diar = Diarization(turns=turns, overlaps=[], refs={}, sample_rate=16_000)
        out = apply_speaker_tags(
            Transcript(language="ru", words=[], sentences=[sent]), diar)
        self.assertEqual(_pairs(out.sentences), [("SPEAKER_00", "без слов")])

    def test_an_exact_overlap_tie_resolves_to_the_lower_speaker_id(self):
        """Determinism: two runs of the same clip must tag identically."""
        words = [_w("tie", 1.0, 2.0)]
        turns = [
            SpeakerTurn(start=1.0, end=2.0, speaker_id="SPEAKER_03"),
            SpeakerTurn(start=1.0, end=2.0, speaker_id="SPEAKER_01"),
        ]
        self.assertEqual(_tag(words, turns)[0].speaker_id, "SPEAKER_01")

    def test_transcript_word_list_is_passed_through_unchanged(self):
        words = [_w("a1", 0.0, 0.4), _w("b1", 1.0, 1.4)]
        turns = [
            SpeakerTurn(start=0.0, end=0.9, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=0.95, end=2.0, speaker_id="SPEAKER_01"),
        ]
        diar = Diarization(turns=turns, overlaps=[], refs={}, sample_rate=16_000)
        out = apply_speaker_tags(
            Transcript(language="ru", words=words, sentences=[_sent(words)]), diar)
        self.assertEqual(out.words, words)
        self.assertEqual(out.language, "ru")

    def test_split_fragments_carry_their_own_word_spans(self):
        words = [_w("a1", 0.0, 0.4), _w("a2", 0.4, 0.8),
                 _w("b1", 1.0, 1.4), _w("b2", 1.4, 1.8)]
        turns = [
            SpeakerTurn(start=0.0, end=0.9, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=0.95, end=2.0, speaker_id="SPEAKER_01"),
        ]
        out = _tag(words, turns)
        self.assertEqual((out[0].start, out[0].end), (0.0, 0.8))
        self.assertEqual((out[1].start, out[1].end), (1.0, 1.8))
        self.assertEqual([w.text for w in out[1].words], ["b1", "b2"])


class NotEvidenceTest(unittest.TestCase):
    """Three situations where the diarization is present but is not evidence of
    who is speaking. In each the word must inherit from its neighbours rather
    than manufacture a turn boundary inside somebody's sentence.
    """

    def test_words_inside_an_overlap_interval_do_not_split_a_sentence(self):
        """Where the diarizer reports two concurrent speakers, max-overlap is a
        coin flip. Measured: "Он еще | долго не очнется" and "Я получил твое |
        сообщение" were each broken in half by turns running concurrently for
        one phrase — the same artifact as a duplicated speaker slot, seen
        locally, which `merge_duplicate_speakers` cannot catch because it tests
        aggregate containment across the whole clip."""
        words = [_w("Я", 91.64, 91.80), _w("получил", 91.80, 92.28),
                 _w("твое", 92.28, 92.52), _w("сообщение", 92.52, 93.00)]
        turns = [
            SpeakerTurn(start=91.76, end=92.64, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=91.84, end=93.04, speaker_id="SPEAKER_01"),
        ]
        overlaps = [OverlapInterval(start=91.84, end=92.64)]
        self.assertEqual(_pairs(_tag(words, turns, overlaps)),
                         [("SPEAKER_01", "Я получил твое сообщение")])

    def test_without_overlap_intervals_the_rule_is_inert(self):
        """Not every diarize candidate emits overlaps. The same input must still
        be attributed, just without the guard."""
        words = [_w("Я", 91.64, 91.80), _w("получил", 91.80, 92.28),
                 _w("твое", 92.28, 92.52), _w("сообщение", 92.52, 93.00)]
        turns = [
            SpeakerTurn(start=91.76, end=92.64, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=91.84, end=93.04, speaker_id="SPEAKER_01"),
        ]
        guarded = _tag(words, turns, [OverlapInterval(start=91.84, end=92.64)])
        bare = _tag(words, turns)
        self.assertEqual(len(guarded), 1)
        # Without the intervals the same input splits — the guard is what fixes
        # it, and every word still gets attributed.
        self.assertEqual(len(bare), 2)
        self.assertTrue(all(x.speaker_id is not None for x in bare))

    def test_a_sliver_of_a_turn_is_not_evidence(self):
        """"Я" at 91.64-91.80 caught 0.04 s of a turn that starts AFTER the word
        does — 25% coverage — and that scrap alone gave the word a different
        speaker from the rest of its sentence."""
        words = [_w("Я", 91.64, 91.80), _w("получил", 91.90, 92.30),
                 _w("сообщение", 92.30, 92.80)]
        turns = [
            SpeakerTurn(start=91.76, end=91.84, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=91.84, end=93.04, speaker_id="SPEAKER_01"),
        ]
        self.assertEqual(_pairs(_tag(words, turns)),
                         [("SPEAKER_01", "Я получил сообщение")])

    def test_a_word_the_best_turn_covers_fully_is_evidence(self):
        """The coverage floor must not swallow a well-supported word."""
        words = [_w("a1", 0.0, 0.5), _w("a2", 0.5, 1.0),
                 _w("b1", 1.1, 1.6), _w("b2", 1.6, 2.1)]
        turns = [
            SpeakerTurn(start=0.0, end=1.05, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=1.05, end=2.2, speaker_id="SPEAKER_01"),
        ]
        self.assertEqual(_pairs(_tag(words, turns)),
                         [("SPEAKER_00", "a1 a2"), ("SPEAKER_01", "b1 b2")])


class FragmentDurationFloorTest(unittest.TestCase):
    """A run too short to synthesize is absorbed whatever the attribution says.

    The measured margin is narrow and both ends are real, so both are pinned:
    "кровь" (0.24 s) is diarization splitting one man's phrase across two
    ADJACENT turns — no overlap, full coverage, so neither the overlap rule nor
    the coverage rule sees it — and must be absorbed; "Зачем" (0.40 s) is a
    genuine one-word line by a third character and must survive.
    """

    def test_a_quarter_second_fragment_is_absorbed_despite_fitting_its_own_turn(self):
        words = [_w("На", 61.62, 62.02), _w("ней", 62.02, 62.18),
                 _w("кровь", 62.18, 62.42)]
        turns = [
            SpeakerTurn(start=61.68, end=62.16, speaker_id="SPEAKER_01"),
            SpeakerTurn(start=62.16, end=62.48, speaker_id="SPEAKER_00"),
        ]
        self.assertEqual(_pairs(_tag(words, turns)),
                         [("SPEAKER_01", "На ней кровь")])

    def test_a_four_hundred_millisecond_line_still_survives(self):
        words = [_w("НьюЙорк", 99.56, 100.12), _w("Зачем", 100.28, 100.68)]
        turns = [
            SpeakerTurn(start=98.72, end=100.24, speaker_id="SPEAKER_02"),
            SpeakerTurn(start=100.32, end=100.80, speaker_id="SPEAKER_00"),
        ]
        self.assertEqual(_pairs(_tag(words, turns)),
                         [("SPEAKER_02", "НьюЙорк"), ("SPEAKER_00", "Зачем")])

    def test_no_emitted_sentence_is_shorter_than_the_floor_unless_it_arrived_that_way(self):
        """A split must never produce a fragment below the floor. A whole input
        sentence that is already shorter is passed through untouched — that is
        upstream's doing, not a split."""
        words = [_w("a1", 0.0, 0.5), _w("a2", 0.5, 1.0), _w("tiny", 1.0, 1.10)]
        turns = [
            SpeakerTurn(start=0.0, end=1.0, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=1.0, end=1.2, speaker_id="SPEAKER_01"),
        ]
        out = _tag(words, turns)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].speaker_id, "SPEAKER_00")


class InheritanceIsConstrainedTest(unittest.TestCase):
    """The second pass must not hand a deferred word to a speaker who does not
    cover it.

    Plain forward-fill was tried first and produced exactly that: "Он еще" went
    to the speaker of the preceding word — who has ZERO coverage of either word —
    because that word happened to be adjacent, which tore the phrase off its own
    sentence and left it in the mother's voice. Both shapes below are the real
    ones, at their real timings and coverages.
    """

    def test_a_deferred_word_prefers_a_neighbour_that_actually_covers_it(self):
        """"Он" and "еще" sit inside an overlap with two qualifying speakers
        (0.80/0.60 and 1.00/1.00), so both defer. The preceding word belongs to a
        third speaker who covers neither; the following word belongs to one of
        their own candidates, and that is who must win."""
        words = [
            _w("малыш", 20.64, 21.28),
            _w("Он", 21.36, 21.76), _w("еще", 21.76, 22.08),
            _w("долго", 22.08, 22.32), _w("не", 22.32, 22.40),
            _w("очнется", 22.40, 22.96),
        ]
        turns = [
            SpeakerTurn(start=18.24, end=21.36, speaker_id="SPEAKER_02"),
            SpeakerTurn(start=21.44, end=22.16, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=21.52, end=23.04, speaker_id="SPEAKER_01"),
            SpeakerTurn(start=22.48, end=22.64, speaker_id="SPEAKER_00"),
        ]
        overlaps = [OverlapInterval(start=21.52, end=22.16),
                    OverlapInterval(start=22.48, end=22.64)]
        self.assertEqual(_pairs(_tag(words, turns, overlaps)), [
            ("SPEAKER_02", "малыш"),
            ("SPEAKER_01", "Он еще долго не очнется"),
        ])

    def test_with_no_plausible_neighbour_a_deferred_word_takes_its_own_best(self):
        """"Сигнал бедствия" opens its sentence, so there is no preceding label,
        and the following word belongs to a speaker who covers neither word. It
        must fall back to its own best candidate rather than inherit forwards."""
        words = [
            _w("Сигнал", 95.80, 96.28), _w("бедствия", 96.28, 96.76),
            _w("И", 97.08, 97.32), _w("деньги", 97.32, 97.64),
        ]
        turns = [
            SpeakerTurn(start=95.84, end=96.88, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=95.92, end=96.56, speaker_id="SPEAKER_01"),
            SpeakerTurn(start=97.04, end=100.16, speaker_id="SPEAKER_02"),
        ]
        overlaps = [OverlapInterval(start=95.92, end=96.56)]
        self.assertEqual(_pairs(_tag(words, turns, overlaps)), [
            ("SPEAKER_00", "Сигнал бедствия"),
            ("SPEAKER_02", "И деньги"),
        ])

    def test_one_qualifying_speaker_inside_an_overlap_is_still_evidence(self):
        """The overlap guard needs TWO speakers clearing the coverage floor. A
        word that merely sits near concurrent speech, with only one speaker
        actually covering it, is attributed normally."""
        words = [_w("a1", 0.0, 0.5), _w("solo", 0.5, 1.0), _w("b1", 1.2, 1.7)]
        turns = [
            SpeakerTurn(start=0.0, end=1.05, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=0.95, end=1.8, speaker_id="SPEAKER_01"),
        ]
        overlaps = [OverlapInterval(start=0.95, end=1.05)]
        out = _tag(words, turns, overlaps)
        self.assertEqual(_pairs(out),
                         [("SPEAKER_00", "a1 solo"), ("SPEAKER_01", "b1")])

    def test_a_word_with_no_candidate_at_all_inherits_unconditionally(self):
        """The sliver case: nothing clears the floor, so there is no candidate to
        constrain the inheritance and any neighbour will do."""
        words = [_w("Я", 91.64, 91.80), _w("сообщение", 92.52, 93.00)]
        turns = [
            SpeakerTurn(start=91.76, end=91.84, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=91.84, end=93.04, speaker_id="SPEAKER_01"),
        ]
        self.assertEqual(_pairs(_tag(words, turns)),
                         [("SPEAKER_01", "Я сообщение")])


class DeferredBlockIsOneDecisionTest(unittest.TestCase):
    """A maximal run of consecutive deferred words is resolved as ONE decision.

    Resolving word by word lets a single phrase alternate. All five words of
    "Он еще долго не очнется" defer — each is covered by two speakers inside one
    overlap interval — and picking the larger per-word coverage independently
    gave SPEAKER_01 | SPEAKER_00 x3 | SPEAKER_01: one utterance in three voices,
    which is worse than the mis-attribution it replaced. Summed across the block
    the same numbers give SPEAKER_01 by 1.52 s to 1.36 s.

    These are the real turns from `runs/20260824-181110-sample_1`; the same
    phrase resolves the same way against three other runs' diarization of the
    identical file, which disagree on turn count (98/102/110/111) and on where
    the boundaries fall.
    """

    def test_a_phrase_wholly_inside_an_overlap_gets_one_voice(self):
        words = [
            _w("тобой", 20.00, 20.56), _w("малыш", 20.64, 21.28),
            _w("Он", 21.36, 21.76), _w("еще", 21.76, 22.08),
            _w("долго", 22.08, 22.32), _w("не", 22.32, 22.40),
            _w("очнется", 22.40, 22.96),
        ]
        turns = [
            SpeakerTurn(start=18.24, end=21.36, speaker_id="SPEAKER_02"),
            SpeakerTurn(start=21.44, end=23.04, speaker_id="SPEAKER_01"),
            SpeakerTurn(start=21.52, end=22.88, speaker_id="SPEAKER_00"),
        ]
        overlaps = [OverlapInterval(start=21.52, end=22.88)]
        self.assertEqual(_pairs(_tag(words, turns, overlaps)), [
            ("SPEAKER_02", "тобой малыш"),
            ("SPEAKER_01", "Он еще долго не очнется"),
        ])

    def test_a_deferred_block_still_yields_to_a_plausible_neighbour(self):
        """Block resolution must not override the plausibility constraint: when a
        boundary neighbour covers the block, it wins over the block's own
        summed-coverage favourite."""
        words = [
            _w("Он", 21.36, 21.76), _w("еще", 21.76, 22.08),
            _w("долго", 22.08, 22.32),
        ]
        turns = [
            SpeakerTurn(start=21.44, end=22.16, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=21.52, end=23.04, speaker_id="SPEAKER_01"),
        ]
        overlaps = [OverlapInterval(start=21.52, end=22.16)]
        out = _tag(words, turns, overlaps)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].speaker_id, "SPEAKER_01")
