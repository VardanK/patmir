"""Tests for the Sortformer diarize post-processing and the NeMo return parser.

Two things are covered here, both previously untested:

* `_postprocess` — the label normalization, same-speaker merge, overlap
  derivation and VAD filter shared by the MLX and NeMo runtimes of the same
  model. These decide the speaker ids and turn boundaries the ref-picker and
  `apply_speaker_tags` build on, so a regression here surfaces as characters
  sharing a voice, not as an exception.

* `sortformer_nemo.parse_segment` / `iter_segments` — NeMo's `diarize()` return
  type is documented only as segments "formatted as 'begin_seconds,
  end_seconds, speaker_index'"; no Sortformer model card shows a literal
  example, so the parser accepts every plausible shape and must REFUSE anything
  else rather than guess.
"""
from __future__ import annotations

import unittest

from patmir.pipeline.stage import SpeakerTurn
from patmir.pipeline.stages.diarize._postprocess import (
    derive_overlaps,
    merge_duplicate_speakers,
    merge_same_speaker,
    norm_speaker,
    passes_vad,
)
from patmir.pipeline.stages.diarize.sortformer_nemo import iter_segments, parse_segment


class NormSpeakerTest(unittest.TestCase):
    def test_every_runtime_label_lands_on_the_same_convention(self):
        """mlx-audio yields ints or `speaker_0`; NeMo yields an index. Downstream
        code keys refs off this string, so it has to be one form."""
        for raw, expected in [
            (0, "SPEAKER_00"),
            (3, "SPEAKER_03"),
            ("0", "SPEAKER_00"),
            ("speaker_1", "SPEAKER_01"),
            ("SPEAKER_2", "SPEAKER_02"),
            (12, "SPEAKER_12"),
        ]:
            self.assertEqual(norm_speaker(raw), expected, raw)

    def test_label_with_no_digits_passes_through(self):
        self.assertEqual(norm_speaker("unknown"), "unknown")


class MergeSameSpeakerTest(unittest.TestCase):
    def test_adjacent_same_speaker_within_gap_merges(self):
        turns = [
            SpeakerTurn(0.0, 1.0, "SPEAKER_00"),
            SpeakerTurn(1.2, 2.0, "SPEAKER_00"),
        ]
        merged = merge_same_speaker(turns, gap=0.5)
        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0].start, merged[0].end), (0.0, 2.0))

    def test_gap_larger_than_threshold_stays_split(self):
        turns = [
            SpeakerTurn(0.0, 1.0, "SPEAKER_00"),
            SpeakerTurn(2.0, 3.0, "SPEAKER_00"),
        ]
        self.assertEqual(len(merge_same_speaker(turns, gap=0.5)), 2)

    def test_different_speakers_never_merge(self):
        turns = [
            SpeakerTurn(0.0, 1.0, "SPEAKER_00"),
            SpeakerTurn(1.1, 2.0, "SPEAKER_01"),
        ]
        self.assertEqual(len(merge_same_speaker(turns, gap=0.5)), 2)

    def test_unsorted_input_is_sorted_first(self):
        turns = [
            SpeakerTurn(2.0, 3.0, "SPEAKER_00"),
            SpeakerTurn(0.0, 1.0, "SPEAKER_00"),
        ]
        merged = merge_same_speaker(turns, gap=1.5)
        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0].start, merged[0].end), (0.0, 3.0))

    def test_a_contained_turn_does_not_shorten_the_merged_span(self):
        """`end` takes the max — a fragment inside an existing span must not
        pull the turn's end backwards."""
        turns = [
            SpeakerTurn(0.0, 5.0, "SPEAKER_00"),
            SpeakerTurn(1.0, 2.0, "SPEAKER_00"),
        ]
        merged = merge_same_speaker(turns, gap=0.5)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].end, 5.0)

    def test_empty_input(self):
        self.assertEqual(merge_same_speaker([], gap=0.5), [])


class DeriveOverlapsTest(unittest.TestCase):
    def test_two_speakers_talking_at_once(self):
        turns = [
            SpeakerTurn(0.0, 2.0, "SPEAKER_00"),
            SpeakerTurn(1.0, 3.0, "SPEAKER_01"),
        ]
        overlaps = derive_overlaps(turns)
        self.assertEqual(len(overlaps), 1)
        self.assertEqual((overlaps[0].start, overlaps[0].end), (1.0, 2.0))

    def test_sequential_speakers_produce_no_overlap(self):
        turns = [
            SpeakerTurn(0.0, 1.0, "SPEAKER_00"),
            SpeakerTurn(1.0, 2.0, "SPEAKER_01"),
        ]
        self.assertEqual(derive_overlaps(turns), [])

    def test_three_way_overlap_is_one_interval(self):
        """The count crossing 2 opens the window and dropping below 2 closes it,
        so a third voice joining mid-window must not split it."""
        turns = [
            SpeakerTurn(0.0, 5.0, "SPEAKER_00"),
            SpeakerTurn(1.0, 4.0, "SPEAKER_01"),
            SpeakerTurn(2.0, 3.0, "SPEAKER_02"),
        ]
        overlaps = derive_overlaps(turns)
        self.assertEqual(len(overlaps), 1)
        self.assertEqual((overlaps[0].start, overlaps[0].end), (1.0, 4.0))

    def test_touching_turns_do_not_count_as_overlap(self):
        """One speaker ending exactly where another begins is a handover."""
        turns = [
            SpeakerTurn(0.0, 1.0, "SPEAKER_00"),
            SpeakerTurn(1.0, 2.0, "SPEAKER_01"),
            SpeakerTurn(2.0, 3.0, "SPEAKER_00"),
        ]
        self.assertEqual(derive_overlaps(turns), [])

    def test_empty_input(self):
        self.assertEqual(derive_overlaps([]), [])


class PassesVadTest(unittest.TestCase):
    def test_no_vad_intervals_keeps_everything(self):
        """The filter refines the diarizer's output; it is not a precondition."""
        self.assertTrue(passes_vad(0.0, 1.0, None))
        self.assertTrue(passes_vad(0.0, 1.0, []))

    def test_any_intersection_passes(self):
        self.assertTrue(passes_vad(0.5, 1.5, [(1.0, 2.0)]))

    def test_disjoint_turn_is_dropped(self):
        self.assertFalse(passes_vad(0.0, 0.5, [(1.0, 2.0)]))

    def test_touching_but_not_overlapping_is_dropped(self):
        self.assertFalse(passes_vad(0.0, 1.0, [(1.0, 2.0)]))


class ParseSegmentTest(unittest.TestCase):
    """The documented format is 'begin_seconds, end_seconds, speaker_index' but
    no model card shows the literal type, so all three plausible shapes parse."""

    def test_comma_separated_string(self):
        self.assertEqual(parse_segment("0.5, 2.25, 1"), (0.5, 2.25, "1"))

    def test_whitespace_separated_string(self):
        self.assertEqual(parse_segment("0.5 2.25 speaker_1"), (0.5, 2.25, "speaker_1"))

    def test_string_with_padding(self):
        self.assertEqual(parse_segment("  0.5 ,  2.25 , 1  "), (0.5, 2.25, "1"))

    def test_tuple_form(self):
        self.assertEqual(parse_segment((0.5, 2.25, 1)), (0.5, 2.25, 1))

    def test_object_form(self):
        class Seg:
            start, end, speaker = 0.5, 2.25, 1

        self.assertEqual(parse_segment(Seg()), (0.5, 2.25, 1))

    def test_object_with_speaker_id_attribute(self):
        class Seg:
            start, end, speaker_id = 0.5, 2.25, 2

        self.assertEqual(parse_segment(Seg()), (0.5, 2.25, 2))

    def test_wrong_field_count_raises_with_the_offending_value(self):
        """A format surprise has to fail loudly: a silently mis-parsed
        diarization shows up much later as two characters sharing a voice."""
        with self.assertRaises(ValueError) as ctx:
            parse_segment("0.5 2.25")
        self.assertIn("0.5 2.25", str(ctx.exception))

    def test_non_numeric_times_raise(self):
        with self.assertRaises(ValueError):
            parse_segment("start end spk")

    def test_unrecognized_type_raises(self):
        with self.assertRaises(ValueError):
            parse_segment(42)


class IterSegmentsTest(unittest.TestCase):
    def test_nested_per_file_shape(self):
        """What the model card's `predicted_segments[0]` example implies."""
        self.assertEqual(
            iter_segments([["0.0 1.0 0", "1.0 2.0 1"]]),
            ["0.0 1.0 0", "1.0 2.0 1"],
        )

    def test_flat_shape(self):
        self.assertEqual(
            iter_segments(["0.0 1.0 0", "1.0 2.0 1"]),
            ["0.0 1.0 0", "1.0 2.0 1"],
        )

    def test_flat_tuple_segments_are_not_mistaken_for_groups(self):
        segs = [(0.0, 1.0, 0), (1.0, 2.0, 1)]
        self.assertEqual(iter_segments(segs), segs)

    def test_nested_group_of_exactly_three_is_not_mistaken_for_a_segment(self):
        """The one real ambiguity: a 3-element sequence is both a valid segment
        and a valid group of three. Times distinguish them."""
        self.assertEqual(
            iter_segments([["0.0 1.0 0", "1.0 2.0 1", "2.0 3.0 0"]]),
            ["0.0 1.0 0", "1.0 2.0 1", "2.0 3.0 0"],
        )

    def test_nested_group_of_three_tuple_segments(self):
        segs = [(0.0, 1.0, 0), (1.0, 2.0, 1), (2.0, 3.0, 0)]
        self.assertEqual(iter_segments([segs]), segs)

    def test_single_tuple_segment_stays_flat(self):
        self.assertEqual(iter_segments([(0.0, 1.0, 0)]), [(0.0, 1.0, 0)])

    def test_empty_and_none(self):
        self.assertEqual(iter_segments([]), [])
        self.assertEqual(iter_segments(None), [])

    def test_nested_with_an_empty_group(self):
        self.assertEqual(iter_segments([[]]), [])


class ObservedNeMoShapeTest(unittest.TestCase):
    """The real thing, captured from NeMo on the RTX 3090 (2026-08-23).

    Until this ran, `diarize()`'s return type was documented only in prose
    ("begin_seconds, end_seconds, speaker_index") on all three Sortformer model
    cards, none of which showed a literal example. The observed shape is:

        [['0.000 1.840 speaker_0', '2.000 3.680 speaker_1', ...]]

    i.e. a list per input file, of **whitespace**-separated strings — not the
    comma-separated form the prose suggests — with `speaker_N` labels rather than
    bare integers. Pinned here so a NeMo upgrade that changes it fails loudly.
    """

    RAW = [[
        "0.000 1.840 speaker_0",
        "2.000 3.680 speaker_1",
        "3.840 5.280 speaker_2",
    ]]

    def test_end_to_end_through_the_real_shape(self):
        segments = iter_segments(self.RAW)
        self.assertEqual(len(segments), 3)
        turns = [
            SpeakerTurn(*parse_segment(seg)[:2], norm_speaker(parse_segment(seg)[2]))
            for seg in segments
        ]
        self.assertEqual(
            [(t.start, t.end, t.speaker_id) for t in turns],
            [(0.0, 1.84, "SPEAKER_00"),
             (2.0, 3.68, "SPEAKER_01"),
             (3.84, 5.28, "SPEAKER_02")],
        )
        # Distinct speakers, non-overlapping -> nothing merges, no overlaps.
        self.assertEqual(len(merge_same_speaker(turns, 0.5)), 3)
        self.assertEqual(derive_overlaps(turns), [])

    def test_the_ambiguous_case_was_not_hypothetical(self):
        """The first real call returned a nested list of EXACTLY three string
        segments — both a valid group of three and, structurally, a valid
        3-element segment. `_is_segment` disambiguating on whether the first two
        elements are times is what makes this parse correctly, and it was load
        bearing on call one."""
        self.assertEqual(len(self.RAW), 1)
        self.assertEqual(len(self.RAW[0]), 3)
        self.assertEqual(iter_segments(self.RAW), self.RAW[0])


class RuntimeParityTest(unittest.TestCase):
    """The MLX and NeMo candidates are two runtimes for ONE model. They must
    share post-processing rather than each carrying a copy that can drift."""

    def test_both_candidates_use_the_shared_postprocess_module(self):
        from patmir.pipeline.stages.diarize import _postprocess, sortformer_mlx, sortformer_nemo

        for module in (sortformer_mlx, sortformer_nemo):
            for fn in ("norm_speaker", "merge_same_speaker", "passes_vad", "derive_overlaps"):
                self.assertIs(
                    getattr(module, fn), getattr(_postprocess, fn),
                    f"{module.__name__}.{fn} is not the shared implementation",
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class MergeDuplicateSpeakersTest(unittest.TestCase):
    """Sortformer sometimes emits ONE person on TWO of its output slots.

    That is not cosmetic: the actor's lines split across two ref cells (so the
    cloned voice changes mid-scene), the twin pair is reported as overlapping
    speech (so `pick_refs` rejects every turn that touches it and the speaker can
    end up with no ref at all), and `apply_speaker_tags` sees a speaker change a
    few hundred ms into a sentence. All three were observed together on
    `runs/20260824-163841-sample_1`.
    """

    def test_two_slots_of_one_person_merge(self):
        turns = [
            SpeakerTurn(start=0.0, end=4.0, speaker_id="SPEAKER_02"),
            SpeakerTurn(start=0.1, end=3.9, speaker_id="SPEAKER_03"),
        ]
        out, aliases = merge_duplicate_speakers(turns, 0.5)
        self.assertEqual(aliases, {"SPEAKER_03": "SPEAKER_02"})
        self.assertEqual({t.speaker_id for t in out}, {"SPEAKER_02"})

    def test_genuinely_distinct_speakers_are_left_alone(self):
        """Real simultaneous speech is partial and ragged. The measured pair of
        distinct speakers on that clip sat at 8-10% mutual concurrency; the
        duplicated pair sat at 52% / 71%."""
        turns = [
            SpeakerTurn(start=0.0, end=10.0, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=9.5, end=20.0, speaker_id="SPEAKER_01"),
        ]
        out, aliases = merge_duplicate_speakers(turns, 0.5)
        self.assertEqual(aliases, {})
        self.assertEqual(len(out), 2)

    def test_the_default_threshold_brackets_every_observed_ratio(self):
        """The model is NOT deterministic: four runs of the identical file put the
        duplicated pair at 0.705, 0.594 and twice above 0.6, while the highest a
        genuinely-distinct pair reached on those same runs was 0.259. A first cut
        at 0.6 was calibrated on one run and silently stopped firing on the next
        — the 0.594 run came back with four speakers and one again borrowing
        another character's reference clip. Both bounds are pinned here so the
        default cannot drift back onto the edge of either."""
        def pair(containment: float) -> list[SpeakerTurn]:
            """Two 10 s speakers overlapping for exactly `containment` of one."""
            shift = 10.0 * (1.0 - containment)
            return [
                SpeakerTurn(start=0.0, end=10.0, speaker_id="SPEAKER_02"),
                SpeakerTurn(start=shift, end=shift + 10.0, speaker_id="SPEAKER_03"),
            ]

        # Lowest duplicated-pair containment seen across four runs.
        self.assertEqual(merge_duplicate_speakers(pair(0.594), 0.5)[1],
                         {"SPEAKER_03": "SPEAKER_02"})
        # Highest genuinely-distinct containment seen across the same four runs.
        self.assertEqual(merge_duplicate_speakers(pair(0.259), 0.5)[1], {})

    def test_a_long_ragged_partial_overlap_is_not_a_duplicate(self):
        """The shape real simultaneous speech takes: one speaker holds long solo
        turns and the concurrency is a small fraction of them."""
        distinct = [
            SpeakerTurn(start=0.0, end=64.0, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=57.5, end=64.0, speaker_id="SPEAKER_01"),
            SpeakerTurn(start=64.0, end=147.0, speaker_id="SPEAKER_01"),
        ]  # shared 6.5 / min(64.0, 89.5) = 0.10
        self.assertEqual(merge_duplicate_speakers(distinct, 0.5)[1], {})

    def test_a_short_pair_needs_min_shared_seconds(self):
        """Two brief turns can hit the ratio on noise alone."""
        turns = [
            SpeakerTurn(start=0.0, end=0.5, speaker_id="SPEAKER_02"),
            SpeakerTurn(start=0.0, end=0.5, speaker_id="SPEAKER_03"),
        ]
        self.assertEqual(merge_duplicate_speakers(turns, 0.5)[1], {})
        self.assertEqual(
            merge_duplicate_speakers(turns, 0.5, min_shared_seconds=0.25)[1],
            {"SPEAKER_03": "SPEAKER_02"},
        )

    def test_the_surviving_id_is_the_one_with_more_speech(self):
        turns = [
            SpeakerTurn(start=0.0, end=2.0, speaker_id="SPEAKER_01"),
            SpeakerTurn(start=0.0, end=9.0, speaker_id="SPEAKER_03"),
        ]
        _out, aliases = merge_duplicate_speakers(turns, 0.5)
        self.assertEqual(aliases, {"SPEAKER_01": "SPEAKER_03"})

    def test_an_exact_tie_on_speech_keeps_the_lower_id(self):
        turns = [
            SpeakerTurn(start=0.0, end=5.0, speaker_id="SPEAKER_01"),
            SpeakerTurn(start=0.0, end=5.0, speaker_id="SPEAKER_03"),
        ]
        self.assertEqual(merge_duplicate_speakers(turns, 0.5)[1],
                         {"SPEAKER_03": "SPEAKER_01"})

    def test_three_slots_of_one_person_collapse_transitively(self):
        turns = [
            SpeakerTurn(start=0.0, end=9.0, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=0.1, end=8.9, speaker_id="SPEAKER_01"),
            SpeakerTurn(start=0.2, end=8.8, speaker_id="SPEAKER_02"),
        ]
        out, aliases = merge_duplicate_speakers(turns, 0.5)
        self.assertEqual(aliases,
                         {"SPEAKER_01": "SPEAKER_00", "SPEAKER_02": "SPEAKER_00"})
        self.assertEqual({t.speaker_id for t in out}, {"SPEAKER_00"})

    def test_relabelled_turns_are_re_merged_so_no_speaker_overlaps_itself(self):
        """`derive_overlaps` documents the invariant that one speaker's merged
        turns never overlap — two slots of one person violate it the moment they
        share a label, and an active count of 2 would then be read as two
        distinct speakers talking at once."""
        turns = [
            SpeakerTurn(start=0.0, end=4.0, speaker_id="SPEAKER_02"),
            SpeakerTurn(start=0.1, end=3.9, speaker_id="SPEAKER_03"),
        ]
        out, _aliases = merge_duplicate_speakers(turns, 0.5)
        self.assertEqual(len(out), 1)
        self.assertEqual(derive_overlaps(out), [])

    def test_phantom_overlaps_disappear_with_the_duplicate(self):
        turns = [
            SpeakerTurn(start=0.0, end=4.0, speaker_id="SPEAKER_02"),
            SpeakerTurn(start=0.1, end=3.9, speaker_id="SPEAKER_03"),
            SpeakerTurn(start=10.0, end=12.0, speaker_id="SPEAKER_00"),
        ]
        self.assertEqual(len(derive_overlaps(turns)), 1)
        out, _ = merge_duplicate_speakers(turns, 0.5)
        self.assertEqual(derive_overlaps(out), [])

    def test_containment_above_one_disables_the_pass(self):
        turns = [
            SpeakerTurn(start=0.0, end=4.0, speaker_id="SPEAKER_02"),
            SpeakerTurn(start=0.1, end=3.9, speaker_id="SPEAKER_03"),
        ]
        out, aliases = merge_duplicate_speakers(turns, 0.5, min_containment=1.01)
        self.assertEqual(aliases, {})
        self.assertEqual(len(out), 2)

    def test_a_single_speaker_and_an_empty_list_are_no_ops(self):
        one = [SpeakerTurn(start=0.0, end=4.0, speaker_id="SPEAKER_00")]
        self.assertEqual(merge_duplicate_speakers(one, 0.5), (one, {}))
        self.assertEqual(merge_duplicate_speakers([], 0.5), ([], {}))

    def test_speakers_that_never_coincide_never_merge(self):
        turns = [
            SpeakerTurn(start=0.0, end=5.0, speaker_id="SPEAKER_00"),
            SpeakerTurn(start=5.0, end=10.0, speaker_id="SPEAKER_01"),
        ]
        self.assertEqual(merge_duplicate_speakers(turns, 0.5)[1], {})
