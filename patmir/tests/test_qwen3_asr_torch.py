"""Unit tests for the transformers-native Qwen3-ASR candidate.

Covers the pure logic only — the chunker, the transformers version gate, and
language normalization. The models themselves are not exercised here (they need
a GPU and ~5 GB of weights); the first-run checks live in
`docs/setup/windows-wsl2-cuda.md`.
"""
from __future__ import annotations

import inspect
import sys
import types
from dataclasses import replace

import pytest

from patmir.pipeline.stage import (
    Diarization,
    Sentence,
    SpeakerTurn,
    Transcript,
    Word,
)
from patmir.pipeline.stages.asr import qwen3_asr
from patmir.pipeline.stages.asr._sentence import (
    group_into_sentences,
    word_in_target_script,
)
from patmir.pipeline.stages.asr.qwen3_asr import (
    MIN_TRANSFORMERS,
    _check_transformers,
    _to_iso,
    build_chunks,
)
from patmir.pipeline.stages.asr.speaker_tag import apply_speaker_tags


# --- chunker ------------------------------------------------------------


def test_no_vad_falls_back_to_fixed_windows():
    chunks = build_chunks(70.0, None, 30.0)
    assert chunks == [(0.0, 30.0), (30.0, 60.0), (60.0, 70.0)]


def test_no_chunk_exceeds_the_cap():
    vad = [(0.0, 5.0), (6.0, 11.0), (12.0, 17.0), (18.0, 23.0), (24.0, 29.0)]
    for cap in (10.0, 15.0, 28.0):
        chunks = build_chunks(30.0, vad, cap)
        assert chunks, cap
        for start, end in chunks:
            assert end - start <= cap + 1e-6, (cap, start, end)


def test_chunks_are_ordered_and_within_the_source():
    vad = [(1.0, 4.0), (10.0, 12.5), (20.0, 28.0)]
    chunks = build_chunks(30.0, vad, 12.0)
    assert chunks == sorted(chunks)
    assert chunks[0][0] >= 0.0
    assert chunks[-1][1] <= 30.0
    for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:]):
        assert prev_end <= next_start + 1e-9, "chunks must not overlap"


def test_speech_is_never_cut_when_it_fits():
    """A gap wider than the cap must not merge, and every VAD interval that
    fits inside the cap must land wholly inside one chunk."""
    vad = [(1.0, 3.0), (50.0, 53.0)]
    chunks = build_chunks(60.0, vad, 28.0)
    assert len(chunks) == 2
    for (v_start, v_end), (c_start, c_end) in zip(vad, chunks):
        assert c_start <= v_start and c_end >= v_end


def test_interval_longer_than_the_cap_is_hard_split():
    chunks = build_chunks(100.0, [(0.0, 70.0)], 28.0)
    assert len(chunks) == 3
    for start, end in chunks:
        assert end - start <= 28.0 + 1e-6


def test_padding_never_crosses_into_a_neighbour():
    # Neighbours 0.1s apart: half the gap each, so the two cannot overlap.
    vad = [(5.0, 6.0), (6.1, 7.0), (40.0, 41.0)]
    chunks = build_chunks(50.0, vad, 0.95)
    for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:]):
        assert prev_end <= next_start + 1e-9


def test_degenerate_and_out_of_range_intervals_are_dropped():
    chunks = build_chunks(10.0, [(5.0, 5.0), (20.0, 25.0), (2.0, 4.0)], 28.0)
    assert len(chunks) == 1
    assert chunks[0][0] <= 2.0 and chunks[0][1] >= 4.0


def test_empty_vad_list_is_treated_as_absent():
    assert build_chunks(40.0, [], 30.0) == build_chunks(40.0, None, 30.0)


def test_zero_cap_is_rejected():
    with pytest.raises(ValueError):
        build_chunks(10.0, None, 0.0)


# --- transformers gate --------------------------------------------------


def _with_transformers_version(monkeypatch, version: str) -> None:
    module = types.ModuleType("transformers")
    module.__version__ = version  # noqa: SLF001
    monkeypatch.setitem(sys.modules, "transformers", module)


@pytest.mark.parametrize("version", ["4.57.6", "5.5.0", "5.12.1", "5.9.0"])
def test_too_old_transformers_raises_with_the_fix(monkeypatch, version):
    _with_transformers_version(monkeypatch, version)
    with pytest.raises(RuntimeError) as excinfo:
        _check_transformers()
    message = str(excinfo.value)
    # The message must carry the fix, and must warn off the package whose pin
    # caused the conflict in the first place.
    assert "transformers>=5.13" in message
    assert "qwen-asr" in message
    assert version in message


@pytest.mark.parametrize("version", ["5.13.0", "5.13.1", "5.15.1", "6.0.0.dev0"])
def test_new_enough_transformers_passes(monkeypatch, version):
    _with_transformers_version(monkeypatch, version)
    _check_transformers()  # must not raise


def test_gate_matches_the_documented_floor():
    assert MIN_TRANSFORMERS == (5, 13)


def test_unparseable_version_is_treated_as_too_old(monkeypatch):
    _with_transformers_version(monkeypatch, "unknown")
    with pytest.raises(RuntimeError):
        _check_transformers()


# --- language normalization --------------------------------------------


@pytest.mark.parametrize(("raw", "expected"), [
    ("English", "en"),
    ("russian", "ru"),
    ("Armenian", "hy"),
    ("ru", "ru"),
    ("Chinese Mandarin", "chinese"),
    (None, None),
    ("", None),
])
def test_to_iso(raw, expected):
    assert _to_iso(raw) == expected


# --- the shared CJK-hallucination guard --------------------------------


@pytest.mark.parametrize(("text", "lang", "kept"), [
    ("привет", "ru", True),
    ("嗯", "ru", False),
    ("hello", "en", True),
    ("滋 巴", "en", False),
    ("բարև", "hy", True),
    ("", "ru", False),
])
def test_word_in_target_script(text, lang, kept):
    assert word_in_target_script(text, lang) is kept


# --- sentence segmentation ----------------------------------------------
#
# Until 2026-08-24 this candidate emitted one `Sentence` per ASR chunk, so a
# "sentence" was up to `chunk_seconds` of audio. Downstream that meant
# `apply_speaker_tags` handed a multi-turn span a single dominant speaker and
# the whole span was synthesized in one character's voice. These tests pin the
# grouping contract the fix relies on; the model itself needs a GPU and is not
# exercised here.

def _w(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end)


def test_qwen3_candidate_uses_the_shared_grouper():
    """Regression guard: the defect was a missing call, so assert the call
    exists rather than only asserting on its behaviour."""
    src = inspect.getsource(qwen3_asr)
    assert "group_into_sentences(" in src


def test_words_with_no_punctuation_still_split_on_gaps():
    """The load-bearing case for this candidate: Qwen3-ASR emits NO terminal
    punctuation (0 of 398 words on the reference clip), so the punctuation
    branch never fires and gap-splitting is the only thing separating
    utterances."""
    words = [
        _w("Привет", 0.0, 0.4),
        _w("Хочешь", 0.4, 0.9),
        # 1.2 s of silence — a speaker change in the source
        _w("Постойте", 2.1, 2.7),
        _w("кто", 2.75, 2.95),
        _w("вы", 2.95, 3.1),
    ]
    sents = group_into_sentences(words, max_gap_s=0.35)
    assert len(sents) == 2
    assert sents[0].text == "Привет Хочешь"
    assert sents[1].text == "Постойте кто вы"
    assert not any(c in s.text for s in sents for c in ".!?")


def test_a_chunk_length_run_becomes_many_sentences():
    """A 26 s chunk of conversational speech must not come out as one
    sentence. Reference failure: 23 sentences for 301 s of video against
    diarization's 108 turns."""
    words = []
    t = 0.0
    for i in range(36):
        words.append(_w(f"w{i}", t, t + 0.4))
        t += 0.4 + (1.2 if i % 3 == 2 else 0.05)  # a pause every third word
    span = words[-1].end - words[0].start
    assert span > 24.0, span  # a chunk-sized run, the shape that used to be 1 sentence
    sents = group_into_sentences(words, max_gap_s=0.35)
    assert len(sents) == 12
    assert max(x.end - x.start for x in sents) < 2.0


def test_sentence_bounds_come_from_the_words_not_the_chunk():
    """Chunks carry VAD padding; the mux places TTS at sentence times, so the
    sentence must be narrowed to where the speaker actually talks."""
    words = [_w("да", 12.80, 13.20), _w("конечно", 13.25, 13.90)]
    sent = group_into_sentences(words, max_gap_s=0.35)[0]
    assert sent.start == 12.80
    assert sent.end == 13.90


def test_grouped_sentences_are_renumbered_contiguously_across_chunks():
    """The candidate concatenates per-chunk grouping results, so `idx` has to
    be reassigned globally — `group_into_sentences` restarts at 0 per call."""
    chunk_a = [_w("a", 0.0, 0.2), _w("b", 1.0, 1.2)]
    chunk_b = [_w("c", 5.0, 5.2), _w("d", 6.0, 6.2)]
    sentences: list[Sentence] = []
    for chunk in (chunk_a, chunk_b):
        for sent in group_into_sentences(chunk, max_gap_s=0.35):
            sentences.append(replace(sent, idx=len(sentences)))
    assert [s.idx for s in sentences] == [0, 1, 2, 3]
    assert [s.text for s in sentences] == ["a", "b", "c", "d"]


def test_a_sentence_spanning_two_speaker_turns_is_split():
    """With short sentences, `apply_speaker_tags` should resolve a gapless
    two-speaker exchange rather than assigning one dominant voice."""
    words = [
        _w("кто", 0.0, 0.3), _w("вы", 0.3, 0.6), _w("такой", 0.6, 0.9),
        _w("тот", 1.0, 1.3), _w("кто", 1.3, 1.6), _w("предпочитает", 1.6, 2.0),
    ]
    sent = Sentence(idx=0, text=" ".join(w.text for w in words),
                    start=0.0, end=2.0, words=words)
    diar = Diarization(turns=[
        SpeakerTurn(start=0.0, end=0.95, speaker_id="SPEAKER_00"),
        SpeakerTurn(start=0.95, end=2.0, speaker_id="SPEAKER_01"),
    ], overlaps=[], refs={}, sample_rate=16_000)
    out = apply_speaker_tags(
        Transcript(language="ru", words=words, sentences=[sent]), diar)
    assert len(out.sentences) == 2
    assert out.sentences[0].speaker_id == "SPEAKER_00"
    assert out.sentences[1].speaker_id == "SPEAKER_01"
    assert out.sentences[0].text == "кто вы такой"
    assert out.sentences[1].text == "тот кто предпочитает"


def test_single_speaker_sentence_is_not_split():
    words = [_w("все", 0.0, 0.3), _w("хорошо", 0.3, 0.8)]
    sent = Sentence(idx=0, text="все хорошо", start=0.0, end=0.8, words=words)
    diar = Diarization(turns=[
        SpeakerTurn(start=0.0, end=1.0, speaker_id="SPEAKER_02"),
    ], overlaps=[], refs={}, sample_rate=16_000)
    out = apply_speaker_tags(
        Transcript(language="ru", words=words, sentences=[sent]), diar)
    assert len(out.sentences) == 1
    assert out.sentences[0].speaker_id == "SPEAKER_02"


def test_a_gapless_run_is_still_capped_by_duration():
    """`max_gap_s` alone has a floor: some dialogue contains no pause above the
    threshold. Measured case — 56 words over 22.9 s whose widest interior gap
    was 0.32 s, spanning 11 diarization turns, assigned to ONE speaker at 86%
    dominance (just over the 0.85 threshold) and handed to the TTS as a single
    22.9 s decode chunk."""
    words = []
    t = 0.0
    for i in range(56):
        words.append(_w(f"w{i}", t, t + 0.3))
        t += 0.3 + (0.32 if i % 7 == 6 else 0.08)
    span = words[-1].end - words[0].start
    assert span > 20.0, span

    uncapped = group_into_sentences(words, max_gap_s=0.35)
    assert len(uncapped) == 1  # the defect this cap exists for

    capped = group_into_sentences(words, max_gap_s=0.35, max_sentence_s=8.0)
    assert len(capped) > 1
    assert max(x.end - x.start for x in capped) <= 8.0


def test_the_duration_cap_splits_at_the_widest_gap():
    """The cut should land on the most plausible utterance boundary available,
    not at the midpoint."""
    words = [
        _w("a", 0.0, 1.0), _w("b", 1.1, 2.0),
        # widest interior gap: 2.0 -> 3.5
        _w("c", 3.5, 4.5), _w("d", 4.6, 5.5),
    ]
    out = group_into_sentences(words, max_gap_s=99.0, max_sentence_s=3.0)
    assert [s.text for s in out] == ["a b", "c d"]


def test_the_cap_is_off_by_default():
    """Backends that already produce short sentences (whisper, faster-whisper,
    scribe, openrouter) must keep their exact previous behaviour."""
    words = [_w("a", 0.0, 1.0), _w("b", 1.05, 30.0)]
    assert len(group_into_sentences(words, max_gap_s=0.35)) == 1


def test_the_cap_cannot_split_a_single_word():
    """A lone word longer than the cap has no interior gap; it must survive
    rather than recurse or vanish."""
    out = group_into_sentences([_w("looong", 0.0, 20.0)], max_gap_s=0.35, max_sentence_s=8.0)
    assert len(out) == 1
    assert out[0].text == "looong"


def test_capped_sentences_keep_their_words_and_bounds():
    words = [_w("a", 0.0, 1.0), _w("b", 3.0, 4.0), _w("c", 4.1, 5.0)]
    out = group_into_sentences(words, max_gap_s=99.0, max_sentence_s=2.0)
    assert sum(len(s.words) for s in out) == 3
    assert [s.idx for s in out] == list(range(len(out)))
    for s in out:
        assert s.start == s.words[0].start
        assert s.end == s.words[-1].end
