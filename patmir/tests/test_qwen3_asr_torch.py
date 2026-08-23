"""Unit tests for the transformers-native Qwen3-ASR candidate.

Covers the pure logic only — the chunker, the transformers version gate, and
language normalization. The models themselves are not exercised here (they need
a GPU and ~5 GB of weights); the first-run checks live in
`docs/setup/windows-wsl2-cuda.md`.
"""
from __future__ import annotations

import sys
import types

import pytest

from patmir.pipeline.stages.asr._sentence import word_in_target_script
from patmir.pipeline.stages.asr.qwen3_asr import (
    MIN_TRANSFORMERS,
    _check_transformers,
    _to_iso,
    build_chunks,
)


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
