"""Post-ASR pass: tag each Sentence with a speaker_id; split sentences
that cross a speaker boundary into per-speaker fragments.

Lives next to ASR because it operates on `Transcript`. Called from the
runner inside the `"asr"` branch right after `stage_obj.transcribe(...)`
returns — keeps every ASR candidate (`whisper_mlx`, `faster_whisper`,
`qwen3_asr`) untouched.

Algorithm per sentence (rewritten 2026-08-24 — see below):

  1. Label every WORD with the speaker whose turns it overlaps most — but only
     where the diarization is evidence of one. A word is DEFERRED when the best
     turn covers less than `MIN_WORD_COVERAGE` of it, when it sits inside an
     overlap interval and two speakers both clear that floor (a coin flip must
     not manufacture a turn boundary), or when no turn covers it at all.
     Deferred words then take a neighbour's label, but only a neighbour who
     actually covers them — otherwise their own best candidate. See
     `_label_words`.
  2. Group the words into consecutive same-speaker runs.
  3. Fold away runs too short to be a plausible utterance: a short run with the
     same speaker on BOTH sides is a diarization wobble and joins it; a short
     run at an edge joins its neighbour only when the neighbour fits it at least
     as well as its own label does; and any run below `MIN_FRAGMENT_SECONDS` is
     absorbed whatever the attribution says, being too short to synthesize. A
     short run the diarizer places squarely on a speaker of its own survives as
     its own fragment. See `_smooth_runs`.
  4. Emit one Sentence per surviving run, re-indexed contiguously.

Why not dominance-plus-one-boundary (the version through 2026-08-24): that
approach asked "does one speaker cover >= 85% of this sentence?" and, if not,
split ONCE at the first turn whose speaker differed from the sentence's opening
speaker. Both halves of it failed on real Sortformer output, which emits
CONCURRENT turns (that is how `derive_overlaps` gets its intervals):

  * The boundary came from a turn START, so a turn running concurrently with the
    opening speaker put the boundary a few hundred ms into the sentence. The left
    half was then one word, that word was usually a function word, the
    `is_non_lexical` guard rejected the split, and the whole sentence fell back
    to its dominant speaker. Measured on `runs/20260824-163841-sample_1`:
    sentence 12 (60.26-62.42) had its boundary land at 60.32 with left half
    `'А'`; sentence 20 (97.08-100.68) at 97.20 with left half `'И'`. In both the
    trailing words belonged to a different character and were dubbed in the
    dominant speaker's voice — "На ней кровь" (a man) spoken by a woman, and
    "Зачем" (a third character) spoken by a second.
  * The 0.85 dominance gate let a genuine intruder through whenever it was
    small: sentence 20 sat at 0.811 and sentence 3 at 7-words-to-1.

Across that clip 13 of 76 sentences carried another speaker's words — 25 words
of 393 in the wrong voice. Per-word attribution resolves every one of them from
data already in the artifacts, and it handles three-plus speakers in one
sentence, which the split-once version structurally could not.

Also exposes `backfill_ref_text(diarization, transcript)` — fills in
`SpeakerRef.ref_text` for each speaker by joining the ASR words whose
midpoint falls inside the reference window.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from patmir.pipeline.stage import (
    Diarization,
    OverlapInterval,
    Sentence,
    SpeakerTurn,
    Transcript,
    Word,
)
from patmir.pipeline.stages.asr._sentence import is_non_lexical

# A run at the EDGE of a sentence with fewer than this many words is a candidate
# for absorption into its one neighbour — but only when the neighbour fits it at
# least as well as its own label does (see `_smooth_runs`).
MIN_RUN_WORDS = 2

# A word must sit on one speaker's turns for at least this fraction of its own
# duration before it counts as evidence of who is speaking. Below it the word is
# deferred to `_label_words`' second pass. Measured: "Я" at 91.64-91.80
# caught 0.04 s of a turn that starts after the word does — 25% coverage — and
# that scrap was enough to break "Я получил твое сообщение" into two speakers.
MIN_WORD_COVERAGE = 0.5

# A run shorter than this is absorbed into a neighbour whatever the attribution
# says, because it is too short to synthesize as a standalone line — the TTS gets
# a fragment, the mux gets a slot it cannot fill, and `voice_no_overlap` has
# nothing to work with. Measured margin is narrow and both ends are real: "кровь"
# (0.24 s) is diarization splitting one man's phrase across two adjacent turns
# and must be absorbed, while "Зачем" (0.40 s) is a genuine one-word line by a
# third character and must survive.
MIN_FRAGMENT_SECONDS = 0.35

# A run of at most this many words with the SAME speaker on both sides is a
# diarization wobble inside one person's utterance, not a turn: absorb it into
# the surrounding speaker without consulting overlap. Two words is measured, not
# arbitrary — "он еще долго не очнется" came back as
# SPEAKER_01 | SPEAKER_00 x2 | SPEAKER_01 with the interior overlap an exact tie
# (0.32 s each way), so no overlap comparison can resolve it, while the A-B-A
# shape says plainly that A is talking. The cap is what keeps a genuine
# three-way exchange in one sentence (A speaks, B replies at length, A answers)
# from having B's line swallowed.
MAX_WOBBLE_WORDS = 2


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _dominant_speaker(sent: Sentence, turns: list[SpeakerTurn]) -> str | None:
    """Speaker with the most overlap across the whole sentence.

    Only used for sentences with no word timings (the Qwen3 candidates fall back
    to a chunk-level Sentence when the forced aligner returns nothing) and as the
    label for a run whose words the diarizer never covers.
    """
    per_speaker: dict[str, float] = defaultdict(float)
    for t in turns:
        ov = _overlap(sent.start, sent.end, t.start, t.end)
        if ov > 0:
            per_speaker[t.speaker_id] += ov
    if not per_speaker:
        return None
    return min(per_speaker.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def _word_coverage(word: Word, turns: list[SpeakerTurn]) -> dict[str, float]:
    """Seconds of this word covered by each speaker's turns."""
    per_speaker: dict[str, float] = defaultdict(float)
    for t in turns:
        cov = _overlap(word.start, word.end, t.start, t.end)
        if cov > 0:
            per_speaker[t.speaker_id] += cov
    return dict(per_speaker)


def _best_speaker(coverage: dict[str, float]) -> str:
    """Highest coverage; exact ties resolve to the lowest id so two runs of the
    same clip tag identically."""
    return min(coverage.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def _midpoint_speaker(word: Word, turns: list[SpeakerTurn]) -> str | None:
    """Fallback for a zero-length word — no turn can overlap a zero-width span,
    and real aligner output contains them (13 on the measured clip)."""
    mid = 0.5 * (word.start + word.end)
    for t in sorted(turns, key=lambda t: (t.start, t.end)):
        if t.start <= mid < t.end:
            return t.speaker_id
    return None


def _in_overlap(word: Word, overlaps: list[OverlapInterval]) -> bool:
    mid = 0.5 * (word.start + word.end)
    return any(ov.start <= mid < ov.end for ov in overlaps)


def _word_overlap_with(words: list[Word], speaker: str, turns: list[SpeakerTurn]) -> float:
    """Total overlap between these words and `speaker`'s turns."""
    return sum(
        _overlap(w.start, w.end, t.start, t.end)
        for w in words
        for t in turns
        if t.speaker_id == speaker
    )


def _label_words(words: list[Word], turns: list[SpeakerTurn],
                 overlaps: list[OverlapInterval]) -> list[str] | None:
    """One speaker label per word, or None if no word matched any turn.

    Two passes. The first labels only words the diarization is actual evidence
    about; the second resolves the rest from their neighbours, but *constrained
    to speakers that plausibly cover the word*.

    A word is deferred to the second pass when:

    * **the best turn covers less than `MIN_WORD_COVERAGE` of it.** Measured:
      "Я" at 91.64-91.80 caught 0.04 s of a turn that starts after the word does
      — 25% coverage — and that scrap alone gave the word a different speaker
      from the rest of its sentence.
    * **its midpoint is inside an overlap interval AND more than one speaker
      clears the coverage floor.** There the diarizer is saying two people are
      talking and both of them cover the word, so picking the larger number is a
      coin flip — and a coin flip must not manufacture a turn boundary inside
      somebody's sentence. This is the duplicate-slot artifact (quality issue
      #16) seen locally instead of clip-wide: aggregate containment cannot catch
      two otherwise-distinct speakers whose turns run concurrently for one
      phrase. Measured: "Он еще | долго не очнется" (0.80 vs 0.60 coverage) and
      "Сигнал бедствия" (0.92 vs 0.75) were each broken off their own sentence
      this way. Requiring TWO qualifying speakers keeps the rule off words that
      merely sit near an overlap; overlap covers under 4% of the measured clip.

    The second pass has two properties that took three attempts to get right.

    It resolves a **maximal run of consecutive deferred words as one decision**,
    not word by word. Resolving individually lets a single phrase alternate: all
    five words of "Он еще долго не очнется" deferred (each covered by two
    speakers inside one overlap interval), and picking the larger coverage per
    word independently produced SPEAKER_01 | SPEAKER_00 x3 | SPEAKER_01 — one
    utterance in three voices. Summed over the block the same numbers give
    SPEAKER_01 by 1.52 s to 1.36 s, which is the whole phrase in one voice.

    And it is **constrained to speakers that plausibly cover the block**. Plain
    forward-fill was tried first and is wrong: it handed "Он еще" to the speaker
    of the preceding word, who has *zero* coverage of either word, purely because
    that word was adjacent. A deferred block takes a neighbour's label only if
    that speaker actually covers it; otherwise it falls back to its own best
    candidate summed across the block; only a block with no candidate at all
    inherits unconditionally, which is what carries a pure-sliver run.
    """
    coverage = [_word_coverage(w, turns) for w in words]
    candidates: list[set[str]] = []
    labels: list[str | None] = []
    for word, cover in zip(words, coverage):
        duration = word.end - word.start
        if duration > 0:
            cands = {spk for spk, cov in cover.items()
                     if cov / duration >= MIN_WORD_COVERAGE}
        else:
            cands = set(cover)
        candidates.append(cands)
        if not cover:
            labels.append(_midpoint_speaker(word, turns))
        elif not cands:
            labels.append(None)
        elif len(cands) > 1 and _in_overlap(word, overlaps):
            labels.append(None)
        else:
            labels.append(_best_speaker(cover))

    if not any(lbl is not None for lbl in labels):
        return None

    resolved = list(labels)
    i = 0
    while i < len(labels):
        if labels[i] is not None:
            i += 1
            continue
        # Maximal run of consecutive deferred words — resolved as ONE decision.
        j = i
        while j < len(labels) and labels[j] is None:
            j += 1
        prev_lbl = labels[i - 1] if i > 0 else None
        next_lbl = labels[j] if j < len(labels) else None

        block_cover: dict[str, float] = defaultdict(float)
        block_cands: set[str] = set()
        for k in range(i, j):
            block_cands |= candidates[k]
            for spk, cov in coverage[k].items():
                block_cover[spk] += cov

        plausible = [n for n in (prev_lbl, next_lbl)
                     if n is not None and n in block_cands]
        if plausible:
            # By how much of the block that speaker covers; a tie goes to the
            # preceding neighbour (list order).
            choice = max(plausible, key=lambda n: block_cover.get(n, 0.0))
        elif block_cands:
            choice = _best_speaker({spk: block_cover[spk] for spk in block_cands})
        else:
            choice = prev_lbl if prev_lbl is not None else next_lbl
        for k in range(i, j):
            resolved[k] = choice
        i = j
    return [str(lbl) for lbl in resolved]


def _runs(labels: list[str]) -> list[tuple[str, int, int]]:
    """Consecutive same-label spans as (label, start_index, end_index_exclusive)."""
    out: list[tuple[str, int, int]] = []
    for i, lbl in enumerate(labels):
        if out and out[-1][0] == lbl:
            out[-1] = (lbl, out[-1][1], i + 1)
        else:
            out.append((lbl, i, i + 1))
    return out


def _smooth_runs(labels: list[str], words: list[Word],
                 turns: list[SpeakerTurn]) -> list[str]:
    """Fold away runs too short to be a plausible standalone utterance.

    Two rules, applied to a fixed point in this order:

    1. **Wobble** — a run of at most `MAX_WOBBLE_WORDS` words with the same
       speaker on BOTH sides is absorbed into that speaker. The A-B-A shape is
       the evidence; overlap is not consulted because in the measured cases it
       is an exact tie. Guarded by the word cap so a real three-way exchange
       inside one ASR sentence keeps its middle speaker.

    2. **Edge** — a run of fewer than `MIN_RUN_WORDS` words is absorbed into the
       adjacent run its own words fit at least as well as they fit their current
       label ("fit" = total overlap with that speaker's turns). If it fits its
       OWN label strictly better it stands as its own fragment: that is the
       diarizer being confident, and it is the case the old dominance path
       silently swallowed. Measured both ways on the same clip — "Зачем" at
       100.28 scores 0.36 s against SPEAKER_00 and 0 against its neighbour, so
       it survives as a one-word line by a third character; "знать" at 111.42
       scores 0.32 against its own label and 0.02 against the neighbour, and is
       kept for the same reason rather than being merged into the middle of
       someone else's sentence.

    Always absorbed regardless of fit: a non-lexical run, so an orphaned
    interjection never becomes its own sentence for translate and TTS; and any
    run shorter than `MIN_FRAGMENT_SECONDS`, which is too short to synthesize as
    a standalone line whatever the attribution says. The duration floor also
    catches the zero-length words real aligner output contains (13 on the
    measured clip), which would otherwise emit a zero-duration sentence for the
    mux to place.

    Each absorption drops the run count by at least one, so both loops
    terminate.
    """
    labels = list(labels)

    # 1. Wobble: same speaker on both sides.
    while True:
        runs = _runs(labels)
        for k in range(1, len(runs) - 1):
            spk, lo, hi = runs[k]
            outer = runs[k - 1][0]
            if outer != runs[k + 1][0] or outer == spk:
                continue
            if hi - lo > MAX_WOBBLE_WORDS:
                continue
            for i in range(lo, hi):
                labels[i] = outer
            break
        else:
            break

    # 2. Edge: short run, absorbed only when a neighbour fits at least as well.
    while True:
        runs = _runs(labels)
        if len(runs) <= 1:
            return labels
        for k, (spk, lo, hi) in enumerate(runs):
            if hi - lo >= MIN_RUN_WORDS:
                continue
            run_words = words[lo:hi]
            neighbours = []
            if k > 0:
                neighbours.append(runs[k - 1][0])
            if k + 1 < len(runs):
                neighbours.append(runs[k + 1][0])
            if not neighbours:
                continue
            # Preceding neighbour wins an exact tie (enumeration order).
            scored = sorted(
                ((_word_overlap_with(run_words, other, turns), i, other)
                 for i, other in enumerate(neighbours)),
                key=lambda item: (-item[0], item[1]),
            )
            best_ov, _, target = scored[0]
            own_ov = _word_overlap_with(run_words, spk, turns)
            span = run_words[-1].end - run_words[0].start
            too_short = span < MIN_FRAGMENT_SECONDS
            forced = too_short or is_non_lexical(" ".join(w.text for w in run_words))
            if best_ov < own_ov and not forced:
                continue
            for i in range(lo, hi):
                labels[i] = target
            break
        else:
            return labels


def apply_speaker_tags(transcript: Transcript, diarization: Diarization) -> Transcript:
    """Return a new Transcript whose sentences carry `speaker_id` and whose
    mixed-speaker sentences are split per speaker. Sentences are re-indexed
    contiguously from 0.

    Non-lexical content is dropped here so it never reaches translate/TTS,
    regardless of which ASR backend produced it: a sentence whose whole text is
    non-lexical is skipped, and a non-lexical fragment is absorbed into a
    neighbouring run rather than emitted (see `_absorb_short_runs`).
    """
    turns = diarization.turns
    overlaps = diarization.overlaps or []
    out: list[Sentence] = []
    next_idx = 0

    def emit(sent: Sentence) -> None:
        nonlocal next_idx
        out.append(replace(sent, idx=next_idx))
        next_idx += 1

    for sent in transcript.sentences:
        # Centralized non-lexical drop. whisper_mlx already filters at the
        # segment level, but other ASR backends (faster_whisper, qwen3_*)
        # don't — keep this check here so the rule holds regardless of backend.
        if is_non_lexical(sent.text):
            continue

        # No word timings (Qwen3's aligner-failed fallback): nothing to split on.
        labels = _label_words(sent.words, turns, overlaps) if sent.words else None
        if labels is None:
            emit(replace(sent, speaker_id=_dominant_speaker(sent, turns)))
            continue

        labels = _smooth_runs(labels, sent.words, turns)
        runs = _runs(labels)
        if len(runs) == 1:
            emit(replace(sent, speaker_id=runs[0][0]))
            continue

        for spk, lo, hi in runs:
            words = sent.words[lo:hi]
            emit(Sentence(
                idx=0,  # emit() assigns the contiguous index
                text=" ".join(w.text for w in words),
                start=words[0].start,
                end=words[-1].end,
                words=words,
                speaker_id=spk,
            ))

    return Transcript(language=transcript.language, words=transcript.words, sentences=out)


def backfill_ref_text(diarization: Diarization, transcript: Transcript) -> None:
    """Fill each SpeakerRef.ref_text in place by joining ASR words whose
    midpoint falls inside the reference window. Mutates `diarization.refs`.

    Refs are never removed here — even when the backfilled text is empty or
    entirely non-lexical (laughter, screams, single interjections). Whether a
    ref-without-text is usable depends on the active TTS candidate:

    - OmniVoice / Piper need `ref_text` (zero-shot cloning takes a reference
      transcript). The adapter's `_is_usable` check skips empty-text refs for
      those candidates.
    - ElevenLabs IVC only needs the audio. Deleting the ref here used to make
      it fall through to the global default voice for that entire speaker.

    The deletion path is therefore moved to the adapter, where the
    candidate's `ref_text_optional` attribute decides the gating.

    `diarization.refs` is keyed by `(speaker_id, emotion_bin)` in Phase 2.x.
    """
    for ref in diarization.refs.values():
        words = [
            w for w in transcript.words
            if ref.source_start <= 0.5 * (w.start + w.end) < ref.source_end
        ]
        ref.ref_text = " ".join(w.text for w in words).strip()
