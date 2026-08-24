# One actor, one voice: duplicate speaker slots and per-word attribution

- **Date:** 2026-08-24
- **Status:** accepted
- **Supersedes:** the dominance-plus-one-boundary half of
  `2026-08-24-asr-sentence-segmentation-and-cuda-baseline.md`
- **Stages:** diarize, asr (speaker tagging), tts (consequence)
- **Quality issues:** #16 (new), #17 (new), extends #14

## The report

The author ear-tested `runs/20260824-163841-sample_1/out.mkv` and listed nine
timestamps where "the same person's voice varies during the dialog, or the person
is being mis-identified". Every one of them traced to one of four mechanisms, and
three of those four turned out to share a single root cause in diarization.

## What the artifacts said

**Sortformer emitted one actor on two of its output slots.** Pairwise
concurrency across the clip:

```
% of each speaker's audio concurrent with each other speaker
  SPEAKER_00: 01=10%   02=0%    03=1%
  SPEAKER_01: 00=8%    02=0%    03=1%
  SPEAKER_02: 00=0%    01=2%    03=52%
  SPEAKER_03: 00=3%    01=9%    02=71%
```

Seven SPEAKER_02 / SPEAKER_03 turn pairs coincided at 90–100 % of the shorter
turn, five of them at total containment (e.g. S02 `98.96–99.68` sits entirely
inside S03 `98.72–100.24`). Genuine simultaneous speech does not look like that —
the one pair of actually-distinct speakers on the clip sat at 8–10 %, ragged and
partial. The author independently corroborated it by ear, hearing *one* woman at
1:00 and at 1:22–1:30 where the diarizer reported two people talking at once.

That single artifact caused three separate downstream failures:

1. **The actor's lines split across two ref cells.** SPEAKER_02 was pinned to a
   2.92 s clip and SPEAKER_03 to a *different character's*, so the same woman's
   voice changed at 85.6 s and again across 1:36–1:44.
2. **SPEAKER_03 got no reference clip at all.** `pick_refs` rejects any turn
   intersecting an overlap interval, and `derive_overlaps` was reporting the twin
   pair as overlapping speech — so 83 % of SPEAKER_03's audio was disqualified by
   its own duplicate. Zero eligible windows, so the TTS stand-in path handed it
   SPEAKER_00's pinned clip. Two characters, byte-identical reference audio.
3. **It defeated the sentence splitter**, below.

**`apply_speaker_tags` left another speaker's words in 13 of 76 sentences** — 25
words of 393 dubbed in the wrong voice. The pass asked "does one speaker cover
≥ 0.85 of this sentence?" and, failing that, split ONCE at the first turn whose
speaker differed from the sentence's opening speaker. On concurrent turns the
boundary therefore landed a few hundred milliseconds into the sentence:

```
idx 12  60.26–62.42  dominant SPEAKER_02 = 0.556
  turns: S03 60.24–61.44 | S02 60.32–61.52 | S01 61.68–62.56
  opening=S03 → boundary=60.32          ← 0.06 s after the sentence starts
  left (1 word): 'А'  non_lexical=True  → split rejected, falls back to dominant
```

Same shape on idx 20 (boundary 97.20 against a start of 97.08, left half `'И'`).
Both of those the author had flagged: `"На ней кровь"` (a man) came out in a
woman's voice, and `"Зачем"` (a third character) in a second character's. The
0.85 gate let small intruders through independently — idx 20 measured 0.811.

**Two more mechanisms, not addressed here.** Only four speaker ids exist for a
clip with at least six speaking characters: the squad roll-call at 192–291 s runs
entirely on SPEAKER_00 and SPEAKER_01 while the other two slots sit idle from
120 s, and one 7.16 s "turn" carries a four-way question-and-answer exchange
(its own `ref_text` reads *"Откуда вы / Из дела закупок вроде / У нас есть имена
/ Санчес и Ороско Кто из вас Ороско"*). And the 0:17.5 change the author asked
about is neither an id change nor an emotion change — sentences 2 and 3 are the
same speaker, the same emotion bin, and the same 2.92 s reference file, so it is
Higgs re-cloning per utterance off the shortest ref in the job.

## Decision

**Merge duplicate speaker slots in diarize post-processing.** New
`merge_duplicate_speakers` in `_postprocess.py` (shared by both Sortformer
runtimes, so they cannot drift): two ids whose aggregate concurrency reaches
`shared / min(total_a, total_b) >= 0.6` on at least 1.0 s of shared audio are one
person. The surviving id is the group member with the most speech, ties by lowest
id; relabelled turns are re-merged, because two slots of one speaker become
self-overlapping turns and every consumer — `derive_overlaps` explicitly — assumes
that cannot happen. Union-find, so three slots collapse transitively.

Aggregate containment is the test because real simultaneous speech cannot come
near the threshold there: a person would have to spend most of their speaking
time talking over exactly one other person and never hold a solo turn. Exposed as
`duplicate_containment` / `duplicate_min_seconds` on both candidates; `> 1.0`
disables the pass.

**The threshold is 0.4, and getting there required learning that the model is not
deterministic.** A first cut at 0.6 came from the single run that started this
investigation, where the duplicated pair measured 0.706 against 0.101 for the
distinct pair — an apparently huge margin. It then silently stopped firing: four
runs of the identical file put the duplicated pair at 0.705, 0.594 and twice
above 0.6, and the 0.594 run came back reporting four speakers with
`SPEAKER_03: no clean reference clip — using global fallback voice`, the exact
symptom the pass exists to prevent. Across those same four runs the highest a
genuinely-distinct pair reached was 0.259. So the real gap is 0.26–0.59, 0.6 sat
on the edge of one side of it, and 0.4 sits in the middle. Both observed bounds
are pinned by test. At 0.4 all four runs resolve to three speakers.

**Attribute speakers per word, not per sentence.** `apply_speaker_tags` now
labels each word by the speaker whose turns it overlaps most, groups words into
runs, and emits one sentence per run. Dominance and the single-boundary search
are gone.

Labelling runs in two passes. The first labels only words the diarization is
actually *evidence* about; a word is deferred to the second pass when:

- **the best turn covers less than `MIN_WORD_COVERAGE` (0.5) of the word.**
  Measured: `"Я"` at 91.64–91.80 caught 0.04 s of a turn that starts *after* the
  word does — 25 % coverage — and that scrap alone gave it a different speaker
  from the rest of its sentence.
- **its midpoint is inside an overlap interval *and* more than one speaker clears
  that floor.** There the diarizer is saying two people are talking and both of
  them cover the word, so taking the larger number is a coin flip — and a coin
  flip must not manufacture a turn boundary inside somebody's sentence. This is
  the duplicate-slot artifact seen *locally* rather than clip-wide: aggregate
  containment cannot catch two otherwise-distinct speakers whose turns run
  concurrently for one phrase. Measured: `"Он еще | долго не очнется"` (0.80
  against 0.60 coverage) and `"Сигнал бедствия"` (0.92 against 0.75) were each
  torn off their own sentence this way. Requiring *two* qualifying speakers keeps
  the rule off words that merely sit near an overlap; overlap covers under 4 % of
  the clip, and the rule is inert for candidates that emit no overlaps.
- no turn covers it at all (a hole in the diarization is not a speaker change).

The second pass has two properties, and both were arrived at by watching a
version fail:

- **A maximal run of consecutive deferred words is resolved as ONE decision.**
  Resolving word by word lets a single phrase alternate: all five words of
  `"Он еще долго не очнется"` defer — each covered by two speakers inside one
  overlap interval — and picking the larger per-word coverage independently gave
  `SPEAKER_01 | SPEAKER_00 ×3 | SPEAKER_01`, one utterance in three voices, which
  is worse than the mis-attribution it replaced. Summed across the block the same
  numbers give SPEAKER_01 by 1.52 s to 1.36 s.
- **It is constrained to speakers that plausibly cover the block.** Plain
  forward-fill was tried first and is wrong: it handed `"Он еще"` to the speaker
  of the preceding word, who covers *neither* of those words, purely because that
  word was adjacent — tearing the phrase off its own sentence and leaving it in
  the mother's voice. A deferred block takes a neighbour's label only if that
  speaker actually covers it; failing that it falls back to its own best
  candidate summed across the block; only a block with no candidate at all
  inherits unconditionally, which is what carries a pure-sliver run.

Three smoothing rules then fold away runs too short to be a plausible utterance:

- **Wobble** — a run of ≤ 2 words with the same speaker on *both* sides joins
  that speaker without consulting overlap. The A-B-A shape is the evidence;
  overlap cannot decide it, because in the measured case (`"он еще долго не
  очнется"`, which arrived as `S01 | S00×2 | S01`) the interior overlap is an
  exact 0.32 s tie. The word cap is what keeps a real three-way exchange inside
  one ASR sentence from losing its middle speaker.
- **Edge** — a one-word run at a sentence edge joins its single neighbour only
  when the neighbour fits it *at least as well as its own label does*. Comparing
  against the run's own fit is what makes this safe: scoring only the neighbour
  absorbed any short run with non-zero neighbour overlap, which on a 1|2|1
  pattern handed the whole sentence to the two-word middle run even though both
  singles fitted the outer speaker better. A run the diarizer places squarely on
  a speaker of its own survives as its own line — that is `"Зачем"`, 0.36 s
  against SPEAKER_00 and 0 against its neighbour.

- **Duration floor** — any run shorter than `MIN_FRAGMENT_SECONDS` (0.35 s) is
  absorbed whatever the attribution says, because it is too short to synthesize
  as a standalone line. This is the only rule that catches diarization splitting
  one phrase across two *adjacent* turns: there is no overlap and coverage is
  100 %, so neither guard above sees it. The measured margin is narrow and both
  ends are real, so both are pinned by test — `"кровь"` (0.24 s) is noise and
  must be absorbed, `"Зачем"` (0.40 s) is a genuine one-word line by a third
  character and must survive. The floor also subsumes the zero-length words real
  aligner output contains (13 on this clip), which would otherwise emit a
  zero-duration sentence for the mux to place.

Non-lexical runs are absorbed regardless of fit, so an orphaned interjection
never becomes its own sentence for translate and TTS.

## Consequences

- Per-word attribution handles three or more speakers in one sentence, which the
  split-once version structurally could not, and needs no data that was not
  already in the artifacts.
- Diarization on this clip goes 4 ids → 3 on every run measured, and
  SPEAKER_02's longest eligible reference window 2.92 s → 4.52 s, which also
  softens the per-utterance timbre drift the author heard at 0:17.5. No character
  clones from another's clip.
- **Sortformer is not deterministic on identical input.** Four runs of the same
  file gave 98/106/110/111 turns, 25–40 overlap intervals, and per-speaker totals
  moving by several seconds. Any threshold in this area has to be calibrated
  against a spread, not a run — and any single-run measurement quoted anywhere in
  these docs should be read with that in mind.
- Sentence count rises — 67 ASR sentences became 83 on the confirming run, where
  the old pass produced 76 — because genuine turn changes now split. Every added
  sentence is another translate call and another TTS line; translate was already
  the #2 cost of a warm batch run at 13.6 %.
- **Every measured failure was re-checked against four independent diarizations
  of the same file**, which disagree on turn count (98/102/110/111) and on where
  boundaries fall. All of them resolve the same way. That is the evidence the
  rules are not fitted to one run.
- One-word sentences rise from 10 to 19. They are real lines (`"Зачем"`,
  `"Третий"`, `"Простите"`) and the pipeline already synthesized 10 of them, but
  a short line is a harder TTS target and `voice_no_overlap` has less slack to
  place it.
- Four constants (`MIN_WORD_COVERAGE`, `MIN_FRAGMENT_SECONDS`,
  `MAX_WOBBLE_WORDS`, `MIN_RUN_WORDS`) are calibrated on one clip, and
  `MIN_FRAGMENT_SECONDS` sits in a 0.24–0.40 s gap between two real measurements.
  They are constants in `speaker_tag.py`, not config, on the grounds that a
  second clip should move them rather than a deployment. Both ends of the
  duration margin are pinned by test so a change has to confront them.
- The rules were arrived at by measuring, not by design. Four versions each
  fixed one class of failure and introduced another: scoring absorption against
  the neighbour alone swallowed correct edge runs; scoring against the run's own
  fit alone shredded single utterances; an unconditional overlap guard let a
  phrase inherit from a speaker with zero coverage of it; and per-word resolution
  of a deferred block put one utterance in three voices. Each of those is pinned
  by a test in `test_speaker_tag.py` naming the shape it broke, so a future
  simplification has to confront them rather than rediscover them.
- The 4-speaker cap is untouched and is now the largest remaining identity
  problem. `pyannote_3_1` is registered on the cuda profile and uncapped; it was
  dropped on 2026-06-15 for merging short *cartoon* turns, which does not
  transfer to live-action drama without measuring.
- The zero-duration `'сэр'` sentence at 134.86 predates this change (it is in the
  shipped output too) and survives it. Its cause is `group_into_sentences`
  splitting on a zero-length aligner word, upstream of both fixes.

## Revisit

- A clip where two people genuinely talk over each other enough to cross 0.4
  aggregate containment — the threshold, not the mechanism, is what would move.
  The observed headroom is 0.259 on one clip, so this is the constant most likely
  to need a second clip's evidence.
- An ear-test finding one-word fragments worse than the mis-voiced words they
  replaced; the lever is `MIN_RUN_WORDS`.
- A wobble longer than two words, or a real interjection being swallowed by the
  wobble rule.
- Diarization gaining more than four identities, which would make most of the
  remaining voice confusion on this clip go away and change what these
  heuristics are compensating for.

## Addendum 2026-08-24: watch-along caught the merge over-firing

Running watch-along after this landed exposed a scale bug in
`merge_duplicate_speakers`. The 0.4 containment threshold was calibrated on
5-minute totals; watch-along diarizes ~35 s sections and a ~40 s prelude, where
containments of **0.73-1.00 between genuinely distinct speakers are routine**
because each speaker only holds a few seconds. Unguarded it collapsed 4 speakers
to 2 in **every** section and the prelude to **1** — the mother and a male
character on one identity, three lines with no speaker at all. The prelude
establishes the cast for the entire stream, so that is the worst possible place.

Fixed with a third guard, `min_speaker_seconds` (default 10.0): a pair is only
considered when the *smaller* speaker holds that much speech, because below it a
single overlapping utterance can reach 0.4 by chance. On the observed data the
two regimes separate with margin — real merges start at 11.68 s, misfires top
out at 9.36 s — and the pass is thereby confined to whole-clip diarization, where
it was validated, without the candidate needing to know which execution mode it
is running under. The failure direction is deliberate: a duplicated bit-part with
under 10 s of speech is now missed, which costs one character a shared voice,
where a false merge costs two characters their separate voices.

The lesson generalises beyond this knob: **a ratio calibrated on whole-clip
totals cannot be assumed valid per section.** Watch-along runs the same stage
code over ~35 s windows, so any diarize threshold tuned on batch needs checking
there before it ships. That check is now part of what running watch-along is for.

Cost of the whole speaker-identity change on watch-along throughput: steady-state
RTF 0.831 → 0.877 (+5.5 %), from more sentences per section. Still under
realtime. Numbers in `performance.md`.
