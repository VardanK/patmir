# The first word's "gunshot" was `ref_text`, and the answer is a redraw

**Date:** 2026-09-03
**Status:** accepted
**Relates:** quality issue #30 (addressed), #28 (`2026-09-03-separation-determinism`
— which is what made this findable), `2026-08-24-tts-sampling-reproducibility`
(the retry-with-reseed machinery reused here)

## Context

The author reported, across five or more videos, that the very first word of a
dub begins with noise — later described as *"like a muffled gun shot"* and
*"it feels longer"* than 50 ms.

## What it is

The decoder sometimes opens an utterance with a **loud pitch-free impulse
instead of speech**. In the raw clip it runs **45 ms at 40-76% of the clip's own
peak** before any pitch appears; in the mix, 20 ms frames step **-45.8 dB ->
-14.8 dB** into a frame with **zero energy below 300 Hz**, peaking *above* the
source's own speech peak, while the original speaker does not start for another
20-40 ms.

It is audible on the first word and effectively nowhere else, because the burst
only shows when a line has no lead-in: a job's first line starts within 10 ms in
**74% of jobs** against **26%** for every other line (median onset 0.000 s vs
0.062 s). The same artifact occurs mid-dialogue and is masked.

## Root cause: `ref_text`

Found by replaying one job's 77 requests directly through `synthesize_batch`,
changing one input at a time:

| what was passed | bursts |
| --- | ---: |
| single row, `ref_text=""` | 0 |
| batch of 12, `ref_text=""` | 0 |
| all 77, `ref_text=""` | 0 |
| all 77, **real `ref_text`** | **2 of 77, incl. line 0000 at amp 0.76** |

The last row reproduces the pipeline exactly. So it is the reference clip's
**transcript** in the cloning prompt — a Russian sentence conditioning an
Armenian target — and it is **not**:

- the reference *audio* (all three of SPEAKER_01's clips give clean onsets when
  synthesized directly, including a variant with its abrupt head trimmed);
- the batch composition (single row, 12 and 77 all behave the same for a given
  `ref_text`);
- `max_batch`, the CUDA graph or the gqa decode kernel;
- **a regression from the performance work.** Runs from 2026-09-01 at
  `max_batch=8`, no graph, no gqa — the branch base — carry it on line 0000 in
  3 of 4 runs at amp 0.62-0.84. The one clean run found (2026-08-30, 0 of 81)
  came from that same code, and at ~2.5% of lines a clean 81-line run happens
  ~13% of the time.

## Decision: redraw the row, keep `ref_text`

A bursting row is treated exactly like a non-terminating one — `attempt + 1`
reseeds that row's generator and the row is decoded again, reusing the retry
loop the candidate already had for overflow.

**Why not drop `ref_text`,** which measurably removes the artifact (0 of 77)?
Because it is real cloning conditioning, so removing it is a **quality**
decision rather than a bug fix, and it would need its own ear-test for voice
similarity. A redraw touches no conditioning, so voice similarity cannot
regress, and at ~2.6% of lines the retry costs nothing measurable.

Detection is narrow on purpose: a word-initial voiceless fricative (/s/, /ʃ/,
/h/) is pitch-free too and must survive, so the clip's **first 5 ms** must reach
**40% of its own peak** — which speech attacks do not, the median first-frame
amplitude over 3040 non-first lines being 0.005 of peak.

A row still bursting after its redraws **keeps its audio** and is counted as
`onset_bursts_kept`. Silence would be worse than an imperfect line, and the mux
trim below is the second line of defence.

**Verified:** the same job re-synthesized goes **2/77 -> 0/77**, with
`onset_bursts: 2` logged (both caught and redrawn); line 0000 amp 0.76 -> 0.00
and low/high 0.000 -> 1.982.

## Two wrong attempts, recorded

Both were plausible readings of the evidence available at the time, and each
cost a round trip with the author.

**1. Shortening the fade-in — a 17 dB regression.** Reasoning from the 74%/26%
asymmetry, the mux's 50 ms raised-cosine fade-in looked like it was ramping over
the first word's *attack*, so it was capped to the clip's leading quiet with a
4 ms anti-click floor. That mistook the burst for the word. Measured at the
first word, the old 50 ms fade held the burst at **-32.1 dB**; the 4 ms guard
let it through at **-14.8 dB**. The author reported no improvement, correctly.

Two things should have ruled a fade out before it was tried: a fade scales
amplitude and leaves spectrum, so it can only ever *mask* a broadband burst; and
the burst **outlasts the fade** — 85 ms at p90 across affected clips, 150 ms
max — which the author's *"it feels longer"* said directly.

**2. Trimming the burst in the mux.** Correct in effect (the first mix frame
goes from low/high 0.00 to 0.56) but it is audio surgery downstream of a defect
that can be prevented, and it leaves the artifact in `tts/*.wav`.

**Both are kept.** The shortened fade-in is right on its own merits now that
clips begin on their attack, and the trim guards a row that still bursts after
its redraws.

## Method note

This was findable only because #28 landed first. With eight different
`vocals.wav` across ten runs, a defect on one word looked like another bad
draw — and at 2.6% incidence, whether a given run showed it at all was close to
a coin flip. Two of the three false starts in this investigation came from
measuring a symptom (onset flatness, then onset level) instead of varying an
input, and the fix came within one experiment of doing the latter.

## Revisit

- An ear-test comparing `ref_text` present against absent, if voice similarity
  turns out not to need it — that would remove the artifact's cause rather than
  its instances.
- `onset_bursts_kept` above zero on real content, meaning redraws are not
  clearing it and the detector or the response needs revisiting.
- A Higgs release that changes the cloning prompt format.
- The extension's unramped `src.start()` (issue #30's closing note), which is a
  different click with a different trigger.
