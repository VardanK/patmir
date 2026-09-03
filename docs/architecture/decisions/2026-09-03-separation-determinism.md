# The pipeline was not reproducible, and the cause was one unseeded `random.randint`

**Date:** 2026-09-03
**Status:** accepted
**Supersedes/relates:** quality issue #28 (addressed), #29 (opened and declined),
`2026-09-02-cuda-perf-rebaseline-and-tts-decode` (the ear-test this unblocked),
`2026-08-24-tts-sampling-reproducibility` (the same class of bug one layer down)

## Context

An author ear-test rejected `max_batch = 12`: *"audio contains artifacts,
pauses and the translation quality itself degraded"*, and separately *"the very
first speech (male actor) starts with garbled sound"*.

None of the three was attributable to `max_batch`. A controlled render — the
same job dir resumed from `tts` at batch 8 and batch 12, so text, refs and
emotion tags were held fixed — put the reported garble at **both** widths
(onset spectral flatness 0.0056 at 8, 0.0076 at 12, against 0.0002 in the run
the author had approved). Batch width was not the variable.

The variable was that **nothing in this pipeline was reproducible above
`separate`**, so the comparison spanned four independent lotteries at once:
sentence splits, emotion labels, translated wording, and TTS draws.

## The measurement that found it

Ten runs of `samples/sample_1.mkv` at one config, spanning the whole
performance session (so: before and after every change in it):

| artifact | spread across 10 runs |
| --- | --- |
| `vocals.wav` | **8 distinct hashes** |
| diarization turns | 91, 91, 95, 96, 107, 107, 110, 111, 111, 117 |
| emotion labels | disagree on **8-24 %** of matched sentences, over all 45 pairs |
| translated Armenian | **24 of 66** shared lines differ |
| ASR coverage | **stable to 0.1 s** — 151.4 s of 301.1 s in every run |

The pair structure is what identified the stage: the only two run pairs whose
`vocals.wav` matched were **exactly** the two pairs whose emotion labels agreed
perfectly. Identical stem, identical everything downstream.

Sortformer was the first suspect and was **exonerated by measurement**: 12
`diarize()` calls in one process gave one output hash, and four separate model
builds in one process gave one output hash and one weight hash — with no
seeding, no `cudnn.deterministic`, and no `use_deterministic_algorithms`.
Diarization was never the source; it was a faithful amplifier of a jittering
input. (Recorded because "diarization is non-deterministic" was written down as
fact in `current-models.md` from four runs on 2026-08-24 — four runs that also
had a jittering stem.)

## Decision

`shifts = 0` on every `separate` candidate.

`demucs.apply.apply_model` **defaults to `shifts=1`**, and at
`demucs/apply.py:245` it draws `random.randint(0, 0.5 * samplerate)` from the
unseeded global `random`, shifts the mix by that offset, separates, and shifts
back. Neither candidate passed `shifts`, so both took the default.

The reason this is a clean fix rather than a trade is that **`shifts=1` is not
an average**. The loop body runs once and divides by 1, so it is one prediction
at an arbitrary offset — test-time augmentation needs `shifts >= 2` to average
anything. It also pads `max_shift` (0.5 s) of extra audio through the model. So
the default cost throughput and bought nothing but variance.

Set in three places, deliberately:

- the constructor default in both candidates, so **determinism does not depend
  on remembering a config line** — bench, tests and `streaming_config()` all
  construct candidates directly;
- the `build()` config default, same reason;
- explicitly in all four `patmir.toml` separate entries, so the choice lands in
  `config_hash` and is **visible in the config as a choice** rather than an
  invisible code default that one machine could have and another not.

`shifts >= 2` remains configurable and is not deprecated — it is a real
quality/throughput trade nobody here has measured. The note in the code says
that if anyone takes it, they must seed `random` in the same change or the
reproducibility comes straight back.

## Verified

Two runs through `stop_after='emotion'`:

| | run A | run B |
| --- | --- | --- |
| `vocals.wav` | `21e18075` | `21e18075` |
| `speech_vocals.wav` | `56ca06e8` | `56ca06e8` |
| `diarize/turns.json` | `1e18aa02` (121 turns) | `1e18aa02` (121 turns) |
| all 5 ref WAVs | identical, identical source windows | " |
| `emotion/segments.json` | `57bedfa7` | `57bedfa7` |

Separation alone, twice on the real clip: byte-identical stems, 42 VAD
intervals both, 8.9 s and 8.5 s.

7 tests pin the default at the three places it can be lost. The load-bearing
one asserts `apply_model` is called **with** an explicit `shifts=`, because
omitting the kwarg is how the bug existed in the first place — a constructor
default that never reaches demucs protects nothing.

## What this does and does not buy

**Does:** voice identity, emotion tags and the translator's prompt are now
stable run to run. That is what makes an ear-test *attributable* — and it was
not, which is why the author's verdict could not be assigned to the change it
was meant to judge.

**Does not:** end-to-end byte-identity. Translate is a cloud call at
temperature 0.2, so the Armenian still varies with an identical prompt, and TTS
re-draws whenever the text changes. Closing that would mean temperature 0 on
the translator, which is a quality decision, not a reproducibility one, and is
not taken here.

## The ref-cleanliness gate, declined

The same investigation proposed a spectral-cleanliness gate on ref picking,
because selection is `max(duration)` with only an RMS floor
(`ref_picker.py:155`), and the TTS pin (`tts/adapter.py:133`) applies no
quality criterion at all. **It was not built, because the evidence for it was
my own measurement error**, and that is worth recording rather than quietly
dropping.

The cited figure — one clip at "onset flatness 0.1451, 55x the clean one" — was
measured on the **first 1024 samples only**, and a quiet speech onset scores
high flatness for the same reason silence does. That clip's median flatness is
0.0165: ordinary. Measured properly over **349 ref clips from 109 runs**,
median spectral flatness spans **0.0055 to 0.0181** — one tight continuous
band, noisiest 3.3x the cleanest, no outlier population — and leading/trailing
silence is **0 clips over 0.5 s** (max 0.10 s). There is nothing to
discriminate, so any threshold rejects nothing or rejects an arbitrary slice of
a healthy distribution.

Shipping it anyway would have added a knob with an invented threshold that
looks load-bearing, which this codebase has been burned by twice already
(`[emotion_guidance]`, inert for two months; issues #11 and #15). Filed as
issue #29 with the trigger that would justify it: a ref clip identified **by
ear** as cloning badly, so the metric can be chosen to separate that clip from
the 349 that are fine, rather than chosen first and calibrated afterwards.

## Revisit

- A separate candidate that wants `shifts >= 2` for real averaging — seed
  `random` in the same change.
- Translate at temperature 0, if end-to-end reproducibility is ever wanted more
  than the wording variety.
- Issue #29's trigger: an ear-identified bad ref clip.
- Any future claim that a model stage is non-deterministic — measure it
  in-process across calls **and** across builds first, as here. The claim about
  Sortformer in `current-models.md` was wrong for four months, and the cost was
  looking for the bug in the wrong stage.
