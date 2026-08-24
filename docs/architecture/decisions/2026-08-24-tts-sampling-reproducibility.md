# TTS sampling: one RNG stream per sentence, and why batch invariance is out of reach

- **Date:** 2026-08-24
- **Status:** accepted
- **Stage:** tts (`higgs_v3_torch`)
- **Quality issues:** #19 (new); corrects the evidence claim in #15

## The report

The author heard wrong emphasis on two lines of
`runs/20260824-181915-sample_1/out.mkv` — `…մեքենայի մեջ Ա` and `…արյուն ԿԱ`,
both with the final short word stressed — and asked whether it was a sentence cut
or a TTS glitch, and whether we had picked up a lower quantization.

Neither. The candidate is `higgs_v3_torch_bf16`, native bf16, no quantization
anywhere on this path (q8 exists only in the MLX port). Both lines are complete
sentences, not fragments. The artifact is present in the raw `tts/*.wav`, so the
mux is not doing it, and both bins map to no emotion tag.

## What it actually is

A measurable signature separates the two lines from the rest: every line the
author did *not* flag ends with its final voiced chunk **6–19 dB below** that
line's own median chunk level; both flagged lines end at or **above** it (−0.5 dB,
+1.2 dB) while being very short (0.26–0.31× the median chunk duration). The
declarative fall is missing, so the last token reads as stressed.

**It is a sampling lottery.** Six runs of the same clip already on disk, same
text, same speaker, same reference, same settings:

| line | six runs |
| --- | --- |
| `Վրան արյուն կա` | −11.6, −9.6, −12.5, −20.0, −12.1, **+1.2** |
| `…մեքենայի մեջ ա/է` | −25.8, +0.1, −2.2, −10.8, −24.2, **−0.5** |

The same sentence lands anywhere across a 26 dB range. The author's run drew
badly; three of four runs render the first line correctly. This is exactly the
report's word, "sometimes".

**A wrong turn worth recording.** Both flagged lines lack terminal punctuation,
and 62 % of translated lines do — Qwen3-ASR emits *zero* punctuation (0 of
`.!?,` in the whole Russian transcript), so the translator returns bare
fragments. Adding `։` to those two lines moved the final chunk from +0.7/+1.2 dB
down to −4…−28 dB, which looked decisive at n=6. It is not: repeating *identical*
input gave spreads of 2.4, 7.2 and 27.9 dB on the same metric, and over 8 lines
the median punctuation effect was +4.6 dB — the wrong direction. The effect was
sampling noise. The punctuation gap is real and may still be worth closing for
the subtitles, but it is not this bug.

## Root cause

`torch.manual_seed()` was called once per *chunk*, and the per-row sampler loop
then drew from that one global stream. A row's draw therefore depended on its
slot index and on how many draws its chunk-mates had already made. Because chunks
are formed by sorting **all** of a job's sentences by length, adding one sentence
anywhere reshuffled every other line's prosody. Three probes:

```
max_batch=1, same text x3            -> 1 distinct output of 3   (deterministic)
max_batch=8, 3 identical in a chunk  -> 3 distinct outputs of 3
same line alone            -> md5 92b4879e
same line + 1 chunk-mate   -> md5 11cd13ac      (changed)
```

The module docstring had described this as "expected for sampled decoding". It was
never measured against the audio.

## Decision

Give every row its own `torch.Generator`, seeded from a **content hash** of the
request (blake2b over prompt text and reference text, not `hash()`, which is
salted per process). The retry attempt is added in, so an overflow retry still
redraws rather than reproducing the run that failed to terminate.

Upstream's `_sample` takes no generator, so `_sample_codes` is a faithful copy of
its 12 lines with the generator added. Only the *draw* is reimplemented: the
sampled codes are handed back to upstream's `_sampler_step` as a one-hot at
temperature 0, which sends it down its argmax path, returns them unchanged, and
then runs its own delay ramp-up and end-of-codebook bookkeeping. Copying a
sampler is cheap; forking a state machine is not.

## What this does and does not buy

**Achieved — reproducibility.** The same sentence set now yields byte-identical
audio on every run, and call order no longer matters (chunking sorts by cap, seeds
are content-addressed). Two identical requests in one chunk now produce identical
audio, which they did not.

**Not achieved — batch invariance, and it is not an RNG problem.** Changing which
sentences share a chunk still changes a line's audio: batched matmuls and the
left-padded prefill (`seq = max` over the chunk) do not give bitwise-identical
logits at different batch shapes, and at temperature 0.8 a one-bit logit
difference can flip a multinomial draw. Fixing that needs bitwise-stable kernels
across shapes, not a seeding change. This is why `test_batched_rows_match_
standalone_rows` pins temperature to 0 — argmax absorbs the drift that sampling
amplifies. Practical consequence: re-running a clip reproduces exactly, but adding
or removing one sentence can redraw unrelated lines.

**The defect itself is not fixed.** Reproducibility makes it stable and
measurable, not correct. At temperature 0.8, **38 % of short lines (9 of 24) still
end flat or rising.**

## Consequences

- **No TTS A/B in this project was measurable at n=1** before today, including the
  emotion-tag rounds of the same date (quality issue #15), which had one sample
  per cell against a per-line variance of 2.4–27.9 dB. That conclusion had an
  independent full-clip confirmation from the author and is not being reopened,
  but #15's presentation of the per-line A/B as strong evidence is corrected.
- Temperature is **not** the lever, measured: 0.8 / 0.75 / 0.7 / 0.6 give 4/10,
  2/10, 5/11, 4/12 flat-or-rising — non-monotonic and inside binomial noise at
  n≈10 — with **0 overflows at every setting**, on short lines. (The ADR
  `2026-06-20-higgs-v3-tts-mlx-q8` finding that low temperature causes overflow
  was measured on a *long* line and is not contradicted for that case.)
- The lever that works is **selection**, now that draws are reproducible and the
  metric identifies a bad one: best-of-2 takes 38 % → 21 %, best-of-3 → 4 %. One
  pass over the 37 short lines of a 5-minute clip measures 21.7–29.5 s, so
  best-of-2 on lines of ≤ 4 words costs roughly +22 s on a 139 s TTS stage
  (+16 % of the stage, +8 % end to end). Not implemented.
- A caution for whoever implements it: best-of-3's selected median is −19.3 dB,
  at the extreme end of the natural range (unflagged lines run −6.8 to −19.3), so
  optimising this metric hard risks trading "final word stressed" for "final word
  swallowed". A target **band** is likely better than a minimum.

## Revisit

- If selection ships, this metric becomes load-bearing and needs ear validation as
  a *selector*, not just as a detector.
- If a future transformers or checkpoint release makes the batched decode
  bitwise-stable across shapes, full batch invariance becomes reachable and the
  temperature-0 pin in the equivalence test can widen.
- The punctuation gap (62 % of lines unpunctuated, from an ASR that emits none) is
  still open on its own merits — subtitles read worse for it.
