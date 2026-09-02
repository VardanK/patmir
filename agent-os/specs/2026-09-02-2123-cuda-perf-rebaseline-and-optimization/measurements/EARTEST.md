# Ear-test: the decode attention kernel — RESOLVED

**Verdict 2026-09-02: approved.** The author listened to
`runs/20260902-230429-sample_1/out.mkv` — the warm repeat of the graph+gqa
bench, 17.33 ms/step, agg RTF 0.277 — and reported it "sounds a little better"
than the sdpa rendering. `decode_attention = "gqa"` is now set in
`patmir.toml`, moving `config_hash` **2cc3dd87 -> 4a56ed45**, so cached dubs
regenerate once.

Read that verdict as **neutral-to-positive, not as an improvement.** Issue #19
measured this TTS's draws spreading 2.4-27.9 dB on identical input, so one
comparison cannot establish a direction. What it does establish is that the
faster kernel is not worse — which is all the decision required.

---

The question, as it stood before the verdict. Everything else in this spec is
byte-identical to what the pipeline produced before, and needed no listening.

## What to compare

| | file | what it is |
| --- | --- | --- |
| **A** | `runs/20260902-222021-sample_1/out.mkv` | current pipeline, unchanged audio |
| **B** | `runs/ab-gqa/out.mkv`, and `runs/20260902-230429-sample_1/out.mkv` (the one judged) | same clip with the faster decode attention |

Both are `samples/sample_1.mkv`, same translation, same refs, same speakers.
Only the TTS sampling draw differs.

## What it buys

| | ms/step | TTS stage | agg RTF |
| --- | ---: | ---: | ---: |
| eager (before this spec) | 45.55 | 60.55 s | 0.377 |
| + CUDA graph (**shipped, byte-identical**) | 27.15 | 39.51 s | 0.309 |
| + this attention kernel (**your call**) | 17.33 | 29.27 s | 0.277 |

So the kernel is worth a further 10 % off the whole pipeline, on top of the
18 % the graph already gave for free.

## What it costs

It changes **all 80 lines**. Not "some lines" — the reduction order differs, and
at temperature 0.8 a one-ulp logit difference flips a multinomial draw, so every
sentence re-rolls. Individual line durations move by up to 1.24 s; total speech
is within 1.4 % (213.6 s -> 210.5 s).

Expected quality is unchanged: it is a different draw from the same
distribution, not a worse model. But quality issue #19 measured this TTS's
draws varying by up to **26 dB** on utterance-final level, with 38 % of short
lines ending flat or rising — so a full re-roll is a re-spin of a known
lottery. B could land better or worse than A by luck. That is exactly what
ears settle and metrics do not.

It stays reproducible: two runs of B are byte-identical to each other.

## How it was enabled

One line in `patmir.toml` under `[[stages.tts.candidates]]` for
`higgs_v3_torch_bf16`:

```toml
decode_attention = "gqa"
```

It goes in the config rather than being defaulted on in code deliberately:
`ui.cache.config_hash` hashes the candidate entry, so setting it there
correctly invalidates cached dubs. **That re-dubs every cached video once** —
the same accepted price the audience guardrail paid on 2026-08-28. Defaulting
it in code would instead leave one cache key able to hold either rendering.

## Reverting

Set `decode_attention = "sdpa"` on that candidate. The code default is still
`sdpa`, so removing the line also reverts. Either way the pipeline returns to
byte-identical output and keeps the graph's 18 %.
