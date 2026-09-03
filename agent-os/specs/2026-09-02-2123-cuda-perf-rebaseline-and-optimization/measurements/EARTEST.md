# Ear-test

Two questions were raised by this spec. One is open and is the last thing
standing between this work and done; the other is already settled.

- **[OPEN] `max_batch = 12`** — the batch size the TTS decode runs at. Moved
  8 -> 12 after the CUDA graph made batch width matter. Re-rolls every line.
- **[RESOLVED, approved] `decode_attention = "gqa"`** — the faster decode
  attention kernel. Also re-rolls every line. Verdict below.

Everything else in this spec is **byte-identical** to what the pipeline
produced before, and needed no listening.

---

## [KEPT, NOT LIKED] `max_batch = 12`

**Verdict 2026-09-03.** The author listened to
`runs/20260902-234938-sample_1/out.mkv` and rejected it — "audio contains
artifacts, pauses and the translation quality itself degraded", and separately
"the very first speech (male actor) starts with garbled sound". On
investigation **none of the three is attributable to `max_batch`** (see
"What the investigation found" below), and after that the author's decision was
to **keep** 12 and the gqa kernel: *"I would not say that I like any of them,
but let's keep for now."*

So this is not an approval. It is a decision to stop paying for a comparison
that the pipeline cannot currently support, because run-to-run variance is
larger than the thing being compared — now filed as **quality issue #28**.

### What the investigation found

The rejected run was compared against the approved one, which confounds four
independent lotteries. Holding text, refs and emotion fixed and varying only
the batch width (`runs/ab-mb8` vs `runs/ab-mb12`, both resumed from the same
job dir at `--resume-from tts`):

| symptom | measured |
| --- | --- |
| garbled first line | onset flatness **0.0002** in the approved run, **0.0076** rejected, **0.0056** at batch 8 with identical text and refs — so noisy at both widths |
| artifacts | 0 non-finite samples, 0 clipped in the mix, no discontinuity; batch 8's worst intra-line step (0.673) is *larger* than batch 12's (0.667) |
| pauses | ASR covers 151.4 s of 301.1 s with 7 gaps over 5 s (81.1 s total, max 23.3 s) in **all ten runs**, the approved one included — structural, not new |
| translation | 24 of 66 shared lines differ; each translate call is one sentence with no history, so concurrency cannot touch wording. The **emotion tag is in the prompt**, and 12 of those 66 sentences carry a different label |
| the age guard | `audience: teenager`, `audience_honored: true` in both runs, clause 1331 chars, no unsteerable warning — the guard was live, and does not govern police slang or formality |

The cause of the garble is the **pinned reference clip changing**:
`tts_pin_speaker_ref` picks `max(duration)` over a speaker's refs, those refs
come from diarization turns, and diarization emitted 91-117 turns across ten
runs of one file. The approved run pinned SPEAKER_01 from 237.4-248.3 s, the
rejected one from 273.2-281.4 s — a different cloned voice for the whole video.
Issue #28.

A test gap the report exposed was closed: the gqa decode kernel had never been
exercised on **ragged per-row masks**, which is exactly what a wider batch
produces. It is correct there, and rows provably do not leak into each other.

---

## The question as it was originally posed

### What to listen to

| | file | what it is |
| --- | --- | --- |
| **A** | `runs/20260902-230429-sample_1/out.mkv` | `max_batch = 8` — the rendering you already approved |
| **B** | `runs/20260902-234938-sample_1/out.mkv` | `max_batch = 12` — the current pipeline |

From Windows:
`\\wsl.localhost\Ubuntu-26.04\home\vardan\orca\workspaces\patmir\performance-improvements\runs\20260902-234938-sample_1\out.mkv`

B is the full final configuration: `max_batch = 12`, the gqa kernel, the cheap
ASR language probe, translate at concurrency 12, batched emotion. 78 sentences,
**0 overflows**, agg RTF 0.211, peak 16.6 GB VRAM, max 78 C.

### What it buys

| | TTS stage | agg RTF | batch wall |
| --- | ---: | ---: | ---: |
| `max_batch = 8` | 29.27 s | 0.277 | 83.3 s |
| `max_batch = 12` | **23.84 s** | **0.211** | **63.4 s** |
| `max_batch = 16` | 27.03 s | 0.228 | — |

16 was measured **worse**, and it produced the only TTS overflow in this whole
body of work: a chunk costs `max(steps)` over its rows, so a wider batch widens
the token-cap spread inside it (total steps rose 888 -> 1032 across one fewer
chunk, live-row fraction 0.592 -> 0.468). VRAM was never the binding
constraint, which the plan had expected it to be.

### What it costs

The same thing the gqa kernel cost: **every line re-rolls.** Batched decode is
not bitwise-invariant across shapes, so regrouping 78 sentences into 7 chunks
of ~11 instead of 10 of ~8 redraws all of them. Same model, same refs, same
speakers, same distribution — a different draw.

Translation text also differs between A and B, independently: the translator is
a cloud call at temperature 0.2, so no two runs agree exactly. **So this is not
a line-by-line A/B.** Listen for defects, not for differences:

1. **Wrong emphasis on a final short word** (quality issue #19 — the known
   defect; 38 % of short lines end flat or rising, and a re-roll re-spins that
   lottery).
2. **A character changing voice mid-scene**, or speaking another character's
   line (issues #16/#17/#18 — should be unchanged, but a re-roll is when you
   would notice a regression).
3. **Garbled or truncated speech** — the overflow signature. The counters say
   0, so hearing one would mean the counter is wrong.
4. **Over-emotive delivery** (issue #15) — the tag map is unchanged, so this
   should sound like the rendering you approved on 2026-08-24.

### Reverting

One line in `patmir.toml`, `[[stages.tts.candidates]]` for
`higgs_v3_torch_bf16`:

```toml
max_batch = 8    # currently 12
```

`max_batch` changes output, so it is correctly in the cache key —
`config_hash` moved **4a56ed45 -> 3f421293**. Reverting moves it back and the
library regenerates once more, so if 12 is going to be rejected, rejecting it
before a bulk re-dub is cheaper.

---

## [RESOLVED] The decode attention kernel

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

### The question, as it stood before the verdict

#### What was compared

| | file | what it is |
| --- | --- | --- |
| **A** | `runs/20260902-222021-sample_1/out.mkv` | current pipeline, unchanged audio |
| **B** | `runs/ab-gqa/out.mkv`, and `runs/20260902-230429-sample_1/out.mkv` (the one judged) | same clip with the faster decode attention |

Both are `samples/sample_1.mkv`, same translation, same refs, same speakers.
Only the TTS sampling draw differs.

### What it buys

| | ms/step | TTS stage | agg RTF |
| --- | ---: | ---: | ---: |
| eager (before this spec) | 45.55 | 60.55 s | 0.377 |
| + CUDA graph (**shipped, byte-identical**) | 27.15 | 39.51 s | 0.309 |
| + this attention kernel (**your call**) | 17.33 | 29.27 s | 0.277 |

So the kernel is worth a further 10 % off the whole pipeline, on top of the
18 % the graph already gave for free.

### What it costs

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

### How it was enabled

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

### Reverting

Set `decode_attention = "sdpa"` on that candidate. The code default is still
`sdpa`, so removing the line also reverts. Either way the pipeline returns to
byte-identical output and keeps the graph's 18 %.
