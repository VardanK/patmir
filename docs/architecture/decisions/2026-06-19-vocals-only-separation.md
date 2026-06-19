# Vocals-only separation (htdemucs_ft vocals-specialist single model)

**Date:** 2026-06-19
**Status:** locked
**Stage:** separate

## Context

`separate` was the single most expensive pipeline stage — ~39s of wall time per
60s of input (≈42% of the batch pipeline), running `htdemucs_ft`, which is a
**bag of 4 fine-tuned Demucs v4 models** (one specialist per source:
drums/bass/other/vocals). Patmir only needs **vocals** (for ASR, ref clips,
loudness-match, and the ducked bed) plus an **accompaniment** music bed — it
never uses the individual drums/bass/other stems.

## Decision

Run **only the vocals-specialist sub-model** (Demucs signature `04573f0d`)
instead of the full 4-model bag, and derive `accompaniment = mix − vocals`.

New candidate `htdemucs_mlx_vocals` (`vocals_only=true` on the existing
`htdemucs_mlx` module). It is the **batch `separate` default** as of 2026-06-19.

## Why this is quality-neutral on vocals

`htdemucs_ft`'s `BagOfModels` weight matrix is the **identity**:

```
model 0 (drums)  → [1,0,0,0]
model 1 (bass)   → [0,1,0,0]
model 2 (other)  → [0,0,1,0]
model 3 (vocals) → [0,0,0,1]   ← sig 04573f0d
```

i.e. the bag's vocals output is **100% the vocals sub-model**, zero contribution
from the other three. So running only it yields **mathematically identical
vocals**. Empirically: A/B on a 30s masha clip → vocals **corr 0.998** vs the
full bag; author ear-confirmed vocals indistinguishable and the `mix − vocals`
music bed clean (2026-06-19).

## Result

- Separate stage: **~39s → ~11s per 60s** of input (~3.6× faster end-to-end; the
  pure model inference is ~7.6× faster — RTF 0.14 vs 1.04 — but ffmpeg extraction
  + Silero VAD are fixed costs that don't shrink).
- Whole batch pipeline: **1.56× → ~1.07× realtime**; separation dropped from 42%
  → 17% of the pipeline (TTS is now the largest stage).

## Mechanics

`demucs.pretrained.get_model("04573f0d")` resolves the single sub-model by
signature; the candidate registers it in `demucs-mlx`'s `MLX_MODEL_REGISTRY` as a
non-bag model at build time. Output contract (`SeparatedAudio`, Silero VAD split,
artifact writing) is unchanged — only the stem-producing call + accompaniment
derivation differ.

## Revisit

- If the `mix − vocals` residual bed proves dirty on some clip (vocal bleed the
  model missed), fall back to the full `htdemucs_mlx` bag (still registered).
- Watch-along still uses `htdemucs_mlx_fast` (base single model); switching it to
  `htdemucs_mlx_vocals` would give ft-quality vocals at similar speed — open option.
- On a CUDA box (MLX is Apple-only), use the PyTorch `htdemucs` fallback.
