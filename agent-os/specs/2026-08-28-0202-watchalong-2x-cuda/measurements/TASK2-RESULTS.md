# Task 2 — where one Higgs decode step actually goes

Measured 2026-08-28 on an idle RTX 3090 (1.3 GB resident), `probe-tts-perstep.py`,
raw output in `perstep.out`. Batch 8, bf16, fixed KV length, **no sampler in the
timed path** — so no termination variance and nothing to confound the arms, which
is what made the earlier full-chunk probe unusable.

## The step, attributed

Decode-active weights (transformer blocks + audio head/embedding, excluding the
389 M tied text embedding a seq-1 audio decode never reads): **7.31 GB**, so the
bandwidth floor is **9.06 ms/step** at 807 GB/s (q8 would be 4.53).

```
kv=300    43.13 ms/step   4.8x floor   79% overhead
kv=600    44.06 ms/step   4.9x floor   79% overhead
kv=1200   51.60 ms/step   5.7x floor   82% overhead
cache-growth slope: 9.42 ms per 1000 tokens of KV
```

| term | ms | share | note |
| --- | --: | --: | --- |
| weight bandwidth (floor) | 9.06 | **21%** | 7.31 GB / 807 GB/s |
| sampler loop (as shipped) | 4.73 | **11%** | per-row draw + one-hot + `.cpu()` per row |
| residual = launch / dispatch | 29.34 | **68%** | ~2800 launches at 10.5 µs, ≈78/layer over 36 layers |

**79% of the step is not arithmetic.** This confirms the plan's central hypothesis
and refutes the prior investigation's B0.5 ("kernel launches are a single-digit-%
overhead on bandwidth-bound decode") on the TTS stage as well as on ASR.

## Two hypotheses of mine, refuted — do not re-chase

1. **`DynamicCache`'s `torch.cat` is NOT a significant cost.** I predicted ~2.9 ms/step
   at kv=1000 from the realloc-and-copy. Measured `StaticCache` against it:
   **0.98× at kv=300 and 1.06× at kv=1200** — i.e. no win at short context and ~6% at
   long. The cache-growth slope (9.42 ms per 1000 tokens) is real but only bites at
   contexts longer than these lines reach. Dropped from Task 3.
2. **fp16 + fp16-accumulate is worthless here, and my 1.46× was a shape artifact.**
   At the model's *true* MLP shape:

   ```
   [8,2560]@[2560,9728]  weight 50 MB
     bf16          0.0637 ms   782 GB/s   1.00x   <- already 97% of the 807 roof
     fp16+f16acc   0.0687 ms   725 GB/s   0.93x   <- SLOWER
   ```

   bf16 already reaches the roof at this shape, so there is no gap for the accumulate
   datapath to close. The 1.46× was measured on `[8,4096]@[4096,14336]` — a 117 MB
   weight this model does not have, where bf16's kernel selection happens to be poor
   (527 GB/s). **Lesson: a controlled, interleaved, reproducible microbenchmark is still
   worthless if the shape is wrong.** It also retroactively explains the two real-model
   fp16 results previously written off as noise (ASR 1.061×, Higgs full-chunk 0.89×):
   fp16 is neutral-to-negative at our shapes. The fp16 arm is dropped from Task 8 and
   from Task 9.

## Levers, all sized against the same 43.13 ms step

| lever | saving | share | risk |
| --- | --: | --: | --- |
| **CUDA graphs / `torch.compile`** on the forward | up to ~29 ms | **68%** | the big one; equivalence testable at temp 0 |
| vectorize the sampler across the batch | 4.56 ms | 10.6% | must preserve per-row generators (see below) |
| drop the per-row `.cpu()` only | 1.79 ms | 4.2% | trivially equivalent |
| **q8** (halves the 9.06 ms floor) | 4.53 ms | 10.5% | *at today's step size* — see sequencing |
| `StaticCache` | ~0 | 0% | **refuted** |

## Sequencing, now quantified rather than argued

q8's value depends entirely on step size, which is why it must come last:

```
today (43.1 ms/step)             q8 saves 4.53 ms -> 1.12x
after graphs (~11.2 ms/step)     q8 saves 4.53 ms -> 1.68x
graphs + q8 together                              -> ~6.4x on the decode step
```

So the ordering in the plan was right, and the magnitudes are now measured rather
than assumed: **overhead first, q8 second, fp16 never.**

## Implementation constraint on the sampler

The 0.17 ms figure came from a single batched `multinomial` over `[B*N, V]`. That is
**not** equivalence-preserving: ADR `2026-08-24-tts-sampling-reproducibility` requires
one RNG stream *per row*, seeded from that request's content, so a line's audio cannot
depend on its batch slot. A shipped vectorization has to keep per-row generators —
batching the top-k/top-p masking and the one-hot/scatter while drawing per row, or
pre-drawing uniforms per row and doing the inverse-CDF lookup batched. Expect less than
the full 4.56 ms. The trivially-safe 1.79 ms (dropping the `.cpu()`) needs none of that
care and should land first.

---

## Why fp16 scored the same as bf16 when the datasheet says 2× — and proof it is not a harness bug

**The roofline answer.** For `[M,K]@[K,N]` where the weight dominates bytes moved,
`FLOPs = 2·M·K·N` and `bytes = K·N·2`, so **arithmetic intensity = M = the batch size**.
The machine's ridge point is peak-FLOPS ÷ peak-bandwidth:

```
bf16          63.3 TFLOP/s / 807 GB/s ~=  78 FLOP/byte
fp16+f16acc  102.0 TFLOP/s / 807 GB/s ~= 126 FLOP/byte
```

Our decode runs at AI = 8, roughly **10× below the bf16 ridge**. In the memory-bound
region `time = bytes / bandwidth` and FLOPS does not enter the expression; doubling peak
TOPS moves the ridge point (78 → 126) without changing the slope we are on. Confirmed by
the measurement: bf16 achieves **784 of 807 GB/s, 97% of the roof**. There was never 2%
available, let alone 2×.

Sweep at the true shape, showing the predicted crossover:

```
   B   AI   bf16 GB/s  TFLOPs | fp16acc GB/s  TFLOPs | ratio  regime
   8    8       784      6.3  |      728      5.8    | 0.93x  memory-bound
  16   16       702     11.2  |      790     12.6    | 1.13x  memory-bound
  24   24       603     14.5  |      700     16.8    | 1.16x  memory-bound
  32   32       767     24.6  |      691     22.1    | 0.90x  memory-bound
  64   64       732     46.9  |      654     41.9    | 0.89x  memory-bound
 128  128       431     55.1  |      636     81.4    | 1.48x  COMPUTE-bound
```

The ratio is ~1.0 (±0.15 noise) everywhere below the ridge and jumps to 1.48× at B=128,
exactly where AI crosses 78. **Consequence for Task 6:** raising `max_batch` to 16-24
takes AI to 16-24, still 3-5× short of the ridge, so fp16 does not become relevant at
any batch size this pipeline would plausibly use.

**And the earlier 1.46× was not TOPS either.** At `[8,4096]@[4096,14336]` fp16+accum was
*flat at ~0.15 ms from B=1 to B=64* — pinned to the memory roof, as the roofline requires
for constant bytes — while bf16 *rose* from 0.167 to 0.347 ms moving identical bytes,
reaching only 337 GB/s. A kernel at the memory roof cannot do that, so bf16 there was
neither memory- nor compute-limited: cuBLAS simply selects a poorly-occupied kernel for
that shape, and fp16-accumulate's lower register pressure dodges it. That win was
kernel-selection quality at a shape this model does not have.

### Validation that the two arms are genuinely distinct

Asked directly whether both measurements could have run the same dtype. Three checks:

1. **Distinct computations.** Dtypes, output dtypes and flag read-backs all as intended;
   three distinct output md5s; and pairwise `max|diff|` non-zero for every pair — which is
   impossible if two arms ran the same computation:

   ```
   bf16        vs fp16_f16acc   1.25e+00
   bf16        vs fp16_f32acc   8.75e-01
   fp16_f16acc vs fp16_f32acc   7.81e-01   <- same dtype; the FLAG alone changes the result
   ```

   The third row isolates the flag, proving `allow_fp16_accumulation` switches the datapath
   rather than being silently ignored.

2. **Positive control in the same harness.** 8192³, deeply compute-bound (AI ~ 8192):

   ```
   bf16          17.18 ms    64.0 TFLOP/s
   fp16_f16acc   10.77 ms   102.1 TFLOP/s    1.60x
   ```

   The harness *can* see the datapath when the work is compute-bound, so the null at our
   shape is a real null.

3. **The null is tight, not noise.** Nine interleaved rounds at the true shape: bf16 median
   0.0634 ms (min 0.0633, max 0.0676) vs fp16+accum median 0.0682 ms (min 0.0680, max
   0.0695). Distributions barely overlap — fp16's *minimum* exceeds bf16's median — so fp16
   is **reliably ~7% slower** here, not equal-within-noise.

Script: `probe-validate-dtype.py`.
