# Measurements — watch-along 2× on WSL2 + RTX 3090

Every number the plan rests on, with the script that produced it. Run all of these
**with nothing else on the GPU**; see "Contention" below for why that is not optional.

| script | what it establishes |
| --- | --- |
| `probe-hardware.py` | roofline + dtype + quantization-path facts. Self-guarding (refuses on a busy card). Raw output in `hardware-probes.out`. |
| `probe-asr-dtype-28s.py <wav>` | real Qwen3-ASR bf16 vs fp16+accumulate at production shapes, with an A/B/A control and a transcript diff. |
| `probe-asr-dtype.py <wav>` | first cut, 3.5 s clips. **Superseded** — clips were too short to be representative. Kept because its flaw is instructive. |
| `probe-tts-fullchunk.py` | real Higgs 4B, 8 sentences. **Inconclusive by construction** — see below. |
| `probe-tts-perstep.py` | isolated per-decode-step forward at fixed KV length. **This is plan Task 2**; not yet run to completion. |

Input for the ASR probes: a 28 s mono 16 kHz chunk, matching the candidate's
`chunk_seconds=28`:

```
ffmpeg -y -i samples/sample_1.mkv -t 28 -ac 1 -ar 16000 -f wav chunk28.wav
```

## Contention — read before trusting any number here

The same `probe-hardware.py` run twice, once with the patmir server resident
(21.8 GB VRAM) and once on an idle card:

| | server resident | idle card |
| --- | --: | --: |
| HBM read+write | 295 GB/s | **807 GB/s** |
| tiny kernel launch | 44.7 µs | **10.5 µs** |
| W8A8 `_int_mm` | 23.7 TOPS | **59.2 TOPS** |
| fp16+accum vs bf16 (decode GEMM) | 1.41× | 1.46× |
| fp8 `_scaled_mm` | unavailable | unavailable |
| fp16-accum relative error | 12× | 12× |

**Ratios survive contention; absolute roofline figures do not** — and it is the
absolute per-step bandwidth floor that decides whether q8 and fp16 are worth
anything at all. `probe-hardware.py` therefore refuses to run above
`VRAM_BUDGET_MB` (2048) rather than quietly reporting a third of the real
bandwidth. The same caution applies to `runs/bench/*.json`: a bench baseline
taken while something else held the card is not comparable to one that was not.

## Results, idle card, 2026-08-28

RTX 3090, SM 8.6, torch 2.11.0+cu128, WSL2. Full output: `hardware-probes.out`.

### The accumulate finding

```
decode GEMM [8,4096]@[4096,14336]   weight 117 MB   HBM roof 0.147 ms
  bf16 default              0.223 ms   527 GB/s   1.00x
  bf16 +reduced_reduction   0.216 ms   544 GB/s   1.03x   <- no bf16 route to the win
  fp16 default (fp32 acc)   0.220 ms   534 GB/s   1.01x   <- the trap: reads as "no gain"
  fp16 +fp16_accum          0.153 ms   769 GB/s   1.46x   <- 95% of the card's roof
```

The 3090 has a second tensor-core datapath — fp16 with **fp16 accumulate** — that
PyTorch does not use by default. Requires **both** `dtype="float16"` and
`torch.backends.cuda.matmul.allow_fp16_accumulation = True`; either alone is worth
~1%, which is exactly how two independent investigations concluded "fp16 is a no-op
on Ampere".

The mechanism is **not** the datasheet's 142-vs-71 TFLOPS line. At B=8 this GEMM
runs at 16 flops/byte and FLOP-bound would need ~80, so FLOPs are not what binds.
fp32 accumulators cost twice the register/shared-memory budget and cut occupancy on
a thin GEMM, so that kernel reaches 527-544 GB/s of the card's 807 while the
fp16-accumulate kernel reaches 769. On genuinely compute-bound shapes it does show
up as FLOPs (B=64: 2.25×).

Cost: fp16-accumulate carries **12× the relative error** of fp32-accumulate
(2.19e-3 vs 1.76e-4). Range is not the problem — activation absmax measured 7.0
against fp16's 65504 ceiling.

### Quantization paths

```
fp8 _scaled_mm      UNAVAILABLE - requires compute capability >= 9.0 or 8.9 (3090 is 8.6)
W8A8 _int_mm        59.2 / 60.0 TOPS at n=4096 / 8192   (datasheet: 284 dense)
```

fp8 is closed by silicon. The `_int_mm` figure rules out **W8A8** compute — it does
**not** bear on q8, which is **W8A16**: weights stored int8, dequantized in-register,
matmul still in bf16/fp16. q8 buys *bytes*, not TOPS, which is the right lever for a
bandwidth-bound decode and the only one that lowers the floor itself.

### Per-step overhead constants

```
tiny kernel launch   10.5 us     (~2-3x a native-Linux box; WSL2 tax)
tensor.cpu() sync    29.8 us     the decode loop pays ONE PER ROW PER STEP
```

### Real Qwen3-ASR-1.7B — why the GEMM win does not reach the stage

8 × 28 s chunks, 382 prompt tokens, **62 generated tokens in every arm**, through the
candidate's own `apply_transcription_request`:

```
bf16          2309.7 ms   1.000x
fp16_f16acc   2176.8 ms   1.061x
bf16_again    2218.7 ms   1.041x   <- A/B/A control: noise floor 4.1%
transcripts byte-identical across all three arms
```

**1.061× reproduced exactly on both a contended and an idle card**, so the effect is
real and small. Against a 4.1% control it is barely outside noise.

Why: **37.3 ms/step against a 4.21 ms/step weight-read floor (1.7B × 2 B / 807 GB/s)
= 8.8× above the floor, ~89% of the stage is not GEMM at all.** At 10.5 µs/launch
that implies ~3500 kernel launches per step. A 1.46× GEMM therefore buys 6% of the
stage. Amdahl, exactly — and this is the plan's central finding: **the bottleneck is
per-step host overhead, not arithmetic precision.**

(The contended run read 41.9 ms/step / 9.8× / 90% — same conclusion.)

### Real Higgs 4B, full chunk — inconclusive, recorded so it is not misread

```
bf16   7288.5 ms      fp16_f16acc   8166.6 ms (0.89x)      bf16_again   7973.1 ms (0.91x)
```

**The bf16 control varied 9.4% against itself, so this probe cannot resolve a dtype
effect**, and the apparent 0.89× for fp16 is not a result. Two confounds: the card was
contended, and `audio_head.weight` loads randomly initialised, so the sampler's
termination is a random draw and the arms did not run the same number of steps.

It did establish two things. bf16 output is **md5-identical run to run** (the
2026-08-24 sampling-reproducibility fix works), so any future audio delta is
attributable. And fp16 changes the audio, as expected from perturbed logits at
temperature 0.8.

The clean version — fixed KV length, no sampler — is `probe-tts-perstep.py`, plan
Task 2. It needs an idle card: two resident 4B models produced
`dxgk: dxgkio_create_allocation: Ioctl failed: -75` and a SIGBUS during this session.

### The 4B decode floors that gate Tasks 2 and 8

```
bf16 (8 GB/step)   9.91 ms/step
q8   (4 GB/step)   4.96 ms/step
```

Whether q8 is worth ~2× or ~1.16× depends entirely on where the measured step lands
relative to 9.91 ms. That is the single number Task 2 exists to produce.
