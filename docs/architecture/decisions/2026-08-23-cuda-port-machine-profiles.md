# 2026-08-23 — CUDA port, slice 1: machine profiles + the torch candidate set

Status: **accepted** (mechanism), **unverified** (the CUDA candidates have not been executed)

## Context

The dev machine moved from an M1 Pro laptop to a WSL2 + RTX 3090 workstation. All five model
stages were MLX-only:

| Stage | Apple Silicon | Runs on CUDA? |
| --- | --- | --- |
| separate | `htdemucs_mlx_vocals` | no — `demucs-mlx` is Apple-only |
| diarize | `sortformer_mlx` | no — Sortformer reaches us through `mlx-audio` |
| emotion | `emotion2vec_plus_base` | **yes, unchanged** — resolves device `auto` to cuda |
| asr | `qwen3_asr_1_7b_mlx_8bit` | no — but a torch sibling was already registered |
| translate | `openrouter_gemini_flash` | **yes** — cloud |
| tts | `higgs_v3_mlx_q8` | no — our own hand-written MLX port |

So the port needs a different candidate for four stages. The question this ADR answers is not
*which* candidates (that is mostly mechanical) but **how the two machines share one repo**.

## Decision 1 — machine profiles in `patmir.toml`, selected by environment

`[pipeline.profiles.<name>]` overlays the `[pipeline]` stage selectors. The CUDA box exports
`PATMIR_PROFILE=cuda`; `patmir dub|bench|ui --profile cuda` is sugar that exports the same variable
so it also reaches `streaming_config()`, which is called deep in the watch-along path with nothing
but a config path.

Resolution order, last wins:

```
[pipeline]  <  [pipeline.profiles.<p>]  <  code overrides  <  [pipeline.profiles.<p>.<layer>]
```

The `<layer>` tier exists for exactly one reason: `streaming/runner.py:streaming_config()` hardcodes
`separate = "htdemucs_mlx_fast"` as a code override, and that candidate does not exist off Apple
Silicon. `[pipeline.profiles.cuda.streaming]` sits above the override so the profile can displace
it, while a Mac watch-along run — no profile active — keeps its own choices untouched.

### Alternatives rejected

- **Edit `[pipeline]` on the CUDA box.** The one file that decides which model runs would diverge
  permanently between two active working copies, on every branch, forever.
- **A second full config file (`patmir.cuda.toml`).** Duplicates a ~700-line candidate registry;
  the copies drift.
- **Autodetect from `torch.cuda.is_available()`.** The tempting option, and the wrong one. Which
  model ran has to be a stated choice or a benchmark stops being attributable, and it would remove
  the ability to run a CPU or MLX set on the CUDA box deliberately. For the same reason an unknown
  profile name is a **hard error** rather than a fall back to the defaults — on a CUDA box, falling
  back means quietly trying to load MLX-only candidates.

Storage roots are unaffected. Profiles change `entries`, which `ui.cache.config_hash` hashes, so
the two machines have different cache keys and cannot serve each other a dub built from a different
model set (verified: `3fceb31d` vs `e9ab98d0`).

## Decision 2 — `higgs_v3_torch_bf16` reuses upstream's model, not its decode loop

The MLX candidate (`eval/candidates/tts/higgs_v3_mlx.py`, 367 lines) exists because torch-MPS
cannot do fast quantized matmul, and the 4B decode is memory-bandwidth-bound: ~69 ms/step at
torch-MPS bf16 versus ~28 ms/step at MLX q8. Quantization was the only lever available, so we ported
the AR graph — weight remap, fused multi-codebook embed/head, delay-pattern sampler, EOC state
machine — by hand.

None of that argument survives on a discrete NVIDIA GPU: bf16 is native, bandwidth is ~4.7x, and
`multimodalart/higgs-audio-v3-tts-4b-transformers` ships a complete transformers implementation with
a public `generate_speech()`. The first cut of this candidate was therefore a thin wrapper around
that call.

**It no longer is, and the reason is batching.** `generate_speech()` is batch-1, and batch-1 is
exactly the wrong shape for a bandwidth-bound decode: every step reads all 4B weights to produce one
sentence's next frame. Reading them once to advance B sentences is close to free, so batching is the
single largest lever on this stage — and TTS is ~69% of batch wall. There is no way to reach it
through upstream's public API, so the decode loop now lives in the candidate.

That is still not a second port. Everything model-specific is reused from the *loaded checkpoint's
own module*, reached through `sys.modules[type(self.model).__module__]` rather than re-imported: the
delay-pattern helpers, the codec constants, and — the part that matters — `_SamplerState` /
`_sampler_step`, so the delay ramp-up and EOC wind-down state machine is upstream's, run once per
row. What the candidate adds is only:

- **the batching mechanics** — left-padded prefill with `position_ids` derived from the attention
  mask, per-row caps and stop state, per-row de-delay and codec decode;
- **reference-code caching** — upstream re-encodes the reference clip on every call; with one pinned
  ref per speaker that is the same clip re-encoded once per sentence;
- **the emotion-bin to inline `<|emotion:..|>` tag map**, byte-identical to the MLX candidate's, so
  a runtime swap cannot change delivery (asserted by comparing the two dicts in
  `test_config_profiles.RuntimeParityTest`);
- **the non-termination guardrail** — length-proportional token cap plus reseed-and-retry.

Three consequences worth recording:

- **Chunks are sorted by expected length.** A chunk costs `max(steps)`, not `mean(steps)`, so a
  one-word line batched with a long sentence waits for it. Requests are sorted by their token cap
  (proportional to text length, which predicts generation length) before chunking, which keeps
  `sum/max` per chunk near the batch size. The sort has to be global to be worth anything, which is
  why the adapter hands the candidate the whole sentence list in one call and takes progress back
  through an `on_chunk` callback instead of chunking itself.
- **Overflow detection became exact.** The `generate_speech` version had to *infer* overflow from
  the output waveform's length against the token cap, calibrating frames-to-samples off the first
  reference encode and giving up entirely on a job with no reference clip. Owning the loop means row
  counts are simply known. The inference, and its calibration, are gone.
- **`synthesize()` is a batch of one.** Deliberately, so there is one decode implementation rather
  than two that can drift.

Sampling is not batch-invariant: rows share one global RNG stream, so a sentence's output depends on
which batch it landed in. That is inherent to sampled decoding rather than a defect, and it is why
the equivalence test pins temperature to 0.

The MLX candidate stays registered. This is a second runtime for one model, not a replacement: the
laptop still needs the MLX one.

### How the batching is verified without a GPU

A tiny randomly-initialized `HiggsMultimodalQwen3` (10M params, 2 layers) runs on CPU in the test
suite, driven through the real decode path. At temperature 0 the comparison is deterministic, and a
row decoded inside a batch comes out **bitwise identical** to the same row decoded alone — across
differing prompt lengths, a reference-conditioned row batched with text-only rows, and a row that
retires mid-batch on a smaller cap while its neighbours keep going. That last case is the one that
matters most: it is the masking path, and if a finished row leaked into a live one the live rows
would diverge. Random weights never emit a stop token, so the EOC path is forced separately by
stubbing the audio head.

This catches the mistakes batching actually invites — padding corrupting a row, `position_ids`
drifting from the attention mask so RoPE sees the wrong positions, finished-row state bleeding
across. It says nothing about GPU throughput, which is still unmeasured.

### A load report that lies, found on first GPU run

The first real load on the 3090 prints:

```
audio_head.weight | MISSING | newly initialized
```

`audio_head` is the fused head behind every audio logit, so a random one means the output is noise.
It is a red herring: the checkpoint omits that key **by design** because upstream lists it in
`_tied_weights_keys` and points it at `audio_embedding.weight` in `tie_weights()`, which runs after
the report is printed. Verified on the card — shared storage, identical norms — and the waveform
statistics agreed (RMS varying 6x across three sentences is content-dependent output, not noise).

Worth recording because the failure mode it *describes* would be invisible. Were the tie ever broken
(a transformers version re-initializing missing keys after tying), the decode would still run,
sentences would still terminate, `overflows` would still read 0 and RTF would look normal. Nothing
in the pipeline or the bench output would flag it; only listening would. So the candidate now asserts
the tie at build time and refuses to start otherwise, and the test suite covers both the invariant
and the refusal.

### One claim from the planning phase that remains wrong

- **Flash-attention is not available.** The upstream port declares `_supports_sdpa` only — no
  `_supports_flash_attn` — so requesting `flash_attention_2` is refused by transformers. The knob
  exists on the candidate (`attn_implementation`) but defaults to `sdpa` and cannot usefully be
  changed until upstream flags support.

(The companion claim, that batched decode was "upstream work or a fork, not a config change", was
half right: it was indeed not a config change. It was ~150 lines in the candidate.)

## Decision 3 — the `cuda` profile's candidate set

| Stage | Candidate | Notes |
| --- | --- | --- |
| separate | `htdemucs_vocals` | **new.** `vocals_only` added to the torch candidate, mirroring the MLX one |
| diarize | `sortformer_nemo` | **new.** Same model as the Mac, on NVIDIA's own NeMo |
| asr | `qwen3_asr_1_7b` | already registered; bf16 rather than 8-bit |
| tts | `higgs_v3_torch_bf16` | **new** |

`separate` is worth the small port rather than falling back to the full `htdemucs_ft` bag: the stage
is 10.7% of batch wall, and the vocals-only trick is ~3.6x on it. The trick's premise was
re-verified independently while writing this — `htdemucs_ft.yaml` lists sub-models
`[f7e0c4bc, d12395a8, 92cfc3b6, 04573f0d]` against an identity weight matrix, so `04573f0d`
contributes weight 1.0 to source index 3 (vocals) and nothing else. It resolves through torch's
`demucs.pretrained.get_model` directly, needing none of the registry patching the MLX path required.

`diarize` was the uncomfortable one and is no longer. The first cut of this profile selected
`pyannote_3_1`, which reintroduced the exact failure that moved diarize off pyannote on 2026-06-15:
community-1 merges short turns from distinct characters into a single speaker, and those characters
then share one pinned ref clip and so one voice. `sortformer_nemo` removes the compromise — it is the
**same model**, run on NVIDIA's own NeMo. NeMo is Sortformer's reference implementation; `mlx-audio`
is the third-party Apple port. So this is two runtimes for one diarizer, exactly like Higgs above,
and the same discipline applies: the label normalization, same-speaker merge, overlap derivation and
VAD filter moved to `_postprocess.py` and are now shared, so the runtimes cannot drift apart in
post-processing and have the difference mistaken for a model difference. That extraction also gave
those functions their first tests — they were entirely uncovered, despite deciding the speaker ids
everything downstream keys off.

Two things about NeMo's runtime are documented thinly, and both are handled defensively rather than
guessed:

- **Input format.** The card specifies mono 16 kHz; the vocals stem arrives at Demucs's native rate
  in stereo. `mlx-audio` and pyannote both resample internally, but whether NeMo's dataloader
  downmixes is documented nowhere, so the stem is converted with ffmpeg into an inspectable job
  artifact (`diarize/vocals_16k_mono.wav`) first.
- **Return shape.** All three Sortformer model cards describe `diarize()` as returning segments
  "formatted as 'begin_seconds, end_seconds, speaker_index'" and **none shows a literal example**, so
  whether an element is a string, a tuple or an object is genuinely unspecified — as is whether the
  result is nested per input file. The parser accepts every plausible shape and **raises** on anything
  else, with the offending value in the message. Guessing here would be the worst option available:
  a silently mis-parsed diarization does not fail, it resurfaces several stages later as two
  characters sharing a voice.

`pyannote_3_1` stays registered as the fallback, because NeMo is the heaviest dependency in the stack
(built from git, wants Cython first) and is the most likely thing to refuse to install.

## Status of verification

Everything in this ADR that is *mechanism* is verified on the Mac: profile resolution and precedence
across all four tiers, the hard failure on an unknown name, distinct cache keys per profile, the
real `cuda` profile selecting no MLX module or `mps` device in either execution mode, and 162 tests
green.

Nothing in it that *touches CUDA* has been executed. `higgs_v3_torch_bf16` was written against the
upstream source and `sortformer_nemo` against the model card, neither against a running GPU;
`htdemucs_vocals` and `qwen3_asr_1_7b` have never run in this pipeline at all.

What *is* covered by tests is the runtime-independent part, which is more than it sounds:

- batch-vs-single decode equivalence, bitwise, on a real (tiny, CPU) model — see above;
- the batching orchestration on stubs: chunking, cap-sorting, output ordering through the reorder,
  the overflow retry touching only failed rows, and the silence fallback;
- the adapter's dispatch to the batch path, including that per-chunk bench timings keep `total_s`
  and `tts_rtf` truthful (attributing a chunk's wall to each of its sentences would multiply both by
  the batch size);
- the shared diarize post-processing and the NeMo return parser, including the one genuinely
  ambiguous case (a 3-element sequence is both a valid segment and a valid group of three).

229 tests green. What remains unverified is throughput: whether batching delivers on the 3090, and
at what `max_batch`. First-run expectations, and what to check, are in
`docs/setup/windows-wsl2-cuda.md` Phase 9.

## Revisit

- Upstream flags flash-attention support (a TTS speedup the candidate can then take for free).
- `max_batch` is tuned against the 3090. 8 is a conservative guess from the memory math, not a
  measurement; the ceiling should be compute, not memory, since the KV cache is small (GQA, 8 KV
  heads, 144 KiB/token).
- NeMo's `diarize()` return shape turns out to be something the tolerant parser rejects — tighten it
  to the real shape and drop the tolerance rather than widening it further.
- The 3090 baseline comes in far from the ~0.35 batch / ~0.51 watch-along projection — the
  projection scales TTS/ASR decode by memory bandwidth and holds cloud translate fixed, so a large
  miss means the model of the bottleneck is wrong, not just the constant.
