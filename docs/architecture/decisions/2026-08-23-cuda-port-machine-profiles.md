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
  the ability to run a CPU or MLX set on the CUDA box deliberately.

### What "unknown profile" should do — corrected 2026-08-23

The first version of this rule was "an unknown profile name is a hard error, never a fall back to the
defaults; on a CUDA box falling back means quietly loading MLX-only candidates." That reasoning is
sound and the rule as stated was still wrong, in two ways found by running it.

**It covered the wrong failure.** A *typo'd* name raised; an *omitted* one did not, because there is
nothing to validate when the variable is unset — resolution just falls through to `[pipeline]`, which
off Apple Silicon selects candidates that cannot load. The symptom was a bare
`libmlx.so: cannot open shared object file`. Fixed downstream instead: `_build_failure_message` now
names the stage, the candidate and the module on any build failure, and points at `$PATMIR_PROFILE`
when an MLX candidate fails with no profile active. That is the real safety net, and it works for
both the typo and the omission.

**It was too aggressive.** `$PATMIR_PROFILE` is machine-scoped and exported from a shell profile, so
it applies to every config the process loads — including small ones with no `[pipeline.profiles]`
table. Hard-failing those is wrong; the variable is not addressed to them. It broke the test suite
outright on the workstation: `test_paths.py` builds minimal configs, so with the variable exported
five of its tests failed on profile resolution rather than on what they test. Green on the laptop,
red on the workstation, same commit.

The rule now keys on provenance:

| Where the name came from | Config defines profiles | Behaviour |
| --- | --- | --- |
| `--profile` / `profile=` | any | raise — the caller named something specific that does not exist |
| `$PATMIR_PROFILE` | yes, but not this one | raise — a real mismatch (typo, or a config needing the profile added) |
| `$PATMIR_PROFILE` | none at all | ignore, use `[pipeline]` |

Nothing is lost by the third row: a profile-less config selecting MLX on a CUDA box still fails, just
at build time with a message that explains itself.

A `conftest.py` now clears `$PATMIR_PROFILE` for the whole test suite as well. Tests that need a
profile set it explicitly, so the suite no longer reads the developer's shell.

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
| asr | `qwen3_asr_1_7b` | **changed twice on the day.** The `qwen-asr` package's exact `transformers==4.57.6` pin forced an interim move to `faster_whisper_large_v3_fp16`; transformers 5.13's native `qwen3_asr` then removed the conflict and restored parity. See Decision 4 and its amendment |
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
  whether an element is a string, a tuple or an object was unspecified — as was whether the result is
  nested per input file. The parser accepts every plausible shape and **raises** on anything else,
  with the offending value in the message. Guessing here would have been the worst option available:
  a silently mis-parsed diarization does not fail, it resurfaces several stages later as two
  characters sharing a voice.

  **Resolved on first run (2026-08-23, RTX 3090):**

  ```
  [['0.000 1.840 speaker_0', '2.000 3.680 speaker_1', '3.840 5.280 speaker_2']]
  ```

  A list per input file, of **whitespace**-separated strings — *not* the comma-separated form the
  prose implies — with `speaker_N` labels rather than bare integers. The tolerance was not wasted:
  had the parser been written to the documented prose it would have split on commas, found one field,
  and raised on the very first call. The tuple and object branches are now insurance against a NeMo
  change rather than live unknowns, and the observed shape is pinned in
  `test_diarize_sortformer.ObservedNeMoShapeTest`.

  The first real call also landed precisely on the one genuinely ambiguous case the parser was
  written for — a nested list of *three* string segments, which is structurally indistinguishable
  from a single 3-element segment. Disambiguating on whether the first two elements parse as times
  was load-bearing on call one, not a hypothetical.

`pyannote_3_1` stays registered as the fallback, because NeMo is the heaviest dependency in the stack
(built from git, wants Cython first) and is the most likely thing to refuse to install.

## Decision 4 — ASR yields to TTS in a hard dependency conflict

The profile originally selected `qwen3_asr_1_7b`, the torch sibling of the Mac's active ASR, which
looked like a free config flip. It is not installable alongside the TTS:

```
qwen-asr>=0.0.6 depends on transformers==4.57.6
patmir[local] depends on transformers>=5.5
```

`qwen-asr` pins transformers **exactly**, and the Higgs checkpoint requires >=5.5. One environment
cannot hold both. Discovered the expensive way: installing `nemo_toolkit` pulled transformers down to
4.57 (nemo declares no constraint — our own `>=4.40` floor simply permitted it), which broke a TTS
stage that had already worked on that machine.

**ASR yields.** TTS is ~69% of batch wall and ASR ~11%, so the stage with the smaller share and a
viable alternative is the one to move. `faster_whisper_large_v3_fp16` is Whisper large-v3 through
CTranslate2, which has no transformers dependency at all — that is what makes it a clean exit rather
than another version negotiation. fp16 rather than the existing int8 entry, which exists only because
CT2 has no MPS backend and falls back to CPU on the Mac.

Two consequences:

- **The ASR stage now differs between machines**, so its per-stage timing is not comparable across
  the two baselines. Aggregate RTF still is, roughly, but not the ASR line. Whisper large-v3 is at
  least well characterized here — it was the Mac default until 2026-07-14: clean on fast cartoon
  speech, prone to hallucinated repeats on singing.
- **There is a path back to parity.** transformers gained native Qwen3-ASR support in 5.13, so a
  candidate written against transformers instead of the `qwen-asr` package would sidestep the pin
  entirely. Not taken now: our candidate imports `qwen_asr.Qwen3ASRModel`, so this is new code, and
  the hard part is word timestamps (the MLX path uses a bundled ForcedAligner). Deferred until a
  working baseline exists rather than written blind.

### Amendment (2026-08-23, later the same day) — parity restored, `faster_whisper` demoted

The "path back to parity" above was taken immediately, because the author's judgement was that
Qwen3-ASR is needed for this pipeline rather than nice to have. Everything in Decision 4's diagnosis
holds; only the conclusion changes.

What was verified before writing code, rather than assumed:

- `qwen-asr` 0.0.6 is the latest release, and its PyPI metadata pins `transformers==4.57.6` —
  an exact pin, alongside exact pins on `accelerate`, `nagisa`, `soynlp`, and a `vllm` extra. The
  conflict is real and has no looser version to wait for.
- `src/transformers/models/qwen3_asr` is present in the v5.13.0 tree and absent from v5.12.0, which
  pins the floor precisely instead of trusting the release note.
- The registered classes are `Qwen3ASRForConditionalGeneration` (transcription) and
  `Qwen3ASRForTokenClassification` (the forced aligner). **The aligner is included** — that was the
  open question that had made this "new code with a hard part", and the answer is that transformers
  ships it, exposed as `prepare_forced_aligner_inputs` / `decode_forced_alignment` on the processor.
- `Qwen/Qwen3-ASR-1.7B-hf` and `Qwen/Qwen3-ForcedAligner-0.6B-hf` exist, declare
  `model_type: qwen3_asr`, and carry no remote code. The **`-hf` suffix is load-bearing**: the
  unsuffixed repos are the layout the `qwen-asr` package reads. The Mac's MLX candidate keeps the
  unsuffixed aligner, since `mlx-qwen3-asr` reads that one.

Since 5.13 also satisfies the Higgs checkpoint's `>=5.5`, there is nothing left to trade: the profile
selects `qwen3_asr_1_7b` again, ASR is the same model family on both machines, and the dependency set
*shrinks* — four exact pins plus `gradio`, `flask`, and `sox` never enter. `nagisa` and `soynlp` turn
out to be Japanese and Korean word-splitting only, which this project does not transcribe.

Two things this costs, stated plainly:

- **Long-form segmentation became ours.** The package hid it inside `transcribe()`; the transformers
  API is single-shot. `build_chunks()` cuts on the separate stage's Silero VAD silences, capped at
  `chunk_seconds`, and the chunks run batched through both models. This is the one place the two
  machines' ASR now genuinely differs in behaviour rather than runtime — and plausibly in our favour,
  since the MLX path's coarse internal chunking is what pushed per-sentence TTS p90 to 21.4 s. That
  is motivation, not a measurement.
- **The version floor is now split**: `[cuda]` carries `transformers>=5.13` while `local` stays at
  `>=5.5`, so the Mac is not forced to upgrade for a candidate it does not select. The module gates
  on the version at build time, so selecting it on a stale environment reports the fix instead of
  `KeyError: 'qwen3_asr'` — the failure that made this conflict confusing the first time.

`faster_whisper_large_v3_fp16` stays registered as the no-Qwen fallback. Its virtue is unchanged and
still worth having on record: CTranslate2 has no transformers dependency at all, so it is the escape
hatch if this ever becomes a version fight again.

Unit tests cover the chunker (no chunk exceeds the cap, chunks never overlap, speech that fits is
never cut, over-long intervals hard-split, padding never crosses a neighbour), the version gate, and
language normalization — 32 tests, 284 in the suite. **The models themselves have still never been
executed on CUDA**, and `chunk_seconds=28` / `batch_size=8` are reasoned defaults, not swept.

## Status of verification

Everything in this ADR that is *mechanism* is verified on the Mac: profile resolution and precedence
across all four tiers, the hard failure on an unknown name, distinct cache keys per profile, the
real `cuda` profile selecting no MLX module or `mps` device in either execution mode, and 162 tests
green.

Nothing in it that *touches CUDA* has been executed. `higgs_v3_torch_bf16` was written against the
upstream source and `sortformer_nemo` against the model card, neither against a running GPU;
`htdemucs_vocals` and `qwen3_asr_1_7b` have never run in this pipeline at all — and `qwen3_asr_1_7b` was rewritten against transformers after this section was first written, so its long-form chunking is unexercised too.

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

## First end-to-end attempt (2026-08-24)

The first real `patmir dub` on the 3090 got **eight of nine stages** through: ingest, separate,
diarize (Sortformer on NeMo, confirmed working on transformers 5.15.1 — so the NeMo-vs-5.x question
in Decision 3 is closed and quality issue #13 stays shut), emotion, asr, translate, and the Higgs
model plus codec both loading on the card.

It died entering TTS synthesis, and not on anything this port introduced:

```
Could not load this library: .../site-packages/torchcodec/libtorchcodec_image.so
```

On torch >= 2.11 `torchaudio.load()` decodes nothing itself — it delegates to `torchcodec`, whose
native libraries link against both torch and the system FFmpeg. Both Higgs candidates used it to read
a speaker's reference clip. This is the *second* time this exact delegation has bitten: the separate
stage hit it through Silero's `read_audio` helper and was worked around there with soundfile plus
`np.interp`.

That workaround was not reusable here. Linear interpolation is acceptable for a VAD frontend and not
for the clip the codec clones a character's voice from — it aliases. So reference clips now decode
through **ffmpeg** (`eval/candidates/tts/_audio_io.py`), which is already a hard dependency of the
project, resamples properly, reads whatever container the clip is in, and costs one subprocess per
speaker because the callers cache. Applied to the MLX sibling too: it carries the identical defect
and simply has not fired on the Mac yet. A stereo ref clip is now also correctly downmixed, which the
old path got wrong.

Worth noting what this says about the `audio_head.weight | MISSING` line in the load report — it is
still expected and still not a problem, and it appeared immediately above the real failure. The
candidate asserts the tie at build time precisely so that noise cannot be mistaken for a fault.

The underlying dlopen failure is untouched and remains an environment landmine: anything else in the
stack that reaches for torchcodec will hit it. Most likely an FFmpeg major version torchcodec has no
build for. Not chased, because nothing we run needs it now.

## Revisit

- Upstream flags flash-attention support (a TTS speedup the candidate can then take for free).
- `max_batch` is tuned against the 3090. 8 is a conservative guess from the memory math, not a
  measurement; the ceiling should be compute, not memory, since the KV cache is small (GQA, 8 KV
  heads, 144 KiB/token).
- A NeMo upgrade changes the `diarize()` return shape. It is now pinned by test, so this surfaces as
  a failure rather than as silent mis-parsing; retarget the parser at the new shape then.
- The 3090 baseline comes in far from the ~0.35 batch / ~0.51 watch-along projection — the
  projection scales TTS/ASR decode by memory bandwidth and holds cloud translate fixed, so a large
  miss means the model of the bottleneck is wrong, not just the constant.
