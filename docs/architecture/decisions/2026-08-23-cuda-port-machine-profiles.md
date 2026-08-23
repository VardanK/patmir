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

## Decision 2 — `higgs_v3_torch_bf16` wraps upstream instead of porting again

The MLX candidate (`eval/candidates/tts/higgs_v3_mlx.py`, 367 lines) exists because torch-MPS
cannot do fast quantized matmul, and the 4B decode is memory-bandwidth-bound: ~69 ms/step at
torch-MPS bf16 versus ~28 ms/step at MLX q8. Quantization was the only lever available, so we ported
the AR graph — weight remap, fused multi-codebook embed/head, delay-pattern sampler, EOC state
machine — by hand.

None of that argument survives on a discrete NVIDIA GPU: bf16 is native, bandwidth is ~4.7x, and
`multimodalart/higgs-audio-v3-tts-4b-transformers` already ships a complete transformers
implementation with a public `generate_speech()`. So the new candidate is a thin wrapper (~250 lines,
most of it commentary) that adds only what the pipeline needs on top:

- **reference-code caching** — upstream re-encodes the reference clip on every call; with one pinned
  ref per speaker that is the same clip re-encoded once per sentence;
- **the emotion-bin to inline `<|emotion:..|>` tag map**, byte-identical to the MLX candidate's, so
  a runtime swap cannot change delivery (asserted by comparing the two dicts in
  `test_config_profiles.RuntimeParityTest`);
- **the non-termination guardrail** — length-proportional token cap plus reseed-and-retry.

The MLX candidate stays registered. This is a second runtime for one model, not a replacement:
the laptop still needs the MLX one.

### Two claims from the planning phase that turned out to be wrong

- **Flash-attention is not available.** The upstream port declares `_supports_sdpa` only — no
  `_supports_flash_attn` — so requesting `flash_attention_2` is refused by transformers. The knob
  exists on the candidate (`attn_implementation`) but defaults to `sdpa` and cannot usefully be
  changed until upstream flags support.
- **Batched decode across sentences is not available either.** `generate_speech()` is batch-1. It
  remains a real optimization on CUDA, but it is upstream work or a fork, not a config change.

Overflow detection is weaker than on MLX and this is a deliberate trade. Our MLX loop knows how many
rows it generated; `generate_speech()` returns only a waveform. The wrapper infers overflow from
output length against the token cap, calibrating frames-to-samples from the first reference encode
(exact for this codec, no magic constant) and skipping detection when a job has no reference clip.
Driving our own decode loop over upstream's private methods would restore the exact signal at the
cost of coupling to internals — not worth it for a net that measured 0 overflows in 122 sentences
across three clips at temperature 0.8.

## Decision 3 — the `cuda` profile's candidate set

| Stage | Candidate | Notes |
| --- | --- | --- |
| separate | `htdemucs_vocals` | **new.** `vocals_only` added to the torch candidate, mirroring the MLX one |
| diarize | `pyannote_3_1` | **interim, a known regression** — see below |
| asr | `qwen3_asr_1_7b` | already registered; bf16 rather than 8-bit |
| tts | `higgs_v3_torch_bf16` | **new** |

`separate` is worth the small port rather than falling back to the full `htdemucs_ft` bag: the stage
is 10.7% of batch wall, and the vocals-only trick is ~3.6x on it. The trick's premise was
re-verified independently while writing this — `htdemucs_ft.yaml` lists sub-models
`[f7e0c4bc, d12395a8, 92cfc3b6, 04573f0d]` against an identity weight matrix, so `04573f0d`
contributes weight 1.0 to source index 3 (vocals) and nothing else. It resolves through torch's
`demucs.pretrained.get_model` directly, needing none of the registry patching the MLX path required.

`diarize` is the uncomfortable one. Sortformer replaced pyannote on 2026-06-15 *because* pyannote
community-1 merged short turns from distinct characters into a single speaker — the exact failure
that drove the original Scribe swap. Selecting `pyannote_3_1` on CUDA reintroduces it. NVIDIA's own
NeMo runs Sortformer natively on CUDA, so the fix is a NeMo candidate; this line is the interim, and
it is tracked as quality issue #13 rather than left to be rediscovered by ear.

## Status of verification

Everything in this ADR that is *mechanism* is verified on the Mac: profile resolution and precedence
across all four tiers, the hard failure on an unknown name, distinct cache keys per profile, the
real `cuda` profile selecting no MLX module or `mps` device in either execution mode, and 162 tests
green.

Nothing in it that *touches CUDA* has been executed. `higgs_v3_torch_bf16` was written against the
upstream source, not against a running GPU; `htdemucs_vocals`, `qwen3_asr_1_7b` and `pyannote_3_1`
have never run in this pipeline at all. First-run expectations, and what to check, are in
`docs/setup/windows-wsl2-cuda.md` Phase 9.

## Revisit

- NeMo Sortformer candidate lands (closes the diarize regression).
- Upstream flags flash-attention support, or batched decode arrives (both are TTS speedups the
  wrapper can then take for free).
- The 3090 baseline comes in far from the ~0.35 batch / ~0.51 watch-along projection — the
  projection scales TTS/ASR decode by memory bandwidth and holds cloud translate fixed, so a large
  miss means the model of the bottleneck is wrong, not just the constant.
