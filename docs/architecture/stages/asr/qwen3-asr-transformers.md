# Qwen3-ASR via transformers (torch / CUDA)

**Status: active under `[pipeline.profiles.cuda]` since 2026-08-23. Never executed —
written and unit-tested, first CUDA run pending.**

Module: `patmir/pipeline/stages/asr/qwen3_asr.py`
Candidates: `qwen3_asr_1_7b` (active), `qwen3_asr_0_6b`

The same model family as the Mac's active ASR (`qwen3_asr_1_7b_mlx_8bit`), driven
through **transformers' own `qwen3_asr` implementation** rather than a wrapper
package. This keeps the ASR stage comparable across the two machines, which the
`faster_whisper_large_v3_fp16` interim choice did not.

## Why not the `qwen-asr` package

The official package from the Qwen team is the obvious way to run these models on
torch, and it is the one thing this candidate must not use.

`qwen-asr` 0.0.6 (latest as of 2026-08-23) declares:

```
transformers==4.57.6
accelerate==1.12.0
nagisa==0.2.11
soynlp==0.0.493
qwen-omni-utils, librosa, soundfile, sox, gradio, flask, pytz
```

`transformers==4.57.6` is an **exact** pin, and the Higgs Audio v3 TTS checkpoint
requires `>=5.5` (its `tokenizer_config` gives `extra_special_tokens` as a list of
62 entries, which 4.x feeds to `_set_model_specific_special_tokens` expecting a
dict). The two cannot share an environment. Installing `qwen-asr` on the
workstation silently resolved transformers down to 4.57 and broke a TTS stage that
had already worked on that machine.

## Why transformers-native works instead

Transformers **5.13.0** added `models/qwen3_asr` (verified against the v5.12.0 and
v5.13.0 trees — the directory appears in 5.13.0 and not before):

| Class | Auto class | Role |
| --- | --- | --- |
| `Qwen3ASRForConditionalGeneration` | `AutoModelForMultimodalLM` | transcription |
| `Qwen3ASRForTokenClassification` | `AutoModelForTokenClassification` | forced aligner |

5.13 satisfies Higgs' `>=5.5`, so the conflict disappears — there is no package
and no pin to negotiate, and four exact pins plus `gradio`/`flask`/`sox` leave the
dependency set. `nagisa` and `soynlp` are only needed for Japanese and Korean word
splitting, neither of which this project transcribes.

The `[cuda]` extra therefore carries `transformers>=5.13` (higher than `local`'s
`>=5.5`, which the Mac keeps), and `_check_transformers()` gates at build time so a
stale environment reports the fix instead of `KeyError: 'qwen3_asr'`.

## Checkpoints — mind the `-hf` suffix

| Repo | Reads |
| --- | --- |
| `Qwen/Qwen3-ASR-1.7B-hf` | transformers (this candidate) |
| `Qwen/Qwen3-ForcedAligner-0.6B-hf` | transformers (this candidate) |
| `Qwen/Qwen3-ASR-1.7B` | the `qwen-asr` package layout — will **not** load here |
| `Qwen/Qwen3-ForcedAligner-0.6B` | `mlx-qwen3-asr`'s own aligner (the Mac candidate keeps this one) |

Both `-hf` repos are plain transformers checkpoints: `model_type: qwen3_asr`,
`timestamp_token_id: 151705`, a `processor_config.json` carrying the feature
extractor, and **no `trust_remote_code`**.

## Long-form audio — segmentation moved into the candidate

The `qwen-asr` package hid long-form segmentation inside its `transcribe()`. The
transformers API is single-shot, so `build_chunks()` owns it here.

The encoder does handle arbitrary length (windowed; ~13 audio tokens per second,
`processor._get_audio_token_length`), but decoding a whole 5-minute transcript in
one autoregressive pass invites repetition loops and needs a large
`max_new_tokens`. Instead:

1. Cut the audio at **Silero VAD silences** (`ctx.extra["vad_intervals"]`, set by
   the separate stage), grouping consecutive speech into windows capped at
   `chunk_seconds` (default 28.0). A speech interval longer than the cap is
   hard-split. With no VAD available it falls back to fixed windows.
2. Pad each window outward by 0.2 s for word onsets, never past the midpoint of
   the gap to its neighbour — so chunks cannot overlap.
3. Run the chunks **batched** (`batch_size`, default 8) through the ASR, then
   batched again through the aligner. Both support batch inference natively.

Two consequences worth stating: no decode budget is spent on silence, and the
sentence granularity is finer than the MLX path's coarse internal chunking, which
is what pushed per-sentence TTS p90 to 21.4 s and max to 30.4 s (see
`qwen3-asr-mlx.md`). Whether that materializes is UNMEASURED.

## Language handling

Auto-detect runs on the **longest** chunk only, then that language is **forced**
for every chunk. Per-chunk auto-detect mislabels short interjections, which is a
documented source of foreign-script hallucination on this model. An explicit
`ctx.source_lang` skips the probe entirely.

Words failing `word_in_target_script` are dropped — Qwen3-ASR hallucinates CJK
characters and emoji on cartoon sound effects. That guard now lives in
`_sentence.py`, shared with the MLX candidate so the two cannot drift.

## Word timestamps

The aligner is one forward pass (no AR decode) with a head predicting each word's
length, decoded via `prepare_forced_aligner_inputs` → forward →
`decode_forced_alignment` into `{text, start_time, end_time}` per word, relative to
the chunk, offset back to absolute time here.

It covers **fewer languages than the ASR**: zh, en, yue, fr, de, it, ja, ko, pt,
ru, es. EN and RU — the project's source languages — are both in. Outside that set,
or if an optional tokenizer package is missing (ja/ko), the stage degrades to
chunk-level sentence timings rather than failing.

## Knobs

| Knob | Default | Effect |
| --- | --- | --- |
| `model_id` | `Qwen/Qwen3-ASR-1.7B-hf` | ASR checkpoint |
| `forced_aligner` | `Qwen/Qwen3-ForcedAligner-0.6B-hf` | set empty to skip word timestamps |
| `dtype` / `aligner_dtype` | `bfloat16` | both checkpoints are bf16-native |
| `chunk_seconds` | 28.0 | cap per ASR window |
| `batch_size` | 8 | chunks per batch, ASR and aligner alike |
| `max_new_tokens` | 256 | per chunk, not per file |
| `attn_implementation` | `sdpa` | the model card advertises flash-attention too |

## Open

- Never executed on CUDA. `batch_size=8` and `chunk_seconds=28.0` are reasoned
  defaults, not swept.
- `attn_implementation` could move to `flash_attention_2` (the model card
  advertises it, unlike Higgs which declares SDPA only) — untried.
- The finer-chunking claim above is the motivation, not a measurement.
