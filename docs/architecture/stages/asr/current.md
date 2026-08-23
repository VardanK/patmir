# Active asr config

**The active candidate differs per machine profile.** ASR is the one stage where
the two machines run different runtimes of (now) the same model family.

```toml
[pipeline]                        # Apple Silicon (default)
asr = "qwen3_asr_1_7b_mlx_8bit"   # mlx-qwen3-asr, 8-bit, MLX ForcedAligner

[pipeline.profiles.cuda]          # RTX 3090 / WSL2
asr = "qwen3_asr_1_7b"            # transformers-native torch, bf16
```

| Profile | Candidate | Runtime | Doc |
| --- | --- | --- | --- |
| default (Mac) | `qwen3_asr_1_7b_mlx_8bit` | `mlx-qwen3-asr` | [qwen3-asr-mlx.md](qwen3-asr-mlx.md) |
| `cuda` | `qwen3_asr_1_7b` | transformers >= 5.13, native `qwen3_asr` | [qwen3-asr-transformers.md](qwen3-asr-transformers.md) |

**Mac:** flipped from `whisper_large_v3_mlx_fp16` on 2026-07-14 — author verdict
"both garble sometimes, pick the fastest". The 2026-08-23 re-baseline puts ASR at
10.8% of batch wall (38.0 s) vs Whisper's 29%. Ear-test still pending
(`runs/20260715-reacher-asr-ab/`).

**CUDA:** `qwen3_asr_1_7b` since 2026-08-23, replacing the same-day interim
`faster_whisper_large_v3_fp16`. The interim existed only because the `qwen-asr`
package pins `transformers==4.57.6`, which the Higgs TTS checkpoint's `>=5.5`
forbids; transformers 5.13 carries `qwen3_asr` natively, so the package is not
needed and the two machines are back on one model family. **Never executed.**

The Whisper-era notes below are retained as history — that candidate stays
registered as a fallback on both profiles.

### Why not Nemotron / turbo (ear-tested 2026-06-19 on masha)

- **Nemotron 3.5** (active 2026-06-15→06-19): clean on slower Russian (Fallout-RU)
  but **garbled fast cartoon dialogue** — Devanagari leakage (`कंसर्ट`/`ना`) +
  mangled words. Fast (~3s/60s) but wrong. Stays registered as a fast alternate.
- **Whisper large-v3-turbo** (same fp16 precision): its **8×-pruned text decoder
  (4 layers vs 32)** produced gibberish and captured only ~half the words (173 vs
  336) on masha — ~1.7× faster but drops ~1/3 of the dialogue → rejected.
- **Whisper large-v3** (chosen): only local option that transcribes masha cleanly.
  Costs ~20s/60s warm and under-covers (~16% of speech; missed dialogue masked
  downstream by `vocals_bed_gain`), but correctness wins (ASR is ~16% of the
  pipeline). Cloud Scribe v2 was the best overall but out of credits. ADR
  `2026-06-15-local-asr-diarize-via-mlx-audio` (revert note appended).

## How it produces the Transcript

- `mlx_audio.stt.load(repo_id).generate(path)` returns an `AlignedResult` with
  `.text` and `.sentences`; each `AlignedSentence` carries `.start/.end` and a
  `.tokens` list of `AlignedToken(.text/.start/.end/.duration)`.
- Tokens are sentencepiece subwords (leading-space word-boundary convention),
  merged back into `Word`s by `_tokens_to_words`.
- **Language is auto-detected** — the result does NOT report the detected
  language, so we infer ISO 639-1 from the transcript script (Cyrillic→ru,
  Armenian→hy, CJK→zh, else en) and set `ctx.source_lang` for translation
  routing. Forcing the wrong locale produces phonetic gibberish, so the default
  is auto-detect; pin a `prompt_dictionary` locale key via `language` only when
  the source language is known.

## Side-car configuration

- **VAD post-filter** via `ctx.extra["vad_intervals"]` (`vad_filter=true`) —
  words outside any VAD interval drop a whole sentence (hallucination guard).
- **Non-lexical filter** (`_sentence.is_non_lexical`) drops pure onomatopoeia /
  letter-run sentences (shared with the whisper candidate).
- **Gap-filler** works — `transcribe_slice` is implemented, so the runner's
  diarization-driven second pass runs.

## Fallbacks (registered, disabled by selector)

- `whisper_large_v3_mlx_fp16` — offline fallback with first-class word
  timestamps. Flip `[pipeline].asr = "whisper_large_v3_mlx_fp16"` if Nemotron's
  quality disappoints (esp. on Russian).
- `elevenlabs_scribe_v2` (cloud, needs credits) and `openrouter_gpt_audio` +
  alternates remain registered.
