# Active asr config

```toml
[pipeline]
asr = "nemotron_mlx"

[[stages.asr.candidates]]
id              = "nemotron_mlx"
module          = "patmir.pipeline.stages.asr.nemotron_mlx"
repo_id         = "mlx-community/nemotron-3.5-asr-streaming-0.6b"
language        = "None"   # None/auto = auto-detect (recommended)
vad_filter      = true
```

**Local** (2026-06-15) — NVIDIA Nemotron 3.5 streaming ASR (FastConformer-RNNT,
0.6B) on MLX via `mlx-audio`. Replaced cloud Scribe v2 after ElevenLabs credits
were exhausted (ADR `2026-06-15-local-asr-diarize-via-mlx-audio`). See
`nemotron-mlx.md` for the model file.

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
