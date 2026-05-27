# Active asr config

```toml
[pipeline]
asr = "openrouter_gpt_audio"

[[stages.asr.candidates]]
id              = "openrouter_gpt_audio"
module          = "patmir.pipeline.stages.asr.openrouter_stt"
model           = "openai/gpt-audio"
api_key_env     = "OPENROUTER_API_KEY"
chunk_seconds   = 240.0
timeout_seconds = 240.0
max_tokens      = 4096
temperature     = 0.0
language        = "ru"   # ctx.source_lang overrides when set upstream
```

## How it works

OpenRouter does not expose a dedicated `/v1/audio/transcriptions` endpoint (verified 2026-05-26 — the `transcribe`-SKU model ids that earlier research suggested actually 500 on OpenRouter). Audio-in is supported only via `/v1/chat/completions` with `input_audio` content blocks, base64-encoded WAV. The cloud model returns plain transcript text with no timing.

Word-level timestamps are reconstructed locally by Meta's MMS-FA (Multilingual Massively-Speech Forced Aligner) inside the candidate (`patmir/pipeline/stages/asr/_aligner.py`). One ~1.2 GB ONNX model, cached at `~/.cache/ctc_forced_aligner/mms_fa.onnx`, downloaded on first run. Lazy-loaded; one shared session per process.

Long inputs are chunked at `chunk_seconds` (default 240 s = 4 min). When `ctx.extra["vad_intervals"]` is available from the separate stage, chunk boundaries snap to the nearest silence within ±8 s to avoid splitting a word. Transcripts are concatenated and the aligner runs once against the full vocals file.

## Side-car configuration

- **No `transcribe_slice()` helper.** The runner's gap-filler is skipped — the cloud model is already strong on laughter/scream regions that triggered the local Whisper hallucinations the gap-filler was working around.
- **Hallucination blocklist still applies** — runs at sentence assembly time inside `_sentence.py`. Same module used by the local Whisper candidate.
- **VAD post-filter is not run** on cloud transcripts. The cloud model doesn't hallucinate on silence the way Whisper does on laughter, so the post-filter would be busywork. If a future cloud model regresses on this, re-enable `filter_words_by_vad` inside the candidate.

## Fallbacks

- **Local Whisper-MLX** — `whisper_large_v3_mlx_fp16` stays registered. Flip `[pipeline].asr` if `OPENROUTER_API_KEY` isn't available or the cloud route is degraded.
- **Proportional word fallback** — if MMS-FA returns nothing (model load fails, language unsupported), `openrouter_stt.py` distributes whitespace-split tokens proportionally to character count across the audio duration. Worse for stretch-cap math but keeps the pipeline moving.

## Alternates (disabled, registered for A/B)

| id                                | model                                  | notes                                                            |
| --------------------------------- | -------------------------------------- | ---------------------------------------------------------------- |
| `openrouter_voxtral_small_24b`    | `mistralai/voxtral-small-24b-2507`     | ~$0.36/hr. Russian quality slightly weaker than gpt-audio.       |
| `openrouter_gpt_audio_mini`       | `openai/gpt-audio-mini`                | Cheapest. Tends to safety-refuse short cartoon clips — unreliable. |
| `openrouter_gpt4o_audio_preview`  | `openai/gpt-4o-audio-preview`          | Listed in OR catalog but upstream OpenAI returns model_not_found for our key. Tripwire entry. |
| `elevenlabs_scribe_v2`            | ElevenLabs Scribe v2 (direct, not OR)  | Documented-only; module not implemented. Policy-gated until ear-tested + budget-sanctioned. Native word timestamps + diarization. |
