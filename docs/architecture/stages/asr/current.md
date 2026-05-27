# Active asr config

```toml
[pipeline]
asr = "whisper_large_v3_mlx_fp16"

[[stages.asr.candidates]]
id              = "whisper_large_v3_mlx_fp16"
module          = "patmir.pipeline.stages.asr.whisper_mlx"
repo_id         = "mlx-community/whisper-large-v3-mlx"
temperature     = 0.0
condition_on_previous_text = true
no_speech_threshold        = 0.6
```

Local-first ASR. Reverted to Whisper-MLX on 2026-05-26 (evening) after the cloud `openrouter_gpt_audio` route was judged not materially better than the local baseline on the masha ear-test, and the first-sentence capture sometimes truncated.

## Side-car configuration

- **VAD post-filter** via `ctx.extra["vad_intervals"]` — words outside any VAD interval get dropped (catches Whisper hallucinations on scream/laugh regions).
- **Hallucination blocklist** — phrases like `"Субтитры сделал DimaTorzok"`, `"thanks for watching"`, etc. removed at sentence assembly time.
- **Gap-filler** runs after the primary pass: any turn ≥1.5 s containing only non-lexical words is re-ASR'd at higher temperature with the blocklist applied. Recovers ~10+ dialogue lines per masha episode.
- **Non-lexical filter** (`patmir/pipeline/stages/asr/_sentence.py:is_non_lexical`) drops sentences that are pure onomatopoeia. Patched 2026-05-26 to also catch single-token "letter-runs" like `'Аааааа...'` (Whisper transcribes long screams as one big mega-token, no whitespace; the old multi-token-repetition heuristic missed these and they killed downstream ElevenLabs TTS).

## Cloud alternate (registered, disabled by selector)

`openrouter_gpt_audio` is still registered with all its alternates (`openrouter_voxtral_small_24b`, `openrouter_gpt_audio_mini`, `openrouter_gpt4o_audio_preview`, and the documented-only `elevenlabs_scribe_v2`). Flip `[pipeline].asr = "openrouter_gpt_audio"` to use the cloud route again. See `openrouter-stt.md` for the chat-completions + MMS-FA forced-alignment architecture and `decisions/2026-05-26-cloud-asr-via-openrouter.md` for the pivot history.
