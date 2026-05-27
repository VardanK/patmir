# References for Cloud ASR migration

## Similar Implementations

### OpenRouter translate candidate (primary pattern)

- **Location:** `patmir/eval/candidates/translation/openrouter.py`
- **Relevance:** Same gateway (OpenRouter), same API key (`OPENROUTER_API_KEY`), same client (`httpx.Client`), same `build(config)` factory shape, same alternates-as-disabled-toml-entries shelving.
- **Key patterns to borrow:**
  - Single persistent `httpx.Client` per instance; `timeout_seconds` configurable.
  - Env-var-driven auth; `RuntimeError` with clear setup instructions if missing.
  - 500-char-truncated error-payload logging on HTTP / response-shape failures.
  - One module, many `[[stages.<stage>.candidates]]` entries differing only by `model` and `id`.

### OpenRouter TTS candidate (audio-endpoint pattern)

- **Location:** `patmir/eval/candidates/tts/openrouter.py`
- **Relevance:** Uses OpenAI-compatible audio sub-endpoint (`/v1/audio/speech`). Our STT candidate will hit `/v1/audio/transcriptions` — same base URL, same auth, similar response-format handling.
- **Key patterns to borrow:** binary-blob handling, response-format branching.

### ElevenLabs TTS shelving (documented-but-not-used pattern)

- **Location:** `patmir/eval/candidates/tts/elevenlabs.py` + `patmir.toml` lines 363–376 (`enabled = false` block with explanatory comment).
- **Relevance:** Same shape we need for the `elevenlabs_scribe_v2` toml stub — register the candidate, mark `enabled = false`, add a comment explaining the tier/policy gate.
- **Key patterns:** comment block above the toml entry naming the blocker and the unlock condition.

### Local Whisper-MLX (the candidate we're displacing)

- **Location:** `patmir/pipeline/stages/asr/whisper_mlx.py` + `docs/architecture/stages/asr/whisper-mlx-large-v3.md`.
- **Relevance:** Shows the `transcribe(audio, ctx) -> Transcript` contract, how word timestamps are flowed through, how sentence grouping is invoked, and how `write_transcript` persists artifacts.
- **What to preserve:** the runner-side post-processing (`apply_speaker_tags`, `apply_emotion_tags`, `_backfill_ref_text`, gap-filler, hallucination blocklist) all assume the same `Transcript` shape — the cloud candidate is a drop-in.

### Local Qwen3-ASR (cautionary tale informing the default-pick decision)

- **Location:** `patmir/pipeline/stages/asr/qwen3_asr.py` + `qwen3_asr_mlx.py`; doc `docs/architecture/stages/asr/qwen3-asr-mlx.md`.
- **Relevance:** Reason we did NOT pick `qwen/qwen3-asr-flash` as the cloud default. The local Qwen3-ASR weights emitted chain-of-thought interleaved with transcription on long inputs (commit `c1b02c5` added reasoning-suppression). Cloud Flash SKU is likely instruction-tuned to suppress this, but the family history pushed us to GPT-4o-transcribe.

### Runner integration (unchanged)

- **Location:** `patmir/runner.py` lines 173–234.
- **Relevance:** ASR invocation block + the three post-ASR passes (speaker tag, emotion tag, ref-text backfill). The cloud candidate must return a well-formed `Transcript` and these passes work without modification.
