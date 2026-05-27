# OpenRouter TTS — Gemini 3.1 Flash (tts — cloud)

- **Repo / version:** OpenRouter `/v1/audio/speech` endpoint; model `google/gemini-3.1-flash-tts-preview`
- **Apple Silicon / CUDA:** n/a (cloud)
- **Languages:** 70+ — explicitly includes Armenian. Dialect ear-verified on the masha clip ("much better than what we had" — user, 2026-05-26).
- **License / gating:** OpenRouter ToS + Google's Gemini ToS via OpenRouter routing. Pay-per-token from a $5 OpenRouter minimum top-up. No tier wall.
- **Download source:** n/a (cloud)
- **API key:** reuses the existing `OPENROUTER_API_KEY` env var (the same one used by the translate stage). Loaded from `.env` at CLI startup by `patmir/env.py`.
- **Integration surface:** `patmir/eval/candidates/tts/openrouter.py`
  ```python
  synthesize(
      text: str,
      *,
      ref_audio: Path | str | None = None,   # extracts speaker_id from parent dir; drives voice-rotation cache
      ref_text:  str | None = None,           # ignored
      guidance_scale: float | None = None,    # ignored
      emotion: str | None = None,             # drives the inline audio tag prefix
  ) -> bytes  # WAV (PCM-16 mono 24 kHz, wrapped from raw PCM by the candidate)
  ```
- **Status in pipeline:** **active** since 2026-05-26
- **Pricing (per OpenRouter 2026-05-26):** $1 in / $20 out per 1M tokens. ~$0.05-0.10 per 5-min clip; ~$0.40-0.80 per 20-min episode.

## Inline emotion tags

Gemini's TTS reads bracketed prosody directives at the start of the input and applies them to the following speech. The candidate maps `s.emotion` → tag and prepends it:

| `s.emotion` | Tag injected before the Armenian text |
| ----------- | ------------------------------------- |
| `neutral`   | *(no tag)*                            |
| `happy`     | `[happy]`                             |
| `sad`       | `[sad]`                               |
| `angry`     | `[angry]`                             |
| `fearful`   | `[whispers]`                          |

Override via `patmir.toml [[eval.tts.candidates]].emotion_tags`. Per Google, ~200 audio tags are supported (`[laughs]`, `[excited]`, `[shouting]`, `[sighs]`, etc.) — feel free to map alternates if a different word elicits a cleaner response for a given emotion.

## Per-speaker voice rotation

Gemini exposes a set of prebuilt voices: `Kore`, `Puck`, `Aoede`, `Charon`, `Leda`, `Orus`, plus more (Zephyr, Erinome, Algenib, etc.). The candidate maintains a `(speaker_id → voice)` cache:

1. Speakers are assigned voices from `voices = [...]` in `patmir.toml` (rotation pool) in encounter order.
2. Once a `(speaker_id → voice)` binding is established, it persists for the rest of the run — consistent character voice across emotions.
3. `voice_map = { "SPEAKER_00": "Kore", ... }` explicitly overrides specific speakers.
4. If `voices` is empty, every speaker uses the single `voice` default (regression to single voice).

Example default in `patmir.toml`:
```toml
voices = ["Kore", "Puck", "Aoede", "Charon", "Leda", "Orus"]
```

For the masha 4-speaker clip: SPEAKER_00 → Kore, SPEAKER_01 → Puck, SPEAKER_02 → Aoede, SPEAKER_03 → Charon (mix of female and male). The voices are *not* cloned from the source — Masha's actual voice isn't preserved. Accepted because emotion delivery matters more than identity match for the target use case.

## Output format

Gemini's TTS only accepts `response_format = "pcm"` (rejects `mp3` with HTTP 400). The candidate wraps the raw PCM bytes in a WAV-44.1k... no, wait — PCM at 24 kHz mono, wrapped in a WAV header at 24 kHz. (For other OpenRouter TTS models that accept MP3, the candidate decodes via ffmpeg subprocess.)

## Related candidates on the same module

`patmir/eval/candidates/tts/openrouter.py` also powers:

- `openrouter_gpt4o_mini_tts` — `openai/gpt-4o-mini-tts-2025-12-15`, cheaper ($0.60 / 1M chars), 6 OpenAI voices, no audio tags. Disabled by default.
- `openrouter_voxtral_mini_tts` — `mistralai/voxtral-mini-tts-2603`, zero-shot voice cloning. Disabled by default.

Flip `enabled = true` and change `[pipeline].tts` to swap.

## ADR

[2026-05-26 — OpenRouter TTS via Gemini 3.1 Flash (active TTS)](../../decisions/2026-05-26-openrouter-tts-cloud-opt-in.md)
