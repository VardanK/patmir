# Active tts config

```toml
[pipeline]
tts = "omnivoice_mlx_q8"

[[eval.tts.candidates]]
id            = "omnivoice_mlx_q8"
enabled       = true
module        = "patmir.eval.candidates.tts.omnivoice_mlx"
repo_id       = "lightsofapollo/omnivoice-mlx-q8-g64"
ref_audio     = "eval_runs/20260524-160019/tts/piper_hye/hy-cartoon-01.wav"
ref_text      = "Արկածի գնանք։"
num_steps     = 32
guidance_scale = 2.0
language      = "None"
lang_code     = "None"
```

MLX-native int8 quantized OmniVoice — `~1 GB` on disk, `~25%` faster RTF on Apple Silicon than the fp16 PyTorch variant, no GPU API key needed. Local-first compliant.

`ElevenLabs Multilingual v2` (`eleven_v3` with `language_code="hy"`) stays registered as the cloud alternate — flip `[pipeline].tts = "elevenlabs_multilingual_v2"` to revert. The in-code per-speaker fallback-to-default-voice retry on HTTP 403 `voice_access_denied` / `detected_captcha_voice` stays in `patmir/eval/candidates/tts/elevenlabs.py` so the ElevenLabs path is still usable for content the captcha system doesn't flag.

OpenRouter Gemini TTS (`openrouter_gemini_flash_tts`) is a second cloud alternate. Piper `hy_AM-gor-medium` is the **no-ref** fallback for any speaker without a usable reference clip.

## Phase 2.x emotion + voice routing

OmniVoice has **no explicit emotion knob**. Emotion transfer is routed through:

1. **Per-`(speaker, emotion_bin)` ref clip selection** — `diarize/refs/<speaker>/<emotion>.wav` (sliced post-emotion by `ref_picker.pick_refs`). The TTS adapter looks up `refs[(speaker, emotion)]` per sentence; falls back to `refs[(speaker, "neutral")]`, then the global default ref, then `piper_hye`.
2. **Per-emotion `guidance_scale`** — pulled from `[emotion_guidance]` in `patmir.toml` and passed per call. Higher scale = more of the ref clip's prosody bleeds into the synth.
3. **Reference clip determines tone** — emotion fidelity is bounded by what's audible in the source character's range; a flat character can't be made angry. Accepted constraint (ADR `2026-05-26-emotion-routed-through-ref-clip-binning`).

## Adapter call passes

`ref_audio` (per-sentence ref clip), `ref_text` (transcript of that ref clip, filled by `backfill_ref_text` from ASR — OmniVoice **requires** this), `emotion` (drives the guidance_scale lookup), `guidance_scale` (resolved per-call from `[emotion_guidance]`).

Refs with empty `ref_text` are not usable by OmniVoice — the adapter falls back to the global default ref for those sentences (a known limitation logged as a TODO; see the audit notes in the closed `quality-issues.md` #1 entry).

## Fallback chain

- `refs[(speaker, emotion)]` → `refs[(speaker, "neutral")]` → global default ref clip (`ref_audio` in the candidate config) → `piper_hye`.
- Per-sentence synth timeout no longer kills the run; on failure the sentence becomes a silent WAV and the rest of the pipeline continues.

## Notes

- MLX requires Apple Silicon; do not run this candidate on x86. The fp16 PyTorch `omnivoice` candidate is the cross-platform alternative (heavier).
- ElevenLabs in-code 403 fallback stays in place so the cloud path remains usable for non-celebrity content; nothing depends on it being the default.
