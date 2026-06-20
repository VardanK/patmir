# Active tts config

**Batch (default):** `higgs_v3_mlx_q8` — Higgs Audio v3 (4B) via our MLX-q8 port +
PyTorch codec. **Watch-along:** `higgs_v3_mlx_q8_fast` (same q8, `max_retries=1`;
`streaming/runner.py` override) — switched off `omnivoice_mlx_bf16_fast` 2026-06-20
for engine consistency (RTF ~1.10 leans on the pre-buffer; OmniVoice-16 stays as
the registered watch-along fallback). Full Higgs write-up → `higgs-audio-v3.md`;
ADR → `decisions/2026-06-20-higgs-v3-tts-mlx-q8.md`.

```toml
[pipeline]
tts = "higgs_v3_mlx_q8"

[[eval.tts.candidates]]
id           = "higgs_v3_mlx_q8"
module       = "patmir.eval.candidates.tts.higgs_v3_mlx"
repo_id      = "multimodalart/higgs-audio-v3-tts-4b-transformers"
codec_id     = "bosonai/higgs-audio-v2-tokenizer"
quant = "q8"; temperature = 0.8; top_p = 0.95; seed = 0; max_retries = 2; codec_device = "mps"
```

Higgs gives **real inline emotion control** (`<|emotion:..|>`, mapped from the 5
bins in the candidate's `EMOTION_TAGS`) — the adapter passes `emotion`, **not**
`guidance_scale`. Faster than OmniVoice-32 (agg RTF 1.10 vs 1.69) and better
quality. q4 rejected; bf16 = cross-platform fallback. Non-termination guardrail:
temp 0.8 + length-cap + reseed (LOW temp causes overflow, measured).

---

## OmniVoice (watch-along + registered batch fallback)

```toml
[[eval.tts.candidates]]
id            = "omnivoice_mlx_bf16"
enabled       = true
module        = "patmir.eval.candidates.tts.omnivoice_mlx"
repo_id       = "mlx-community/OmniVoice-bfloat16"
ref_audio     = "eval_runs/20260524-160019/tts/piper_hye/hy-cartoon-01.wav"
ref_text      = "Արկածի գնանք։"
num_steps     = 32
guidance_scale = 2.0
language      = "None"
lang_code     = "None"
```

Now the **registered fallback** for both modes (`_fast`, 16 steps, RTF 0.91 <
Higgs 1.10) — flip the `streaming/runner.py` override / `[pipeline].tts` back to
the OmniVoice ids to revert. MLX-native bf16 OmniVoice — **switched from the prior `lightsofapollo/omnivoice-mlx-q8-g64` (q8) on 2026-06-16 after an ear-test.** Ranking (same `hy` line + clean ref): **bf16@32 EXCELLENT > q8@32 (almost excellent) > bf16@16 (good, slight nasal) > q8@16 (good, some inter-word pauses).** bf16 is also faster at every step count in our pipeline — RTF **0.46/0.89 @ 16/32** vs q8 **0.67/1.20** — with a faster load (1.7s vs 6.1s), and loads + voice-clones cleanly through mlx-audio. Active: batch = `omnivoice_mlx_bf16` (32 steps), watch-along = `omnivoice_mlx_bf16_fast` (16 steps). The q8 candidates (`omnivoice_mlx_q8` / `_q8_fast`) stay registered as fallbacks — flip `[pipeline].tts` / the streaming override to revert.

Rejected TTS variants: `mlx-community/OmniVoice-8bit` + `OmniVoice-4bit` (config.json missing the quant block → won't load on mlx-audio; built for the separate `ailuntx/OmniVoice-MLX` runtime), `OmniVoice-bf16` (audio tokenizer not wired → can't clone). The `ailuntx/OmniVoice-MLX` runtime loads + clones but is slower (RTF 0.97 @16 fp16; q4 1.07 @16 / 1.26 @32) — precision is not the speed lever, `num_steps` is. q4@16 was ear-rejected. Piper `hy_AM-gor-medium` was tried for watch-along and rejected (quality too low); it remains the **no-ref** fallback only.

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
