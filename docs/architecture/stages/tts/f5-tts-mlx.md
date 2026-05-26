# F5-TTS-MLX (tts)

- **Repo / version:** `lucasnewman/f5-tts-mlx` (MLX port of `SWivid/F5-TTS`)
- **Parameters / size / runtime memory:** 335M params; ~1.5 GB checkpoint; ~3 GB MPS minimum, ~6-8 GB on longer paragraphs
- **Quantization available:** fp16 / bf16; no GGUF; no INT8 in official
- **Apple Silicon (MLX):** ✅ native; **first-class MLX port** (~11 s/sample on M3 Max)
- **CUDA / CPU:** original PyTorch repo supports both
- **Languages:** **English + Chinese (base only).** Community fine-tunes exist for Arabic, Finnish, Spanish, French, German. **No Eastern Armenian fine-tune published.**
- **License / gating:** CC-BY-NC; no HF token needed
- **Integration surface (hypothetical):** `f5-tts-mlx` package; zero-shot voice cloning via 1-5 s reference clip; speaker embedding extracted by a reference encoder, fused with text + mel features in a DiT (Diffusion Transformer) backbone; flow-matching (non-autoregressive).
- **Status in pipeline:** **rejected (2026-05-26)** on Eastern-Armenian gap
- **Last evaluated:** 2026-05-26 (web-research investigation only; not actually run)
- **Our experience:** Investigation only. The MLX port + small footprint + non-autoregressive memory profile make F5-TTS attractive deployment-wise. The language gap is the blocker — feeding Armenian text to a model that has never seen Armenian phonemes produces garbled output (well-documented in the F5-TTS community discussions for other unsupported languages).
- **Notes:**
  - Emotion handling: "partially carries" emotion from the reference clip. No explicit emotion knob. Strictly worse than IndexTTS-2 on emotion control, strictly better on deployment.
  - Path to use F5-TTS for EA: build an EA tokenizer + collect ~50-100 h of EA speech + fine-tune. **A project of its own; not a TTS-swap.**

## Why rejected

See the [OmniVoice locked for EA](../../decisions/2026-05-26-omnivoice-locked-for-ea.md) ADR. Short version: no published EA support; fine-tuning is out of scope for the emotion-transfer spec.

## Revisit trigger

A published EA fine-tune of F5-TTS-MLX appears on HuggingFace / similar.
