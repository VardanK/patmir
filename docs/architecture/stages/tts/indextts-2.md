# IndexTTS-2 (tts)

- **Repo / version:** `IndexTeam/IndexTTS-2` (Bilibili, 2026); `IndexTeam/Index-TTS` (original)
- **Parameters / size / runtime memory:** three-module cascade — T2S (Qwen3-finetuned autoregressive LM) + S2M (Zipformer + flow matching) + BigVGANv2 vocoder. **16 GB RAM minimum** per the install guide; 24 GB+ storage recommended.
- **Quantization available:** **none published** — no GGUF, INT8, or quantized variants in the official repo
- **Apple Silicon (MLX/MPS):** **no MLX port.** PyTorch-only; MPS unverified for the autoregressive T2S + Zipformer ops
- **CUDA:** ✅ recommended
- **CPU:** ✅ works but slow
- **Languages:** **Chinese, English, Japanese, Spanish** (IndexTTS 2.5). **No Eastern Armenian fine-tune published.**
- **License / gating:** check the model card; non-commercial use typical
- **Standout feature:** **explicit 8-dim emotion vector** (happy / angry / sad / afraid / disgusted / melancholic / surprised / calm, each 0-1), **independent of speaker identity**. Outperforms SOTA zero-shot on emotional fidelity per arXiv 2506.21619.
- **Voice cloning:** 3-10 s reference clip; reference fed to T2S module as conditioning. Same shape as OmniVoice's interface.
- **Integration surface (hypothetical):** not implemented in Patmir. Would require Python+PyTorch+MPS plumbing, no MLX wrapper.
- **Status in pipeline:** **rejected (2026-05-26)** on Eastern-Armenian gap
- **Last evaluated:** 2026-05-26 (web-research investigation only)
- **Our experience:** Investigation only. IndexTTS-2 is the most attractive emotion-control story in any current open TTS, but the language gap is structural — the model was trained primarily on Chinese + English and lists 4 supported languages, none of them Armenian.
- **Notes:**
  - **Why we didn't try the 8-dim emotion vector anyway:** Armenian text → garbled phonetics regardless of emotion control. The vector is useless without the underlying language coverage.
  - On a 32 GB Mac, the 16 GB RAM minimum is tight alongside the rest of the pipeline (NLLB-3.3B + Whisper + Demucs + pyannote). PyTorch+MPS for an autoregressive LM stage is plausibly slow.

## Why rejected

See the [OmniVoice locked for EA](../../decisions/2026-05-26-omnivoice-locked-for-ea.md) ADR.

## Revisit trigger

An EA fine-tune of IndexTTS-2 is published, OR an EA tokenizer + fine-tune project becomes scoped in (currently out of scope per local-first / spec scope).

## Sources

- arXiv 2506.21619 (IndexTTS2 paper)
- `huggingface.co/IndexTeam/IndexTTS-2` model card
- 2026-05-26 web-research investigation
