# emotion2vec_plus_base (emotion)

- **Repo / version:** `iic/emotion2vec_plus_base` on ModelScope; mirrored to HuggingFace as `emotion2vec/emotion2vec_plus_base`. Loaded via `funasr.AutoModel`.
- **Download source:** **HuggingFace** (Cloudflare-fronted, tens of MB/s). funasr defaults to ModelScope which is hosted in China and crawls at ~500-700 KB/s from the US; the candidate hardcodes `hub="hf"` and rewrites the id to the HF mirror. See [decisions/2026-05-26-prefer-fast-model-mirrors.md](../../decisions/2026-05-26-prefer-fast-model-mirrors.md).
- **Parameters / size / runtime memory:** ~90M params, ~360 MB on disk, ~1-2 GB peak during inference
- **Quantization available:** none official (fp32)
- **Apple Silicon (MLX/MPS) / CUDA / CPU:** PyTorch — MPS workable, CUDA full, CPU full
- **Languages:** multilingual training (English, Mandarin primarily; cross-lingual transfer for paralinguistic cues is generally strong)
- **License / gating:** Apache 2.0
- **Integration surface:** `Emotion2VecCandidate.detect(separated, diarization, ctx) -> list[EmotionPrediction]` (Phase 2.x — new). Lazy-loads via `funasr.AutoModel`; per-turn `model.generate(...)` returns class probabilities over 9 emotion labels.
- **Status in pipeline:** **active** (Phase 2.x — landing with `2026-05-26-emotion-transfer-and-quality-docs`)
- **Last evaluated:** 2026-05-26 (initial integration)
- **Our experience:** TBD — to be filled in after end-to-end verification.
- **Notes:**
  - 9 output classes: `angry`, `disgusted`, `fearful`, `happy`, `neutral`, `other`, `sad`, `surprised`, `unknown`. Collapsed to 5 bins via `patmir.toml [emotion_bins]`.
  - Confidence floor at 0.5: below that, the prediction is replaced with `neutral` (avoids confidently-wrong tags on ambiguous/quiet segments).
  - **Bias risk:** the masha clip is Russian-source. emotion2vec is biased toward EN/ZH training. Cross-language transfer for paralinguistic cues is OK in the literature but not Russian-specific. Mitigation = the confidence floor + the always-available `neutral` fallback.
  - **First-run download:** ~500 MB including modelscope's dependency footprint. Documented in setup notes.

## Alternatives we considered but skipped

- `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` — outputs dimensional valence/arousal/dominance instead of categorical emotions; harder to map to ref-clip bins.
- `superb/wav2vec2-base-superb-er` — only 4 classes (ang/hap/neu/sad), fine-tuned on IEMOCAP only. Less expressive than emotion2vec.
- emotion-from-text approaches — rejected per the ADR: emotion is detected from the *source vocal*, not from translated text, because the vocal carries ground-truth paralinguistic cues.

## ADR

[2026-05-26 — Emotion routed through ref-clip binning](../../decisions/2026-05-26-emotion-routed-through-ref-clip-binning.md)
