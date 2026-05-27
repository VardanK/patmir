# Prefer fast/nearby model-host mirrors

**Date:** 2026-05-26
**Status:** locked (general policy across all stages)

## Context

While verifying the Phase 2.x emotion-transfer work, `funasr.AutoModel` defaulted to ModelScope to download `iic/emotion2vec_plus_base`. ModelScope is hosted in mainland China and the Pacific peering for US-bound traffic chokes outbound speeds to ~500-700 KB/s. A 1.29 GB bundle took 43 minutes before we caught it.

The model is also mirrored to HuggingFace (`emotion2vec/emotion2vec_plus_base`), which is Cloudflare-fronted and tops out at tens of MB/s from the US. The fix was a one-line change to pass `hub="hf"` and rewrite the model id. Same weights, ~50× faster download.

This wasn't an emotion2vec-specific bug — it's a category of avoidable slow downloads we'll keep hitting whenever we adopt a new model. So it gets a rule, not just a fix.

## Decision

**When adding a new model candidate, prefer the fastest/nearest published mirror.** Order of preference, top to bottom:

1. **HuggingFace** (Cloudflare-fronted, global edge). Default for any model that publishes there.
2. **GitHub releases** (Fastly, similar coverage to Cloudflare). Common for ONNX bundles and small VAD/TTS models.
3. **AWS S3 / GCS / Cloudflare R2** when used directly by the publishing project. Generally fast.
4. **Project-specific CDNs** (e.g. NVIDIA NGC) when fronted by Akamai/Cloudfront.
5. **ModelScope (Alibaba)** — only when the model is *not* mirrored anywhere else, and only with a hard expectation that the first download will be slow + must be cached.
6. **University FTP / homepages / unfronted origin servers** — last resort. Note the slow first run in the model's docs file.

For any model under consideration, the candidate's docs file (`docs/architecture/stages/<stage>/<model>.md`) must record:

- Which hub the candidate downloads from by default.
- Whether a faster mirror exists, and the kwarg / env var / config switch to use it.
- The actually-observed download speed and total size on the author's network, if measured.

For libraries that default to a slow hub (funasr → ModelScope is the canonical example), the candidate's `build()` must override the default and pass the fast mirror explicitly — *don't* rely on environment overrides users would have to know to set.

## Consequences

- New model files added to `docs/architecture/stages/<stage>/<model>.md` must include a "Download source" line in addition to the existing fields.
- When a model can't be found on a fast mirror at all (rare), the docs file says so and the candidate code adds a one-line `progress(...)` warning on first load so users aren't surprised by the wait.
- For models that publish on both ModelScope and HuggingFace (common for Chinese-origin research models — emotion2vec, Qwen, ChatTTS, SenseVoice, CosyVoice, Index-TTS, IndexTTS-2, FishSpeech), the candidate code defaults to the HF mirror and only falls back to ModelScope if HF is unavailable for that specific model.

## Revisit trigger

If a model is published *only* on ModelScope (with no HF / GitHub / S3 mirror) and we need it badly enough to wait, the docs file calls that out explicitly. The general rule still applies — the exception is a measured one-time cost, not a policy change.

## Source

This conversation (2026-05-26 emotion-transfer verification — emotion2vec downloaded from ModelScope at ~600 KB/s for 43 min before being switched to HF mirror at multi-MB/s).
