# Tech Stack

Patmir is a local-first ML pipeline running on a single workstation with a GPU. The stack is chosen to (a) make every pipeline stage independently swappable, (b) prefer mature local-model tooling, and (c) keep the orchestration boring so that model evaluation — not plumbing — is where time is spent.

## Orchestration & Language

- **Python 3.11+** as the primary language. Every candidate model in this space (Whisper, NLLB, MMS, XTTS, Coqui, Bark, etc.) ships with first-class Python bindings; any other choice would be net-negative.
- **Pipeline framework:** plain Python with a typed `Stage` protocol per logical step (ingest, ASR, translate, TTS, mux, output). Avoid heavy workflow engines until needed; a clean module boundary is enough for v1.
- **Configuration:** per-stage config (model name, device, precision, optional cloud fallback) loaded from a single `patmir.toml` so model swaps are config changes, not code changes.
- **Dependency management:** `uv` or `poetry` (whichever the author prefers — `uv` recommended for speed). Single project, no monorepo.

## GPU / Acceleration

- **CUDA** is the assumed acceleration backend on the workstation. Track VRAM budget per stage so the full pipeline fits and concurrent stages can be scheduled without OOM.
- **`torch`** as the model runtime baseline (every candidate ships PyTorch weights).
- **`ctranslate2`** or vendor-specific optimized runtimes where they offer large speedups (e.g. `faster-whisper` for ASR).

## Candidate Models per Stage

Each stage lists current best-known local candidates plus a cloud fallback option. **Final selection is a Phase 0 decision**, not a Phase 1 decision — the table below is the starting search space.

### Stage 1 — Audio Ingest / Extraction
- **`ffmpeg`** (via `ffmpeg-python` or subprocess) — extraction, remux, time-stretch. Mature, no ML model needed.
- **`yt-dlp`** for YouTube and stream sources (Phase 2).

### Stage 2 — Source-Language ASR (with timestamps + diarization)
- **`faster-whisper`** (`large-v3`) — best quality/speed tradeoff, word-level timestamps. ~6–10 GB VRAM in fp16.
- **`whisperx`** — wraps Whisper with better timestamps and optional `pyannote` diarization. Good Phase 2 candidate.
- **`pyannote.audio`** for diarization specifically (Phase 2).
- Cloud fallback: AssemblyAI / Deepgram. Generally unnecessary — Whisper is genuinely state-of-the-art locally.

### Stage 3 — Translation → Eastern Armenian (HIGHEST RISK STAGE)
Eastern Armenian (`hy_AM` / `hye_Armn`) is lower-resource for most translation models; expect quality variance. Translation models usually do not distinguish Eastern vs Western Armenian in their training labels — the dialect of the output depends on training-data composition and must be judged by ear during Phase 0.
- **Local options to evaluate in Phase 0:**
  - **NLLB-200** (`facebook/nllb-200-3.3B` or distilled) — explicitly multilingual, includes Armenian as `hye_Armn`. Likely the local baseline.
  - **MADLAD-400** — newer multilingual, includes Armenian.
  - General-purpose local LLMs (e.g. `Llama-3.x-70B`, `Qwen2.5-72B`, `gemma-2-27b`) prompted for translation — surprisingly competitive for higher-resource pairs, less proven for Eastern Armenian.
- **Cloud fallback:** GPT-4.x / Claude / Gemini. Per the local-first policy (see `mission.md`), cloud is **opt-in per-stage only**, brought in if the local set above is judged inadequate. Translation is the stage most likely to need cloud, because it benefits most from frontier-scale models and is the smallest data leak.
- Plan to evaluate **both** English→Eastern-Armenian and Russian→Eastern-Armenian, since Russian-source content is common and may translate better via Russian as a pivot.

### Stage 4 — Eastern-Armenian TTS (HIGHEST RISK STAGE)
The bottleneck for whether this product is worth building. The local Eastern-Armenian TTS ecosystem is *extremely* thin — at time of writing there is essentially one community model, and several models that look like candidates at first glance are actually Western Armenian (the wrong dialect, see `mission.md`).
- **Local candidates to evaluate in Phase 0:**
  - **Piper community voice `davit312/piper-TTS-Armenian` (`hy_AM-gor-medium`)** — Eastern Armenian. ONNX, runs on CPU, fast. *This is the de-facto local baseline for Eastern Armenian.* Quality unproven and must be judged by ear; if it fails the "watchable for hours?" bar, there is no obvious local alternative.
  - **Coqui XTTS-v2** — multilingual, supports voice cloning. Lists Armenian but does not distinguish Eastern vs Western and the model card is silent on dialect. Treat as unknown-dialect until verified by ear. GPU pass only.
  - **Bark** — multilingual generative TTS. Armenian dialect handling unknown; prosody can be uneven. GPU pass only.
- **Explicitly not viable for v1:**
  - **Meta MMS-TTS** — does not publish an Eastern-Armenian voice. `facebook/mms-tts-hye` returns 404 on HuggingFace; only `facebook/mms-tts-hyw` (Western Armenian) exists. Keep on the radar in case Meta adds `hye` later, but do not plan around it.
- **Voice variety (Phase 2):** *Preset-pool first.* Before reaching for voice cloning, assemble a small fixed cast of 2–4 Eastern-Armenian preset voices — additional community Piper voices if they appear, a Piper fine-tune trained on a different speaker, or one-off recordings via friends/family. A small cast (mom voice / dad voice / kid voice) delivers roughly 80% of the multi-character experience without any cloning gamble, and the produced dialect is guaranteed Eastern.
- **Voice cloning (Phase 2 stretch, optional):** Only attempted once the preset pool feels limiting. Local first: **XTTS-v2** is the only credible zero-shot local cloner that even *claims* Armenian — cloning mechanism is speaker-embedding-based and will produce *something*, but XTTS doesn't distinguish Eastern vs Western in its language tags and dialect quality is undocumented. Must be judged by ear. Cross-lingual cloning via F5-TTS, OpenVoice, Sesame CSM, etc. is essentially undefined behavior for Armenian and is not worth investing in. **Cloud cloning fallback (last resort):** ElevenLabs Instant Voice Cloning + Multilingual v2 — supports Armenian text, decouples timbre from language, dialect must still be ear-verified. Azure / Google Custom Voice are enterprise-tier and not realistic for a personal workstation product.
- **Cloud fallback for TTS (when local fails the bar):** **ElevenLabs Multilingual v2** is the primary cloud candidate — it includes Armenian and supports prosody/emotion/pause control via SSML-ish prompt syntax. Azure (`hy-AM-AnahitNeural` / `hy-AM-HaykNeural`) is a workable secondary. Google TTS (`hy-AM-Standard-A`) is roughly Piper-quality and not a meaningful upgrade — keep on the radar but not the primary cloud option. Per the local-first policy, cloud is **opt-in per-stage only**, brought in if the local Piper voice (and any GPU-pass candidates) is judged inadequate.
- **Hard requirement:** subjective listening eval by a fluent Eastern-Armenian speaker (and the target child) **before** committing to a TTS choice for Phase 1.

### Stage 5 — Audio Mux / Sync
- **`ffmpeg`** for remuxing and basic time alignment.
- **`pyrubberband`** or **`soundstretch`** for high-quality time-stretching when synthesized Armenian audio is longer/shorter than the source segment.
- Sentence-level alignment only; no lip sync.

### Stage 6 — Output
- **Mode A:** MKV container (supports multiple audio tracks + subtitles cleanly), Armenian as default audio, original audio retained, optional Armenian SRT.
- **Mode B (Phase 3):** local HTTP audio stream consumable by `mpv` or a small web player. Out of scope until Phase 3 is greenlit.

## Local UI (v1)

- **CLI** as the source of truth — every pipeline stage runnable independently, intermediate artifacts written to a per-job working directory.
- **Lightweight local web UI** (FastAPI + a minimal HTMX or vanilla-JS front end) for queue + per-stage progress + previewing intermediate artifacts. No auth, no deployment — it binds to `localhost` only.
- Explicitly **not** a polished consumer UI.

## Hardware Sizing (target workstation)

To be confirmed during Phase 0, but the initial assumption is:

- **GPU:** single consumer GPU with **≥ 24 GB VRAM** (e.g. RTX 4090 / 5090, or an A6000-class card). 24 GB comfortably fits Whisper-large + an NLLB-3.3B + a TTS model with headroom.
- **System RAM:** 64 GB recommended (32 GB workable).
- **Disk:** 500 GB+ SSD for model weights + working files. Model weights alone will run 30–80 GB depending on what's pinned.
- Phase 0 must produce **measured VRAM/RAM/disk numbers per chosen model** so hardware can be sized confidently before any Phase 1 build-out.

## What's Deliberately *Not* in the Stack

- No web framework beyond the minimal local UI.
- No container orchestration, no cloud deployment, no auth, no telemetry.
- No tightly-coupled "end-to-end" model that bundles ASR+translation+TTS — modularity is non-negotiable.
