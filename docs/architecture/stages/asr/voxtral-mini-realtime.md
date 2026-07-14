# Voxtral Mini 4B Realtime 2602 (asr)

- **Repo / version:** `mistralai/Voxtral-Mini-4B-Realtime-2602` (2026-02, Apache 2.0); evaluated as `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` via mlx-audio `voxtral_realtime`
- **Languages:** multilingual incl. Russian (verified on masha); truly causal streaming (80ms–2.4s delay, 480ms default)
- **Apple Silicon:** ✅ MLX 4-bit via mlx-audio
- **Status in pipeline:** **evaluated 2026-07-14, not adopted** — quality good, speed insufficient
- **Our experience (2026-07-14, masha section slices):**
  - Transcription quality ≈ Whisper large-v3 on section 1 (even rendered "Бонжорно" cleanly); no Nemotron-style garbling observed.
  - **RTF ~0.64 at 4-bit** (24.3s for a 38s slice, warm) — **slower than Whisper large-v3's ~0.40** in our pipeline, and ~13× slower than Qwen3-ASR-1.7B (0.05). The "realtime" design point is per-token latency for live streaming, not batch throughput over slices.
  - **Input gotcha:** mlx-audio's `generate()` produced only `"."` on our 44.1 kHz stereo `vocals.wav` stems; works only on **16 kHz mono** input (ref clips + resampled stems). Any future integration must resample first.
- **Verdict:** its causal-streaming property only pays off if watch-along moves to true sample-streaming ASR (not section slices). For the current section architecture, Qwen3-ASR-1.7B dominates it on speed at comparable quality. Revisit if D3 moves to continuous streaming ASR.
