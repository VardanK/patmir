# qwen3-asr-mlx (asr)

- **Repo / version:** experimental Qwen3-based ASR via MLX; **2026-07-14 eval used `mlx-community/Qwen3-ASR-1.7B-8bit` via mlx-audio's `qwen3_asr`** (a different runtime than our Phase-1.x candidate module)
- **Status in pipeline:** **registered, not active.** Alternative ASR architecture (LLM-style decoder). Kept as a candidate; Whisper-MLX is the proven baseline.
- **Last evaluated:** **2026-07-14** (masha watch-along section slices — the fast-cartoon-RU stress test); Phase 1.x exploratory before that
- **Our experience:**
  - **2026-07-14 A/B vs Whisper large-v3 on 3 masha 38s section slices** (`runs/20260714-asr-ab/asr_ab_qwen3.json`): **RTF ~0.05 vs Whisper's ~0.40 (≈8× faster)** — the biggest available D3 ASR lever. Quality: sections 1–2 at or *above* Whisper (caught an "Ого!" Whisper missed; no foreign-script leakage, none of Nemotron's failure mode); section 3 (singing/screaming) degraded ("Анекдай, что мы начали? Не посмеши" vs Whisper's "Ах, не гаджай! Что вы наделали? Как мы посмели!"). Verdict: **not a clean win on masha → Whisper stays**; strong candidate for watch-along where speed matters most. Ear-test/decision pending.
  - **Gotchas (2026-07-14):** `generate()` returns **empty text unless `language="ru"` is passed** (auto-detect silently fails on these stems); output has only segment-level timing — **no word timestamps**, so pipeline integration needs mlx-audio's `qwen3_forced_aligner` companion (its cost is unmeasured; remember the MMS-FA alignment pain, ADR 2026-05-26-cloud-asr-via-openrouter).
  - Phase 1.x: Qwen-style reasoning suppression had to be added (commit `c1b02c5`) because the model would emit chain-of-thought interleaved with transcription on long inputs.
- **Notes:**
  - Adoption spec = new candidate wrapping mlx-audio `qwen3_asr` + forced-align pass + `language` plumbed from config/first-section detection; gate on a full watch-along ear-test.
  - Not part of the Phase 2 verification path.
