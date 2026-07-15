# qwen3-asr-mlx (asr)

- **Repo / version:** experimental Qwen3-based ASR via MLX; **2026-07-14 eval used `mlx-community/Qwen3-ASR-1.7B-8bit` via mlx-audio's `qwen3_asr`** (a different runtime than our Phase-1.x candidate module)
- **Status in pipeline:** **registered, not active.** Alternative ASR architecture (LLM-style decoder). Kept as a candidate; Whisper-MLX is the proven baseline.
- **Last evaluated:** **2026-07-14** (masha watch-along section slices — the fast-cartoon-RU stress test); Phase 1.x exploratory before that
- **Our experience:**
  - **2026-07-14 A/B vs Whisper large-v3 on 3 masha 38s section slices** (`runs/20260714-asr-ab/asr_ab_qwen3.json`): **RTF ~0.05 vs Whisper's ~0.40 (≈8× faster)**, but **author-REJECTED on quality 2026-07-14** — the transcripts read as "a lot of nonsense" to a native speaker (mangled words beyond the singing section, e.g. "Анекдай, что мы начали? Не посмеши" for "Ах, не гаджай! Что вы наделали? Как мы посмели!"). Speed doesn't buy anything if the words are wrong → **Whisper large-v3 stays for both modes**. Reopen only with a newer/larger Qwen3-ASR checkpoint and a fresh masha ear-test.
  - **Gotchas (2026-07-14):** `generate()` returns **empty text unless `language="ru"` is passed** (auto-detect silently fails on these stems); output has only segment-level timing — **no word timestamps**, so pipeline integration needs mlx-audio's `qwen3_forced_aligner` companion (its cost is unmeasured; remember the MMS-FA alignment pain, ADR 2026-05-26-cloud-asr-via-openrouter).
  - Phase 1.x: Qwen-style reasoning suppression had to be added (commit `c1b02c5`) because the model would emit chain-of-thought interleaved with transcription on long inputs.
- **2026-07-15 in-pipeline A/B on Reacher (new default clip), watch-along, warm loop:**
  - **The standalone 8× speed win does NOT survive integration** — the ForcedAligner-0.6B pass (needed because Qwen3 lacks native word timestamps; Whisper has them) eats it. ASR stage: qwen3 118.7s (RTF 0.394) vs whisper 136.3s (0.453); agg RTF 2.10 vs 2.16 — a ~3% total-run edge.
  - **Granularity: 29 sentences vs Whisper's 104** over the same 5 min (~10s run-on chunks vs ~3s sentences) — coarser mux/duck windows, longer single TTS renders, coarser translation units.
  - Transcript quality on Reacher (live-action RU dub) reads clean — proper names intact, no masha-style mangling.
  - Ear-test material: `runs/20260715-reacher-asr-ab/{reacher-dub-qwen3.wav,reacher-dub-whisper.wav,transcripts.md}`.
  - **Flipped active 2026-07-14** (`[pipeline].asr = qwen3_asr_1_7b_mlx_8bit`) per author decision "both garble sometimes → pick the fastest"; note the speed premise weakened to ~3% in-pipeline. Whisper stays registered — one-line revert.
- **2026-07-15 precision A/B (8bit vs bf16, same slices, same runtime+aligner):** transcripts **character-for-character identical** on Reacher (3 slices) and masha (singing section) — same phrasing, same minor errors ("парашюток", "все планах чертам"). bf16 is 2–4× slower (RTF 0.32–0.56 vs 0.10–0.15) and +1.1 GB. **8-bit is strictly better; quality errors are model-level (1.7B), not quantization.** Data: `runs/20260715-reacher-asr-ab/qwen_prec_{8bit,bf16}.json`. If quality must improve, the lever is a bigger checkpoint (no Qwen3-ASR >1.7B exists on mlx-community as of 2026-07-15), not precision.
- **Notes:**
  - If Qwen3 is kept: candidate follow-ups = split run-on chunks at aligner word gaps (recover sentence granularity), and consider skipping the aligner in watch-along (sentence-level timing may suffice for mux ducks) which would restore most of the 8×.
  - Not part of the Phase 2 verification path.
