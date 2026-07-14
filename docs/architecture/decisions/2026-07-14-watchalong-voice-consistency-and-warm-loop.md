# 2026-07-14 — Watch-along voice consistency (refs remap bug) + warm-model loop + P1 perf wins

## Status

Accepted (code landed 2026-07-14; bench + ear-test verification tracked in quality issue #12).

## Context

A full pipeline review (user-requested, with a SOTA sweep) found that the two
top user complaints — watch-along not being realtime and characters swapping
voices — had *code-level* causes, not model-quality causes:

1. **Voice swapping was a wiring bug.** `_remap_and_trim_transcript`
   (`streaming/section.py`) rewrote transcript `speaker_id`s from
   section-local `SPEAKER_NN` to global registry ids, but `diarize/refs.json`
   kept the local keys. The dub pass's TTS ref lookup — keyed by the
   transcript's global id — matched nothing, so **every watch-along line
   synthesized with `ref_audio=None`** (Higgs' anonymous default voice,
   varying per section). The registry's `lock_ref`/`locked_ref` (the intended
   stable per-character ref) was written but never read in production. Batch
   mode was unaffected (no remap), which is why the bug survived batch
   ear-tests.
2. **The rebuild tax was structural.** Every `run_job` built and freed each
   stage model; `process_section` called it twice per section — a measured
   11–13% of watch-along wall (performance.md).

## Decision

### Voice consistency (quality issue #12)

- **`_remap_refs`** (`streaming/section.py`) rewrites `diarize/refs.json`
  speaker ids/keys local→global in lockstep with the transcript remap. On a
  `(speaker, emotion)` key collision (two locals resolved to one character)
  the longer clip wins.
- **`ctx.extra["locked_refs"]`**: the streaming dub passes now carry
  `SpeakerRegistry.locked_refs()` (path + ref_text), and the TTS adapter
  prefers a speaker's locked ref over per-section clips — a character clones
  from the SAME audio in every section. `lock_ref` now also stores the clip's
  transcript so text-requiring candidates can use locks.
- **Hybrid stand-in pool** (user decision: "hybrid pool"): a speaker with no
  usable ref (or diarization-unattributed speakers with ids that never
  resolve) gets a stable stand-in instead of the anonymous default —
  (1) a clip from the curated `[[tts_ref_pool]]` in patmir.toml,
  gender-matched when the speaker's profile is known (registry profiles are
  now exported via `ctx.extra["speaker_profiles"]`); (2) with no pool
  configured, another character's pinned ref from the same job (two
  characters sharing a voice beats a voice from nowhere — explicitly
  user-approved); (3) only when nothing exists at all, the old default.
  Assignment is stable per speaker and spreads least-used.
- **Ref loudness gate**: `ref_min_rms_db` (default −33 dBFS) in
  `ref_picker.pick_refs` — quiet (far-field/bleed-heavy) windows lose to any
  louder eligible window; if all windows are quiet the longest still wins.

### Performance (performance.md roadmap #1, #2, #5)

- **Warm-model loop (#2):** `run_job(..., stage_cache=)` keeps built stage
  instances warm in a caller-owned dict. The watch-along paths (`ui/stream.py`
  scheduler loop, `streaming/runner.stream_sections`, `bench_watch_along`)
  pass one cache for the whole stream: every stage model is built once, reused
  across sections and across the two per-section `run_job` calls. Batch runs
  pass nothing and keep the one-model-resident-at-a-time behavior. All-stages-
  resident is ~9 GB with the current active set (fine on 32 GB). Warm-cached
  TTS candidates get `reset_summary()` per job so cloning summaries stay
  per-section.
- **Concurrent translate (#1):** the per-sentence translate loop fires
  concurrently when the candidate sets `translate_concurrency`
  (OpenRouterTranslator: 6; httpx.Client is thread-safe; round-trips are
  ~1s TTFT-dominated). Output is byte-identical to the serial loop — no
  ear-test needed. Local candidates (NLLB) stay serial.
- **Pad usage (#5):** the section end now VAD-snaps as far into the padded
  slice as safe (`slice_len − END_GUARD_SECONDS`, guard 1.5s) instead of
  stopping at the 30s target. The 8s pad is analyzed either way; dubbing into
  it makes sections ~20% longer → fewer sections → proportionally less
  re-analyzed overlap. VAD ends inside the guard are distrusted (a mid-speech
  slice cut masquerades as silence).

## Verification

- 118/118 unit tests, including new coverage: `_remap_refs` (key remap,
  collision, no-op), locked-ref precedence + text gating, stand-in pool
  (gender match, stability, spread, reuse-other-speaker, text gating), RMS
  gate (quiet-loses / all-quiet-keeps-longest).
- Live watch-along run mid-bench confirmed: `refs.json` keys now match
  remapped transcript ids (`spk0__sad`), per-section cloning summary resets.
- `patmir bench --mode watch-along` before/after in performance.md.

## P2 measurements (same review, 2026-07-14)

- **ASR (watch-along speed lever): both candidates rejected.**
  `Qwen3-ASR-1.7B-8bit` = RTF **0.05** vs Whisper's 0.40 (≈8×) but
  **author-rejected on quality 2026-07-14** (transcripts read as nonsense to a
  native speaker). `Voxtral Mini 4B Realtime 4bit` = quality ≈ Whisper but RTF
  0.64 (slower than Whisper for section slices; 16 kHz-mono-only input).
  **Whisper large-v3 stays for both modes.** Details:
  `stages/asr/qwen3-asr-mlx.md`, `stages/asr/voxtral-mini-realtime.md`.
- **TTS runtime: no swap.** mlx-audio native Higgs runtime measured — native-q8
  ≈ parity with our port; native-q4 = RTF 0.744 (−29%) without collapse, but
  the **author ear-test 2026-07-14 rejected native-q4 (worst of three) and
  ranked ours-q8 (the active engine) best**. Prefix-KV caching on our port is
  again the TTS speed lever. Details: `stages/tts/higgs-audio-v3.md`.
- **Speaker embeddings:** ReDimNet-b2 (ptn + ft_lm) calibrated against ECAPA
  on the bench ref clips — **both backends' same/different-speaker cosine
  distributions overlap** on short Demucs-separated emotional cartoon clips
  (ECAPA: min-same +0.088 < max-diff +0.143; ReDimNet: −0.075 < +0.112).
  Per-clip matching is marginal regardless of model → the locked-ref +
  multi-embedding + RMS-gate fixes are the load-bearing mitigations. ECAPA
  stays default; `PATMIR_EMBED_BACKEND=redimnet` is an experimental flag
  (threshold 0.02). A proper cast-list pass needs a larger labeled pair set —
  follow-up spec.

## Revisit

- If resident-set memory grows past ~20 GB (bigger models), make the
  stage_cache selective (e.g. cache only asr+tts).
- The stand-in pool for *unattributed* sentences (speaker_id=None) still
  falls to the default voice — revisit with the cast-list pass (P2).
- OpenRouter rate limits: drop `translate_concurrency` if 429s appear.
