# 2026-06-13 — Near-realtime watch-along (D3): spike GO + architecture

## Status

Accepted, 2026-06-13. Reopens the D3 path that `2026-05-27-low-friction-translation-d1.md`
rejected. Throughput spike passed; Slice 2 (streaming runner) + Slice 3 (streaming client)
are now **both built**. Watch-along ships as an **additive mode**, not a replacement — D1
Quality mode stays the default and untouched.

**Slice 3 (#10, streaming client) — DONE 2026-06-13.** `patmir/pipeline/streaming/`
gained `coverage.py` (merged dubbed-interval tracker) + `scheduler.py` (playhead-driven
`StreamScheduler`) + `dub_prelude_window` (reuse prelude analysis for the opening dub).
`patmir/ui/stream.py` (`StreamManager`) runs prelude→scheduler in the background and writes
each section to the per-section stream cache. Server gained `mode` on `POST /jobs` +
`/cache/lookup`, and `/stream/<key>/{sections,section/<i>.wav,playhead}`. `mode` is folded
into the cache key (non-default only → existing quality caches unbusted). Extension gained a
Quality/Watch-along popup toggle + a content-script chunked Web Audio feed (offset-addressed
sections, buffer-gated playback, playhead/seek/pause reporting). Pure logic unit-tested
(coverage/scheduler/cache-mode/stream-cache); full suite 97/97. Live in-browser E2E needs
`.env` (ElevenLabs + OpenRouter keys) + the real source clip.

## Context

The author asked to **watch a video while it dubs** (5–15s buffer) instead of waiting for the
full batch dub. D1 had rejected this on the assumption the pipeline was ~0.64× realtime and
TTS-bound with no fix short of a multi-week rewrite. The author overrode the rejection and
asked to **spike first** to decide the architecture from real numbers.

## What the spike measured

Live config on `runs/20260528-010843-source` (127.3s clip, 33 sentences, 4 speakers). Full
data in `spike/d3_throughput/FINDINGS.md`.

- Pipeline is actually **0.25× realtime** (not 0.64×); **TTS was 74% of wall-clock** — the sole gate.
- **OmniVoice MLX `num_steps` is a near-linear speed lever** (not previously documented):
  32→16→8 steps cut TTS ~3.8×. Only **num_steps=8 (~0.79× video-length) clears realtime**;
  16 steps (1.5×) still drains the buffer. Thread-concurrency is a **dead end** — MLX serializes
  on one GPU stream (conc 2/4 left wall-clock flat).
- **ElevenLabs `eleven_v3` = ~1.57s/sentence at full quality** → ~0.41× video-length, **clears
  realtime even single-threaded**, ~7× faster than OmniVoice-32. Tier **caps at 3 concurrent**
  (429 above). This is the watch-along TTS winner.
- `separate` (Demucs) on a 20s slice = **0.54× realtime** (already internally windowed via
  `split=True`) — chunkable, faster than playback, not a blocker. `diarize`+`asr` (Scribe) =
  0.10× whole-file. No stage other than TTS was ever the constraint.

## Decisions

1. **GO.** Watch-along is feasible. With ElevenLabs TTS the slowest stage becomes `separate`
   at 0.55×, so a pipelined runner produces dub faster than it plays — the buffer fills.
2. **Watch-along is an additive mode.** Selectable in the extension. Quality mode (D1, cached,
   full batch) stays the default. The two modes cache independently (mode in the cache key).
3. **TTS per mode:**
   - Watch-along default = **ElevenLabs eleven_v3**, capped at **3 concurrent** requests.
   - **OmniVoice with a configurable `num_steps`** = the free/offline fallback engine
     (32 = full quality but drains the buffer; 8 = clears realtime at reduced fidelity).
     Author ear-tested: 16 watchable, 32 good; default 32, configurable.
   - Engine is selectable.
4. **ElevenLabs voice strategy = fixed preset pool, NO source cloning.** ElevenLabs intermittently
   blocks IVC uploads as celebrity clones even on non-celebrity content (`voice_access_denied`),
   so watch-along assigns each character a stable voice from a pool of **ElevenLabs prebuilt
   library voices** (curated/own designed voices later). Removes the IVC upload from first-dub
   latency and deletes the captcha-fallback path. **OmniVoice keeps zero-shot source cloning**
   (local, no upload, no captcha). Trade: ElevenLabs characters get distinct consistent voices,
   not the original actor's timbre — acceptable for a kid's cartoon dub.
5. **Architecture = first-chunk prelude + stream.** Analyze the first ~45–60s to establish the
   cast + assign pool voices, start playback (~20–30s to first dub), then stream the rest,
   matching later speakers to the established cast via a global speaker registry. Chosen over
   chunk-everything (marginally lower wait not worth the extra engineering + early mis-assignment
   risk — fixed costs dominate latency, not chunk size) and whole-file prelude (wait scales with
   length; not watch-along for full episodes).

## Consequences / what Slice 2+ must build

- A **streaming/pipelined runner** (today's `runner.py:82-114` is strictly sequential): section N's
  translate→tts→mux overlaps upstream work on N+1; processing continues while paused; seeking
  re-prioritizes the section at the playhead.
- A **global speaker registry** so a character maps to the same voice across sections (Scribe gives
  no cross-call identity). For ElevenLabs: registry → pool-voice assignment. For OmniVoice: registry
  → locked ref clip.
- A **mode toggle** in the extension + a `mode` field on `/jobs` and in the cache key; watch-along
  writes to the D1 cache as it streams so rewatches are free.
- ElevenLabs candidate gains a **preset-pool / no-clone mode** (current code is IVC-only).

## Slice 2 implementation decisions (2026-06-13)

Settled while building; recorded here so they're not re-litigated:

- **#6 done** — ElevenLabs `clone=false` preset-pool mode + `speaker_id` plumbed through the TTS adapter
  + `elevenlabs_pool` candidate (disabled) in `patmir.toml`. Fixed a latent MP3→WAV header bug (bogus
  durations that would mis-stretch mux).
- **#7 done** — `patmir/pipeline/streaming/{registry.py,speaker_profile.py}`. Speakers profiled for
  **gender + 4-way age** (child/young/adult/elder) via the audeering `wav2vec2-large-robust-24-ft-age-gender`
  model (MPS, lazy, pitch fallback). **Gender + child detection reliable; age regression skews high → thresholds
  conservative, age treated as approximate.** Pool voices are tagged with an age *range* so 8 voices cover all
  buckets; assignment is gender-matched + age-range-matched + variety-spread; OmniVoice keeps ref-locking.
- **Section cuts (#8/#9) = VAD-snapped windows** — ~30s steady-state target ended at the next Silero-VAD
  silence gap (sentence-safe; reuses VAD from `separate`), hard cap ~90s. Prelude = larger first window (~45–60s).
- **Cross-section speaker identity (#9) = ECAPA-TDNN embeddings** (`speechbrain/spkrec-ecapa-voxceleb`): embed
  each section's local speakers, cosine-match against the registry's stored per-character embeddings; below
  threshold → new character. Voice mapping stays locked once resolved (no mid-video voice flip).
  - **Calibration (validated 2026-06-13):** threshold = **0.12**, NOT the literature ~0.25–0.40. On
    Demucs-separated emotional speech, same-speaker cosine across emotions drops to 0.157 (angry vs neutral)
    while different speakers stay ≤0.06. Each speaker stores **multiple embeddings** (one per emotion clip;
    resolve = best match) + online enrichment from matched sections. See `project_ecapa_identity_calibration`
    memory + `spike/d3_throughput/test_positive_match.py`. Positive (returning speaker matches) and negative
    (new speaker → new voice) paths both validated on the real source.

## 2026-06-14 — Scribe upload downsampled to 16 kHz mono (perf fix)

First watch-along dogfood showed "minutes" to first audio. Per-stage timing on
the real source found the gate was **diarize/ASR (Scribe), not TTS**: ~169s for
a 60s prelude, ~82–110s per 30s section (≈2.8× realtime) — the ASR stage then
reads the shared cache instantly. This contradicted the spike's "Scribe 0.10×
realtime." A size sweep settled it: Scribe runs at ~0.7–1.1× realtime on a
**16 kHz mono** upload (30s→20s, 60s→64s, 120s→110s) but the **44.1 kHz stereo**
Demucs vocal stem the pipeline was sending is ~6× bigger and ~6× slower —
the 120s slice took **658s (5.48× realtime)** as stereo vs **110s (0.92×)** as
16k mono (same 444 vs 451 words). Rate is ~constant across chunk size, so there's
no fixed per-call overhead for bigger chunks to amortize — the file size was the
whole story. The spike's fast number held only because its probe fed light audio.

Fix: `ScribeV2Client` now transcodes the upload to 16 kHz mono before POSTing
(knobs `upload_sample_rate`=16000, `upload_mono`=true; word timestamps are in
seconds so the timeline is unaffected). ~3× faster diarize/ASR for **both** modes
(Quality + watch-along), no quality change (STT models run at 16 kHz internally).
This removes Scribe as the watch-along throughput gate. (The streaming runner is
still sequential per section — pipelining stages across sections remains the
open follow-up for steady-state headroom.)

## 2026-06-15 — Time-to-first-audio (prelude) reduced

After the Scribe fix, the prelude (analyze + dub the opening before playback)
still took ~2–2.5 min. Per-stage timing (60s window): separate 35s, diarize 22s
(was 169s), translate 20–46s, **tts 40s (sequential)**, mux 1.5s. Three changes
cut it toward ~40–50s, all behind the watch-along streaming config:

- **TTS 3-concurrent.** The shared TTS adapter synthesized per-sentence calls
  one at a time. It now dispatches them through a thread pool sized by the
  candidate's `tts_concurrency` (ElevenLabs pool = 3, the tier cap; OmniVoice/MLX
  stays 1 — single GPU stream). Output order preserved. ~3× on the TTS stage.
- **Batched translate.** Watch-along's `translate` override is now
  `openrouter_gemini_flash_batch` (already in the tree); the translate adapter
  auto-uses its `translate_batch`, so the opening's ~10 sentences translate in
  one call instead of ~10 round-trips.
- **Configurable prelude window** (`prelude_window_seconds`, default 60→40):
  smaller window = less analysis + fewer opening sentences to dub before play,
  at the cost of fewer characters established up front (later ones register
  on the fly).

Then dogfood showed the new wall was **`separate` (Demucs) ≈ 34s/section** — the
`htdemucs_ft` variant is a bag of 4 fine-tuned sub-models (~4× the base cost).
Per-section wall ≈ 57s for ~22s of dub (~2.4× realtime), so the buffer drained
and playback hung repeatedly. Two fixes (watch-along streaming config only):
- **separate → `htdemucs_fast`** (base single-model Demucs, ~4× faster). The
  vocal stem in watch-along only feeds Scribe + the mux duck/loudness-match (no
  cloning), so the slight extra bleed is fine. Quality mode keeps `htdemucs_ft`.
- **Section pad 15s → 8s** — every padded second is Demucs compute re-analyzed
  by the next section anyway.

These bring per-section to ~1.2× realtime. Getting reliably **under 1×** on long
videos still needs **pipelining** — overlapping section N+1's Demucs with N's
light stages (diarize/translate/tts) so the bound is the slowest single stage,
not the sum. The runner is still sequential per section; pipelining is the next
structural piece (the standing "warm-model loop" follow-up, now load-bearing).

Also fixed same-speaker overlap in mux (quality issue #8): a hybrid that
compresses a line (pitch-preserving, capped at `voice_max_compress`=1.5×) toward
the gap to the next line and ripples the remainder — pure ripple alone let drift
reach ~11s on dense sections.

## 2026-06-15 — Cost guard (watch-along was dubbing the whole video)

The first build of the scheduler ran to **completion** — it dubbed the entire
video (backfilling every gap "for free rewatch"), and the client's Stop only
tore down local playback without telling the server. So a single watch-along
submit on a 34-min video ran ElevenLabs TTS + per-section Scribe over **all ~68
sections** regardless of how much was watched, and kept going after the tab
closed. Result: ~30K ElevenLabs credits burned in a day. (Watch-along is also
structurally pricier than Quality mode: per-section Scribe ≈ 1 STT call/section
vs Quality's single whole-file upload.)

Guards added:
- **Bounded look-ahead** — the scheduler only dubs up to `lookahead_seconds`
  (120s) past the playhead, then idles; it never races ahead to dub the whole
  video. The cache now holds what you watched (+buffer), not the full film.
- **Abandon timeout** — if no playhead update arrives for `abandon_seconds`
  (45s; the client posts every ~1s while watching), the viewer has left and the
  scheduler stops.
- **Explicit stop** — `POST /stream/<key>/stop`; the content script calls it on
  teardown so dubbing halts immediately on Stop / tab close, not 45s later.

## Revisit triggers

- ElevenLabs cost runs high on full episodes → lean on the OmniVoice-8 local path or cache-and-rewatch.
- A streaming-capable EA local TTS appears that clears realtime at full quality → drop the cloud dependency.
- Early-scene voice consistency proves bad in practice → revisit online speaker matching (the
  chunk-everything matcher) or lengthen the prelude.
- ElevenLabs lifts the 3-concurrent cap (tier change) → raise the TTS concurrency.
