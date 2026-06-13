# 2026-06-13 — Near-realtime watch-along (D3): spike GO + architecture

## Status

Accepted, 2026-06-13. Reopens the D3 path that `2026-05-27-low-friction-translation-d1.md`
rejected. Throughput spike passed; Slice 2 (streaming runner) + Slice 3 (streaming client)
are now scheduled. Watch-along ships as an **additive mode**, not a replacement — D1 Quality
mode stays the default and untouched.

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

## Revisit triggers

- ElevenLabs cost runs high on full episodes → lean on the OmniVoice-8 local path or cache-and-rewatch.
- A streaming-capable EA local TTS appears that clears realtime at full quality → drop the cloud dependency.
- Early-scene voice consistency proves bad in practice → revisit online speaker matching (the
  chunk-everything matcher) or lengthen the prelude.
- ElevenLabs lifts the 3-concurrent cap (tier change) → raise the TTS concurrency.
