# D3 near-realtime throughput spike — findings (2026-06-13)

Test inputs: `runs/20260528-010843-source` (Fallout clip, 127.3s, 33 sentences, 4 speakers).
Live config: OmniVoice MLX q8 TTS, Demucs htdemucs_ft, ElevenLabs Scribe v2.

## The gate question

Watch-along needs the pipeline to produce dub *faster than it plays*. Today it
runs at **0.25× realtime**; TTS is **74%** of wall-clock. Can TTS clear realtime?

## Result: YES, conditional on 8-step quality

### TTS sweep (OmniVoice MLX, 10-sentence subset)

| Config | wall (10) | per-sentence | TTS × video-length |
| ------ | --------- | ------------ | ------------------ |
| 32 steps, conc 1 | 115.5s | 11.55s | ~2.9× (matches 369s baseline) |
| 16 steps, conc 1 | 59.1s | 5.91s | ~1.5× |
| **8 steps, conc 1** | **30.6s** | **3.06s** | **~0.79× (under realtime)** |
| 32 steps, conc 2 | 106.8s | 20.29s | ~2.9× (no gain) |
| 32 steps, conc 4 | 107.7s | 38.08s | ~2.9× (no gain) |

- **`num_steps` is a near-linear speed lever** (not in the original plan). 32→8 cuts TTS ~3.8×.
- **Only 8 steps clears realtime.** 16 steps (1.5×) still drains the buffer.
- **Thread-concurrency is a dead end** — MLX serializes on one GPU stream; conc 2/4 leave wall-clock flat and just inflate per-sentence latency.

### ElevenLabs eleven_v3 (cloud) — the full-quality winner

Cloud calls parallelize (httpx releases GIL), unlike MLX. 8-sentence subset:

| concurrency | wall | per-sentence | notes |
| ----------- | ---- | ------------ | ----- |
| 1 | 12.6s | **1.57s** | clean |
| 4 | (4.0s) | — | 429s — tier caps at **3 concurrent** |
| 8 | (1.8s) | — | 429s (silent-line failures) |

- **1.57s/sentence at full quality** → ~52s for the 33-sentence clip = **0.41× video-length, clears realtime even single-threaded.** ~7× faster than OmniVoice-32.
- **IVC cloning succeeded** on this clip (both speakers `cloned`, no captcha) — the 2026-05-27 captcha block was YouTube-celebrity-specific; cartoons should clone fine.
- Caveats: costs credits per sentence (one-time per video if cached); needs a **3-concurrent cap** to avoid 429→silent-line; IVC upload ~7–16s/speaker one-time during opening buffer.
- Metric bug: `output_audio_s`/RTF were garbage (ffmpeg-pipe WAV header reports bogus nframes); wall + per-sentence are valid.

**Recommendation:** Watch-along TTS = ElevenLabs eleven_v3 (full quality, clears realtime). OmniVoice with configurable `num_steps` = free/offline fallback engine. Engine selectable per the dual-mode design.

### Other stages (no blockers)

- **separate (Demucs):** whole-file 0.55×; 20s-slice probe = **0.54× realtime** → fully chunkable, faster than playback (already uses `split=True` internal windowing).
- **diarize + asr (Scribe):** 0.10× realtime whole-file (single shared upload). Huge headroom.
- emotion 0.07×, translate 0.25×, mux/output ~0.03×.

### Streaming math (pipelined, num_steps=8)

Steady-state throughput = slowest stage = TTS at 0.79× → produces dub at ~1.27×
realtime → **buffer fills faster than it drains.** Feasible.

## The catch (author decision required)

8-step diffusion sounds worse than 32-step. **Watch-along must use a faster/lower-fidelity
TTS config than Quality mode.** Maps cleanly onto the dual-mode design:
- **Quality mode (D1):** num_steps=32, cached, full fidelity.
- **Watch-along mode (D3):** num_steps=8, streamed, lower fidelity.

GO/NO-GO hinges on: **is 8-step Armenian watchable for the son?**
Ear-test samples: `spike/d3_throughput/ear_test/*__steps{08,16,32}.wav` (5 sentences × 3 step-counts).

## Architecture constraint surfaced

A true **5–15s buffer requires chunk-everything** (process sections at the playhead):
a whole-file prelude (separate+diarize+asr ≈ 0.72× realtime) = ~90s upfront on this
127s clip, far past 15s.

But chunk-everything with **per-section Scribe** has a quality blocker: Scribe assigns
`speaker_0/1/2` **per call with no cross-call identity** — same character gets different
labels (and different cloned voices) across section boundaries. Options:
1. **Whole-file diarize prelude** (fast, 0.10×) once upfront → preserves identity, but needs
   whole audio downloaded+separated first → larger initial wait (architecture B-ish).
2. **Online speaker-embedding matching** across sections (local model) → true streaming,
   more engineering.
3. **First-chunk prelude + online matching** hybrid: establish speakers/refs on the first
   ~60-90s, start playback, match later chunks against the established set.

This is the main open design fork for Slice 2.
