# Product Roadmap

The roadmap is gated by a **Phase 0 feasibility check** on Eastern-Armenian (`hy_AM`) translation and Eastern-Armenian TTS quality. Until that gate passes, no work on later phases is justified, because Eastern-Armenian model availability and quality is the single largest technical risk and determines whether the product is worth building at all. Western Armenian (`hyw`) is **not** a substitute: see `mission.md`.

## Phase 0 — Armenian feasibility gate (do this first)

The goal of Phase 0 is to know, with confidence, whether currently available models can produce Armenian audio that is good enough for a child to enjoy. No pipeline plumbing yet.

- Assemble a small eval set: ~20 short utterances spanning child-content register (cartoon dialogue, simple narration, exclamations), formal narration, idioms, and proper nouns. Use both English→Armenian and Russian→Armenian source examples.
- **Translation eval (local first):** Run each utterance through 2–3 local candidate translators. Have a fluent Armenian speaker score: meaning preserved, register appropriate, naturalness. Decide whether the best local option is "good enough." Cloud baselines (GPT, Claude, etc.) are **opt-in only**, brought in per-stage *after* local has been tried and judged inadequate — they are never the default, even as a ceiling reference, in keeping with `mission.md`'s local-first principle.
- **TTS eval (local first):** Take the reference Armenian translations and synthesize them through 2–3 local Armenian TTS engines. Side-by-side listening test with a fluent Armenian speaker (and ideally the target child). Score: intelligibility, naturalness, prosody, "would I let a child watch hours of this?" Cloud TTS is again opt-in only if local fails.
- **Gate decision:** Document the chosen (local-by-default) translator and TTS engine, the quality floor they hit, and whether the product is worth building. If a local stage is clearly inadequate, the explicit next steps are, in order: (1) re-run on the workstation GPU with larger local candidates, (2) opt in to a cloud fallback for that specific stage, (3) if both fail to clear the bar, pause and reassess — wait for the ecosystem to mature, or scope down to subtitles-with-narration.

**Exit criteria for Phase 0:** A written one-pager naming the winning translator + TTS per language pair, with audio samples attached, and an explicit go/no-go on Phase 1.

## Phase 1 — MVP: Mode A on local files (must-have)

End-to-end offline batch translation for local video files. This is the floor of the product — if nothing else ships, this alone is valuable.

- **Pipeline skeleton** with the six logical stages as swappable modules behind clear interfaces: ingest, ASR, translate, TTS, mux, output. Each stage has a typed input/output contract and at least one concrete implementation.
- **Ingest:** local video file → audio track + metadata (via ffmpeg).
- **ASR:** Whisper-family (e.g. `whisper-large-v3` or `faster-whisper`) producing word-level timestamps. Diarization optional in Phase 1 (single-voice TTS is acceptable).
- **Translation:** the winner from Phase 0.
- **TTS:** the winner from Phase 0, single voice.
- **Mux:** align synthesized Armenian audio to sentence-level source timing (time-stretch or pad with silence as needed; rough alignment is fine, lip sync is not a goal).
- **Output:** remuxed video file with Armenian audio as the default track. Original audio retained as a secondary track. Subtitle track (Armenian) as a free byproduct.
- **Minimal local UI:** a small local app or CLI to point at a video file and watch the pipeline run, with per-stage progress and the ability to inspect intermediate artifacts (transcript, translation, generated audio chunks).

**Definition of done for Phase 1:** Author can drop a 20-minute cartoon episode onto the workstation, run the pipeline, and the son finds the resulting Armenian-dubbed file watchable end-to-end.

## Phase 2 — YouTube, multi-speaker, better voices

Once Phase 1 is solid, broaden inputs and improve perceived quality.

- **YouTube + downloadable streams** as ingest sources (via `yt-dlp`).
- **Speaker diarization** in the ASR stage; assign distinct Eastern-Armenian voices per speaker from a small **preset pool** (2–4 voices: mom / dad / kid / narrator). This is the *default* multi-voice path and is sufficient for the target use case — distinct preset voices are not a "fallback" from cloning, they are the primary plan.
- **Voice cloning (stretch, optional)** — only pursued if the preset pool feels limiting in practice. Order of attempts: (1) **XTTS-v2** local — the only credible zero-shot local cloner that claims Armenian, dialect output unverified and must be ear-judged on the GPU pass; (2) **ElevenLabs** Instant Voice Cloning + Multilingual v2 as the cloud last-resort, opt-in per local-first policy. If neither produces Eastern Armenian acceptably, cloning is dropped from Phase 2 — the preset pool is enough.
- **Quality-of-life:** batch queue, resume-from-stage on failure, side-by-side preview (original vs. Armenian).
- Re-run the Phase 0 eval as the model ecosystem shifts; swap in better translators / TTS engines without changing the pipeline shape.

## Phase 3 — Mode B: near-realtime stream translation (stretch)

**Only enter Phase 3 if a feasibility study confirms it can run on the target workstation without cloud-heavy components.** If feasibility fails, drop Mode B permanently and reinvest in Phase 2 quality.

- Buffer the first N seconds of a stream; begin playback once buffer is full.
- Sustained pipeline throughput must exceed playback rate so the buffer never drains.
- Likely requires streaming variants at every stage (streaming ASR, chunked translation, low-latency TTS). Expect harder tradeoffs on quality.
- **Hard rule:** Mode B must not compromise Mode A quality. If a model swap helps Mode B latency but degrades Mode A output, it goes behind a flag, not into the default pipeline.

## Risks (visible at the roadmap level)

- **Eastern-Armenian TTS availability and quality is the dominant risk.** Phase 0 exists specifically to surface this before plumbing is built. Note: a first scan of the model ecosystem found that `facebook/mms-tts-hye` (Eastern Armenian) does **not** exist on HuggingFace — only `facebook/mms-tts-hyw` (Western Armenian, wrong dialect). The actually-usable local Eastern-Armenian candidate is the community-trained Piper voice `davit312/piper-TTS-Armenian` (`hy_AM-gor-medium`). Phase 0 will judge whether that one voice clears the "watchable for hours?" bar.
- **Eastern-Armenian translation quality** is a secondary risk; per-stage cloud fallback (GPT, Claude) is acceptable here because translation is the easiest stage to keep swappable, but only opt-in after local is judged inadequate.
- **Mode B feasibility** on a single workstation is uncertain; the roadmap treats it as a stretch and is willing to drop it.
- **Voice cloning in Eastern Armenian** may simply not exist at usable quality during the relevant horizon; Phase 2 must not depend on it — the preset-pool plan is the default. Cloning is treated as a stretch attempt (XTTS-v2 local → ElevenLabs cloud), to be dropped cleanly if neither produces Eastern Armenian acceptably.
