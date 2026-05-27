# Cloud ASR via OpenRouter (gpt-audio chat-completions + MMS-FA forced alignment)

- **Date:** 2026-05-26
- **Status:** locked (per-stage cloud opt-in for ASR, via OpenRouter)
- **Supersedes:** local `whisper_large_v3_mlx_fp16` as the active ASR (kept as offline fallback)

## Context

After translate (`openrouter_gemini_flash`) and TTS (`openrouter_gemini_flash_tts`) moved to OpenRouter as the active cloud stages, ASR was the last major local stage. Local Whisper Large-v3 (MLX) was workable but required a scaffold of compensating logic — VAD post-filter, hallucination blocklist, and a gap-filler that re-ran ASR on laughter-dominated regions — to be usable on cartoon source audio. Quality-issue tracker #1 (voice cloning fidelity) and #3 (emotion transfer) had recently been improved by escalating downstream stages to cloud; the natural next step was setting a cleaner ASR baseline so per-sentence text quality stopped capping translate output.

## Decision

Opt in to **cloud ASR via OpenRouter** as the active ASR.

- **Active model:** `openai/gpt-audio` — clean Russian transcription on Masha clips, same model family as the (no-longer-exposed) `gpt-4o-transcribe` SKU. Approximate cost: $1.20 / hr of audio at posted rates.
- **Alternates registered as `enabled = false`:** `mistralai/voxtral-small-24b-2507`, `openai/gpt-audio-mini`, `openai/gpt-4o-audio-preview`. Same module — flip `[pipeline].asr` to A/B.
- **Word-level timestamps:** reconstructed locally by Meta's MMS-FA via the `ctc-forced-aligner` package. One ~1.2 GB ONNX model cached at `~/.cache/ctc_forced_aligner/mms_fa.onnx`.
- **ElevenLabs Scribe v2:** registered as a `[[stages.asr.candidates]]` stub with `enabled = false` and no candidate module. Documentation-only future option; revisit when we want diarization-via-ASR or to drop MMS-FA.
- **Local Whisper-MLX stays registered** as the offline fallback.

## Why not the originally-planned route

The plan called for `openai/gpt-4o-transcribe` via OpenRouter's `/v1/audio/transcriptions` endpoint with native `timestamp_granularities=["word"]`. Smoke probe revealed:

1. OpenRouter does **not** expose `/v1/audio/transcriptions`. The endpoint 500s for every transcribe-SKU model id (`openai/gpt-4o-transcribe`, `openai/whisper-1`, etc.).
2. The transcribe-SKU model ids that earlier research (Gemini deep-research doc; Explore agent) suggested were OpenRouter-routable (`qwen/qwen3-asr-flash`, `google/chirp-3`, `mistralai/voxtral-mini-transcribe`) are **not in OpenRouter's catalog**. Enumerated 356 models — only 4 audio-capable models exist, all on the chat-completions multimodal route.
3. Audio-in on OpenRouter goes through `POST /v1/chat/completions` with `input_audio` content blocks. These models return plain transcript text with no timing.

Pivoted to chat-completions multimodal audio. Forced alignment (originally planned as a fallback) became mandatory for every call.

## Why not Soniox / Gladia / Deepgram / Speechmatics / direct-vendor APIs

User constraint: "Use what's available on OpenRouter." Keeping a single cloud key (`OPENROUTER_API_KEY`) was deliberate — same key drives translate, TTS, and ASR. Soniox / Gladia have richer native features (per-token timestamps, end-to-end translation including Armenian) but require a separate key and account. Direct OpenAI STT (`api.openai.com/v1/audio/transcriptions` with `gpt-4o-transcribe`) would give native word timestamps but adds a second cloud key. Both off the table this round.

## Why not stay on local Whisper-MLX

It works, but produces hallucinations on cartoon laughter/scream regions that the gap-filler + hallucination blocklist were patching around. Quality-issue trackers acknowledge the scaffolding. The cloud upgrade cleanly removes both the gap-filler invocation path and the largest source of hallucinations.

## Architecture (one-paragraph summary)

`patmir/pipeline/stages/asr/openrouter_stt.py` POSTs `/v1/chat/completions` with a verbatim-transcription system prompt and an `input_audio` content block (base64 WAV). Long inputs are split into ≤240 s chunks; chunk boundaries snap to VAD silence when available. Transcripts are concatenated. `_aligner.py` runs MMS-FA on the full vocals stem with the concatenated text, returning per-word `start`/`end`. Sentence grouping uses the existing `_sentence.group_into_sentences` helper (same module the local Whisper candidate uses). The runner's post-ASR passes (`apply_speaker_tags`, `apply_emotion_tags`, `_backfill_ref_text`) are unchanged — the cloud candidate is a drop-in `ASRStage`.

## Revisit triggers

- Cost surprise — gpt-audio's per-call cost exceeds expectations on a real episode.
- An OpenRouter-routable STT model with native word timestamps appears (would let us drop MMS-FA).
- ElevenLabs Scribe is ear-tested and approved — would replace both gpt-audio and MMS-FA in one swap.
- Russian transcription regresses on a representative episode vs. local Whisper-MLX.
- A frontier-quality local Armenian/Russian ASR model appears (revert to local).

## Verification

- Smoke probe (this commit): 5-min masha-2 clip → 33 sentences of clean Russian, downstream pipeline ran clean through translate / tts / mux / output. Artifact: `runs/20260526-183305-masha-2-5min/out.mkv`.
- Full-episode ear-test deferred (user direction during the bring-up session).
