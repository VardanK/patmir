# elevenlabs-scribe (asr) — documented, not used

- **Repo / version:** ElevenLabs Scribe v2 (and Scribe v2 Realtime), via the ElevenLabs REST API. NOT routed through OpenRouter.
- **Status in pipeline:** **registered, disabled** (`enabled = false`). No candidate module is implemented — the `[[stages.asr.candidates]]` toml entry is documentation-only.
- **Why it's gated:** project policy (not API tier). Scribe is actually available on the ElevenLabs Free tier — the gate mirrors how `elevenlabs_multilingual_v2` TTS is shelved: cloud candidates stay disabled until ear-tested and budget-sanctioned. This avoids accidental activation when flipping `[pipeline].asr`.
- **What it offers:**
  - Native word-level timestamps (no separate forced-aligner pass).
  - Built-in speaker diarization (up to 32 speakers per request).
  - 90+ languages with Russian in the top WER tier per published benchmarks.
  - Up to 3 GB / 10 hr per request.
  - Pricing from ~$0.40/hr at scale.
- **When to revisit:**
  - We want to evaluate whether dropping MMS-FA forced alignment (native timestamps) improves dub-time alignment.
  - We want to A/B against `gpt-audio` on a representative episode.
  - We want diarization-via-ASR for a future spec that collapses the diarize + asr stages.
- **Activation steps (when revisited):**
  1. Write the candidate module `patmir/pipeline/stages/asr/elevenlabs_scribe.py` (mirror the OpenRouter STT candidate's structure).
  2. Set `ELEVENLABS_API_KEY` (`.env` already has the value).
  3. Flip the toml entry to `enabled = true` and `[pipeline].asr = "elevenlabs_scribe_v2"`.
  4. Compare a masha-2-5min run output ear-to-ear against the OpenRouter baseline.
- **References:**
  - ElevenLabs API docs (https://elevenlabs.io/docs/api-reference/speech-to-text).
  - Local mirror of the gating pattern: `docs/architecture/stages/tts/elevenlabs.md` + ADR `2026-05-26-elevenlabs-tts-cloud-opt-in.md`.
