# 2026-06-15 — Local ASR + diarize (and watch-along TTS) via mlx-audio

**Status:** accepted

## Context

ElevenLabs credits were exhausted with no plan to top up. The pipeline depended
on ElevenLabs `scribe_v2` for **both** `asr` and `diarize` (one shared API call,
cached at `<job>/scribe/result.json`), and the watch-along (D3) streaming mode
hard-coded ElevenLabs `elevenlabs_pool` for TTS. With no credits, the default
batch pipeline *and* watch-along both failed. The project's local-first policy
(`decisions/2026-05-24-local-first-strict.md`) makes moving these stages local
the correct response, not a stopgap.

The author proposed two specific Apple-Silicon models, both served by the
**mlx-audio** library (`github.com/Blaizzy/mlx-audio`) — which already hosts our
OmniVoice TTS, so the whole audio stack converges on one dependency.

## Decision

- **ASR → `nemotron_mlx`** — `mlx-community/nemotron-3.5-asr-streaming-0.6b`
  (NVIDIA FastConformer-RNNT, 0.6B) via `mlx_audio.stt`. Replaces cloud Scribe
  v2. `whisper_large_v3_mlx_fp16` stays the registered offline fallback.
- **Diarize → `sortformer_mlx`** — `mlx-community/diar_streaming_sortformer_4spk-v2.1-fp32`
  (NVIDIA Sortformer, end-to-end, ≤4 speakers) via `mlx_audio.vad`. Replaces
  cloud Scribe v2. `pyannote_3_1` stays the registered fallback.
- **Watch-along TTS → `omnivoice_mlx_q8_fast`** — the working
  `lightsofapollo/omnivoice-mlx-q8-g64` weights with `num_steps=8` for realtime.
  Replaces `elevenlabs_pool` in `streaming_config()`. Batch TTS
  (`omnivoice_mlx_q8`, `num_steps=32`) is unchanged.

`translate = openrouter_gemini_flash` (OpenRouter, `OPENROUTER_API_KEY`) is the
only remaining cloud stage and is unaffected.

## Why these

- **Nemotron timing risk cleared by probe.** The model card only documented a
  `.text` output, but `generate()` returns an `AlignedResult` with `.sentences`
  (each `AlignedSentence` has `.start/.end` and a `.tokens` list of
  `AlignedToken(.start/.end/.duration)`). Full token+sentence timing — enough for
  `apply_speaker_tags`, `apply_emotion_tags`, the VAD post-filter, and mux. Tokens
  are sentencepiece subwords (leading-space word boundary), merged into words by
  the candidate. Auto-detect picks the language correctly; **forcing the wrong
  locale produces phonetic gibberish**, so the candidate auto-detects and infers
  ISO 639-1 from the transcript script for translation routing.
- **Sortformer fits better than pyannote.** End-to-end neural diarization (no
  clustering hyperparameters) resolves short cartoon turns far better than
  pyannote — the exact failure that drove the Scribe swap
  (`2026-05-27-scribe-v2-for-asr-and-diarize.md`). It also emits concurrent
  speakers, so the candidate re-derives overlap intervals — **restoring the
  overlap output Scribe dropped** (quality issue #5). The Scribe ADR's own
  revisit trigger named NeMo Sortformer as the local diarizer to watch for.

## Alternatives rejected

- **`mlx-community/OmniVoice-8bit`** — won't load on mlx-audio (config.json is
  missing the `quantization` block, so the model isn't quantized and the `.scales`
  weights are rejected). Its model card targets a *different* runtime
  (`ailuntx/OmniVoice-MLX`) and states its slim tokenizer "do[es] not provide a
  reliable speaker-encoding path" — i.e. no reliable voice cloning. Disqualified.
- **`mlx-community/OmniVoice-bf16`** — loads, but synth raises "tokenizer
  (HiggsAudioTokenizer) is required for voice cloning via ref_audio"; the repo's
  audio-tokenizer assets aren't wired up by mlx-audio's non-quantized load path.
  No cloning → disqualified for our per-character TTS. Noted as a future upgrade
  path only if a working clone-capable build appears.

## Consequences

- Fully local except translate. No ElevenLabs dependency in either execution mode.
- ASR and diarize are now independent stages (no shared upload). `diarize` runs
  before `asr` as before; `apply_speaker_tags` maps ASR sentences onto Sortformer
  turns — the same local path that predated Scribe.
- Sortformer caps at 4 speakers; content with >4 distinct speakers will under-cluster.
- Watch-along voice consistency now leans on the ECAPA `SpeakerRegistry` feeding
  stable per-identity refs to OmniVoice cloning (vs preset pool voices before) —
  watch for early-scene consistency.

## Revisit triggers

- Nemotron Russian quality disappoints on a representative clip → flip
  `[pipeline].asr = "whisper_large_v3_mlx_fp16"`.
- Sortformer mis-handles >4 speakers or merges turns → flip
  `[pipeline].diarize = "pyannote_3_1"`.
- OmniVoice `num_steps=8` doesn't clear realtime in watch-along → drop steps or
  fall back to `piper_hye`; if quality is too low, raise steps.
- ElevenLabs credits return and Scribe is worth the cost again → flip both stages
  back (candidates stay registered).

---

## Update 2026-06-19 — ASR reverted to Whisper large-v3 for Russian quality

The first Russian-cartoon (masha) ear-test exposed Nemotron 3.5 **garbling fast
cartoon dialogue** — Devanagari script leakage (`कंसर्ट`/`ना`) + mangled words —
even though it was clean on slower Russian (Fallout-RU). Flipped `[pipeline].asr`
back to `whisper_large_v3_mlx_fp16` (the registered offline fallback), which
transcribes masha cleanly. Whisper large-v3-**turbo** was also ear-tested and
rejected (8×-pruned decoder → gibberish + ~half the word coverage). Tradeoff
accepted: Whisper is ~6× slower than Nemotron (~20s/60s) but correct; ASR is only
~16% of the pipeline. Nemotron stays registered as a fast alternate for
slower/clearer-speech content. See `stages/asr/current.md`.
