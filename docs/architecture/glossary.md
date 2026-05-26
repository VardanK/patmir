# Glossary

Terms used across the codebase, docs, and conversations. Kept short on purpose — for full context, follow the linked stage or decision file.

## A

- **AR / AR voice** — Armenian (output language). Sometimes "the AR track" = the synthesized Armenian audio in `dub.wav`.
- **afade** — `ffmpeg` audio-fade filter. Used at sentence boundaries in mux (50 ms in / 50 ms out) to suppress voice leak.
- **Annotation** (pyannote) — pyannote's primary timeline-of-labeled-segments object. `annotation.itertracks(yield_label=True)` yields the turns; `annotation.get_overlap()` returns overlap regions.
- **Autoregressive** — a model that generates one token at a time, conditioned on previous tokens. Whisper, IndexTTS-2 T2S. Memory grows with sequence length.

## B

- **BigVGANv2** — neural vocoder used by IndexTTS-2's S2M module. Mel → waveform.
- **Bin (emotion bin)** — collapsed emotion category. The 9 SER classes from emotion2vec collapse to 5 bins: `neutral`, `happy`, `sad`, `angry`, `fearful`. Mapping configured in `patmir.toml [emotion_bins]`.

## C

- **CTranslate2 (CT2)** — efficient inference runtime used by `faster-whisper`. Supports int8 quantization on CPU.

## D

- **Demucs / htdemucs** — source-separation model. Splits audio into vocals / drums / bass / other stems. Patmir uses `htdemucs_ft` and sums non-vocal stems into the `accompaniment` bed.
- **Diarize / diarization** — assigning "who spoke when" labels to an audio stream. Patmir uses pyannote 3.1.
- **DiT** — Diffusion Transformer. Backbone of F5-TTS (rejected for Patmir on language grounds). Non-autoregressive.

## E

- **Emilia** — the large multilingual speech dataset F5-TTS trains on.
- **emotion2vec** — wav2vec-style self-supervised speech model fine-tuned for SER. Patmir uses `iic/emotion2vec_plus_base` via funasr.

## F

- **Flow matching** — generative-modeling technique used by F5-TTS. Learns a smooth ODE path from Gaussian noise to the target mel-spectrogram. Few inference steps vs full diffusion.
- **funasr** — the modelscope-published inference framework that loads emotion2vec (and several Chinese-origin SER/ASR models).

## G

- **Gap-filler (Patmir-specific)** — a re-ASR pass over turns ≥1.5 s containing only non-lexical words. Added in Phase 2 to recover dialogue Whisper missed in laughter-dominated regions.
- **GGUF** — quantized model file format used by `llama.cpp` and friends. NLLB has no upstream GGUF support (only a fork-based attempt exists).
- **guidance_scale** — OmniVoice TTS knob. Strength of conditioning during denoising. Higher = more faithful to ref + text, lower = more freedom. Patmir uses a per-emotion-bin table.

## H

- **HF token / `HF_TOKEN`** — HuggingFace access token. Required by pyannote 3.1 (license-gated). Patmir reads it from env or the `hf auth login` cached token.
- **`hy_AM`** — Eastern Armenian. The project's target output dialect.
- **`hyw`** — Western Armenian. Rejected by Patmir on dialect grounds.

## I

- **IndexTTS-2** — Bilibili's industrial TTS. 8-dim emotion vector, decouples speaker from emotion. Rejected for Patmir on language grounds (no Armenian).

## M

- **MLX** — Apple's array/ML framework optimized for Apple Silicon. Used by mlx-whisper, OmniVoice MLX, f5-tts-mlx (Patmir uses the first two).
- **MMS** — Meta's massively multilingual speech project. `mms-tts-hyw` exists (Western), `mms-tts-hye` does not.
- **MPS** — Metal Performance Shaders. PyTorch's Apple-Silicon GPU backend. Used by pyannote (partial coverage), NLLB.

## N

- **NLLB** — "No Language Left Behind". Meta's multilingual translator. Patmir uses `nllb-200-3.3B` at fp32 on MPS as the active translator.

## O

- **OmniVoice MLX** — Patmir's active TTS. Zero-shot voice cloning via reference audio. Supports per-call `ref_audio`/`ref_text`/`guidance_scale`. EA support verified.
- **Overlap interval** — a time window where ≥2 speakers are simultaneously active per pyannote. Used to filter ref-clip candidates (clean ref clips must not intersect any overlap).

## P

- **pyannote** — speaker-diarization toolkit. Patmir uses `pyannote/speaker-diarization-3.1`.
- **pyrubberband** — Python wrapper for `rubberband-cli`. Pitch-preserving time-stretch used in mux.

## Q

- **q8 / Q8** — 8-bit quantization. OmniVoice MLX runs at q8 to fit comfortably alongside other models on a 32 GB Mac.
- **Qwen3 / Qwen3-Instruct-MLX** — instruction-tuned LLM, MLX-ported. Registered as a translator candidate with the system prompt "You preserve tone, register, and emotion." Not the current default; gets activated when its emotion-aware translation is needed (or when ear-judged to beat NLLB-3.3B).

## R

- **ref clip / reference clip** — a short audio sample (typically 3-10 s) used to clone a speaker's voice via OmniVoice. Patmir auto-extracts these from `speech_vocals.wav` per `(speaker, emotion_bin)`.
- **ref_text** — the transcript of the ref clip's audio. OmniVoice consumes both. Backfilled by the runner after ASR runs (ASR words inside the ref's source-time window are joined).

## S

- **SER** — Speech Emotion Recognition. New `emotion` stage in Phase 2.x. Detects per-segment emotion from audio.
- **Silero VAD** — Voice Activity Detector. Produces `vad_intervals` in the separate stage. Used as an ASR post-filter (drop hallucinated words outside any VAD window) and to build `speech_vocals` / `non_speech_vocals`.
- **Stage / stage candidate** — A pipeline step (`STAGE_ORDER`) is implemented by one *candidate* selected in `patmir.toml [pipeline]`. Each stage has a `Protocol` in `stage.py` and a registry of candidates in `[[stages.<stage>.candidates]]` (or legacy `[[eval.<stage>.candidates]]`).
- **stretch_cap** — max ±20% time-stretch ratio per sentence in the mux stage. Preserves pitch via pyrubberband.

## T

- **T2S** — Text-to-Semantic. The Qwen3-finetuned first module of IndexTTS-2's three-module cascade.
- **Turn (speaker turn)** — one contiguous span attributed to one speaker by the diarizer.

## V

- **VAD** — Voice Activity Detection. See Silero VAD.
- **vocals.wav** — Demucs vocal stem (full). What ASR + diarize consume.

## W

- **Whisper / whisper-large-v3 / mlx-whisper** — OpenAI's ASR. Patmir uses the MLX port at fp16 by default.
