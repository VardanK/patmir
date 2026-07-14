# Higgs Audio v3 TTS (4B) — `higgs_v3_mlx_q8`

**Status:** ACTIVE — batch (`higgs_v3_mlx_q8`) **and** watch-along
(`higgs_v3_mlx_q8_fast`, `max_retries=1`), locked 2026-06-20. ADR:
`decisions/2026-06-20-higgs-v3-tts-mlx-q8.md`.

Model: `bosonai/higgs-audio-v3-tts-4b` (4B chat-native TTS, June 2026), loaded
via the transformers `trust_remote_code` port
`multimodalart/higgs-audio-v3-tts-4b-transformers`. Candidate:
`patmir/eval/candidates/tts/higgs_v3_mlx.py`.

## Why it's the batch TTS

First **local** TTS that does all three: production-quality **Eastern Armenian**,
**zero-shot cloning** (ref clip + optional transcript), and **real inline emotion
control** (`<|emotion:..|>` / `<|style:..|>` / `<|prosody:..|>` tags). That last
point is what OmniVoice can't do — it routed emotion through a guidance_scale
proxy + per-emotion ref binning, which swapped voices mid-character and
over-emoted (quality issues #3, #11). Ear-gate passed 2026-06-20 (EA not Western;
clean clone; the anger tag audibly changes delivery).

## Architecture (hybrid MLX + torch)

Custom arch `HiggsMultimodalQwen3ForConditionalGeneration` = a standard **Qwen3-4B**
backbone + a **fused multi-codebook** audio embedding/head (8 codebooks × 1026
vocab, delay pattern) + the **`higgs-audio-v2-tokenizer`** codec (waveform ↔ codes).

We run it as a **hybrid** because the decode is memory-bandwidth-bound on the 4B
weights and only MLX does fast quantized matmul on Apple Silicon (no upstream MLX
runtime exists — mlx-audio #766 open; stock transformers can't load the arch):

- **MLX** (`mlx_lm` Qwen3 backbone, **q8**) — the AR decode + our port of the fused
  embed/head + delay-pattern sampler. KV-cached (`make_prompt_cache`).
- **PyTorch** (`transformers` `higgs-audio-v2-tokenizer`, fp32, MPS) — ref-encode
  + waveform-decode, once per utterance (~125 ms), ref-encode cached per speaker.

Weight load: `mx.load` the port's `model.safetensors`, remap `body.*`→Qwen3,
`tied.embedding.text_embedding`→`embed_tokens`, `tied.embedding.modality_embeddings.0.embedding`
→ the fused audio matrix (head is tied to it). Codec weights in the ckpt
(`...modality_embeddings.0.model.*`) are ignored — loaded separately.

## Performance

Decode per-step: torch-bf16 **69 ms** → MLX-bf16 **48** → **MLX-q8 28**. Quantization
is the only lever (decode is bandwidth-bound; torch micro-opt + `torch.compile`
measured 0% — they don't reduce bytes/token). Aggregate RTF (wall/audio, same
sentences + clean ref): **Higgs q8 1.10** vs OmniVoice-32 **1.69** (batch win),
vs OmniVoice-16 **0.91**. Long-line q8 RTF 0.92. Watch-along uses
`higgs_v3_mlx_q8_fast` (same q8, `max_retries=1`) despite the RTF gap — it leans
on the pre-buffer; prefix-KV caching is the follow-up to clear realtime there.

- **q4 rejected** — collapses (runaway / no EOC); matches "Higgs is quant-sensitive".
- **bf16** = cross-platform fallback (~1.9 RTF, 2.5× slower).

## Non-termination guardrail (important)

The AR decoder sometimes fails to emit EOC and runs to the token cap (known issue,
sglang-omni #720). **Measured here, counterintuitive: LOW temperature causes
overflow, not high.** Long line: temp 0.3 → 5/5 overflow, 0.7–0.9 → 0/5. Low temp
→ near-greedy → repetition lock → never samples EOC. Short-line overflows were
caused by a **repetitive/noisy ref** (cartoon "ням-ням"), not temp; a clean
adult-speech ref was 0/5 at every temp. The cited 0.01–0.3 advice is wrong for our
setup (it targets long-paragraph serving).

Candidate defaults: `temperature=0.8`, `top_p=0.95`, `max_new_tokens =
min(1024, max(200, 4×chars))` (overflow caught in ~1 s), `max_retries=2` (reseed
on overflow). Overflow count logged to `tts/voice_cloning_summary.json` via
`summarize()`.

## Adapter call passes

The `SentenceTTS` adapter probes the signature and passes `ref_audio`, `ref_text`
(optional — `ref_text_optional=True`), and **`emotion`** (→ inline tag via
`EMOTION_TAGS`: happy→elation, sad→sadness, angry→anger, fearful→fear, neutral→none).
It does **not** pass `guidance_scale` (Higgs has no such knob). Ref selection still
honors `tts_pin_speaker_ref` (one clean ref per speaker → stable voice).

## Config (`patmir.toml`)

```toml
[[eval.tts.candidates]]
id = "higgs_v3_mlx_q8"
module = "patmir.eval.candidates.tts.higgs_v3_mlx"
repo_id = "multimodalart/higgs-audio-v3-tts-4b-transformers"
codec_id = "bosonai/higgs-audio-v2-tokenizer"
quant = "q8"; temperature = 0.8; top_p = 0.95; seed = 0; max_retries = 2; codec_device = "mps"
```

## mlx-audio native runtime A/B (2026-07-14 — the ADR's revisit trigger fired)

mlx-audio merged a native `higgs_audio_v3` runtime (PR #770, 2026-06-05,
closing #766): fully-MLX codec (no torch bridge), inline control tokens,
zero-shot cloning via `ref_audio`/`ref_text`. Measured against our hybrid port
on 8 EA sentences with the same masha ref clip (`runs/20260714-higgs-ab/`):

| engine | internal RTF | note |
| --- | --- | --- |
| ours-q8 (hybrid port) | 1.052 | baseline |
| native-q8 (bf16 load + LM-only `nn.quantize` 8-bit/g64) | 1.018 | parity |
| **native-q4** (same, 4-bit) | **0.744** | **−29%, no collapse** — comparable audio lengths (slightly longer/slower-paced) |

- Our port's q4 collapsed (ADR 2026-06-20); **mlx-audio's runtime at q4 does
  not** — likely the LM-only quant predicate (codec/codebooks excluded; naive
  full-model quantize breaks the codec's VQ math — reproduced).
- The community `Reza2kn/Higgs-Audio-v3-TTS-4bit-MLX` repo does **not** load in
  mlx-audio's runtime (different weight layout, `backbone.*` naming) — quantize
  in-memory from `bosonai/higgs-audio-v3-tts-4b` instead.
- **Author ear-test 2026-07-14: native-q4 REJECTED** (worst of the three —
  faster but audibly degraded; "no collapse" ≠ good quality). **ours-q8 ranked
  best** and is the engine already active (`higgs_v3_mlx_q8`) → **no swap**.
  native-q8 was only parity, so the native runtime brings no benefit at equal
  quality. Our hybrid port stays; the remaining TTS speed lever is per-speaker
  prefix-KV caching (roadmap #4).

## Open items / notes

- **Per-speaker prefix-KV caching** — each sentence re-prefills the cloned ref's
  tokens; caching that KV per speaker would cut short-line overhead and likely
  bring watch-along under realtime. Not done.
- Optional Higgs **"scene description" system prompt** (`<|scene_desc_start|>…`)
  — our `generate_speech`-style path doesn't expose it; possible quality lever.
- License **research/non-commercial** (fine for this personal tool). MLX →
  Apple-Silicon only; bf16 torch path is the cross-platform fallback.
- Quality differences between seeds/quants are **sampling**, not quant — voice is
  anchored by the pinned ref, not invented (zero-shot has no anchor → varies).
