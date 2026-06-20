# Higgs Audio v3 TTS (4B) via MLX-q8 port — active batch TTS

**Date:** 2026-06-20
**Status:** locked
**Stage:** tts (batch)

## Context

OmniVoice (the prior active TTS) was the only verified Eastern-Armenian local
TTS, but it has **no explicit emotion control** — emotion was faked through
`guidance_scale` + per-`(speaker, emotion)` ref-clip binning, which caused
characters to swap voices mid-scene and over-emote (quality issue #11) and never
really delivered emotion transfer (deferred issue #3).

`bosonai/higgs-audio-v3-tts-4b` (June 2026, 4B chat-native TTS) is the first
local TTS we found that does **production-quality Eastern Armenian**, zero-shot
cloning, **and** real **inline emotion/style/prosody control** (`<|emotion:..|>`
tags). That hit the revisit trigger in ADR `2026-05-26-omnivoice-locked-for-ea`.

## Gate (ear-tested 2026-06-20)

Ran the stock model on MPS via the transformers `trust_remote_code` port
(`multimodalart/higgs-audio-v3-tts-4b-transformers`). Author-confirmed:
**Eastern Armenian (not Western), clean cloning, and the `<|emotion:anger|>` tag
audibly changes delivery vs no-tag.** Passes the EA product gate.

## Decision

Active TTS = Higgs `higgs_v3_mlx` for **both** modes: `higgs_v3_mlx_q8` (batch,
`[pipeline].tts`) and `higgs_v3_mlx_q8_fast` (watch-along, `max_retries=1`,
`streaming/runner.py` override). Watch-along was switched off
`omnivoice_mlx_bf16_fast` on the same day for engine consistency despite the RTF
gap (see Performance) — it leans on the pre-buffer, with prefix-KV caching as the
follow-up. OmniVoice candidates remain registered as fallbacks; flip
`[pipeline].tts` / the streaming override to revert.

## Runtime: hybrid MLX + torch (our own port, not sglang/vllm)

The decode is **memory-bandwidth-bound** on the 4B weights — ~69 ms/step in
torch-MPS bf16 (re-reading ~8 GB/token; ~116 GB/s ≈ this Mac's ceiling). Neither
torch micro-opt nor `torch.compile` helps (measured: Tier-1 loop opts gave 0%).
The only lever is **quantized decode**, and the only Apple-Silicon runtime that
does fast quantized matmul is **MLX** (torchao/bitsandbytes absent/weak on MPS).
There is no working MLX runtime for this custom arch upstream (mlx-audio #766
open; stock transformers can't load `higgs_multimodal_qwen3`), so we ported it:

- **MLX** (`mlx_lm` Qwen3-4B backbone, **q8**) for the AR decode + Higgs' fused
  multi-codebook embed/head + delay-pattern sampler;
- **PyTorch/transformers** for the 0.8 GB codec (`higgs-audio-v2-tokenizer`,
  ref-encode + waveform-decode) on MPS — runs once/utterance, not per step.

KV cache is active (essential): `mlx_lm` `make_prompt_cache` + per-step
`update_and_fetch`. Per-speaker ref-prefix KV reuse across sentences is an open
optimization (the watch-along headroom).

## Performance (head-to-head, same sentences + ref, aggregate RTF = wall/audio)

| Engine | agg RTF | note |
| ------ | ------- | ---- |
| OmniVoice-32 (prior batch) | 1.69 | |
| OmniVoice-16 (watch-along)  | 0.91 | clears realtime |
| **Higgs q8 (codec MPS + ref-cache)** | **1.10** | long-line 0.92 |
| Higgs bf16 | 1.69 | |
| Higgs q4 | — | **rejected: runaway/collapse** |

- **Batch: Higgs q8 (1.10) beats OmniVoice-32 (1.69) — faster AND better quality.**
- **Watch-along: OmniVoice-16 (0.91) is faster than Higgs q8 (1.10)** — but
  watch-along was switched to Higgs anyway (2026-06-20) for engine consistency;
  it leans on the pre-buffer and prefix-KV caching is the follow-up to get it
  comfortably under realtime. Per-step: torch-bf16 69 ms → MLX-bf16 48 → MLX-q8 28.
- q8 chosen over q4 (q4 collapses) and over bf16 (2.5× slower, no quality need).

## Non-termination guardrail (the key operational finding)

The AR decoder occasionally fails to emit its stop token and runs to the cap
(known Higgs issue, sglang-omni #720). **Measured, counterintuitively: LOW
temperature causes overflow, not high** — at temp 0.3 a long line overflowed
5/5; 0.7–0.9 was 0/5. Low temp → near-greedy → repetition lock → never samples
EOC. The cited generic advice (temp 0.01–0.3 for stability) is **wrong for our
setup** (it targets long-paragraph serving). Also: short-line overflows were
caused by a **repetitive/noisy ref** (cartoon "ням-ням" munching), not temp — a
clean adult-speech ref was 0/5 at every temp.

Guardrail in the candidate: **temp 0.8** (stable zone), `max_new_tokens`
length-proportional (`min(1024, max(200, 4×chars))`) so overflow is caught in
~1 s, and **reseed-and-retry** on overflow (≤2 retries). `summarize()` logs the
overflow count to `tts/voice_cloning_summary.json`.

## Emotion mapping

Pipeline bins → Higgs tags (candidate `EMOTION_TAGS`, tunable): happy→`elation`,
sad→`sadness`, angry→`anger`, fearful→`fear`, neutral→none. Injected at sentence
start. The adapter passes `emotion` (not `guidance_scale` — Higgs has no such knob).

## Notes / open items

- License: **research/non-commercial** — fine for this personal workstation tool.
- MLX → Apple-Silicon only; the torch-bf16 path (stock transformers port) is the
  cross-platform fallback (≈1.9 RTF).
- Quality difference between seeds/quants is **sampling**, not quant — voice is
  anchored by the pinned per-speaker ref (`tts_pin_speaker_ref`), not invented.
- **Open:** per-speaker prefix-KV caching (→ watch-along under realtime);
  optional Higgs "scene description" system prompt (not currently used).

## Revisit

If mlx-audio #766 ships a maintained runtime (replace our port), if prefix-KV
caching lands (reconsider Higgs for watch-along), if a clip shows EA regression
or emotion mis-fire, or if the non-commercial license becomes a constraint.
