# Higgs Audio v3 TTS (4B) — `higgs_v3_mlx_q8`

**Status:** ACTIVE — `higgs_v3_mlx_q8` for **both** batch and watch-along, locked
2026-06-20, unified 2026-08-23. ADR:
`decisions/2026-06-20-higgs-v3-tts-mlx-q8.md`.

**Unified 2026-08-23:** watch-along dropped its `higgs_v3_mlx_q8_fast` override and now shares `higgs_v3_mlx_q8` with batch. The two differed *only* in `max_retries` (1 vs 2) — identical repo, quant, temperature, top_p and seed — so `_fast` was never a faster model: a cleanly-generating sentence cost the same under both, and the cap only bounded the reseed path of the non-termination guardrail (measured 0 overflows / 25 Reacher, 0/37 masha, 0/60 Fallout). One candidate id means one 4B instance serves both modes from the warm-model cache instead of two, which is required on a 24 GB discrete GPU. Tune `max_retries` on the shared candidate if overflows appear.

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
vs OmniVoice-16 **0.91**. Long-line q8 RTF 0.92. Watch-along runs the same
`higgs_v3_mlx_q8` (see the unification note above) and leans on the pre-buffer.
Prefix-KV caching was the intended follow-up but was **dropped 2026-08-23** — the
CUDA move deletes this MLX port for the upstream PyTorch implementation, which
brings bf16, flash-attention and batched decode instead.

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

### The tag vocabulary we are barely using (surveyed 2026-08-24)

The checkpoint's `tokenizer_config.json` declares **62 control tokens**; the model card documents
them. We map five emotion bins onto five tags and use nothing else. What is actually available:

| category | count | tokens |
| --- | --: | --- |
| emotion | **21** | elation, amusement, enthusiasm, determination, pride, contentment, affection, relief, contemplation, confusion, surprise, awe, longing, arousal, anger, fear, disgust, bitterness, sadness, shame, helplessness |
| prosody | 10 | `speed_very_slow/slow/fast/very_fast`, `pitch_low/high`, `pause`, `long_pause`, **`expressive_high`**, **`expressive_low`** |
| style | 3 | singing, shouting, whispering |
| sfx | 9 | cough, laughter, crying, screaming, burping, humming, sigh, sniff, sneeze |
| env | 2 | music, noise |

Three things follow, all relevant to the over-emoting the author reported on 2026-08-24:

1. **`expressive_low` is the only intensity control that exists.** The card documents no per-tag
   strength parameter, so emotion is otherwise binary — the tag fires at whatever intensity the
   model associates with it. `expressive_low` is therefore the dial, and we have never used it.
2. **Our bin→tag map picked the most intense variants.** `happy → elation` is the strongest positive
   available when `contentment`, `amusement`, `affection` and `relief` all exist; `angry → anger`
   where `bitterness` is the milder register. That was never a considered choice — the five tags were
   simply the five obvious name matches for the five bins.
3. **`disgust` is a real tag and we throw it away.** `patmir.toml [emotion_bins]` folds emotion2vec's
   `disgusted` into `angry` (and `surprised` into `happy`), a collapse designed when the target was
   five *ref-clip* bins. With 21 tags available the collapse is now lossy for no reason — though
   widening it is not a one-liner, because `Emotion` is a 5-value `Literal` that the ref-picker,
   the runs UI and `[emotion_bins]` all share.

**Placement is correct and should stay that way.** The card: *"Delivery tokens first. Emotion,
style, and the prosody speed / pitch / expressive tokens shape the whole turn — put them at the
start of input."* `_prepare` builds `f"<|emotion:{tag}|>{text}"`, so the tag leads.

### The map, chosen by ear 2026-08-24 (supersedes the name-match guesses)

Two A/B rounds on real lines with the real pinned refs (`runs/emotion-ab/`,
`runs/emotion-ab2/`). **The original mapping lost every line it was entered in**, in
every bin — it had been five obvious name matches that nobody had listened to.

| bin | original | chosen | why |
| --- | --- | --- | --- |
| `angry` | `anger` | **(none)** | no-tag won 2 of 3 lines; the third preferred `bitterness` + `expressive_low`. 35% of sentences on the test clip. |
| `happy` | `elation` | **(none)** | nothing read as happy; `contentment` introduced a flirtatious tone. 9% of sentences. |
| `sad` | `sadness` | **`helplessness`** | won all three sad lines |
| `fearful` | `fear` | **`shame`** | won the fearful line; `fear` + low second |
| `neutral` | (none) | (none) | unchanged |

`<|prosody:expressive_low|>` is **available and real but not adopted**: an identical
prompt with and without it produces different audio (confirmed by md5), so the dial
works — but it won only one line of nine, and on another the author judged the
lower-intensity variant *more* emotional than the plain tag.

**Treat every tag name as a label, not a description.** `expressive_low` increasing
perceived intensity, `elation` not reading as happy, and `contentment` reading as
flirtation are all cases where the name predicted the wrong thing. There is nothing
upstream to check against — see below.

**Documentation status, since it gets asked:** the tag list is documented in the model card only.
There is **no paper or technical report** linked, so nothing upstream defines what any tag means
acoustically, how strong it is, or whether an emotion tag and `expressive_low` compose. Any mapping
we choose is an ear-test result, not a citation.

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
  tokens; caching that KV per speaker cuts prefill. **Measured 2026-07-15:**
  saving = **14.3% of TTS time** (direct cached-prefix experiment; decode is
  73% and is the real wall). Does NOT bring watch-along under realtime on its
  own. Data: `runs/20260715-reacher-asr-ab/higgs_prefill_split.json`. Not done.
- Optional Higgs **"scene description" system prompt** (`<|scene_desc_start|>…`)
  — our `generate_speech`-style path doesn't expose it; possible quality lever.
- License **research/non-commercial** (fine for this personal tool). MLX →
  Apple-Silicon only; bf16 torch path is the cross-platform fallback.
- Quality differences between seeds/quants are **sampling**, not quant — voice is
  anchored by the pinned ref, not invented (zero-shot has no anchor → varies).
