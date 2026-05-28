# TranslateGemma (translate — local MLX, **rejected**)

**Status:** rejected (2026-05-27)

## What it is

Google's TranslateGemma family — Gemma 3 base, fine-tuned for translation across 55+ "production-tier" language pairs (160+ codes total, with the long tail trained from GATITOS/SMOL datasets and flagged as experimental). Decoder-only, 2K-token context. Open weights under the Gemma license. MLX builds published by `mlx-community/`:

| Variant | On-disk | Inference RAM | Available |
| --- | --- | --- | --- |
| `mlx-community/translategemma-4b-it-4bit`  | 2.2 GB | ~3 GB  | ✓ |
| `mlx-community/translategemma-4b-it-8bit`  | 4.5 GB | ~6 GB  | ✓ |
| `mlx-community/translategemma-12b-it-4bit` | 7 GB   | ~9 GB  | ✓ |
| `mlx-community/translategemma-12b-it-8bit` | 13 GB  | ~16 GB | ✓ |
| `mlx-community/translategemma-27b-it-4bit` | 15 GB  | ~30 GB peak | ✓ ← benchmarked |
| `mlx-community/translategemma-27b-it-8bit` | 28 GB  | n/a    | doesn't fit 32 GB |

Armenian (`hy`) is present in the chat-template language map. Whether `hy` is in the 55 production-tier set or the experimental tail is undocumented (would require the technical report).

## Why we evaluated it

User asked for a local Gemma-family translation benchmark. Smoke-tested the 27B 4-bit variant on the first 10 sentences of `runs/20260526-183305-masha-2-5min/translated.json` (Gemini 3.5 Flash as the reference target).

## Verdict

Rejected for two reasons:

### 1. No register / style control surface

TranslateGemma's chat template is **structured** and accepts only `{type, source_lang_code, target_lang_code, text}` per message — there's no system prompt and no way to inject "translate into casual spoken Yerevan, not literary Armenian." The casual-Yerevan register prompt + per-sentence emotion hint that Gemini 3.5 Flash uses is simply not honored.

In practice, this means output skews literary:

| Russian | Gemini (casual) | TranslateGemma 27B (literary) |
| --- | --- | --- |
| `Хочешь попробовать?` | `Ուզո՞ւմ ես փորձես` | `Ցանկանո՞ւմ ես փորձել` (formal "do you wish to try") |
| `А ну стой!` | `Արա կա՛նգնի։` | `Այդպես, կանգնի!` (neutralized) |
| `Буэ,` | `Ֆո՜ւյ` | `Բուե։` (literal transliteration of the Russian sound, not the Armenian interjection) |

### 2. Output verbosity + max-tokens latency

Without a system-prompt surface, the model occasionally appends a parenthesized Latin transliteration: `Փխ, ինչ անհաճոյ։ (Phkh, inch anhachoy)`. We can't tell it not to.

The 27B 4-bit on MPS produces ~7 tok/s. With `max_tokens=256`, runs that don't hit a natural EOS race the cap, giving **37 s/sentence wall time** on the 10-sample probe (vs ~1 s on Gemini Flash). A 41-sentence masha clip would take ~25 min to translate.

A tuned version (`max_tokens=48` + stop-string on `(`) would land ~6 s/sentence — viable but still 5-6× slower than Gemini, and quality is still the bigger issue.

## ml-explore/mlx-lm gotcha

mlx-lm v0.31.x does **not** apply TranslateGemma's structured chat template — see [ml-explore/mlx-lm#783](https://github.com/ml-explore/mlx-lm/issues/783). The candidate at `patmir/eval/candidates/translation/translategemma_mlx.py` bypasses `apply_chat_template` and reproduces the Jinja template's prompt string by hand. The full template is at `mlx-community/translategemma-*-it-*/chat_template.jinja`.

## When to revisit

- TranslateGemma adds a system-prompt or register-hint surface in a future revision (would require an architecture change in the chat template).
- A community fine-tune on the Gemma base with explicit register/colloquial-Armenian training surfaces.
- The active translator (Gemini Flash via OpenRouter) becomes unavailable or much more expensive and a local fallback with literary register would be better than no dub.

## Registered candidates (`patmir.toml`)

All `enabled = false`:

| Candidate id | model_id |
| --- | --- |
| `translategemma_27b_mlx_4bit` | `mlx-community/translategemma-27b-it-4bit` |
| `translategemma_12b_mlx_4bit` | `mlx-community/translategemma-12b-it-4bit` |
| `translategemma_4b_mlx_4bit`  | `mlx-community/translategemma-4b-it-4bit` |
