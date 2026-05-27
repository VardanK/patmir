# Active translate config

```toml
[pipeline]
translate = "openrouter_gemini_flash"

[[eval.translation.candidates]]
id          = "openrouter_gemini_flash"
enabled     = true
module      = "patmir.eval.candidates.translation.openrouter"
model       = "google/gemini-3.5-flash"
api_key_env = "OPENROUTER_API_KEY"
max_tokens  = 512
temperature = 0.2
top_p       = 0.9
timeout_seconds = 60.0
```

API key in `.env` (auto-loaded by `patmir/env.py`): `OPENROUTER_API_KEY=...`. Same key used by the TTS stage when on the OpenRouter Gemini TTS alternate.

## Model history (latest first)

- **`google/gemini-3.5-flash`** — active since 2026-05-26 (evening). Locked after a casual-register A/B vs `gemini-3.1-flash-lite`. 3.5-flash produced more emotive/playful Eastern Armenian (diminutives like `քչիկ`, exclamatory `Ի՜նչ թթու ա՜`, casual vocative `քեռի՛`) at the same speed. Full A/B table archived at `runs/20260526-183305-masha-2-5min/translate-ab-gemini35.json`.
- **`google/gemini-2.5-flash`** — active 2026-05-26 (morning) → 2026-05-26 (evening). Worked but tilted formal-textbook even with the casual-register prompt rewrite; users described follow-on output as "academic." See `translate-ab.json` for the earlier 5-model comparison (Opus / GPT-5 / GPT-5.4-mini / Pro / Flash) that established casual register matters more than raw model size.
- **`facebook/nllb-200-3.3B`** — active Phase 1.3 → 2026-05-26. Stays registered as the offline-fallback path (`nllb_200_3.3b`). Flip `[pipeline].translate = "nllb_200_3.3b"` to revert.

## System prompt

The casual-register prompt (rewritten 2026-05-26) lives at `patmir/eval/candidates/translation/openrouter.py:_SYSTEM`. Key directives:

- Push spoken Yerevan register, not formal book Armenian.
- Prefer everyday colloquial vocabulary (`լավ` over `բարի`, `ուզում եմ` over `ցանկանում եմ`).
- Use spoken contractions/elisions; keep sentences short to match source rhythm.
- Preserve emotion and tone; sound-effect words use Armenian equivalents (`Վա՜յ`, `Ֆու՜`).
- Never produce Western Armenian (hyw).

Per-sentence user message also gets an emotion hint (`The speaker's emotion is angry; the Armenian should sound angry too.`) when the source sentence carries one.

## Disabled alternates

- `openrouter_claude_opus` — registered, **rejected for Patmir** (user 2026-05-26: cost). Do not re-enable without explicit ask.
- `openrouter_gpt5` — registered. A/B 2026-05-26 showed casual register but 1/16 None-content errors (reasoning model burning tokens internally). Reliability concern for full-episode runs.

## Phase 2.x

Each `Sentence` and `TranslatedSentence` carries an `emotion` field. The OpenRouter candidate injects this into the per-sentence user message; NLLB ignores it. See ADR `decisions/2026-05-26-cloud-translate-via-openrouter.md`.
