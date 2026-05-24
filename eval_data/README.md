# Phase 0 Eval Set

**Target dialect: Eastern Armenian (`hy_AM`).** All Armenian utterances and reference translations are written in Eastern Armenian. Models that only produce Western Armenian (`hyw`) are not viable candidates — see `agent-os/product/mission.md`.



~20 short utterances per source language across four categories drawn from the roadmap:

| category    | count | what it stresses                                            |
|-------------|------:|-------------------------------------------------------------|
| `cartoon`   |     8 | short, exclamatory, child register — the primary use case   |
| `narration` |     5 | longer, more formal prosody                                 |
| `idiom`     |     3 | translation quality, not TTS                                |
| `proper`    |     4 | proper-noun handling, transliteration into Armenian script  |

## Files

- `translation/en.jsonl` — English source utterances. Fields: `id`, `category`, `source`, `reference_hy`.
- `translation/ru.jsonl` — Russian source utterances. Same fields.
- `tts/hy.jsonl` — Armenian utterances to synthesize (seeded from the gold translations so TTS is evaluated independently of translation quality). Fields: `id`, `category`, `text`.

## Important — the `reference_hy` field is an AI-drafted starter

The author is the fluent Armenian speaker on this project; these reference translations were drafted by an LLM as a starting point so the harness has *something* to compare against. **Review them before drawing any conclusions** — replacing weak reference translations is part of the eval, not contamination of it. Edits are welcome and expected.

The `tts/hy.jsonl` text comes from those same starter translations and should be updated in lockstep when references are revised.
