# Stage: translate

Translates source-language sentences into Eastern Armenian (`hy_AM`).

## I/O

- **Input:** [`Transcript`](../../data-contracts.md#transcript) (with `language` set by ASR)
- **Output:** [`TranslatedTranscript`](../../data-contracts.md#translatedtranscript)

## Protocol

```python
class TranslateStage(Protocol):
    def translate(self, transcript: Transcript, ctx: JobContext) -> TranslatedTranscript: ...
```

## Candidates

### Active

- **[nllb-200-3-3b](nllb-200-3-3b.md)** — active default since Phase 1.3.

### Registered (local — alternates / future)

- **[nllb-200-distilled-600m](nllb-200-distilled-600m.md)** — superseded; kept for legacy.
- **[qwen3-instruct-mlx](qwen3-instruct-mlx.md)** — emotion-aware via system prompt + per-sentence user-prompt injection. Activates when chosen as the pipeline translator.
- **[opus-mt](opus-mt.md)** — fast-path A/B candidate (EN→hy + RU→hy variants).
- **[hygpt-10b-it](hygpt-10b-it.md)** — future: Gemma-2-9b + Armenian fine-tune. GPU-only.
- **[madlad-400](madlad-400.md)** — future: GPU re-eval.

### Cloud (opt-in only — local-first policy)

- **[gpt-cloud](gpt-cloud.md)** — disabled. Needs an explicit local-exhaustion ADR before activation.
- **[claude-cloud](claude-cloud.md)** — disabled. Same gate.

## Sentence-level mechanics

The translate adapter (`stages/translate/adapter.py`) iterates the source sentences, calls the inner translator's `translate(text, source_lang=...)`, and constructs `TranslatedSentence`s passing `speaker_id` (and Phase 2.x: `emotion`) through unchanged. Inner translators are stateless per call — no cross-sentence context unless a candidate adds it explicitly.

## See also

- [current.md](current.md) — active config
- ADR: [NLLB-3.3B supersedes 600M](../../decisions/2026-05-25-nllb-3-3b-over-600m.md)
- `language-support.md` — full list of Armenian-capable translators
