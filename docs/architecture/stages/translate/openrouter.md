# OpenRouter (translate — cloud)

- **Repo / version:** `openrouter.ai` — OpenAI-compatible chat-completions gateway
- **Apple Silicon / CUDA:** n/a (cloud)
- **Languages:** broad — depends on the chosen routed model
- **License / gating:** OpenRouter ToS + per-token cost (variable by model)
- **Download source:** n/a (cloud API; no model download)
- **API key:** read from `OPENROUTER_API_KEY` env var. Sign up at openrouter.ai → create a key → `export OPENROUTER_API_KEY=sk-or-...`. The candidate raises a clear error if the env var is unset.
- **Integration surface:** `patmir/eval/candidates/translation/openrouter.py`
  ```python
  translate(text: str, source_lang: str, emotion: str | None = None) -> str
  ```
  Same shape as `qwen3_instruct_mlx` — emotion tag injected into the user message when set; SentenceTranslator adapter forwards it via signature probing.
- **Status in pipeline:** **active** (Gemini 2.5 Flash variant) as of 2026-05-26
- **Last evaluated:** 2026-05-26 (initial enablement)
- **Our experience:** TBD — to be filled after the masha-clip re-run.
- **Notes:**
  - Same system prompt as Qwen3 ("preserve tone, register, and emotion"). The cloud LLM follows instructions much more reliably than NLLB (which has no prompt surface at all) or Qwen3 (which is fine but smaller).
  - **No new Python deps** — uses `httpx` already in the venv via huggingface_hub / fastapi.
  - Per-sentence HTTP round-trip is 200-800 ms; ~comparable to local NLLB-3.3B's per-sentence latency on MPS.
  - OpenRouter rate-limits per-key; the default is generous for our use volume but a 20-minute episode (~200 sentences) is well within free-tier-style budgets.

## Registered models

| Candidate id                 | OpenRouter model                | Status   | When to switch                                              |
| ---------------------------- | ------------------------------- | -------- | ----------------------------------------------------------- |
| `openrouter_gemini_flash`    | `google/gemini-3.5-flash`       | **active** | Default — cheap + competitive on multilingual. 3.5 picked over 3.1-flash-lite 2026-05-26 evening A/B (more emotive output). |
| `openrouter_claude_opus`     | `anthropic/claude-opus-4.7`     | alt      | Strongest on register / nuance; most expensive.             |
| `openrouter_gpt5`            | `openai/gpt-5`                  | alt      | Comparable to Claude; cheaper. Slightly weaker on Eastern Armenian. |
| `openrouter_deepseek_v4_pro` | `deepseek/deepseek-v4-pro`      | rejected (2026-05-27) | Cheap on paper ($0.435/M in, $0.87/M out — ~3× Gemini output cost). Benchmarked on masha-2-5min: **4/10 chain-of-thought leaks into the `content` field even with `thinking={type: "disabled"}`** (DeepSeek-native passthrough; the model ignores OpenRouter's `reasoning.effort=minimal`). **Avg latency 10.9 s/sentence**, ~10× Gemini 3.5 Flash. Production-unusable for the per-sentence translate loop until DeepSeek fixes the no-thinking-mode content-leak bug. |

To swap, change `[pipeline].translate` in `patmir.toml`. To add a new model, copy a `[[eval.translation.candidates]]` block and change `id` + `model`.

## Failure semantics (2026-09-15)

| Response | Per-sentence candidate | Batched candidate |
| --- | --- | --- |
| 200 | text (a `content: null` becomes `""` — a per-line refusal, reported by the adapter) | parsed; idx mismatch / empty text -> `_BatchError` -> per-sentence fallback for that chunk |
| 408 / 409 / 425 / 429 / 5xx / transport | retried (`max_attempts`, exponential `retry_backoff_s`) | same, then `_BatchError` -> fallback |
| **401 / 402 / 404** | **`TranslatorUnavailable`**, first response, no retry | **`TranslatorUnavailable`**, not a `_BatchError`, so **no** per-sentence fallback; the fallback's `_one` also re-raises it |
| 403 (provider moderation flagged the input) | `RuntimeError` on that line -> in the batched fallback, `""` for that line | `_BatchError` -> fallback -> `""` for the flagged line |
| other 4xx (400) | `RuntimeError` | `_BatchError` -> fallback |

`TranslatorUnavailable` ends the stream or job `failed` with the one-line message as `error`
(`OpenRouter: HTTP 402 from google/gemini-3.5-flash — credits exhausted (...)`). Two days of
402s in 2026-09 had instead produced sections with every line empty and every stage `done`
(quality issue #35). Watch the account balance: there is no preflight probe by decision.

## ADR

[2026-05-26 — Cloud translate via OpenRouter (Gemini 2.5 Flash default)](../../decisions/2026-05-26-cloud-translate-via-openrouter.md)
[2026-09-15 — A translator outage must fail the dub, not mute it](../../decisions/2026-09-15-translator-outage-fails-loudly.md)
