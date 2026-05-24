# References for Phase 1 MVP pipeline

## Phase 0 verdict (drives Phase 1 model choices)

- **Location:** `agent-os/specs/2026-05-24-1430-phase-0-armenian-feasibility/verdict.md`
- **Relevance:** The soft-go decision and the winning candidates (NLLB-200-distilled-600M for translation, Piper `hy_AM-gor-medium` for TTS) are the Phase 1 v1 defaults. Idiom and Russian-narration weaknesses are explicitly accepted into Phase 1.
- **Key patterns:** GPU re-eval order documented there is the same order used to register the deferred candidates in `patmir.toml`.

## Candidate loader contract

- **Location:** `patmir/eval/_loader.py`
- **Relevance:** Phase 1's `[pipeline]` selector reuses `load_candidates(config_path, stage)` to instantiate one candidate per stage by id.
- **Key patterns:**
  - `[[eval.<stage>.candidates]]` TOML tables filtered by `enabled = true`
  - Each candidate module must export `def build(config: dict[str, Any]) -> <CandidateClass>`
  - Loader returns `list[tuple[id, raw_config, built_obj]]`
- **Phase 1 extension:** four new stages (ingest, ASR, mux, output) reuse the exact same loader by adding their own `[[eval.<stage>.candidates]]` tables.

## Translation candidate interface (reused as-is)

- **Location:** `patmir/eval/candidates/translation/nllb_200_distilled_600m.py`
- **Relevance:** Phase 1's translate stage adapter wraps this exact class — no rewrite.
- **Key patterns:** `class NLLBTranslator: def translate(self, text: str, source_lang: str) -> str`. Source-lang code mapping (`en` → `eng_Latn`, `ru` → `rus_Cyrl`, target `hye_Armn`) handled internally. `build(config)` reads `model_id` and `device`.
- **Sibling cloud candidates:** `claude_cloud.py`, `gpt_cloud.py` — same interface, kept disabled by local-first policy.

## TTS candidate interface (reused as-is)

- **Location:** `patmir/eval/candidates/tts/piper_hye.py`
- **Relevance:** Phase 1's TTS stage adapter wraps this exact class.
- **Key patterns:** `class PiperHye: def synthesize(self, text: str) -> bytes` returning WAV bytes (22.05 kHz mono PCM). `build(config)` reads `repo_id`, `onnx_file`, `config_file`; downloads via `huggingface-hub`.

## Eval-run output conventions (mirror in pipeline)

- **Location:** `patmir/eval/translation_eval.py`, `patmir/eval/tts_eval.py`, `patmir/eval/summarize.py`
- **Relevance:** Phase 1 job dirs follow the same `runs/<ts>-<stem>/` pattern as `eval_runs/<ts>/`. Per-stage artifacts are JSON files alongside generated WAVs.
- **Key patterns:** flat per-run directory, JSONL for streamed records, JSON for snapshots, WAV for audio.

## Roadmap and tech-stack (canonical product context)

- **Locations:** `agent-os/product/roadmap.md` §"Phase 1", `agent-os/product/tech-stack.md`
- **Relevance:** Phase 1 implements the exact stages listed there. The MKV output format, sentence-level alignment, and "minimal local UI" decisions all come from these docs.
- **Open correction:** `tech-stack.md` §"Stage 4 — Eastern-Armenian TTS" overstates XTTS-v2's Armenian support — the model card lists 17 languages and Armenian isn't one. Phase 1 Task 3 includes a correction note.
