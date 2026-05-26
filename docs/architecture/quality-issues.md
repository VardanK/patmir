# Quality issues — live tracker

Observed problems with the current pipeline output, with owner stage, status, and which spec is tackling them. Updated whenever a fix lands or a regression is observed.

Status values:

- `open` — observed, not yet scheduled
- `in-progress: <spec>` — actively being worked on
- `addressed (<date> — spec: <spec>)` — fixed; ear-verified on a representative clip
- `partial (<date> — spec: <spec>)` — fix shipped but didn't fully resolve; deeper work scheduled
- `regression-watch` — was addressed, but a regression is suspected

## Active issues

| # | Title                                | Owner stage | Status                                                                                          | Notes                                                                                                                       |
| - | ------------------------------------ | ----------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 1 | Voice cloning fidelity inconsistent  | tts         | `open` (likely improved by #3 fix)                                                              | Reported on 2026-05-26 by author after listening to masha-2-5min output. Theory: many "wrong character" reports are actually emotion-mismatched refs, not identity errors. Re-eval after #3 lands. |
| 2 | Translation quality not there yet    | translate   | `open`                                                                                          | NLLB-3.3B is current default. Qwen3-Instruct-MLX is registered. Likely needs OpenRouter (frontier-model) comparison ceiling before a real verdict. |
| 3 | **Emotions not transferred**         | tts (+emotion stage) | `in-progress: 2026-05-26-emotion-transfer-and-quality-docs`                            | The biggest user-flagged issue. Adding a new `emotion` stage (emotion2vec) + per-`(speaker, emotion)` ref-clip selection + emotion-aware `guidance_scale`. |
| 4 | Original voice leaks at sentence boundaries | mux  | `in-progress: 2026-05-26-emotion-transfer-and-quality-docs`                                     | Folded into spec #3. Fix attempt: per-sentence 50 ms afade in/out on synthesized track. Bumped to its own spec if the simple fix doesn't land. |
| 5 | Overlap handling unclear             | mux (+diarize) | `open`                                                                                       | Pyannote detects overlap intervals; downstream handling of simultaneous speakers is not defined. Needs its own spec — likely involves ducking the lower-priority voice or channel-separating. |

## Closed issues

(none yet — keep a section here so closed entries don't crowd the active list)

## How to keep this file useful

- When a spec starts, set the relevant rows to `in-progress: <spec-folder-name>`.
- When verification passes, flip to `addressed (<date> — spec: <spec>)`.
- When a partial fix ships, flip to `partial` and add a note explaining what's left.
- If a closed issue regresses, change the status to `regression-watch` and add a note with the reproducing case.
- Add new rows whenever a quality observation is logged (commit messages, dogfood notes, ear-tests).
