# Phase 0 — Armenian Feasibility Verdict

> Eval runs:
> - Translation: `eval_runs/20260524-155553/translation/`
> - TTS: `eval_runs/20260524-160019/tts/`
> Scored by the author (fluent Eastern-Armenian speaker), 1–5 rubric per metric, Y/N/maybe for the watchability column.

## Decision

**SOFT GO on Phase 1**, scoped to cartoons and short-form children's content.

The pipeline as designed for the primary use case (children's cartoons in Eastern Armenian, for the author's son) is buildable today using only local models — no cloud APIs required, no GPU workstation required to *start*. The roadmap's hard-requirement — *"would I let a child watch hours of this?"* — was answered Y on **all 20 of 20** TTS utterances across every category, with no `maybe`s. That clears the bar.

The "soft" qualifier reflects two real ceilings exposed during the eval that should be addressed *before* Phase 1 locks in its model choices:
- **Idiom translation is unusable** on NLLB-200-distilled-600M (1.3 / 1.3 / 1.5 means literal calques across the board).
- **Russian-source narration is weak** (mostly 1/2 on meaning_preserved). English-source narration is acceptable.

Neither blocks shipping cartoons — idioms and long-form Russian-source narration are not the dominant content types for the target user — but both will limit what Patmir can dub well, and both are likely to improve significantly with a larger NLLB or MADLAD-400 on the workstation GPU. The expected next step before Phase 1 build-out is therefore the **GPU re-eval**, not cloud opt-in.

## Winning candidates

Target dialect: Eastern Armenian (`hy_AM`). Any candidate that produces Western-Armenian (`hyw`) output is rejected regardless of fluency score.

| stage       | EN → hy_AM                     | RU → hy_AM                     |
|-------------|--------------------------------|--------------------------------|
| Translation | `nllb_200_distilled_600m`      | `nllb_200_distilled_600m`      |
| TTS         | `piper_hye` (`hy_AM-gor-medium`) | `piper_hye` (`hy_AM-gor-medium`) |

Only one local candidate was evaluated per stage on this CPU/Mac pass; the table reflects winners-by-default. The GPU pass below is where real candidate selection happens.

## Quality floor reached

### Translation (1–5)

**NLLB-200-distilled-600M, EN→hy_AM and RU→hy_AM combined, n=40:**

| metric | overall | cartoon | proper | narration | idiom |
|---|---:|---:|---:|---:|---:|
| meaning_preserved | 3.33 | 4.00 | **5.00** | 2.10 | 1.33 |
| register | 3.15 | 3.56 | **4.88** | 2.20 | 1.33 |
| naturalness | 3.12 | 3.38 | **4.50** | 2.60 | 1.50 |

**Notable per-pair split**: RU→hy_AM narration meaning_preserved is 1.20 (vs EN→hy_AM narration at 3.00). The 600M NLLB is meaningfully weaker on Russian-source long-form content than on English-source.

### TTS (1–5, plus Y/N watchability)

**Piper `hy_AM-gor-medium`, n=20:**

| metric | overall | cartoon | proper | narration | idiom |
|---|---:|---:|---:|---:|---:|
| intelligibility | 3.00 | 3.00 | 3.00 | 3.00 | 3.00 |
| naturalness    | 3.00 | 3.00 | 3.00 | 3.00 | 3.00 |
| prosody        | 2.90 | 2.88 | 2.75 | 3.00 | 3.00 |
| child_would_watch_hours | **20/20 Y** | 8/8 Y | 4/4 Y | 5/5 Y | 3/3 Y |

Score uniformity (3/3/3 essentially across the board) is itself a signal: Piper has consistent baseline quality but no standout strengths and no per-category collapse. Acceptable but not delightful — exactly the "watchable for hours but not premium" range.

## Audio samples

Representative WAVs from the chosen TTS candidate:

- Cartoon dialogue: `../../../eval_runs/20260524-160019/tts/piper_hye/hy-cartoon-01.wav`
- Narration:        `../../../eval_runs/20260524-160019/tts/piper_hye/hy-narration-01.wav`
- Idiom:            `../../../eval_runs/20260524-160019/tts/piper_hye/hy-idiom-01.wav`
- Proper noun:      `../../../eval_runs/20260524-160019/tts/piper_hye/hy-proper-01.wav`

Full A/B/C listening page: `../../../eval_runs/20260524-160019/tts/index.html`

## Notable failures and surprises

- **All three EN idioms scored 1/1/1.** NLLB-200-distilled-600M does not handle idioms — it produces literal word-by-word translations that lose the figurative meaning entirely. Russian idioms mostly the same, except `ru-idiom-02` (the bear-skin idiom) which scored 3/3/3 — possibly closer to the surface form of an idiom NLLB has actually seen.
- **Proper nouns surprisingly strong** (5.00 / 4.88 / 4.50). Transliteration of foreign names into Armenian script came out cleanly, which was an open risk going in.
- **Russian-source narration noticeably weaker than English-source narration.** This is worth knowing for content sourcing — Russian dubs of Western cartoons exist and seemed like a useful pivot source, but the translation step is currently the bottleneck for that path. Recommend prioritizing English-source content for early Phase 1 dogfood.
- **No `maybe` watchability votes on TTS** across 20 utterances. That's a stronger signal than the uniform 3/3/3 scores might suggest — there were no edge cases where the author hesitated.
- **No model errors or crashes.** Both candidates loaded on MPS and ran every utterance to completion.

## What changes when the GPU workstation arrives

Run the **second local pass** before Phase 1 commits to translator/TTS choices. Re-eval order, per local-first policy:

1. **Translation, GPU local pass** — flip these to `enabled = true` in `patmir.toml`:
   - `nllb_200_3.3b` (full NLLB, the obvious upgrade from the 600M)
   - `madlad_400`
   - Optionally a prompted local LLM (e.g. Qwen2.5-72B) as a free-form translator
   Re-run `translation_eval`. If idiom and narration scores improve meaningfully, the local stack is enough and cloud stays off.

2. **TTS, GPU local pass** — flip on:
   - `xtts_v2` — verify dialect by ear before scoring; XTTS doesn't distinguish Eastern vs Western and could come out wrong-dialect, in which case it's immediately disqualified.
   - `bark` — same dialect caveat.
   If either matches or beats Piper *and* produces Eastern Armenian, swap the winner in `patmir.toml`. Otherwise Piper stays.

3. **Cloud fallback — only if both local passes fall short for a given stage.** Per `mission.md` local-first policy, do not enable cloud candidates until the local set has been exhausted for that specific stage:
   - Translation: `gpt_cloud` / `claude_cloud` (frontier models, biggest quality jump expected on idioms).
   - TTS: ElevenLabs Multilingual v2 (primary cloud option per updated `tech-stack.md` — supports Armenian + emotion + SSML pauses; significantly above Piper's baseline). Azure neural voices secondary; Google `hy-AM-Standard-A` is roughly Piper-tier, not an upgrade.

4. **If cloud also falls short**, follow the roadmap: pause and wait for the model ecosystem to mature, or scope down to subtitles-with-narration.

## Hardware sizing notes

CPU/Mac pass measurements:

| candidate | model size on disk | runtime on MPS | notes |
|---|---|---|---|
| `nllb_200_distilled_600m` | ~2.4 GB | ~0.5–2 s/utterance | first-token latency dominant; batch would help |
| `piper_hye` (hy_AM-gor-medium) | ~63 MB | ~0.1 s/utterance | trivially fast; CPU-only is fine for this stage |

To be filled in during the GPU pass: NLLB-3.3B, MADLAD-400, XTTS-v2, Bark — per-stage peak VRAM, RAM, and disk numbers per `tech-stack.md`'s hardware-sizing requirement.

## Phase 0 exit

The exit criteria from `roadmap.md` are met:

- ✅ Written one-pager: this document.
- ✅ Winning translator + TTS per language pair (named above).
- ✅ Audio samples attached (paths above; full listening page at the linked `index.html`).
- ✅ Explicit go/no-go on Phase 1: **soft go, scoped to cartoons, with GPU translation re-eval recommended before Phase 1 model selection is locked.**

Phase 1 spec can begin. The Phase 2 voice-variety strategy (preset pool first, cloning as stretch with XTTS-v2 → ElevenLabs ordering) is already locked in `tech-stack.md` Stage 4 and `roadmap.md` Phase 2.
