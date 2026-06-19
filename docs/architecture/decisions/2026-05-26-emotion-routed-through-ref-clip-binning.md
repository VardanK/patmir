# Emotion routed through ref-clip binning

**Date:** 2026-05-26
**Status:** locked (spec in flight — `2026-05-26-emotion-transfer-and-quality-docs`)

## Context

User flagged "emotions are not transferred" as the #3 quality issue (biggest of 5). After investigating, OmniVoice MLX has no explicit emotion knob — see [omnivoice-locked-for-ea](2026-05-26-omnivoice-locked-for-ea.md) for why a TTS swap to an emotion-controllable engine (IndexTTS-2, F5-TTS) is not viable for Eastern Armenian.

Emotion must be expressed through the levers OmniVoice does have:

1. The reference clip's expressive content (the dominant signal — emotion in the ref bleeds into the synthesized output).
2. `guidance_scale` (strength of conditioning).
3. The translated text's emotional register (a flat translation can't be emoted well even with a good ref).

## Decision

A four-part emotion pipeline:

1. **Detect emotion from the source vocal stem**, not from the translated text. The source vocal carries the ground truth; translation rephrasing should not redefine the emotion. New `emotion` stage using `iic/emotion2vec_plus_base` (via `funasr`), run between `diarize` and `asr`.
2. **Bin the 9 SER classes into 5 emotion bins** — `neutral`, `happy`, `sad`, `angry`, `fearful`. The 9-class output (angry/disgusted/fearful/happy/neutral/other/sad/surprised/unknown) collapses via a configurable `[emotion_bins]` table in `patmir.toml`. Confidence below 0.5 maps to `neutral`.
3. **Per-`(speaker, emotion_bin)` reference clips.** `ref_picker` no longer picks one ref per speaker. It picks one ref per `(speaker, bin)` — slicing the longest non-overlapping turn whose dominant emotion matches that bin. Fallback chain (4 steps): exact → `(speaker, "neutral")` → **any `(speaker, *)`** → global default ref. The third step preserves voice *identity* when a speaker only has one emotion bucket's worth of clean clips — without it, sentences in non-covered emotions would drop to the global Piper voice (a different character entirely), which is a regression vs Phase 2's "one ref per speaker" behavior.
4. **Per-emotion `guidance_scale`.** Small lookup table (`neutral=0.7`, `angry/happy=0.9`, `sad=0.6`, `fearful=0.8`), tunable in `patmir.toml`. Not a magic bullet; the dominant signal is still the ref clip.

The Qwen3-Instruct-MLX translator candidate also receives the emotion tag in its user prompt (when active) so the translated register matches. NLLB-3.3B (current default) has no prompt surface; it ignores the field — no behavior change there.

## Consequences

- Emotion fidelity is **bounded by what's audible in the source character's range.** A character who is always calm in the source cannot be made to sound angry. Accepted constraint, documented in the spec's "Risks" section.
- Per-character emotion coverage may be sparse (some characters may only have `neutral` + one other bin). The fallback chain handles this — quietly drops to neutral-for-that-speaker when an emotion has no matching ref turn. Logged in the run manifest so the user can see coverage gaps.
- `Diarization.refs` keying changes from `dict[str, SpeakerRef]` to `dict[(speaker_id, emotion_bin), SpeakerRef]` (or string-keyed `"spk__bin"` for JSON portability). This is a manifest-schema-bump migration; old runs can't `--resume-from` past `diarize`.
- The pipeline gains a new stage — `STAGE_ORDER` grows from 8 to 9 entries.

## Revisit trigger

- OmniVoice exposes an explicit emotion knob in a future release.
- TTS engine is swapped (see the OmniVoice ADR's revisit triggers).
- emotion2vec is replaced by a better SER model (probably also Russian-tuned, since the masha clip is Russian-source and emotion2vec is biased toward EN/ZH training).

## Source

- This conversation (2026-05-26 emotion-transfer investigation).
- `agent-os/specs/2026-05-26-<HHMM>-emotion-transfer-and-quality-docs/plan.md`
- emotion2vec paper: arXiv 2312.15185
- EMO-SUPERB: arXiv 2402.13018

---

## Update 2026-06-19 — emotion-ref binning relaxed (voice consistency won)

Revisit trigger fired (ear-test failure). On masha + a 2nd watch-along clip,
per-`(speaker, emotion)` ref binning made each character's cloned voice **swap
mid-scene** (e.g. `SPEAKER_00` cloned from 4 different short/noisy cartoon
snippets), and the high per-emotion `guidance_scale` (happy 2.8 / angry 3.0)
**over-emoted**. Added `tts_pin_speaker_ref=true` (default): the TTS adapter now
uses ONE ref per speaker (longest/cleanest clip, emotion ignored for ref choice),
and `[emotion_guidance]` was lowered/flattened to 1.3–1.7. Emotion still varies
via guidance but the voice stays consistent. The binning path is preserved
(`tts_pin_speaker_ref=false`). Quality issue #11.
