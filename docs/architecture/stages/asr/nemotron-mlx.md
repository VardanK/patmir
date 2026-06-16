# nemotron-mlx (asr)

- **Repo / version:** `mlx-community/nemotron-3.5-asr-streaming-0.6b`
- **Library:** `mlx-audio` (`mlx_audio.stt`) — `github.com/Blaizzy/mlx-audio`
- **Architecture:** NVIDIA cache-aware streaming FastConformer-RNNT (0.6B)
- **Parameters / size:** 0.6B; ~1.2 GB download
- **Apple Silicon (MLX):** ✅ first-class. CUDA/CPU: n/a (MLX port)
- **Languages:** 40 language-locales incl. English + Russian; auto-detect or
  forced via a `prompt_dictionary` locale key (`en-US`, `ru`, …)
- **License / gating:** non-gated HF download
- **Integration surface:** `patmir/pipeline/stages/asr/nemotron_mlx.py`
- **Status in pipeline:** **active** (2026-06-15) — replaced cloud
  `elevenlabs_scribe_v2` after ElevenLabs credits were exhausted. ADR:
  `decisions/2026-06-15-local-asr-diarize-via-mlx-audio.md`.
- **Last evaluated:** 2026-06-15 (probe + 90s end-to-end run).

## Output contract

`generate(path)` → `AlignedResult`:
- `.text: str`
- `.sentences: list[AlignedSentence]`, each with `.text`, `.start`, `.end`,
  `.duration`, and `.tokens: list[AlignedToken]`
- `AlignedToken`: `.id`, `.text`, `.start`, `.duration`, `.end` — sentencepiece
  subwords with a leading-space word boundary (`' '`,`'R'`,`'ight'`,`' '`,`'f'`…)

The candidate merges subwords into whitespace words (`_tokens_to_words`), then
**re-segments from the timed words** (`_segment_words`) rather than trusting
Nemotron's own `AlignedSentence` boundaries, applies `is_non_lexical` + VAD
post-filter, and sets `ctx.source_lang` from a script-based language guess.

### Why we re-segment (no-punctuation caveat)

Nemotron 3.5 streaming emits **no sentence punctuation**, so its `AlignedSentence`
boundaries collapse continuous narration into one giant run-on per audio chunk.
Unsplit, that run-on (a) is dropped by the batch translator (returns empty → no
dub, you hear the original) and (b) can't be placed by the mux. `_segment_words`
splits on terminal punctuation (when present), a word gap > `max_gap_s`, **or**
hard caps on duration / word count — the caps are load-bearing here. Surfaced
2026-06-15 on watch-along of a narration video (one 683-char "sentence",
empty translation); fixed by re-segmentation.

## Our experience

- **Timestamps:** probe confirmed full token+sentence timing despite the model
  card only documenting `.text`. No forced aligner needed.
- **Speed:** ~0.03–0.05× realtime on a 45s clip (well under realtime).
- **Language:** auto-detect picks EN/RU correctly. **Forcing the wrong locale
  (e.g. `ru-RU` on English audio) produces phonetic Cyrillic gibberish** — so we
  auto-detect and never force unless the source language is known.

## Knobs (`[[stages.asr.candidates]]`)

| Key | Default | Effect |
| --- | --- | --- |
| `repo_id` | — | HF repo |
| `language` | `"None"` | `None`/`auto` = auto-detect (recommended); else a prompt-dict locale key |
| `vad_filter` | `true` | Drop sentences whose words fall entirely outside VAD speech |
| `max_gap_s` | `0.6` | Split a sentence when the gap to the next word exceeds this |
| `max_sentence_seconds` | `10.0` | Hard cap: split a sentence at this duration even mid-clause (no-punctuation guard) |
| `max_sentence_words` | `40` | Hard cap: split a sentence at this word count |

## Revisit

If Russian (or any source) quality disappoints on a representative clip, flip
`[pipeline].asr = "whisper_large_v3_mlx_fp16"` (the offline fallback, which has
first-class word timestamps).
