# Doc drift — files that disagree with `patmir.toml`

Recorded 2026-08-24 while assembling [sota-review-brief.md](sota-review-brief.md).

`patmir.toml [pipeline]` is the single source of truth for which candidate runs each
stage. The files below state something else. **Nothing here has been fixed** — this is a
record, not a changelog, so that a reader who lands on one of these files knows not to
trust its status line.

Every row was verified against `patmir.toml` and, where relevant, the source file. The
maintenance contract in `CLAUDE.md` says these should track the active set; they drifted
because a stage swap updated `current-models.md` and `<stage>/current.md` without
sweeping the per-candidate files and stage `index.md`s.

## The active set, for reference

| Stage | Mac (`[pipeline]`) | CUDA (`[pipeline.profiles.cuda]`) |
| --- | --- | --- |
| separate | `htdemucs_mlx_vocals` | `htdemucs_vocals` |
| diarize | `sortformer_mlx` | `sortformer_nemo` |
| asr | `qwen3_asr_1_7b_mlx_8bit` | `qwen3_asr_1_7b` |
| translate | `openrouter_gemini_flash` | (same) |
| tts | `higgs_v3_mlx_q8` | `higgs_v3_torch_bf16` |

## Stale status claims

| File | Says | Actually |
| --- | --- | --- |
| `stages/tts/index.md` | "**Active (since 2026-05-26)**: openrouter-gemini-flash-tts". Lists OmniVoice + Piper as the local fallbacks and **does not mention Higgs at all** (0 occurrences). | Higgs has been the active TTS on both modes since 2026-06-20. This is the most misleading file in the tree — a reader looking for "what TTS does Patmir use" gets a cloud model retired in May. |
| `stages/tts/elevenlabs-multilingual-v2.md` | "**active** since 2026-05-26 (key scopes broadened — IVC permission added)" | Registered, inactive since 2026-05-27 PM (captcha blocked cloned celebrity voices), and again from 2026-06-15 (credits exhausted). |
| `stages/translate/index.md` | "**Active**: nllb-200-3-3b — active default since Phase 1.3" | `openrouter_gemini_flash` since 2026-05-26. NLLB-3.3B is the offline fallback. (`stages/translate/current.md` is correct — only the index disagrees.) |
| `stages/asr/whisper-mlx-large-v3.md` | "**active** (Phase 1.3 → 2026-05-26 morning …)" | Superseded 2026-07-14 by `qwen3_asr_1_7b_mlx_8bit`; registered fallback. |
| `stages/asr/nemotron-mlx.md` | "**active** (2026-06-15) — replaced cloud" | Rejected 2026-06-19 (garbled fast cartoon Russian); registered as a fast alternate. |
| `stages/asr/qwen3-asr-transformers.md` + `stages/asr/index.md` | "Never executed — written and unit-tested, first CUDA run pending" | Executed end to end on the 3090 on 2026-08-24 and baselined: 20.53 s warm, 14.8 % of batch wall, 5.51 GB resident. |
| `stages/diarize/pyannote-3-1.md` | "**active**" | Superseded 2026-06-15 by `sortformer_mlx`; registered fallback. Also: the filename and the candidate id are both misnomers — the `model_id` it loads is `pyannote/speaker-diarization-community-1`, not `speaker-diarization-3.1`. |
| `stages/separate/htdemucs.md` | "**active**" | The MLX vocals-only candidate is active on the Mac; the torch twin `htdemucs_vocals` on cuda. Plain `htdemucs` is a fallback. |
| `stages/separate/index.md` | "**htdemucs-mlx** — active (2026-06-15)" | `htdemucs_mlx_vocals` since 2026-06-19 (vocals-specialist sub-model only). The full ft bag is a registered fallback; the index does not mention the vocals-only variant. |
| `stages/tts/current.md` | Describes `higgs_v3_mlx_q8` for both modes — correct for the Mac | Incomplete rather than wrong: it does not mention the cuda profile's `higgs_v3_torch_bf16`. |
| `language-support.md` | OmniVoice MLX (q8) "**Active**" for tts; `facebook/nllb-200-3.3B` "**Active**" for translate | Higgs + Gemini Flash. The rest of the file (the EA gate, the dialect-ambiguity policy, the rejected-on-dialect tables) is still accurate and is the authoritative statement of the Eastern-Armenian constraint. |
| `docs/architecture/index.md` | Its `stages/` tree annotates `diarize/ (pyannote 3.1)`, `asr/ (mlx-whisper large-v3 + alternates)`, `translate/ (nllb-3.3b + …)`, `tts/ (omnivoice-mlx-q8 + …)` | All four are superseded. The annotations describe the Phase-1.3 set. |

## Accurate, noted to prevent a redundant "fix"

- **`stages/asr/index.md`** correctly records `qwen3-asr-mlx` as active on the default
  profile *and* records the unresolved contradiction (a 2026-07-14 eval note called the
  transcripts nonsense to a native speaker; the flip verdict was "both garble sometimes,
  pick the fastest"; the ear-test is still pending). Only its "Never executed" line about
  the transformers candidate is stale.
- **`stages/diarize/index.md`** and **`stages/translate/current.md`** are correct.
- **`decisions/2026-08-23-cuda-port-machine-profiles.md` and the `sortformer_nemo`
  docstring** were both **already corrected** on 2026-08-24 — they now say
  `nemo_toolkit 3.0.0` installs cleanly from PyPI with no git checkout and no Cython
  prerequisite, and the docstring explicitly flags the old claim as wrong. Listed here
  only because the earlier text is memorable enough to be repeated from memory.

## Non-file discrepancies

- **Three TTS candidates point at a directory that does not exist.** The OmniVoice entries
  at `patmir.toml:392`, `:488` and `:505` set
  `ref_audio = "eval_runs/20260524-160019/tts/piper_hye/hy-cartoon-01.wav"`. There is no
  `eval_runs/` in the repo (only `runs/`), so those eval defaults are dead paths. Harmless
  in the pipeline — the adapter supplies real per-speaker refs — but a smoke-test of a
  registered OmniVoice candidate would fail on it.

- **The canonical clip's language is contested.** `performance.md`'s Apple-Silicon rows
  describe `sample_1.mkv` as "English, dialogue-dense live action"; the file on disk has
  exactly one audio track and it is **Russian**, which is what the CUDA baselines
  measured. Either the Mac had a different copy or the label was always wrong. This is
  unresolved and it matters: ASR and TTS cost both scale with what is actually spoken, so
  the two machines' per-stage numbers may not be directly comparable. Noted in
  `eval-clips.md`.

- **Command lines in pre-2026-08-24 docs fail as written.** The canonical clip was renamed
  from `Reacher-S02E01-5min.mkv` to `sample_1.mkv`; older docs use the old filename in
  runnable commands.
