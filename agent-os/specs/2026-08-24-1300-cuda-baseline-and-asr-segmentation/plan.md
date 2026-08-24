# Make the CUDA pipeline correct, then benchmark it

## Context

The project moved from the 32 GB Apple Silicon laptop to an RTX 3090 / WSL2 box.
`decisions/2026-08-23-cuda-port-machine-profiles.md` landed the `cuda` machine
profile plus four torch candidates (`htdemucs_vocals`, `sortformer_nemo`,
`qwen3_asr_1_7b`, `higgs_v3_torch_bf16`), all documented as **UNMEASURED / never
executed**. That is no longer true, and nothing has been written down.

The intent was to start benchmarking. The survey found a correctness bug that has
to be fixed first, because fixing it changes the sentence count and the TTS chunk
distribution — any baseline measured before it would be stale on arrival.

### The blocking defect: "sentence" means "28-second chunk"

`patmir/pipeline/stages/asr/qwen3_asr.py:382-388` appends one `Sentence` per ASR
chunk and never calls `group_into_sentences` from
`patmir/pipeline/stages/asr/_sentence.py:208`, whose whole job is splitting words
into sentences on terminal punctuation **or** a word gap over `max_gap_s` (0.6 s).
`whisper_mlx`, `faster_whisper`, `elevenlabs_scribe` and `openrouter_stt` all call
it; **both** Qwen3 candidates (`qwen3_asr.py` and `qwen3_asr_mlx.py`) skip it.

Measured on `runs/20260824-022033-sample_1`:

- **23 sentences for 301 s** of video against diarization's **108 turns**.
- **Zero punctuation** across all 398 words, so nothing downstream can re-split.
- Sentence spans mean 9.7 s, median 8.6 s, **max 26.3 s**.
- The runner's `apply_speaker_tags` assigns one *dominant* speaker per sentence, so
  a span covering several turns is dubbed in **one character's voice**. The
  14.3–27.4 s "SPEAKER_02" line contains a mother comforting a child *and*
  someone saying "call the cops, give them this".
- It also produces the 44–55 s TTS decode chunks and the ref clips whose
  `ref_text` reads like several people talking.

This is the 2026-07-14 ASR flip silently redefining "sentence".
`performance.md` recorded the symptom ("Qwen3 chunks ~3.5× coarser — 25 vs ~104
sentences per 5 min") and filed it as a TTS-tail *performance* follow-up. It is a
quality bug, it affects **both machines**, and the CUDA port only made it visible.
`faster_whisper_large_v3_fp16` is registered on the cuda profile and does group
properly, which gives a same-box A/B to confirm the mechanism.

### Everything else the survey established

- **Batch runs end to end on CUDA.** `runs/bench/reacher-cuda-batch.json`: 3
  repeats on `samples/sample_1.mkv` (301.11 s), agg RTF 0.904 / 0.815 / 0.803, all
  9 stages `done`. Diarization is healthy — `sortformer_nemo` gave 4 speakers, 108
  turns, 33 derived overlaps, and the NeMo return-shape parser did not raise, so
  **quality issue #13 is resolved in practice**. TTS: `overflows: 0`, `batches: 3`.
- **The profile resolves correctly in both modes** (verified): every id is the
  profile's, no MLX leaks. `streaming_config` does not override `tts`, so
  watch-along inherits `higgs_v3_torch_bf16`.
- **Those RTFs are not the warm baseline the docs compare against.** `bench_batch`
  (`patmir/bench.py:195-218`) calls `run_job` with **no `stage_cache`**, so every
  repeat rebuilds every model; `bench_watch_along` (line 266) already shares one and
  its comment says "bench must measure the warm path". Cold builds are ~110 s of a
  242 s run — **45 % of wall** (asr `build_s` 42–48 s, tts 50–62 s). Subtracting
  that gives ≈0.44, near the doc's ~0.35 projection — so the bottleneck model was
  roughly right and the report was measuring nine model loads per repeat.
- **Part of the build tax is filesystem, but only part.** `HF_HOME` is
  `/mnt/e/models_cache` on a **9p** mount: **~360 MB/s** warm, **~176 MB/s** cold,
  with no sensitivity to block size above 1 MB (`msize=65536`) — transport-bound,
  not disk-bound. The ext4 root reads **3.4–4.0 GB/s** under O_DIRECT and is a vhdx
  on **C:**, not the D: HDD. At the warm rate Higgs' 8.7 GB is ~24 s of read against
  a 50–62 s `build_s`, and Qwen3-ASR + aligner (5.9 GB) ~16 s against 42–48 s — so
  roughly **half** the tax is I/O and the rest is parse + host→device copy +
  `from_pretrained` init. C: has only 51 GB free against 27 GB of models; E: has 152.
- **Watch-along has no baseline** — both prelude runs were author interrupts.
- **`uv.lock` is stale and dangerous.** It predates the `[cuda]` extra entirely,
  still carries `transformers>=4.40`, has no `nemo-toolkit`, and no platform markers
  on the MLX deps. A `uv sync` could re-resolve transformers below 5.13 and
  reproduce the exact breakage commit `f6e367d` was written to prevent. Also
  `.python-version` says 3.11 against a 3.12.14 venv.
- **Clip references drifted.** `samples/sample_1.mkv` *is* the Reacher clip
  (301.113 s, matching the docs' 301.1 s), but `sample_1` appears in no tracked
  file while every doc command names the absent `samples/Reacher-S02E01-5min.mkv`.
  `masha-2-5min.mkv` (deliberately retired — poor translation candidate) and
  `Fallout-S02E01-5min.mkv` are gone, yet `performance.md` still calls masha a
  "canonical clip". The clip's only audio track is **Russian** (video stream tagged
  `eng`), so the "English, dialogue-dense" label on the Mac's Reacher rows needs
  checking before the two baselines are compared.

**Outcome:** a dub that is correct on this box, then a trustworthy warm baseline for
both execution modes, then docs that match reality.

---

## Task 1 — Save spec documentation

Create `agent-os/specs/2026-08-24-1300-cuda-baseline-and-asr-segmentation/`:

- `plan.md` — this plan
- `shape.md` — scope, the shaping decisions below, and the survey findings above
- `standards.md` — records that `agent-os/standards/index.yml` holds only a header,
  so no standards apply
- `references.md` — `patmir/pipeline/stages/asr/_sentence.py:208`
  (`group_into_sentences`) and the four candidates that call it;
  `patmir/bench.py` (the batch/watch-along cache asymmetry);
  `patmir/pipeline/runner.py:55-89,175-200`; `docs/setup/windows-wsl2-cuda.md`
  Phase 9; `agent-os/specs/2026-06-20-1656-pipeline-benchmark-perf/`
- no `visuals/` — none provided

Shaping decisions confirmed with the author:

1. **Correctness before benchmarking.** Make the pipeline produce a good dub first;
   a segmentation fix moves sentence count and TTS distribution, so any earlier
   baseline is void.
2. `transformers` floor must be **>= 5.13** — below that the pipeline fails.
3. `HF_HOME` stays on the 9p mount; measure the I/O-vs-init split only. A 980 PRO
   is going in shortly and will change that decision.
4. The `max_batch` sweep counts as measurement, not optimization. **The VRAM
   setting is decided by the sweep, not chosen up front** — see Task 6.
5. Docs adopt `samples/sample_1.mkv`; a clip registry carries provenance. Nothing
   on disk is renamed.
6. masha cleanup touches live docs only; ADRs and closed specs stay as history.

## Task 2 — Raise the `transformers` floor to >= 5.13 and fix the lockfile

- `pyproject.toml` — the `local` extra pins `transformers>=5.5`; raise it to
  **`>=5.13`**. That is the floor `qwen3_asr` needs for transformers' native
  `qwen3_asr` module, and it still satisfies the Higgs checkpoint's own `>=5.5`.
  The `cuda` extra already says `>=5.13`; keep it as documentation of *why*.
- `uv.lock` — regenerate so it stops advertising `transformers>=4.40` and starts
  carrying the `cuda`/`dev` extras and `nemo-toolkit`. **Verify the resolved
  transformers is >= 5.13 and NeMo is present before committing it, and do not
  `uv sync` into the working venv** — the venv was built with `uv pip install` and
  currently holds a good set (`torch 2.11.0+cu128`, `transformers 5.15.1`,
  `nemo-toolkit 3.0.0`). If a clean re-resolve proves messy, deleting the stale
  lock is better than keeping one that lies.
- Leave the runtime gate in place as the backstop: `_check_transformers`
  (`patmir/pipeline/stages/asr/qwen3_asr.py:117-137`, `MIN_TRANSFORMERS=(5,13)`)
  already fails loudly with a useful message.
- **Risk to flag, not resolve here:** raising the shared `local` extra also raises
  the floor on the Mac, and no MLX package's compatibility with transformers ≥ 5.13
  has been checked from this box. Note it in the change so the Mac's next install
  is verified rather than assumed.
- Do not touch `.python-version` in this spec — correcting 3.11 → 3.12 invites a
  `uv venv` recreate that would orphan the working install. Note it as a follow-up.

## Task 3 — Fix ASR sentence segmentation (the correctness gate)

- `patmir/pipeline/stages/asr/qwen3_asr.py:355-388` — instead of one `Sentence` per
  chunk, run each chunk's VAD-filtered `chunk_words` through the existing
  `group_into_sentences(...)` and extend the transcript with the result, renumbering
  `idx` across chunks. The helper already handles the no-punctuation case by
  splitting on word gaps, which is exactly this transcript's situation.
- Apply the same fix to `patmir/pipeline/stages/asr/qwen3_asr_mlx.py` — the defect
  is shared, and leaving the Mac on chunk-sentences would make the two baselines
  incomparable for the wrong reason.
- **Split at speaker-turn boundaries too.** Gap-splitting alone will not separate
  two characters in rapid exchange with no silence between them. Diarization already
  supplies 108 turns via `ctx.extra["diarization"]`; a sentence spanning a turn
  boundary should be split there rather than handed a dominant speaker. Decide
  whether that belongs in the candidate or in the runner's `apply_speaker_tags`
  (`patmir/pipeline/stages/asr/speaker_tag.py`) — the latter is shared by every ASR
  candidate, which argues for putting it there.
- Tune `max_gap_s` against this clip rather than accepting the 0.6 s default
  unexamined; report the resulting sentence count. For reference, Whisper produced
  ~104 sentences per 5 min on masha and Scribe-era runs were in that range.
- Tests: cover chunk→multi-sentence splitting, the no-punctuation path, `idx`
  continuity across chunks, and a sentence spanning two speaker turns being split.
  Extend the existing `patmir/tests/test_qwen3_asr_torch.py`.

## Task 4 — Verify the dub is actually good (gate before any benchmarking)

No timings in this task. Re-run one batch job and judge the output.

- Ear-test `runs/<job>/out.mkv`: each character keeps one recognizable voice, no
  character speaking another's lines, no original dialogue leaking, Armenian
  default audio, subtitles aligned.
- Confirm the numbers moved the right way: sentence count up from 23 toward ~100,
  max span well under 26.3 s, per-sentence speaker assignment consistent with
  `diarize/turns.json`, `overflows: 0` in `tts/voice_cloning_summary.json`.
- Inspect `diarize/refs/*/*.wav` and their `ref_text` — a ref whose text reads like
  several people talking means the fix did not take.
- If quality is still wrong, A/B against `faster_whisper_large_v3_fp16` (already
  registered on the cuda profile, and it groups sentences properly) to separate a
  segmentation problem from a Qwen3 transcription problem.
- **This task gates Tasks 5–8.** If the dub is not good, fix that first; the
  benchmark is worthless against a broken pipeline.

## Task 5 — Make `bench_batch` measure the warm path

- `patmir/bench.py:195-218` — `bench_batch(..., warm_cache: bool = True)`; create one
  plain `stage_cache = {}` **above** the repeat loop and thread it into every
  `run_job`, so repeat 0 is cold and repeats 1..N-1 are warm (a cache hit sets
  `st.build_s = 0.0`, `runner.py:186-188`). Clear it in a `finally` so ~15 GB of VRAM
  is not pinned while the report is written. Do **not** use `prewarm_stage_cache`
  here: it would zero repeat 0 too, destroying the cold datapoint, and it swallows
  build failures (`runner.py:84-86`) — a benchmark must fail loudly. No runner
  refactor.
- Give `bench_watch_along` the same parameter but **keep its cache creation inside
  the repeat loop**. Hoisting it would let repeats 1+ skip the section-0 builds and
  break comparability with the stored 1.71 / 1.611 / 1.47 rows.
- Add `--no-warm-cache` on `patmir bench` (`patmir/cli.py` after :108) so the old
  behaviour stays reachable and N independent cold samples are obtainable. Prefer
  that name over `--cold`, which reads like "drop the page cache". Reword
  `--repeat`'s help to say repeat 0 is cold and the rest warm.
- Store `warm_cache`, `warm`, `build_total_s` and `profile` on `BenchResult`, all
  defaulted (it is serialized with `asdict`, and the stored Mac reports in
  `agent-os/specs/2026-06-20-1656-pipeline-benchmark-perf/results/` must still
  load). Make `warm` *measured* — `warm_cache and build_total_s <= 0.05` — so the
  JSON says what happened. `PipelineConfig.profile` exists
  (`patmir/pipeline/config.py:67`) with a comment that it is there "so a bench
  report can record what actually ran", but `BenchResult` never carried it.
- Record `peak_vram_gb` / `peak_vram_reserved_gb` / `vram_free_gb` / `peak_rss_gb`
  in `notes`, via a helper that returns `{}` on any exception — never fail a
  four-minute benchmark on a telemetry call. `peak_rss_gb` guards the swap trap that
  produced the MLX 1.71 → 3.02 regression on a 23 GB host.
- `format_report` labels each repeat (`repeat=0 (cold: builds models)` /
  `repeat=1 (warm)`) and never averages cold with warm — the whole point.
- `_tts_stats` (`patmir/bench.py:123-148`) — surface the row unit. Under batched
  decode `tts/timings.jsonl` rows are per **chunk** (`"kind": "chunk"`,
  `patmir/pipeline/stages/tts/adapter.py:286-293,361-396`), so the current report
  reads `count: 3` beside `sentences: 24` with `p90_s: 51.2` — chunk numbers a
  reader takes for the per-sentence p90 the Mac rows quote. Add `unit`, `chunks`,
  `chunk_sentences`, `sentences_per_chunk` and make `format_report` unit-aware
  (`p90 51.20s/chunk`). Two constraints: the new keys go **above** the
  `voice_cloning_summary.json` merge (:139-147), and none may be named `sentences`
  or the per-job total loses that key. Leave `total_s` / `audio_s` / `tts_rtf`
  alone — already correct under batching. Do not invent per-sentence walls.
- Watch-along `agg_rtf` uses `dubbed_audio_s` (`patmir/bench.py:296-298,328`) while
  batch and the stated convention (`performance.md:22-23`) use source seconds. Keep
  `agg_rtf` and add `notes["rtf_denominator"]` + `notes["agg_rtf_source"]`, printing
  both. Reflooring the primary field would invalidate every historical watch-along
  row with no way to tell which convention a row used.
- Two accepted side effects of warm caching across repeats, both timing-neutral and
  both already true of the UI server: `HiggsV3Torch._ref_cache`
  (`higgs_v3_torch.py:252`) is keyed by absolute path so it never hits across
  repeats; `TTSAdapter._standin` (`adapter.py:94-95,192-223`) is keyed by
  `speaker_id`, which does repeat, so a ref-less speaker on repeat 1 can be handed a
  stand-in from repeat 0's job dir. Note it; the fix belongs in the adapter.
- Update `patmir/bench.py`'s module docstring (:1-18) — it describes the semantics
  this change replaces.
- Tests: nothing covers `patmir/bench.py` today, so add `patmir/tests/test_bench.py`
  (~19 tests). Synthetic manifests + `tmp_path` in the style of
  `test_stage_cache.py`, with a `run_job` stub honouring the cache contract
  (`build_s = 0.0` when the passed dict already holds the key) rather than keying on
  call index. Groups: shared-cache identity and `--no-warm-cache` passing `None`;
  repeat 0 cold / repeat 1 warm; the default staying `True`; cache released on
  return; backward compatibility by rehydrating every stored report JSON (skip if
  absent so a bare checkout stays green); TTS unit reporting including `sentences`
  and `sentences_per_chunk` coexisting — the exact confusion in the current report;
  report labelling; both watch-along RTF denominators; CLI wiring of the new flag.

## Task 6 — Batch baseline, and let the sweep decide `max_batch`

```bash
patmir bench samples/sample_1.mkv --mode batch --repeat 3 \
  --report runs/bench/sample_1-cuda-batch.json --label cuda-baseline
```

After Task 5 this one command yields both numbers: repeat 0 cold (comparable to
today's 0.904) and repeats 1–2 warm (the new baseline). Report separately, never
averaged. Run it in an uninterrupted shell — `performance.md:342-346` explains why a
suspended run yields impossible per-stage rows.

**On the VRAM question — nothing to decide up front.** The estimate is ~14.5 GB
resident (Higgs 8.5–9.5, asr + aligner ~4.6, the rest under 1.5) peaking ~18–21 GB
against ~22.7 GB usable after WSL's 1.3 GB baseline, so `max_batch = 8` should fit
and 16 may not. The sweep answers it empirically:

```bash
patmir bench samples/sample_1.mkv --mode batch --repeat 2 \
  --report runs/bench/sample_1-cuda-b<N>.json --label "max_batch=<N>"
```

for N in 4 / 8 / 16, hand-editing the `higgs_v3_torch_bf16` entry between runs.
Decision rule: pick the largest N whose warm TTS wall improves and whose
`vram_free_gb` stays above ~2 GB. An OOM at 16 is a result, not a failure — record
it. Then leave the winner in `patmir.toml` and replace the "conservative guess from
memory arithmetic, not a measurement" comment at `patmir.toml:456-459` with the
measurement. Also verify the Task 3 fix in the timings: sentence count up, TTS p90
per chunk down.

## Task 7 — Watch-along baseline

```bash
patmir bench samples/sample_1.mkv --mode watch-along \
  --report runs/bench/sample_1-cuda-watch.json --label cuda-baseline
```

Never completed on this box, so run it attended and let it finish. The two earlier
preludes were interrupts, but they stopped at the identical point ten hours apart
with no error recorded — if an attended run stops there too, that is a real fault
and this task becomes a debugging task. Baselines to beat (warm M1 Pro, same clip):
batch **1.17**, watch-along **1.611**; the doc projects ~0.35 / ~0.51, explicitly
"a projection from memory bandwidth — measure it, do not assume it". Note per-section
RTF and the rebuild tax (`notes.total_build_s`).

## Task 8 — Attribute the model-build tax (no cache move)

Measure, don't move; the pending drive swap makes relocation premature.

- Time one build with the phases separated: read the checkpoint files to `/dev/null`
  (I/O alone) at the current `HF_HOME`, then copy just the Higgs and Qwen3-ASR
  snapshots to ext4 and time the same build there. ~15 GB fits C:'s 51 GB free
  without committing to the full 27 GB. Delete the ext4 copies afterwards.
- Record the split per stage: how much of asr's 42–48 s and tts's 50–62 s is file
  read versus parse + host→device copy + init.
- `HF_HOME` stays on E:. Note the deferral explicitly so a later reader knows it was
  a decision, and that this is the number to re-check when the 980 PRO lands.
- Document in `docs/setup/windows-wsl2-cuda.md`: 9p is transport-bound at ~360 MB/s
  warm / ~176 MB/s cold regardless of the underlying SSD, versus 3.4–4.0 GB/s on the
  ext4 vhdx (which lives on C:, not the D: HDD), plus the free-space tradeoff.

## Task 9 — Clip registry and reference cleanup

- New `docs/architecture/eval-clips.md`: one row per clip —
  `samples/sample_1.mkv` (Reacher S02E01, first ~5 min, **Russian** audio,
  301.113 s, ~60 % speech, 4 speakers; the canonical clip),
  `samples/clip-en-test.mkv` (7.18 s, English, fixed-cost sanity only), and a
  removed section for `masha-2-5min.mkv` (retired) and
  `Fallout-S02E01-5min.mkv`. State plainly that measurements citing removed clips
  are historical and not reproducible.
- Rewrite the runnable commands to `samples/sample_1.mkv`:
  `docs/architecture/performance.md:508-513`,
  `docs/setup/windows-wsl2-cuda.md:243-246,282-289`. Align the report filename the
  WSL doc prescribes with what is written.
- Verify whether the Mac's Reacher rows really were English before comparing RTFs;
  the clip on disk has a single Russian track. Correct the label if wrong and say
  which language each baseline measured.
- Fix `performance.md:19` — it claims traces are "emitted only under `--bench`";
  there is no such flag, the mechanism is `ctx.extra["bench"]`, which `patmir bench`
  sets and `patmir dub` does not. While in that block: add `--no-warm-cache` to the
  usage line (:15), the denominator note (:22-23), and to the cold-vs-warm callout
  (:32-38) that on torch the delta is *model builds*, not MLX kernel JIT.
- Remove the dead `load_pipeline_config` import at `patmir/bench.py:28`.
- One-liners found while auditing: `docs/architecture/runs.md` omits
  `mux_placements.json` and the CUDA-only `diarize/vocals_16k_mono.wav` and calls
  `manifest.json`'s `stages` a list when it is a dict;
  `patmir/pipeline/stages/diarize/sortformer_nemo.py:45-47` and the CUDA ADR still
  say NeMo must be built from git, but `nemo_toolkit 3.0.0` installed from PyPI with
  no Cython.

## Task 10 — Update the knowledge base (the maintenance contract)

- `docs/architecture/performance.md` — CUDA/RTX 3090 baseline sections for both
  modes beside the M1 rows, stating hardware and the `HF_HOME` filesystem. Demote
  masha and Fallout to a marked historical block noting the clips are gone. Update
  the bottleneck matrix and the "realtime gaps to close" line. **Rewrite roadmap
  item #3's chunking note**: it predicted finer sentences from
  `qwen3_asr.build_chunks()`; the measured answer is that Qwen3 never grouped
  sentences at all, and that was a quality bug, not a tail-shape nicety. Mark #6
  (batched TTS decode) measured.
- `docs/architecture/current-models.md` — replace every `UNMEASURED` in the CUDA
  profile table with the measured figure; drop the "never been executed" paragraph.
- `docs/architecture/quality-issues.md` — flip **#13** to `addressed (2026-08-24)`
  (4 speakers, 108 turns, 33 overlaps, parser did not raise). Add a **new row** for
  the ASR chunk-as-sentence defect: owner `asr (+tts)`, the multi-speaker-in-one-
  voice consequence, that it affected both machines from 2026-07-14, and its fix
  date. Cross-reference #11, whose "character voice swaps mid-scene" symptom this
  reproduces from a different cause.
- `docs/architecture/stages/asr/qwen3-asr-transformers.md` and
  `stages/asr/qwen3-asr-mlx.md` — record the segmentation defect and fix;
  `stages/diarize/`, `stages/tts/higgs-audio-v3.md`, `stages/separate/htdemucs.md` —
  first-execution results per stage.
- Clear the stale never-executed claims: `patmir.toml:465-468`,
  `performance.md:392,465`, ADR
  `2026-08-23-cuda-port-machine-profiles.md:3,334-335,344-346,359`. That ADR's
  "First end-to-end attempt (2026-08-24)" section documents only the torchcodec
  failure because it was written at 00:30, before the 02:11–02:24 successful runs;
  extend it rather than leaving it as the final word.
- `decisions/2026-08-23-warm-model-residency-and-prewarm.md` — its Consequences end
  with "**`bench.py` batch still passes no cache deliberately**, so `--repeat`
  continues to measure the cold-build path." Task 5 makes that false. Add a dated
  addendum rather than editing the record.
- New ADR `decisions/2026-08-24-asr-sentence-segmentation-and-cuda-baseline.md` (+ a
  line in `decisions.md`): the segmentation defect and fix, the measured CUDA
  baseline, the `max_batch` result, the 9p finding and the deliberate non-move.
  Append a "measured" note to the CUDA port ADR, whose revisit trigger ("the
  measured 3090 baseline lands far from the projection") this fires.
- Record the live torchcodec landmine in the WSL setup doc: `torchaudio.load` raises
  on `libtorchcodec_image.so` under ffmpeg 8.0.1 / torchcodec 0.16. It does not
  block the CUDA path — Higgs reads via
  `patmir/eval/candidates/tts/_audio_io.py` (ffmpeg), `htdemucs.py` via soundfile —
  but new code reaching for `torchaudio.load` will hit it.

---

## Verification

Do not run `uv sync` or recreate the venv at any point — see Task 2. Use the
existing `.venv` (`torch 2.11.0+cu128`, `transformers 5.15.1`, `nemo-toolkit 3.0.0`,
CUDA available, bf16 supported).

1. `uv run pytest` — the existing suite (284 tests) plus the new
   `patmir/tests/test_bench.py` and the segmentation tests stay green.
2. Profile resolution, both modes — already passing; keep as a regression check:
   ```bash
   PATMIR_PROFILE=cuda .venv/bin/python -c "
   from pathlib import Path
   from patmir.pipeline.config import load_pipeline_config
   from patmir.pipeline.streaming.runner import streaming_config
   b = load_pipeline_config(Path('patmir.toml')); s = streaming_config(Path('patmir.toml'))
   assert not [v['id'] for v in list(b.entries.values()) + list(s.entries.values()) if 'mlx' in v['id']]
   print({k: v['id'] for k, v in b.entries.items()})"
   ```
3. **The quality gate (Task 4) is the real acceptance test.** Ear-confirmed dub on
   `samples/sample_1.mkv`: one voice per character, no character speaking another's
   lines, no original-dialogue leak. Sentence count up from 23 toward ~100, max span
   well under 26.3 s, `overflows: 0`.
4. Warm-path fix: repeat 0 carries non-zero `build_s`, repeats 1–2 read `0.0`,
   `--no-warm-cache` reproduces the old behaviour, and the report's `warm` flag
   agrees with its own `build_total_s`.
5. `max_batch` chosen from measured warm TTS wall with `vram_free_gb` above ~2 GB,
   and `patmir.toml`'s comment replaced by the measurement.
6. Both baselines land in `runs/bench/`, and every number quoted in
   `performance.md` traces to one of those JSON files.
7. Every command in `performance.md` "Reproducing" and `windows-wsl2-cuda.md`
   Phase 9 runs as written against files that exist.
