# CUDA baseline + ASR segmentation — Shaping Notes

## Scope

The project moved from the 32 GB Apple Silicon laptop to an RTX 3090 / WSL2 box.
The ask was "start benchmarking". Shaping found a correctness bug that has to be
fixed first, so the spec is two things in order:

1. **Make the dub correct on this box.** ASR was emitting one `Sentence` per
   28-second chunk, so multiple characters were being dubbed in one character's
   voice. Fixing it changes sentence count and TTS chunk distribution, which means
   any baseline measured before the fix is void.
2. **Then establish a trustworthy baseline** for both execution modes, with a bench
   harness that measures the warm path it claims to measure, and docs/clip
   references that match what is on disk.

Explicitly out of scope: pipeline optimizations. The `max_batch` sweep is in scope
because it is measurement — the docs left the value explicitly unmeasured and acting
on the result is a one-line config change.

## Decisions

1. **Correctness before benchmarking.** Author's call, and the right one: a
   segmentation fix moves the numbers, so benchmarking first would produce a
   baseline that is stale on arrival.
2. **`transformers` floor raised to `>= 5.13`** in the shared `local` extra, not
   just `cuda`. Below that the pipeline fails (`qwen3_asr` needs transformers'
   native `qwen3_asr` module). 5.13 still satisfies the Higgs checkpoint's `>= 5.5`.
   The stale `uv.lock` advertising `>= 4.40` is the actual hazard and is fixed too.
3. **`HF_HOME` stays on the 9p mount.** Measure the I/O-vs-init split only — a
   980 PRO is going in shortly and will change the relocation decision. Deferred
   deliberately, not overlooked.
4. **`max_batch` is decided by the sweep, not chosen up front.** The author was
   unsure; the answer is that nobody should have to guess. Sweep 4/8/16, pick the
   largest N whose warm TTS wall improves with `vram_free_gb` above ~2 GB, and treat
   an OOM at 16 as a result.
5. **Docs adopt `samples/sample_1.mkv`**; a clip registry carries provenance.
   Nothing on disk is renamed.
6. **masha cleanup touches live docs only.** ADRs and closed specs stay as
   historical record — rewriting them would falsify the decision log.
7. **Do not run `uv sync` or recreate the venv.** The lock is stale and
   `.python-version` disagrees with the installed interpreter; either action could
   destroy a working environment mid-spec.

## Context

- **Visuals:** none provided.
- **References:** see `references.md`.
- **Product alignment:** this is Phase 3 / platform work. `agent-os/product/roadmap.md`
  Phase 3 carries the hard rule that Mode B (watch-along) must not compromise Mode A
  quality — which is exactly why the segmentation fix gates the benchmark rather than
  following it. The roadmap also anticipated this hardware: Phase 0's escalation path
  named "re-run on the workstation GPU with larger local candidates" as step 1.

## The blocking defect, as found

`patmir/pipeline/stages/asr/qwen3_asr.py:382-388` appends one `Sentence` per ASR
chunk and never calls `group_into_sentences`
(`patmir/pipeline/stages/asr/_sentence.py:208`), which splits words on terminal
punctuation **or** a word gap over `max_gap_s`. `whisper_mlx`, `faster_whisper`,
`elevenlabs_scribe` and `openrouter_stt` all call it; **both** Qwen3 candidates skip
it.

Measured on `runs/20260824-022033-sample_1`:

| signal | value |
| --- | --- |
| sentences / 301 s of video | **23** |
| diarization turns | **108** |
| words containing punctuation | **0 of 398** |
| sentence span mean / median / max | 9.7 s / 8.6 s / **26.3 s** |

Because the runner's `apply_speaker_tags` assigns one *dominant* speaker per
sentence, a span covering several turns is dubbed in one character's voice. The
14.3–27.4 s "SPEAKER_02" line contains a mother comforting a child *and* someone
saying "call the cops, give them this". It also produces the 44–55 s TTS decode
chunks and ref clips whose `ref_text` reads like several people talking.

This is the 2026-07-14 ASR flip silently redefining "sentence". `performance.md`
recorded the symptom as a TTS-tail *performance* follow-up ("Qwen3 chunks ~3.5×
coarser — 25 vs ~104 sentences per 5 min"). It is a quality bug, it affects **both
machines**, and the CUDA port only made it visible.

## What was already working (so the spec doesn't re-litigate it)

- Batch runs end to end on CUDA: 3 repeats, all 9 stages `done`.
- `sortformer_nemo` is healthy — 4 speakers, 108 turns, 33 derived overlaps, parser
  did not raise. **Quality issue #13 is resolved in practice.**
- The `cuda` profile resolves correctly in batch *and* watch-along, no MLX leaks.
- TTS: `overflows: 0`, batched decode live (`batches: 3`).

## Standards Applied

None. `agent-os/standards/index.yml` contains only a header comment, so there are no
standards to apply to this work.
