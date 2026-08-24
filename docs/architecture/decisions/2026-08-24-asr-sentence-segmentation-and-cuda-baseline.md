# 2026-08-24 — ASR sentence segmentation, and what the first CUDA baseline actually measured

Status: **accepted**. Segmentation fix landed and verified on `samples/sample_1.mkv`; ear-confirm
pending. Bench harness correction landed. CUDA baseline captured.

## Context

`2026-08-23-cuda-port-machine-profiles` left the four torch candidates written but never executed.
The intent for this session was simply to benchmark them. Two things had to be fixed first, because
both meant the numbers would have described something other than the pipeline.

## Decision 1 — a "sentence" is an utterance, not an ASR chunk

`qwen3_asr.py` appended one `Sentence` per ASR chunk and never called `group_into_sentences`
(`patmir/pipeline/stages/asr/_sentence.py`), the shared helper that splits words on terminal
punctuation **or** a word gap. `whisper_mlx`, `faster_whisper`, `elevenlabs_scribe` and
`openrouter_stt` all call it. **Both** Qwen3 candidates — torch and MLX — did not.

So from the 2026-07-14 ASR flip onward, on **both machines**, "sentence" silently meant "up to
`chunk_seconds` (28 s) of audio".

Measured on `samples/sample_1.mkv` (301.1 s, Russian, 4 speakers):

| signal | value |
| --- | --- |
| sentences | **23** |
| diarization turns over the same audio | **108** |
| words containing terminal punctuation | **0 of 398** |
| sentence span mean / median / max | 9.67 s / 8.56 s / **26.32 s** |

The zero-punctuation row is why this could not self-correct: Qwen3-ASR emits no terminal
punctuation at all, so even a caller that *did* group would have had only the word-gap branch to
work with.

### Why this was a quality bug and not a performance note

The runner's `apply_speaker_tags` assigns one **dominant** speaker per sentence. A sentence covering
several diarization turns therefore gets one speaker id, one pinned reference clip, and one cloned
voice. Concretely, on that clip:

- A 22.9 s "sentence" spanned **11 turns** with the dominant speaker at 86 % — just over the 0.85
  `DOMINANT_THRESHOLD` — and so swallowed another character's 1.5 s line entirely.
- A 13.1 s "sentence" contained a mother comforting a child *and* a different character saying
  "call the cops, give them this", all synthesized in one voice.
- Reference clips inherited it: `SpeakerRef.ref_text` read like several people talking, which is the
  text the TTS clones a character's voice against.
- The 44–55 s TTS decode chunks were a **symptom**, not the disease.

`performance.md` had recorded the symptom for a month — "Qwen3 chunks ~3.5× coarser, 25 vs ~104
sentences per 5 min ... splitting chunks at word gaps would flatten that tail" — filed under the
performance roadmap as a watch-along buffer concern. The tail was real. The reason for it was not
what the entry said, and the cost was voice attribution, not latency.

This is worth naming as a process failure, not just a code one: a coarse-sentence-count observation
was treated as a throughput datapoint because it showed up in a benchmark. Nothing checked it
against `diarize/turns.json`, which would have shown 23 against 108 immediately.

### The fix, in three parts

Each part was found because the previous one exposed it.

1. **Both Qwen3 candidates call `group_into_sentences`.** The MLX sibling is fixed too — leaving the
   Mac on chunk-sentences would have made the two machines' baselines differ for a reason unrelated
   to hardware.

2. **`_boundary_time` now measures against the speaker who *opens* the sentence.** It previously
   looked for the first turn whose speaker differed from the **dominant** one, which silently failed
   whenever the dominant speaker was the *later* of two: dominant is B, the first non-B turn is A
   starting at or before the sentence start, the `change > sent.start` guard rejects it, the function
   returns `None`, and the sentence is never split — B's voice speaks A's words. That is half of all
   two-speaker sentences, and it had been latent since the function was written.

3. **New `max_sentence_s` cap**, splitting recursively at the widest interior gap. `max_gap_s` alone
   has a floor: the 22.9 s run above had a widest interior gap of **0.32 s** across 56 words, so no
   gap threshold that keeps normal speech intact would ever have cut it. The cap **defaults to
   `None`** in the shared helper, so the four backends that were already producing short sentences
   keep their exact prior behaviour; only the Qwen3 candidates pass a value (8.0 s), because they are
   the ones with no punctuation signal.

Result on the same clip, identical word count:

| | before | after |
| --- | --: | --: |
| sentences | 23 | **81** |
| span mean / median | 9.67 s / 8.56 s | **1.88 s / 1.28 s** |
| max span | 26.32 s | **6.96 s** |
| spans > 8 s | 12 | **0** |
| TTS overflows | 0 | **0** |

16 unit tests in `patmir/tests/test_qwen3_asr_torch.py`. Ear-confirm pending on
`runs/20260824-141543-sample_1/out.mkv`.

**Consequence for every earlier measurement:** per-sentence TTS percentiles recorded before this
date, on either machine, describe chunk-sized units and are not comparable to anything after it.

## Decision 2 — `patmir bench --mode batch` measures the warm path

`bench_batch` called `run_job` with **no `stage_cache`**, so every `--repeat` rebuilt every stage
model. `bench_watch_along` had shared one since 2026-07-14, and its own comment said "bench must
measure the warm path, not the legacy rebuild path".

This was invisible on Apple Silicon, where MLX mmap loads put `build_s` at 0–3 s. On torch it is
~110 s of a 242 s run — **45 % of wall** (asr `build_s` 42–48 s, tts 50–62 s). `--repeat 3` was
timing nine model loads three times over, and `performance.md`'s stated convention ("the warm run is
the baseline") did not hold for the mode most people run.

`2026-08-23-warm-model-residency-and-prewarm` had made this explicit — "`bench.py` batch still
passes no cache **deliberately**, so `--repeat` continues to measure the cold-build path" — which was
a defensible choice when a build cost 2 s and is not when it costs 110 s.

Now: one `stage_cache` hoisted above the repeat loop, so **repeat 0 is cold and repeats 1..N-1 are
warm**, and both numbers come from one command. `--no-warm-cache` restores the old behaviour, for N
independent cold samples or on a box where all-models-resident is tight.

Deliberately **not** `prewarm_stage_cache`: it would zero repeat 0's `build_s` too, destroying the
cold datapoint, and it swallows build failures by contract — a benchmark must fail loudly rather
than quietly report a run with a stage missing.

`bench_watch_along` keeps its cache **inside** the repeat loop. Watch-along warmness is per-section
(section 0 builds, later sections reuse), so each repeat must pay its own section-0 builds to stay
comparable with the stored 1.71 / 1.611 / 1.47 baselines.

### Two reporting corrections that came with it

- **TTS percentiles are per row, and a row is not always a sentence.** Under batched decode the
  adapter emits one row per *chunk* (`"kind": "chunk"`), since a per-sentence wall does not exist
  once the chunk's cost is shared. The report read `count: 3` beside `sentences: 24` with
  `p90_s: 51.2` — chunk figures a reader takes for the per-sentence p90 the Apple-Silicon rows
  quote. `tts["unit"]` now says which, and the report labels it (`p90 51.20s/chunk`). No
  per-sentence walls are invented.
- **Watch-along `agg_rtf` divides by dubbed audio, not source seconds**, unlike batch and unlike the
  convention `performance.md` states. Left as-is on purpose — every historical watch-along number
  was computed that way, and reflooring the field would leave a reader unable to tell which
  convention a row used. The denominator is now named (`notes.rtf_denominator`) and the
  source-second figure published beside it (`notes.agg_rtf_source`).

`BenchResult` also gained `profile`, `warm_cache`, `warm`, `build_total_s` and VRAM/RSS telemetry in
`notes`. All defaulted, so the stored reports still deserialize. `warm` is **measured**
(`warm_cache and build_total_s <= 0.05`), not declared, so the JSON says what happened rather than
what was requested.

## Decision 3 — `transformers>=5.13` belongs in the shared `local` extra

The floor was `>=5.5` in `local` and `>=5.13` only in `cuda`. But `qwen3_asr` needs 5.13 for
transformers' native `qwen3_asr` module, so the base install advertised a floor the pipeline cannot
run on: a resolver satisfying `>=5.5` could hand back a transformers that breaks ASR with
`KeyError: 'qwen3_asr'`. Raised to `>=5.13`, which still satisfies the Higgs checkpoint's own
`>=5.5`. The `cuda` line is kept as documentation of *which* candidate needs it.

`uv.lock` was worse: it predated the `cuda` extra entirely, still advertised `transformers>=4.40`,
carried no `nemo-toolkit`, and had no platform markers on the MLX deps — so a `uv sync` could
re-resolve transformers below 5.13 and reproduce exactly the breakage commit `f6e367d` was written
to prevent. Regenerated; it now declares `>=5.13` for both extras, carries `cuda`/`dev` and
`nemo-toolkit 3.0.0`, and gates the MLX packages on `sys_platform == 'darwin'`.

**Unverified and flagged:** raising the shared `local` floor also raises it on Apple Silicon, and no
MLX package's compatibility with transformers >= 5.13 was checked from the CUDA box. Verify on the
Mac's next install rather than assuming it.

## Finding — the VRAM ceiling on this box is the Windows commit limit

Not a decision, but the single most expensive thing learned this session, and it
invalidated three of my own earlier explanations before the measurement settled it.

`patmir bench --mode batch` with the warm cache OOMed while loading Higgs. The failure
was not what it looked like:

```
Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 24.00 GiB of which
14.46 GiB is free. Including non-PyTorch memory, this process has
17179869184.00 GiB memory in use.
```

A **20 MiB** request refused with **14.46 GiB free**, and a "non-PyTorch memory" figure
of exactly 2^34 GiB — i.e. garbage from a subtraction that went negative.

Measured cause: **a CUDA allocation under WSL2 consumes Windows *commit charge* at
~1.06 GB per GB of VRAM.** Under the Windows display driver model a GPU allocation
needs a system-memory backing store (so the driver can evict it off the card), and
Windows commits that backing up front even though the data lives on the card and the
backing is never touched. Commit is a promise, and
`commit limit = physical RAM + pagefile size`.

This box was configured with a manually-pinned **4 GB** pagefile on 31.84 GB of RAM →
commit limit 35.84 GB, idle baseline ~19.7 GB → **~16 GB of headroom on a 24 GB card**.
The warm cache plus Higgs wants ~15 GB, which is why it presented as intermittent.

Every observed ceiling lines up with the commit headroom at that moment:

| scenario | allocated at failure | why |
| --- | --: | --- |
| bare torch, quiet machine | 20.75 GB | lowest baseline |
| four non-TTS models resident | 15.23 GB | ~19.7 GB baseline |
| bench warm cache, loading Higgs | 13.06 GB | higher baseline |
| `patmir dub` with Chrome playing video | 8.23 GB | Chrome holding several GB of commit |

Confirmed by raising the pagefile to 20 GB: commit limit went **35.84 → 51.84 GB**,
exactly the +16 GB, and the warm path stopped failing.

Three explanations I advanced before measuring, all wrong, recorded so the next person
does not spend the time again:

- **"`.wslconfig memory=24GB` starves Windows."** It is a cap, not a reservation —
  `vmmemWSL` held 3.98 GB against it with 21.85 GB of RAM free.
- **"The warm cache retains GPU inference state."** A per-stage trace showed it holds
  6.49 GB, exactly the weights.
- **"Fragmentation."** `expandable_segments:True` changed nothing, and
  reserved-but-unallocated was 15 MiB — there was no fragmented pool.

Also ruled out, since both were reasonable suspicions: **the VRAM is not faulty** (19 GiB
tested with four patterns = 76 GiB written and verified, 0 errors, 0 disturb errors, no
Xid entries) and **the Windows "shared GPU memory" pool is not involved** (per-process
shared usage stayed at 0.21 GB flat; CUDA under WSL2 does not use it).

Operational consequence: **benchmarks need the Windows desktop quiet**, because anything
holding commit shrinks CUDA's ceiling. Same class of hazard as the `caffeinate` warning
in `performance.md`. Full write-up, including the sizing table, is in
`docs/setup/windows-wsl2-cuda.md`.

## Consequences

- Sentence counts and per-sentence TTS percentiles from before 2026-08-24 are not comparable to
  anything after, on either machine.
- Batch bench reports now carry a cold row and warm rows; quoting a single "RTF" from a multi-repeat
  run without saying which is a mistake the report format now makes hard.
- `docs/architecture/eval-clips.md` is new and is the record of which clips exist. masha and Fallout
  are gone; measurements citing them cannot be reproduced.
- The 9p `HF_HOME` and Windows-side-GPU-contention findings are in
  `docs/setup/windows-wsl2-cuda.md`, not here — they are environment properties, not decisions.

## Revisit

- An ear-test finds the 8.0 s `max_sentence_s` cap cutting mid-utterance often enough to hurt
  prosody. The cap splits at the widest available gap, so the failure mode is a cut at a weak
  boundary, not a cut mid-word.
- A clip shows three or more speakers inside one grouped sentence. `apply_speaker_tags` splits at
  most **once** per sentence; that was load-bearing when sentences were 28 s chunks and is not now.
  Make it recursive rather than widening the cap.
- Qwen3-ASR gains punctuation output, at which point `max_gap_s` stops being the only real signal
  and the cap can probably be relaxed.
- `max_batch` is swept on the 3090. It is still the conservative guess from
  `2026-08-23-cuda-port-machine-profiles`, and the measured headroom says there is room.
