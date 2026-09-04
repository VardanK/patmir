# 2026-09-04 — Review of the performance branch: what it fixed, and two decisions it forced

**Status:** accepted
**Scope:** `patmir/ui/*`, `patmir/pipeline/streaming/{registry,pipeline}.py`, `patmir/pipeline/stages/asr/_language.py`, `patmir/eval/candidates/translation/openrouter_batch.py`, `patmir/eval/candidates/tts/higgs_v3_torch.py`, `extension/`

## Context

A code review of the 43-commit `performance-improvements` branch before merge.
The performance work itself held up: the CUDA-graph design recomputes both
masks per replay from a static-address `cumulative_length` tensor rather than
freezing them at capture, the measurements are sequenced one lever at a time,
and both suites were green. What the review found was in the surrounding
changes, and in claims that the code did not enforce. Every item was verified
against the code before it was acted on; the two that turned out to be
docstring-only defects were retracted.

## What was broken

- **The watch-along progress bar could not work.** `StreamRecord.dubbed_s` was
  maintained on the record and read by the extension, and `to_public()` never
  emitted it. Both halves had tests; neither crossed the seam.
- **A race in section pipelining.** `translate_section` runs outside the GPU
  lock by design and snapshots the `SpeakerRegistry` dicts; the prefetch
  thread's `resolve_or_register` inserts into the same dicts, and the registry
  had no lock. "Dictionary changed size during iteration" mid-section, or a
  half-updated cast handed to translate.
- **The ASR language probe accepted garbage.** `_to_iso` returned any
  unrecognised string verbatim, so a tag the 24-token budget had truncated
  (`"Engl"`) became the language `"engl"`, was forced onto every chunk, and
  skipped the "no tag, run the full probe" fallback the budget comment promised.
  Both Qwen3 candidates carried their own copy of the function.
- **The batched translator ignored `concurrency` and the retry knobs.** Its
  per-sentence fallback was built with class defaults whatever `patmir.toml`
  said, and watch-along runs that candidate exclusively.
- **`--host 0.0.0.0` exposed more than its warning admitted.** `/` listed every
  job id with no token, and `/run/<id>/file/<path>` then served the source
  audio, dub and transcript. The note said ids had to be guessed.
- **Extension:** revive-once could restart a stream the viewer had just
  cancelled; its stall clock measured "time since a new section" rather than
  time starved, so a resumed cached stream burned the revive with zero grace;
  and Չեղարկել stopped nothing server-side while a watch-along job was still in
  prelude.
- **TTS bookkeeping:** `reset_summary` never zeroed the burst counters (a warm
  watch-along instance accumulated them across sections), bursts were counted
  per attempt rather than per line, the eager fallback from graph capture was
  documented as bitwise identical and is not once the gqa kernel is on, and
  `prefill_s` nested `graph_capture_s`.
- **`_cached_opening`** applied its 1 s end tolerance to the start too, which
  would leave `[0, 0.9)` uncovered and schedule a sub-second section at 0.
- **`SectionPipeline.close()`** bounded its wait with a timeout and then
  re-blocked in `shutdown(wait=True)` with none; and a resumed stream discarded
  every prefetch because the scheduler passes `max_end` for every section
  there, paying two analyses per section serially.

## Decisions

### `CACHE_VERSION` 1 → 2

`config_hash` sees the TOML entries only. Four output-affecting changes landed
between 2026-09-02 and 2026-09-03 as code constants with no key movement: the
mux onset-burst trim and fade-in cap (#30), `EMOTION_MAX_CLIP_SECONDS`
(changes the label of every turn over 20 s, hence ref choice and the
translator prompt), the 8 s ASR language probe, and `shifts = 0` on the two
torch fallback separate entries that had no `shifts` key (the docs said all
four entries carried it; two did). A dub cached under version 1 can carry any
mix of the old behaviours, and the author already accepted a regeneration for
the gqa kernel and `max_batch` on 2026-09-02, so one more that folds all of
this in is cheaper than reasoning about which cached dub has which onset. The
two `PATMIR_TTS_*` env overrides that bypassed the hash for `max_batch` and
`decode_attention` are removed; `PATMIR_TTS_CUDA_GRAPHS` stays because
`cuda_graphs` is in `_NON_OUTPUT_KEYS`.

**Rejected:** recording the accepted drift in the ADR and leaving the key.
Cheaper today, but it leaves a library where two dubs under one key differ on
a defect the author ear-confirmed as fixed.

### Progress routes: loopback client or token

`/` and `/run/*` are browser navigations, which send no `Authorization`
header, so they cannot simply take `TokenAuth` without breaking the local
progress UI. `LoopbackOrToken` passes a request whose client address is
loopback and otherwise requires the token exactly as `/jobs` does. This keeps
the pages working on the machine itself and closes the LAN leak.

**Rejected:** gating only when `--host` is non-loopback. The server bound to
`0.0.0.0` is also the one a local browser talks to, so the gate has to be
per-request, not per-bind.

## Verification

- Every fix carries a test that crosses the seam it lives on: `to_public()`
  contains `dubbed_s`; a writer thread registering 400 speakers against three
  readers snapshotting the registry; `_to_iso("Engl") is None` and the probe
  spy returns None for four garbage tags; `build()` of the batched translator
  forwards the knobs to its fallback; a remote `TestClient` gets 401 on all
  five progress routes and 200/404 with the token or from loopback; bursts
  counted once per line through the **production** `_synthesize_chunk`, whose
  test helper used to be a hand-copied replica; a cached section at 0.9 s is
  not the opening.
- Python suite and `node --test "extension/tests/*.test.js"` green.

## Not done, on purpose

- The onset detector's "no pitch" test is one 200 Hz FFT bin over 5 ms — an
  energy ratio, not a pitch test. Widening the spectral window changes
  behaviour on real audio and needs the 3040-line corpus re-measured first.
- Bursts and overflows share `max_retries`; a separate budget is a policy call.
- `optional_host_permissions: ["http://*/*", "https://*/*"]` in the extension
  manifest permits sending the bearer token over plaintext HTTP to any host the
  viewer saves; `permissions.request` after an `await` may forfeit the user
  gesture. Both need a manual check in Chrome before changing.
- The root logger at INFO into a 40 MiB rotating file can rotate away the
  crash traceback the file exists to keep.

## Revisit

- A cached dub that regenerates and sounds worse: `CACHE_VERSION` back to 1
  reverts the key, not the behaviour.
- A reverse proxy in front of `patmir ui`, which makes every client loopback
  and would void the progress-route gate.
