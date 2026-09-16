# A translator outage must fail the dub, not mute it

- **Date:** 2026-09-15
- **Status:** accepted, shipped
- **Spec:** `agent-os/specs/2026-09-15-2107-translator-outage-fails-loudly/`
- **Quality issue:** #35

## The report

Author, 2026-09-15: *"the server was running, the browser plugin (latest version) was
showing that it translates the video and chunks are being translated (1, 2, .... 5,...) but
I was playing the video original (non translated, english) voice was playing (a little bit
muffled compared to the voice played when the translation was off)."* Then, mid-shaping:
*"openrouter credit went to 0, toped up the account."*

## What the logs showed

OpenRouter began answering **HTTP 402 Payment Required** at 2026-09-13 09:49:01 (credits API
at shaping time: 20.00 total, 20.18 used). 1,546 402s on 09-13, 223 on 09-14. The translate
stage never failed. For a 12-sentence watch-along section the log holds **13** 402s and then

    12 of 12 line(s) came back empty from the translator (audience=teenager) — those lines will be silent

and the stage reports `done`. 226 sections over two days. The reported evening (cache key
`70be3ae34a039a08`, 7 sections) had 206 of 206 sentences empty.

The section artifact says the rest: `translated.json` all `target_text: ""`,
`synthesized.json` 0 sentences, `mux_placements.json` empty, and `section_dub.wav` at a mean
of -31.4 dB against the source's -14.6 — exactly `0.6 * accompaniment + 0.12 * vocals`, the
mux's output when its sentence loop never runs and the duck mask stays all-ones. **The
"muffled English" was the original vocals at `vocals_bed_gain`, played by the extension over
the element it had muted.** The extension counted the sections, because every stage said
`done` and nothing in `StreamRecord.to_public()` or `sections.json` said otherwise.

## Why the pipeline did that

Three layers, each behaving as designed, composed into a dub with no speech in it:

1. **The batched candidate degraded an outage into per-line failures.** 402 is not in
   `_RETRYABLE`, so `_translate_chunk` raised `_BatchError`; `translate_batch` caught it and
   ran `_fallback_chunk`, whose `_one` caught *every* exception and returned `""` — documented
   as "the empty-translation case the adapter counts and reports, rather than a silent gap".
   That is 1 + 12 calls, all 402, twelve empty strings, and a normal return. Watch-along uses
   this candidate exclusively (`streaming/runner.py`); quality mode's per-sentence candidate
   raises on a non-retryable status and was never affected.
2. **The adapter counted blanks with no threshold.** The audience guardrail ADR
   (`2026-08-28-audience-age-guardrail`, "Empty results are now counted") made a per-line
   provider refusal a reported empty line rather than a failed run, on the stated reasoning
   that "the rest of the dub is good". `blank == len(sentences)` took the same branch.
3. **Nothing downstream could tell.** The TTS adapter skips empty lines silently; the mux
   emits the bed; `StreamRecord` has no warnings field; the manifest entry carries only
   timing (the one tell was `stage_wall_s.tts == 0.0`); `content.js` read only `waiting` off
   the `/sections` status overlay; and `session.js` checked `pb.playing` before
   `job.status === "failed"`, so even a failed job would have rendered as "Playing dub.".

The audience ADR's revisit trigger — "provider-side refusals show up as empty lines often
enough to need a retry rather than a report" — fired, but the case is an **outage**, not a
refusal, and the two must stay distinct: the ADR's rule that a refused line becomes an empty
line (never a bleep, never a dropped dub) is right and stands.

## Decision

**Fail loudly.** A translator that cannot translate anything is a different type of thing
from a translator that declined one line.

- `TranslatorUnavailable(RuntimeError)` in `pipeline/context.py`, next to `JobCancelled`,
  with `message` and optional `status`. Every layer lets it through the way it lets
  `JobCancelled` through.
- **Candidates:** `_FATAL = {401, 402, 404}` on both OpenRouter candidates (shared like
  `_RETRYABLE`). A fatal status raises `TranslatorUnavailable` from the first response — no
  retry, and in the batched candidate *not* a `_BatchError*, so `translate_batch` does not
  fall back and the 12-call storm is gone. `_fallback_chunk._one` re-raises
  `TranslatorUnavailable` and still blanks any other failure. **403 is deliberately not
  fatal**: OpenRouter returns it when the provider's moderation flags one input, which is the
  per-line refusal the audience ADR routes to an empty line. 400 stays a plain
  `RuntimeError` (usually our payload). The message is one readable line —
  `OpenRouter: HTTP 402 from google/gemini-3.5-flash — credits exhausted (...)` — because the
  extension prints it under the button.
- **Adapter backstop:** every line of `N >= ALL_EMPTY_MIN_LINES` (2) empty raises
  `TranslatorUnavailable` and writes no `translated.json`; partial blanks keep the warning. A
  one-line section whose only line is refused stays a reported gap, not a dead stream.
- **Propagation:** `StreamManager._process` re-raises it beside `JobCancelled`; any other
  section exception is still skipped, now counted in `sections_failed` (exposed in
  `to_public()`). `_run_locked` and `JobManager` record `exc.message` as `error` — the plain
  line, no traceback — where every other exception keeps `Type: msg\ntraceback`.
- **Cache guard:** `SectionResult` gains `lines` / `voiced_lines` (read from
  `translated.json` by `line_counts`), `section_timing` writes them into the manifest entry
  (additive, absent-able, like every timing key), and `_emit` refuses to `append_section` a
  result with `lines >= 2 and voiced_lines == 0`. Unreachable if the layers above work; it
  exists because the cache is the free-rewatch store.
- **Extension:** `content.js`'s sections poll reads the overlaid `status`/`error` and on
  `failed` (or a `cancelled` it did not issue — `shouldReleaseOnServerStatus` in `lib/sync.js`)
  calls `stopWatchAlong("server-failed")` and resumes the video, without the revive path (a
  diagnosed failure is deterministic). `session.js` checks `failed` before `pb.playing`, and
  `pollJob` calls `player.stop()` on a `failed` job while playback is attached.

## Decided against

- **Automatic local fallback** (MADLAD / NLLB on cuda). Every local translate entry in
  `patmir.toml` hardcodes `device = "mps"` except `madlad_400`; a fallback would have to
  record which translator made the dub in the cache key or a cached dub would misattribute
  its translator; and the register changes mid-video with no prompt surface. Real option,
  separate spec.
- **A preflight credit probe** at submit / connect. Author's call: the first section's
  failure is now the signal, seconds after the prelude.
- **Making 403 fatal.** See above.
- **Surgical purge.** Nine cache keys held sections produced after the cutoff
  (`3f9ce3f2c3cfe9be`, `47db7fc315969105`, `5f7d4767ec45afbb`, `70be3ae34a039a08`,
  `9b6cc113acc9b825`, `9bc1d6b1ec513e95`, `c7e79dd6873c515b`, `c8e666c513e8fa71`, and
  `17fc1b081e03f729` with 25 of 71 post-cutoff); the author chose deleting the directories
  wholesale, which also lost 46 good sections in the last one. The list with URLs is in the
  spec's `shape.md`.

## Incident during implementation

`uv run pytest` **removed and recreated `.venv`**: `.python-version` pins 3.11 while this
box's environment is Python 3.12 (the setup doc's own instruction), so uv treated the venv as
stale and rebuilt it with only `patmir` in it. The live server broke on its next lazy import.
Rebuilt per `docs/setup/windows-wsl2-cuda.md` Phase 7, pinned to the wheels still in the uv
cache (torch 2.11.0+cu128, transformers 5.16.1, nemo_toolkit 3.0.0), and the server restarted.
Recorded in the setup doc's gotchas; run tests here with `.venv/bin/python -m pytest`.

## Verification

- `patmir/tests`: `OutagePolicy` (11 cases incl. the 1 + 12 storm shape, 403 fallback, fatal
  raised inside the concurrent fallback), `EmptyLineGuard` (all-empty raises and writes no
  artifact; one empty line only reports; batch dict of `""` raises),
  `test_translator_outage.py` (voiceless rule, `line_counts` on the incident's artifact shape,
  manifest fields, `_emit` refusal, stream and job end `failed` with the plain message, the
  `/sections` overlay carries it). Full suite green.
- `extension/tests`: `failed` outranks `playing`; a failing watch-along job stops the player
  and shows the reason; `shouldReleaseOnServerStatus` truth table. 257 pass.
- Live reproduction with an invalid key is the remaining step; see the spec's plan.

## Revisit

- A one-line `child` refusal killing a stream (would mean the `ALL_EMPTY_MIN_LINES` floor is
  in the wrong place).
- OpenRouter changing what 402/403 mean.
- Appetite for a local translate fallback — this ADR is where its constraints are listed.
- `sections_failed` going non-zero on a healthy stream.
