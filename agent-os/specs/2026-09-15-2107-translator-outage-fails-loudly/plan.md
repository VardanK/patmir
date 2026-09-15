# A translator outage must fail the dub, not mute it

Spec folder: `agent-os/specs/2026-09-15-2107-translator-outage-fails-loudly/`

## Context

Author report (2026-09-15): yesterday evening the server was running, the extension showed
sections 1, 2, ... 5 being translated, but the original English voice played, "a little bit
muffled" compared to translation-off. Author has since confirmed the OpenRouter balance hit 0
and topped it up (translation works again as of today: key `efa962dcde154507`, 18 sections, all
with TTS work).

What the logs and artifacts show (`~/.patmir/logs/server.log`, `runs/20260914-210424-section`):

- OpenRouter began answering **HTTP 402 Payment Required** at 2026-09-13 09:49:01 (credits API:
  20.00 total, 20.18 used). 1,546 402s on 09-13, 223 on 09-14. The first ~10 minutes were mixed
  200/402 (partial-empty sections), then 100% 402.
- The translate stage never failed. Watch-along uses `openrouter_gemini_flash_batch`
  (`streaming/runner.py:68`). Per 12-sentence section: 1 batch POST raises `_BatchError`
  (402 is not in `_RETRYABLE`, `openrouter_batch.py:323-327`), `translate_batch` catches it and
  degrades to `_fallback_chunk` (`:245-258`), whose `_one` catches **every** exception and
  returns `""` (`:375-382`, documented as intentional: "the empty-translation case the adapter
  counts and reports"). 12 more 402s, 12 empty strings, `translate_batch` returns normally.
- The adapter counts blanks and logs `12 of 12 line(s) came back empty ... those lines will be
  silent` (`translate/adapter.py:217-224`), with a comment "never fail the run". No threshold:
  100% empty is treated like 1 empty. Stage `done`.
- TTS adapter skips empty `target_text` (`tts/adapter.py:283-285`), 0 sentences synthesized;
  mux loop never runs, `duck_mask` stays all-ones, output = `0.6*accompaniment + 0.12*vocals`
  (`mux/ffmpeg_pad_stretch.py:188, 353-356`), peak-normalised. Measured `section_dub.wav`
  mean -31.4 dB. That is the muffled English, played by the extension over the muted element.
- Nothing reached the client: `StreamRecord.to_public()` has no warnings field; `sections.json`
  entries carry only timing (the only tell is `stage_wall_s.tts == 0.0`); `content.js:903`
  reads only `data.status === "waiting"` from the `/sections` overlay, never `failed`/`error`;
  `session.js:263-271` checks `pb.playing` before `job.status === "failed"`, so a failure during
  playback would render as "playing" anyway; `pollJob` (`session.js:866-896`) does nothing on
  `failed` (no `player.stop()`).
- The quality-mode (per-sentence) candidate is **not** affected: `_post_with_retry` raises on a
  non-retryable status (`openrouter.py:259-264`) and the adapter's per-sentence path re-raises
  (`adapter.py:171`). Only the batched candidate swallows.
- Blast radius: 09-13 after 09:49: 209 runs, 1,421 of 2,492 sentences empty; 09-14: 17 runs,
  206 of 206 empty. **Nine cache keys** hold poisoned sections (produced_at >= cutoff, tts=0):
  `3f9ce3f2c3cfe9be` (11/11), `47db7fc315969105` (29/29), `5f7d4767ec45afbb` (18/18),
  `70be3ae34a039a08` (7/7), `9b6cc113acc9b825` (4/4), `9bc1d6b1ec513e95` (17/17),
  `c7e79dd6873c515b` (22/22), `c8e666c513e8fa71` (4/4), and `17fc1b081e03f729` (25 of 71
  post-cutoff, partially empty; 46 good sections before the cutoff).

The audience-guardrail ADR (`decisions/2026-08-28-audience-age-guardrail.md:108-119`) is why
empties are counted rather than raised: a per-line provider refusal under `child` must not
become silence. Its revisit trigger ("provider-side refusals show up as empty lines often enough
to need a retry rather than a report", `:163-164`) has effectively fired, but the case here is
an **outage**, not a refusal, and the design must keep the two apart.

## Decisions (author, 2026-09-15)

1. **Fail loudly.** Translator unavailability raises a typed error; the stream/job ends `failed`
   with a readable reason; the extension stops the dub and gives the original audio back;
   nothing is written to the cache. No local auto-fallback in this spec.
2. **Delete the nine poisoned cache dirs** wholesale (the author's choice; surgical removal of
   `17fc1b08`'s 25 post-cutoff entries is possible via `produced_at` but not chosen).
3. **No preflight credit probe.** The first section's failure is the signal.

## Design

### A. Typed unavailability at the candidate (`patmir/eval/candidates/translation/`)

- New exception `TranslatorUnavailable(RuntimeError)` in `patmir/pipeline/context.py` next to
  `JobCancelled` (both are control-flow signals every layer must let through). Message is
  human-readable and short, e.g. `OpenRouter: HTTP 402 Payment Required (credits exhausted)`;
  carries `status: int | None`.
- `openrouter.py`: `_FATAL = frozenset({401, 402, 403?})` — decide 403 carefully. OpenRouter
  returns 403 for **per-input moderation flags** (a per-line refusal, which the guardrail ADR
  says must become an empty line, not a failure), so **403 stays a per-line failure**. `_FATAL =
  {401, 402}` plus 404 on the model id (the whole job would fail identically). In
  `_post_with_retry`, a `_FATAL` status raises `TranslatorUnavailable(status=...)` instead of a
  bare `RuntimeError`; other non-retryable statuses keep raising `RuntimeError` as today.
- `openrouter_batch.py`: `_BatchError` gains `status`. In `_translate_chunk` a `_FATAL` status
  raises `TranslatorUnavailable` directly (not `_BatchError`), so `translate_batch`'s
  `except _BatchError` does **not** catch it and there is no 12-call fallback storm. In
  `_fallback_chunk._one`, `except TranslatorUnavailable: raise` before the blanket
  `except Exception`; with the thread pool, `ex.map` re-raises the first worker exception — fine.
- Keep `openrouter.py:320-325` (200 with `content: null` -> `""`): that is the refusal path.

### B. Outage detection at the adapter (`patmir/pipeline/stages/translate/adapter.py`)

Belt and braces for outage modes that are not a status code (provider returns 200 with empty
content for everything, a different cloud candidate, a misconfigured model):

- After the existing `blank` count: if `blank == len(sentences)` and `len(sentences) >=
  ALL_EMPTY_MIN_LINES` (2), raise `TranslatorUnavailable(f"translator returned no text for any
  of {n} lines (audience=...)")`. A single-line section that comes back empty stays a warning
  (a one-line `child` refusal must not kill a stream — the ADR's case).
- Partial blanks (1 <= blank < n) keep today's warning and `ctx.progress` message. The comment
  block at `:208-216` is rewritten to state both rules and why the boundary sits there.

### C. Propagation: stream, pipeline, jobs

- `patmir/pipeline/runner.py:214-242`: no change needed — a raising stage is marked `failed`
  with `st.error` and re-raised. Verify `TranslatorUnavailable` survives `run_job` unchanged
  (it is a plain subclass; no wrapping happens).
- `patmir/pipeline/streaming/pipeline.py:153-170`: `translate_section` raises straight out of
  `process()`; `close()` still waits for the in-flight prefetch analysis (existing behaviour,
  required — GPU thread). Confirm no `except Exception` around translate there (only around
  prefetch analysis at `:199, :220`).
- `patmir/ui/stream.py:652-660` `_process`: add `except TranslatorUnavailable: raise` alongside
  `JobCancelled`, so it propagates out of `scheduler.run()` to `_run_locked` (`:506-523`), which
  already sets `rec.status = "failed"` and `rec.error`. Change `rec.error` for this exception to
  the **plain message** (no traceback) — the extension shows `errorText` verbatim; keep the
  traceback form for unexpected exceptions. Also: the prelude/opening dub path
  (`_run_blocking` -> `run_prelude` / `dub_prelude_window`) — verify the error propagates there
  too (it should: no swallow between them and `_run_locked`).
- `patmir/ui/jobs.py:446-461` (quality mode): same plain-message treatment for
  `TranslatorUnavailable`; the per-sentence candidate already raises, so this is a message fix.
- Other section exceptions (non-fatal) still return `None` and skip the section as today; add a
  `sections_failed: int` counter on `StreamRecord` exposed in `to_public()` so a skipped section
  is at least visible (small, additive).

### D. Cache must never hold a voiceless "dub" (`patmir/ui/stream.py`, `streaming/section.py`, `timing.py`)

- `SectionResult` gains `lines: int` and `voiced_lines: int` (sentences in `translated.json`
  and sentences with non-empty `target_text`), filled in `speak_section` from the job dir.
- `section_timing()` writes `lines` / `voiced_lines` into the manifest entry (additive,
  absent-able, like every other timing key) so a cached section is self-describing.
- `_emit` refuses to `append_section` a result with `lines >= 2 and voiced_lines == 0`, logs it,
  and does not count it toward `sections_done` / `dubbed_s`. With A+B this should be unreachable
  — it is the last line of defence, same role as the mux onset trim in issue #30.

### E. Extension: a failed stream releases the video (`extension/`)

- `content.js` `pollWatchAlongSections` (`:891-936`): read the overlaid `data.status` and
  `data.error`. On `status === "failed"`: `stopWatchAlong("server-failed")`, `waPlay()` if
  paused (mirror the stall-abandon branch at `:1085-1097`), and surface the error to the
  session (extend the playback status message at `:186-195` with `error: string|null`). Also
  on `cancelled` when we did not initiate it. Do **not** trigger the revive path for `failed`.
- `lib/session.js` `deriveView` (`:263-271`): check `job.status === "failed"` **before**
  `pb.playing`, so a failure during playback renders `job-failed` (RETRY, tone `error`,
  `errorText`). `pollJob` (`:866-896`): on `failed`, call `player.stop()` (the same path
  `stop()` uses) so the element is unmuted even if the content script missed the poll. Add an
  Armenian/English string for the translator case only if `errorText` is not enough — the plain
  message from C should read fine as the subtitle (`formatSubtitle` appends it, `:491`).
- Quality mode already fails at the job level; verify the popup shows the new plain message.

### F. Purge the poisoned cache (one-off, operator step in the spec, not code)

Delete `~/.patmir/cache/<key>/` for the nine keys listed in Context. Record the key list, URLs
and the cutoff timestamp in `shape.md`. Downloads under `~/.patmir/downloads/` stay (keyed
audience-free, still valid).

### G. Docs (maintenance contract, same change)

- `quality-issues.md`: new row **#35** "A translator outage produced two days of muffled-original
  'dubs' and the extension called them translated" — owner translate (+streaming, extension),
  status `addressed (2026-09-15 — spec: ...)` awaiting a live confirmation; notes carry the
  numbers above and the two-mechanism explanation (batched fallback swallow + adapter never
  failing). Mention #23(a) coverage-fudge interaction (a skipped section is still fudged as
  covered) as still open.
- New ADR `decisions/2026-09-15-translator-outage-fails-loudly.md` + one-line entry in
  `decisions.md`: the refusal-vs-outage boundary (per-line 403/`content: null` -> empty line;
  401/402/404 or all-of-N>=2 empty -> `TranslatorUnavailable`), why no local auto-fallback
  (device fields hardcoded `mps`, cache key would misattribute the translator, quality change
  mid-video), why no preflight, and the purge. Revisit triggers: a one-line `child` refusal
  killing a stream (lower `ALL_EMPTY_MIN_LINES` logic), OpenRouter changing status semantics,
  appetite for a local fallback.
- `current-models.md` translate row: short note on the outage semantics and the
  `sections.json` `lines`/`voiced_lines` fields. `stages/translate/openrouter.md`: same.
- Append to the audience ADR's revisit section that the trigger fired 2026-09-13 and points here.

## Tasks

1. **Save spec documentation** — `agent-os/specs/2026-09-15-2107-translator-outage-fails-loudly/`
   with `plan.md` (this plan), `shape.md` (report, log evidence, decisions, poisoned-key list),
   `references.md` (files/lines above), `standards.md` (none apply — `agent-os/standards/index.yml`
   is empty; say so). No visuals.
2. **Candidates** (A): `context.py` exception; `openrouter.py` `_FATAL` + raise;
   `openrouter_batch.py` `_translate_chunk` / `_fallback_chunk._one`. Tests in
   `patmir/tests/test_translate_batch.py::RetryPolicy`: `test_402_raises_unavailable_without_fallback`
   (exactly 1 request, no per-sentence calls), `test_403_still_falls_back_per_sentence`,
   `test_fallback_one_reraises_unavailable`; per-sentence candidate `test_402_is_unavailable`.
3. **Adapter** (B): all-empty rule + comment rewrite. Tests in `test_audience.py::EmptyLineGuard`:
   `test_all_lines_empty_raises_unavailable`, `test_single_empty_line_is_only_reported`,
   existing partial test unchanged.
4. **Propagation** (C): `stream.py::_process`, `_run_locked` plain message, `sections_failed`;
   `jobs.py` message. Tests: `test_streaming_section.py` / new `test_stream_translator_outage.py`
   — a `process_fn` raising `TranslatorUnavailable` ends the stream `failed` with the plain
   message and releases the accelerator lock (pattern: the cancellation test that asserts a
   second job runs afterwards); a generic exception still skips one section and increments
   `sections_failed`.
5. **Cache guard** (D): `SectionResult` fields, `section_timing`, `_emit` refusal. Tests:
   `test_stream_cache.py` (entry carries `lines`/`voiced_lines`), `_emit` refuses a voiceless
   result and leaves the manifest untouched.
6. **Extension** (E): `content.js`, `lib/session.js`; tests in `extension/tests/session.test.js`
   (`failed` beats `playing`; `pollJob` on `failed` stops the player) and a `content`-level
   assertion where testable via `lib/sync.js` if the status handling is factored into a pure
   helper (`shouldReleaseOnServerStatus(status)`), test in `sync.test.js`.
7. **Purge** (F): delete the nine cache dirs; verify `/cache/lookup` reports `cached: false`
   for one of the URLs.
8. **Docs** (G).

## Verification

- `uv run pytest patmir/tests/test_translate_batch.py patmir/tests/test_audience.py
  patmir/tests/test_stream*.py patmir/tests/test_streaming*.py` green; full suite green.
- `node --test "extension/tests/*.test.js"` green.
- **Live reproduction on the 3090 (PATMIR_PROFILE=cuda):** start `patmir ui` with
  `OPENROUTER_API_KEY` set to a deliberately invalid value (401 — same `_FATAL` path as 402),
  submit a watch-along URL from the extension. Expect: stream `failed` within seconds after the
  prelude, `rec.error` reads `OpenRouter: HTTP 401 ...`, in-page button shows Չհաջողվեց with
  the message, the video keeps playing with **original audio unmuted**, no `section_*.wav`
  written under the new cache key, and `server.log` shows exactly one 401 per attempt (no
  per-sentence storm). Repeat in quality mode: job `failed`, same message in the popup.
- With the real key restored: one watch-along section streams normally and its `sections.json`
  entry carries `lines`/`voiced_lines` with `voiced_lines > 0`.
- Replay the 2026-09-14 section: `patmir dub runs/20260914-210424-section/audio.wav
  --resume-from translate` with the bad key -> translate stage `failed`, no `dub.wav`.
