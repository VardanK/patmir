# The download inside the accelerator lock, and a cancel that missed `waiting`

Date: 2026-09-04. Status: accepted. Quality issue #33.

## The report

Author, on a live watch-along: *"it's stuck. A moment ago it was saying that
it's waiting for GPU and after page refresh it's stuck on loading. It is the
single client, no other client tries to translate."* Reported as a scheduling
and cancellation bug, which it was — but the thing holding the GPU slot was
not GPU work.

## What the server was doing

Measured on the running process (pid 321, `--profile cuda`), GPU at 4 %
utilization with 16.7 GB of resident weights and nothing running:

| record | submitted | status | what it was doing |
| --- | --- | --- | --- |
| `stream-1788549907196-f95d94` | 23:25:07 | `cancelling`, `download_pct 1.0`, `download_s None` | inside `yt_dlp.extract_info`, **holding the accelerator lock**, for 6+ minutes |
| `stream-1788550026871-f95d94` | 23:27:06 | `cancelling`, `started_at None` | parked on the lock; a "cancelled" record that would have run in full |
| `stream-1788550053988-f95d94` | 23:27:33 | `waiting` | what the client was polling |

Same video, one viewer, three records. The URL was
`watch?v=x2Yl0ALK3-4&list=RDx2Yl0ALK3-4&start_radio=1` — a YouTube Mix.

### Cause 1: the URL was a playlist to yt-dlp and one video to us

`canonicalize_url` reduces any watch URL to `yt:<videoId>`, so the cache key
and the download dir are per video. `_download_url` passed no `noplaylist`, so
yt-dlp took the `list=` half and walked the Mix entry by entry. Evidence, in
the download dir `736569b1cfc3f23a`: **two files, `source.m4a` (3.5 MB) and
`source.webm` (4.6 MB), both written 23:25** — two different videos' audio
under one `outtmpl`. Every later entry whose extension matched an existing
file was skipped as "already downloaded", so nothing was written after 23:25
while extraction went on. A Mix is effectively endless. The progress hook's
`finished` fired after the *first* entry, so the record read "download
complete" for the rest of the hang, which is what the popup showed.

The download runs inside the accelerator lock — recorded as a "known wart,
wasteful not incorrect" in `2026-08-25-one-accelerator-at-a-time` — and the
cancel token was never consulted inside it, recorded as a known limit in
`2026-08-25-cooperative-job-cancellation`. Both limits were priced against a
download that ends. This one did not.

### Cause 2: `StreamManager.cancel` did not know `waiting`

`JobManager.cancel` treats `queued` and `waiting` alike as not-started:
instant, exact, and the worker skips the body when the lock frees.
`StreamManager.cancel` had only `queued`. A `waiting` stream fell into the
running branch, found no token (none exists before `_run_locked`), and was
left `cancelling`; `_run` skips only `cancelled`, so the moment the lock was
released that stream would have started from scratch — a stop that had been
acknowledged with HTTP 200 and then ignored.

### Cause 3: `stop_stream` reached only the newest record

`_by_cache_key` points at the latest submission, and `submit()` creates a new
record whenever the existing one is `cancelling`. So after one Stop +
re-submit the record actually holding the lock was unreachable from
`/stream/<key>/stop`, and each further press stopped only the newcomer.
`submit()` also did not count `waiting` as live, so a re-submit during the
wait minted another record behind the first.

## Decision

1. **`noplaylist: True`.** A URL that names both a video and a playlist is
   the video. This matches the cache key's own definition of the source.
2. **A bare playlist, channel or home-page URL fails fast** with a stated
   reason, instead of yt-dlp fetching N entries and `prepare_filename`
   producing `source.NA` (observed the same evening: the extension submitted
   `https://www.youtube.com/`).
3. **The cancel token reaches into the download.** yt-dlp's abort mechanism is
   a `DownloadCancelled` raised from a hook, which `extract_info` re-raises
   verbatim; we raise it from the per-entry `match_filter` and the per-chunk
   progress hook, and translate it to `JobCancelled` at the boundary so the
   record reads `cancelled`, not `failed`. Checked before the download starts
   too. Both managers pass their token.
4. **`waiting` is not-started** in `StreamManager.cancel`, as it always was in
   `JobManager.cancel`; `submit()` returns a `waiting` record rather than
   duplicating it.
5. **`stop_stream(ck)` cancels every live record for the key.**

Not done: moving the download outside the lock. It would have shortened this
hang by nothing — the stuck call would still have been stuck, just without a
lock — and it changes the ordering semantics the one-accelerator ADR chose
deliberately. It remains the right next step if downloads of long videos are
felt as GPU dead time; that is now a measurable wait rather than a hang.

## Verification

- 7 new tests in `test_cancellation.py`: a `waiting` stream cancels instantly
  and its body never runs; `stop_stream` reaches an older record holding the
  token; `_download_url` sets `noplaylist` and wires both hooks; a token set
  before the download aborts without constructing `YoutubeDL`; a token set
  mid-transfer surfaces as `JobCancelled`; a playlist result raises; the
  no-token path downloads as before. Full suite: 780 passed.
- The stuck process itself cannot be recovered by this change — a Python
  thread in a C-level network wait cannot be interrupted from outside — so
  the running server has to be restarted once to clear it.

## Also in this change: the in-page button during start-up

The author added: *"the green progress does not appear (the button stays gray
while loading)."* Two mechanisms, both extension-side. A watch-along record
has no `stage_states`, so `runningStage` never saw its download and the
button read "prelude" with no percentage for the whole fetch; the record does
carry `live_stage`/`download_pct` from the yt-dlp hook, so that is now the
`downloading` kind with its own `stagePct` in the subtitle. And every busy
state with nothing measurable (submitting, queued, waiting, prelude, 0 %
streamed) rendered the fill hidden — a flat grey button. It now shows a
sweeping band. A percentage was rejected for those states because it would
have to fall back to ~1 % when the first section of a long video lands, which
is the "progress bar goes backwards" failure the code already guards against.
5 new JS tests; 253 pass. Extension version 0.2.1.

*Revisit:* the download hold being felt as GPU dead time on long videos (move
it ahead of the lock); yt-dlp changing how hooks abort; a legitimate need to
dub a whole playlist.
