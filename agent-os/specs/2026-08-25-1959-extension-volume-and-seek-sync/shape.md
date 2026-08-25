# Extension volume + seek sync — Shaping Notes

## Scope

Two bugs the author reported against the MV3 Chrome extension in `extension/`:

1. **Changing the audio volume mixes the original audio back in with the dub.**
2. **Seeking while the audio is being dubbed puts the dub out of sync with the video.**

Both were confirmed to affect **both** playback modes — Quality (D1, one cached
`dub.wav`) and watch-along (D3, streamed sections). Both live primarily in one file,
`extension/content.js` (649 lines, the entire client).

The second bug turned out to have a server-side half as well: `StreamScheduler` emits
sections whose spans overlap after a seek, so the client has two renderings of the
same seconds to choose between. That half is in scope.

Explicitly **not** in scope: VAD-snapping the seek start in the mux. It is a plausible
second contributor to post-seek drift, but the evidence for it is a hypothesis rather
than a measurement, and `mux_placements.json` already carries the data needed to
decide. Measure first.

Also out of scope, filed rather than fixed: the scheduler's coverage fudge, the
removed-but-still-documented backfill, the >45 s pause that kills a stream, and
double-dubbed boundary-straddling sentences. See `plan.md` § "Out of scope".

## Decisions

1. **Both modes.** They share the mute mechanism and the `audioCtx`, so a fix in one
   that skipped the other would be half a fix.
2. **The page volume slider drives the dub, via a GainNode.** The alternative
   considered — just re-assert `video.muted = true` and leave the dub at fixed full
   volume — silences the original but leaves the viewer with a slider that does
   nothing audible. Rejected.
3. **YouTube's mute button mutes the dub.** A viewer who presses mute and still hears
   Armenian reads it as broken.
4. **Continuous drift correction, not seek-only.** A one-shot re-anchor at seek time
   would fix the reported symptom and leave slow clock drift over a feature-length
   film unfixed.
5. **Discriminate our own mute write with a synchronous pending-write counter**, not
   a timeout. The file already has a timeout-based guard for the analogous problem
   (`waProgrammatic`, 400 ms), and a 400 ms window can swallow a genuine viewer mute.
   The counter is exact because `forceMute` only writes when `muted` is currently
   `false`, and that write always emits exactly one `volumechange`.
6. **`createMediaElementSource` was considered and rejected** for silencing the
   original. It would remove the write-back war entirely, but it is irreversible for
   the element's lifetime, survives our teardown, and would leave the page video
   permanently silent if our AudioContext were closed. Too much blast radius on
   someone else's page for the problem it solves.
7. **Delete the `originalVolume` save/restore rather than fix it.** Since we never
   write `video.volume`, the element already holds the viewer's choice. This also
   disposes of a latent bug: the singleton is never reset to `null`, so after an SPA
   navigation a later stop wrote the *first* video's volume onto a different element.
8. **Tests via Node's built-in runner**, no npm dependencies. Requires installing Node
   in WSL — the box has only a Windows `node.exe`, which runs `--test <file>` from a
   WSL cwd but cannot walk a WSL directory.
9. **Measure the post-seek mux drift before changing the mux.** Consistent with how
   this project has handled every other audio-quality hypothesis.

## Context

- **Visuals:** none provided. The author opted to have the repro steps written from
  the code rather than supplying a recording, so `plan.md` carries a 10-step manual
  checklist instead of a `visuals/` folder.
- **References:** see `references.md`.
- **Product alignment:** `agent-os/product/roadmap.md` Phase 3 carries the hard rule
  that Mode B (watch-along) must not compromise Mode A quality. Both halves of this
  work respect it — the client fixes are shared by both modes, and the server change
  narrows what the scheduler emits rather than changing how anything is dubbed.

## Standards Applied

None. `agent-os/standards/index.yml` contains only a header comment with no entries.
See `standards.md` for the conventions that govern instead.

## Rebase note (2026-08-25, after `7f8d4a9`)

`origin/main` moved twice while this spec was being executed. The second move
(PR #5, watch-along startup latency and stall causes) landed in the same files
and changed three things this spec had planned:

1. **The resumed section-index fix was done upstream**, as `_index_seed`, with
   real evidence (corrupted cache key `d5af5653d745ed66`) and 15 tests. Our
   duplicate `_seed_section_index` and its 6 tests were **dropped on rebase**.
   The `max_end` span clamp is unaffected and still needed — see the ADR §6 for
   why the two are complementary rather than redundant.
2. **The >45 s pause that killed a stream was fixed upstream** (their #21). It
   had been filed here as part of the out-of-scope row and is removed from it.
3. **`_snap_window_end` was rewritten** (their #20). Our clamp feeds it, so the
   interaction was re-checked; it is sound, and the reasoning is recorded at the
   clamp site in `section.py` and in the ADR's Verification section.

Quality-issue numbers also collided: upstream took 20/21/22, so this spec's rows
were renumbered to **25 (volume), 24 (seek), 23 (coverage holes)**.
