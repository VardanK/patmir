# The in-page toggle, and what ads were doing to the dub

**Date:** 2026-08-28
**Status:** accepted
**Spec:** `agent-os/specs/2026-08-28-0033-audience-guardrail-and-inpage-toggle/`

## Context

Two things, related only by living in the extension.

**1. Every control was in the toolbar popup.** To start a dub the viewer had to
leave the video, open the extension menu and read English labels. The ask was a
big button under the YouTube player, labelled `Թարգմանել` / `Չթարգմանել`, doing
what the popup CTA does, with the popup still working and in agreement.

**2. A mid-roll ad killed the dub permanently.** Ads play through the *same*
`<video>` element as the content. Two consequences, and the second was fatal:
`currentTime` during an ad is the ad's timeline, so every anchor computed from it
is wrong; and the src swap fires `emptied`/`loadstart`, which
`onVideoSourceChanged` read as "the viewer navigated to another video" and
answered with a full `stopDub()`. So an unskippable mid-roll ended the dub for
the rest of the video, with no path back short of pressing play again. This was
not reported as a bug — it was found while adding ad detection for the button —
and it is filed as quality issue #26.

## Decision

### One session, owned by the content script

The job protocol — which statuses are terminal, that `waiting` means the
accelerator lock, that `download_pct` is special-cased, that `live_pct` is only
meaningful when `live_stage` matches the running stage — moves out of `popup.js`
into `extension/lib/session.js`, following the `lib/sync.js` pattern exactly: no
DOM, no timers, no `chrome.*`, everything through an injected transport, so the
whole state machine is driven by hand under `node --test`.

On a YouTube tab, the content script owns the only session and the popup renders
it over two messages (`patmir:session-state`, `patmir:session-command`). That
makes "whichever surface you touch, both agree" true **by construction**: there
is no second state to reconcile. The popup cannot delegate unconditionally — it
also works on any http(s) page, where no content script is injected — so it
keeps a second path running the *same* module locally with `NULL_PLAYER`,
reproducing its pre-button behaviour off-YouTube exactly.

Duplicating the protocol into `content.js` was rejected: this repo already has
that scar, where the injected-file list drifted between the manifest and
`popup.js` and produced a self-inflicted "Receiving end does not exist". A
service-worker-owned controller was also rejected — MV3 workers are killed after
~30 s idle and `setInterval` does not survive, while the `chrome.alarms` floor
of 30 s is useless for a 2 s job poll.

### The DOM lives in a separate content script

`extension/overlay.js`, listed after `content.js`, consuming a
`globalThis.PatmirPlayer` facade that `content.js` publishes at the very end of
its IIFE. This keeps every `document` mutation out of `content.js`, so its
sandbox test still boots with **no DOM globals at all** — which is the
regression guard that stops button code creeping back into the file that owns
the audio graph, the volume arbitration and the drift tick. Manifest order is
now load-bearing three times (`sync` < `content`, `session` < `overlay`,
`content` < `overlay`) and each edge is asserted.

`content.js` changed in three small ways: the message-handler bodies were
extracted into functions so the overlay reaches *exactly* the same code path
(critical for the `setVideoDisabled` bookkeeping — a stop that skipped it would
be undone by the auto-play listener within a tick, and the button would look
broken); an `onUrlChange` fan-out was added to its existing 500 ms poll so the
overlay needs no second poll; and its listener stopped returning `true` for
messages it does not own, which had been claiming a response port for the
overlay's messages and never closing it.

### Placement and labels

Watch and live pages get a full-width button under the player, mounted by an
ordered probe (`#below` → `#primary-inner > #player` → `ytd-watch-metadata` →
`#primary-inner` → walk up from `findVideo()`), with the policy as a pure
`mountPlan(present)` over booleans so it is testable without jsdom. Nothing
resolving means **no button**, not a guess at `document.body`.

Shorts and embeds get a compact fixed-position pill instead: Shorts is a
recycler that re-parents nodes as you scroll, and an embed is someone else's
article layout — neither can take a full-width block.

Everything lives in a shadow root with a constructable stylesheet. A shadow
boundary *removes* the possibility of style bleed in both directions, where
scoped class names only lower the odds, and with it the temptation to fight with
`!important`. YouTube's theme still reaches in through custom properties, so
dark mode is free.

Re-mounting is driven by `content.js`'s existing URL poll, `yt-navigate-finish`,
and one owned 500 ms tick doing `document.contains`. **No MutationObserver:**
`#below` is YouTube's churniest subtree and a subtree observer there fires
thousands of times per navigation on the main thread of a video-decoding page,
to re-mount one frame sooner than a `contains` check.

The label is only ever one of four Armenian words, and **progress never enters
the label** — a 52 px full-width control whose text changes every two seconds
cannot be aimed at. Progress goes in an Armenian-only subtitle as a single
percentage: the viewer is a child watching a video, not an operator reading a
pipeline. `Թարգմանել` deliberately unifies "submit a job" and "play the cache",
because from the viewer's side both are the same request.

`Չթարգմանել` switches the **audio track** off and nothing else: the dub stops,
the mute control hands `video.muted` back to the viewer's own intent so the
original returns, and the picture keeps rolling. It never pauses and never
seeks. It also sets the per-video opt-out, which is both correct ("off" should
stay off) and load-bearing (without it the auto-play listener restarts the dub
within a tick). `jobs:cancel` was added to `sw.js` so switching off mid-job also
stops the GPU, which had no path from the extension at all.

### Ads: suspend, resume, and skip

`classifyAd` and `adSkipPlan` are pure functions in `lib/sync.js`; `content.js`
reads the DOM signals (`ad-showing` on `#movie_player` primarily, with the
`.ytp-ad-*` overlay nodes corroborating) from the sync tick that already runs.
No new timer.

On ad start the dub is **silenced, not torn down** — listeners, decoded buffer,
`playing` and the mute control all stay, which is the whole difference between
"the dub pauses for the ad" and "the dub is gone". `shouldScheduleAudio` gained
an `adShowing` term, checked before `readyState` because during an ad the
element is perfectly healthy and simply playing something else. The
source-change guards now exempt an ad's swap. Watch-along stops posting the
playhead during an ad, because an ad's `currentTime` starts near zero and the
scheduler would read it as a seek to the beginning and re-dub covered sections.

Auto-skip clicks the skip control when it becomes live — the viewer's own
gesture, a second early (`patmirSkipAds`, default on). Seeking to the end of an
unskippable ad is a **separate opt-in** (`patmirSkipAdSeek`, default off).

### The seek shipped broken, and why the guard did not save it

First release: `classifyAd` treated any signal as sufficient — the player class
*or* the presence of `.ytp-ad-player-overlay` / `.ytp-ad-module` — reasoning
that "a false positive costs a moment of silence, a false negative costs the
desync". Both halves were wrong.

Those overlay nodes are **containers YouTube ships on every watch page and
leaves in the DOM when no ad is running**, so the predicate returned true on
every ordinary video, permanently. And a false positive does not cost silence:
it authorizes the seek. Every video the author opened jumped to its end, three
times over (the attempt cap was the only thing that stopped it looping).

`applyAdAction` re-checked `showing` against live DOM immediately before acting,
exactly as designed — and that was worthless, because it re-checked *the same
broken predicate*. **A guard that re-evaluates the wrong signal is not a guard.**
The fix is therefore at the signal, not the guard:

* Only `ad-showing` / `ad-interrupting` on `#movie_player` can establish
  `showing`. The overlay nodes are demoted to corroboration, reported as
  `corroborated`, and can never originate an ad — and the caller must now pass
  "present **and** non-empty", not "the selector matched".
* `classifyAd` reports `classConfirmed` separately, and **only that** may
  authorize a seek.
* The seek additionally requires an opt-in, a duration under `MAX_AD_SECONDS`
  (180 — a clip longer than that is not an ad, so believing it is means the
  detection is wrong), and a duration that does not match the one recorded for
  this page's content video while no ad was showing. Three independent
  conditions, any of which refuses.
* `offsetParent !== null` was dropped as the skip-button visibility test: it is
  null inside any `position: fixed` ancestor, which would make a live skip
  button look unclickable and push us onto the seek path — the dangerous one.

The general lesson, written down because it generalizes past this feature: the
asymmetry of consequences has to reach the predicate. A signal that can trigger
a destructive action does not get to be liberal "just in case", and the place to
encode that is the classifier, not a re-check downstream of it.

Related: `tick()` ran mounting, ad handling and rendering inside one try/catch,
so a single throw in the mount probe silently disabled ad handling on every
subsequent tick. Each concern is now guarded independently and warns once.

**Blocking ads is deliberately out of scope.** Ad segments come from the same
`googlevideo.com` CDN and URL shapes as content, so `declarativeNetRequest` has
nothing clean to match, and MV3 removed `webRequestBlocking`. The technique that
does work — a `world: "MAIN"` script patching `JSON.parse` /
`Response.prototype.json` to strip `adPlacements` — is an arms race that breaks
on YouTube's schedule, draws playback stalls that would be blamed on the dub,
and rewrites the very player response the drift correction is anchored to. With
uBlock Origin or Premium installed, everything here degrades to a no-op:
`ad-showing` never appears, nothing suspends, nothing skips. The two compose
rather than compete.

## Consequences

* 126 new JS tests (168 total). Two real bugs were found by them before
  shipping: `deriveView` read through a null job record on the very first press,
  and a `mountedAt` of `0` was treated as "unknown", which kept the video grace
  window open forever and left the button permanently disabled on a page with no
  player.
* A large always-visible button makes an accidental watch-along start (which
  costs ElevenLabs credit) likelier than a popup CTA did. The not-cached
  subtitle names the mode; a confirm step is still worth considering.
* Regenerate stays popup-only and two-click-armed: it deletes a finished dub and
  reruns a multi-minute pipeline, which has no business being one press away on
  the page.
* Nothing in the audio graph, the volume arbitration (`resolveVolumeEvent`,
  `pendingForcedWrites`, `dubGain`) or the 250 ms drift tick was touched. The
  only new coupling is the four-method facade, which reuses the existing handler
  bodies verbatim.

## Revisit

* YouTube renames the player class or the skip control (both are pinned in one
  place each: `adState()` and `SKIP_SELECTORS`).
* The floating pill proves intrusive on Shorts, or the mount probe starts
  missing on watch pages — `mountPlan` is where to look, and it is pure.
* An ear-test finds the ad suspend/resume re-anchor audible at the resume point.
