# Patmir Chrome extension

One-click Eastern Armenian dub for YouTube videos, backed by your local Patmir server.

## The button under the player

On a YouTube watch page a full-width button appears under the video:

- **Թարգմանել** — give me the Armenian. One word for both "run the pipeline" and
  "play the dub you already have"; which one it is shows in the small line under
  the label, as an Armenian-only percentage (no stage names — the viewer is
  watching a video, not reading a pipeline).
- **Չթարգմանել** — switch the Armenian off. The dub stops and the original audio
  comes back; the **video keeps playing**, and is never paused or seeked. It also
  remembers the choice for that video, so it will not auto-start next time.

Where it appears: watch and `/live` pages get the inline button. **Shorts and
embeds get a compact floating pill instead** — Shorts is a recycler that
re-parents nodes as you scroll and an embed is someone else's layout, so neither
can take a full-width block. On any other page there is no button and the popup
does the job.

The popup keeps working and the two always agree: on a YouTube tab the content
script owns the only session and the popup is a window onto it, so pressing
either flips both. Regenerate stays popup-only (and two-click-armed) — it deletes
a finished dub and reruns a multi-minute pipeline.

## Audience age guardrail

The options page has an **Audience** selector: `child` (about 5-10), `teenager`
(about 11-15, the default) or `adult`. It steers the translator toward age-appropriate
Armenian wording — **nothing is ever cut, muted or bleeped**, every line is still
spoken, in gentler words. `adult` is exactly the behaviour that existed before
the selector.

The level is sent on every lookup and submit (attached in `sw.js`, so the two can
never disagree) and folded into the cache key, so **each level caches separately
and switching re-translates the video once**. The extension's value overrides
`[pipeline.options].audience` in `patmir.toml`, which is the default for
`patmir dub` / `patmir submit`.

## Ads

Ads share the page's single `<video>` element, so the extension handles them
explicitly:

- The dub is **suspended** for the duration of an ad and resumes afterwards.
  (Before this, a mid-roll ad ended the dub for the rest of the video — the
  source swap read as "the viewer navigated away".)
- **Auto-skip** (on by default) presses *Skip Ad* as soon as it goes live.
- **Jumping to the end of an unskippable ad** is a *separate* setting and is
  **off by default**. It is the only thing here that writes to the player's
  position, so if ad detection is ever wrong it moves the video you are actually
  watching — which is precisely what the first release did, on every video, via
  a predicate that mistook YouTube's always-present empty ad containers for an
  ad. It now needs YouTube's own player class, a clip short enough to be an ad,
  and a length that does not match the video you opened.

If ad handling misbehaves, paste `tools-ad-probe.js` into the YouTube tab's
console and let an ad play: it logs what the detector can see once a second, so
a selector that YouTube has renamed shows up as data rather than a guess. (That
file is a developer tool, not part of the extension — it is not in the manifest
and is never injected.)

Ads are not *blocked* — that is out of scope, and deliberately so; see the ADR
`docs/architecture/decisions/2026-08-28-inpage-toggle-and-ads.md`. If you run
uBlock Origin or have Premium, none of the above ever fires and everything still
works.

## Two dub modes (popup toggle)

- **Quality** (default, D1) — runs the full batch pipeline → one cached `dub.wav`.
  Best quality, one-time wait, instant rewatch. *Translate this → Play dub.*
- **Watch-along** (D3) — analyzes a first-chunk prelude (~45–60 s) to establish the
  cast, then streams the dub in VAD-snapped sections behind a short buffer while you
  watch. ~20–30 s to first audio; the video auto-pauses to buffer if the playhead
  outruns production. Sections are cached as they land, so a rewatch is free.
  *Start watch-along.* Uses the ElevenLabs `eleven_v3` preset voice pool (no source
  cloning). See ADR `docs/architecture/decisions/2026-06-13-near-realtime-d3-spike.md`.

The chosen mode is sent on every lookup + submit and folded into the cache key, so the
two modes cache independently.

## Install (unpacked)

1. Start the local server:

   ```
   patmir ui --port 8765
   ```

   On first start it generates `~/.patmir/secret` and prints the token on stderr.

2. Open `chrome://extensions`, enable **Developer mode**, click **Load unpacked**, and select this `extension/` directory.

3. Right-click the Patmir toolbar icon → **Options**. Paste the bearer token,
   pick an **Audience** level, and Save.

4. Open a public YouTube video. Press **Թարգմանել** under the player (or use the
   toolbar popup — either works).
   - First time: the pipeline runs (~7-8 min for a 5-min clip) with progress under
     the label.
   - Done: the dub starts by itself and the button reads **Չթարգմանել**. The
     `<video>` element is held muted; the Armenian plays in-page, synced to
     `<video>.currentTime`.
   - Rewatch: cache hit, so it is immediate.

## What works

- URL-cache lookup on the active tab (`GET /cache/lookup?...&mode=`)
- Job submission + status polling (`POST /jobs` / `GET /jobs/{id}`), both modes
- Quality: in-page Web Audio overlay with play / pause / seek / playbackRate following
- Watch-along: chunked, offset-addressed Web Audio feed (`GET /stream/<key>/sections`,
  `/section/<i>.wav`), buffer-gated playback, playhead/seek/pause reporting
  (`POST /stream/<key>/playhead`)
- Both modes: the page volume slider drives the dub and the page mute button mutes
  it, while the original stays silenced no matter what the player does to `muted`
- Both modes: continuous A/V drift correction on a shared 250 ms tick — audio only
  starts once the picture is confirmed moving, small errors are walked out by an
  inaudible resampling nudge, and a hard re-anchor needs three agreeing samples
- Loopback-only server with bearer-token auth
- Local files via `patmir submit <file>` CLI
- An in-page `Թարգմանել` / `Չթարգմանել` button on watch pages, a floating pill on
  Shorts and embeds, in a shadow root so YouTube's CSS cannot reach it
- Ad detection: the dub suspends and resumes across an ad break, and ads are
  auto-skipped where the player allows it
- A `child` / `teenager` / `adult` audience guardrail on the translator's register

## Tests

Pure decision logic lives in `lib/` so it can be tested without a browser:

- `lib/sync.js` — volume/mute arbitration, drift policy, section selection, and
  the ad predicates (`classifyAd`, `adSkipPlan`).
- `lib/session.js` — the whole job state machine and the button's view
  derivation. No DOM, no timers, no `chrome.*`: the transport is injected, so the
  tests drive it by hand with a fake clock.

```
node --test "extension/tests/*.test.js"

## Reaching a server on another machine

The manifest grants `http://127.0.0.1:8765/*` and nothing else, so pointing the
extension at a server elsewhere on the network used to fail *inside the
browser* — the fetch never left, and the symptom was indistinguishable from a
server that was down.

The permission now follows the address you configure:

1. Start the server bound to the network: `patmir ui --profile cuda --host 0.0.0.0`
   (it binds loopback by default; see `docs/setup/windows-wsl2-cuda.md` for the
   WSL2 side, which needs `networkingMode=mirrored` and a firewall rule).
2. In the options page set the server to `http://<that machine>:<port>` and
   paste the token, then press Save. Chrome asks once for access to that host;
   granting it is what makes the extension able to talk to it.

If you decline, **nothing is saved** — deliberately, because an address stored
without permission is a configuration that looks right and cannot work.

The requested pattern is host-only (`http://192.168.1.50/*`, no port): Chrome's
match-pattern grammar has no port component, so a host-only pattern is
unambiguous and covers whichever port the server runs on. `127.0.0.1` and
`localhost` need no prompt — they are already in the manifest.
```

(On Node 26 the bare-directory form was dropped; use
`node --test "extension/tests/*.test.js"`.)

No dependencies beyond Node itself. **Manifest load order is load-bearing three
times over** and each edge is asserted by `tests/bootstrap.test.js`:

```
lib/sync.js  <  content.js      (PatmirSync)
lib/session.js  <  overlay.js   (PatmirSession)
content.js   <  overlay.js      (PatmirPlayer, published at the end of content.js)
```

`content.js` deliberately touches **no** DOM-construction globals — all element
creation is in `overlay.js`. A test boots `content.js` in a sandbox with no
`createElement`, no `MutationObserver` and no `CSSStyleSheet` precisely to keep it
that way, because that file owns the audio graph, the mute arbitration and the
drift tick and should not also own a button.

## Not yet

- SSE-driven progress (currently 2 s polling)
- Desktop notifications on job completion
- DRM detection (Netflix / Prime / Disney+ — not feasible regardless)
- macOS Finder Quick Action
- Subtitle overlay from `transcript.json`

## Known gotchas

- **Mixed content**: YouTube is HTTPS; `http://127.0.0.1` fetches are routed through the service worker (`sw.js`) because the page context blocks them. Audio bytes come back as `Uint8Array` via `chrome.runtime.sendMessage`.
- **Drift**: corrected automatically (2026-08-25). `lib/sync.js` owns the policy — a dead band below 40 ms, a ±0.5 % resampling nudge up to 250 ms, and a hard re-anchor above that only after three agreeing samples. The thresholds are wider than the ±80 ms the original spec asked for on purpose: `video.currentTime` is coarsely quantized, and a hard re-anchor is an audible cut, so a tight threshold re-anchors on noise. Toggling Stop / Play is no longer needed.
- **Volume**: while a dub is playing the `<video>` element is held muted and the page's slider drives a GainNode on the dub instead. The player's UI will therefore show the video as muted — that is deliberate, and the mute button still works (it mutes the dub).
- **The button is missing on a watch page**: nothing resolved among the five mount
  points, which is deliberate — guessing a parent on someone else's page is worse
  than showing no button, and the popup still works. `mountPlan` in
  `lib/session.js` is where the probe order lives.
- **A manifest change needs a full Reload** on `chrome://extensions`, not a page
  refresh — and this release adds two files to `content_scripts`, so the first
  load after updating must be a reload or the in-page button will log that
  `lib/session.js` / `PatmirPlayer` is missing and disable itself.
- **YouTube SPA navigation**: handled — `yt-navigate-finish` plus a 400 ms URL/element poll as a backstop for Shorts, which uses a recycler and swaps `<video>` elements without always firing the event.
