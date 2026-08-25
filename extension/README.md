# Patmir Chrome extension

One-click Eastern Armenian dub for YouTube videos, backed by your local Patmir server.

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

3. Right-click the Patmir toolbar icon → **Options**. Paste the bearer token. Save.

4. Open a public YouTube video. Click the extension icon.
   - First time: popup says **Translate this** → click. The pipeline runs (~7-8 min for a 5-min clip). Popup polls every 2 s.
   - Done: popup says **Play dub** → click. Video mutes; Armenian dub plays in-page, synced to `<video>.currentTime`.
   - Rewatch: cache hit; popup immediately shows **Play dub**.

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

## Tests

Pure decision logic (volume/mute arbitration, drift policy, section selection) lives
in `lib/sync.js` so it can be tested without a browser:

```
node --test extension/tests/
```

No dependencies beyond Node itself. `lib/sync.js` is listed **before** `content.js` in
the manifest's `content_scripts` and must stay that way.

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
- **YouTube SPA navigation**: handled — `yt-navigate-finish` plus a 400 ms URL/element poll as a backstop for Shorts, which uses a recycler and swaps `<video>` elements without always firing the event.
