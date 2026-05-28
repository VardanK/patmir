# Patmir Chrome extension (slice 1)

One-click Eastern Armenian dub for YouTube videos, backed by your local Patmir server.

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

## What works in slice 1

- URL-cache lookup on the active tab (`POST` / `GET /cache/lookup`)
- Job submission + status polling (`POST /jobs` / `GET /jobs/{id}`)
- In-page Web Audio overlay with play / pause / seek / playbackRate following
- Loopback-only server with bearer-token auth
- Local files via `patmir submit <file>` CLI

## Not yet (slice 2 / 3)

- SSE-driven progress (currently 2 s polling)
- Desktop notifications on job completion
- DRM detection (Netflix / Prime / Disney+ — not feasible regardless)
- ±80 ms drift correction loop
- macOS Finder Quick Action
- Subtitle overlay from `transcript.json`

## Known gotchas

- **Mixed content**: YouTube is HTTPS; `http://127.0.0.1` fetches are routed through the service worker (`sw.js`) because the page context blocks them. Audio bytes come back as `Uint8Array` via `chrome.runtime.sendMessage`.
- **Drift**: slice 1 has no active correction. If `<video>.playbackRate` changes mid-clip the dub will drift; toggle Stop / Play to resync. Slice 2 adds an automatic correction loop.
- **YouTube SPA navigation**: clicking from one video to the next does not re-trigger the content script. Reload the page to attach Patmir to the new video. Slice 2 will add a `yt-navigate-finish` listener.
