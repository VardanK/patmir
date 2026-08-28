# Audience guardrail + in-page toggle + ad handling — shaping notes

## Scope

Three additive changes, none of which touches the nine pipeline stages:

1. **Audience age guardrail.** A `child` / `teenager` / `adult` selector in the
   extension options page, sent on every submission, that steers the translator's
   system prompt toward age-appropriate Armenian. Prompt-only softening.
2. **In-page toggle button.** A big button under the YouTube player labelled
   `Թարգմանել` / `Չթարգմանել` driving the same flow the popup drives. Popup keeps
   working and stays in sync.
3. **YouTube ad handling.** Detect ads, suspend and resume the dub across them
   (today a mid-roll kills the dub permanently), and auto-skip where possible.

## Decisions

| Question | Decision |
| --- | --- |
| Guardrail strength | Prompt-only softening. No redaction, no skipped lines, no bleeps. |
| Popup vs button | Both work, kept in sync. |
| Level scope | Global extension setting, sent per submission. No per-video override. |
| Default level | `child` at shaping; **changed to `teenager` on 2026-08-28** at the author's request, after the feature was built. `child` stays one click away. |
| CLI | `--audience` on `patmir dub` and `patmir submit`. |
| Picker UI | Options page only. |
| What `Չթարգմանել` does | Switches the audio track off. Dub stops, original returns, video keeps playing. Never pauses, never seeks. |
| Progress text | Armenian only. No English stage names in the page. |
| Button placement | Watch pages: full-width under the player. Shorts / embeds: floating pill. |
| Ad blocking | Out of scope — detect and skip only. |

### The cache-key consequence, stated up front

`adult` reproduces the pre-feature prompt byte-for-byte, so `adult` is the level that
appends **nothing** to the cache key — existing dubs keep their keys and still describe
what produced them. The **default** is `child`, which does append a suffix. So on first
use after this ships, every video is re-dubbed once. Accepted during shaping; the
alternative is serving adult-register audio under a child-labelled setting.

This splits two constants that the `mode` precedent (`ui/cache.py:146-152`) keeps as one:

    DEFAULT_AUDIENCE = "teenager"  # what a submission gets when it says nothing
    NEUTRAL_AUDIENCE = "adult"     # byte-identical to the old prompt -> no key suffix

Both names exist so the split cannot be collapsed by accident, and a test pins each.

## Rejected alternatives

- **Per-job derived `PipelineConfig` with `audience` inside `entries["translate"]`.**
  Gets the cache key for free via `config_hash`, and cannot work: `stage_cache_key` is
  `(stage, candidate_id)` only (`pipeline/runner.py:43-54`), so the id is unchanged and
  the first job's warm translator is handed to every later job regardless of level. And
  `prewarm_stage_cache` (`ui/app.py:93-124`) builds translate at boot, before any
  submission exists, so there is nothing to bake in. A test in `test_stage_cache.py`
  now pins the invariant this design depends on.
- **`[pipeline.options]` as the only source.** Fails the requirement (per-submission)
  and would need a server restart per level.
- **`sw.js` as a per-tab session controller.** MV3 service workers are killed after
  ~30 s idle and `setInterval` does not survive; the `chrome.alarms` minimum period is
  30 s, useless for a 2 s job poll.
- **Duplicating the job protocol into `content.js`.** The repo already carries this
  exact scar, documented at `popup.js:57-65` and pinned by three tests in
  `bootstrap.test.js` — the injected-file list drifted from the manifest and produced a
  self-inflicted "Receiving end does not exist".
- **`MutationObserver` on `#below`.** YouTube's churniest subtree; thousands of
  callbacks per navigation on the main thread of a video-decoding page. The existing
  500 ms URL poll and `yt-navigate-finish` already cover navigation.
- **`createMediaElementSource`** on the page video — already rejected in ADR
  `2026-08-25-extension-volume-and-av-sync`; irreversible for the element's lifetime.
- **Blocking ads.** Ad segments come from the same `googlevideo.com` CDN and URL shapes
  as content, so `declarativeNetRequest` has nothing clean to match, and MV3 removed
  `webRequestBlocking`. The working technique — a `world: "MAIN"` script patching
  `JSON.parse` / `Response.prototype.json` to strip `adPlacements` — is an arms race
  that breaks on YouTube's schedule, draws playback stalls that would be blamed on the
  dub, and rewrites the very player response the dub's drift correction anchors to.
  With uBlock Origin or Premium installed, our ad handling degrades to a no-op, so the
  two compose rather than compete.

## Context

- **Visuals:** none provided.
- **References:** see `references.md`.
- **Product alignment:** `agent-os/product/` mission is a local-first workstation tool
  for one viewer. A child-safe register and an in-page control are both squarely that;
  neither adds a service dependency, and the guardrail is prompt-side so it costs no
  extra model.

## Standards applied

`agent-os/standards/index.yml` is an empty stub, so no numbered standards apply. The
de-facto conventions this work follows are recorded in `standards.md`.
