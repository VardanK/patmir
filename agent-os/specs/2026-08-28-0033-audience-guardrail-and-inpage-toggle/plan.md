# Audience guardrail, in-page translate toggle, and YouTube ad handling

## Context

Three viewer-facing gaps, all about the person actually watching.

**1. Nothing constrains the register for a child viewer.** Patmir dubs foreign video
into Eastern Armenian for the author's hearing-aid-wearing son, but the translator's
system prompt (`_SYSTEM`, `patmir/eval/candidates/translation/openrouter.py:34`) asks
only for "casual spoken Eastern Armenian… playful, cartoon register". Profanity, slurs,
innuendo and graphic violence pass straight through, and the dub *is* the only audio
track this viewer hears. We add a `child` / `teenager` / `adult` selector in the
extension options that steers word choice — prompt-only softening, meaning preserved,
no line ever dropped. A silent gap is worse for this viewer than a gentler word.

**2. Starting a dub requires the toolbar popup.** Every control lives in `popup.html`,
so the viewer must leave the video, open the extension menu, and read English labels.
The ask is a big button under the YouTube player labelled `Թարգմանել` /
`Չթարգմանել`, driving the same flow, with the popup still working and in sync.

**3. Ads break the dub, permanently.** During a YouTube ad the `<video>` element's
`currentTime` belongs to the ad, not the content, and the src swap fires `emptied` /
`loadstart` — which `onVideoSourceChanged` (`extension/content.js:482-491`) reads as
"the viewer navigated away" and answers with a full `stopDub()`. So today an
unskippable mid-roll **kills the dub for the rest of the video**, and there is no path
back short of pressing Play again. The dub must suspend for the ad and resume after.
And, since we are detecting ads anyway, auto-skip them.

Ads are purely a client-side playback problem: the server downloads with yt-dlp, which
never sees an ad, so nothing in the nine pipeline stages is affected. Worth stating so
nobody goes looking there.

## Decisions taken during shaping

| Question | Decision |
| --- | --- |
| Guardrail strength | Prompt-only softening. No redaction, no skipped lines, no bleeps. |
| Popup vs button | Both work, kept in sync. |
| Level scope | Global extension setting, sent on every submission. No per-video override. |
| Default level | **`child`** — the product's actual viewer. |
| CLI | `--audience` on `patmir dub` and `patmir submit`, for ear-testing. |
| Picker UI | Options page only. |
| What `Չթարգմանել` does | Switches the **audio track** off — dub stops, original comes back, **video keeps playing**. Never pauses, never seeks. |
| Progress text | Armenian only (`Թարգմանում… 45 %`). No English stage names in the page. |
| Where the button appears | Watch pages: full-width under the player. Shorts / embeds: a floating pill. |

### The one consequence to state up front

`adult` reproduces today's prompt byte-for-byte, so **`adult` is the level that appends
nothing to the cache key** — existing dubs keep their keys and still describe what
produced them. But the **default is `child`**, which does append a suffix. So on first
use after this ships, every video is re-dubbed once. Intentional, and accepted during
shaping: the alternative is serving adult-register audio under a child-labelled setting.

This deliberately splits two constants that the `mode` precedent (`cache.py:146-152`)
keeps as one. Both names exist so the split cannot be collapsed by accident:

```python
DEFAULT_AUDIENCE = "child"    # what a submission gets when it says nothing
NEUTRAL_AUDIENCE = "adult"    # byte-identical to the pre-feature prompt -> no key suffix
```

---

## Task 1 — Save spec documentation (runs first)

`agent-os/specs/2026-08-28-<HHMM>-audience-guardrail-and-inpage-toggle/` containing
`plan.md` (this file), `shape.md` (scope, the nine decisions, the cache-key
consequence, and the rejected alternatives listed below), `standards.md`,
`references.md`, and an empty `visuals/`.

`agent-os/standards/index.yml` is an empty stub (header comment only), so
`standards.md` records that and lists the de-facto conventions instead: pure logic in
`extension/lib/*.js` behind the dual-export shim, `node --test` with zero deps, no
Pydantic on the server, candidate knobs read with `config.get`, every non-trivial
decision gets an ADR.

Rejected alternatives worth recording, each with its reason:

- **Per-job derived `PipelineConfig` with `audience` inside `entries["translate"]`** —
  gets the cache key for free via `config_hash`, and cannot work.
  `stage_cache_key` is `(stage, candidate_id)` only (`patmir/pipeline/runner.py:43-54`);
  the id is unchanged, so the first job's warm translator is handed to every later job
  regardless of level. And `prewarm_stage_cache` (`ui/app.py:93-124`) builds translate
  at boot, before any submission exists, so there is nothing to bake in.
- **`[pipeline.options]` as the only source** — fails the requirement (per-submission)
  and needs a server restart per level.
- **`sw.js` as a per-tab session controller** — MV3 service workers are killed after
  ~30 s idle and `setInterval` does not survive; the minimum `chrome.alarms` period is
  30 s, useless for a 2 s job poll.
- **Duplicating the job protocol in `content.js`** — the repo already has this exact
  scar, documented at `popup.js:57-65` and pinned by three tests in
  `bootstrap.test.js:116-135` (the injected-file list drifted from the manifest and
  produced a self-inflicted "Receiving end does not exist").
- **`MutationObserver` on `#below`** — YouTube's churniest subtree; thousands of
  callbacks per navigation on the main thread of a video-decoding page. The existing
  500 ms URL poll plus `yt-navigate-finish` already cover navigation.
- **`createMediaElementSource`** on the page video — already rejected in ADR
  `2026-08-25-extension-volume-and-av-sync`; irreversible for the element's lifetime.

---

# Feature 1 — Audience age guardrail

## Task 2 — Canonical vocabulary

In `patmir/pipeline/context.py`, above `PipelineOptions`:

```python
AUDIENCE_LEVELS = ("child", "teenager", "adult")
DEFAULT_AUDIENCE = "child"     # product default
NEUTRAL_AUDIENCE = "adult"     # no clause, no cache-key suffix

def normalize_audience(value: str | None) -> str: ...   # None -> default; unknown -> ValueError
def resolve_audience(requested: str | None, options: PipelineOptions) -> str: ...
```

`resolve_audience` is the single place the precedence rule lives — per-submission wins,
else the server's `[pipeline.options].audience`. The resolved string drives **both** the
prompt and the cache-key suffix, so the two cannot disagree.

New `PipelineOptions.audience: str = DEFAULT_AUDIENCE`, with a comment marking it as
**the one exception** to "options never touch the cache key" (`ui/cache.py:10-12`):
every other field there is audio tuning, this one changes the Armenian *text*. It stays
out of `config_hash` and is folded into `cache_key` as an explicit suffix instead.

Loader: `config.py`'s `PipelineOptions(...)` construction gains
`audience=normalize_audience(opts_raw.get("audience", DEFAULT_AUDIENCE))`. A bad value
in `patmir.toml` is a hard load error, consistent with the unknown-profile stance.

## Task 3 — The prompt

`openrouter.py` gains `_AUDIENCE_CLAUSES` and `_audience_clause(level) -> str`, each
clause carrying its own leading `"\n\n"` so composition is a bare
`_SYSTEM + _audience_clause(level)` and `adult` returns `""` — byte-identical to today.
`_audience_clause` is tolerant of an unknown value (returns `""`); validation lives at
the edges.

`openrouter_batch.py` already imports `_SYSTEM` from that module
(`openrouter_batch.py:39-44`), so it imports `_audience_clause` alongside it and
composes at line 247 as `_SYSTEM + _audience_clause(audience) + "\n\n" + _BATCH_SYSTEM_SUFFIX`
— clause before the JSON-shape rules, so the schema instructions stay last. The
per-sentence and batch/watch-along candidates therefore cannot drift.

Both clauses must carry, in this order: what to soften (profanity → mild Armenian
interjections a child actually hears; slurs → translate the hostility, never the
epithet; sex → non-explicit; violence → keep the fact, drop the gore; substances →
named plainly, never glamorised), then a "do not moralize, do not add a warning, do not
change who says what, the plot facts, names or numbers", then a **HARD RULE** paragraph:
every input line gets one real speakable Armenian sentence; never delete, blank, bleep,
star out, mark `[...]`, or replace with a refusal. Write the gentlest true version and
move on. That paragraph exists because of who is watching, and the comment above the
clauses should say so.

`child` targets roughly 5-10, `teenager` roughly 11-15 (keep the edge, lose the
obscenity). Draft text is in the shape doc; it is prose and will want an ear-test pass.

Signatures gain a trailing `audience: str | None = None`:
`OpenRouterTranslator.translate`, `BatchTranslator.translate` / `.translate_batch` /
`._translate_chunk` / `._fallback_chunk`. The fallback path matters: a chunk that fails
validation degrades to per-sentence, and must degrade **with the same audience** rather
than silently reverting to the adult prompt.

## Task 4 — Adapter

`patmir/pipeline/stages/translate/adapter.py`: hoist `import inspect` to module scope
(there are two probe sites now), add `_accepts(fn, name)` and use it for `emotion` and
`audience` on `self.inner.translate`, and for `audience` on `self.inner.translate_batch`
— the batch branch currently probes nothing. The level comes from
`resolve_audience(None, ctx.options)`.

When the level is not `adult` and the candidate accepts no `audience`, emit **one**
warning per job through `log.warning` **and** `ctx.progress("translate", 0.0, …)` naming
the candidate. Not fatal (that would break the offline NLLB fallback outright — NLLB /
MADLAD / OPUS-MT are seq2seq and `translategemma_mlx.py:7-11` says it has no prompt
surface at all), and not silent. The repo has a scar here: `emotion_guidance` sat inert
for two months for exactly this reason, and the NOTE at `context.py:122-133` explicitly
says a restored version must warn when the active candidate cannot accept it.

Two additive fields on `translated.json` via `_write_translated`: `audience` and
`audience_honored` (the probe result). That file is copied into the cache as
`transcript.json` (`cache.py:243-245`), so a cached dub becomes self-describing.
Readers must treat both as absent-able.

**Also count empty results.** A provider safety refusal under `child` returns
`content: null`, which `openrouter.py:172-177` turns into `""` and `tts/adapter.py:248`
then skips — a silent hole, the exact failure this feature exists to prevent. The batch
path is already protected (`empty translation for idx=` → per-sentence fallback); the
per-sentence path is not. Count blank `target_text` and report via `ctx.progress` +
`log.warning`. This belongs in this task, not "later".

## Task 5 — Cache key

```python
def cache_key(source_id, cfg, mode=DEFAULT_MODE, audience=NEUTRAL_AUDIENCE) -> str:
    payload = f"{source_id}|{cfg_hash}"
    if mode and mode != DEFAULT_MODE:          payload += f"|{mode}"
    if audience and audience != NEUTRAL_AUDIENCE: payload += f"|audience={audience}"
```

`|audience=<level>` rather than a bare `|<level>`, so a future reader cannot confuse it
positionally with mode. **`CACHE_VERSION` stays 1** — adult keys are byte-identical by
construction and nothing on disk changed; a bump would needlessly invalidate every
watch-along `sections.json` too.

Callers, all of which must resolve once and reuse: `ui/jobs.py:177` and the
`lookup_cache_by_url` / `cache_key_for_url` helpers, `ui/stream.py:216` and `:226`,
`ui/cache.py:189` (`lookup_by_url` gains the parameter), and
`GET /cache/lookup` (`app.py:234-259`) — which gains an `audience` query param. A lookup
that forgets it reports "not dubbed" for an existing dub, so it gets its own test.

## Task 6 — Server wire-up

- `JobSpec` (`ui/jobs.py:42-57`): `audience: str | None = None`, normalized in
  `__post_init__` beside the `mode` check; the `ValueError` reaches the existing 400
  path at `app.py:271-272`.
- `JobRecord` / `StreamRecord`: a plain `audience: str` field (so `asdict` persistence
  still works), echoed in `to_public()`.
- **One derived config per job**, and only `options`:

  ```python
  def _cfg_for(self, audience: str) -> PipelineConfig:
      if audience == self._cfg.options.audience:
          return self._cfg
      return replace(self._cfg, options=replace(self._cfg.options, audience=audience))
  ```

  `entries` is shared by reference, so `stage_cache_key` is identical to the default
  config's and the startup prewarm still hits. Add a one-line comment at
  `runner.py:43-54` saying the key is derived from `entries` only, and why.
- Watch-along needs **no signature changes** anywhere in `patmir/pipeline/streaming/`:
  `run_prelude`, `dub_prelude_window` and `process_section` all already take `cfg`, so
  `_run_blocking` deriving once is enough. A running stream keeps the level it started
  with — the level is in the cache key, so a new level is a new stream.
- `POST /jobs` pulls `payload.get("audience")`. Untyped dict, no Pydantic, matching the
  rest of the file. `GET /cache/lookup` echoes the resolved level in its response so a
  mismatch is debuggable.

**One wart to fix in the same change:** `_download_url` writes to
`downloads_root/<cache_key>` (`jobs.py:466`), so with the audience in the key, flipping
level on the same video **re-downloads the source**. Since re-picking a level on the
same episode is the expected workflow here, give the records a second, audience-free
`download_key` computed at `NEUTRAL_AUDIENCE` and use it for the download dir and for
`force` cleanup. If this is dropped to keep the diff tight, say so explicitly in the ADR
rather than letting someone discover it as a four-minute re-download.

## Task 7 — Extension + CLI surface

- `options.html` / `options.js`: a `patmirAudience` `<select>` (child / teenager /
  adult, default child) with a hint saying nothing is ever cut or bleeped and that each
  level caches separately. While editing, fix the latent quirk at `options.js:36` —
  `load()` is bound to `DOMContentLoaded` from a script tag at the end of `<body>`, so
  the event may already have fired; call it directly.
- `sw.js`: a `getAudience()` next to `getServer()`, injected **by the service worker
  itself** into both `jobs:submit`'s body and `cache:lookup`'s query string. Doing it
  here rather than at each call site is what keeps lookup and submit in agreement, and
  it makes `content.js`'s auto-play lookup (`content.js:169`) correct with no change.
  Update the message-contract comment at the top of the file.
- Precedence is therefore **extension > `patmir.toml`**: the extension always sends a
  value, so `[pipeline.options].audience` is the default for the CLI and any client that
  omits the field.
- CLI: `--audience {child,teenager,adult}` on `patmir dub` (set on the freshly loaded
  `cfg.options` in `cmd_dub`, and printed in the header block at `cli.py:214-220` so an
  ear-test log records it) and on `patmir submit` (added to the POST body).
  `patmir bench` is left alone — it measures throughput, and a ~250-token system-prompt
  delta on a cached prefix is not what it measures. Note that in the ADR.
- `patmir.toml` `[pipeline.options]`: `audience = "child"`, with a comment marking it as
  the one option that participates in the cache key and how.

## Task 8 — Tests (feature 1)

- `test_ui_cache.py`, new class beside `CacheKeyInvariants`: adult key equals the
  legacy key (assert against `sha256(f"{src}|{config_hash(cfg)}")[:16]`, pinning the
  exact payload); three levels pairwise distinct; audience composes with mode;
  `cfg.options.audience` alone does **not** change the key (documents that the resolved
  argument carries it); `lookup_by_url` respects the level.
- `test_translate_batch.py`: adult system prompt unchanged; the child clause sits
  between `_SYSTEM` and `_BATCH_SYSTEM_SUFFIX`; **the per-sentence fallback carries the
  same audience** (the bug this prevents is a fallen-back chunk silently reverting to
  the adult prompt).
- New `test_audience.py`: `_audience_clause("adult") == ""` and
  `_SYSTEM + _audience_clause("adult") == _SYSTEM` byte-for-byte; child ≠ teenager, both
  non-empty, both containing the never-drop language (assert on stable substrings, so
  the test fails if someone edits the softening prose into redaction language);
  per-sentence prompt composition with and without an emotion hint; adapter probing in
  all four shapes (accepts / does not accept, `translate` and `translate_batch`) with
  the warning firing exactly once at non-adult and never at adult; `JobSpec` validation;
  `resolve_audience` precedence.
- `test_stage_cache.py`: two configs differing only in `options.audience` produce the
  **same** `stage_cache_key`. This is the test that would have caught the rejected
  entry-injection design shipping by accident.
- New `extension/tests/audience.test.js` (vm sandbox over `sw.js`, same idiom as
  `bootstrap.test.js`): the stored level reaches both the lookup URL and the submit
  body; nothing stored → `child`; mode and audience coexist in the query string.

---

# Feature 2 — In-page toggle button

## Task 9 — `extension/lib/session.js` (new)

One copy of the job protocol, two surfaces. Follows `lib/sync.js` exactly: **no DOM, no
timers, no direct `chrome.*`** — everything through an injected `transport`, so it is
driven by hand under `node --test`.

`createSession({ transport, mode, autoPlayOnDone })` where `transport` supplies
`sw(msg)`, `player`, `getUrl()`, `storage`, `now()`. The `player` interface is exactly
four methods mirroring the existing tab messages: `play(cacheKey)`,
`playWatchAlong(cacheKey)`, `stop()`, `status()`.

Methods: `refresh()`, `primary()`, `start()`, `stop()`, `regenerate()`, `setMode()`,
`tick()` (host-driven — the host owns the interval), `getState()`, `onChange(cb)`.

Three race guards, all unit-testable: a `seq` counter (every command bumps it; a reply
whose captured seq differs is dropped — kills double-click double-submit and
mode-flip-mid-submit); a URL guard (same defence `content.js:178` already applies in
`autoPlayIfApplicable`); and terminal-status handling matching `pollOnce`
(`popup.js:300-313`) **plus** `cancelled` / `cancelling`, which the server produces
(`jobs.py:274,282`) and the popup currently falls through on — a small pre-existing bug
this fixes for both surfaces.

Pure derivation, split in two so one state machine serves two languages:

```
deriveView(state) -> { kind, action, disabled, busy, tone, progress, errorText }
formatLabel(view, strings) / formatSubtitle(view, strings)
STRINGS_HY (page)   STRINGS_EN (popup, unchanged UX)
pageKind(href)      MOUNTS / mountPlan(present)
```

## Task 10 — Button states

The label is only ever one of four Armenian words; **progress never enters the label**,
because a 52 px full-width control whose text changes every 2 s cannot be aimed at.

| Situation | Label | Subtitle (Armenian only) |
| --- | --- | --- |
| checking | `Թարգմանել` (disabled) | `Ստուգվում է…` |
| no token | `Կարգավորել` | `Անհրաժեշտ է token` |
| server unreachable | `Կրկին փորձել` | `Սերվերը հասանելի չէ` |
| video not ready yet | `Թարգմանել` (disabled) | `Վիդեոն դեռ չի բացվել…` |
| not cached / partial / cancelled | **`Թարգմանել`** | — / `N հատված պահված է` |
| submitting, queued, waiting for GPU | `Չթարգմանել` | `Ուղարկվում է…` / `Հերթում է…` / `Սպասում է GPU-ին…` |
| running, downloading, prelude | `Չթարգմանել` | `Թարգմանում… NN %` (one number, no stage names) |
| streaming | `Չթարգմանել` | `Հեռարձակվում է — N հատված` |
| failed | `Կրկին փորձել` | `Չհաջողվեց` + trimmed error |
| cached, idle | **`Թարգմանել`** | `Պատրաստ է` |
| starting / buffering | `Չթարգմանել` | `Բեռնվում է…` |
| **ad showing** | unchanged (`Չթարգմանել` if on) | `Գովազդ…` |

Two design rules to write into the file header:

- **`Թարգմանել` unifies "submit a job" and "play the cache."** From the viewer's side
  both are "give me the Armenian"; whether the server has to work for it is a detail for
  the subtitle. A deliberate departure from the popup's *Translate this* / *Play dub*
  split, which exists because the popup is also a developer tool.
- **`Չթարգմանել` stays live during a job**, so the viewer always has an exit. That is
  what makes the `jobs:cancel` addition below load-bearing rather than cosmetic.

The "video not ready" row matters: `findVideo()` filters on `duration > 1`
(`content.js:351-359`), so it returns `null` for the first few hundred ms of a watch
page. Latch that row for the first ~8 s after mount (the budget `waitForVideo` already
uses at `content.js:175`), then fall through. And never call `player.stop()` because
`hasVideo` went false — `content.js`'s own navigation and source-change handlers own
that decision, and a second authority would fight them.

**What pressing `Չթարգմանել` does, precisely.** It calls `player.stop()`, which is
`stopPlayback()` — that stops the dub source, detaches the sync tick, and
`endMuteControl()` restores `video.muted` to the viewer's intent (`content.js:272`), so
the original audio comes back. It **never touches `video.pause()` or `currentTime`**,
so the picture keeps rolling; verified against `stopDub` (`content.js:396-407`) and
`stopPlayback` (`:524-527`). It also sets the per-video opt-out
(`setVideoDisabled(url, true)`), which is **required, not incidental**: without it the
auto-play path (`content.js:158-185`, re-triggered by the `patmirDisabledVideos` storage
listener at `:205-210`) restarts the dub within a tick and the button looks broken.
Semantically that is right anyway — "translation off" for this video should stick.

## Task 11 — `extension/overlay.js` (new) — the DOM

A **separate content script**, listed after `content.js`, consuming a
`globalThis.PatmirPlayer` facade that `content.js` publishes. This keeps all DOM
construction out of `content.js`, so `tests/bootstrap.test.js`'s vm sandbox needs no new
stubs and the bail-before-registration guard (`content.js:26-34`) is untouched.

Manifest becomes `["lib/sync.js", "lib/session.js", "content.js", "overlay.js"]`.
Order is load-bearing three times and each ordering gets an assertion.
`popup.js:contentScriptFiles()` reads the manifest, so its re-injection fallback picks
up the new files for free — but its literal default array must be updated, and the
existing test at `bootstrap.test.js:126-135` will say so if it is not.

**Mount points, watch pages** — first hit wins, resolved through a pure
`mountPlan(present)` over booleans so the policy is testable without jsdom:

1. `ytd-watch-flexy #primary-inner #below` → prepend (immediately under the player)
2. `#primary-inner > #player` → after
3. `ytd-watch-metadata` → before
4. `#primary-inner` → prepend
5. walk up from `PatmirPlayer.findVideo()` via `closest("#player, ytd-player")` → after
   — selector-free last resort, in the spirit of `findVideo()` itself

Theater mode needs no code (`#below` still exists, just lower). Fullscreen hides
`#below` along with the rest of the page chrome, which is correct — do **not** try an
in-player overlay, it would sit inside `#movie_player` and fight the controls' pointer
capture and auto-hide.

**Shorts / embeds get the floating pill.** There is no stable region under the player
there — Shorts is a recycler that re-parents nodes as you scroll — so instead of
mounting into the layout, mount a compact fixed-position pill (same shadow root, same
`deriveView`, shorter label) anchored to the player's bounding box, `position: fixed`,
bottom-right of the video, with a modest `z-index` and `pointer-events` confined to the
pill. It must re-anchor on resize and on the Shorts pager swapping videos, and it must
hide itself in fullscreen. This is the riskier of the two placements and should be built
second, after the watch-page button is working.

**Isolation:** a `#patmir-toggle-root` div with `attachShadow({mode:"open"})`, styles via
a constructable stylesheet (`new CSSStyleSheet()` + `replaceSync` + `adoptedStyleSheets`,
feature-detected with an appended `<style>` fallback). A shadow boundary *removes* the
possibility of style bleed in both directions, where scoped class names only lower the
odds — and with it the temptation to fight with `!important`. YouTube's theme still
reaches in through custom properties (`var(--yt-spec-text-primary, #0f0f0f)`), so dark
mode is free. Set an Armenian-capable font stack explicitly rather than relying on
inheritance you just severed. `lang="hy"`, `type="button"`, `aria-live="polite"` on the
subtitle, `aria-busy` while working.

**Keyboard:** `stopPropagation()` on `keydown` for space, `k`, `j`, `l`, `f`, `m` and
arrows, or a focused button will both activate itself and trigger YouTube's
document-level shortcut. Do **not** `preventDefault()` on space — that would break the
button's own activation.

**Re-mount** on: `yt-navigate-finish` (+250 ms, the delay `content.js:200-202` already
found sufficient), a URL change delivered through a new `PatmirPlayer.onUrlChange(cb)`
fanned out from the **existing** 500 ms interval (`content.js:942-947`), and one owned
500 ms tick doing `document.contains(root)` plus a diffed render. Diffing before writing
`textContent` is not optional: unconditional DOM writes at 2 Hz inside YouTube's layout
on a video-decoding page is exactly what shows up as dropped frames.

Two guards it needs from day one: `window.__patmirOverlayInjected`, and a defensive
`document.getElementById("patmir-toggle-root")?.remove()` before mounting, because
`executeScript` re-injection can leave a stale root whose interval was cleared. And a
central catch for `"Extension context invalidated"` that self-destructs the root —
otherwise a stale button sits on the page reading `Սերվերը հասանելի չէ` forever.

## Task 12 — `content.js` and `popup.js` changes

`content.js`, three edits, none touching the audio graph, volume arbitration, or the
sync tick:

1. **Publish the facade** at the end of the IIFE (after all `let` declarations, so
   nothing is reachable during the TDZ window the bootstrap guard exists to protect):
   `{ play, playWatchAlong, stop, status, findVideo, canonicalUrl, onUrlChange, adState }`.
   `play`/`playWatchAlong`/`stop` are the **existing message-handler bodies** extracted
   into functions, with the handlers rewritten to call them — one implementation, so the
   `setVideoDisabled` bookkeeping is shared.
2. `onUrlChange(cb)` — a callback list the existing 500 ms interval fans out to, so
   `overlay.js` needs no URL poll of its own. Two intervals on an idle page, same as today.
3. **Narrow `return true`** in the message listener (`content.js:116`) to the five known
   types. Today it claims the response channel for every message including the new
   `patmir:session-*` ones and never answers — it happens to work because the overlay's
   listener replies first, but it produces dangling ports and races on a slow machine.

`popup.js` keeps every control and every DOM id (`popup.html` is untouched) and keeps
`sw()`, `tabMessage()`, `contentScriptFiles()`, the mode toggle and the two-click
regenerate arming. Its orchestration bodies are replaced by a session:

- **remote session** when `tabMessage` reaches a content script — `getState()` and each
  command are one round trip to the overlay, which is the single owner. Sync is then
  true by construction: on a YouTube tab there is only ever one state.
- **local session** otherwise — same module, `player: NULL_PLAYER`,
  `autoPlayOnDone: false`, reproducing today's behaviour on non-YouTube http pages,
  which is why the popup cannot simply delegate unconditionally.

New messages: `patmir:session-state` (`{popupOpen}` → `{ok, state}`) and
`patmir:session-command` (`{action, mode?, force?}` → post-action snapshot). The existing
`patmir:play` / `patmir:play-watchalong` / `patmir:stop` / `patmir:status` stay exactly
as they are — the popup's local path and the injection fallback both need them.

Cross-surface signals ride `chrome.storage.local`, which `content.js` already watches:
`patmirMode` (popup flips it → button re-looks-up), `patmirDisabledVideos`,
`patmirAudience`, and a new bounded `patmirActiveJobs` map (`{cache_key: job_id}`) so a
popup opened mid-button-job, or a tab navigated away and back, re-adopts the running job
instead of offering a fresh submit. Self-healing, because the server already dedupes by
cache key (`jobs.py:189-193`). Do **not** broadcast the whole session state through
storage — it would fire `storage.onChanged` at 2 Hz and re-enter the auto-play listener
every tick.

One `sw.js` addition: **`jobs:cancel`** → `POST /jobs/{id}/cancel`, which exists
(`app.py:293`) and is idempotent but has no extension message. Without it, pressing
`Չթարգմանել` during a running quality job stops the UI while the GPU keeps grinding —
the same waste `stream:stop` already prevents for watch-along.

**Watch-along costs money** (ElevenLabs TTS). A large always-visible button makes
accidental starts likelier than a popup CTA does. At minimum the not-cached subtitle
should name the active mode; a one-time confirm in watch-along mode is worth
considering.

## Task 13 — Tests (feature 2)

- `extension/tests/session.test.js` — `deriveView` one case per row above, plus two
  invariant loops asserting that no state introduces a third primary label; progress
  formatting including the `live_pct`-ignored-unless-`live_stage === running` subtlety
  (`popup.js:334-336`, currently untested); `pageKind` and `mountPlan` (all 32 boolean
  combinations, first-hit ordering, and the null case); and `createSession` against a
  recording fake transport with a controllable clock — one submit per double click, poll
  interval respected, `done` → play only when `autoPlayOnDone`, watch-along `done` →
  no polling at all, `stop()` → `jobs:cancel` for quality and not for watch-along,
  stale-URL and stale-seq replies dropped, `patmirActiveJobs` adoption and pruning.
- `extension/tests/overlay.test.js` — vm plus a ~60-line hand-rolled mini-DOM (`FakeEl`
  with `attachShadow`, a `document.querySelector` backed by a test-provided Map, and a
  `contains` that walks `parentElement`). Same stubbing idiom as `bootstrap.test.js`, no
  jsdom. Covers: bails loudly without `PatmirSession` and without `PatmirPlayer`; mounts
  into `#below` by prepending; falls down the list as selectors disappear; no mount and
  no throw when everything is missing; the floating pill on Shorts; re-mount when
  detached and on URL change; click → `primary()` once, suppressed while disabled;
  answers both new messages; render is idempotent (assert via a setter-counting FakeEl).
- `extension/tests/bootstrap.test.js` — the four-file manifest ordering; the
  `contentScriptFiles()` literal default; **`content.js` still boots clean in a sandbox
  with no `createElement` / `MutationObserver` / `CSSStyleSheet`**, which is the
  regression guard that keeps button code out of `content.js`; the facade is published
  with exactly the expected keys; and `content.js` no longer returns `true` for an
  unknown message type.

---

# Feature 3 — YouTube ads

## Task 14 — Detect the ad, suspend the dub, resume after

Detection, cheapest signal first, in a pure `classifyAd(present)` in `lib/sync.js` (it
already owns the playback predicates and has a test file):

1. `#movie_player` carries the class **`ad-showing`** (and `ad-interrupting`) — the
   primary signal.
2. `.ytp-ad-player-overlay-layout` / `.ytp-ad-module` present and non-empty.
3. `.ytp-ad-preview-container` (the countdown) for the skippable case.

Polled from the **existing** 250 ms sync tick while a dub is active, plus the 500 ms
idle interval when it is not. No new timer.

Three behaviour changes, each pinned to a concrete failure:

- **`shouldScheduleAudio` gains an `adShowing` input** and returns false for it. It is
  already the single gate on "is the picture really moving", already pure, and already
  tested — this is one line and one row in the truth table.
- **On ad start:** `stopDubSource()` / `stopAllNodes()` and `resetSyncWindow()` —
  silence, **not** `stopDub()`. Keep the listeners, the decoded buffer, `playing = true`
  and the mute control, so the original ad audio is still suppressed only to the extent
  the viewer's own mute setting says. On ad end, the existing "picture confirmed moving"
  path re-anchors from `video.currentTime`, which is back on the content timeline, and
  audio resumes on its own.
- **`onVideoSourceChanged` must not treat an ad's src swap as a navigation.**
  `content.js:482-491` currently answers any `currentSrc !== videoSrcAtStart` with
  `stopDub()`, which is why an unskippable mid-roll kills the dub permanently today.
  Gate it on `!adShowing`, and on ad end compare against the remembered content src
  before deciding anything. Same for `onVideoSourceChangedWA` (`:922-926`).
- **Watch-along:** suppress `reportPlayhead` while an ad shows. Posting the ad's
  `currentTime` to `/stream/<key>/playhead` makes the scheduler think the viewer
  seeked to near zero and start dubbing sections that are already covered.

## Task 15 — Auto-skip

Two mechanisms, one setting (`patmirSkipAds`, default on):

1. **Click the skip control when it appears** — `.ytp-skip-ad-button`,
   `.ytp-ad-skip-button-modern`, `.ytp-ad-skip-button` (try in that order; YouTube has
   renamed it repeatedly), only when it is present and not disabled. This is exactly
   what the viewer would do by hand a second later.
2. **Unskippable ads: seek to the end** — `video.currentTime = video.duration` while
   `ad-showing` is set. This is the ad-skipper-userscript move: more fragile (YouTube
   sometimes restarts or stalls the ad), and more clearly circumvention than clicking a
   visible button. Keep it behind the same setting, cap it to a few attempts per break
   so a stall cannot become a loop, and never seek unless `ad-showing` is set — a bug
   here would seek the *content* video, which is the one genuinely destructive failure
   in this whole plan.

Worth knowing rather than discovering: this will break whenever YouTube changes its
player markup, and it is the kind of behaviour a Web Store review would look at if this
extension were ever distributed (it is loaded unpacked, for one person, so that is not
today's problem). YouTube Premium is the robust version of this feature.

Decision logic goes in a pure `adSkipPlan({adShowing, skipButtonReady, adCurrentTime, adDuration, attempts, enabled})`
→ `{action: "click" | "seek" | "none"}`, unit-tested; `overlay.js` performs the action.

### Blocking ads outright — deliberately out of scope

Not by network rules. YouTube's ad segments come from the same `googlevideo.com` CDN,
often the same URL shapes, as the content, so there is no clean pattern for
`declarativeNetRequest` to block; and MV3 removed `webRequestBlocking`, so a dynamic
per-response decision is not available either.

There *is* a way, and it is how real blockers do it: a `world: "MAIN"` content script at
`document_start` that patches `JSON.parse` / `Response.prototype.json` before YouTube's
own scripts run, and deletes `adPlacements` / `playerAds` / `adSlots` from the player
response. We are not doing it, for three reasons that are about this project rather than
about principle: it is an active arms race that breaks on YouTube's schedule, not ours;
YouTube responds to it with playback stalls and interstitials that would be blamed on
the dub; and rewriting the player response is precisely the kind of thing that could
perturb the timeline the dub's drift correction is anchored to. That is a lot of
fragility to take on inside a tool whose job is audio.

The right division of labour is that an ad blocker blocks and Patmir dubs. Which gives
a property worth stating: **with uBlock Origin (or Premium) installed, everything in
Tasks 14-15 degrades to a no-op** — `ad-showing` never appears, `classifyAd` returns
false, nothing suspends and nothing skips. So the ad handling is insurance for the
unblocked case, not a competing mechanism, and installing a blocker later costs us
nothing.

## Task 16 — Tests (feature 3)

`classifyAd` over each signal in isolation and in combination; `shouldScheduleAudio`
returns false with `adShowing: true` regardless of the other inputs; `adSkipPlan`
returns `none` when disabled, `click` when the button is ready, `seek` only for an
unskippable ad with attempts remaining, and `none` once the cap is hit. In the vm
sandbox: an `emptied` + src change **with** `ad-showing` set does not tear the dub down,
and the same event **without** it still does — the two halves of the fix, pinned
against each other.

---

## Task 17 — Documentation (CLAUDE.md maintenance contract)

- New ADR `docs/architecture/decisions/2026-08-28-audience-age-guardrail.md` — the three
  levels; prompt-only softening and the hearing-aid rationale that makes a silent gap
  the worst outcome; why the level rides on `options` and a per-call kwarg rather than
  `entries` (stage cache + prewarm, with the concrete failure mode); why the cache key
  gains a suffix instead of a `CACHE_VERSION` bump; why `DEFAULT_AUDIENCE` and
  `NEUTRAL_AUDIENCE` differ and the one-time re-dub that implies; precedence
  extension > toml; warn-don't-refuse on candidates with no prompt surface, citing the
  inert-`emotion_guidance` scar; open items (provider refusals still land as `""`, the
  download-dir question, no mid-stream level change, bench left at adult).
  *Revisit:* an ear-test finds child over-softened or teenager still too coarse; or a
  local translate candidate becomes active, at which point the guardrail silently stops
  applying and the warning is the tripwire.
- New ADR `docs/architecture/decisions/2026-08-28-inpage-toggle-and-ads.md` — the
  ownership rule (content script owns the session on a YouTube tab; popup is a view);
  why `overlay.js` is separate from `content.js`; shadow DOM over scoped classes; the
  poll-not-observe choice; and the ad findings, especially that a mid-roll previously
  killed the dub permanently.
- One-line entries in `docs/architecture/decisions.md` for both.
- `current-models.md` translate row: the composed audience clause, `adult` byte-identical,
  softening is wording-only. The knob is prompt-side, so it is profile-independent.
- `stages/translate/current.md`: `_audience_clause`, the three levels' concrete rules,
  and that the batch candidate inherits it through the shared import.
- `data-contracts.md`: `PipelineOptions.audience` (the one option in the cache key, and
  how); the additive `translated.json` keys; the `audience` field on `JobSpec`,
  `POST /jobs`, `GET /cache/lookup` and both `to_public()`s.
- `pipeline.md`, beside the "`JobContext.cancel` is NOT in `extra`" section: one
  paragraph on the level riding `ctx.options` (config channel) rather than `extra` (data
  hand-off) or `entries` (model selection), forwarded per call by signature probe like
  `emotion`.
- `extension/README.md`: the audience select, the in-page button and where it appears,
  the ad handling and the skip setting, `lib/session.js` in the Tests section, and the
  now-four-file manifest ordering requirement.

`quality-issues.md` gets **one** new row: the mid-roll ad permanently killing the dub is
an observed output-affecting defect, not a feature. The rest is additive.

---

## Implementation order

Each step leaves the tree green.

**Feature 1** — context/config/toml → prompt clause + batch threading → adapter probing,
warning and empty-line counter → cache key → server (`jobs.py`, `stream.py`, `app.py`)
→ extension options + `sw.js` → CLI flags → docs. Nothing calls with a non-adult level
until the server step, so every earlier step is provably behaviour-preserving.

**Feature 2** — `lib/session.js` pure half (the whole state machine, testable before any
wiring) → `createSession` → `sw.js jobs:cancel` → `content.js` facade (verify popup
play/stop by hand *before* touching the manifest; this step must be provably
behaviour-preserving on its own) → manifest → `overlay.js` watch-page button →
`popup.js` remote/local split → `patmirActiveJobs` adoption → floating pill for
Shorts/embeds (last; it is the riskiest placement).

**Feature 3** — `classifyAd` + `shouldScheduleAudio` + `adSkipPlan` pure, with tests →
suspend/resume wiring in `content.js` → playhead suppression → auto-skip in `overlay.js`.
Doing detection before skipping means the dub is already correct across ads even if
skipping is later turned off.

---

## Verification

**Automated**

```bash
node --test extension/tests/          # existing 42 + session, overlay, audience, ads
python -m pytest patmir/tests -q
```

**Feature 1, end to end**

```bash
patmir dub samples/sample_1.mkv --audience adult
patmir dub samples/sample_1.mkv --audience child
```

Then diff the two `runs/*/translated.json`: the `idx` sets and sentence counts must be
**identical**, no `target_text` may be empty, and the wording should differ only in
register. Comparing counts matters as much as listening — the failure mode this feature
can introduce is a dropped line, not a bad word. Then via `patmir ui`: set the options
page to `child`, submit, and confirm a **new** directory under `~/.patmir/cache/`; flip
to `adult` and confirm the original key is reused.

**Feature 2, in the browser** — reload the unpacked extension (a manifest change needs a
full reload). Then: watch page not cached → press `Թարգմանել` → Armenian progress →
audio starts, label becomes `Չթարգմանել`; press again → dub stops, **original audio
returns, video never pauses**; open the popup mid-job and confirm the same progress;
press Stop in the popup and confirm the in-page button flips within one tick; start from
the popup and confirm the button shows progress; SPA-navigate away and back (exactly one
`#patmir-toggle-root` on the page); theater mode; fullscreen enter/exit; a Short (pill
appears); an embed; server stopped mid-job; token cleared. Smoke-test the volume slider
and a seek — those paths are untouched, so this is a check, not a re-derivation.

**Feature 3** — a video with a pre-roll (dub waits, then starts in sync), one with an
unskippable mid-roll (**dub suspends and resumes; before this change it stopped for
good**), and one with a skippable ad (skip fires once the button enables). With
auto-skip off, confirm the suspend/resume still works on its own. Confirm no seek ever
lands on the content video.
