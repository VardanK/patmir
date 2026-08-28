# Standards for the audience guardrail + in-page toggle

`agent-os/standards/index.yml` contains only a header comment — there are no indexed
standards in this repo yet. What follows are the de-facto conventions this codebase
actually enforces, read off the existing code and tests, which this work follows.

## Extension

- **Pure logic lives in `extension/lib/*.js`** behind the dual-export shim: assigns to
  `globalThis.<Name>` when loaded as a content script, `module.exports` under
  `node --test`. No DOM, no Web Audio, no timers, no `chrome.*`. `lib/sync.js:1-14`
  states the rule; `lib/session.js` extends it by taking a transport object.
- **Tests are `node --test extension/tests/` with zero dependencies.** No jsdom, no
  package.json. DOM-dependent code is exercised in a `node:vm` sandbox with hand-rolled
  stubs (`tests/bootstrap.test.js`).
- **One source of truth for the content-script file list**: the manifest.
  `popup.js:contentScriptFiles()` reads it rather than hardcoding, and a test asserts
  the literal fallback still matches. Load order is load-bearing and asserted.
- **A missing dependency bails before registering anything**, with a message naming the
  cause and the fix (`content.js:8-34`). Half-initialized modules produce confusing TDZ
  errors instead of the real reason.

## Server / pipeline

- **No Pydantic.** Request bodies are untyped `dict` with manual field pulls; validation
  lives in the dataclass `__post_init__` and surfaces as HTTP 400.
- **Candidate knobs are read with `config.get(...)`** from the raw TOML entry dict, so
  an unknown key is inert until a `build()` line reads it.
- **Optional capabilities are discovered by signature probing**, not by declaration —
  `adapter.py:68-89` probes for `emotion`. A capability the active candidate cannot
  honour must **warn**, never silently vanish: `context.py:122-133` records two months
  of `emotion_guidance` being inert for exactly that reason.
- **`JobContext.extra` is the cross-stage data hand-off channel only.** Control signals
  and configuration get their own field or ride `ctx.options` (`pipeline.md`).
- **The cache key hashes candidate entries, never `[pipeline.options]`** — audio tuning
  must not invalidate an ear-test loop. `audience` is the single, documented exception,
  folded in as an explicit suffix rather than through `config_hash`.
- **Every non-trivial decision gets an ADR** in `docs/architecture/decisions/` plus a
  one-line index entry, and the `CLAUDE.md` maintenance contract names which docs to
  update for which kind of change.
