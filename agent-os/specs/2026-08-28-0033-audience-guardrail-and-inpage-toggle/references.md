# References for the audience guardrail + in-page toggle

## Similar implementations studied

### Per-call capability threading — the `emotion` kwarg

- **Location:** `patmir/pipeline/stages/translate/adapter.py:68-89`
- **Relevance:** the exact shape `audience` needs. The adapter probes
  `inspect.signature(self.inner.translate)` for `emotion` and passes it only when the
  candidate accepts it, so the candidate stays stateless and one warm instance serves
  every call.
- **Key patterns:** signature probe, kwargs built conditionally, no candidate-side
  registration. Extended here to cover `translate_batch`, which probed nothing.

### The `mode` cache-key suffix

- **Location:** `patmir/ui/cache.py:140-152`
- **Relevance:** the precedent for folding a per-submission value into the key without
  busting existing entries — append only when non-default.
- **Key patterns:** a named constant for the no-suffix value plus a comment explaining
  which keys are being preserved and why. `audience` copies the mechanism but splits the
  constant in two, because its default and its no-suffix value differ.

### Pure decision logic behind a dual-export shim

- **Location:** `extension/lib/sync.js` (+ `extension/tests/sync.test.js`)
- **Relevance:** the template for `lib/session.js` and for the new ad predicates.
- **Key patterns:** no DOM / no timers / no `chrome.*`; every decision a pure function
  over a plain object; the host owns the interval and calls in.

### Popup orchestration being replaced

- **Location:** `extension/popup.js:240-360` (`submitJob`, `startPolling`, `pollOnce`,
  `renderJob`, `renderStreamJob`)
- **Relevance:** ~120 lines of job-protocol knowledge that must move into
  `lib/session.js` rather than be copied — which statuses are terminal, that `waiting`
  means the accelerator lock, the `download_pct` special case, that `live_pct` is only
  meaningful when `live_stage === running`.

### The vm sandbox for DOM-touching code

- **Location:** `extension/tests/bootstrap.test.js`
- **Relevance:** how this repo tests a content script with no jsdom — evaluate the file
  text in `node:vm` against hand-rolled `chrome` / `document` / `AudioContext` stubs and
  capture timers. `overlay.test.js` extends it with a ~60-line mini-DOM.

### `JobSpec` validation

- **Location:** `patmir/ui/jobs.py:42-57`
- **Relevance:** where a new per-submission field is validated. `__post_init__` raises
  `ValueError`, which `app.py:271-272` turns into a 400.

## Prior decisions that constrain this work

- `docs/architecture/decisions/2026-08-25-extension-volume-and-av-sync.md` — the volume
  arbitration and drift machinery this work must not disturb, and the rejection of
  `createMediaElementSource`.
- `docs/architecture/decisions/2026-08-23-warm-model-residency-and-prewarm.md` — the
  process-wide `stage_cache` and startup prewarm that rule out baking the audience level
  into a candidate's constructor.
