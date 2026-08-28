# Audience age guardrail: steer the register, never drop a line

**Date:** 2026-08-28
**Status:** accepted
**Spec:** `agent-os/specs/2026-08-28-0033-audience-guardrail-and-inpage-toggle/`

## Context

Patmir exists to dub foreign video into Eastern Armenian for the author's
hearing-aid-wearing son. Until now nothing in the pipeline knew that. The
translator's system prompt (`_SYSTEM`, `openrouter.py`) asks for "casual spoken
Eastern Armenian… playful, cartoon register" and nothing else, so profanity,
slurs, sexual innuendo and graphic violence in the source pass straight through
into the Armenian the child hears.

The constraint that shapes everything below: **the dub is the only audio track
this viewer has.** He is not reading subtitles over an audible original — if a
line is missing, the story has a hole in it. A silent gap is therefore strictly
worse than a gentler word, which rules out every "just skip it" design.

## Decision

Three levels — `child`, `teenager`, `adult` — selected in the Chrome extension's
options page, sent on every submission, and threaded to the translator as a
per-call keyword that appends a clause to the system prompt.

**Prompt-only softening.** The clauses instruct the model to choose safer
wording and explicitly forbid deleting, blanking, bleeping, starring out,
marking `[...]`, or replacing a line with a refusal or a warning. They also
forbid changing who says what, the plot facts, the names or the numbers: the
words soften, the story does not. `adult` contributes the empty string.

### The value rides `options`, not `entries`

The tempting design — inject `audience` into the translate candidate's TOML
entry dict — gets the cache key for free through `config_hash`, and cannot work:

* `stage_cache_key` is `(stage, candidate_id)` only (`pipeline/runner.py`). The
  candidate id does not move when you add a key to its entry, so the first job's
  warm translator is handed to every later job **regardless of level**. A child
  submission would get whatever register the previous job built.
* `prewarm_stage_cache` builds the translate stage at server boot
  (`ui/app.py`), before any submission exists. There is nothing to bake in.
* Fixing the first point by putting the level *into* the stage-cache key
  fragments the warm cache three ways for one stage, and breaks the invariant
  that the key is derived from the candidate id.

So the server derives a per-job `PipelineConfig` replacing **only `options`**
(`JobManager._cfg_for`, `StreamManager._cfg_for`); `entries` is shared by
reference. The adapter reads `ctx.options.audience` and forwards it as a
per-call kwarg, discovered by signature probe — exactly the precedent `emotion`
already set. One warm candidate serves all three levels.
`test_stage_cache.AudienceDoesNotFragmentTheCache` pins this.

The level lives on `PipelineOptions` rather than a new `JobContext` field
because `options` *is* the configuration channel (`ctx.extra` is the cross-stage
data hand-off; `cancel` has its own field because it is an unserializable
control signal, not config). It also means watch-along needed **no signature
changes at all** in `pipeline/streaming/` — `run_prelude`, `dub_prelude_window`
and `process_section` already take `cfg`.

### Two constants, deliberately different

```python
DEFAULT_AUDIENCE = "teenager"  # what a submission gets when it says nothing
NEUTRAL_AUDIENCE = "adult"     # byte-identical prompt -> no cache-key suffix
```

`mode` keeps these as one value; here they must differ. `adult` is the level
that appends nothing to the key, because every dub cached before this feature
was produced by exactly the adult prompt and its key must keep pointing at it.
The default is a *softening* level so that not choosing is not the same as
opting out.

Shaping set the default to `child`; the author moved it to `teenager` the same
day. `child` is one click away and is the right choice for cartoons, but
`teenager` is the better *default* across a mixed library because it is wrong in
the smaller direction — too mild for an adult drama costs some bite, where too
coarse for a young viewer costs the thing this feature exists to prevent.

**The consequence: shipping this re-dubs every video once**, at the new default.
The alternative — making the default level the no-suffix one — would serve
adult-register audio under a softened label, which is the failure the feature
exists to prevent.

The value is pinned by its own single-line test rather than only by a
relationship assertion, so moving it again is one obvious edit; the extension
carries the constant twice (`sw.js` attaches it to requests, `options.js` seeds
the picker) and a test asserts those two agree with each other and with Python.
If they drift, the picker shows one level while submissions use another — and
the cache key follows the submission, so the picker would be quietly lying.

`CACHE_VERSION` stays at 1: adult keys are byte-identical by construction and
nothing on disk changed. Bumping it would needlessly invalidate every
watch-along `sections.json` too.

### Candidates that cannot honour it: warn, don't refuse

NLLB, MADLAD and OPUS-MT are seq2seq with no prompt surface, and
`translategemma_mlx.py` documents having none. Refusing a non-adult level there
would take out the offline fallback entirely. Staying silent is worse: this repo
lost two months to `[emotion_guidance]` being inert because the adapter dropped
an unsupported kwarg without saying so, and the NOTE left in `context.py` at the
time explicitly asked that any restored version warn. So the adapter emits one
`log.warning` **and** one `ctx.progress` line per job naming the candidate, and
translates anyway.

### Empty results are now counted

A provider that declines a line under `child` returns `content: null`, which the
candidate turns into `""` and the TTS adapter then skips — a silent hole, which
is precisely the outcome this feature is supposed to make impossible. The batch
path was already protected (`empty translation for idx=` → per-sentence
fallback); the per-sentence path was not. The adapter now counts blank
`target_text` and reports it to the progress log. `translated.json` also gains
`audience` and `audience_honored`, both additive and absent-able, and that file
is copied into the cache as `transcript.json` — so a cached dub is
self-describing about the register it was made at.

### Precedence

`extension > patmir.toml`. `sw.js` attaches the level to **both**
`cache:lookup` and `jobs:submit` itself, rather than at each call site, so the
two cannot resolve differently — a lookup that used a different level would
report "not dubbed" for a dub that exists, or offer play for a key nothing will
write. It also means `content.js`'s auto-play lookup is correct without knowing
the feature exists. `[pipeline.options].audience` is the default for `patmir
dub`, `patmir submit` and anything else that omits the field.

## Consequences

* Up to 3 dubs × 2 modes per video. Acceptable for a single-user tool, and only
  paid for levels actually used. The LRU-cache idea from the D1 ADR gets
  slightly more relevant.
* The download directory is keyed **audience-free** (`download_key`, computed at
  `NEUTRAL_AUDIENCE`): the source file is identical at every level, and
  re-picking a level on the same episode is the expected workflow, so keying the
  download by the audience-bearing key would re-download the whole video every
  time.
* A running stream keeps the level it started with. The level is in the cache
  key, so a new level is a new stream rather than a mid-flight switch.
* `patmir bench` gets **no flag**: it measures throughput, and a ~1700-character
  system-prompt delta on a cached prefix is not what it measures. It does pick up
  `[pipeline.options].audience` like any other CLI entry point, so benchmarks
  taken after this change run at `child` — which is the right default to measure,
  since it is what the extension sends.
* The clause text is prose and has not been **ear**-tested, but it was checked
  live against the active candidate on four deliberately coarse Russian lines at
  all three levels (2026-08-28). Direction is right and, more importantly,
  **nothing came back empty** — a threat kept its force while losing the gore
  ("blood all over the wall" -> "you'll hit the wall"), "sleep with her" became
  "be with her", and a `գրողի ծոցը` curse became "get out of here". The weakest
  case was substance reference: `child` still rendered "I want to get properly
  drunk", which the clause does permit ("named as plainly as the plot needs")
  but which an ear-test may want to soften further. Sample size four, one draw
  each — this is a smoke test, not evidence.

## Revisit

* An ear-test finds `child` over-softened, or `teenager` still too coarse.
* A local translate candidate becomes active — the guardrail then silently stops
  applying, and the adapter's warning is the tripwire that says so.
* Provider-side refusals show up as empty lines often enough to need a retry
  rather than a report.
