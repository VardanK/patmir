# Configurable storage roots — shaping notes

## Scope

Make every filesystem root configurable from one `[storage]` table in `patmir.toml`, and
add a new `library_root` that receives the finished deliverables of each run.

Motivation (author, 2026-08-01): put translated videos and other translated files on
network storage, keep the code and the working directories on the processing machine.
Today the only way to get a finished dub off the machine is to dig into
`runs/<job-id>/out.mkv` — that file is written there and never copied anywhere.

## Path inventory before the change

Four roots baked in, no single choke point:

| Root | Defined at | Value |
| --- | --- | --- |
| `runs/` | `patmir/cli.py:29` | `Path("runs")` — CWD-relative |
| cache | `patmir/ui/cache.py:29` | `~/.patmir/cache` |
| downloads | `patmir/ui/jobs.py:33` | `~/.patmir/downloads` |
| server state | `patmir/ui/jobs.py:34`, `patmir/ui/auth.py:20` | `~/.patmir/jobs.json`, `~/.patmir/secret` |

`runs/` had **five** independent CWD-relative fallbacks that silently changed meaning with
the invoking directory: `cli.py:29`, `runner.py:64`, `ui/app.py:333`,
`streaming/prelude.py:111`, `streaming/section.py:163` and `:206`.

Only `patmir ui --runs-root` was overridable. `serve()` accepted a `cache_root` that no
caller ever passed, so the cache root was always `~/.patmir/cache` regardless.

## Decisions

- **All roots configurable**, not just a publish destination. Author picked the widest
  option so any root can move later without a code change.
- **Config file is the only knob.** No env vars, no new per-run CLI flags. Keeps one
  obvious place to look. (`patmir ui --runs-root` already existed and stays, now
  defaulting to the config value.)
- **Relative paths resolve against the config file's directory**, not CWD. This is what
  replaces the five divergent fallbacks with one rule.
- **Publish layout: per-job subdirectory** — `library_root/<job_id>/{out.mkv, out.hy.srt,
  dub.wav}`. Repeat runs of the same source sit side by side with no collision handling.
- **Copy, not move.** `runs/<job>/` stays intact so `--resume-from` and
  `cache.materialize` keep working (both read absolute paths out of the job dir).
- **Write `.partial` then rename** inside the destination dir, so an interrupted transfer
  to the share never leaves a truncated `.mkv` that looks valid.
- **Single writer, no locking** — author confirmed one machine writes the share.
- **Unknown keys under `[storage]` raise**, so a typo'd `runs-root` fails loudly instead
  of silently writing to the default.
- **Defaults reproduce today's behavior exactly**, so an install with no `[storage]` table
  is unchanged.

## Deliberately out of scope

1. **ECAPA model cache stays hardcoded** at `~/.patmir/models/ecapa`
   (`streaming/embeddings.py:32`) — a model download, not a translated file. Threading it
   would mean changing the `lru_cache`d `embed()` signature through `registry.py` /
   `prelude.py` / `section.py`, and it should stay on fast local disk anyway.
2. **Existing runs won't resume after `runs_root` changes.** Sidecar JSONs
   (`audio.meta.json`, `separated.meta.json`, `refs.json`, `synthesized.json`) store
   absolute paths, read back verbatim by `_load_resume_artifact`. Making them job-dir
   relative is a separate, larger change.
3. **No multi-writer locking.** There is no `flock`/sqlite anywhere in `patmir/`; safety
   today rests on the single asyncio worker at concurrency 1 (`ui/jobs.py:120, 244-266`).
   Two `patmir` processes against a shared cache root would corrupt it.
4. **Working dirs on network storage stay risky if pointed there.** `progress.jsonl` is
   appended and byte-offset-seek-tailed by the SSE loop, `manifest.json` is rewritten
   non-atomically per stage, and ffmpeg/soundfile write in place with no temp-then-rename.
   The knob exists because all roots are configurable; the recommended configuration is
   `runs_root` local, `library_root` (and optionally `cache_root`) on the share.

Verified safe during shaping: the `resolve()`-based path containment checks
(`ui/app.py:78-84, 171-177`) still work when the roots themselves are symlinks, because
both roots are `resolve()`d at app build. They break only if an individual job/key dir is
a symlink.

## Context

- **Visuals:** None — no UI surface.
- **References:** see `references.md`.
- **Product alignment:** consistent with local-first strict (ADR
  `2026-05-24-local-first-strict`) — network storage here is the author's own NAS, not a
  cloud service, and no stage gains a network dependency. The default configuration stays
  fully local.

## Standards Applied

None. `agent-os/standards/index.yml` contains only a header comment — no standards are
indexed in this repo yet. See `standards.md`.
