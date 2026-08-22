# 2026-08-01 — Configurable storage roots (`[storage]`) + `library_root` publish

**Status:** accepted
**Supersedes:** nothing
**Affects:** `patmir/paths.py` (new), `patmir/pipeline/publish.py` (new), `patmir/pipeline/config.py`, `patmir/pipeline/runner.py`, `patmir/pipeline/manifest.py`, `patmir/cli.py`, `patmir/ui/{app,jobs,stream,cache,auth}.py`, `patmir/pipeline/streaming/{prelude,section}.py`

## Context

Every filesystem root was hardcoded at its point of use, and the finished dub was never
copied out of the working directory. Getting a translated video off the processing machine
meant digging into `runs/<job-id>/out.mkv` by hand.

Four roots, no single choke point:

| Root | Was defined at | Value |
| --- | --- | --- |
| `runs/` | `cli.py:29` | `Path("runs")` — CWD-relative |
| cache | `ui/cache.py:29` | `~/.patmir/cache` |
| downloads | `ui/jobs.py:33` | `~/.patmir/downloads` |
| server state | `ui/jobs.py:34`, `ui/auth.py:20` | `~/.patmir/jobs.json`, `~/.patmir/secret` |

`runs/` additionally had **five** independent `Path.cwd() / "runs"` fallbacks that silently
changed meaning with the invoking directory (`cli.py`, `runner.py`, `ui/app.py`,
`streaming/prelude.py`, `streaming/section.py` ×2). Only `patmir ui --runs-root` was
overridable, and `serve()` accepted a `cache_root` that no caller ever passed — so the
cache root was always `~/.patmir/cache` regardless.

Author's requirement: translated videos and other translated files on network storage, code
and working directories on the processing Mac.

## Decision

One `[storage]` table in `patmir.toml` decides every root. `patmir/paths.py` owns the
`StoragePaths` dataclass and the loader; `PipelineConfig.storage` carries the resolved
roots to the CLI, the server, the runner, and the streaming loop.

```toml
[storage]
# runs_root      = "runs"                        # relative → next to patmir.toml
# cache_root     = "~/.patmir/cache"
# downloads_root = "~/.patmir/downloads"
# state_root     = "~/.patmir"                   # bearer token + job index
# library_root   = "/Volumes/media/patmir/library"
```

- **Config file is the only knob.** No env vars, no new per-run flags. (`patmir ui
  --runs-root` predates this and stays, now defaulting to the config value.) `runs ls`,
  `runs show`, and `submit` gained `--config` so they can find the roots.
- **Relative paths resolve against the config file's directory**, not CWD. This one rule
  replaces the five divergent fallbacks.
- **Defaults reproduce the pre-change paths exactly** — an install with no `[storage]`
  table is unchanged.
- **Unknown keys raise**, so a typo'd `runs-root` fails loudly instead of silently writing
  to the default.
- **`library_root` is new and unset by default.** When set, each finished run's
  deliverables (`out.mkv`, `out.hy.srt`, `dub.wav`) are copied to
  `library_root/<job_id>/`. Per-job subdirectory, so repeat runs of the same source sit
  side by side with no collision handling.
- **Publish runs in the runner**, right after the `output` stage — same pattern as the
  other post-stage runner steps (`pick_refs`, `apply_speaker_tags`). `stages/output/mkv.py`
  keeps writing to `ctx.job_dir` and stays destination-agnostic.
- **Copy, not move.** The job dir has to survive: `--resume-from` reads absolute paths back
  out of the sidecar JSONs, and `cache.materialize` pulls `dub.wav` from there.
- **`.partial` then rename** inside the destination dir (same-filesystem), so an
  interrupted transfer never leaves a truncated `out.mkv` that looks valid. `copyfile`
  rather than `copy2` — `copystat` can raise or no-op on SMB/NFS and no mtime downstream
  depends on it.
- **Publish failures never fail the run.** The dub is already complete in the job dir; an
  unmounted share is reported via the progress log and swallowed.
- **Storage is excluded from the cache key.** `cache.config_hash` hashes only
  `cfg.entries`, so relocating the share never orphans an existing dub. Pinned by a test.
- **`_make_job_dir` no longer hard-fails on collision.** Job ids are second-granularity and
  watch-along calls `run_job` once per section; it now suffixes `-2`, `-3` instead of
  raising `FileExistsError`.

`Manifest` gained `published_dir`, surfaced by `patmir runs show` / `_print_summary`.

## Which roots are safe on network storage

`library_root` and `cache_root` are written in whole-file copies — fine.

`runs_root` is the live working directory and is best left local: `progress.jsonl` is
appended and byte-offset seek-tailed by the SSE loop (`ui/app.py`, `ui/jobs.py`),
`manifest.json` is rewritten non-atomically at every stage transition (`manifest.py`), and
every ffmpeg/soundfile write goes in place with no temp-then-rename. `state_root` holds the
server's bearer token and job index and should also stay local. The knobs exist because all
roots are configurable; the recommended configuration is `runs_root` + `state_root` local,
`library_root` (and optionally `cache_root`) on the share.

## Alternatives considered

- **A publish destination only, leaving the other roots hardcoded.** Smaller change, but
  the author explicitly wanted every root movable, and it would have left the five
  CWD-relative `runs/` fallbacks in place.
- **Env vars and/or per-run CLI flags in addition to the config table.** Rejected by the
  author — one obvious place to look beats three layers of precedence.
- **Flat `<stem>.hy.mkv` naming at the destination** instead of a per-job subdirectory.
  Rejected: repeat runs of the same source would collide and need suffixing.
- **Moving rather than copying the deliverables.** Would break `--resume-from` and
  `cache.materialize`, both of which read out of the job dir after the run returns.

## Out of scope

1. **ECAPA model cache stays hardcoded** at `~/.patmir/models/ecapa`
   (`streaming/embeddings.py:32`) — a model download, not a translated file. Threading it
   means changing the `lru_cache`d `embed()` signature through `registry.py` /
   `prelude.py` / `section.py`, and it belongs on fast local disk anyway.
2. **Existing runs won't resume after `runs_root` changes.** Sidecar JSONs
   (`audio.meta.json`, `separated.meta.json`, `refs.json`, `synthesized.json`) store
   absolute paths, read back verbatim by `_load_resume_artifact`. Making them job-dir
   relative is a separate change.
3. **No multi-writer locking.** Confirmed single machine. There is no `flock`/sqlite
   anywhere in `patmir/`; safety rests on the single asyncio worker at concurrency 1. Two
   `patmir` processes against a shared cache root would corrupt it.
4. **Watch-along mode publishes nothing.** Sections run `run_job(stop_after="mux")`, so the
   `output` stage — and therefore the publish step — never fires. This is the right default
   (a library full of 20 per-section MKVs would be useless), but it means the only way to
   get a library entry today is a Quality-mode (`patmir dub` / batch) run. Watch-along
   sections still land in `cache_root/<key>/section_NNN.wav`, so a "stitch the cached
   sections into one library MKV" step is the obvious follow-up if the author wants it.

## Verification

- `patmir/tests/test_paths.py` — defaults match the old constants; a relative `runs_root`
  resolves against the config dir asserted from a different CWD; `~` expands; unknown /
  empty / non-table values raise; `cache_key` identical across storage changes.
- `patmir/tests/test_publish.py` — deliverables land in `library_root/<job_id>/`,
  intermediates don't; job dir untouched; no `.partial` after success *or* failure;
  idempotent rerun; missing deliverables skipped; nothing-to-publish raises.
- Full suite: 134 tests green.
- E2E: a staged job resumed with `--resume-from output` against a config with
  `library_root` set published all three files; the MKV probed clean (h264 video, `hye`
  default audio, `ru` original retained, `hye` subs), byte-identical to the job dir, no
  `.partial`, `published_dir` recorded in the manifest.
- Server: with all five roots redirected, `ensure_token` wrote to the configured
  `state_root`, `materialize` wrote to the configured `cache_root`, and `/cache/lookup` +
  `/cache/<key>/dub.wav` served from it (1300260 bytes).
- Unconfigured `patmir.toml` resolves to exactly the pre-change locations.

## Revisit trigger

If the cache moves to network storage in practice and per-section `append_section` copies
(synchronous inside the streaming loop) hurt watch-along latency; if a second machine ever
needs to write the share (then locking becomes load-bearing); or if publishing needs a
filename template rather than a per-job directory.
