# Configurable storage roots

## Context

Every filesystem root in Patmir is hardcoded today. Finished dubs are written to
`runs/<job-id>/out.mkv` and **never copied anywhere** — the only way to get a translated
video off the processing machine is to dig into `runs/`. The author wants translated
videos and other translated files on network storage while the code and working
directories stay on the processing Mac.

Four separate roots are currently baked in, with no single choke point:

| Root | Defined at | Value |
| --- | --- | --- |
| `runs/` | `patmir/cli.py:29` | `Path("runs")` — **CWD-relative** |
| cache | `patmir/ui/cache.py:29` | `~/.patmir/cache` |
| downloads | `patmir/ui/jobs.py:33` | `~/.patmir/downloads` |
| server state | `patmir/ui/jobs.py:34`, `patmir/ui/auth.py:20` | `~/.patmir/jobs.json`, `~/.patmir/secret` |

`runs/` additionally has **five** independent CWD-relative fallbacks that silently change
meaning with the invoking directory: `cli.py:29`, `runner.py:64`, `ui/app.py:333`,
`streaming/prelude.py:111`, `streaming/section.py:163` and `:206`. Only `patmir ui
--runs-root` is overridable today, and `serve()` accepts a `cache_root` that no caller
ever passes — so the cache root is always `~/.patmir/cache` regardless.

Outcome: one `[storage]` table in `patmir.toml` decides all roots, plus a new
`library_root` that receives the finished deliverables in a per-job subdirectory.
Defaults stay byte-identical to today's behavior, so an unconfigured install is unchanged.

### Decisions locked during shaping

- **All roots configurable** (not just a publish dir), via a `[storage]` table.
- **Config file is the only knob** — no env vars, no new per-run CLI flags. Relative
  paths resolve against the config file's directory, not CWD.
- **Publish layout: per-job subdirectory** — `library_root/<job_id>/{out.mkv, out.hy.srt, dub.wav}`.
- **Single writer** — one machine writes the share; no locking.
- `agent-os/standards/index.yml` is empty, so no coding standards apply.

---

## Task 1: Save spec documentation

Create `agent-os/specs/2026-08-01-1640-configurable-storage-roots/` with `plan.md`
(this plan), `shape.md` (scope + the four locked decisions + the path inventory above),
`standards.md` (records that the standards index is empty), `references.md` (the config
loader as the extension point, `[emotion_guidance]`/`[[tts_ref_pool]]` as the precedent
for a top-level table, `--report` in `cmd_bench` as the precedent for a path flag).
No `visuals/` — nothing visual here.

## Task 2: `patmir/paths.py` — `StoragePaths` + loader

New module, the single choke point.

```python
@dataclass(frozen=True)
class StoragePaths:
    runs_root: Path
    cache_root: Path
    downloads_root: Path
    state_root: Path
    library_root: Path | None = None      # None = don't publish

    @property
    def jobs_index(self) -> Path:  return self.state_root / "jobs.json"
    @property
    def secret_path(self) -> Path: return self.state_root / "secret"
```

`load_storage(toml: dict, config_path: Path) -> StoragePaths` reads the `[storage]`
table. Resolution per key: `expanduser()`, then resolve relative paths against
`config_path.parent` (**not** `Path.cwd()` — this is what kills the five divergent
fallbacks). Defaults exactly reproduce today's constants:

| Key | Default |
| --- | --- |
| `runs_root` | `<config_dir>/runs` |
| `cache_root` | `~/.patmir/cache` |
| `downloads_root` | `~/.patmir/downloads` |
| `state_root` | `~/.patmir` |
| `library_root` | unset → `None` |

Unknown keys under `[storage]` raise, so a typo'd `runs-root` fails loudly instead of
silently writing to the default.

## Task 3: Attach storage to `PipelineConfig`

`patmir/pipeline/config.py` — add `storage: StoragePaths` to the `PipelineConfig`
dataclass (`config.py:42-51`) and populate it in `load_pipeline_config` (it already has
both `toml` and `config_path` in hand, at `:110-111`). Follows the existing
`[emotion_guidance]` / `[[tts_ref_pool]]` pattern of reading a top-level table.

`cache.config_hash` (`ui/cache.py:107-124`) hashes only `cfg.entries`, so storage
**cannot** bust the cache — moving the share must not orphan existing dubs. Add a comment
plus a test pinning that.

## Task 4: Thread it through the entry points

Mechanical replacement of the hardcoded roots. Every site already has `cfg` in scope or
one call away.

- **`patmir/cli.py`** — delete `RUNS_ROOT` (`:29`); every command reads
  `cfg.storage.runs_root`. `runs ls` / `runs show` currently take no config, so add
  `--config` to both (`:52-54`) and pass the root into `_print_diarize_summary`
  (`:295-299`, drops its `RUNS_ROOT` reference). `cmd_ui`: `--runs-root` default becomes
  `None`, falling back to `cfg.storage.runs_root`; pass `cache_root` through to `serve()`
  — today it never is. `ensure_token(cfg.storage.secret_path)` at `:191`.
- **`patmir/ui/app.py`** — `_create_app` / `serve` take `storage: StoragePaths` instead of
  the two separate roots. Keep `runs_root` / `cache_root` as locals derived from it
  (`:48-50`) so the ~40 route bodies below are untouched. Drop the
  `Path("runs")` / `DEFAULT_CACHE_ROOT` fallbacks at `:333-334`.
- **`patmir/ui/jobs.py`** — `JobManager(cfg, *, storage)`; `self._downloads_dir` comes
  from `storage.downloads_root` (the `downloads_dir` param exists at `:112` but nothing
  ever passes it); `JOBS_INDEX_PATH` (`:34`, used at `:287-289`) becomes
  `self._storage.jobs_index`.
- **`patmir/ui/stream.py`** — same `storage=` swap as `JobManager`.
- **`patmir/pipeline/runner.py:64`** — `runs_root = runs_root or cfg.storage.runs_root`.
  `cfg` is already a parameter, so the fallback becomes correct rather than
  CWD-dependent. Same substitution at `streaming/prelude.py:111` and
  `streaming/section.py:163, 206`.

Also relax `_make_job_dir`'s `mkdir(..., exist_ok=False)` (`runner.py:26`) to retry with a
`-2`, `-3` suffix. Job ids are second-granularity and watch-along calls `run_job` once per
section; a slow-metadata network mount makes the current hard failure reachable.

## Task 5: Publish deliverables to `library_root`

New `patmir/pipeline/publish.py`:

```python
def publish_job(job_dir: Path, job_id: str, artifacts: OutputArtifacts,
                library_root: Path) -> Path
```

Creates `library_root/<job_id>/` and copies `out.mkv`, `out.hy.srt`, and `dub.wav` into
it. Each copy goes to `<name>.partial` first, then `.replace()` into place **within the
destination directory** (same-filesystem rename) — otherwise an interrupted transfer
leaves a truncated `.mkv` on the share that looks like a valid file.

Called from `runner.py` immediately after the `output` stage's `_record_artifacts`
(`runner.py:322-331`), guarded by `if cfg.storage.library_root`. This mirrors the existing
"post-stage runner steps" pattern (`pick_refs`, `apply_speaker_tags`) rather than adding
a stage or teaching `output/mkv.py` about destinations — `mkv.py:42-43` keeps writing to
`ctx.job_dir` and stays destination-agnostic.

**Copy, not move.** `runs/<job>/` must stay intact: `--resume-from` reads absolute paths
back out of the sidecar JSONs (`runner.py:410, 413-421, 445, 504-512`), and
`cache.materialize` (`ui/cache.py:230-239`) pulls `dub.wav` from there.

Record the published directory in the manifest so `patmir runs show` surfaces it.

## Task 6: Docs (maintenance contract)

- New ADR `docs/architecture/decisions/2026-08-01-configurable-storage-roots.md` +
  one-line entry in `decisions.md`.
- `docs/architecture/data-contracts.md` — the new `[storage]` table and `StoragePaths`.
- `docs/architecture/runs.md` (on-demand tier) — the file map gains the
  `library_root/<job_id>/` publish destination.
- `docs/architecture/pipeline.md` — the ingest-sources and cache paragraphs name
  `~/.patmir/...` literals; note they are now defaults.

---

## Deliberately out of scope

State these rather than silently skipping them:

1. **ECAPA model cache stays hardcoded** at `~/.patmir/models/ecapa`
   (`streaming/embeddings.py:32`). It's a model download, not a translated file, and
   threading it means changing the `lru_cache`d `embed()` signature through
   `registry.py` / `prelude.py` / `section.py` — churn for something that should stay on
   fast local disk anyway.
2. **Existing runs won't resume after `runs_root` changes.** Sidecar JSONs
   (`audio.meta.json`, `separated.meta.json`, `refs.json`, `synthesized.json`) store
   absolute paths, read back verbatim by `_load_resume_artifact`. Making them job-dir
   relative is a separate, larger change.
3. **No multi-writer locking** — confirmed single machine. There is no `flock`/sqlite
   anywhere in `patmir/`; safety today rests entirely on the single asyncio worker at
   concurrency 1 (`ui/jobs.py:120, 244-266`). Two `patmir` processes against a shared
   cache root would corrupt it.
4. **Working dirs on network storage remain risky if pointed there.** `progress.jsonl` is
   appended and byte-offset-seek-tailed by the SSE loop (`ui/app.py:128-137`,
   `ui/jobs.py:454-465`), `manifest.json` is rewritten non-atomically per stage
   (`manifest.py:58`), and ffmpeg/soundfile write in place with no temp-then-rename. The
   knob will exist because all roots are configurable; the recommended configuration is
   `runs_root` local, `library_root` (and optionally `cache_root`) on the share.

Verified safe: the `resolve()`-based path containment checks (`ui/app.py:78-84, 171-177`)
still work when the roots themselves are symlinks, because both roots are `resolve()`d at
app build (`:48-49`). They break only if an individual job/key dir is a symlink.

---

## Verification

**Unit** — new `patmir/tests/test_paths.py`:
- No `[storage]` table → every default matches today's hardcoded constant.
- A relative `runs_root` resolves against the config file's directory, asserted from a
  *different* CWD (this is the regression the five old fallbacks had).
- `~` expands; `library_root` absent → `None`; an unknown key raises.
- `config_hash` is identical for two configs differing only in `[storage]`.

New `patmir/tests/test_publish.py`: `publish_job` lands all three files under
`<library>/<job_id>/`, leaves no `.partial` behind, and leaves `job_dir` untouched.

**Existing suite** — `python -m pytest patmir/tests`. `test_ui_cache.py` /
`test_stream_cache.py` pass `cache_root` directly to `cache_mod` functions whose
signatures don't change, so they should be unaffected; any test constructing
`JobManager` / `StreamManager` needs the `storage=` kwarg.

**E2E, defaults unchanged** — with no `[storage]` table:
`patmir dub samples/Reacher-S02E01-5min.mkv` still produces `runs/<job>/out.mkv`, and
`patmir runs ls` still lists it. Nothing appears under a library dir.

**E2E, network storage** — add `[storage]` with `library_root` (and `cache_root`) on the
mounted share, rerun the same clip, then confirm:
- `library_root/<job_id>/` holds `out.mkv`, `out.hy.srt`, `dub.wav`, no `.partial`;
- the MKV plays with Armenian as the default audio track;
- `runs/<job>/` is still complete and `patmir dub --resume-from tts --resume-job <dir>`
  still works.

**Server + extension** — `patmir ui`, then `patmir submit --url <youtube>`:
`/cache/<key>/dub.wav` must serve from the configured `cache_root` (not
`~/.patmir/cache`), the download must land under the configured `downloads_root`, and
the extension's "Play dub" path must still resolve. Then a watch-along submit to confirm
`sections.json` + `section_NNN.wav` land in the configured cache root and the browser
client streams them.
