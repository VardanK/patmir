# References for configurable storage roots

## Extension point

### The config loader

- **Location:** `patmir/pipeline/config.py` — `load_pipeline_config` (`:90-169`)
- **Relevance:** the only TOML reader in the codebase. Every root has to come from here,
  and it already holds both the parsed `toml` dict and the `config_path` needed to resolve
  relative paths against the config file's directory.
- **Key patterns:** `PipelineConfig` (`:42-51`) is the object every entry point already
  receives, so a `storage` field on it reaches the CLI, the server, the runner, and the
  streaming loop with no new plumbing.

### Top-level tables, not `[pipeline.options]` scalars

- **Location:** `config.py:126-136` (`[emotion_guidance]`) and `config.py:154-156`
  (`[[tts_ref_pool]]`)
- **Relevance:** the precedent for reading a top-level table rather than adding scalars to
  `[pipeline.options]`. Storage roots are neither per-stage nor audio-tuning knobs, so
  `[storage]` is the right shape.
- **Key patterns:** `toml.get("<table>", {}) or {}`, then fold into a typed object.

### Cache key derivation

- **Location:** `patmir/ui/cache.py:107-124` (`config_hash`), `:141-147` (`cache_key`)
- **Relevance:** hashes only `cfg.entries` + `CACHE_VERSION`, deliberately excluding
  `[pipeline.options]` so ear-test tweaks don't bust the cache. Storage inherits that
  exclusion for free — moving the share must not orphan existing dubs. Worth a test that
  pins it.

## Patterns to borrow

### Post-stage runner steps

- **Location:** `patmir/pipeline/runner.py` — `pick_refs` after `emotion`,
  `apply_speaker_tags` / `apply_emotion_tags` / `_backfill_ref_text` after `asr`
  (documented in `docs/architecture/pipeline.md`)
- **Relevance:** the publish step is the same shape — work that bridges a stage and
  something outside it, with no independent value as a stage. Putting it in the runner
  keeps `output/mkv.py:42-43` writing to `ctx.job_dir` and destination-agnostic.

### Path arguments on the CLI

- **Location:** `patmir/cli.py:223-228` (`cmd_bench --report`)
- **Relevance:** the one existing output-path flag. Pattern:
  `path.parent.mkdir(parents=True, exist_ok=True)` before writing.

### Atomic-ish writes

- **Location:** `patmir/ui/cache.py:197-202` (`_save_by_url`), `:367-371` (stream manifest)
- **Relevance:** `tmp.write_text(...)` then `tmp.replace(path)` within the same directory.
  The publish step uses the same shape (`.partial` then `.replace()`) so a half-copied MKV
  never appears on the share as a valid file.

### Already-parameterized roots

- `patmir/ui/auth.py:23` — `ensure_token(secret_path=...)` already takes a path.
- `patmir/ui/jobs.py:112` — `JobManager(..., downloads_dir=...)` already takes one; nothing
  ever passes it (`ui/app.py:53`), so the default always won.
- `patmir/ui/app.py:315-323` — `serve(cache_root=...)` likewise accepted but never passed.

These three are why the change is mostly wiring rather than new abstraction.

## Sites that assume same-filesystem or local behavior

Catalogued during shaping; informs which roots are safe to point at a share.

- **Non-atomic JSON rewrites:** `manifest.py:58` (per stage transition),
  `ui/jobs.py:285-291` (jobs index).
- **Append + byte-offset seek-tail:** `progress.jsonl` written at `context.py:132-134`,
  read with `f.seek(pos)` at `ui/app.py:128-137` and `f.seek(size - 4096)` at
  `ui/jobs.py:454-465`.
- **In-place writes with no temp-then-rename:** every ffmpeg subprocess
  (`ingest/ffmpeg.py:35`, `output/mkv.py:76`, `streaming/section.py:59-68`) and every
  `soundfile.write` (`mux/ffmpeg_pad_stretch.py:243`, `separate/htdemucs*.py`,
  `ref_picker.py:168`).
- **Full byte copies into the cache:** `shutil.copy2` at `ui/cache.py:235, 239, 391`. The
  section copy at `:391` runs synchronously inside the streaming loop
  (`ui/stream.py:352-358`), so a network cache root adds a round trip per section.
- **Absolute paths inside sidecar JSONs:** `audio.meta.json` (`ingest/ffmpeg.py:56-60`),
  `separated.meta.json`, `dub.meta.json`, `synthesized.json`, `diarize/refs.json`; read
  back verbatim by `runner._load_resume_artifact` (`:410, 413-421, 445, 504-512`).
- **No locking anywhere:** no `flock`, sqlite, mmap, hardlinks, or symlink creation in
  `patmir/`.
