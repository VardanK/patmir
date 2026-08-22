"""Tests for `patmir.paths` — the `[storage]` table and root resolution.

The load-bearing properties:

* With no `[storage]` table, every root matches the pre-change hardcoded path,
  so an existing install is unaffected.
* Relative paths resolve against the CONFIG FILE's directory, not the CWD.
  This is the regression that made `runs/` mean different things depending on
  where `patmir` was invoked from.
* A typo'd key fails loudly instead of silently using the default.
* Storage never contributes to the cache key — relocating the share must not
  orphan existing dubs.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from patmir import paths
from patmir.pipeline.config import PipelineConfig, load_pipeline_config
from patmir.pipeline.context import PipelineOptions
from patmir.pipeline.stage import STAGE_ORDER
from patmir.ui import cache as cache_mod

_HOME = Path.home()

# A minimal but complete config: one candidate per stage so
# `load_pipeline_config` resolves every selector.
_MINIMAL_STAGES = """
[pipeline]
ingest = "i"
separate = "s"
diarize = "d"
emotion = "e"
asr = "a"
translate = "t"
tts = "v"
mux = "m"
output = "o"

[[stages.ingest.candidates]]
id = "i"
module = "x"
[[stages.separate.candidates]]
id = "s"
module = "x"
[[stages.diarize.candidates]]
id = "d"
module = "x"
[[stages.emotion.candidates]]
id = "e"
module = "x"
[[stages.asr.candidates]]
id = "a"
module = "x"
[[eval.translation.candidates]]
id = "t"
module = "x"
[[eval.tts.candidates]]
id = "v"
module = "x"
[[stages.mux.candidates]]
id = "m"
module = "x"
[[stages.output.candidates]]
id = "o"
module = "x"
"""


def _write_config(directory: Path, storage_table: str = "") -> Path:
    path = directory / "patmir.toml"
    path.write_text(_MINIMAL_STAGES + storage_table)
    return path


class LoadStorageTest(unittest.TestCase):
    def test_defaults_match_prechange_hardcoded_paths(self):
        """No [storage] table → exactly the old locations."""
        with tempfile.TemporaryDirectory() as td:
            cfg_path = _write_config(Path(td))
            st = load_pipeline_config(cfg_path).storage

            self.assertEqual(st.runs_root, Path(td) / "runs")
            self.assertEqual(st.cache_root, _HOME / ".patmir" / "cache")
            self.assertEqual(st.downloads_root, _HOME / ".patmir" / "downloads")
            self.assertEqual(st.state_root, _HOME / ".patmir")
            self.assertIsNone(st.library_root)
            # Derived paths the server used to hardcode.
            self.assertEqual(st.jobs_index, _HOME / ".patmir" / "jobs.json")
            self.assertEqual(st.secret_path, _HOME / ".patmir" / "secret")

    def test_relative_path_resolves_against_config_dir_not_cwd(self):
        """The five old `Path.cwd() / "runs"` fallbacks made this ambiguous."""
        with tempfile.TemporaryDirectory() as project, tempfile.TemporaryDirectory() as elsewhere:
            cfg_path = _write_config(Path(project), '\n[storage]\nruns_root = "work/runs"\n')
            prior = Path.cwd()
            try:
                os.chdir(elsewhere)
                st = load_pipeline_config(cfg_path).storage
            finally:
                os.chdir(prior)

            self.assertEqual(st.runs_root, Path(project) / "work" / "runs")
            self.assertNotIn(Path(elsewhere).name, str(st.runs_root))

    def test_absolute_and_tilde_paths(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = _write_config(Path(td), (
                '\n[storage]\n'
                'cache_root = "/Volumes/media/patmir/cache"\n'
                'library_root = "~/media/patmir/library"\n'
            ))
            st = load_pipeline_config(cfg_path).storage

            self.assertEqual(st.cache_root, Path("/Volumes/media/patmir/cache"))
            self.assertEqual(st.library_root, _HOME / "media" / "patmir" / "library")
            # Unspecified keys keep their defaults.
            self.assertEqual(st.downloads_root, _HOME / ".patmir" / "downloads")

    def test_partial_table_leaves_other_roots_at_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = _write_config(Path(td), '\n[storage]\nlibrary_root = "/tmp/lib"\n')
            st = load_pipeline_config(cfg_path).storage

            self.assertEqual(st.library_root, Path("/tmp/lib"))
            self.assertEqual(st.runs_root, Path(td) / "runs")
            self.assertEqual(st.state_root, _HOME / ".patmir")

    def test_unknown_key_raises(self):
        """A typo must not silently write to the default location."""
        with tempfile.TemporaryDirectory() as td:
            cfg_path = _write_config(Path(td), '\n[storage]\nruns-root = "/tmp/x"\n')
            with self.assertRaises(RuntimeError) as ctx:
                load_pipeline_config(cfg_path)
            self.assertIn("runs-root", str(ctx.exception))

    def test_empty_value_raises(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = _write_config(Path(td), '\n[storage]\nruns_root = "  "\n')
            with self.assertRaises(RuntimeError):
                load_pipeline_config(cfg_path)

    def test_non_table_storage_raises(self):
        with tempfile.TemporaryDirectory() as td:
            # The bare key has to precede every table header, or TOML nests it
            # inside the last table instead of at the top level.
            cfg_path = Path(td) / "patmir.toml"
            cfg_path.write_text('storage = "nope"\n' + _MINIMAL_STAGES)
            with self.assertRaises(RuntimeError):
                load_pipeline_config(cfg_path)

    def test_defaults_with_no_config_path_are_cwd_relative(self):
        """Matches the old `Path("runs")` behavior for config-less callers."""
        with tempfile.TemporaryDirectory() as td:
            prior = Path.cwd()
            try:
                os.chdir(td)
                st = paths.defaults(None)
            finally:
                os.chdir(prior)
            self.assertTrue(st.runs_root.is_absolute())
            self.assertEqual(st.runs_root.name, "runs")


class StorageWithoutFullConfigTest(unittest.TestCase):
    """`runs ls` / `runs show` / `submit` must not require a patmir.toml.

    They worked from any directory before `[storage]` existed — they only need
    the roots, never a stage. Regression guard: routing them through
    `load_pipeline_config` made them crash with a raw FileNotFoundError.
    """

    def test_missing_config_falls_back_to_defaults(self):
        from patmir.cli import _storage_for

        with tempfile.TemporaryDirectory() as td:
            prior = Path.cwd()
            try:
                os.chdir(td)
                st = _storage_for(Path("patmir.toml"))  # does not exist
            finally:
                os.chdir(prior)

            self.assertEqual(st.runs_root.name, "runs")
            self.assertEqual(st.secret_path, _HOME / ".patmir" / "secret")

    def test_storage_read_without_pipeline_selectors(self):
        """A config with [storage] but no [pipeline] still yields roots."""
        from patmir.cli import _storage_for

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "patmir.toml"
            cfg_path.write_text('[storage]\nlibrary_root = "/tmp/lib"\n')

            st = _storage_for(cfg_path)

            self.assertEqual(st.library_root, Path("/tmp/lib"))
            self.assertEqual(st.runs_root, Path(td) / "runs")

    def test_malformed_storage_still_raises(self):
        """Falling back on a MISSING file must not mask a broken one."""
        from patmir.cli import _storage_for

        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "patmir.toml"
            cfg_path.write_text('[storage]\nruns-root = "/tmp/x"\n')
            with self.assertRaises(RuntimeError):
                _storage_for(cfg_path)


class StorageExcludedFromCacheKeyTest(unittest.TestCase):
    """Moving the share must not orphan existing dubs."""

    def _cfg(self, storage) -> PipelineConfig:
        entries = {name: {"id": f"{name}_x", "module": "m"} for name in STAGE_ORDER}
        return PipelineConfig(
            entries=entries, options=PipelineOptions(), storage=storage,
        )

    def test_cache_key_identical_across_storage_changes(self):
        local = paths.defaults(None)
        networked = paths.StoragePaths(
            runs_root=Path("/Volumes/media/patmir/runs"),
            cache_root=Path("/Volumes/media/patmir/cache"),
            downloads_root=Path("/Volumes/media/patmir/downloads"),
            state_root=Path("/Volumes/media/patmir/state"),
            library_root=Path("/Volumes/media/patmir/library"),
        )
        self.assertEqual(
            cache_mod.cache_key("url:yt:abc", self._cfg(local)),
            cache_mod.cache_key("url:yt:abc", self._cfg(networked)),
        )


if __name__ == "__main__":
    unittest.main()
