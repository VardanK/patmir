"""Tests for `_free_accelerator_cache` — the between-stage memory hygiene call.

This exists because of a real outage. `_free_accelerator_cache()` runs after
EVERY stage and probed MLX under `except ImportError`. On the CUDA workstation
the `mlx` wheel was installed (an unmarked Apple-only dependency in the `local`
extra), and importing it fails inside dlopen with:

    OSError: libmlx.so: cannot open shared object file: No such file or directory

`OSError` is not an `ImportError`, so it escaped the guard and killed every run
after the first stage — on a machine whose active profile selected no MLX
candidate anywhere. Nothing about a run depends on this function succeeding, so
it must swallow whatever any accelerator throws.
"""
from __future__ import annotations

import builtins
import unittest

from patmir.pipeline import runner


class _MemoReset(unittest.TestCase):
    def setUp(self):
        self._saved = runner._MLX_CLEAR_CACHE
        runner._MLX_CLEAR_CACHE = None

    def tearDown(self):
        runner._MLX_CLEAR_CACHE = self._saved


def _blocking_import(exc: Exception):
    """An `__import__` that raises `exc` for any mlx module, else behaves."""
    real = builtins.__import__

    def fake(name, *args, **kwargs):
        if name.split(".")[0] == "mlx":
            raise exc
        return real(name, *args, **kwargs)

    return fake


class MlxProbeTest(_MemoReset):
    def test_oserror_from_dlopen_does_not_escape(self):
        """The exact production failure: OSError, not ImportError."""
        original = builtins.__import__
        builtins.__import__ = _blocking_import(
            OSError("libmlx.so: cannot open shared object file: No such file or directory")
        )
        try:
            runner._free_accelerator_cache()   # must not raise
        finally:
            builtins.__import__ = original
        self.assertIs(runner._MLX_CLEAR_CACHE, False)

    def test_missing_package_does_not_escape(self):
        original = builtins.__import__
        builtins.__import__ = _blocking_import(ModuleNotFoundError("No module named 'mlx'"))
        try:
            runner._free_accelerator_cache()
        finally:
            builtins.__import__ = original
        self.assertIs(runner._MLX_CLEAR_CACHE, False)

    def test_probe_is_memoized_so_it_is_not_retried_per_stage(self):
        """A failed import is not cached by Python, so without memoization this
        redoes the whole dlopen attempt nine times per job."""
        original = builtins.__import__
        calls = {"n": 0}

        def counting(name, *args, **kwargs):
            if name.split(".")[0] == "mlx":
                calls["n"] += 1
                raise OSError("libmlx.so: cannot open shared object file")
            return original(name, *args, **kwargs)

        builtins.__import__ = counting
        try:
            for _ in range(5):
                runner._free_accelerator_cache()
        finally:
            builtins.__import__ = original
        self.assertEqual(calls["n"], 1, "MLX should be probed once, not once per stage")

    def test_a_working_clear_cache_is_called(self):
        """Stub real modules in `sys.modules` rather than patching `__import__`:
        `import mlx.core as mx` returns the TOP-level module and then does
        `getattr(..., "core")`, so a single fake object is not enough."""
        import sys
        import types

        called = {"n": 0}
        mlx = types.ModuleType("mlx")
        core = types.ModuleType("mlx.core")
        core.clear_cache = lambda: called.__setitem__("n", called["n"] + 1)
        mlx.core = core
        saved = {k: sys.modules.get(k) for k in ("mlx", "mlx.core")}
        sys.modules["mlx"], sys.modules["mlx.core"] = mlx, core
        try:
            runner._free_accelerator_cache()
            runner._free_accelerator_cache()
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
        self.assertEqual(called["n"], 2, "resolved callable should be reused, not re-probed")

    def test_the_deprecated_metal_api_is_used_as_a_fallback(self):
        """MLX moved `clear_cache` off `mx.metal`; support both."""
        import sys
        import types

        called = {"n": 0}
        mlx = types.ModuleType("mlx")
        core = types.ModuleType("mlx.core")
        metal = types.ModuleType("mlx.core.metal")
        metal.clear_cache = lambda: called.__setitem__("n", called["n"] + 1)
        core.metal = metal          # no `core.clear_cache` — force the fallback
        mlx.core = core
        saved = {k: sys.modules.get(k) for k in ("mlx", "mlx.core")}
        sys.modules["mlx"], sys.modules["mlx.core"] = mlx, core
        try:
            runner._free_accelerator_cache()
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
        self.assertEqual(called["n"], 1)

    def test_a_broken_clear_cache_does_not_escape(self):
        runner._MLX_CLEAR_CACHE = None

        def boom():
            raise RuntimeError("metal is unhappy")

        runner._MLX_CLEAR_CACHE = boom
        runner._free_accelerator_cache()   # must not raise


class TorchProbeTest(_MemoReset):
    def test_a_broken_torch_does_not_escape(self):
        """Same reasoning as MLX: an accelerator that isn't in use must not get a
        vote on whether the job finishes."""
        original = builtins.__import__

        def fake(name, *args, **kwargs):
            if name.split(".")[0] == "torch":
                raise OSError("libtorch_cuda.so: cannot open shared object file")
            return original(name, *args, **kwargs)

        builtins.__import__ = fake
        try:
            runner._free_accelerator_cache()   # must not raise
        finally:
            builtins.__import__ = original


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
