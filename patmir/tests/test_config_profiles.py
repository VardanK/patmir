"""Tests for machine profiles — `[pipeline.profiles.<name>]`.

One committed `patmir.toml` has to serve two machines with different
accelerators (Apple Silicon / MLX and the WSL2 + RTX 3090 / CUDA workstation).
The load-bearing properties:

* No profile → the `[pipeline]` selectors, byte for byte. An existing install
  must be unaffected.
* A profile overlays only the stages it names; the rest inherit.
* Resolution order is `[pipeline] < profile < overrides < profile.<layer>`.
  The `<layer>` tier exists because `streaming_config()` hardcodes MLX-only
  candidates as `overrides`, and a CUDA machine has to be able to replace them.
* An unknown profile name is a HARD ERROR. Falling back to the defaults would
  mean silently trying to load MLX-only candidates on a CUDA box.
* The real `cuda` profile selects no MLX/MPS candidate in either mode — the
  regression this whole mechanism exists to prevent.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from patmir.pipeline.config import active_profile, load_pipeline_config

_REPO_CONFIG = Path(__file__).resolve().parents[2] / "patmir.toml"

# Minimal but complete: two candidates for the stages the profile swaps, one
# for the rest, so every selector resolves without importing a model.
_CONFIG = """
[pipeline]
ingest = "i"
separate = "s_mlx"
diarize = "d"
emotion = "e"
asr = "a"
translate = "t"
tts = "v_mlx"
mux = "m"
output = "o"

[pipeline.profiles.cuda]
separate = "s_torch"
tts = "v_torch"
not_a_stage = "ignored"

[pipeline.profiles.cuda.streaming]
separate = "s_torch_fast"

[pipeline.profiles.bare]
tts = "v_torch"

[[stages.ingest.candidates]]
id = "i"
module = "x"
[[stages.separate.candidates]]
id = "s_mlx"
module = "x"
[[stages.separate.candidates]]
id = "s_torch"
module = "x"
[[stages.separate.candidates]]
id = "s_torch_fast"
module = "x"
[[stages.separate.candidates]]
id = "s_mlx_fast"
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
id = "v_mlx"
module = "x"
[[eval.tts.candidates]]
id = "v_torch"
module = "x"
[[stages.mux.candidates]]
id = "m"
module = "x"
[[stages.output.candidates]]
id = "o"
module = "x"
"""


def _ids(cfg) -> dict[str, str]:
    return {stage: entry.get("id") for stage, entry in cfg.entries.items()}


class ProfileFixture(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.cfg_path = Path(self._td.name) / "patmir.toml"
        self.cfg_path.write_text(_CONFIG)
        # Never let the developer's own shell leak into these assertions.
        self._saved = os.environ.pop("PATMIR_PROFILE", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["PATMIR_PROFILE"] = self._saved
        else:
            os.environ.pop("PATMIR_PROFILE", None)
        self._td.cleanup()


class ProfileResolutionTest(ProfileFixture):
    def test_no_profile_uses_pipeline_selectors(self):
        ids = _ids(load_pipeline_config(self.cfg_path))
        self.assertEqual(ids["separate"], "s_mlx")
        self.assertEqual(ids["tts"], "v_mlx")

    def test_profile_overlays_only_named_stages(self):
        ids = _ids(load_pipeline_config(self.cfg_path, profile="cuda"))
        self.assertEqual(ids["separate"], "s_torch")
        self.assertEqual(ids["tts"], "v_torch")
        # Untouched stages inherit.
        self.assertEqual(ids["asr"], "a")
        self.assertEqual(ids["diarize"], "d")

    def test_non_stage_keys_in_a_profile_are_ignored(self):
        """`not_a_stage` must not become a selector or blow up the load."""
        cfg = load_pipeline_config(self.cfg_path, profile="cuda")
        self.assertNotIn("not_a_stage", cfg.entries)

    def test_unknown_profile_raises_instead_of_falling_back(self):
        with self.assertRaises(RuntimeError) as ctx:
            load_pipeline_config(self.cfg_path, profile="nope")
        self.assertIn("nope", str(ctx.exception))
        # The message has to name what IS defined, or a typo is a guessing game.
        self.assertIn("cuda", str(ctx.exception))

    def test_profile_requested_but_none_defined_raises(self):
        """A config with no [pipeline.profiles] at all — same hard failure."""
        stripped = _CONFIG.replace("""
[pipeline.profiles.cuda]
separate = "s_torch"
tts = "v_torch"
not_a_stage = "ignored"

[pipeline.profiles.cuda.streaming]
separate = "s_torch_fast"

[pipeline.profiles.bare]
tts = "v_torch"
""", "")
        self.assertNotIn("profiles", stripped)
        path = Path(self._td.name) / "noprofiles.toml"
        path.write_text(stripped)
        with self.assertRaises(RuntimeError):
            load_pipeline_config(path, profile="cuda")
        # ...but with no profile requested it still loads fine.
        self.assertEqual(_ids(load_pipeline_config(path))["tts"], "v_mlx")


class PrecedenceTest(ProfileFixture):
    """`[pipeline] < profile < overrides < profile.<layer>`."""

    def test_overrides_beat_the_profile_base(self):
        ids = _ids(load_pipeline_config(
            self.cfg_path, profile="bare", overrides={"separate": "s_mlx_fast"}))
        self.assertEqual(ids["separate"], "s_mlx_fast")
        self.assertEqual(ids["tts"], "v_torch")

    def test_profile_layer_beats_overrides(self):
        """The reason the layer tier exists: streaming_config() hardcodes an
        MLX-only `separate`, and the CUDA profile has to win over it."""
        ids = _ids(load_pipeline_config(
            self.cfg_path, profile="cuda", layer="streaming",
            overrides={"separate": "s_mlx_fast"}))
        self.assertEqual(ids["separate"], "s_torch_fast")

    def test_layer_without_a_matching_subtable_is_inert(self):
        ids = _ids(load_pipeline_config(
            self.cfg_path, profile="bare", layer="streaming",
            overrides={"separate": "s_mlx_fast"}))
        self.assertEqual(ids["separate"], "s_mlx_fast")

    def test_layer_ignored_when_no_profile_is_active(self):
        """A Mac watch-along run keeps its own overrides."""
        ids = _ids(load_pipeline_config(
            self.cfg_path, layer="streaming",
            overrides={"separate": "s_mlx_fast"}))
        self.assertEqual(ids["separate"], "s_mlx_fast")


class EnvSelectionTest(ProfileFixture):
    def test_env_var_selects_the_profile(self):
        os.environ["PATMIR_PROFILE"] = "cuda"
        self.assertEqual(_ids(load_pipeline_config(self.cfg_path))["tts"], "v_torch")

    def test_explicit_argument_wins_over_env(self):
        os.environ["PATMIR_PROFILE"] = "cuda"
        ids = _ids(load_pipeline_config(self.cfg_path, profile="bare"))
        self.assertEqual(ids["tts"], "v_torch")
        self.assertEqual(ids["separate"], "s_mlx")  # 'bare' doesn't touch separate

    def test_blank_env_var_is_treated_as_unset(self):
        os.environ["PATMIR_PROFILE"] = "   "
        self.assertIsNone(active_profile())
        self.assertEqual(_ids(load_pipeline_config(self.cfg_path))["tts"], "v_mlx")


class RepoCudaProfileTest(unittest.TestCase):
    """Guards the real patmir.toml: the `cuda` profile must not leave an
    Apple-Silicon-only candidate selected in either execution mode."""

    def setUp(self):
        if not _REPO_CONFIG.exists():  # pragma: no cover - source checkout only
            self.skipTest(f"{_REPO_CONFIG} not present")
        self._saved = os.environ.pop("PATMIR_PROFILE", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["PATMIR_PROFILE"] = self._saved

    def _assert_no_apple_only(self, cfg, mode: str):
        for stage, entry in cfg.entries.items():
            module = str(entry.get("module", ""))
            self.assertNotIn(
                "mlx", module.lower(),
                f"{mode}: stage {stage!r} selects MLX module {module!r} under the cuda profile",
            )
            for key in ("device", "codec_device"):
                self.assertNotEqual(
                    str(entry.get(key, "")), "mps",
                    f"{mode}: stage {stage!r} pins {key}=mps under the cuda profile",
                )

    def test_batch_mode_is_cuda_clean(self):
        self._assert_no_apple_only(
            load_pipeline_config(_REPO_CONFIG, profile="cuda"), "batch")

    def test_watch_along_mode_is_cuda_clean(self):
        """streaming_config()'s hardcoded overrides name MLX candidates; the
        profile's streaming layer has to displace them."""
        cfg = load_pipeline_config(
            _REPO_CONFIG, profile="cuda", layer="streaming",
            overrides={"separate": "htdemucs_mlx_fast",
                       "translate": "openrouter_gemini_flash_batch"},
        )
        self._assert_no_apple_only(cfg, "watch-along")

    def test_default_profile_still_selects_the_mlx_set(self):
        """The Mac path must be untouched by the profile mechanism."""
        ids = _ids(load_pipeline_config(_REPO_CONFIG))
        self.assertEqual(ids["tts"], "higgs_v3_mlx_q8")
        self.assertEqual(ids["asr"], "qwen3_asr_1_7b_mlx_8bit")
        self.assertEqual(ids["separate"], "htdemucs_mlx_vocals")
        self.assertEqual(ids["diarize"], "sortformer_mlx")


class RuntimeParityTest(unittest.TestCase):
    """The MLX and torch Higgs candidates are two runtimes for ONE model, so the
    knobs that shape delivery must not drift apart between them."""

    def test_emotion_tag_maps_are_identical(self):
        from patmir.eval.candidates.tts import higgs_v3_mlx, higgs_v3_torch

        self.assertEqual(higgs_v3_torch.EMOTION_TAGS, higgs_v3_mlx.EMOTION_TAGS)

    def test_both_expose_the_same_adapter_contract(self):
        """`SentenceTTS` probes these by signature/attribute, so a missing one
        degrades silently (a ref-less voice, or a lost overflow count) rather
        than raising."""
        import inspect

        from patmir.eval.candidates.tts import higgs_v3_mlx, higgs_v3_torch

        mlx_cls, torch_cls = higgs_v3_mlx.HiggsV3MLX, higgs_v3_torch.HiggsV3Torch
        for attr in ("ref_text_optional", "tts_concurrency"):
            self.assertEqual(getattr(torch_cls, attr), getattr(mlx_cls, attr), attr)
        for method in ("synthesize", "summarize", "reset_summary"):
            self.assertTrue(callable(getattr(torch_cls, method, None)), method)
        self.assertEqual(
            list(inspect.signature(torch_cls.synthesize).parameters),
            list(inspect.signature(mlx_cls.synthesize).parameters),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
