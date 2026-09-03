"""`separate` must be reproducible, because everything below it inherits.

Quality issue #28: `demucs.apply.apply_model` defaults to `shifts=1`, which
draws `random.randint(0, 0.5 * samplerate)` from the *unseeded* global `random`
(`demucs/apply.py:245`), shifts the mix by that offset, separates, and shifts
back. With shifts=1 the loop body runs once and divides by 1, so it is not an
average — it is one prediction at an arbitrary offset.

Ten runs of `samples/sample_1.mkv` at one config produced eight distinct
`vocals.wav` files, and the two pairs that happened to match were exactly the
two pairs whose emotion labels agreed perfectly downstream. Everything below
separate inherited the jitter: VAD intervals, diarization turns (91-117 on one
file), which window became each speaker's reference clip — so a character's
cloned voice changed between runs — emotion labels, and hence the translator's
prompt, since the emotion rides in the user message.

These tests pin `shifts=0` in both candidates and in the config, at the three
places it can be lost: the constructor default, the `build()` config default,
and the call into demucs.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock


class ShiftsDefaultTest(unittest.TestCase):
    def test_torch_candidate_constructor_defaults_to_zero(self):
        import inspect

        from patmir.pipeline.stages.separate.htdemucs import HTDemucs
        sig = inspect.signature(HTDemucs.__init__)
        self.assertEqual(sig.parameters["shifts"].default, 0)

    def test_mlx_candidate_constructor_defaults_to_zero(self):
        # Kept in lockstep with the torch twin: the two machines must not
        # disagree about whether the pipeline is reproducible.
        import inspect

        from patmir.pipeline.stages.separate import htdemucs_mlx
        sig = inspect.signature(htdemucs_mlx.HTDemucsMLX.__init__)
        self.assertEqual(sig.parameters["shifts"].default, 0)

    def test_torch_build_defaults_to_zero_when_config_is_silent(self):
        from patmir.pipeline.stages.separate import htdemucs

        with mock.patch.object(htdemucs, "HTDemucs") as ctor:
            htdemucs.build({"model": "04573f0d"})
        self.assertEqual(ctor.call_args.kwargs["shifts"], 0)

    def test_config_can_still_ask_for_averaging(self):
        # Not forbidden, just not the default: shifts >= 2 is a real average
        # and a legitimate quality/throughput trade. It reintroduces the
        # randomness, which is why it must be asked for explicitly.
        from patmir.pipeline.stages.separate import htdemucs

        with mock.patch.object(htdemucs, "HTDemucs") as ctor:
            htdemucs.build({"model": "04573f0d", "shifts": 4})
        self.assertEqual(ctor.call_args.kwargs["shifts"], 4)


class ApplyModelCallTest(unittest.TestCase):
    """The default must survive the call into demucs, not just the constructor.

    This is the assertion that actually protects reproducibility: passing no
    `shifts=` to `apply_model` silently gets demucs' default of 1, which is how
    the bug existed in the first place.
    """

    def _run_separate(self, tmp: Path, shifts: int):
        import numpy as np
        import soundfile as sf

        from patmir.pipeline.stages.separate import htdemucs

        sr = 44100
        sf.write(str(tmp / "src.wav"),
                 np.zeros((sr, 2), dtype="float32"), sr, subtype="PCM_16")

        cand = object.__new__(htdemucs.HTDemucs)
        cand.model = object()
        cand.model_name = "04573f0d"
        cand.device = "cpu"
        cand.vocals_only = True
        cand.shifts = shifts
        cand.stem_names = ["vocals"]
        cand.sample_rate = sr
        cand.channels = 2

        import torch
        captured = {}

        def fake_apply(model, wav, **kwargs):
            captured.update(kwargs)
            return torch.zeros(1, 1, 2, wav.shape[-1])

        ctx = mock.Mock()
        ctx.job_dir = tmp
        ctx.options.keep_intermediate = True
        ctx.probe.section = mock.MagicMock()
        ctx.probe.bump = mock.MagicMock()

        with mock.patch("demucs.apply.apply_model", fake_apply), \
             mock.patch.object(htdemucs, "_split_vocals_with_vad",
                               return_value=(tmp / "s.wav", tmp / "n.wav", [])):
            cand.separate(tmp / "src.wav", ctx)
        return captured

    def test_shifts_is_passed_explicitly(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            kw = self._run_separate(Path(d), 0)
        self.assertIn("shifts", kw,
                      "apply_model called without shifts= inherits demucs' "
                      "random-shift default of 1 (quality issue #28)")
        self.assertEqual(kw["shifts"], 0)

    def test_a_configured_value_reaches_demucs(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            kw = self._run_separate(Path(d), 3)
        self.assertEqual(kw["shifts"], 3)


class ConfiguredSeparatorsTest(unittest.TestCase):
    def test_every_registered_separate_candidate_pins_shifts_to_zero(self):
        """The value lives in `patmir.toml` as well as in code.

        Two reasons: `config_hash` hashes the candidate entry, so the choice is
        recorded in the cache key rather than being an invisible code default
        that one machine could have and another not; and a reader of the config
        can see that reproducibility was chosen, not defaulted into.
        """
        import tomllib

        cfg = tomllib.loads(Path("patmir.toml").read_text())
        cands = cfg["stages"]["separate"]["candidates"]
        self.assertTrue(cands)
        for c in cands:
            module = c.get("module", "")
            if "htdemucs" not in module:
                continue
            self.assertEqual(
                c.get("shifts", 0), 0,
                f"separate candidate {c['id']!r} would reintroduce demucs' "
                f"random shift (quality issue #28)",
            )


if __name__ == "__main__":
    unittest.main()
