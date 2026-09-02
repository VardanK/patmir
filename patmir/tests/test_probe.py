"""Unit tests for the intra-stage probe (`patmir/pipeline/probe.py`).

The contract these pin, in order of how much a violation would cost:

1. **Disabled is free and silent.** Candidates call `section()` on hot paths
   (once per decode step, once per sentence), so a disabled probe must record
   nothing, write nothing, and never touch torch.
2. **`dump()` merges.** A stage can have two instruments aimed at it — the
   candidate's own probe (TTS owns one because its decode loop never sees a
   `JobContext`) and the job probe the runner dumps per stage. Overwriting
   would mean whichever wrote last silently erased the other.
3. **Step spans are opt-in.** They need a device sync per step, which inflates
   the total they exist to explain, so `chunk` level must not record them.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from patmir.pipeline.probe import Probe, probe_from_ctx


class _Ctx:
    def __init__(self, extra, job_dir="."):
        self.extra = extra
        self.job_dir = Path(job_dir)


class DisabledProbeIsFreeTest(unittest.TestCase):
    def test_sections_record_nothing(self):
        p = Probe()
        with p.section("model_s"):
            pass
        with p.step_section("forward_s"):
            pass
        p.bump("steps", 3)
        p.set("engine", "x")
        p.detail({"B": 8})
        self.assertEqual(p.summary(), {})

    def test_dump_writes_no_file(self):
        p = Probe()
        with TemporaryDirectory() as td:
            out = Path(td) / "tts" / "bench_probe.json"
            p.dump(out)
            self.assertFalse(out.exists())

    def test_section_returns_a_reusable_noop(self):
        # One shared nullcontext is handed out; using it twice, and nested,
        # must be safe or a disabled probe would break the pipeline.
        p = Probe()
        with p.section("a"), p.section("b"):
            with p.section("a"):
                pass


class EnabledProbeRecordsTest(unittest.TestCase):
    def test_spans_accumulate_across_calls(self):
        p = Probe()
        p.start()
        for _ in range(3):
            with p.section("decode_s"):
                pass
        self.assertIn("decode_s", p.spans)
        self.assertGreaterEqual(p.spans["decode_s"], 0.0)

    def test_counters_and_values_and_details(self):
        p = Probe()
        p.start()
        p.bump("steps", 5)
        p.bump("steps")
        p.set("engine", "higgs")
        p.detail({"B": 8})
        s = p.summary()
        self.assertEqual(s["steps"], 6)
        self.assertEqual(s["engine"], "higgs")
        self.assertEqual(s["details"], [{"B": 8}])

    def test_step_spans_need_step_level(self):
        p = Probe()
        p.start()                      # default level="chunk"
        with p.step_section("forward_s"):
            pass
        self.assertNotIn("forward_s", p.spans)
        p.start(level="step")
        with p.step_section("forward_s"):
            pass
        self.assertIn("forward_s", p.spans)

    def test_start_resets_prior_state(self):
        p = Probe()
        p.start()
        p.bump("steps", 9)
        p.start()
        self.assertEqual(p.summary(), {})


class DumpMergesTest(unittest.TestCase):
    def test_second_writer_does_not_erase_the_first(self):
        with TemporaryDirectory() as td:
            out = Path(td) / "asr" / "bench_probe.json"
            candidate = Probe()
            candidate.start()
            candidate.set("generate_s", 10.5)
            candidate.dump(out)

            runner = Probe()
            runner.start()
            runner.set("tag_s", 1.25)
            runner.dump(out)

            data = json.loads(out.read_text())
            self.assertEqual(data["generate_s"], 10.5)
            self.assertEqual(data["tag_s"], 1.25)

    def test_corrupt_prior_file_is_overwritten_not_fatal(self):
        with TemporaryDirectory() as td:
            out = Path(td) / "bench_probe.json"
            out.write_text("{not json")
            p = Probe()
            p.start()
            p.set("x", 1)
            p.dump(out)
            self.assertEqual(json.loads(out.read_text()), {"x": 1})

    def test_empty_summary_leaves_a_prior_file_alone(self):
        with TemporaryDirectory() as td:
            out = Path(td) / "bench_probe.json"
            p = Probe()
            p.start()
            p.set("keep", 1)
            p.dump(out)
            Probe.start(other := Probe())   # armed but recorded nothing
            other.dump(out)
            self.assertEqual(json.loads(out.read_text()), {"keep": 1})


class ArmFromContextTest(unittest.TestCase):
    def test_no_bench_flag_leaves_it_disabled(self):
        p = Probe()
        p.start()
        probe_from_ctx(_Ctx({}), p, stage="tts")
        self.assertFalse(p.enabled)

    def test_bench_flag_arms_at_chunk_level(self):
        p = Probe()
        probe_from_ctx(_Ctx({"bench": True}), p, stage="tts")
        self.assertTrue(p.enabled)
        self.assertEqual(p.level, "chunk")
        self.assertIsNone(p.trace_path)

    def test_per_stage_level_and_trace_keys(self):
        p = Probe()
        probe_from_ctx(
            _Ctx({"bench": True, "tts_probe": "step", "tts_trace": True}, job_dir="/j"),
            p, stage="tts")
        self.assertEqual(p.level, "step")
        self.assertEqual(p.trace_path, Path("/j/tts/trace_tts.json"))

    def test_another_stages_keys_are_ignored(self):
        p = Probe()
        probe_from_ctx(_Ctx({"bench": True, "asr_probe": "step"}), p, stage="tts")
        self.assertEqual(p.level, "chunk")


if __name__ == "__main__":
    unittest.main()
