"""Unit tests for GPU/host telemetry (`patmir/telemetry.py`).

Why this module exists at all is the thing worth remembering: every CUDA figure
in `performance.md` before 2026-09-02 was measured on a card that had been
power-limited to 280 W for thermal reasons, and no report recorded it — so a
throttled run and a clean one were indistinguishable after the fact. The
assertions below are therefore mostly about the summary being *honest*:
`throttled` must be true when the driver said so, `GpuIdle` must not be counted
as throttling, and a machine with no GPU must yield None rather than an empty
shell that reads like a measurement.

No GPU is required: the reader is injected.
"""
from __future__ import annotations

import unittest
from unittest import mock

from patmir import telemetry
from patmir.telemetry import GpuSampler


class _FakeReader:
    name = "fake"
    device_name = "Fake 3090"
    power_limit_w = 350.0
    power_limit_default_w = 350.0

    def __init__(self, rows):
        self._rows = list(rows)
        self.closed = False

    def sample(self):
        return self._rows.pop(0) if self._rows else None

    def close(self):
        self.closed = True


def _run(rows, interval_s=0.01):
    """Drive the sampler with a fixed row list and return its summary."""
    reader = _FakeReader(rows)
    s = GpuSampler(interval_s=interval_s)
    with mock.patch.object(telemetry, "_reader", return_value=reader):
        s.start()
        # Drain deterministically instead of sleeping: pull every row on this
        # thread, then stop. The loop's own polling is timing-dependent and is
        # not what these tests are about.
        s._stop.set()
        if s._thread is not None:
            s._thread.join(timeout=2)
        s._samples = [r for r in rows]
        return s.stop(), reader


class SummaryShapeTest(unittest.TestCase):
    def test_aggregates_power_clocks_temp_util(self):
        out, _ = _run([
            {"power_w": 100.0, "sm_mhz": 1500, "temp_c": 60, "util_pct": 90,
             "mem_used_mb": 16000, "reasons": 0},
            {"power_w": 300.0, "sm_mhz": 1900, "temp_c": 70, "util_pct": 98,
             "mem_used_mb": 17000, "reasons": 0},
        ])
        self.assertEqual(out["power_w"], {"avg": 200.0, "min": 100.0, "max": 300.0})
        self.assertEqual(out["sm_mhz"]["max"], 1900.0)
        self.assertEqual(out["temp_c"]["max"], 70.0)
        self.assertEqual(out["util_pct"]["avg"], 94.0)
        self.assertEqual(out["mem_used_mb"]["max"], 17000.0)
        self.assertEqual(out["samples"], 2)
        self.assertEqual(out["source"], "fake")
        self.assertEqual(out["power_limit_w"], 350.0)

    def test_missing_fields_are_omitted_not_zeroed(self):
        out, _ = _run([{"power_w": 50.0}])
        self.assertIn("power_w", out)
        self.assertNotIn("temp_c", out)
        self.assertNotIn("sm_mhz", out)


class ThrottleReportingTest(unittest.TestCase):
    def test_clean_run_reports_no_throttling(self):
        out, _ = _run([{"power_w": 300.0, "reasons": 0}] * 3)
        self.assertEqual(out["throttle_reasons"], {})
        self.assertFalse(out["throttled"])
        self.assertEqual(out["at_power_ceiling_samples"], 0)

    def test_gpu_idle_is_not_a_throttle(self):
        # 0x1 is GpuIdle: it means nothing was queued, which says something
        # about the pipeline and nothing about the card being limited.
        out, _ = _run([{"power_w": 30.0, "reasons": 0x1}] * 4)
        self.assertEqual(out["throttle_reasons"], {})
        self.assertFalse(out["throttled"])
        self.assertEqual(out["idle_pct"], 100.0)

    def test_power_cap_and_thermal_are_named_and_counted(self):
        out, _ = _run([
            {"reasons": 0x004},           # sw_power_cap
            {"reasons": 0x004 | 0x020},   # + sw_thermal_slowdown
            {"reasons": 0x000},
        ])
        self.assertEqual(out["throttle_reasons"]["sw_power_cap"], 2)
        self.assertEqual(out["throttle_reasons"]["sw_thermal_slowdown"], 1)
        # Thermal slowdown is a finding at any power limit.
        self.assertTrue(out["thermal_slowdown"])
        self.assertTrue(out["throttled"])

    def test_stock_ceiling_is_not_reported_as_throttled(self):
        # A 350 W 3090 under a bandwidth-bound decode is SUPPOSED to sit at its
        # ceiling. Calling that "throttled" would conflate a healthy card with
        # the 280 W thermal cap this box actually ran under, which is the
        # confound the whole module exists to expose.
        out, _ = _run([{"reasons": 0x004}] * 3)
        self.assertFalse(out["throttled"])
        self.assertFalse(out["capped_below_stock"])
        self.assertTrue(out["power_limit_is_stock"])
        self.assertEqual(out["at_power_ceiling_samples"], 3)

    def test_a_limit_below_stock_is_loud(self):
        class _Capped(_FakeReader):
            power_limit_w = 280.0          # the old thermal mitigation
            power_limit_default_w = 350.0

        reader = _Capped([{"reasons": 0}])
        s = GpuSampler(interval_s=0.01)
        with mock.patch.object(telemetry, "_reader", return_value=reader):
            s.start()
            s._stop.set()
            if s._thread is not None:
                s._thread.join(timeout=2)
            s._samples = [{"reasons": 0}]
            out = s.stop()
        self.assertTrue(out["capped_below_stock"])
        self.assertTrue(out["throttled"])
        self.assertFalse(out["power_limit_is_stock"])

    def test_hardware_slowdown_bits_are_distinguished(self):
        out, _ = _run([{"reasons": 0x008 | 0x040 | 0x080}])
        self.assertEqual(set(out["throttle_reasons"]),
                         {"hw_slowdown", "hw_thermal_slowdown",
                          "hw_power_brake_slowdown"})


class NoGpuTest(unittest.TestCase):
    def test_summary_is_none_without_a_reader(self):
        s = GpuSampler(interval_s=0.01)
        with mock.patch.object(telemetry, "_reader", return_value=None):
            s.start()
            out = s.stop()
        self.assertIsNone(out)

    def test_rss_is_still_sampled_without_a_gpu(self):
        # Host RSS needs no GPU, and host commit — not VRAM — is what produced
        # the documented 1.71 -> 3.02 watch-along regression.
        s = GpuSampler(interval_s=0.01)
        with mock.patch.object(telemetry, "_reader", return_value=None):
            s.start()
            s._rss.append(3.5)
            s.stop()
        self.assertEqual(s.peak_rss_gb, 3.5)

    def test_summary_is_none_when_the_reader_yielded_nothing(self):
        reader = _FakeReader([])
        s = GpuSampler(interval_s=0.01)
        with mock.patch.object(telemetry, "_reader", return_value=reader):
            s.start()
            out = s.stop()
        self.assertIsNone(out)


class LifecycleTest(unittest.TestCase):
    def test_reader_is_closed_on_stop(self):
        _out, reader = _run([{"power_w": 1.0}])
        self.assertTrue(reader.closed)

    def test_a_raising_reader_does_not_kill_the_run(self):
        class _Boom(_FakeReader):
            def sample(self):
                raise RuntimeError("nvml exploded")

        reader = _Boom([{}])
        s = GpuSampler(interval_s=0.01)
        with mock.patch.object(telemetry, "_reader", return_value=reader):
            s.start()
            out = s.stop()
        self.assertIsNone(out)      # no samples survived, but nothing raised


if __name__ == "__main__":
    unittest.main()
