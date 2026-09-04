"""GPU + host telemetry sampled for the duration of one benchmark repeat.

Why this exists: until 2026-09-02 `patmir bench` recorded no power, clock,
temperature or throttle state at all, so **no stored number could be told apart
from a throttled one**. The 3090 in this box had in fact been running under a
reduced power limit (the 80% / 280 W mitigation in
`docs/setup/windows-wsl2-cuda.md`) while the GDDR6X cooling was bad, and that
limit was never recorded anywhere in the repo — which retroactively invalidated
every CUDA figure in `performance.md`. The driver's since-boot counters showed
337 s of SW power-cap and 203 s of SW thermal slowdown, so the throttling was
real, not hypothetical.

A benchmark that cannot say "the card was at its ceiling and never throttled"
is not reproducible, so this is now part of every run.

Two backends, in preference order:

  * **NVML** (`nvidia-ml-py`, declared in the `cuda` extra) — in-process, ~50 us
    per sample. Works under WSL2, where it loads `/usr/lib/wsl/lib/libnvidia-ml.so.1`.
  * **`nvidia-smi`** subprocess — the fallback when NVML will not import. One
    spawn per sample, which costs real CPU and pollutes `notes["cpu_s"]`, so
    the summary names the backend it used.

Never raises. A four-minute benchmark must not die on an instrumentation call,
and a missing GPU (the Mac, CI) simply yields `None`.

`GpuIdle` is deliberately NOT reported as a throttle reason. It means "no work
queued", which during a pipeline run says something about the pipeline (serial
stages, host-bound sections) and nothing about the card being limited. It is
published separately as `idle_pct`.
"""
from __future__ import annotations

import subprocess
import threading
import time
from typing import Any

# Bit -> name for NVML's clocks-event (formerly clocks-throttle) reason mask.
# Only the reasons that actually reduce clocks; GpuIdle is handled separately.
_THROTTLE_BITS: list[tuple[int, str]] = [
    (0x004, "sw_power_cap"),
    (0x008, "hw_slowdown"),
    (0x010, "sync_boost"),
    (0x020, "sw_thermal_slowdown"),
    (0x040, "hw_thermal_slowdown"),
    (0x080, "hw_power_brake_slowdown"),
    (0x002, "applications_clocks_setting"),
    (0x100, "display_clock_setting"),
]
_IDLE_BIT = 0x001

_SMI_FIELDS = (
    "power.draw", "clocks.sm", "clocks.mem", "temperature.gpu",
    "utilization.gpu", "utilization.memory", "memory.used",
)


def _host_rss_gb() -> float | None:
    """Current process RSS. `resource.getrusage` only gives the high-water mark
    for the whole process lifetime, which across `--repeat` cannot say which
    repeat drove host commit — and host commit, not VRAM, is what produced the
    documented 1.71 -> 3.02 watch-along regression."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return None


class _Nvml:
    """In-process NVML reader. `ok` is False when the bindings or device are
    unavailable, and the caller falls back."""

    name = "nvml"

    def __init__(self, index: int = 0) -> None:
        self.ok = False
        self._n = None
        self._h = None
        self.power_limit_w: float | None = None
        self.power_limit_default_w: float | None = None
        self.device_name: str | None = None
        try:
            import pynvml  # type: ignore[import-not-found]
            pynvml.nvmlInit()
            self._n = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(index)
            dn = pynvml.nvmlDeviceGetName(self._h)
            self.device_name = dn.decode() if isinstance(dn, bytes) else str(dn)
            try:
                self.power_limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(self._h) / 1000.0
            except Exception:  # noqa: BLE001
                pass
            try:
                # The board's own design ceiling. Needed to separate "this card
                # is doing all it can" from "someone capped it below stock" —
                # the second is what silently invalidated a month of numbers.
                self.power_limit_default_w = (
                    pynvml.nvmlDeviceGetPowerManagementDefaultLimit(self._h) / 1000.0)
            except Exception:  # noqa: BLE001
                pass
            self.ok = True
        except Exception:  # noqa: BLE001
            self.ok = False

    def sample(self) -> dict[str, Any] | None:
        n, h = self._n, self._h
        if n is None or h is None:
            return None
        out: dict[str, Any] = {}
        try:
            out["power_w"] = n.nvmlDeviceGetPowerUsage(h) / 1000.0
        except Exception:  # noqa: BLE001
            pass
        try:
            out["sm_mhz"] = n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM)
            out["mem_mhz"] = n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_MEM)
        except Exception:  # noqa: BLE001
            pass
        try:
            out["temp_c"] = n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU)
        except Exception:  # noqa: BLE001
            pass
        try:
            u = n.nvmlDeviceGetUtilizationRates(h)
            out["util_pct"] = u.gpu
            out["mem_util_pct"] = u.memory
        except Exception:  # noqa: BLE001
            pass
        try:
            out["mem_used_mb"] = n.nvmlDeviceGetMemoryInfo(h).used / 1024 ** 2
        except Exception:  # noqa: BLE001
            pass
        try:
            getter = getattr(n, "nvmlDeviceGetCurrentClocksThrottleReasons", None) \
                or getattr(n, "nvmlDeviceGetCurrentClocksEventReasons", None)
            if getter is not None:
                out["reasons"] = int(getter(h))
        except Exception:  # noqa: BLE001
            pass
        return out or None

    def close(self) -> None:
        try:
            if self._n is not None:
                self._n.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


class _Smi:
    """`nvidia-smi` fallback. One subprocess per sample — noted in the summary
    because it charges the benchmark's own CPU accounting."""

    name = "nvidia-smi"

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.power_limit_w: float | None = None
        self.power_limit_default_w: float | None = None
        self.device_name: str | None = None
        # Driver 600+ renamed the field; try the new name first and remember
        # which one answered so the hot path does not re-probe.
        self._reason_field: str | None = None
        for field in ("clocks_event_reasons.active", "clocks_throttle_reasons.active"):
            if self._query([field]) is not None:
                self._reason_field = field
                break
        head = self._query(["name", "power.limit", "power.default_limit"])
        self.ok = head is not None
        if head:
            self.device_name = head[0]
            try:
                self.power_limit_w = float(head[1])
                self.power_limit_default_w = float(head[2])
            except (ValueError, IndexError):
                pass

    def _query(self, fields: list[str]) -> list[str] | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={','.join(fields)}",
                 "--format=csv,noheader,nounits", "-i", str(self.index)],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode != 0 or not out.stdout.strip():
                return None
            return [c.strip() for c in out.stdout.strip().splitlines()[0].split(",")]
        except Exception:  # noqa: BLE001
            return None

    def sample(self) -> dict[str, Any] | None:
        fields = list(_SMI_FIELDS) + ([self._reason_field] if self._reason_field else [])
        vals = self._query(fields)
        if vals is None:
            return None
        out: dict[str, Any] = {}
        keys = ["power_w", "sm_mhz", "mem_mhz", "temp_c",
                "util_pct", "mem_util_pct", "mem_used_mb"]
        for k, v in zip(keys, vals):
            try:
                out[k] = float(v)
            except ValueError:
                pass
        if self._reason_field and len(vals) > len(keys):
            try:
                out["reasons"] = int(vals[len(keys)], 16)
            except ValueError:
                pass
        return out or None

    def close(self) -> None:
        return None


def _reader(index: int = 0):
    nv = _Nvml(index)
    if nv.ok:
        return nv
    nv.close()
    smi = _Smi(index)
    return smi if smi.ok else None


class GpuSampler:
    """Poll GPU + host telemetry on a daemon thread for one repeat.

    Usage:

        s = GpuSampler(); s.start()
        ...one bench repeat...
        notes["gpu"] = s.stop()

    `stop()` returns None when no GPU reader is available, so the caller can
    omit the key entirely rather than publish an empty shell that reads like a
    measurement.
    """

    def __init__(self, interval_s: float = 1.0, device_index: int = 0) -> None:
        self.interval_s = interval_s
        self.device_index = device_index
        self._samples: list[dict[str, Any]] = []
        self._rss: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reader = None

    # -- lifecycle -- #
    def start(self) -> None:
        self._reader = _reader(self.device_index)
        self._samples.clear()
        self._rss.clear()
        self._stop.clear()
        # RSS needs no GPU, so sample it even with no reader — that is the
        # instrument the swap regression actually needed.
        self._thread = threading.Thread(target=self._loop, name="patmir-telemetry",
                                        daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._reader is not None:
                try:
                    s = self._reader.sample()
                except Exception:  # noqa: BLE001
                    s = None
                if s:
                    self._samples.append(s)
            r = _host_rss_gb()
            if r is not None:
                self._rss.append(r)
            self._stop.wait(self.interval_s)

    def stop(self) -> dict[str, Any] | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 3)
            self._thread = None
        summary = self._summarize()
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        return summary

    # -- summary -- #
    @property
    def peak_rss_gb(self) -> float | None:
        return round(max(self._rss), 2) if self._rss else None

    def _summarize(self) -> dict[str, Any] | None:
        if self._reader is None or not self._samples:
            return None
        rows = self._samples

        def agg(key: str) -> dict[str, float] | None:
            vals = [float(r[key]) for r in rows if r.get(key) is not None]
            if not vals:
                return None
            return {
                "avg": round(sum(vals) / len(vals), 1),
                "min": round(min(vals), 1),
                "max": round(max(vals), 1),
            }

        masks = [int(r["reasons"]) for r in rows if r.get("reasons") is not None]
        reasons: dict[str, int] = {}
        for bit, name in _THROTTLE_BITS:
            hits = sum(1 for m in masks if m & bit)
            if hits:
                reasons[name] = hits
        idle = sum(1 for m in masks if m & _IDLE_BIT)

        limit = self._reader.power_limit_w
        stock = self._reader.power_limit_default_w
        # Two very different conditions share one NVML bit, and conflating them
        # is how a benchmark lies to you.
        #
        #   * `sw_power_cap` while the enforced limit IS the board's stock limit
        #     means the card spent part of the run at its own design ceiling.
        #     That is what a 350 W 3090 is supposed to do under a bandwidth-bound
        #     decode, and it is not a defect or a confound.
        #   * The enforced limit sitting BELOW stock means someone capped the
        #     card — the 280 W thermal mitigation this box ran under while every
        #     CUDA number in performance.md was measured. That makes a run
        #     incomparable with one taken at stock, and it must be loud.
        #
        # Thermal slowdown is always a finding, at any limit.
        capped_below_stock = bool(limit and stock and limit < stock - 1.0)
        thermal = any(k.endswith("thermal_slowdown") for k in reasons)
        out: dict[str, Any] = {
            "source": self._reader.name,
            "device": self._reader.device_name,
            "power_limit_w": limit,
            "power_limit_default_w": stock,
            "power_limit_is_stock": (None if not (limit and stock)
                                     else not capped_below_stock),
            "samples": len(rows),
            "interval_s": self.interval_s,
            "throttle_reasons": reasons,
            # Reserved for conditions that make this run INCOMPARABLE with a
            # clean one. Reaching a stock ceiling is not one of them.
            "throttled": bool(capped_below_stock or thermal),
            "at_power_ceiling_samples": reasons.get("sw_power_cap", 0),
            "thermal_slowdown": thermal,
            "capped_below_stock": capped_below_stock,
        }
        if masks:
            out["idle_pct"] = round(100 * idle / len(masks), 1)
        for key in ("power_w", "sm_mhz", "mem_mhz", "temp_c",
                    "util_pct", "mem_util_pct", "mem_used_mb"):
            a = agg(key)
            if a is not None:
                out[key] = a
        return out


def gpu_static_info(device_index: int = 0) -> dict[str, Any] | None:
    """One-shot device identity + power ceiling, for a report header."""
    rd = _reader(device_index)
    if rd is None:
        return None
    try:
        return {"device": rd.device_name, "power_limit_w": rd.power_limit_w,
                "source": rd.name}
    finally:
        rd.close()
