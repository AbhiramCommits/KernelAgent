"""Profilers.

``DerivedProfiler`` is the CPU path: it combines the task's FLOP/byte model
with a measured STREAM-style bandwidth ceiling. ``NcuProfiler`` runs Nsight
Compute and is only used when ``ncu`` and a CUDA device are actually present.
Every :class:`ProfileResult` records which source produced it.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from ..env import probe_environment
from ..types import KernelTask, ProfileResult, RunResult, Shape
from ..workload import bytes_for, flops_for
from .base import find_executable, run_subprocess

_BW_CACHE: dict[int, float] = {}

NCU_METRICS = [
    "gpu__time_duration.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
]


def measure_bandwidth_ceiling(nbytes: int, reps: int = 3) -> float:
    """Measure an achievable copy bandwidth (GB/s) for a working set.

    A copy streams one read and one write, so ``2 * nbytes`` crosses the memory
    system per repetition. Results are cached per size in-process and the fastest
    repetition is used, which is the least noisy estimate of the ceiling.
    """
    size = max(int(nbytes), 1 << 20)
    size -= size % 4
    if size in _BW_CACHE:
        return _BW_CACHE[size]
    count = max(size // 4, 1)
    src = np.ones(count, dtype=np.float32)
    dst = np.empty_like(src)
    dst[:] = src  # warm the pages
    best = float("inf")
    for _ in range(reps):
        started = time.perf_counter()
        dst[:] = src
        elapsed = time.perf_counter() - started
        best = min(best, elapsed)
    bandwidth = (2.0 * count * 4.0) / max(best, 1e-9) / 1e9
    _BW_CACHE[size] = bandwidth
    return bandwidth


class DerivedProfiler:
    """Analytic counters backed by one real bandwidth measurement."""

    def __init__(self, task: KernelTask, shape: Shape) -> None:
        self.task = task
        self.shape = shape
        self.flops = flops_for(task, shape)
        self.bytes = bytes_for(task, shape)

    def profile(self, run: RunResult) -> ProfileResult:
        """Compute achieved throughput and the fraction of measured bandwidth."""
        seconds = run.ms_median / 1000.0
        gflops = self.flops / max(seconds, 1e-12) / 1e9
        gbps = self.bytes / max(seconds, 1e-12) / 1e9
        ceiling = measure_bandwidth_ceiling(int(self.bytes))
        env = probe_environment()
        caches = env.get("cache_bytes", {}) or {}
        largest_cache = max(caches.values()) if caches else 0
        counters = {
            "flops": self.flops,
            "bytes": self.bytes,
            "gflops": gflops,
            "gbps": gbps,
            "arithmetic_intensity": self.flops / max(self.bytes, 1.0),
            "working_set_bytes": self.bytes,
            "bandwidth_ceiling_gbps": ceiling,
            "bandwidth_pct": 100.0 * gbps / max(ceiling, 1e-9),
            "largest_cache_bytes": float(largest_cache),
            "working_set_over_cache": self.bytes / float(largest_cache) if largest_cache else 0.0,
            "kernel_ms": run.ms_median,
        }
        return ProfileResult(counters=counters, source="derived")


class NcuProfiler:
    """Nsight Compute counter collection, gated on ``ncu`` being installed."""

    def __init__(self, timeout_s: float = 120.0) -> None:
        self.timeout_s = timeout_s
        self.executable = find_executable("ncu")

    @staticmethod
    def is_available() -> bool:
        """True only when ``ncu`` exists and a CUDA device is visible."""
        if find_executable("ncu") is None:
            return False
        env = probe_environment()
        return bool(env.get("has_torch") or env.get("has_nvcc"))

    def profile(self, binary: str | Path, shape: Shape) -> ProfileResult:
        """Run one kernel under ncu and parse the CSV counters."""
        if not self.executable:
            raise RuntimeError("ncu is not available on this machine")
        from .diagnostics import parse_ncu_csv

        metrics = ",".join(NCU_METRICS)
        with tempfile.TemporaryDirectory(prefix="ka-ncu-") as tmp:
            csv_path = Path(tmp) / "ncu.csv"
            cmd = [
                self.executable,
                "--csv",
                "--target-processes",
                "all",
                "--metrics",
                metrics,
                "--launch-count",
                "1",
                str(binary),
                "--shape",
                shape.key(),
                "--reps",
                "1",
                "--warmup",
                "0",
            ]
            result = run_subprocess(cmd, timeout_s=self.timeout_s, retries=0)
            if not result.ok:
                return ProfileResult(
                    counters={"error": 1.0},
                    source="ncu",
                    notes=f"ncu failed: {result.error}",
                )
            csv_path.write_text(result.stdout, encoding="utf-8")
            parsed = parse_ncu_csv(csv_path)
        counters = dict(parsed.get("counters", {}))
        for reason, value in parsed.get("warp_stalls", {}).items():
            counters[f"warp_stall_{reason}"] = float(value)
        return ProfileResult(
            counters=counters,
            source="ncu",
            notes=f"parsed {parsed.get('rows', 0)} metric rows",
        )


def profile_binary(
    task: KernelTask,
    shape: Shape,
    binary: str | Path,
    run: RunResult,
    prefer_ncu: bool = True,
) -> ProfileResult:
    """Use ncu when it is genuinely available, otherwise the derived profiler."""
    if prefer_ncu and NcuProfiler.is_available():
        result = NcuProfiler().profile(binary, shape)
        if result.counters and "error" not in result.counters:
            return result
    return DerivedProfiler(task, shape).profile(run)
