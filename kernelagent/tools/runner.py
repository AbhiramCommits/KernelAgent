"""Benchmark runner: validates the binary's JSON contract and noise."""

from __future__ import annotations

import json
from pathlib import Path

from .. import config
from ..types import RunResult, Shape
from .base import run_subprocess

_REQUIRED_KEYS = ("ms_median", "ms_mad", "ms_min", "reps")


class RunnerError(RuntimeError):
    """Raised when a benchmark binary breaks the output contract."""

    def __init__(self, message: str, stderr: str = "") -> None:
        super().__init__(message)
        self.stderr = stderr


def _parse_json_line(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and all(key in payload for key in _REQUIRED_KEYS):
            return payload
    raise RunnerError("benchmark binary did not emit a valid JSON timing line")


def _to_result(payload: dict[str, object], flops: float | None) -> RunResult:
    ms_median = float(payload["ms_median"])  # type: ignore[arg-type]
    if not (ms_median > 0.0) or ms_median != ms_median:
        raise RunnerError(f"invalid median time: {ms_median}")
    reps = int(str(payload["reps"]))
    if reps < 1:
        raise RunnerError(f"invalid rep count: {reps}")
    throughput = 0.0
    if flops and flops > 0:
        throughput = flops / (ms_median / 1000.0) / 1e9
    return RunResult(
        ms_median=ms_median,
        ms_mad=float(payload.get("ms_mad", 0.0)),  # type: ignore[arg-type]
        ms_min=float(payload.get("ms_min", ms_median)),  # type: ignore[arg-type]
        reps=reps,
        throughput=throughput,
        checksum=float(payload.get("checksum", 0.0)),  # type: ignore[arg-type]
    )


class BenchRunner:
    """Runs a compiled kernel binary and normalises its timing report."""

    def __init__(self, timeout_s: float = config.RUN_TIMEOUT_S) -> None:
        self.timeout_s = timeout_s

    def run(
        self,
        binary: str | Path,
        shape: Shape,
        reps: int = config.DEFAULT_REPS,
        warmup: int = config.DEFAULT_WARMUP,
        flops: float | None = None,
        allow_rerun: bool = True,
    ) -> RunResult:
        """Time ``binary`` on ``shape``; re-run once when the noise is high."""
        binary_path = Path(binary)
        if not binary_path.exists():
            raise RunnerError(f"binary does not exist: {binary_path}")

        def invoke(run_reps: int) -> RunResult:
            result = run_subprocess(
                [
                    str(binary_path),
                    "--shape",
                    shape.key(),
                    "--reps",
                    str(run_reps),
                    "--warmup",
                    str(warmup),
                ],
                timeout_s=self.timeout_s,
                retries=0,
            )
            if not result.ok:
                message = result.error or f"exit {result.returncode}"
                if result.timed_out:
                    message = f"benchmark timed out after {self.timeout_s:.0f}s"
                raise RunnerError(message, stderr=result.stderr)
            try:
                payload = _parse_json_line(result.stdout)
            except RunnerError as exc:
                exc.stderr = result.stderr
                raise
            return _to_result(payload, flops)

        first = invoke(reps)
        if not allow_rerun or first.noise_ratio <= config.NOISE_RERUN_RATIO:
            return first
        second = invoke(min(config.MAX_REPS, max(reps, first.reps) * 2))
        if second.noise_ratio < first.noise_ratio:
            return second
        if second.ms_median < first.ms_median:
            return second
        return first

    def run_with_dump(
        self,
        binary: str | Path,
        shape: Shape,
        dump_path: Path,
        input_dir: Path | None = None,
        timeout_s: float | None = None,
    ) -> None:
        """Run the kernel once, dumping its output for the correctness checker."""
        cmd = [
            str(binary),
            "--shape",
            shape.key(),
            "--reps",
            "1",
            "--warmup",
            "0",
            "--dump",
            str(dump_path),
        ]
        if input_dir is not None:
            cmd += ["--input-dir", str(input_dir)]
        result = run_subprocess(cmd, timeout_s=timeout_s or config.CHECK_TIMEOUT_S, retries=0)
        if not result.ok:
            raise RunnerError(result.error or "dump run failed", stderr=result.stderr)
