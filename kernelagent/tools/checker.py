"""Correctness gate: compare a kernel's dumped output to the NumPy reference."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from .. import config
from ..types import CheckResult, KernelTask, Shape
from .base import run_subprocess
from .runner import BenchRunner, RunnerError


class CorrectnessChecker:
    """Runs the compiled kernel and compares against an independent reference.

    Inputs are dumped once per (task, shape) under ``.kacache/inputs`` and reused
    by every variant, so all variants are compared on byte-identical data and
    large inputs are not rewritten for every trial.
    """

    def __init__(self, task: KernelTask, cache_root: Path | None = None) -> None:
        self.task = task
        self.cache_root = Path(cache_root) if cache_root else config.CACHE_DIR / "inputs"
        self.runner = BenchRunner()

    def _source_stamp(self) -> str:
        task_dir = Path(self.task.directory)
        digest = hashlib.sha256()
        for name in ("meta.yaml", self.task.reference_module, self.task.kernel_file):
            path = task_dir / name
            digest.update(path.read_bytes() if path.exists() else b"")
        return digest.hexdigest()

    def input_dir(self, shape: Shape) -> Path:
        """Cache directory holding this shape's dumped inputs."""
        safe_shape = shape.key().replace("=", "-").replace(",", "_")
        return self.cache_root / self.task.name / safe_shape

    def ensure_inputs(self, shape: Shape, binary: str | Path) -> tuple[Path, np.ndarray]:
        """Prepare (or reuse) inputs and the cached reference output."""
        directory = self.input_dir(shape)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = self._source_stamp()
        manifest_path = directory / "manifest.json"
        ref_path = directory / "ref.npy"
        if manifest_path.exists() and ref_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
            if manifest.get("stamp") == stamp:
                return directory, np.load(ref_path)

        # Dump the exact inputs this kernel consumes, then recompute in NumPy.
        result = run_subprocess(
            [
                str(binary),
                "--shape",
                shape.key(),
                "--reps",
                "1",
                "--warmup",
                "0",
                "--dump-dir",
                str(directory),
            ],
            timeout_s=config.CHECK_TIMEOUT_S,
            retries=0,
        )
        if not result.ok:
            raise RunnerError(
                f"input dump failed: {result.error}", stderr=result.stderr
            )
        from ..tasks.registry import load_reference

        module = load_reference(self.task)
        reference_fn = getattr(module, "reference", None)
        if reference_fn is None:
            raise AttributeError(f"{self.task.name} reference.py has no reference()")
        expected = np.asarray(reference_fn(dict(shape.dims), directory), dtype=np.float32)
        np.save(ref_path, expected)
        dumped = sorted(path.name for path in directory.iterdir() if path.is_file())
        manifest_path.write_text(
            json.dumps({"stamp": stamp, "files": dumped, "shape": shape.dims}, indent=2),
            encoding="utf-8",
        )
        stale_output = directory / "out.bin"
        if stale_output.exists():
            stale_output.unlink()
        return directory, expected

    def check(self, shape: Shape, binary: str | Path) -> CheckResult:
        """Run the kernel twice and compare with the NumPy reference."""
        try:
            input_dir, expected = self.ensure_inputs(shape, binary)
        except (RunnerError, OSError, AttributeError, ValueError) as exc:
            return CheckResult(passed=False, message=str(exc))

        with tempfile.TemporaryDirectory(prefix="ka-check-") as tmp:
            first_path = Path(tmp) / "out1.bin"
            second_path = Path(tmp) / "out2.bin"
            try:
                self.runner.run_with_dump(binary, shape, first_path, input_dir)
                self.runner.run_with_dump(binary, shape, second_path, input_dir)
            except RunnerError as exc:
                return CheckResult(passed=False, message=f"kernel run failed: {exc}")

            expected_count = expected.size
            try:
                raw = np.fromfile(first_path, dtype=np.float32)
            except OSError as exc:
                return CheckResult(passed=False, message=f"cannot read output: {exc}")
            if raw.size != expected_count:
                return CheckResult(
                    passed=False,
                    message=f"output has {raw.size} floats, reference has {expected_count}",
                )
            actual = raw.reshape(expected.shape)
            raw2 = np.fromfile(second_path, dtype=np.float32)
            deterministic = raw2.size == raw.size and bool(np.array_equal(raw, raw2))

        nan_or_inf = bool(not np.all(np.isfinite(actual)))
        if nan_or_inf:
            return CheckResult(
                passed=False,
                nan_or_inf=True,
                deterministic=deterministic,
                message="kernel output contains NaN or Inf",
            )

        difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        max_abs = float(np.max(difference)) if difference.size else 0.0
        denominator = np.maximum(np.abs(expected.astype(np.float64)), 1e-6)
        max_rel = float(np.max(difference / denominator)) if difference.size else 0.0
        passed = bool(
            np.allclose(
                actual,
                expected,
                rtol=self.task.rtol,
                atol=self.task.atol,
                equal_nan=False,
            )
        )
        message = "ok" if passed else (
            f"mismatch: max_abs={max_abs:.4g} (atol={self.task.atol:g}), "
            f"max_rel={max_rel:.4g} (rtol={self.task.rtol:g})"
        )
        if passed and not deterministic:
            message = "numerically correct but not bitwise deterministic across runs"
        return CheckResult(
            passed=passed,
            max_abs_err=max_abs,
            max_rel_err=max_rel,
            nan_or_inf=False,
            deterministic=deterministic,
            message=message,
        )
