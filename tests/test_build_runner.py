"""Build, run and correctness integration tests on the real compiler."""

from __future__ import annotations

from kernelagent.tasks.registry import load_tasks
from kernelagent.tools.build import CxxBuilder
from kernelagent.tools.checker import CorrectnessChecker
from kernelagent.tools.runner import BenchRunner
from kernelagent.types import Shape


def test_build_run_check_sgemm() -> None:
    task = load_tasks()["sgemm_tiled"]
    variant = task.default_variant()
    build = CxxBuilder().build(task, variant)
    assert build.ok, build.raw_stderr
    assert build.compile_time_s > 0.0
    assert build.binary is not None
    diagnostics = build.diagnostics
    assert diagnostics["vectorized_loops"] >= 1
    assert diagnostics["source"] == "opt-record"

    shape = Shape(dims={"M": 128, "N": 128, "K": 128})
    runner = BenchRunner()
    run = runner.run(build.binary, shape, reps=3, warmup=1)
    assert run.ms_median > 0
    assert run.reps >= 1
    assert run.noise_ratio >= 0

    check = CorrectnessChecker(task).check(shape, build.binary)
    assert check.passed, check.message
    assert check.deterministic is True
    assert check.max_abs_err is not None and check.max_abs_err < 0.1


def test_build_cache_hit() -> None:
    task = load_tasks()["sgemm_tiled"]
    variant = task.default_variant()
    builder = CxxBuilder()
    first = builder.build(task, variant)
    second = CxxBuilder().build(task, variant)
    assert first.ok and second.ok
    assert second.cached is True
    assert second.binary == first.binary


def test_build_rejects_unlisted_flag() -> None:
    task = load_tasks()["sgemm_tiled"]
    variant = task.default_variant().model_copy(update={"cflags": ["-o", "/tmp/evil"]})
    result = CxxBuilder().build(task, variant)
    assert not result.ok
    assert "not allowed" in (result.error or "")


def test_build_compile_error_is_structured(tmp_path) -> None:
    task = load_tasks()["sgemm_tiled"]
    broken_source = tmp_path / "broken.cpp"
    broken_source.write_text("int main( { return 0; }\n")
    copied = task.model_copy(update={"directory": str(tmp_path), "kernel_file": "broken.cpp"})
    result = CxxBuilder().build(copied, task.default_variant())
    assert not result.ok
    assert result.binary is None
    assert "error:" in result.raw_stderr
