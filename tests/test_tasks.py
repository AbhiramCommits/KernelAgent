"""Task-suite integrity: every task compiles, is correct, and has a legal space."""

from __future__ import annotations

import pytest

from kernelagent.eval.harness import (
    compile_time_total,
    geomean_speedup,
    pass_at_k,
    solve_rate,
    trials_to_best,
)
from kernelagent.tasks.registry import enumerate_variants, load_tasks, variant_from_knobs
from kernelagent.tools.build import CxxBuilder
from kernelagent.tools.checker import CorrectnessChecker
from kernelagent.types import BuildResult, CheckResult, KernelVariant, RunResult, Shape, Trial

TASKS = load_tasks()


@pytest.mark.parametrize("task_name", sorted(TASKS))
def test_default_compiles_and_matches_reference(task_name: str) -> None:
    task = TASKS[task_name]
    build = CxxBuilder().build(task, task.default_variant())
    assert build.ok, build.raw_stderr
    assert build.binary is not None
    for shape_name, shape in task.shapes.items():
        check = CorrectnessChecker(task).check(shape, build.binary)
        assert check.passed, f"{task_name}/{shape_name}: {check.message}"
        assert check.nan_or_inf is False
        assert check.deterministic is True


@pytest.mark.parametrize("task_name", sorted(TASKS))
def test_knob_space_is_legal(task_name: str) -> None:
    task = TASKS[task_name]
    sizes = 1
    for spec in task.knobs.values():
        sizes *= spec.size()
    assert task.knob_space_size() == sizes
    seen = 0
    for variant in enumerate_variants(task):
        seen += 1
        assert set(variant.knobs) == set(task.knobs)
        rebuilt = variant_from_knobs(task, variant.knobs, variant.cflags, "test")
        assert rebuilt.key() == variant.key()
    assert seen == sizes
    assert task.default_variant().key() in {v.key() for v in enumerate_variants(task)}


@pytest.mark.parametrize("task_name", sorted(TASKS))
def test_shapes_cover_small_and_large(task_name: str) -> None:
    task = TASKS[task_name]
    assert len(task.shapes) >= 2
    assert "small" in task.shapes and "large" in task.shapes
    small_bytes = _working_set(task, task.shapes["small"])
    large_bytes = _working_set(task, task.shapes["large"])
    assert large_bytes > small_bytes
    assert large_bytes > 8 * 1024 * 1024, f"{task_name} has no large memory-bound shape"


def _working_set(task, shape: Shape) -> float:
    from kernelagent.workload import bytes_for

    return bytes_for(task, shape)


def _trial(ms: float, speedup: float, passed: bool = True) -> Trial:
    variant = KernelVariant(knobs={"x": 1})
    return Trial(
        trial_id=f"t{ms}",
        task="toy",
        shape=Shape(dims={"N": 1}),
        variant=variant,
        build=BuildResult(ok=True, compile_time_s=0.5),
        run=RunResult(ms_median=ms, ms_mad=0.0, ms_min=ms, reps=3),
        check=CheckResult(passed=passed),
        speedup=speedup if passed else None,
    )


def test_metrics() -> None:
    trials = [_trial(10.0, 0.9), _trial(9.0, 1.0), _trial(5.0, 2.0), _trial(1.0, 99.0, passed=False)]
    assert trials_to_best(trials) == 3
    assert pass_at_k(trials, 2) == 0.0
    assert pass_at_k(trials, 3) == 1.0
    assert solve_rate(trials) == pytest.approx(1 / 4)
    assert geomean_speedup(trials) == pytest.approx((0.9 * 1.0 * 2.0) ** (1 / 3))
    assert compile_time_total(trials) == pytest.approx(2.0)
