"""Round-trip serialization tests for the typed contracts."""

from __future__ import annotations

from kernelagent.types import (
    BuildResult,
    CheckResult,
    KernelTask,
    KernelVariant,
    KnobSpec,
    ProfileResult,
    RunResult,
    SearchRun,
    Shape,
    Trial,
)


def _task() -> KernelTask:
    return KernelTask(
        name="toy",
        directory="/tmp/toy",
        shapes={"small": Shape(dims={"M": 64, "N": 64})},
        knobs={"TILE": KnobSpec(name="TILE", values=[8, 16, 32])},
        default_knobs={"TILE": 16},
        flops_expr="M*N",
        bytes_expr="4*M*N",
    )


def _trial() -> Trial:
    return Trial(
        trial_id="t1",
        task="toy",
        shape=Shape(dims={"M": 64, "N": 64}),
        variant=KernelVariant(knobs={"TILE": 16}, cflags=["-O3"], provenance="test"),
        build=BuildResult(ok=True, compile_time_s=0.25, binary="/tmp/a.out"),
        run=RunResult(ms_median=1.5, ms_mad=0.05, ms_min=1.4, reps=5, throughput=3.2),
        profile=ProfileResult(counters={"gflops": 1.0}, source="derived"),
        check=CheckResult(passed=True, max_abs_err=1e-6, max_rel_err=1e-6),
        speedup=1.25,
        wall_s=2.0,
    )


def test_shape_key_is_stable() -> None:
    assert Shape(dims={"N": 4, "M": 2}).key() == "M=2,N=4"


def test_task_round_trip() -> None:
    task = _task()
    restored = KernelTask.model_validate_json(task.model_dump_json())
    assert restored == task
    assert restored.knob_space_size() == 3
    assert restored.default_variant().knobs == {"TILE": 16}


def test_trial_round_trip() -> None:
    trial = _trial()
    restored = Trial.model_validate_json(trial.model_dump_json())
    assert restored == trial
    assert restored.solved(1.05)
    assert restored.run is not None and restored.run.noise_ratio > 0


def test_search_run_round_trip() -> None:
    run = SearchRun(strategy="random", seed=3, budget=4, trials=[_trial()], best=_trial())
    restored = SearchRun.model_validate_json(run.model_dump_json())
    assert restored.best is not None
    assert restored.best.variant.knobs["TILE"] == 16


def test_variant_define_flags() -> None:
    variant = KernelVariant(knobs={"B": 2, "A": 1}, cflags=["-O3"])
    assert variant.define_flags() == ["-DA=1", "-DB=2"]
    assert variant.with_knob("A", 9).knobs["A"] == 9
