"""Scored evaluation harness around the typed tool layer."""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .. import config
from ..ledger import TrialLedger
from ..tasks.registry import default_variant, load_tasks, variant_from_knobs
from ..tools.build import CxxBuilder
from ..tools.checker import CorrectnessChecker
from ..tools.profiler import profile_binary
from ..tools.runner import BenchRunner, RunnerError
from ..types import (
    BuildResult,
    CheckResult,
    Confirmation,
    KernelTask,
    KernelVariant,
    RunResult,
    Shape,
    Trial,
)
from ..workload import flops_for

_TASK_CACHE: dict[str, KernelTask] = {}
_BASELINES: dict[tuple[str, str], float] = {}
_BASELINE_TRIALS: dict[tuple[str, str], Trial] = {}


def all_tasks() -> dict[str, KernelTask]:
    """Load (and memoise) the task suite."""
    global _TASK_CACHE
    if not _TASK_CACHE:
        _TASK_CACHE = load_tasks()
    return _TASK_CACHE


def get_task(name: str) -> KernelTask:
    """Task by name with a clear error."""
    tasks = all_tasks()
    if name not in tasks:
        raise KeyError(f"unknown task {name!r}; available={sorted(tasks)}")
    return tasks[name]


def get_shape(task: KernelTask, shape: str | Shape) -> Shape:
    """Resolve a shape name or object."""
    if isinstance(shape, Shape):
        return shape
    return task.shape_key(shape)


def parse_variant(task: KernelTask, spec: str | dict[str, Any] | KernelVariant | None) -> KernelVariant:
    """Accept a KernelVariant, a JSON string, a knob dict or ``"default"``."""
    if isinstance(spec, KernelVariant):
        return variant_from_knobs(
            task, dict(spec.knobs), list(spec.cflags), spec.provenance or "manual"
        )
    if spec is None or spec == "default":
        return default_variant(task)
    data: dict[str, Any]
    if isinstance(spec, str):
        data = json.loads(spec)
    elif isinstance(spec, dict):
        data = spec
    else:
        raise TypeError(f"cannot interpret variant spec of type {type(spec).__name__}")
    if "knobs" in data:
        knobs = dict(data["knobs"])
        cflags = list(data.get("cflags", []))
        provenance = str(data.get("provenance", "manual"))
    else:
        knobs = dict(data)
        cflags = []
        provenance = "manual"
    return variant_from_knobs(task, knobs, cflags, provenance)


def evaluate_variant(
    task: KernelTask,
    shape: Shape,
    variant: KernelVariant,
    *,
    strategy: str = "",
    seed: int = 0,
    ledger: TrialLedger | None = None,
    reps: int = config.DEFAULT_REPS,
    warmup: int = config.DEFAULT_WARMUP,
    do_profile: bool = True,
    builder: CxxBuilder | None = None,
    runner: BenchRunner | None = None,
    checker: CorrectnessChecker | None = None,
    baseline_ms: float | None = None,
) -> Trial:
    """Build -> check -> time -> profile one variant, returning a scored Trial.

    Compile or correctness failures produce a failed Trial carrying the real
    diagnostic; no timing or speedup is ever reported for an incorrect kernel.
    """
    started = time.perf_counter()
    trial_id = uuid.uuid4().hex[:12]
    builder = builder or CxxBuilder()
    runner = runner or BenchRunner()
    checker = checker or CorrectnessChecker(task)

    build: BuildResult = builder.build(task, variant)
    trial = Trial(
        trial_id=trial_id,
        task=task.name,
        shape=shape,
        variant=variant,
        build=build,
        strategy=strategy,
        seed=seed,
        timestamp=datetime.now(UTC).isoformat(),
    )
    if not build.ok or build.binary is None:
        trial.error = f"build failed: {build.error or 'unknown error'}"
        trial.wall_s = time.perf_counter() - started
        if ledger is not None:
            ledger.append(trial)
        return trial

    check: CheckResult = checker.check(shape, build.binary)
    trial.check = check
    if not check.passed:
        trial.error = f"correctness failed: {check.message}"
        trial.wall_s = time.perf_counter() - started
        if ledger is not None:
            ledger.append(trial)
        return trial

    try:
        run: RunResult = runner.run(
            build.binary,
            shape,
            reps=reps,
            warmup=warmup,
            flops=flops_for(task, shape),
        )
    except RunnerError as exc:
        trial.error = f"run failed: {exc}"
        trial.wall_s = time.perf_counter() - started
        if ledger is not None:
            ledger.append(trial)
        return trial
    trial.run = run

    if do_profile:
        trial.profile = profile_binary(task, shape, build.binary, run)

    if baseline_ms is None:
        baseline_ms = baseline_time(task, shape, ledger=None)
    trial.speedup = baseline_ms / run.ms_median if run.ms_median > 0 else None
    trial.wall_s = time.perf_counter() - started
    if ledger is not None:
        ledger.append(trial)
    return trial


def baseline_time(
    task: KernelTask, shape: Shape, reps: int = config.DEFAULT_REPS, ledger: TrialLedger | None = None
) -> float:
    """Median time of the task's default variant, measured once per process."""
    key = (task.name, shape.key())
    if key in _BASELINES:
        return _BASELINES[key]
    builder = CxxBuilder()
    variant = default_variant(task)
    build = builder.build(task, variant)
    if not build.ok or build.binary is None:
        raise RuntimeError(f"baseline for {task.name}/{shape.key()} did not compile: {build.error}")
    checker = CorrectnessChecker(task)
    check = checker.check(shape, build.binary)
    if not check.passed:
        raise RuntimeError(
            f"baseline for {task.name}/{shape.key()} is incorrect: {check.message}"
        )
    runner = BenchRunner()
    run = runner.run(build.binary, shape, reps=reps, flops=flops_for(task, shape))
    trial = Trial(
        trial_id=uuid.uuid4().hex[:12],
        task=task.name,
        shape=shape,
        variant=variant,
        build=build,
        run=run,
        check=check,
        profile=None,
        speedup=1.0,
        wall_s=0.0,
        strategy="baseline",
        seed=-1,
        timestamp=datetime.now(UTC).isoformat(),
    )
    _BASELINES[key] = run.ms_median
    _BASELINE_TRIALS[key] = trial
    if ledger is not None:
        ledger.append(trial)
    return run.ms_median


def baseline_trial(task: KernelTask, shape: Shape) -> Trial:
    """The cached baseline trial (measured on first access)."""
    baseline_time(task, shape)
    return _BASELINE_TRIALS[(task.name, shape.key())]


def confirm_best(trial: Trial, repeats: int = 3) -> Confirmation:
    """Re-measure a winner several times and report median, spread and speedup."""
    task = get_task(trial.task)
    shape = trial.shape
    if trial.build.binary is None:
        raise ValueError("cannot confirm a trial without a binary")
    checker = CorrectnessChecker(task)
    check = checker.check(shape, trial.build.binary)
    runner = BenchRunner()
    times: list[float] = []
    noises: list[float] = []
    for _ in range(max(1, repeats)):
        run = runner.run(trial.build.binary, shape, reps=config.DEFAULT_REPS, warmup=config.DEFAULT_WARMUP)
        times.append(run.ms_median)
        noises.append(run.noise_ratio)
    median_ms = float(np.median(times))
    baseline = baseline_time(task, shape)
    return Confirmation(
        task=trial.task,
        shape=shape,
        variant=trial.variant,
        ms_median=median_ms,
        ms_min=float(min(times)),
        ms_max=float(max(times)),
        spread_pct=100.0 * (max(times) - min(times)) / median_ms if median_ms > 0 else 0.0,
        baseline_ms=baseline,
        speedup=baseline / median_ms if median_ms > 0 else 0.0,
        repeats=len(times),
        noise_ratios=noises,
        check_passed=check.passed,
    )


def check_variant(task_name: str, shape_name: str, variant_spec: Any) -> Trial:
    """Build and check one variant without timing it."""
    task = get_task(task_name)
    shape = get_shape(task, shape_name)
    variant = parse_variant(task, variant_spec)
    build = CxxBuilder().build(task, variant)
    trial = Trial(
        trial_id=uuid.uuid4().hex[:12],
        task=task.name,
        shape=shape,
        variant=variant,
        build=build,
        timestamp=datetime.now(UTC).isoformat(),
    )
    if build.ok and build.binary:
        trial.check = CorrectnessChecker(task).check(shape, build.binary)
        trial.error = None if trial.check.passed else trial.check.message
    else:
        trial.error = build.error
    return trial


def run_variant(
    task_name: str,
    shape_name: str,
    variant_spec: Any = "default",
    reps: int = config.DEFAULT_REPS,
    ledger: TrialLedger | None = None,
) -> Trial:
    """Build, check and time one variant (the CLI ``run`` path)."""
    task = get_task(task_name)
    shape = get_shape(task, shape_name)
    variant = parse_variant(task, variant_spec)
    return evaluate_variant(task, shape, variant, reps=reps, ledger=ledger)


def speedup(trial: Trial) -> float | None:
    """Measured speedup, only for correct trials."""
    if trial.check is None or not trial.check.passed or trial.run is None:
        return None
    return trial.speedup


def geomean(values: Iterable[float]) -> float:
    """Geometric mean of positive values (0.0 for an empty sequence)."""
    positive = [float(v) for v in values if v is not None and v > 0]
    if not positive:
        return 0.0
    return float(math.exp(sum(math.log(v) for v in positive) / len(positive)))


def geomean_speedup(trials: Iterable[Trial]) -> float:
    """Geometric mean speedup over correct trials."""
    return geomean([s for t in trials if (s := speedup(t)) is not None])


def pass_at_k(trials: list[Trial], k: int, threshold: float = config.SPEEDUP_SOLVE_THRESHOLD) -> float:
    """1.0 when any of the first ``k`` trials is correct and fast enough."""
    for trial in trials[:k]:
        if trial.solved(threshold):
            return 1.0
    return 0.0


def trials_to_best(trials: list[Trial]) -> int | None:
    """1-based index of the first trial attaining the maximum measured speedup."""
    best_index = None
    best_value = -1.0
    for index, trial in enumerate(trials):
        value = speedup(trial)
        if value is not None and value > best_value:
            best_value = value
            best_index = index + 1
    return best_index


def solve_rate(trials: Iterable[Trial], threshold: float = config.SPEEDUP_SOLVE_THRESHOLD) -> float:
    """Fraction of trials that are correct and at least ``threshold`` faster."""
    materialised = list(trials)
    if not materialised:
        return 0.0
    return sum(1 for trial in materialised if trial.solved(threshold)) / len(materialised)


def compile_time_total(trials: Iterable[Trial]) -> float:
    """Sum of wall compile time for builds that actually ran (cache misses)."""
    return sum(t.build.compile_time_s for t in trials if not t.build.cached)


def compiles(trials: Iterable[Trial]) -> int:
    """Number of cache-miss compilations."""
    return sum(1 for t in trials if not t.build.cached)


def best_trial(trials: list[Trial]) -> Trial | None:
    """Correct trial with the highest measured speedup."""
    correct = [t for t in trials if speedup(t) is not None]
    if not correct:
        return None
    return max(correct, key=lambda t: t.speedup or 0.0)


def problem_index(tasks: dict[str, KernelTask] | None = None) -> list[tuple[KernelTask, str, Shape]]:
    """Every (task, shape name, shape) problem in stable order."""
    suite = tasks or all_tasks()
    problems: list[tuple[KernelTask, str, Shape]] = []
    for task in suite.values():
        for shape_name, shape in task.shapes.items():
            problems.append((task, shape_name, shape))
    return problems


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with a stable key order and a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)
