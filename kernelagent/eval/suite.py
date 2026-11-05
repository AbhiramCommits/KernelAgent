"""Run every (task, shape) problem with one strategy and summarise the outcome."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import config
from ..env import probe_environment
from ..ledger import TrialLedger, git_sha
from ..types import SearchRun, Trial
from .harness import (
    all_tasks,
    baseline_time,
    best_trial,
    compile_time_total,
    evaluate_variant,
    geomean,
    parse_variant,
    problem_index,
    write_json,
)
from .library_baseline import library_reference

SUPPORTED_STRATEGIES = ("default", "random", "greedy", "mutation", "evolution", "guided", "llm")


def _library_for(task, shape, binary) -> dict[str, Any] | None:
    from ..tools.checker import CorrectnessChecker

    try:
        input_dir, _ = CorrectnessChecker(task).ensure_inputs(shape, binary)
    except Exception:
        return None
    return library_reference(task, shape, input_dir)


def _row(task, shape_name, shape, trial: Trial, library: dict[str, Any] | None) -> dict[str, Any]:
    run = trial.run
    profile = trial.profile.counters if trial.profile else {}
    library_ms = library["ms_median"] if library else None
    return {
        "task": task.name,
        "shape": shape_name,
        "shape_dims": dict(shape.dims),
        "variant": trial.variant.model_dump(mode="json"),
        "build_ok": trial.build.ok,
        "compile_time_s": trial.build.compile_time_s,
        "cached": trial.build.cached,
        "check_passed": bool(trial.check and trial.check.passed),
        "check_message": trial.check.message if trial.check else "",
        "ms_median": run.ms_median if run else None,
        "ms_mad": run.ms_mad if run else None,
        "reps": run.reps if run else None,
        "gflops": profile.get("gflops"),
        "gbps": profile.get("gbps"),
        "profile_source": trial.profile.source if trial.profile else None,
        "speedup": trial.speedup,
        "library_ms": library_ms,
        "library_name": library["name"] if library else None,
        "library_speedup": (library_ms / run.ms_median) if (library_ms and run) else None,
        "error": trial.error,
    }


def run_suite(
    strategy: str = "default",
    budget: int = config.DEFAULT_BUDGET,
    seeds: list[int] | None = None,
    out: Path | None = None,
    tasks: list[str] | None = None,
) -> dict[str, Any]:
    """Execute the suite and write the summary JSON; return the same payload."""
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {SUPPORTED_STRATEGIES}")
    seeds = seeds or [0]
    started = time.perf_counter()
    ledger = TrialLedger()
    suite = all_tasks()
    if tasks:
        suite = {name: suite[name] for name in tasks}
    problems = problem_index(suite)
    rows: list[dict[str, Any]] = []
    all_trials: list[Trial] = []
    runs: list[SearchRun] = []

    for task, shape_name, shape in problems:
        base = baseline_time(task, shape)
        if strategy == "default":
            variant = parse_variant(task, "default")
            trial = evaluate_variant(task, shape, variant, strategy="default", seed=0, ledger=ledger)
            all_trials.append(trial)
            if trial.check is None or not trial.check.passed:
                trial.speedup = None
            else:
                trial.speedup = base / trial.run.ms_median if trial.run else None
            library = _library_for(task, shape, trial.build.binary) if trial.build.binary else None
            rows.append(_row(task, shape_name, shape, trial, library))
            continue

        from ..agent.loop import run_search

        best: Trial | None = None
        for seed in seeds:
            search = run_search(
                task=task.name,
                shape=shape_name,
                strategy=strategy,
                budget=budget,
                seed=seed,
                patience=config.DEFAULT_PATIENCE,
                ledger=ledger,
            )
            runs.append(search)
            all_trials.extend(search.trials)
            candidate = search.best or best_trial(search.trials)
            if candidate is not None and (
                best is None or (candidate.speedup or 0.0) > (best.speedup or 0.0)
            ):
                best = candidate
        if best is not None:
            library = _library_for(task, shape, best.build.binary) if best.build.binary else None
            rows.append(_row(task, shape_name, shape, best, library))
        else:
            failed = SearchRun(strategy=strategy, seed=seeds[0], budget=budget)
            trial = Trial(
                trial_id="failed",
                task=task.name,
                shape=shape,
                variant=parse_variant(task, "default"),
                build=failed.trials[0].build if failed.trials else _empty_build(),
                error="no successful trial",
            )
            rows.append(_row(task, shape_name, shape, trial, None))

    wall_s = time.perf_counter() - started
    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "environment": probe_environment(),
        "strategy": strategy,
        "budget": budget,
        "seeds": seeds,
        "problems": len(rows),
        "trials": len(all_trials),
        "builds": sum(1 for t in all_trials if t.build.ok),
        "compile_time_total_s": compile_time_total(all_trials),
        "wall_s": wall_s,
        "geomean_speedup": geomean(
            [row["speedup"] for row in rows if row.get("speedup") is not None]
        ),
        "solve_rate": (
            sum(1 for row in rows if (row.get("speedup") or 0) >= config.SPEEDUP_SOLVE_THRESHOLD)
            / len(rows)
            if rows
            else 0.0
        ),
        "library_geomean_speedup": geomean(
            [row["library_speedup"] for row in rows if row.get("library_speedup") is not None]
        ),
        "rows": rows,
    }
    if strategy != "default":
        payload["search_runs"] = [
            {
                "task": run.strategy,
                "seed": run.seed,
                "trials": len(run.trials),
                "stop_reason": run.stop_reason,
                "wall_s": run.wall_s,
            }
            for run in runs
        ]
    target = Path(out) if out else config.RESULTS_DIR / "baseline.json"
    write_json(target, payload)
    return payload


def _empty_build():
    from ..types import BuildResult

    return BuildResult(ok=False, error="no trial")
