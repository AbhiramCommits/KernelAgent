"""exp02: feedback-signal ablation for the mutation agent."""

from __future__ import annotations

import os
from typing import Any

from kernelagent import config
from kernelagent.agent.loop import run_search
from kernelagent.eval.harness import (
    all_tasks,
    confirm_best,
    geomean,
    problem_index,
    trials_to_best,
)
from kernelagent.ledger import TrialLedger
from kernelagent.types import SearchRun, Trial

from .common import search_record, write_experiment

SIGNAL_SETS: list[tuple[str, list[str]]] = [
    ("runtime", ["runtime"]),
    ("runtime+build_errors", ["runtime", "build_errors"]),
    ("runtime+build_errors+compiler_remarks", ["runtime", "build_errors", "compiler_remarks"]),
    (
        "all+runtime+build_errors+compiler_remarks+profile_counters",
        ["runtime", "build_errors", "compiler_remarks", "profile_counters"],
    ),
]


def main() -> dict[str, Any]:
    """Run the ablation and write exp02 JSON."""
    budget = int(os.environ.get("KA_EXP_BUDGET", "8"))
    seeds = [int(part) for part in os.environ.get("KA_EXP_SEEDS", "0,1").split(",") if part]
    task_filter = [name for name in os.environ.get("KA_EXP_TASKS", "").split(",") if name]
    suite = all_tasks()
    if task_filter:
        suite = {name: suite[name] for name in task_filter}
    problems = problem_index(suite)

    ledger = TrialLedger()
    runs: list[SearchRun] = []
    run_labels: dict[int, str] = {}
    all_trials: list[Trial] = []
    for task, shape_name, _shape in problems:
        for label, signals in SIGNAL_SETS:
            for seed in seeds:
                run = run_search(
                    task=task.name,
                    shape=shape_name,
                    strategy="mutation",
                    budget=budget,
                    seed=seed,
                    signals=signals,
                    ledger=ledger,
                )
                run_labels[id(run)] = label
                runs.append(run)
                all_trials.extend(run.trials)
                print(
                    f"{task.name:18s} {shape_name:5s} {label:52s} seed={seed} "
                    f"best={(run.best.speedup if run.best and run.best.speedup else 0):.3f}",
                    flush=True,
                )

    # Confirm the best run per (signal set, problem).
    best_runs: dict[tuple[str, str, str], SearchRun] = {}
    for candidate_run in runs:
        if not candidate_run.trials or candidate_run.best is None:
            continue
        label = run_labels.get(id(candidate_run), candidate_run.strategy)
        key = (label, candidate_run.trials[0].task, candidate_run.trials[0].shape.key())
        existing = best_runs.get(key)
        existing_speed = existing.best.speedup if existing and existing.best else 0.0
        candidate_speed = candidate_run.best.speedup or 0.0
        if existing is None or candidate_speed > (existing_speed or 0.0):
            best_runs[key] = candidate_run
    for confirm_run in best_runs.values():
        if confirm_run.best is None:
            continue
        try:
            confirm_run.confirmation = confirm_best(confirm_run.best, repeats=3)
        except Exception as exc:  # pragma: no cover
            print(f"confirmation failed: {exc}", flush=True)

    by_signal: dict[str, Any] = {}
    for label, signals in SIGNAL_SETS:
        per_problem: list[dict[str, Any]] = []
        for task, shape_name, shape in problems:
            key = (label, task.name, shape.key())
            best_run = best_runs.get(key)
            if best_run is None or best_run.best is None:
                continue
            confirmation = best_run.confirmation
            per_problem.append(
                {
                    "task": task.name,
                    "shape": shape_name,
                    "speedup": (
                        confirmation.speedup if confirmation else best_run.best.speedup
                    ),
                    "raw_speedup": best_run.best.speedup,
                    "spread_pct": confirmation.spread_pct if confirmation else None,
                    "trials_to_best": trials_to_best(best_run.trials),
                    "seed": best_run.seed,
                }
            )
        values: list[float] = [
            float(row["speedup"]) for row in per_problem if row["speedup"] is not None
        ]
        signal_runs = [
            search_run
            for search_run in runs
            if run_labels.get(id(search_run), "") == label
        ]
        by_signal[label] = {
            "signals": signals,
            "solve_rate": (
                sum(1 for value in values if value >= config.SPEEDUP_SOLVE_THRESHOLD)
                / len(per_problem)
                if per_problem
                else 0.0
            ),
            "geomean_speedup": geomean(values),
            "per_problem": per_problem,
            "trials": sum(len(run.trials) for run in signal_runs),
            "compiles": sum(
                1 for run in signal_runs for trial in run.trials if not trial.build.cached
            ),
            "wall_s": sum(run.wall_s for run in signal_runs),
            "tokens": sum(run.tokens for run in signal_runs),
        }

    labels = [label for label, _ in SIGNAL_SETS]
    deltas: dict[str, Any] = {}
    for previous, next_label in zip(labels, labels[1:], strict=False):
        deltas[f"{previous} -> {next_label}"] = {
            "solve_rate_delta": by_signal[next_label]["solve_rate"]
            - by_signal[previous]["solve_rate"],
            "geomean_delta": by_signal[next_label]["geomean_speedup"]
            - by_signal[previous]["geomean_speedup"],
        }

    payload: dict[str, Any] = {
        "budget": budget,
        "seeds": seeds,
        "problems": len(problems),
        "signal_sets": by_signal,
        "deltas": deltas,
        "runs": [search_record(run) for run in runs],
        "notes": "Same mutation proposer, budget and seeds; only the rendered feedback signal blocks differ.",
    }
    write_experiment("exp02_ablation", payload, all_trials, runs)
    return payload


if __name__ == "__main__":
    result = main()
    print(f"wrote exp02: {len(result['signal_sets'])} signal sets")
