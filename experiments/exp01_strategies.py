"""exp01: strategy comparison on every (task, shape) problem, 3 seeds."""

from __future__ import annotations

import os
from typing import Any

from kernelagent import config
from kernelagent.agent.loop import run_search
from kernelagent.agent.proposer import LLMProposer
from kernelagent.eval.harness import (
    all_tasks,
    confirm_best,
    geomean,
    problem_index,
    trials_to_best,
)
from kernelagent.ledger import TrialLedger
from kernelagent.types import SearchRun, Trial

from .common import aggregate_best, assemble_summary, search_record, write_experiment


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def main() -> dict[str, Any]:
    """Run the strategy sweep and write exp01 JSON."""
    budget = int(os.environ.get("KA_EXP_BUDGET", "8"))
    seeds = [int(part) for part in os.environ.get("KA_EXP_SEEDS", "0,1,2").split(",") if part]
    task_filter = [name for name in os.environ.get("KA_EXP_TASKS", "").split(",") if name]
    suite = all_tasks()
    if task_filter:
        suite = {name: suite[name] for name in task_filter}
    problems = problem_index(suite)

    strategies = ["random", "greedy", "mutation", "evolution", "guided"]
    llm_available = LLMProposer.available()
    if llm_available:
        strategies.append("llm")

    ledger = TrialLedger()
    runs: list[SearchRun] = []
    all_trials: list[Trial] = []
    for task, shape_name, _shape in problems:
        for strategy in strategies:
            for seed in seeds:
                run = run_search(
                    task=task.name,
                    shape=shape_name,
                    strategy=strategy,
                    budget=budget,
                    seed=seed,
                    ledger=ledger,
                    confirm=False,
                )
                runs.append(run)
                all_trials.extend(run.trials)
                print(
                    f"{task.name:18s} {shape_name:5s} {strategy:9s} seed={seed} "
                    f"trials={len(run.trials):2d} best="
                    f"{(run.best.speedup if run.best and run.best.speedup else 0):.3f}",
                    flush=True,
                )

    # Confirm each strategy's best variant per problem, not just the raw sample.
    for (_strategy, task_name, shape_key), entry in aggregate_best(runs).items():
        best_run = entry["run"]
        if best_run.best is not None:
            try:
                best_run.confirmation = confirm_best(best_run.best, repeats=3)
            except Exception as exc:  # pragma: no cover - confirmation is best-effort
                print(f"confirmation failed for {task_name}/{shape_key}: {exc}", flush=True)

    confirmed = aggregate_best(runs)
    summary: dict[str, Any] = {}
    for strategy in strategies:
        per_problem: list[dict[str, Any]] = []
        for task, shape_name, shape in problems:
            best_entry = confirmed.get((strategy, task.name, shape.key()))
            if best_entry is None:
                continue
            confirmation = best_entry["confirmation"]
            best = best_entry["run"].best
            raw = best.speedup if best and best.speedup else None
            value = confirmation.speedup if confirmation else raw
            per_problem.append(
                {
                    "task": task.name,
                    "shape": shape_name,
                    "speedup": value,
                    "raw_speedup": raw,
                    "confirmed_spread_pct": confirmation.spread_pct if confirmation else None,
                    "seed": best_entry["seed"],
                    "trials_to_best": trials_to_best(best_entry["run"].trials),
                    "stop_reason": best_entry["run"].stop_reason,
                }
            )
        per_task: list[dict[str, Any]] = []
        for task in suite.values():
            rows = [row for row in per_problem if row["task"] == task.name]
            values = [row["speedup"] for row in rows if row["speedup"] is not None]
            if values:
                per_task.append(
                    {
                        "task": task.name,
                        "geomean": geomean(values),
                        "best": max(values),
                        "worst": min(values),
                        "shapes": len(values),
                    }
                )
        strategy_runs = [run for run in runs if run.strategy == strategy]
        scored = [row["speedup"] for row in per_problem if row["speedup"] is not None]
        summary[strategy] = {
            "solve_rate": (
                sum(1 for value in scored if value >= config.SPEEDUP_SOLVE_THRESHOLD)
                / len(per_problem)
                if per_problem
                else 0.0
            ),
            "geomean_speedup": geomean(scored),
            "best_task_speedup": max((row["geomean"] for row in per_task), default=0.0),
            "worst_task_speedup": min((row["geomean"] for row in per_task), default=0.0),
            "per_task": per_task,
            "per_problem": per_problem,
            "trials_to_best_mean": _mean(
                [row["trials_to_best"] for row in per_problem if row["trials_to_best"]]
            ),
            "compiles": sum(1 for run in strategy_runs for trial in run.trials if not trial.build.cached),
            "compile_time_s": sum(
                trial.build.compile_time_s
                for run in strategy_runs
                for trial in run.trials
                if not trial.build.cached
            ),
            "wall_s": sum(run.wall_s for run in strategy_runs),
            "tokens": sum(run.tokens for run in strategy_runs),
            "runs": len(strategy_runs),
            "scored_problems": len(scored),
        }

    payload: dict[str, Any] = {
        "budget": budget,
        "seeds": seeds,
        "problems": len(problems),
        "strategies": summary,
        "runs": [search_record(run) for run in runs],
        "llm": {
            "available": llm_available,
            "note": (
                "LLM proposer ran with a live API key"
                if llm_available
                else "implemented and fixture-tested offline; no API key, so it did not run"
            ),
        },
        "notes": "Every strategy/budget/seed is identical across problems; winners are confirmed by 3 re-measurements.",
    }
    write_experiment("exp01_strategies", payload, all_trials, runs)
    assemble_summary()
    return payload


if __name__ == "__main__":
    result = main()
    print(f"wrote exp01: {result['problems']} problems, {len(result['runs'])} runs")
