"""exp03: cost-model ranking quality, guided compiles-to-best, oracle gap."""

from __future__ import annotations

import json
import os
from typing import Any

from kernelagent import config
from kernelagent.eval.harness import all_tasks, problem_index
from kernelagent.select.baselines import oracle_gap, run_oracle
from kernelagent.select.evaluate import (
    collect_loto_predictions,
    run_loto_evaluation,
)

from .common import RESULTS, write_experiment


def _load_exp01() -> dict[str, Any]:
    path = RESULTS / "exp01_strategies.json"
    if not path.exists():
        raise FileNotFoundError("exp03 needs results/exp01_strategies.json; run exp01 first")
    return json.loads(path.read_text(encoding="utf-8"))


def _compiles_to_best(run: dict[str, Any]) -> tuple[int, int] | None:
    """(build requests, cache misses) up to the first best-speedup trial.

    A cache hit still represents a compiled candidate from the strategy's point
    of view, so build requests are the fair equal-budget comparison; cache
    misses are reported separately as the true compiler cost in this run.
    """
    speedups = run.get("speedups") or []
    compiled = run.get("compiled") or []
    best_index = None
    best_value = -1.0
    for index, value in enumerate(speedups):
        if value is not None and value > best_value:
            best_value = value
            best_index = index
    if best_index is None:
        return None
    return (best_index + 1, int(sum(compiled[: best_index + 1])))


def _guiding_stats(exp01: dict[str, Any]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    problems = {(run["task"], run["shape"]) for run in exp01.get("runs", [])}
    for strategy in ("random", "guided"):
        values = []
        cache_values = []
        for task, shape in sorted(problems):
            candidates = [
                run
                for run in exp01["runs"]
                if run["strategy"] == strategy and run["task"] == task and run["shape"] == shape
            ]
            if not candidates:
                continue
            best = max(candidates, key=lambda run: run.get("best_speedup") or 0.0)
            value = _compiles_to_best(best)
            if value is not None:
                values.append(value[0])
                cache_values.append(value[1])
        stats[strategy] = {
            "compiles_to_best_mean": float(sum(values) / len(values)) if values else None,
            "cache_misses_to_best_mean": (
                float(sum(cache_values) / len(cache_values)) if cache_values else None
            ),
            "problems": len(values),
            "budget": exp01.get("budget"),
        }
    if stats["random"]["compiles_to_best_mean"] and stats["guided"]["compiles_to_best_mean"]:
        stats["guided_saving_pct"] = 100.0 * (
            1.0
            - stats["guided"]["compiles_to_best_mean"]
            / stats["random"]["compiles_to_best_mean"]
        )
    return stats


def main() -> dict[str, Any]:
    """Run ranking evaluation, guided comparison and oracles."""
    exp01 = _load_exp01()
    loto = run_loto_evaluation(config.RANK_DATASET)
    predictions = collect_loto_predictions(config.RANK_DATASET)

    maximum_oracle = int(os.environ.get("KA_ORACLE_MAX", "256"))
    suite = all_tasks()
    best_by_problem: dict[tuple[str, str], float] = {}
    for run in exp01.get("runs", []):
        key = (run["task"], run["shape"])
        value = run.get("confirmed_speedup") or run.get("best_speedup")
        if value is not None:
            best_by_problem[key] = max(best_by_problem.get(key, 0.0), float(value))

    oracle_rows = []
    oracle_variants = 0
    for task, shape_name, shape in problem_index(suite):
        if not (task.knob_space_size() <= maximum_oracle):
            continue
        try:
            result = run_oracle(task, shape, max_variants=maximum_oracle)
        except Exception as exc:  # pragma: no cover
            print(f"oracle failed for {task.name}/{shape_name}: {exc}", flush=True)
            continue
        oracle_variants += int(result["variants"])
        found = best_by_problem.get((task.name, shape.key()))
        oracle_rows.append(
            {
                "task": task.name,
                "shape": shape_name,
                "variants": result["variants"],
                "oracle_speedup": result["oracle_speedup"],
                "oracle_raw_speedup": result.get("oracle_raw_speedup"),
                "oracle_spread_pct": result.get("oracle_spread_pct"),
                "found_speedup": found,
                "oracle_gap": oracle_gap(found, result["oracle_speedup"]),
                "wall_s": result["wall_s"],
            }
        )
        print(
            f"oracle {task.name:18s} {shape_name:5s} variants={result['variants']:3d} "
            f"oracle={result['oracle_speedup'] or 0:.3f} found={found or 0:.3f}",
            flush=True,
        )

    gaps = [row["oracle_gap"] for row in oracle_rows if row["oracle_gap"] is not None]
    payload: dict[str, Any] = {
        "budget": exp01.get("budget"),
        "seeds": exp01.get("seeds"),
        "loto": loto,
        "predictions": predictions,
        "guiding": _guiding_stats(exp01),
        "oracle": {
            "rows": oracle_rows,
            "mean_oracle_gap": float(sum(gaps) / len(gaps)) if gaps else None,
            "problems": len(oracle_rows),
            "max_variants": maximum_oracle,
        },
        "oracle_variants": oracle_variants,
        "notes": (
            "Compiles-to-best counts build requests (one per trial) in the equal-budget exp01 "
            "runs, with cache misses reported separately; the oracle enumerates every legal "
            "variant for tasks with a small knob space."
        ),
    }
    write_experiment("exp03_cost_model", payload, [], [])
    return payload


if __name__ == "__main__":
    result = main()
    print(f"wrote exp03: {result['oracle']['problems']} oracle problems")
