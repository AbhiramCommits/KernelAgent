"""Shared experiment plumbing: compact records, JSON/JSONL writers, summary."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

from kernelagent import config
from kernelagent.env import probe_environment
from kernelagent.eval.harness import (
    best_trial,
    compile_time_total,
    compiles,
    trials_to_best,
)
from kernelagent.ledger import git_sha
from kernelagent.types import SearchRun, Trial

RESULTS = config.RESULTS_DIR

_PROFILE_KEYS = (
    "gflops",
    "gbps",
    "bandwidth_pct",
    "arithmetic_intensity",
    "working_set_over_cache",
    "occupancy_pct",
    "dram_throughput_pct",
    "sm_throughput_pct",
)


def now() -> str:
    """UTC timestamp."""
    return datetime.now(UTC).isoformat()


def compact_trial(trial: Trial) -> dict[str, Any]:
    """A ledger-sized trial record without megabytes of YAML remarks."""
    build = trial.build
    diagnostics = build.diagnostics
    profile = trial.profile
    counters = profile.counters if profile else {}
    return {
        "trial_id": trial.trial_id,
        "task": trial.task,
        "shape": trial.shape.key(),
        "shape_dims": trial.shape.dims,
        "variant": trial.variant.model_dump(mode="json"),
        "strategy": trial.strategy,
        "seed": trial.seed,
        "build": {
            "ok": build.ok,
            "compile_time_s": build.compile_time_s,
            "cached": build.cached,
            "source_hash": build.source_hash,
            "vectorized_loops": diagnostics.get("vectorized_loops"),
            "missed_count": diagnostics.get("missed_count"),
            "unroll": diagnostics.get("unroll"),
            "compile_errors": len(diagnostics.get("compile_errors") or []),
        },
        "check": trial.check.model_dump(mode="json") if trial.check else None,
        "run": trial.run.model_dump(mode="json") if trial.run else None,
        "profile": {
            "source": profile.source if profile else None,
            "counters": {key: counters[key] for key in _PROFILE_KEYS if key in counters},
        },
        "speedup": trial.speedup,
        "wall_s": trial.wall_s,
        "error": trial.error,
        "timestamp": trial.timestamp,
    }


def search_record(run: SearchRun) -> dict[str, Any]:
    """Compact description of one search episode."""
    best = run.best or best_trial(run.trials)
    confirmation = run.confirmation
    return {
        "strategy": run.strategy,
        "seed": run.seed,
        "budget": run.budget,
        "task": run.trials[0].task if run.trials else "",
        "shape": run.trials[0].shape.key() if run.trials else "",
        "trials": len(run.trials),
        "best_speedup": best.speedup if best else None,
        "confirmed_speedup": confirmation.speedup if confirmation else None,
        "confirmation_spread_pct": confirmation.spread_pct if confirmation else None,
        "best_variant": best.variant.model_dump(mode="json") if best else None,
        "trials_to_best": trials_to_best(run.trials),
        "speedups": [trial.speedup for trial in run.trials],
        "compiled": [0 if trial.build.cached else 1 for trial in run.trials],
        "compiles": compiles(run.trials),
        "compile_time_s": compile_time_total(run.trials),
        "wall_s": run.wall_s,
        "tokens": run.tokens,
        "repair_attempts": run.repair_attempts,
        "stop_reason": run.stop_reason,
        "solved": bool(best and best.solved(config.SPEEDUP_SOLVE_THRESHOLD)),
    }


def write_experiment(
    name: str,
    payload: dict[str, Any],
    trials: list[Trial],
    runs: list[SearchRun],
) -> dict[str, Any]:
    """Write ``results/<name>.json``, its trial JSONL and return the payload."""
    results = RESULTS
    results.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": name,
        "generated_at": now(),
        "git_sha": git_sha(),
        "environment": probe_environment(),
        "budget": int(os.environ.get("KA_EXP_BUDGET", payload.get("budget", config.DEFAULT_BUDGET))),
        "seeds": payload.get("seeds", [0]),
        **payload,
    }
    (results / f"{name}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    with (results / f"{name}_trials.jsonl").open("w", encoding="utf-8") as handle:
        for trial in trials:
            handle.write(
                json.dumps({"experiment": name, **compact_trial(trial)}, sort_keys=True, default=str)
                + "\n"
            )
    return payload


def aggregate_best(
    runs: list[SearchRun],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Best run per (strategy, task, shape), confirming each winner once."""
    grouped: dict[tuple[str, str, str], list[SearchRun]] = {}
    for run in runs:
        if not run.trials:
            continue
        key = (run.strategy, run.trials[0].task, run.trials[0].shape.key())
        grouped.setdefault(key, []).append(run)
    output: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key, candidates in grouped.items():
        best_run = max(
            candidates,
            key=lambda run: (run.best.speedup or 0.0) if run.best else 0.0,
        )
        output[key] = {
            "run": best_run,
            "confirmation": best_run.confirmation,
            "seed": best_run.seed,
        }
    return output


def assemble_summary() -> dict[str, Any]:
    """Combine every experiment JSON into ``results/summary.json``."""
    experiments: dict[str, Any] = {}
    for name in ("exp01_strategies", "exp02_ablation", "exp03_cost_model", "exp04_reliability"):
        path = RESULTS / f"{name}.json"
        if path.exists():
            experiments[name] = json.loads(path.read_text(encoding="utf-8"))
    trials = 0
    compiles_total = 0
    compile_time = 0.0
    wall = 0.0
    for name in ("exp01_strategies", "exp02_ablation"):
        payload = experiments.get(name)
        if not payload:
            continue
        for run in payload.get("runs", []):
            trials += int(run.get("trials", 0))
            compiles_total += int(run.get("compiles", 0))
            compile_time += float(run.get("compile_time_s", 0.0))
            wall += float(run.get("wall_s", 0.0))
    oracle_variants = int(experiments.get("exp03_cost_model", {}).get("oracle_variants", 0))
    summary = {
        "generated_at": now(),
        "git_sha": git_sha(),
        "environment": probe_environment(),
        "budgets": {
            "main_budget": experiments.get("exp01_strategies", {}).get("budget"),
            "main_seeds": experiments.get("exp01_strategies", {}).get("seeds"),
            "ablation_budget": experiments.get("exp02_ablation", {}).get("budget"),
            "ablation_seeds": experiments.get("exp02_ablation", {}).get("seeds"),
        },
        "totals": {
            "problems": experiments.get("exp01_strategies", {}).get("problems", 0),
            "search_trials": trials,
            "compiles": compiles_total,
            "compile_time_s": compile_time,
            "search_wall_s": wall,
            "oracle_variants": oracle_variants,
            "trials_including_oracle": trials + oracle_variants,
        },
        "experiments": experiments,
    }
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return summary


def load_summary() -> dict[str, Any]:
    """Read ``results/summary.json``."""
    path = RESULTS / "summary.json"
    if not path.exists():
        raise FileNotFoundError("results/summary.json is missing; run `make experiments` first")
    return json.loads(path.read_text(encoding="utf-8"))
