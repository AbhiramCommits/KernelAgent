"""Search baselines, including an exhaustive oracle for small knob spaces."""

from __future__ import annotations

import time
from typing import Any

from .. import config
from ..agent.loop import run_search
from ..eval.harness import confirm_best, evaluate_variant
from ..ledger import TrialLedger
from ..tasks.registry import enumerate_variants
from ..types import KernelTask, SearchRun, Shape, Trial


def random_search(
    task: KernelTask,
    shape: Shape,
    budget: int = config.DEFAULT_BUDGET,
    seed: int = 0,
    ledger: TrialLedger | None = None,
) -> SearchRun:
    """Uniform random baseline."""
    return run_search(task, shape, strategy="random", budget=budget, seed=seed, ledger=ledger)


def greedy_search(
    task: KernelTask,
    shape: Shape,
    budget: int = config.DEFAULT_BUDGET,
    seed: int = 0,
    ledger: TrialLedger | None = None,
) -> SearchRun:
    """Greedy one-knob hill-climb baseline."""
    return run_search(task, shape, strategy="greedy", budget=budget, seed=seed, ledger=ledger)


def oracle_supported(task: KernelTask, max_variants: int = 256) -> bool:
    """Whether the full knob space is small enough to enumerate."""
    return task.knob_space_size() <= max_variants


def run_oracle(
    task: KernelTask,
    shape: Shape,
    ledger: TrialLedger | None = None,
    max_variants: int = 256,
) -> dict[str, Any]:
    """Evaluate every legal variant to give a true upper bound on speedup."""
    if not oracle_supported(task, max_variants):
        raise ValueError(
            f"{task.name} has {task.knob_space_size()} variants, above the oracle limit {max_variants}"
        )
    started = time.perf_counter()
    trials: list[Trial] = []
    for variant in enumerate_variants(task):
        trials.append(evaluate_variant(task, shape, variant, strategy="oracle", ledger=ledger))
    best = max(
        (trial for trial in trials if trial.check and trial.check.passed and trial.speedup),
        key=lambda trial: trial.speedup or 0.0,
        default=None,
    )
    confirmation = confirm_best(best, repeats=3) if best is not None else None
    return {
        "task": task.name,
        "shape": shape.key(),
        "variants": len(trials),
        "best": best,
        "oracle_speedup": confirmation.speedup if confirmation else (best.speedup if best else None),
        "oracle_raw_speedup": best.speedup if best else None,
        "oracle_spread_pct": confirmation.spread_pct if confirmation else None,
        "wall_s": time.perf_counter() - started,
    }


def oracle_gap(found_speedup: float | None, oracle_speedup: float | None) -> float | None:
    """Relative speedup gap to the oracle (0 is optimal)."""
    if not found_speedup or not oracle_speedup or oracle_speedup <= 0:
        return None
    return max(0.0, 1.0 - found_speedup / oracle_speedup)
