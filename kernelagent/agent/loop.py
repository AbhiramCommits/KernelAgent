"""The agent loop: propose -> build (with repair) -> check -> time -> profile."""

from __future__ import annotations

import time
from collections.abc import Iterable

from .. import config
from ..eval.harness import (
    all_tasks,
    baseline_time,
    best_trial,
    confirm_best,
    evaluate_variant,
    get_shape,
)
from ..ledger import TrialLedger
from ..types import Confirmation, KernelTask, KernelVariant, SearchRun, Shape, Trial
from .feedback import DEFAULT_SIGNALS, format_feedback, repair_feedback
from .proposer import Proposer, make_proposer


def run_search(
    task: str | KernelTask,
    shape: str | Shape,
    strategy: str = "mutation",
    budget: int = config.DEFAULT_BUDGET,
    seed: int = 0,
    patience: int = config.DEFAULT_PATIENCE,
    replay: str | None = None,
    ledger: TrialLedger | None = None,
    proposer: Proposer | None = None,
    signals: Iterable[str] | None = None,
    wall_cap_s: float = config.MAX_WALL_S,
    confirm: bool = False,
) -> SearchRun:
    """Run one search episode and return every trial plus the best variant.

    The budget counts every evaluated variant, including failed builds and
    repairs. A variant that does not pass the correctness gate can never become
    ``best`` and never carries a speedup.
    """
    kernel_task = task if isinstance(task, KernelTask) else all_tasks()[task]
    kernel_shape = get_shape(kernel_task, shape)
    chosen: Proposer = proposer or make_proposer(strategy, seed=seed, replay=replay)
    signals_set = frozenset(signals) if signals is not None else DEFAULT_SIGNALS

    # Measure (or reuse) the baseline before any feedback is produced.
    baseline_ms = baseline_time(kernel_task, kernel_shape)

    started = time.perf_counter()
    trials: list[Trial] = []
    repair_attempts = 0
    stop_reason = "budget"
    feedback = "no trials yet; propose the first variant"
    best_value = -1.0
    non_improving = 0

    while len(trials) < budget:
        proposals = chosen.propose(kernel_task, kernel_shape, trials, feedback)
        if not proposals:
            stop_reason = "no_proposal"
            break
        variant: KernelVariant = proposals[0]

        trial = evaluate_variant(
            kernel_task,
            kernel_shape,
            variant,
            strategy=chosen.name,
            seed=seed,
            ledger=ledger,
            baseline_ms=baseline_ms,
        )
        trials.append(trial)

        # Repair loop: feed the real compiler error back to the proposer.
        attempts = 0
        while (
            not trial.build.ok
            and attempts < config.MAX_REPAIR_ATTEMPTS
            and len(trials) < budget
        ):
            repair_text = repair_feedback(trial)
            repairs = chosen.propose(kernel_task, kernel_shape, trials, repair_text)
            if not repairs:
                break
            attempts += 1
            repair_attempts += 1
            trial = evaluate_variant(
                kernel_task,
                kernel_shape,
                repairs[0],
                strategy=chosen.name,
                seed=seed,
                ledger=ledger,
                baseline_ms=baseline_ms,
            )
            trials.append(trial)

        score = trial.speedup if (trial.check and trial.check.passed and trial.speedup) else None
        if score is not None and score > best_value:
            best_value = score
            non_improving = 0
        else:
            non_improving += 1

        feedback = format_feedback(trial, signals_set)

        elapsed = time.perf_counter() - started
        if elapsed >= wall_cap_s:
            stop_reason = "wall_cap"
            break
        if non_improving >= patience:
            stop_reason = "patience"
            break
    else:
        stop_reason = "budget"

    best = best_trial(trials)
    confirmation: Confirmation | None = None
    if confirm and best is not None:
        try:
            confirmation = confirm_best(best)
            if not confirmation.check_passed:
                confirmation = None
        except Exception:
            confirmation = None

    return SearchRun(
        strategy=chosen.name,
        seed=seed,
        budget=budget,
        trials=trials,
        best=best,
        wall_s=time.perf_counter() - started,
        tokens=int(getattr(chosen, "tokens", 0)),
        repair_attempts=repair_attempts,
        stop_reason=stop_reason,
        confirmation=confirmation,
    )
