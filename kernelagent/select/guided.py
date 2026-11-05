"""Model-guided search: rank a large unbuilt candidate pool, compile top-k only.

The cost model is trained on measured candidates from *other* tasks (leave the
current task out) plus any trials already measured in this episode. Candidates
are featurised without build diagnostics, because those remarks do not exist
until a candidate has been compiled -- exactly the information budget the guided
strategy is trying to save.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from .. import config
from ..agent.proposer import sample_variant
from ..tasks.registry import default_variant, enumerate_variants, load_tasks, variant_from_knobs
from ..types import KernelTask, KernelVariant, Shape, Trial
from .features import featurize, featurize_many
from .model import CostModel


class ModelGuidedSearch:
    """Generate, score and rank a candidate pool with a learned cost model."""

    def __init__(
        self,
        seed: int = 0,
        pool_size: int = 512,
        dataset_path: Path | None = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.pool_size = pool_size
        self.dataset_path = Path(dataset_path) if dataset_path else config.RANK_DATASET
        self.last_model: CostModel | None = None

    def training_data(self, exclude_task: str) -> tuple[np.ndarray, np.ndarray]:
        """Static features and speedups from recorded candidates of other tasks."""
        rows: list[np.ndarray] = []
        targets: list[float] = []
        if not self.dataset_path.exists():
            return np.empty((0, 0)), np.empty((0,))
        tasks = load_tasks()
        import json

        with self.dataset_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("task") == exclude_task or record.get("speedup") is None:
                    continue
                task = tasks.get(record.get("task", ""))
                if task is None or "shape_dims" not in record:
                    continue
                shape = Shape(dims=record["shape_dims"])
                try:
                    variant = KernelVariant.model_validate(record["variant"])
                except Exception:
                    continue
                rows.append(featurize(task, shape, variant, None))
                targets.append(float(record["speedup"]))
        if not rows:
            return np.empty((0, 0)), np.empty((0,))
        return np.vstack(rows), np.asarray(targets)

    def fit(self, task: KernelTask, shape: Shape, history: list[Trial]) -> CostModel | None:
        """Fit on other tasks plus this episode's measured trials."""
        x_base, y_base = self.training_data(task.name)
        history_rows = []
        history_targets = []
        for trial in history:
            if trial.check is None or not trial.check.passed or trial.speedup is None:
                continue
            history_rows.append(featurize(task, shape, trial.variant, None))
            history_targets.append(trial.speedup)
        if history_rows:
            x_hist = np.vstack(history_rows)
            if x_base.size:
                x = np.vstack([x_base, x_hist])
                y = np.concatenate([y_base, np.asarray(history_targets)])
            else:
                x, y = x_hist, np.asarray(history_targets)
        else:
            x, y = x_base, y_base
        if len(y) < 4:
            return None
        self.last_model = CostModel().fit(x, y)
        return self.last_model

    def candidate_pool(self, task: KernelTask, shape: Shape) -> list[KernelVariant]:
        """Every variant when the space is small, otherwise a seeded pool."""
        space = task.knob_space_size()
        if space <= self.pool_size:
            return list(enumerate_variants(task))
        pool: list[KernelVariant] = [default_variant(task)]
        tried = {pool[0].key()}
        for name in sorted(task.knobs):
            spec = task.knobs[name]
            if all(isinstance(value, int) for value in spec.values):
                low, high = min(spec.values), max(spec.values)
            else:
                low, high = spec.values[0], spec.values[-1]
            for value in (low, high):
                candidate = variant_from_knobs(
                    task, {**task.default_knobs, name: value}, [], "guided:extreme"
                )
                if candidate.key() not in tried:
                    pool.append(candidate)
                    tried.add(candidate.key())
        while len(pool) < self.pool_size:
            pool.append(sample_variant(task, self.rng, tried, "guided:pool"))
        return pool

    def rank_pool(
        self, task: KernelTask, shape: Shape, history: list[Trial], exclude: set[str] | None = None
    ) -> list[KernelVariant]:
        """Return candidates ordered best-first by the fitted cost model."""
        pool = [variant for variant in self.candidate_pool(task, shape) if variant.key() not in (exclude or set())]
        if not pool:
            return []
        model = self.fit(task, shape, history)
        if model is None:
            return self._heuristic_order(task, pool)
        matrix = featurize_many(task, shape, pool, None)
        order = model.rank(matrix)
        return [pool[index] for index in order]

    def _heuristic_order(self, task: KernelTask, pool: list[KernelVariant]) -> list[KernelVariant]:
        default = default_variant(task)

        def distance(variant: KernelVariant) -> tuple[int, str]:
            changes = sum(1 for key in task.knobs if variant.knobs[key] != default.knobs[key])
            return (changes, variant.key())

        return sorted(pool, key=distance)


class GuidedProposer:
    """Proposer adapter: repeatedly take the next untried top-ranked candidate."""

    name = "guided"

    def __init__(
        self,
        seed: int = 0,
        pool_size: int = 512,
        dataset_path: Path | None = None,
    ) -> None:
        self.search = ModelGuidedSearch(seed=seed, pool_size=pool_size, dataset_path=dataset_path)
        self.tried: set[str] = set()
        self.rng = random.Random(seed)

    def propose(self, task, shape, history, feedback):  # type: ignore[no-untyped-def]
        ranked = self.search.rank_pool(task, shape, history, exclude=self.tried)
        for variant in ranked:
            if variant.key() not in self.tried:
                self.tried.add(variant.key())
                return [variant]
        return [sample_variant(task, self.rng, self.tried, "guided:fallback")]


def compiles_to_best(search_rank: int | None) -> int | None:
    """Compiles needed to reach the best candidate, from a 1-based rank."""
    return search_rank
