"""Cost regression and pairwise ranking models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from joblib import dump, load
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from ..types import KernelTask, KernelVariant, Shape, Trial
from .features import feature_names, featurize


class CostModel:
    """Gradient-boosted regression on log speedup."""

    def __init__(self, random_state: int = 0, **kwargs: Any) -> None:
        params = {
            "n_estimators": 150,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "random_state": random_state,
        }
        params.update(kwargs)
        self.model = GradientBoostingRegressor(**params)
        self.fitted = False

    def fit(self, features: np.ndarray, speedups: np.ndarray) -> CostModel:
        """Fit on measured speedups (target is log speedup)."""
        x = np.asarray(features, dtype=np.float64)
        y = np.log(np.clip(np.asarray(speedups, dtype=np.float64), 1e-6, None))
        self.model.fit(x, y)
        self.fitted = True
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predicted speedup for each row."""
        x = np.atleast_2d(np.asarray(features, dtype=np.float64))
        return np.exp(self.model.predict(x))

    def rank(self, features: np.ndarray) -> np.ndarray:
        """Candidate indices sorted by predicted speedup, best first."""
        return np.argsort(-self.predict(features))

    def save(self, path: str | Path) -> None:
        """Persist the fitted model."""
        dump({"model": self.model, "fitted": self.fitted}, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> CostModel:
        """Load a persisted model."""
        payload = load(Path(path))
        instance = cls()
        instance.model = payload["model"]
        instance.fitted = bool(payload.get("fitted", True))
        return instance


class PairwiseRanker:
    """Logistic model over feature differences from measured within-pool pairs."""

    def __init__(self, random_state: int = 0, max_pairs: int = 20000) -> None:
        self.model = LogisticRegression(max_iter=2000, C=1.0, random_state=random_state)
        self.max_pairs = max_pairs
        self.fitted = False

    @staticmethod
    def _pairs(
        features: np.ndarray, speedups: np.ndarray, groups: list[str] | None, max_pairs: int
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(speedups, dtype=np.float64)
        differences: list[np.ndarray] = []
        labels: list[int] = []
        if groups is None:
            group_indices = [list(range(len(y)))]
        else:
            buckets: dict[str, list[int]] = {}
            for index, group in enumerate(groups):
                buckets.setdefault(group, []).append(index)
            group_indices = list(buckets.values())
        for indices in group_indices:
            for offset, i in enumerate(indices):
                for j in indices[offset + 1 :]:
                    if y[i] == y[j]:
                        continue
                    high, low = (i, j) if y[i] > y[j] else (j, i)
                    difference = x[high] - x[low]
                    differences.append(difference)
                    labels.append(1)
                    differences.append(-difference)
                    labels.append(0)
                    if len(labels) >= max_pairs:
                        return np.vstack(differences), np.asarray(labels)
        if not differences:
            return np.empty((0, x.shape[1])), np.empty((0,), dtype=int)
        return np.vstack(differences), np.asarray(labels)

    def fit(
        self,
        features: np.ndarray,
        speedups: np.ndarray,
        groups: list[str] | None = None,
    ) -> PairwiseRanker:
        """Fit on within-group orderings."""
        x, y = self._pairs(features, speedups, groups, self.max_pairs)
        if len(y) < 2:
            raise ValueError("need at least one non-tied pair to fit a ranker")
        self.model.fit(x, y)
        self.fitted = True
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Pairwise score (higher is better)."""
        x = np.atleast_2d(np.asarray(features, dtype=np.float64))
        return self.model.decision_function(x)

    def rank(self, features: np.ndarray) -> np.ndarray:
        """Candidate indices sorted by score, best first."""
        return np.argsort(-self.predict(features))

    def save(self, path: str | Path) -> None:
        """Persist the fitted model."""
        dump({"model": self.model, "fitted": self.fitted}, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> PairwiseRanker:
        """Load a persisted model."""
        payload = load(Path(path))
        instance = cls()
        instance.model = payload["model"]
        instance.fitted = bool(payload.get("fitted", True))
        return instance


def training_arrays(trials: list[Trial], task: KernelTask, shape: Shape) -> tuple[np.ndarray, np.ndarray]:
    """Build (features, speedups) from correct trials only."""
    rows = []
    targets = []
    for trial in trials:
        if trial.check is None or not trial.check.passed or trial.speedup is None:
            continue
        rows.append(featurize(task, shape, trial.variant, trial.build))
        targets.append(trial.speedup)
    if not rows:
        return np.empty((0, len(feature_names()))), np.empty((0,))
    return np.vstack(rows), np.asarray(targets, dtype=np.float64)


def featurize_variant(
    task: KernelTask, shape: Shape, variant: KernelVariant, build: Any = None
) -> np.ndarray:
    """Convenience wrapper used by guided search."""
    return featurize(task, shape, variant, build)
