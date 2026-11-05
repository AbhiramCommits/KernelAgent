"""Leave-one-task-out evaluation of the cost model and pairwise ranker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kendalltau, spearmanr

from .. import config
from .features import feature_names
from .model import CostModel, PairwiseRanker


def load_dataset(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Read measured records from the ranking dataset."""
    target = Path(path) if path else config.RANK_DATASET
    if not target.exists():
        raise FileNotFoundError(
            f"ranking dataset not found: {target}; run `kernelagent rank-eval --collect`"
        )
    records = []
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get("speedup") is not None:
                records.append(record)
    if not records:
        raise ValueError(f"no usable records in {target}")
    return records


def loto_folds(records: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]:
    """Split records into (held-out task, train, test) folds."""
    tasks = sorted({record["task"] for record in records})
    folds = []
    for task in tasks:
        train = [record for record in records if record["task"] != task]
        test = [record for record in records if record["task"] == task]
        if train and test:
            folds.append((task, train, test))
    return folds


def _matrix(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    expected = feature_names()
    rows = []
    targets = []
    groups = []
    for record in records:
        features = record.get("features")
        if features is None or len(features) != len(expected):
            continue
        rows.append(np.asarray(features, dtype=np.float64))
        targets.append(float(record["speedup"]))
        groups.append(f"{record['task']}|{record.get('shape', '')}")
    if not rows:
        return np.empty((0, len(expected))), np.empty((0,)), []
    return np.vstack(rows), np.asarray(targets), groups


def _group_metrics(
    actual: np.ndarray, scores: np.ndarray, predicted: np.ndarray | None = None
) -> dict[str, float]:
    if len(actual) < 2:
        return {}
    order = np.argsort(-scores)
    best = float(np.max(actual))
    picked = float(actual[order[0]])
    top1_regret = 1.0 - picked / best if best > 0 else 0.0

    gains = np.asarray(actual, dtype=np.float64)
    top = order[:5]
    discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
    dcg = float(np.sum((np.power(2.0, gains[top]) - 1.0) * discounts))
    ideal_order = np.argsort(-gains)[:5]
    ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal_order) + 2))
    idcg = float(np.sum((np.power(2.0, gains[ideal_order]) - 1.0) * ideal_discounts))
    ndcg = dcg / idcg if idcg > 0 else 1.0

    spearman = float(spearmanr(scores, actual).statistic) if np.std(scores) > 0 else 0.0
    kendall = float(kendalltau(scores, actual).statistic) if np.std(scores) > 0 else 0.0
    if not np.isfinite(spearman):
        spearman = 0.0
    if not np.isfinite(kendall):
        kendall = 0.0
    metrics = {
        "spearman": spearman,
        "kendall": kendall,
        "top1_regret": max(0.0, top1_regret),
        "ndcg_at_5": ndcg,
    }
    if predicted is not None:
        metrics["mae_log"] = float(
            np.mean(np.abs(np.log(np.clip(predicted, 1e-6, None)) - np.log(actual)))
        )
    return metrics


def run_loto_evaluation(dataset_path: str | Path | None = None) -> dict[str, Any]:
    """Fit on all-but-one task, rank the held-out task's measured pool."""
    records = load_dataset(dataset_path)
    folds = loto_folds(records)
    if not folds:
        raise ValueError("leave-one-task-out needs at least two tasks in the dataset")

    per_task: list[dict[str, Any]] = []
    cost_metrics: list[dict[str, float]] = []
    ranker_metrics: list[dict[str, float]] = []
    leakage = False

    for held_out, train_records, test_records in folds:
        train_tasks = {record["task"] for record in train_records}
        if held_out in train_tasks:  # defensive; this must never be true
            leakage = True
            continue
        x_train, y_train, train_groups = _matrix(train_records)
        x_test, y_test, test_groups = _matrix(test_records)
        if len(y_train) < 4 or len(y_test) < 2:
            continue

        cost = CostModel().fit(x_train, y_train)
        ranker_candidate = PairwiseRanker()
        ranker: PairwiseRanker | None
        try:
            ranker_candidate.fit(x_train, y_train, groups=train_groups)
            ranker = ranker_candidate
        except ValueError:
            ranker = None

        cost_predictions = cost.predict(x_test)
        ranker_scores = ranker.predict(x_test) if ranker is not None else np.zeros_like(y_test)

        buckets: dict[str, list[int]] = {}
        for index, group in enumerate(test_groups):
            buckets.setdefault(group, []).append(index)

        task_cost: list[dict[str, float]] = []
        task_ranker: list[dict[str, float]] = []
        for indices in buckets.values():
            if len(indices) < 2:
                continue
            group_idx = np.asarray(indices)
            task_cost.append(
                _group_metrics(
                    y_test[group_idx], cost_predictions[group_idx], cost_predictions[group_idx]
                )
            )
            if ranker is not None:
                task_ranker.append(_group_metrics(y_test[group_idx], ranker_scores[group_idx]))

        def mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
            if not items:
                return {}
            keys = sorted(items[0])
            return {key: float(np.mean([item[key] for item in items])) for key in keys}

        cost_fold = mean_metrics(task_cost)
        ranker_fold = mean_metrics(task_ranker)
        cost_metrics.append(cost_fold)
        if ranker_fold:
            ranker_metrics.append(ranker_fold)
        per_task.append(
            {
                "task": held_out,
                "groups": len(task_cost),
                "cost_model": cost_fold,
                "pairwise_ranker": ranker_fold,
            }
        )

    def macro(items: list[dict[str, float]]) -> dict[str, Any]:
        if not items:
            return {}
        keys = sorted(items[0])
        merged = {key: float(np.mean([item[key] for item in items])) for key in keys}
        merged["folds"] = len(items)
        return merged

    return {
        "dataset": str(Path(dataset_path) if dataset_path else config.RANK_DATASET),
        "records": len(records),
        "tasks": sorted({record["task"] for record in records}),
        "leakage": leakage,
        "cost_model": macro(cost_metrics),
        "pairwise_ranker": macro(ranker_metrics),
        "per_task": per_task,
    }
