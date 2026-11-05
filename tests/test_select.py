"""Selection-layer tests: features, models, LOTO leakage, evolution, guided."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from kernelagent.select.baselines import oracle_gap, oracle_supported
from kernelagent.select.evaluate import loto_folds, run_loto_evaluation
from kernelagent.select.evolution import EvolutionProposer
from kernelagent.select.features import feature_names, featurize
from kernelagent.select.guided import GuidedProposer, ModelGuidedSearch
from kernelagent.select.model import CostModel, PairwiseRanker
from kernelagent.tasks.registry import load_tasks, variant_from_knobs
from kernelagent.types import BuildResult, CheckResult, KernelVariant, RunResult, Trial


def _toy_trial(variant: KernelVariant, speedup: float) -> Trial:
    return Trial(
        trial_id=f"t-{variant.key()}",
        task="softmax_rows",
        shape=load_tasks()["softmax_rows"].shapes["small"],
        variant=variant,
        build=BuildResult(ok=True),
        run=RunResult(ms_median=1.0, ms_mad=0.0, ms_min=1.0, reps=3),
        check=CheckResult(passed=True),
        speedup=speedup,
    )


def test_featurizer_shape_and_determinism() -> None:
    task = load_tasks()["sgemm_tiled"]
    shape = task.shapes["small"]
    variant = variant_from_knobs(task, dict(task.default_knobs), ["-funroll-loops"], "test")
    build = BuildResult(
        ok=True,
        diagnostics={
            "source": "opt-record",
            "vectorized_loops": 3,
            "vector_widths": [4, 4],
            "missed_count": 2,
            "unroll": {"fully": 1, "partially": 2, "not": 0},
            "inlined": 4,
        },
    )
    first = featurize(task, shape, variant, build)
    second = featurize(task, shape, variant, build)
    assert first.shape == (len(feature_names()),)
    assert np.array_equal(first, second)
    assert np.all(np.isfinite(first))
    without_build = featurize(task, shape, variant, None)
    assert without_build.shape == first.shape
    assert not np.array_equal(first, without_build)
    other = variant_from_knobs(
        task, {**task.default_knobs, "TILE_M": 64}, [], "test"
    )
    assert not np.array_equal(featurize(task, shape, other, build), first)


def test_cost_model_beats_shuffled_control() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 8))
    y = np.exp(0.8 * x[:, 0] - 0.5 * x[:, 1])
    model = CostModel().fit(x, y)
    predicted = model.predict(x)
    assert spearmanr(predicted, y).statistic > 0.6

    shuffled = CostModel().fit(x, rng.permutation(y))
    control = spearmanr(shuffled.predict(x), y).statistic
    assert control < 0.3


def test_pairwise_ranker_orders_groups() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(120, 6))
    y = np.exp(x[:, 0] + 0.2 * rng.normal(size=120))
    groups = ["a"] * 40 + ["b"] * 40 + ["c"] * 40
    ranker = PairwiseRanker().fit(x, y, groups=groups)
    assert spearmanr(ranker.predict(x), y).statistic > 0.5
    order = ranker.rank(x[:40])
    assert len(order) == 40


def test_model_save_load(tmp_path: Path) -> None:
    x = np.random.default_rng(2).normal(size=(80, 4))
    y = np.exp(x[:, 0])
    model = CostModel().fit(x, y)
    model.save(tmp_path / "cost.joblib")
    restored = CostModel.load(tmp_path / "cost.joblib")
    assert np.allclose(restored.predict(x), model.predict(x))

    ranker = PairwiseRanker().fit(x, y)
    ranker.save(tmp_path / "ranker.joblib")
    restored_ranker = PairwiseRanker.load(tmp_path / "ranker.joblib")
    assert np.allclose(restored_ranker.predict(x), ranker.predict(x))


def test_evolution_never_duplicates_measured_variants() -> None:
    task = load_tasks()["softmax_rows"]
    shape = task.shapes["small"]
    proposer = EvolutionProposer(seed=0, population_size=12)
    history: list[Trial] = []
    for index in range(8):
        knobs = {name: spec.values[index % len(spec.values)] for name, spec in task.knobs.items()}
        variant = variant_from_knobs(task, knobs, [], "seed")
        history.append(_toy_trial(variant, 1.0 + 0.1 * index))
    known = {trial.variant.key() for trial in history}
    produced: set[str] = set()
    for _ in range(12):
        proposals = proposer.propose(task, shape, history, "")
        assert len(proposals) == 1
        key = proposals[0].key()
        assert key not in known
        assert key not in produced
        produced.add(key)
        history.append(_toy_trial(proposals[0], 1.0 + 0.05 * len(history)))


def test_loto_split_has_no_leakage(tmp_path: Path) -> None:
    names = feature_names()
    rng = np.random.default_rng(3)
    records = []
    for task_name in ("task_a", "task_b", "task_c"):
        for index in range(12):
            features = rng.normal(size=len(names)).astype(float)
            features[0] = index / 12.0
            records.append(
                {
                    "task": task_name,
                    "shape": "small",
                    "features": features.tolist(),
                    "speedup": float(np.exp(features[0])),
                }
            )
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    folds = loto_folds(records)
    for held_out, train, test in folds:
        assert held_out not in {record["task"] for record in train}
        assert {record["task"] for record in test} == {held_out}
    metrics = run_loto_evaluation(path)
    assert metrics["leakage"] is False
    assert metrics["records"] == len(records)
    assert metrics["cost_model"]["folds"] == 3
    assert "kendall" in metrics["pairwise_ranker"]


def test_guided_pool_and_unique_proposals() -> None:
    task = load_tasks()["softmax_rows"]
    shape = task.shapes["small"]
    search = ModelGuidedSearch(seed=0, pool_size=64, dataset_path=Path("/nonexistent.jsonl"))
    pool = search.candidate_pool(task, shape)
    assert 0 < len(pool) <= 64
    assert len({variant.key() for variant in pool}) == len(pool)
    assert task.default_variant().key() in {variant.key() for variant in pool}
    proposer = GuidedProposer(seed=0, pool_size=64, dataset_path=Path("/nonexistent.jsonl"))
    keys = set()
    for _ in range(10):
        variant = proposer.propose(task, shape, [], "")[0]
        assert variant.key() not in keys
        keys.add(variant.key())


def test_oracle_gate_and_gap() -> None:
    tasks = load_tasks()
    assert oracle_supported(tasks["softmax_rows"])
    assert not oracle_supported(tasks["sgemm_tiled"])
    assert oracle_gap(2.0, 2.0) == 0.0
    assert oracle_gap(1.0, 2.0) == 0.5
    assert oracle_gap(None, 2.0) is None
