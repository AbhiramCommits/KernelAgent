"""Collect a measured ranking dataset for the learned selection layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import config
from ..agent.proposer import sample_variant
from ..env import probe_environment
from ..eval.harness import all_tasks, baseline_time, evaluate_variant
from ..ledger import git_sha
from ..tasks.registry import default_variant
from ..types import KernelTask, KernelVariant
from .features import feature_names, featurize


def candidate_variants(task: KernelTask, per_task: int, seed: int = 0) -> list[KernelVariant]:
    """Default + single-knob extremes + seeded random fill."""
    import random

    rng = random.Random(seed)
    tried: set[str] = set()
    variants: list[KernelVariant] = [default_variant(task)]
    tried.add(variants[0].key())
    knob_names = rng.sample(sorted(task.knobs), min(5, len(task.knobs)))
    for name in knob_names:
        spec = task.knobs[name]
        if all(isinstance(value, int) for value in spec.values):
            low, high = min(spec.values), max(spec.values)
        else:
            low, high = spec.values[0], spec.values[-1]
        for value in (low, high):
            candidate = KernelVariant(
                knobs={**task.default_knobs, name: value},
                provenance=f"collect:{name}",
            )
            if candidate.key() not in tried:
                variants.append(candidate)
                tried.add(candidate.key())
            if len(variants) >= per_task:
                return variants
    while len(variants) < per_task:
        variants.append(sample_variant(task, rng, tried, "collect:random"))
    return variants[:per_task]


def collect_dataset(
    per_task: int = 10,
    out: Path | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Measure candidates and write one JSON record per (task, shape, variant)."""
    target = Path(out) if out else config.RANK_DATASET
    target.parent.mkdir(parents=True, exist_ok=True)
    env = probe_environment()
    sha = git_sha()
    suite = all_tasks()
    records: list[dict[str, Any]] = []
    names = feature_names()

    for task in suite.values():
        variants = candidate_variants(task, per_task, seed=seed)
        for shape_name, shape in task.shapes.items():
            baseline_ms = baseline_time(task, shape)
            for variant in variants:
                trial = evaluate_variant(task, shape, variant, do_profile=False)
                features = featurize(task, shape, variant, trial.build).tolist()
                record = {
                    "task": task.name,
                    "shape": shape_name,
                    "shape_dims": dict(shape.dims),
                    "variant": variant.model_dump(mode="json"),
                    "features": features,
                    "feature_names": names,
                    "speedup": trial.speedup,
                    "ms_median": trial.run.ms_median if trial.run else None,
                    "baseline_ms": baseline_ms,
                    "build_ok": trial.build.ok,
                    "check_passed": bool(trial.check and trial.check.passed),
                    "compile_time_s": trial.build.compile_time_s,
                    "git_sha": sha,
                    "environment": {
                        "cpu_model": env["cpu_model"],
                        "compiler_version": env["compiler_version"],
                    },
                }
                records.append(record)
        if verbose:
            print(f"collected {len(records)} records ({task.name})")

    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    usable = sum(1 for record in records if record["speedup"] is not None)
    return {
        "path": str(target),
        "records": len(records),
        "usable": usable,
        "tasks": sorted(suite),
        "per_task": per_task,
    }
