"""Numeric featurisation of a (task, shape, variant, build) point.

The schema is global across the whole task suite so leave-one-task-out models
see the same feature dimension for every task. Build-dependent features are
zero when no build exists (the model-guided selector ranks *unbuilt* candidates
with the same featuriser, so it never cheats by reading remarks it has not
measured).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import numpy as np

from ..config import CANDIDATE_FLAGS
from ..env import probe_environment
from ..tasks.registry import load_tasks
from ..types import BuildResult, KernelTask, KernelVariant, Shape
from ..workload import bytes_for, flops_for


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


@lru_cache(maxsize=1)
def _schema() -> tuple[list[str], list[str], list[str]]:
    tasks = load_tasks()
    knob_names = sorted({name for task in tasks.values() for name in task.knobs})
    dim_names = sorted({name for task in tasks.values() for shape in task.shapes.values() for name in shape.dims})
    flags = [_sanitize(flag) for flag in CANDIDATE_FLAGS]
    return knob_names, dim_names, flags


@lru_cache(maxsize=1)
def feature_names() -> list[str]:
    """Global, stable ordered feature names."""
    knob_names, dim_names, flags = _schema()
    names: list[str] = []
    for name in knob_names:
        names.extend(
            [f"knob_{name}_present", f"knob_{name}_norm", f"knob_{name}_is_min", f"knob_{name}_is_max"]
        )
    for flag in flags:
        names.append(f"flag_{flag}")
    names.append("n_cflags")
    names.extend(
        [
            "diag_available",
            "vectorized_loops",
            "missed_count",
            "mean_vector_width",
            "unroll_full",
            "unroll_partial",
            "unroll_not",
            "inlined",
            "not_inlined",
            "compile_ok",
            "compile_time_s",
            "registers",
            "spills",
            "shared_bytes",
        ]
    )
    names.extend(
        [
            "log2_total_elements",
            "log2_flops",
            "log2_bytes",
            "ai_log",
            "working_set_over_cache",
            "log2_working_set",
            "num_dims",
        ]
    )
    for name in dim_names:
        names.extend([f"dim_{name}_present", f"dim_{name}_log2"])
    names.extend(["ratio_M_over_N", "ratio_K_over_N", "ratio_L_over_K", "ratio_CIN_over_COUT"])
    return names


def _largest_cache() -> float:
    caches = probe_environment().get("cache_bytes", {}) or {}
    return float(max(caches.values())) if caches else 0.0


def featurize(
    task: KernelTask,
    shape: Shape,
    variant: KernelVariant,
    build: BuildResult | None,
) -> np.ndarray:
    """Return the numeric feature vector for one candidate."""
    knob_names, dim_names, flags = _schema()
    values: list[float] = []

    for name in knob_names:
        spec = task.knobs.get(name)
        if spec is None:
            values.extend([0.0, 0.0, 0.0, 0.0])
            continue
        current = variant.knobs.get(name)
        numeric = [float(v) for v in spec.values if isinstance(v, (int, float))]
        low, high = (min(numeric), max(numeric)) if numeric else (0.0, 0.0)
        number = float(current) if isinstance(current, (int, float)) else 0.0
        norm = (number - low) / (high - low) if high > low else 0.0
        values.extend(
            [
                1.0,
                norm,
                1.0 if high == low or number == low else 0.0,
                1.0 if high == low or number == high else 0.0,
            ]
        )

    for flag in CANDIDATE_FLAGS:
        values.append(1.0 if flag in variant.cflags else 0.0)
    values.append(float(len(variant.cflags)))

    diagnostics: dict[str, Any] = build.diagnostics if build is not None else {}
    available = 1.0 if (build is not None and build.diagnostics.get("source") == "opt-record") else 0.0
    widths = diagnostics.get("vector_widths") or []
    unroll = diagnostics.get("unroll", {}) if isinstance(diagnostics.get("unroll"), dict) else {}
    values.extend(
        [
            available,
            float(diagnostics.get("vectorized_loops", 0) or 0),
            float(diagnostics.get("missed_count", 0) or 0),
            float(np.mean(widths)) if widths else 0.0,
            float(unroll.get("fully", 0) or 0),
            float(unroll.get("partially", 0) or 0),
            float(unroll.get("not", 0) or 0),
            float(diagnostics.get("inlined", 0) or 0),
            float(diagnostics.get("not_inlined", 0) or 0),
            1.0 if (build is not None and build.ok) else 0.0,
            float(build.compile_time_s) if build is not None else 0.0,
            float(diagnostics.get("registers", 0) or 0),
            float(diagnostics.get("spills", 0) or 0),
            float(diagnostics.get("shared_bytes", 0) or 0),
        ]
    )

    flops = flops_for(task, shape)
    nbytes = bytes_for(task, shape)
    dims = shape.dims
    total = 1.0
    for value in dims.values():
        total *= float(value)
    cache = _largest_cache()
    values.extend(
        [
            np.log2(1.0 + total),
            np.log2(1.0 + flops),
            np.log2(1.0 + nbytes),
            np.log2(1.0 + flops / max(nbytes, 1.0)),
            (nbytes / cache) if cache else 0.0,
            np.log2(1.0 + nbytes),
            float(len(dims)),
        ]
    )
    for name in dim_names:
        present = name in dims
        values.extend([1.0 if present else 0.0, np.log2(1.0 + dims.get(name, 0))])

    def ratio(a: str, b: str) -> float:
        if a in dims and b in dims and dims[b] > 0:
            return float(dims[a]) / float(dims[b])
        return 0.0

    values.extend(
        [ratio("M", "N"), ratio("K", "N"), ratio("L", "K"), ratio("C_IN", "C_OUT")]
    )
    vector = np.asarray(values, dtype=np.float32)
    if vector.size != len(feature_names()):
        raise AssertionError(f"feature size {vector.size} != names {len(feature_names())}")
    return vector


def featurize_many(
    task: KernelTask,
    shape: Shape,
    variants: list[KernelVariant],
    builds: list[BuildResult | None] | None = None,
) -> np.ndarray:
    """Stack feature vectors for a candidate pool."""
    if builds is None:
        builds = [None] * len(variants)
    return np.vstack([featurize(task, shape, v, b) for v, b in zip(variants, builds, strict=False)])
