"""Library reference timings (NumPy / BLAS / SciPy) for the same problem data.

This is a separate column, never the baseline used to score the agent: it shows
how far a tuned C++ kernel is from a mature library implementation on the same
machine and inputs.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..types import KernelTask, Shape


def _median_ms(fn: Callable[[], Any], reps: int = 3) -> float:
    fn()
    times = []
    for _ in range(reps):
        started = time.perf_counter()
        fn()
        times.append((time.perf_counter() - started) * 1000.0)
    return float(np.median(times))


def _load(dump_dir: Path, name: str, dtype: type[np.floating[Any]] | type[np.integer[Any]]) -> np.ndarray:
    return np.fromfile(dump_dir / name, dtype=dtype)


def library_reference(task: KernelTask, shape: Shape, dump_dir: Path) -> dict[str, Any] | None:
    """Time an equivalent NumPy/SciPy operation on the dumped inputs.

    Returns ``None`` when a task has no library analogue.
    """
    try:
        return _build(task.name, shape, dump_dir)
    except (OSError, ValueError, MemoryError):
        return None


def _build(task_name: str, shape: Shape, dump_dir: Path) -> dict[str, Any] | None:
    dims = dict(shape.dims)

    if task_name == "sgemm_tiled":
        m, n, k = dims["M"], dims["N"], dims["K"]
        a = _load(dump_dir, "a.bin", np.float32).reshape(m, k)
        b = _load(dump_dir, "b.bin", np.float32).reshape(k, n)
        return _result(_median_ms(lambda: a @ b), "numpy.matmul (BLAS)")

    if task_name == "spmv_csr":
        from scipy.sparse import csr_matrix

        m, n = dims["M"], dims["N"]
        rowptr = _load(dump_dir, "rowptr.bin", np.int32)
        cols = _load(dump_dir, "cols.bin", np.int32)
        vals = _load(dump_dir, "vals.bin", np.float32)
        x = _load(dump_dir, "x.bin", np.float32)
        matrix = csr_matrix((vals, cols, rowptr), shape=(m, n))
        return _result(_median_ms(lambda: matrix.dot(x)), "scipy.sparse csr_matvec")

    if task_name == "softmax_rows":
        nrows, ncols = dims["ROWS"], dims["COLS"]
        x = _load(dump_dir, "x.bin", np.float32).reshape(nrows, ncols)

        def run() -> np.ndarray:
            shifted = x - x.max(axis=1, keepdims=True)
            e = np.exp(shifted)
            return e / e.sum(axis=1, keepdims=True)

        return _result(_median_ms(run), "numpy softmax")

    if task_name == "layernorm_fwd":
        nrows, ncols = dims["ROWS"], dims["COLS"]
        x = _load(dump_dir, "x.bin", np.float32).reshape(nrows, ncols)
        gamma = 1.0 + 0.1 * _load(dump_dir, "gamma.bin", np.float32)
        beta = 0.05 * _load(dump_dir, "beta.bin", np.float32)

        def run() -> np.ndarray:
            mean = x.mean(axis=1, keepdims=True)
            var = x.var(axis=1, keepdims=True)
            return (x - mean) / np.sqrt(var + 1e-5) * gamma + beta

        return _result(_median_ms(run), "numpy layernorm")

    if task_name == "conv1d_direct":
        c_in, c_out, length, kernel = dims["C_IN"], dims["C_OUT"], dims["L"], dims["K"]
        x = _load(dump_dir, "x.bin", np.float32).reshape(c_in, length)
        w = _load(dump_dir, "w.bin", np.float32).reshape(c_out, c_in, kernel)
        windows = np.lib.stride_tricks.sliding_window_view(x, kernel, axis=1)
        return _result(_median_ms(lambda: np.einsum("ilk,oik->ol", windows, w, optimize=True)), "numpy einsum conv1d")

    if task_name == "transpose_blocked":
        m, n = dims["M"], dims["N"]
        x = _load(dump_dir, "x.bin", np.float32).reshape(m, n)
        return _result(_median_ms(lambda: x.T.copy()), "numpy transpose")

    if task_name == "reduce_sum":
        m, n = dims["M"], dims["N"]
        x = _load(dump_dir, "x.bin", np.float32).reshape(m, n)
        return _result(_median_ms(lambda: x.sum(axis=1)), "numpy sum(axis=1)")

    if task_name == "bias_relu_fused":
        nrows, ncols = dims["ROWS"], dims["COLS"]
        x = _load(dump_dir, "x.bin", np.float32).reshape(nrows, ncols)
        bias = _load(dump_dir, "bias.bin", np.float32).reshape(nrows, 1)
        return _result(_median_ms(lambda: np.maximum(x + bias, 0.0)), "numpy bias+relu")

    return None


def _result(ms: float, name: str) -> dict[str, Any]:
    return {"ms_median": ms, "name": name, "numpy": np.__version__}
