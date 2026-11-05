"""NumPy reference for spmv_csr, rebuilt from the dumped CSR arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def reference(dims: dict[str, int], dump_dir: Path) -> np.ndarray:
    """Compute A x with float64 accumulation from the dumped CSR data."""
    m = dims["M"]
    rowptr = np.fromfile(dump_dir / "rowptr.bin", dtype=np.int32).astype(np.int64)
    cols = np.fromfile(dump_dir / "cols.bin", dtype=np.int32).astype(np.int64)
    vals = np.fromfile(dump_dir / "vals.bin", dtype=np.float32).astype(np.float64)
    x = np.fromfile(dump_dir / "x.bin", dtype=np.float32).astype(np.float64)
    counts = np.diff(rowptr)
    rows = np.repeat(np.arange(m, dtype=np.int64), counts)
    y = np.zeros(m, dtype=np.float64)
    np.add.at(y, rows, vals * x[cols])
    return y.astype(np.float32)
