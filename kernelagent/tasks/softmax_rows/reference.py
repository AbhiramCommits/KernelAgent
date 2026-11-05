"""NumPy reference for softmax_rows."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def reference(dims: dict[str, int], dump_dir: Path) -> np.ndarray:
    """Stable row-wise softmax in float64, returned as float32."""
    rows, cols = dims["ROWS"], dims["COLS"]
    x = np.fromfile(dump_dir / "x.bin", dtype=np.float32).reshape(rows, cols).astype(np.float64)
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=1, keepdims=True)).astype(np.float32)
