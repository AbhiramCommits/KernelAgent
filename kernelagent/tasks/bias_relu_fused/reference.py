"""NumPy reference for bias_relu_fused."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def reference(dims: dict[str, int], dump_dir: Path) -> np.ndarray:
    """Broadcast the row bias, add, and clamp at zero."""
    rows, cols = dims["ROWS"], dims["COLS"]
    x = np.fromfile(dump_dir / "x.bin", dtype=np.float32).reshape(rows, cols)
    bias = np.fromfile(dump_dir / "bias.bin", dtype=np.float32).reshape(rows, 1)
    return np.maximum(x + bias, np.float32(0.0)).astype(np.float32)
