"""NumPy reference for reduce_sum."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def reference(dims: dict[str, int], dump_dir: Path) -> np.ndarray:
    """Row sums accumulated in float64 then cast back to float32."""
    m, n = dims["M"], dims["N"]
    x = np.fromfile(dump_dir / "x.bin", dtype=np.float32).reshape(m, n)
    return x.astype(np.float64).sum(axis=1).astype(np.float32)
