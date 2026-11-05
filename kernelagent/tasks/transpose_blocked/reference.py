"""NumPy reference for transpose_blocked."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def reference(dims: dict[str, int], dump_dir: Path) -> np.ndarray:
    """Transpose the dumped matrix."""
    m, n = dims["M"], dims["N"]
    x = np.fromfile(dump_dir / "x.bin", dtype=np.float32).reshape(m, n)
    return np.ascontiguousarray(x.T)
