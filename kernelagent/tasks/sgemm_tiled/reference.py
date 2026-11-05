"""NumPy reference for sgemm_tiled.

The checker dumps the exact inputs the C++ kernel consumed and this module
recomputes the result independently with NumPy/BLAS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def reference(dims: dict[str, int], dump_dir: Path) -> np.ndarray:
    """Compute A @ B from the dumped inputs."""
    m, n, k = dims["M"], dims["N"], dims["K"]
    a = np.fromfile(dump_dir / "a.bin", dtype=np.float32).reshape(m, k)
    b = np.fromfile(dump_dir / "b.bin", dtype=np.float32).reshape(k, n)
    return a @ b
