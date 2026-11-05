"""NumPy reference for layernorm_fwd."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def reference(dims: dict[str, int], dump_dir: Path) -> np.ndarray:
    """Biased-variance layer norm with affine transform, float64 then float32."""
    rows, cols = dims["ROWS"], dims["COLS"]
    x = np.fromfile(dump_dir / "x.bin", dtype=np.float32).reshape(rows, cols).astype(np.float64)
    gamma = np.fromfile(dump_dir / "gamma.bin", dtype=np.float32).astype(np.float64)
    beta = np.fromfile(dump_dir / "beta.bin", dtype=np.float32).astype(np.float64)
    # Re-apply the kernel's deterministic affine pre-transform of the raw dumps.
    gamma = 1.0 + 0.1 * gamma
    beta = 0.05 * beta
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    out = (x - mean) / np.sqrt(var + 1e-5) * gamma + beta
    return out.astype(np.float32)
