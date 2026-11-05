"""NumPy reference for conv1d_direct."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def reference(dims: dict[str, int], dump_dir: Path) -> np.ndarray:
    """Valid cross-correlation computed with a strided-view einsum."""
    c_in, c_out, length, kernel = dims["C_IN"], dims["C_OUT"], dims["L"], dims["K"]
    x = np.fromfile(dump_dir / "x.bin", dtype=np.float32).reshape(c_in, length)
    w = np.fromfile(dump_dir / "w.bin", dtype=np.float32).reshape(c_out, c_in, kernel)
    windows = np.lib.stride_tricks.sliding_window_view(x, kernel, axis=1).astype(np.float64)
    out = np.einsum("ilk,oik->ol", windows, w.astype(np.float64), optimize=True)
    return np.ascontiguousarray(out).astype(np.float32)
