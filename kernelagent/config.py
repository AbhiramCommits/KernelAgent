"""Global paths and tunable defaults for KernelAgent.

All paths are resolved relative to the repository root (the parent of the
``kernelagent`` package) so that the CLI behaves identically regardless of the
caller's working directory. Environment variables can override the cache and
results locations, which is useful in CI.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.environ.get("KA_CACHE_DIR", ROOT / ".kacache"))
RESULTS_DIR = Path(os.environ.get("KA_RESULTS_DIR", ROOT / "results"))
FIGURES_DIR = Path(os.environ.get("KA_FIGURES_DIR", ROOT / "figures"))
TASKS_DIR = Path(__file__).resolve().parent / "tasks"
TRANSCRIPTS_DIR = RESULTS_DIR / "transcripts"
BUILD_DIR = CACHE_DIR / "build"
RANK_DATASET = RESULTS_DIR / "rank_dataset.jsonl"

DEFAULT_WARMUP = 2
DEFAULT_REPS = 5
# The harness self-calibrates: it runs at least MIN_REPS and at most MAX_REPS,
# stopping when the timed loop exceeds TARGET_TIMED_SECONDS.
MIN_REPS = 1
MAX_REPS = 9
TARGET_TIMED_SECONDS = 0.08

DEFAULT_TIMEOUT_S = 120.0
COMPILE_TIMEOUT_S = 90.0
RUN_TIMEOUT_S = 120.0
CHECK_TIMEOUT_S = 120.0

# Search defaults (kept small so the CPU-only sweep stays tractable).
DEFAULT_BUDGET = 16
DEFAULT_PATIENCE = 8
DEFAULT_SEEDS = (0, 1, 2)

# Correctness gates.
DEFAULT_RTOL = 1e-4
DEFAULT_ATOL = 1e-4
SPEEDUP_SOLVE_THRESHOLD = 1.05
NOISE_RERUN_RATIO = 0.15  # MAD/median above this triggers one re-run
MAX_WALL_S = 300.0  # per-search wall-clock cap
MAX_REPAIR_ATTEMPTS = 2

# Compiler detection order. The first available compiler is the default.
COMPILER_CANDIDATES = ("clang++", "g++")

# knobs / flags that are safe to feed to a compiler
MAX_TOOL_OUTPUT_CHARS = 6000
