"""Environment probing: what this machine can actually do.

The result of :func:`probe_environment` is embedded in every ledger record so a
number can always be traced back to the machine, compiler and optional backends
that produced it.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import COMPILER_CANDIDATES


def _first_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _compiler_info() -> dict[str, Any]:
    for candidate in COMPILER_CANDIDATES:
        path = shutil.which(candidate)
        if not path:
            continue
        try:
            proc = subprocess.run(
                [path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            version = _first_line(proc.stdout or proc.stderr)
        except (OSError, subprocess.SubprocessError):
            version = "unknown"
        return {"name": candidate, "path": path, "version": version}
    return {"name": "none", "path": "", "version": "not found"}


def _sysctl(name: str) -> int | None:
    try:
        proc = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, timeout=5, check=False
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return int(proc.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return None


def _cache_sizes() -> dict[str, int]:
    sizes: dict[str, int] = {}
    for level, key in (("l1", "hw.l1dcachesize"), ("l2", "hw.l2cachesize"), ("l3", "hw.l3cachesize")):
        value = _sysctl(key)
        if value:
            sizes[level] = value
    if "l2" not in sizes:
        # Apple Silicon exposes the shared performance cache under this name.
        value = _sysctl("hw.perflevel0.l2cachesize")
        if value:
            sizes["l2"] = value
    return sizes


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _llm_provider() -> str | None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


@functools.lru_cache(maxsize=1)
def probe_environment() -> dict[str, Any]:
    """Return a JSON-serialisable capability table for this machine."""
    compiler = _compiler_info()
    nvcc_path = shutil.which("nvcc")
    nvcc_version = ""
    if nvcc_path:
        try:
            proc = subprocess.run(
                [nvcc_path, "--version"], capture_output=True, text=True, timeout=10, check=False
            )
            nvcc_version = _first_line(proc.stdout)
        except (OSError, subprocess.SubprocessError):
            nvcc_version = "unknown"
    return {
        "os": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_model": _cpu_model(),
        "cores": os.cpu_count() or 1,
        "memory_bytes": _memory_bytes(),
        "cache_bytes": _cache_sizes(),
        "compiler": compiler["name"],
        "compiler_path": compiler["path"],
        "compiler_version": compiler["version"],
        "has_nvcc": nvcc_path is not None,
        "nvcc_version": nvcc_version,
        "has_ncu": shutil.which("ncu") is not None,
        "has_triton": _module_available("triton"),
        "has_torch": _module_available("torch"),
        "llm_provider": _llm_provider(),
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def _cpu_model() -> str:
    if platform.system() == "Darwin":
        try:
            proc = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if proc.stdout.strip():
                return proc.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        text = Path("/proc/cpuinfo").read_text()
        for line in text.splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.machine()


def _memory_bytes() -> int:
    if platform.system() == "Darwin":
        value = _sysctl("hw.memsize")
        if value:
            return value
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return 0


def fingerprint() -> str:
    """Short stable hash-ish fingerprint of the machine + compiler."""
    env = probe_environment()
    raw = json.dumps(
        {
            "cpu": env["cpu_model"],
            "cores": env["cores"],
            "os": env["os"],
            "compiler": env["compiler_version"],
        },
        sort_keys=True,
    )
    import hashlib

    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def capability_rows() -> list[tuple[str, str, str]]:
    """Rows for the ``doctor`` table: (capability, value, status)."""
    env = probe_environment()
    rows = [
        ("OS", str(env["os"]), "ok"),
        ("CPU", f"{env['cpu_model']} ({env['cores']} cores)", "ok"),
        ("Memory", f"{int(env['memory_bytes']) / 2**30:.1f} GiB", "ok"),
        ("Compiler", f"{env['compiler']} - {env['compiler_version']}", "ok"),
        ("NVCC", str(env.get("nvcc_version") or "not found"), "ok" if env["has_nvcc"] else "skip"),
        ("Nsight Compute", "ncu" if env["has_ncu"] else "not found", "ok" if env["has_ncu"] else "skip"),
        ("Triton", "installed" if env["has_triton"] else "not installed", "ok" if env["has_triton"] else "skip"),
        ("PyTorch", "installed" if env["has_torch"] else "not installed", "ok" if env["has_torch"] else "skip"),
        (
            "LLM provider",
            str(env.get("llm_provider") or "no API key"),
            "ok" if env.get("llm_provider") else "skip",
        ),
    ]
    return rows
