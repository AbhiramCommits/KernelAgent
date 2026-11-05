"""C++17 kernel builder with a content-hash cache and real compiler remarks."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .. import config
from ..types import BuildResult, KernelTask, KernelVariant
from .base import DiskCache, content_hash, run_subprocess
from .diagnostics import parse_opt_record

# Flags an agent is allowed to add. Anything else is rejected rather than
# handed to the compiler, so a hallucinated flag cannot redirect the output or
# execute an arbitrary program.
ALLOWED_CFLAGS = {
    "-ffast-math",
    "-fno-math-errno",
    "-fno-trapping-math",
    "-funroll-loops",
    "-fomit-frame-pointer",
    "-fvectorize",
    "-fslp-vectorize",
    "-fno-vectorize",
    "-fno-slp-vectorize",
    "-march=native",
    "-mtune=native",
    "-O2",
    "-O3",
    "-Ofast",
}

_CLANG_REMARK_FLAGS = [
    "-fsave-optimization-record",
    "-Rpass=loop-vectorize",
    "-Rpass-missed=loop-vectorize",
    "-Rpass-analysis=loop-vectorize",
]


def find_compiler(preferred: str | None = None) -> str:
    """Locate a C++ compiler, preferring clang++ then g++."""
    if preferred:
        path = shutil.which(preferred) or preferred
        if Path(path).exists():
            return path
        raise FileNotFoundError(f"compiler not found: {preferred}")
    for candidate in config.COMPILER_CANDIDATES:
        found = shutil.which(candidate)
        if found:
            return found
    raise FileNotFoundError("no C++ compiler found (looked for clang++, g++)")


def compiler_version(compiler: str) -> str:
    """Version banner of the compiler, used in cache keys and the ledger."""
    result = run_subprocess([compiler, "--version"], timeout_s=10, retries=0)
    for line in result.stdout.splitlines():
        if line.strip():
            return line.strip()
    return "unknown"


def _compile_errors(stderr: str) -> list[str]:
    errors = [line.strip() for line in stderr.splitlines() if "error:" in line]
    return errors[:12]


class CxxBuilder:
    """Compiles one parameterised C++ kernel variant."""

    def __init__(
        self,
        compiler: str | None = None,
        cache_dir: Path | None = None,
        extra_flags: list[str] | None = None,
    ) -> None:
        self.compiler = find_compiler(compiler)
        self.version = compiler_version(self.compiler)
        self.is_clang = "clang" in Path(self.compiler).name.lower() or "clang" in self.version.lower()
        self.cache = DiskCache(cache_dir or config.BUILD_DIR)
        self.extra_flags = list(extra_flags or [])
        self._hash_note = ""

    def base_flags(self) -> list[str]:
        """Portable optimisation flags shared by every variant."""
        flags = ["-std=c++17", "-O3", "-DNDEBUG", "-fno-math-errno", "-Wno-unknown-pragmas"]
        flags.extend(self.extra_flags)
        return flags

    def build(self, task: KernelTask, variant: KernelVariant, force: bool = False) -> BuildResult:
        """Compile ``task`` with ``variant`` knobs, or return a cached binary."""
        try:
            self._validate_cflags(variant.cflags)
        except ValueError as exc:
            return BuildResult(ok=False, error=str(exc), raw_stderr=str(exc))

        task_dir = Path(task.directory)
        source = task_dir / task.kernel_file
        harness = task_dir.parent / "harness.hpp"
        source_text = source.read_text(encoding="utf-8")
        harness_text = harness.read_text(encoding="utf-8") if harness.exists() else ""
        define_flags = variant.define_flags()
        flags = (
            self.base_flags()
            + define_flags
            + list(variant.cflags)
            + (self._remark_flags() if self.is_clang else [])
        )
        key = content_hash(
            [
                self.compiler,
                self.version,
                task.name,
                source_text,
                harness_text,
                summary := " ".join(flags),
            ]
        )
        self._hash_note = summary
        entry = self.cache.entry(key)
        binary = entry / "kernel"
        meta_path = entry / "build.json"
        opt_path = entry / "opt.yaml"

        if not force and binary.exists() and meta_path.exists():
            cached_meta = self.cache.read_json(meta_path)
            if cached_meta:
                cached_diagnostics = cached_meta.get("diagnostics", {})
                return BuildResult(
                    ok=bool(cached_meta.get("ok", True)),
                    compile_time_s=float(cached_meta.get("compile_time_s", 0.0)),
                    binary=str(binary),
                    diagnostics=cached_diagnostics,
                    raw_stderr=str(cached_meta.get("raw_stderr", "")),
                    cached=True,
                    source_hash=key,
                    command=list(cached_meta.get("command", [])),
                )

        cmd = (
            [self.compiler]
            + self.base_flags()
            + define_flags
            + list(variant.cflags)
        )
        if self.is_clang:
            cmd += self._remark_flags(opt_path)
        cmd += [str(source), "-o", str(binary)]

        started = time.perf_counter()
        result = run_subprocess(
            cmd,
            timeout_s=config.COMPILE_TIMEOUT_S,
            retries=0,
            cwd=task_dir,
        )
        compile_time = time.perf_counter() - started

        diagnostics: dict[str, Any] = {}
        if self.is_clang and opt_path.exists():
            try:
                diagnostics = parse_opt_record(
                    opt_path, interest_files={source.name, harness.name}
                )
            except Exception as exc:  # defensive: a bad record must not hide a good build
                diagnostics = {"source": "opt-record", "parse_error": str(exc)}
        errors = _compile_errors(result.stderr)

        build = BuildResult(
            ok=result.ok and binary.exists(),
            compile_time_s=round(compile_time, 6),
            binary=str(binary) if (result.ok and binary.exists()) else None,
            diagnostics=diagnostics,
            raw_stderr=result.stderr,
            cached=False,
            source_hash=key,
            command=cmd,
            error=None if (result.ok and binary.exists()) else (result.error or "compilation failed"),
        )
        if errors:
            build.diagnostics["compile_errors"] = errors
        self.cache.write_json(
            meta_path,
            {
                "ok": build.ok,
                "compile_time_s": build.compile_time_s,
                "diagnostics": build.diagnostics,
                "raw_stderr": build.raw_stderr,
                "command": cmd,
                "variant": variant.model_dump(mode="json"),
            },
        )
        return build

    def _remark_flags(self, opt_path: Path | None = None) -> list[str]:
        flags = list(_CLANG_REMARK_FLAGS)
        if opt_path is not None:
            flags.insert(1, f"-foptimization-record-file={opt_path}")
        return flags

    @staticmethod
    def _validate_cflags(cflags: list[str]) -> None:
        unknown = [flag for flag in cflags if flag not in ALLOWED_CFLAGS]
        if unknown:
            raise ValueError(
                f"cflags not allowed: {unknown}; allowed={sorted(ALLOWED_CFLAGS)}"
            )


def build_variant(task: KernelTask, variant: KernelVariant, force: bool = False) -> BuildResult:
    """Convenience wrapper used by the CLI and the evaluation harness."""
    return CxxBuilder().build(task, variant, force=force)


def build_result_json(result: BuildResult) -> str:
    """Compact JSON rendering for the CLI."""
    return json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True)
