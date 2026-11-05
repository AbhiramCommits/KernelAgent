"""Optional GPU backends.

Both builders are fully implemented but import their dependencies lazily and
report unavailability through ``is_available()``. On a CPU-only machine they are
never constructed by the default path.
"""

from __future__ import annotations

import importlib.util
import shutil
import time
from pathlib import Path
from typing import Any

from .. import config
from ..types import BuildResult, KernelVariant
from .base import content_hash, run_subprocess
from .diagnostics import parse_ptxas_verbose, parse_triton_log

DEFAULT_CUDA_ARCH = "sm_80"


class CudaBuilder:
    """Compile a CUDA C++ source with ``nvcc -Xptxas -v``."""

    def __init__(self, arch: str | None = None, cache_dir: Path | None = None) -> None:
        self.nvcc = shutil.which("nvcc")
        self.arch = arch or DEFAULT_CUDA_ARCH
        self.cache_dir = Path(cache_dir) if cache_dir else config.CACHE_DIR / "cuda"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def is_available() -> bool:
        """True when nvcc is installed."""
        return shutil.which("nvcc") is not None

    def build(
        self,
        source: str | Path,
        knobs: dict[str, int | str] | None = None,
        cflags: list[str] | None = None,
    ) -> BuildResult:
        """Compile ``source`` and return ptxas register/spill diagnostics."""
        if not self.is_available():
            return BuildResult(ok=False, error="nvcc is not available")
        source_path = Path(source)
        if not source_path.exists():
            return BuildResult(ok=False, error=f"source not found: {source_path}")
        knobs = knobs or {}
        defines = [f"-D{k}={v}" for k, v in sorted(knobs.items())]
        flags = ["-std=c++17", "-O3", "-Xptxas", "-v", f"-arch={self.arch}"] + list(cflags or [])
        key = content_hash(
            [source_path.read_text(encoding="utf-8"), " ".join(defines + flags), self.arch]
        )
        binary = self.cache_dir / key[:2] / f"{key}.out"
        binary.parent.mkdir(parents=True, exist_ok=True)
        cmd = [self.nvcc or "nvcc", *flags, *defines, str(source_path), "-o", str(binary)]
        started = time.perf_counter()
        result = run_subprocess(cmd, timeout_s=config.COMPILE_TIMEOUT_S, retries=0)
        elapsed = time.perf_counter() - started
        diagnostics = parse_ptxas_verbose(result.stderr)
        return BuildResult(
            ok=result.ok and binary.exists(),
            compile_time_s=round(elapsed, 6),
            binary=str(binary) if result.ok else None,
            diagnostics=diagnostics,
            raw_stderr=result.stderr,
            source_hash=key,
            command=cmd,
            error=None if result.ok else (result.error or "nvcc failed"),
        )


class TritonBuilder:
    """Compile and benchmark a Triton matmul variant, reading ptxas metadata."""

    def __init__(self) -> None:
        self._compiled: Any = None

    @staticmethod
    def is_available() -> bool:
        """True when Triton is importable and a CUDA device is present."""
        if importlib.util.find_spec("triton") is None:
            return False
        if importlib.util.find_spec("torch") is None:
            return False
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:  # pragma: no cover - broken driver installs
            return False

    def _kernel(self) -> Any:
        import triton
        import triton.language as tl

        @triton.jit
        def matmul_kernel(  # type: ignore[no-untyped-def]
            a_ptr,
            b_ptr,
            c_ptr,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
        ):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_K)):
                k_remaining = K - k * BLOCK_K
                a = tl.load(
                    a_ptr + offs_m[:, None] * stride_am + (k * BLOCK_K + offs_k)[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
                    other=0.0,
                )
                b = tl.load(
                    b_ptr + (k * BLOCK_K + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
                    other=0.0,
                )
                acc += tl.dot(a, b)
            tl.store(
                c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
                acc,
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            )

        return matmul_kernel

    def build(
        self,
        variant: KernelVariant,
        m: int = 512,
        n: int = 512,
        k: int = 512,
        dtype: str = "float32",
    ) -> BuildResult:
        """Compile and benchmark one Triton matmul variant on the GPU."""
        if not self.is_available():
            return BuildResult(ok=False, error="Triton/CUDA not available")
        import torch
        import triton

        kernel = self._kernel()
        torch_dtype = torch.float32 if dtype == "float32" else torch.float16
        a = torch.randn((m, k), device="cuda", dtype=torch_dtype)
        b = torch.randn((k, n), device="cuda", dtype=torch_dtype)
        c = torch.empty((m, n), device="cuda", dtype=torch_dtype)
        block_m = int(variant.knobs.get("BLOCK_M", 64))
        block_n = int(variant.knobs.get("BLOCK_N", 64))
        block_k = int(variant.knobs.get("BLOCK_K", 32))
        num_warps = int(variant.knobs.get("NUM_WARPS", 4))
        num_stages = int(variant.knobs.get("NUM_STAGES", 3))
        grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
        try:
            compiled = kernel.warmup(
                a,
                b,
                c,
                m,
                n,
                k,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                grid=(1,),
                num_warps=num_warps,
                num_stages=num_stages,
            )
        except Exception as exc:  # pragma: no cover - requires GPU
            return BuildResult(ok=False, error=f"Triton compile failed: {exc}")
        try:
            compiled._init_handles()  # type: ignore[attr-defined]
        except Exception:
            pass
        self._compiled = compiled
        kernel[grid](
            a,
            b,
            c,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        torch.cuda.synchronize()
        timings = []
        for _ in range(10):
            torch.cuda.synchronize()
            started = time.perf_counter()
            kernel[grid](
                a,
                b,
                c,
                m,
                n,
                k,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - started) * 1000.0)
        timings.sort()
        registers = int(getattr(compiled, "n_regs", 0) or 0)
        spills = int(getattr(compiled, "n_spills", 0) or 0)
        metadata = getattr(compiled, "metadata", None)
        shared = int(getattr(metadata, "shared", 0) or 0)
        diagnostics = {
            "source": "triton",
            "registers": registers,
            "spills": spills,
            "shared_bytes": shared,
            "gpu_ms_median": timings[len(timings) // 2],
            "gpu_ms_min": timings[0],
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
        return BuildResult(
            ok=True,
            compile_time_s=0.0,
            binary=None,
            diagnostics=diagnostics,
            command=[],
        )

    @staticmethod
    def parse_log(text: str) -> dict[str, Any]:
        """Parse a captured Triton/ptxas log (used by fixture tests)."""
        return parse_triton_log(text)
