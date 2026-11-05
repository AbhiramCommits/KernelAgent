"""Optional-backend tests.

Availability is asserted here because this repository is built and validated on
a CPU-only machine; the GPU paths themselves are marked ``gpu`` and skipped when
no CUDA device is present.
"""

from __future__ import annotations

import pytest

from kernelagent.tools.gpu import CudaBuilder, TritonBuilder


def test_cuda_builder_reports_unavailable_or_builds() -> None:
    if CudaBuilder.is_available():
        pytest.skip("nvcc present; covered by the gpu-marked tests")
    builder = CudaBuilder()
    result = builder.build("does-not-exist.cu")
    assert not result.ok
    assert "not available" in (result.error or "")


def test_triton_builder_reports_unavailable_or_compiles() -> None:
    if TritonBuilder.is_available():
        pytest.skip("Triton+CUDA present; covered by the gpu-marked tests")
    builder = TritonBuilder()
    from kernelagent.types import KernelVariant

    result = builder.build(KernelVariant(knobs={"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}))
    assert not result.ok
    assert "not available" in (result.error or "")


@pytest.mark.gpu
def test_cuda_builder_compiles_real_kernel(tmp_path) -> None:
    if not CudaBuilder.is_available():
        pytest.skip("nvcc not available")
    source = tmp_path / "add.cu"
    source.write_text(
        """
__global__ void add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
int main() { return 0; }
"""
    )
    result = CudaBuilder().build(source, {"BLOCK": 256})
    assert result.ok, result.raw_stderr
    assert result.diagnostics["registers"] > 0


@pytest.mark.gpu
def test_triton_builder_compiles_and_benchmarks() -> None:
    if not TritonBuilder.is_available():
        pytest.skip("Triton/CUDA not available")
    from kernelagent.types import KernelVariant

    result = TritonBuilder().build(
        KernelVariant(knobs={"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "NUM_WARPS": 4})
    )
    assert result.ok, result.error
    assert result.diagnostics["gpu_ms_median"] > 0
