"""Typed contracts shared by every layer of KernelAgent.

The agent never sees raw shell output. Every backend call is normalised into one
of the pydantic models below, which are JSON-serialisable and validated at the
boundary. This keeps transcripts, the ledger and the analysis layer stable.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Shape(BaseModel):
    """A concrete problem shape, e.g. ``{"M": 256, "N": 256, "K": 256}``."""

    model_config = ConfigDict(frozen=True)

    dims: dict[str, int]

    @field_validator("dims")
    @classmethod
    def _positive_dims(cls, value: dict[str, int]) -> dict[str, int]:
        if not value:
            raise ValueError("shape must have at least one dimension")
        for key, dim in value.items():
            if not key or not key.isidentifier():
                raise ValueError(f"invalid dimension name: {key!r}")
            if int(dim) <= 0:
                raise ValueError(f"dimension {key} must be positive, got {dim}")
        return {k: int(v) for k, v in value.items()}

    def key(self) -> str:
        """Stable string key used for caches and reports."""
        return ",".join(f"{k}={v}" for k, v in sorted(self.dims.items()))

    def __getitem__(self, item: str) -> int:
        return self.dims[item]


class KnobSpec(BaseModel):
    """One legal knob and its allowed values."""

    name: str
    kind: Literal["int", "enum"] = "int"
    values: list[int | str]
    doc: str = ""

    @field_validator("values")
    @classmethod
    def _nonempty(cls, value: list[int | str]) -> list[int | str]:
        if not value:
            raise ValueError("knob must have at least one legal value")
        return value

    def legal(self, value: int | str) -> bool:
        """Return whether ``value`` is in this knob's legal set."""
        return value in self.values

    def size(self) -> int:
        """Number of legal values."""
        return len(self.values)


class KernelTask(BaseModel):
    """Static description of a kernel optimisation problem."""

    name: str
    description: str = ""
    directory: str
    shapes: dict[str, Shape]
    knobs: dict[str, KnobSpec]
    default_knobs: dict[str, int | str]
    flops_expr: str
    bytes_expr: str
    rtol: float = 1e-4
    atol: float = 1e-4
    reference_module: str = "reference.py"
    kernel_file: str = "kernel.cpp"
    notes: dict[str, Any] = Field(default_factory=dict)

    def shape_key(self, shape_name: str) -> Shape:
        """Look up a named shape, raising a helpful error when absent."""
        try:
            return self.shapes[shape_name]
        except KeyError as exc:
            raise KeyError(f"task {self.name} has no shape {shape_name!r}") from exc

    def knob_space_size(self) -> int:
        """Size of the full cartesian knob space."""
        total = 1
        for spec in self.knobs.values():
            total *= spec.size()
        return total

    def default_variant(self) -> KernelVariant:
        """The baseline variant declared in ``meta.yaml``."""
        return KernelVariant(knobs=dict(self.default_knobs), provenance="default")


class KernelVariant(BaseModel):
    """A concrete point in knob space plus extra compiler flags."""

    knobs: dict[str, int | str] = Field(default_factory=dict)
    cflags: list[str] = Field(default_factory=list)
    provenance: str = "default"

    def key(self) -> str:
        """Deterministic identity of this variant for caches and dedup."""
        knobs = ",".join(f"{k}={v}" for k, v in sorted(self.knobs.items()))
        flags = " ".join(sorted(self.cflags))
        return f"{knobs}|{flags}"

    def define_flags(self) -> list[str]:
        """Translate knobs into ``-D`` macros."""
        return [f"-D{k}={v}" for k, v in sorted(self.knobs.items())]

    def with_knob(self, name: str, value: int | str) -> KernelVariant:
        """Copy this variant with one knob replaced."""
        knobs = dict(self.knobs)
        knobs[name] = value
        return KernelVariant(knobs=knobs, cflags=list(self.cflags), provenance=self.provenance)


class BuildResult(BaseModel):
    """Outcome of compiling one variant."""

    ok: bool
    compile_time_s: float = 0.0
    binary: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    raw_stderr: str = ""
    cached: bool = False
    source_hash: str = ""
    command: list[str] = Field(default_factory=list)
    error: str | None = None


class RunResult(BaseModel):
    """Timing result emitted by the benchmark binary."""

    ms_median: float
    ms_mad: float
    ms_min: float
    reps: int
    throughput: float = 0.0
    checksum: float = 0.0
    unit: str = "ms"

    @property
    def noise_ratio(self) -> float:
        """Measurement noise proxy: MAD / median."""
        return self.ms_mad / self.ms_median if self.ms_median > 0 else 0.0


class ProfileResult(BaseModel):
    """Hardware or analytically derived counters for one trial."""

    counters: dict[str, float] = Field(default_factory=dict)
    source: Literal["ncu", "derived"] = "derived"
    notes: str = ""


class CheckResult(BaseModel):
    """Numerical correctness comparison against the NumPy reference."""

    passed: bool
    max_abs_err: float | None = None
    max_rel_err: float | None = None
    nan_or_inf: bool = False
    deterministic: bool | None = None
    message: str = ""


class Trial(BaseModel):
    """Everything measured for one (task, shape, variant) evaluation."""

    trial_id: str
    task: str
    shape: Shape
    variant: KernelVariant
    build: BuildResult
    run: RunResult | None = None
    profile: ProfileResult | None = None
    check: CheckResult | None = None
    speedup: float | None = None
    wall_s: float = 0.0
    strategy: str = ""
    seed: int = 0
    error: str | None = None
    timestamp: str = ""

    def solved(self, threshold: float = 1.05) -> bool:
        """True when the trial is correct and at least ``threshold`` faster."""
        return bool(
            self.check is not None
            and self.check.passed
            and self.speedup is not None
            and self.speedup >= threshold
        )


class Confirmation(BaseModel):
    """Re-measurement of a winning trial, to avoid reporting a lucky sample."""

    task: str
    shape: Shape
    variant: KernelVariant
    ms_median: float
    ms_min: float
    ms_max: float
    spread_pct: float
    baseline_ms: float
    speedup: float
    repeats: int
    noise_ratios: list[float] = Field(default_factory=list)
    check_passed: bool = True


class SearchRun(BaseModel):
    """A complete search episode over one task and shape."""

    strategy: str
    seed: int
    budget: int
    trials: list[Trial] = Field(default_factory=list)
    best: Trial | None = None
    wall_s: float = 0.0
    tokens: int = 0
    repair_attempts: int = 0
    stop_reason: str = ""
    confirmation: Confirmation | None = None


class ToolError(BaseModel):
    """Structured error returned by the tool layer instead of raw stderr."""

    kind: str
    message: str
    retryable: bool = False
    duration_s: float = 0.0
