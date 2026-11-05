"""Discovery and validation of the kernel task suite."""

from __future__ import annotations

import importlib.util
import itertools
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import yaml

from ..config import TASKS_DIR
from ..types import KernelTask, KernelVariant, KnobSpec, Shape


def load_tasks(root: Path | None = None) -> dict[str, KernelTask]:
    """Load and validate every ``tasks/<name>/meta.yaml``."""
    base = Path(root) if root else TASKS_DIR
    tasks: dict[str, KernelTask] = {}
    for meta_path in sorted(base.glob("*/meta.yaml")):
        with meta_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f"{meta_path}: meta.yaml must be a mapping")
        directory = meta_path.parent
        knobs = {
            name: KnobSpec(
                name=name,
                kind=spec.get("kind", "int"),
                values=spec["values"],
                doc=spec.get("doc", ""),
            )
            for name, spec in raw["knobs"].items()
        }
        task = KernelTask(
            name=raw["name"],
            description=raw.get("description", ""),
            directory=str(directory),
            shapes={name: Shape(dims=dims) for name, dims in raw["shapes"].items()},
            knobs=knobs,
            default_knobs=raw["default"],
            flops_expr=raw["flops"],
            bytes_expr=raw["bytes"],
            rtol=float(raw.get("rtol", 1e-4)),
            atol=float(raw.get("atol", 1e-4)),
            kernel_file=raw.get("kernel", "kernel.cpp"),
            notes=raw.get("notes", {}),
        )
        validate_task(task)
        if task.name in tasks:
            raise ValueError(f"duplicate task name {task.name}")
        tasks[task.name] = task
    if not tasks:
        raise FileNotFoundError(f"no tasks found under {base}")
    return tasks


def validate_task(task: KernelTask) -> None:
    """Check that files, default knobs and the declared space are coherent."""
    directory = Path(task.directory)
    for filename in (task.kernel_file, "meta.yaml", task.reference_module):
        path = directory / filename
        if not path.exists():
            raise FileNotFoundError(f"{task.name}: missing {path}")
    if set(task.default_knobs) != set(task.knobs):
        missing = set(task.knobs) - set(task.default_knobs)
        extra = set(task.default_knobs) - set(task.knobs)
        raise ValueError(f"{task.name}: default knobs mismatch (missing={missing}, extra={extra})")
    for name, spec in task.knobs.items():
        value = task.default_knobs[name]
        if not spec.legal(value):
            raise ValueError(f"{task.name}: default {name}={value} is not in {spec.values}")
    if not task.shapes:
        raise ValueError(f"{task.name}: no shapes")
    for shape in task.shapes.values():
        for key in shape.dims:
            if key not in task.flops_expr and key not in task.bytes_expr:
                # Not an error, but catches typos in dimension names early.
                raise ValueError(f"{task.name}: dimension {key!r} unused in workload model")


def variant_from_knobs(
    task: KernelTask,
    knobs: dict[str, int | str],
    cflags: list[str] | None = None,
    provenance: str = "manual",
) -> KernelVariant:
    """Build a validated variant, rejecting unknown or illegal knob values."""
    unknown = set(knobs) - set(task.knobs)
    if unknown:
        raise ValueError(f"{task.name}: unknown knobs {sorted(unknown)}")
    missing = set(task.knobs) - set(knobs)
    if missing:
        raise ValueError(f"{task.name}: missing knobs {sorted(missing)}")
    for name, value in knobs.items():
        if not task.knobs[name].legal(value):
            raise ValueError(f"{task.name}: illegal {name}={value!r}, legal={task.knobs[name].values}")
    return KernelVariant(knobs=dict(knobs), cflags=list(cflags or []), provenance=provenance)


def enumerate_variants(task: KernelTask, limit: int | None = None) -> Iterator[KernelVariant]:
    """Yield the cartesian product of the legal knob space."""
    names = list(task.knobs)
    value_lists = [task.knobs[name].values for name in names]
    for index, combo in enumerate(itertools.product(*value_lists)):
        if limit is not None and index >= limit:
            return
        yield KernelVariant(knobs=dict(zip(names, combo, strict=False)), provenance="enumerate")


def default_variant(task: KernelTask) -> KernelVariant:
    """The baseline declared in the task metadata."""
    return task.default_variant()


def load_reference(task: KernelTask) -> ModuleType:
    """Import a task's NumPy reference module by file path."""
    path = Path(task.directory) / task.reference_module
    module_name = f"kernelagent_reference_{task.name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reference for {task.name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
