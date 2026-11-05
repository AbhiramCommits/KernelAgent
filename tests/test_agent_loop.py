"""Agent-loop tests: budgets, repair, correctness gate and wall caps."""

from __future__ import annotations

import shutil
from pathlib import Path

from kernelagent.agent.loop import run_search
from kernelagent.agent.proposer import Proposer
from kernelagent.types import KernelTask, KernelVariant, KnobSpec, Shape

TOY_KERNEL = r"""
#include "../harness.hpp"

#ifndef OK
#define OK 1
#endif
#ifndef WRONG
#define WRONG 0
#endif

#if !OK
#error deliberately broken variant
#endif

int main(int argc, char** argv) {
  ka::Args args = ka::parse_args(argc, argv);
  const int N = static_cast<int>(args.dim("N", 64));
  std::vector<float> x(N), out(N, 0.0f);
  ka::load_or_random(x.data(), x.size(), args.input_dir, "x.bin", 42);
  ka::maybe_dump_input(args, "x.bin", x.data(), x.size() * sizeof(float));
  auto fn = [&]() -> double {
    for (int i = 0; i < N; ++i) out[i] = x[i] + static_cast<float>(WRONG);
    return static_cast<double>(out[0]);
  };
  ka::BenchResult result = ka::time_kernel(fn, args.warmup, args.reps);
  ka::emit_result(result);
  ka::dump_output(args, "out.bin", out.data(), out.size());
  return 0;
}
"""

TOY_REFERENCE = """
from pathlib import Path
import numpy as np

def reference(dims, dump_dir):
    n = dims["N"]
    return np.fromfile(dump_dir / "x.bin", dtype=np.float32).reshape(n)
"""


def make_toy_task(tmp_path: Path) -> KernelTask:
    task_dir = tmp_path / "toy"
    task_dir.mkdir()
    shutil.copy(Path(__file__).parent.parent / "kernelagent/tasks/harness.hpp", tmp_path / "harness.hpp")
    (task_dir / "kernel.cpp").write_text(TOY_KERNEL, encoding="utf-8")
    (task_dir / "reference.py").write_text(TOY_REFERENCE, encoding="utf-8")
    return KernelTask(
        name="toy",
        directory=str(task_dir),
        shapes={"small": Shape(dims={"N": 128})},
        knobs={
            "OK": KnobSpec(name="OK", values=[0, 1]),
            "WRONG": KnobSpec(name="WRONG", values=[0, 1]),
        },
        default_knobs={"OK": 1, "WRONG": 0},
        flops_expr="N",
        bytes_expr="4*N",
        rtol=1e-4,
        atol=1e-4,
    )


class ScriptedProposer:
    name = "scripted"

    def __init__(self, variants: list[KernelVariant]) -> None:
        self.variants = list(variants)
        self.calls = 0

    def propose(self, task, shape, history, feedback):  # type: ignore[no-untyped-def]
        self.calls += 1
        if not self.variants:
            return [task.default_variant()]
        return [self.variants.pop(0)]


class AlwaysProposer(ScriptedProposer):
    def __init__(self, variant: KernelVariant) -> None:
        super().__init__([])
        self.variant = variant

    def propose(self, task, shape, history, feedback):  # type: ignore[no-untyped-def]
        return [self.variant]


class RepairProposer(ScriptedProposer):
    def propose(self, task, shape, history, feedback):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == 1:
            return [KernelVariant(knobs={"OK": 0, "WRONG": 0}, provenance="broken")]
        if "repair:" in feedback:
            return [KernelVariant(knobs={"OK": 1, "WRONG": 0}, provenance="repaired")]
        return [KernelVariant(knobs={"OK": 1, "WRONG": 1}, provenance="wrong")]


def test_loop_honours_budget(tmp_path: Path) -> None:
    task = make_toy_task(tmp_path)
    proposer: Proposer = ScriptedProposer([])
    run = run_search(task=task, shape="small", proposer=proposer, budget=3, patience=99)
    assert len(run.trials) == 3
    assert run.stop_reason == "budget"
    assert run.best is not None


def test_loop_repairs_compile_error(tmp_path: Path) -> None:
    task = make_toy_task(tmp_path)
    proposer = RepairProposer([])
    run = run_search(task=task, shape="small", proposer=proposer, budget=5, patience=99)
    assert run.repair_attempts >= 1
    assert any(not trial.build.ok for trial in run.trials)
    assert run.best is not None
    assert run.best.check is not None and run.best.check.passed


def test_failed_correctness_never_reports_speedup(tmp_path: Path) -> None:
    task = make_toy_task(tmp_path)
    wrong = KernelVariant(knobs={"OK": 1, "WRONG": 1}, provenance="wrong")
    run = run_search(task=task, shape="small", proposer=AlwaysProposer(wrong), budget=2, patience=99)
    assert run.best is None
    for trial in run.trials:
        assert trial.speedup is None
        assert trial.check is not None and not trial.check.passed


def test_loop_stops_on_patience(tmp_path: Path) -> None:
    task = make_toy_task(tmp_path)
    wrong = KernelVariant(knobs={"OK": 1, "WRONG": 1}, provenance="wrong")
    run = run_search(
        task=task, shape="small", proposer=AlwaysProposer(wrong), budget=50, patience=2
    )
    assert run.stop_reason == "patience"
    assert len(run.trials) == 2  # patience counts the first non-improving trial


def test_loop_stops_on_wall_cap(tmp_path: Path) -> None:
    task = make_toy_task(tmp_path)
    run = run_search(
        task=task,
        shape="small",
        proposer=ScriptedProposer([]),
        budget=50,
        patience=99,
        wall_cap_s=0.0,
    )
    assert run.stop_reason == "wall_cap"
    assert len(run.trials) == 1
