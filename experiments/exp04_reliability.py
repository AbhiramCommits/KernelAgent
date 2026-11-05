"""exp04: tool-layer reliability, measurement noise and rerun variance."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from kernelagent import config
from kernelagent.agent.loop import run_search
from kernelagent.agent.proposer import Proposer
from kernelagent.eval.harness import geomean
from kernelagent.ledger import TrialLedger
from kernelagent.tasks.registry import load_tasks
from kernelagent.tools.base import run_subprocess
from kernelagent.tools.build import CxxBuilder
from kernelagent.types import KernelTask, KernelVariant, KnobSpec, SearchRun, Shape

from .common import RESULTS, search_record, write_experiment

RERUN_PROBLEMS = [
    ("sgemm_tiled", "small"),
    ("reduce_sum", "large"),
    ("conv1d_direct", "large"),
    ("transpose_blocked", "small"),
]

_PROBE_KERNEL = r"""
#include "../harness.hpp"
#ifndef FIX
#define FIX 0
#endif
#if !FIX
#error probe: this variant is deliberately broken
#endif
int main(int argc, char** argv) {
  ka::Args args = ka::parse_args(argc, argv);
  const int N = static_cast<int>(args.dim("N", 64));
  std::vector<float> x(N), out(N, 0.0f);
  ka::load_or_random(x.data(), x.size(), args.input_dir, "x.bin", 7);
  ka::maybe_dump_input(args, "x.bin", x.data(), x.size() * sizeof(float));
  auto fn = [&]() -> double {
    for (int i = 0; i < N; ++i) out[i] = x[i];
    return static_cast<double>(out[0]);
  };
  ka::BenchResult result = ka::time_kernel(fn, args.warmup, args.reps);
  ka::emit_result(result);
  ka::dump_output(args, "out.bin", out.data(), out.size());
  return 0;
}
"""

_PROBE_REFERENCE = """
from pathlib import Path
import numpy as np

def reference(dims, dump_dir):
    return np.fromfile(dump_dir / "x.bin", dtype=np.float32).reshape(dims["N"])
"""


class _RepairProbe(Proposer):
    name = "repair-probe"

    def __init__(self) -> None:
        self.calls = 0

    def propose(self, task, shape, history, feedback):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == 1:
            return [KernelVariant(knobs={"FIX": 0}, provenance="broken")]
        if "repair:" in feedback:
            return [KernelVariant(knobs={"FIX": 1}, provenance="repaired")]
        return [KernelVariant(knobs={"FIX": 0}, provenance="broken")]


def _noise_stats() -> dict[str, Any]:
    path = RESULTS / "exp01_strategies_trials.jsonl"
    if not path.exists():
        return {"available": False}
    ratios = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            run = record.get("run") or {}
            median = run.get("ms_median")
            mad = run.get("ms_mad")
            if median and mad is not None:
                ratios.append(float(mad) / float(median))
    if not ratios:
        return {"available": False}
    values = np.asarray(ratios)
    return {
        "available": True,
        "count": int(values.size),
        "median_mad_over_median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "over_rerun_threshold": int(np.sum(values > config.NOISE_RERUN_RATIO)),
    }


def _fault_injection() -> dict[str, Any]:
    timeouts = 0
    attempts = 0
    durations = []
    for _ in range(3):
        result = run_subprocess(
            [sys.executable, "-c", "import time; time.sleep(2)"],
            timeout_s=0.2,
            retries=2,
            backoff_s=0.01,
        )
        timeouts += int(result.timed_out)
        attempts += result.attempts
        durations.append(result.duration_s)
    return {
        "probe_runs": 3,
        "timeouts": timeouts,
        "attempts": attempts,
        "mean_duration_s": float(np.mean(durations)),
        "max_duration_s": float(max(durations)),
    }


def _compile_failure_probe(tmp: Path) -> dict[str, Any]:
    source = tmp / "broken.cpp"
    source.write_text("int main( { return 0; }\n", encoding="utf-8")
    task = load_tasks()["sgemm_tiled"].model_copy(
        update={"directory": str(tmp), "kernel_file": "broken.cpp", "name": "broken_probe"}
    )
    result = CxxBuilder().build(task, task.default_variant())
    return {
        "ok": result.ok,
        "error": result.error,
        "has_compile_errors": bool(result.diagnostics.get("compile_errors")),
        "stderr_lines": len(result.raw_stderr.splitlines()),
        "compile_time_s": result.compile_time_s,
    }


def _repair_probe(tmp: Path) -> dict[str, Any]:
    task_dir = tmp / "repair_probe"
    task_dir.mkdir()
    shutil.copy(Path(__file__).parent.parent / "kernelagent/tasks/harness.hpp", tmp / "harness.hpp")
    (task_dir / "kernel.cpp").write_text(_PROBE_KERNEL, encoding="utf-8")
    (task_dir / "reference.py").write_text(_PROBE_REFERENCE, encoding="utf-8")
    task = KernelTask(
        name="repair_probe",
        directory=str(task_dir),
        shapes={"small": Shape(dims={"N": 64})},
        knobs={"FIX": KnobSpec(name="FIX", values=[0, 1])},
        default_knobs={"FIX": 1},
        flops_expr="N",
        bytes_expr="4*N",
        rtol=1e-5,
        atol=1e-5,
    )
    successes = 0
    attempts = 0
    runs = []
    for _ in range(3):
        run = run_search(task=task, shape="small", proposer=_RepairProbe(), budget=3, patience=99)
        attempts += run.repair_attempts
        if run.best is not None and run.best.check is not None and run.best.check.passed:
            successes += 1
        runs.append({"repairs": run.repair_attempts, "best": run.best.speedup if run.best else None})
    return {"probes": 3, "successes": successes, "repair_attempts": attempts, "runs": runs}


def _rerun_variance(ledger: TrialLedger, budget: int) -> tuple[dict[str, Any], list[Any]]:
    suite = load_tasks()
    passes: list[dict[str, Any]] = []
    all_runs: list[SearchRun] = []
    all_trials: list[Any] = []
    for pass_index in range(2):
        per_problem = []
        for task_name, shape_name in RERUN_PROBLEMS:
            task = suite[task_name]
            run = run_search(
                task=task.name,
                shape=shape_name,
                strategy="mutation",
                budget=budget,
                seed=0,
                ledger=ledger,
            )
            all_runs.append(run)
            all_trials.extend(run.trials)
            per_problem.append(
                {
                    "task": task.name,
                    "shape": shape_name,
                    "speedup": run.best.speedup if run.best else None,
                    "trials_to_best": None,
                }
            )
        values = [float(row["speedup"]) for row in per_problem if row["speedup"]]
        passes.append(
            {
                "pass": pass_index + 1,
                "geomean_speedup": geomean(values),
                "per_problem": per_problem,
            }
        )
    first, second = passes[0], passes[1]
    deltas = []
    for row_a, row_b in zip(first["per_problem"], second["per_problem"], strict=False):
        if row_a["speedup"] and row_b["speedup"]:
            deltas.append(abs(row_a["speedup"] - row_b["speedup"]) / row_a["speedup"])
    payload = {
        "budget": budget,
        "seed": 0,
        "passes": passes,
        "geomean_abs_delta_pct": (
            100.0
            * abs(first["geomean_speedup"] - second["geomean_speedup"])
            / first["geomean_speedup"]
            if first["geomean_speedup"]
            else None
        ),
        "per_problem_abs_delta_pct_mean": 100.0 * float(np.mean(deltas)) if deltas else None,
        "runs": [search_record(run) for run in all_runs],
    }
    return payload, all_trials


def main() -> dict[str, Any]:
    """Run reliability probes and a fixed-seed rerun."""
    budget = int(os.environ.get("KA_EXP_RERUN_BUDGET", "6"))
    exp01_trials = 0
    exp01_build_failures = 0
    trial_path = RESULTS / "exp01_strategies_trials.jsonl"
    if trial_path.exists():
        with trial_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                exp01_trials += 1
                if not record.get("build", {}).get("ok", True):
                    exp01_build_failures += 1

    ledger = TrialLedger()
    with tempfile.TemporaryDirectory(prefix="ka-rel-") as tmp:
        tmp_path = Path(tmp)
        compile_probe = _compile_failure_probe(tmp_path)
        repair_probe = _repair_probe(tmp_path)

    rerun, rerun_trials = _rerun_variance(ledger, budget)
    noise = _noise_stats()
    fault = _fault_injection()

    payload: dict[str, Any] = {
        "budget": budget,
        "exp01_trials": exp01_trials,
        "compile_failures": {
            "exp01_failed_builds": exp01_build_failures,
            "exp01_trials": exp01_trials,
            "failed_build_rate": (
                exp01_build_failures / exp01_trials if exp01_trials else None
            ),
            "fault_injection": compile_probe,
        },
        "repair": repair_probe,
        "tool_layer": fault,
        "measurement_noise": noise,
        "rerun_variance": rerun,
        "notes": (
            "Timeout/retry counts come from a deliberate sleep probe; compile failures are "
            "counted over exp01 and a deliberate syntax error; rerun variance repeats the same "
            "strategy, seed and budget twice."
        ),
    }
    write_experiment("exp04_reliability", payload, rerun_trials, [])
    from .common import assemble_summary

    assemble_summary()
    return payload


if __name__ == "__main__":
    result = main()
    print(f"wrote exp04: noise p50={result['measurement_noise'].get('median_mad_over_median')}")
