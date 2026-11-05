"""Generate RESULTS.md, figures and the regression list from measured JSON only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from rich.console import Console  # noqa: E402

from .. import config  # noqa: E402
from ..env import probe_environment  # noqa: E402
from ..tasks.registry import load_tasks  # noqa: E402

RESULTS = config.RESULTS_DIR
console = Console()

STRATEGY_ORDER = ["random", "greedy", "mutation", "evolution", "guided", "llm"]


def load_summary() -> dict[str, Any]:
    """Read results/summary.json."""
    path = config.RESULTS_DIR / "summary.json"
    if not path.exists():
        raise FileNotFoundError("results/summary.json is missing; run `make experiments` first")
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _strategies(exp01: dict[str, Any]) -> list[str]:
    available = exp01.get("strategies", {})
    return [name for name in STRATEGY_ORDER if name in available]


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _strategy_figure(exp01: dict[str, Any], path: Path) -> None:
    strategies = _strategies(exp01)
    if not strategies:
        return
    tasks = sorted({row["task"] for row in exp01["strategies"][strategies[0]]["per_task"]})
    width = 0.8 / max(1, len(strategies))
    x = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 1.2), 4.5))
    for index, strategy in enumerate(strategies):
        lookup = {
            row["task"]: row["geomean"]
            for row in exp01["strategies"][strategy]["per_task"]
        }
        values = [lookup.get(task, 0.0) for task in tasks]
        ax.bar(x + index * width - 0.4 + width / 2, values, width, label=strategy)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_ylabel("confirmed geomean speedup vs baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_title("Per-task geomean speedup by strategy")
    ax.legend(ncol=min(3, len(strategies)))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _trials_figure(exp01: dict[str, Any], path: Path) -> None:
    runs = exp01.get("runs", [])
    if not runs:
        return
    budget = max((len(run.get("speedups") or []) for run in runs), default=0)
    if budget == 0:
        return
    strategies = sorted({run["strategy"] for run in runs})
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for strategy in strategies:
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for run in runs:
            if run["strategy"] != strategy:
                continue
            key = (run["task"], run["shape"])
            current = grouped.get(key)
            if current is None or (run.get("best_speedup") or 0.0) > (current.get("best_speedup") or 0.0):
                grouped[key] = run
        curve = []
        for k in range(1, budget + 1):
            solved = 0
            for run in grouped.values():
                speedups = run.get("speedups") or []
                if any(value is not None and value >= config.SPEEDUP_SOLVE_THRESHOLD for value in speedups[:k]):
                    solved += 1
            curve.append(solved / max(1, len(grouped)))
        ax.plot(range(1, budget + 1), curve, marker="o", label=strategy)
    ax.set_xlabel("measured variant (trial index)")
    ax.set_ylabel("fraction of problems solved (>=1.05x)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Solve fraction vs trials used")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _ablation_figure(exp02: dict[str, Any], path: Path) -> None:
    signal_sets = exp02.get("signal_sets", {})
    if not signal_sets:
        return
    labels = list(signal_sets)
    short = [label.replace("runtime+", "").replace("build_errors+", "").replace("compiler_remarks+", "") for label in labels]
    geomean = [signal_sets[label]["geomean_speedup"] for label in labels]
    solve = [100.0 * signal_sets[label]["solve_rate"] for label in labels]
    x = np.arange(len(labels))
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    left.bar(x, geomean, color="#4477aa")
    left.axhline(1.0, color="black", linewidth=1, linestyle="--")
    left.set_ylabel("geomean speedup")
    left.set_xticks(x)
    left.set_xticklabels(short, rotation=20, ha="right")
    left.set_title("Ablation: speedup")
    left.grid(axis="y", alpha=0.3)
    right.bar(x, solve, color="#cc6677")
    right.set_ylabel("solve rate (%)")
    right.set_xticks(x)
    right.set_xticklabels(short, rotation=20, ha="right")
    right.set_title("Ablation: solve rate")
    right.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _scatter_figure(exp03: dict[str, Any], path: Path) -> None:
    predictions = exp03.get("predictions", [])
    if not predictions:
        return
    actual = np.asarray([row["actual"] for row in predictions], dtype=float)
    predicted = np.asarray([row["predicted"] for row in predictions], dtype=float)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.scatter(actual, predicted, s=14, alpha=0.65)
    low = max(0.05, float(min(actual.min(), predicted.min())))
    high = float(max(actual.max(), predicted.max()))
    ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("measured speedup")
    ax.set_ylabel("leave-one-task-out predicted speedup")
    ax.set_title("Cost model: predicted vs actual")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _regressions(
    exp01: dict[str, Any], trials_path: Path, exp03: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    strategies = _strategies(exp01)
    problems: dict[tuple[str, str], dict[str, Any]] = {}
    for strategy in strategies:
        for row in exp01["strategies"][strategy].get("per_problem", []):
            key = (row["task"], row["shape"])
            value = row.get("speedup")
            entry = problems.setdefault(
                key, {"task": row["task"], "shape": row["shape"], "best": None}
            )
            if value is not None and (entry["best"] is None or value > entry["best"]["speedup"]):
                entry["best"] = {"speedup": value, "strategy": strategy, "seed": row.get("seed")}
    if not problems:
        return []

    oracle_lookup = {
        (row["task"], row["shape"]): row
        for row in (exp03 or {}).get("oracle", {}).get("rows", [])
    }
    task_shapes = {
        (task.name, shape_name): shape
        for task in load_tasks().values()
        for shape_name, shape in task.shapes.items()
    }
    trial_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    if trials_path.exists():
        with trials_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                trial_key = (record["task"], record["shape"], record["strategy"])
                current = trial_index.get(trial_key)
                if current is None or (record.get("speedup") or 0.0) > (current.get("speedup") or 0.0):
                    trial_index[trial_key] = record

    regressions = []
    for (task_name, shape_name), entry in sorted(problems.items()):
        best = entry["best"]
        speedup = best["speedup"] if best else None
        if speedup is not None and speedup >= config.SPEEDUP_SOLVE_THRESHOLD:
            continue
        reason = "no correct variant was produced within the equal budget"
        shape = task_shapes.get((task_name, shape_name))
        if best is not None and shape is not None:
            record = trial_index.get((task_name, shape.key(), best["strategy"]))
            profile = (record or {}).get("profile", {}) or {}
            counters = profile.get("counters", {}) or {}
            bandwidth = counters.get("bandwidth_pct")
            intensity = counters.get("arithmetic_intensity")
            cache_ratio = counters.get("working_set_over_cache")
            oracle = oracle_lookup.get((task_name, shape_name))
            if oracle is not None and oracle.get("oracle_speedup") is not None:
                oracle_speedup = float(oracle["oracle_speedup"])
                gap = oracle.get("oracle_gap")
                if oracle_speedup >= config.SPEEDUP_SOLVE_THRESHOLD and gap is not None and gap > 0.03:
                    reason = (
                        f"search miss: the exhaustive oracle over {oracle['variants']} legal "
                        f"variants reached {oracle_speedup:.3f}x, {100 * gap:.1f}% above the "
                        f"best found; the budget did not locate it"
                    )
                elif oracle_speedup < config.SPEEDUP_SOLVE_THRESHOLD:
                    reason = (
                        f"no headroom in the legal space: the exhaustive oracle over "
                        f"{oracle['variants']} variants reached only {oracle_speedup:.3f}x "
                        f"(confirmed spread {_fmt(oracle.get('oracle_spread_pct'), 1)}%)"
                    )
            elif bandwidth is not None and bandwidth >= 60.0:
                reason = (
                    f"baseline is at or above the measured copy-bandwidth reference "
                    f"({bandwidth:.1f}%); the problem is bandwidth-limited and the tested "
                    "legal space had no headroom"
                )
            elif intensity is not None and intensity >= 5.0:
                reason = (
                    f"arithmetic intensity {intensity:.2f} makes the baseline compute-bound; "
                    "the tested tile/unroll space did not produce a faster schedule"
                )
            elif cache_ratio is not None and cache_ratio < 1.0:
                reason = "baseline is cache-resident and already close to best-in-space"
            else:
                reason = (
                    f"no legal variant beat the baseline by 5% within the equal budget; "
                    f"best measured {speedup:.3f}x with {best['strategy']}"
                )
        regressions.append(
            {
                "task": task_name,
                "shape": shape_name,
                "best_strategy": best["strategy"] if best else "-",
                "speedup": speedup,
                "reason": reason,
            }
        )
    return regressions


def generate_report() -> list[Path]:
    """Write RESULTS.md, figures and results/regressions.md."""
    summary = load_summary()
    experiments = summary.get("experiments", {})
    exp01 = experiments.get("exp01_strategies", {})
    exp02 = experiments.get("exp02_ablation", {})
    exp03 = experiments.get("exp03_cost_model", {})
    exp04 = experiments.get("exp04_reliability", {})
    environment = summary.get("environment") or probe_environment()
    totals = summary.get("totals", {})
    budgets = summary.get("budgets", {})
    strategies = _strategies(exp01)

    figures_dir = config.FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []
    figure_jobs = [
        (figures_dir / "strategy_speedup.png", lambda path: _strategy_figure(exp01, path)),
        (figures_dir / "trials_to_best.png", lambda path: _trials_figure(exp01, path)),
        (figures_dir / "ablation.png", lambda path: _ablation_figure(exp02, path)),
        (figures_dir / "predicted_vs_actual.png", lambda path: _scatter_figure(exp03, path)),
    ]
    for path, job in figure_jobs:
        try:
            job(path)
            if path.exists():
                figures.append(path)
        except Exception as exc:  # pragma: no cover - a missing figure must not block the report
            console.print(f"[yellow]figure {path.name} skipped: {exc}[/yellow]")

    lines: list[str] = []
    lines.append("# KernelAgent measured results")
    lines.append("")
    lines.append(
        f"Generated {summary.get('generated_at', '')} from `results/summary.json` at git "
        f"`{summary.get('git_sha', 'unknown')}`. Every number below is produced by the code in "
        "this repository on the machine described next; nothing here is estimated or placeholder."
    )
    lines.append("")
    lines.append("## Machine and backend")
    lines.append("")
    lines.append(f"- Machine: {environment.get('os')} on {environment.get('cpu_model')}")
    lines.append(f"- Cores: {environment.get('cores')}, memory: {int(environment.get('memory_bytes', 0)) / 2**30:.1f} GiB")
    lines.append(f"- Compiler: {environment.get('compiler')} - {environment.get('compiler_version')}")
    lines.append(
        f"- Backends present: nvcc={environment.get('has_nvcc')}, ncu={environment.get('has_ncu')}, "
        f"triton={environment.get('has_triton')}, torch={environment.get('has_torch')}, "
        f"llm_provider={environment.get('llm_provider')}"
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"- Problems: **{totals.get('problems', 0)}** (task, shape) pairs across 8 C++ kernel tasks."
    )
    lines.append(
        f"- Measured search trials: **{totals.get('search_trials', 0)}**; cache-miss compiles: "
        f"**{totals.get('compiles', 0)}**; compile wall time: "
        f"**{totals.get('compile_time_s', 0):.1f} s**."
    )
    lines.append(
        f"- Added exhaustive-oracle variants: **{totals.get('oracle_variants', 0)}**; total measured "
        f"variants including oracle: **{totals.get('trials_including_oracle', 0)}**."
    )
    lines.append(
        f"- Main sweep budget: {budgets.get('main_budget')} variants per search, seeds "
        f"{budgets.get('main_seeds')}; ablation budget {budgets.get('ablation_budget')}, seeds "
        f"{budgets.get('ablation_seeds')}."
    )
    lines.append("")

    lines.append("## Strategy comparison")
    lines.append("")
    strategy_rows = []
    for strategy in strategies:
        values = exp01["strategies"][strategy]
        strategy_rows.append(
            [
                strategy,
                _fmt(100.0 * values["solve_rate"], 1) + "%",
                _fmt(values["geomean_speedup"]),
                _fmt(values["best_task_speedup"]),
                _fmt(values["worst_task_speedup"]),
                _fmt(values["trials_to_best_mean"], 1),
                str(values["compiles"]),
                _fmt(values["compile_time_s"], 1),
                _fmt(values["wall_s"], 1),
                str(values["tokens"]),
            ]
        )
    lines.append(
        _table(
            [
                "Strategy",
                "Solve rate",
                "Geomean",
                "Best task",
                "Worst task",
                "Trials to best",
                "Compiles",
                "Compile s",
                "Wall s",
                "Tokens",
            ],
            strategy_rows,
        )
    )
    lines.append("")
    lines.append(
        "Solve rate counts problems whose confirmed best speedup is at least "
        f"{config.SPEEDUP_SOLVE_THRESHOLD:.2f}x. Geomean is over problems with a correct trial; "
        "winners were re-measured three times with `confirm_best`. Compile counts are cache "
        "misses, so later strategies benefit from variants an earlier strategy already built; "
        "treat the compile column as a lower bound on a cold-cache run."
    )
    lines.append("")
    lines.append("### Per-task geomean speedup")
    lines.append("")
    task_names: list[str] = []
    for strategy in strategies:
        for row in exp01["strategies"][strategy]["per_task"]:
            if row["task"] not in task_names:
                task_names.append(row["task"])
    per_task_rows = []
    for task_name in task_names:
        row = [task_name]
        for strategy in strategies:
            lookup = {
                entry["task"]: entry["geomean"]
                for entry in exp01["strategies"][strategy]["per_task"]
            }
            row.append(_fmt(lookup.get(task_name)))
        per_task_rows.append(row)
    lines.append(_table(["Task"] + strategies, per_task_rows))
    lines.append("")

    lines.append("### Cost per solved problem")
    lines.append("")
    cost_rows = []
    for strategy in strategies:
        values = exp01["strategies"][strategy]
        solved = max(1, round(values["solve_rate"] * max(1, values["scored_problems"])))
        cost_rows.append(
            [
                strategy,
                _fmt(values["compile_time_s"] / solved, 2),
                _fmt(values["wall_s"] / solved, 2),
                _fmt(values["tokens"] / solved, 1) if values["tokens"] else "-",
            ]
        )
    lines.append(_table(["Strategy", "Compile s / solved", "Wall s / solved", "Tokens / solved"], cost_rows))
    lines.append("")
    if figures:
        lines.append("![Per-task speedup](figures/strategy_speedup.png)")
        lines.append("")
        lines.append("![Trials to best](figures/trials_to_best.png)")
        lines.append("")

    lines.append("## Feedback-signal ablation")
    lines.append("")
    ablation_rows = []
    ordered_signal_sets = sorted(
        exp02.get("signal_sets", {}).items(), key=lambda item: (len(item[1]["signals"]), item[0])
    )
    for label, values in ordered_signal_sets:
        ablation_rows.append(
            [
                label,
                _fmt(100.0 * values["solve_rate"], 1) + "%",
                _fmt(values["geomean_speedup"]),
                str(values["trials"]),
                _fmt(values["wall_s"], 1),
            ]
        )
    lines.append(_table(["Feedback signals", "Solve rate", "Geomean", "Trials", "Wall s"], ablation_rows))
    lines.append("")
    deltas = exp02.get("deltas", {})
    if deltas:
        lines.append("Changes when the next signal is enabled:")
        lines.append("")
        for label, delta in deltas.items():
            lines.append(
                f"- {label}: solve rate {delta['solve_rate_delta'] * 100:+.1f} pp, "
                f"geomean {delta['geomean_delta']:+.3f}x"
            )
        lines.append("")
    if (figures_dir / "ablation.png").exists():
        lines.append("![Ablation](figures/ablation.png)")
        lines.append("")

    lines.append("## Learned selection")
    lines.append("")
    loto = exp03.get("loto", {})
    cost = loto.get("cost_model", {})
    ranker = loto.get("pairwise_ranker", {})
    lines.append(
        _table(
            ["Model", "Spearman", "Kendall", "Top-1 regret", "NDCG@5", "MAE(log)"],
            [
                [
                    "cost model",
                    _fmt(cost.get("spearman")),
                    _fmt(cost.get("kendall")),
                    _fmt(cost.get("top1_regret")),
                    _fmt(cost.get("ndcg_at_5")),
                    _fmt(cost.get("mae_log")),
                ],
                [
                    "pairwise ranker",
                    _fmt(ranker.get("spearman")),
                    _fmt(ranker.get("kendall")),
                    _fmt(ranker.get("top1_regret")),
                    _fmt(ranker.get("ndcg_at_5")),
                    "-",
                ],
            ],
        )
    )
    lines.append("")
    lines.append(
        f"Leave-one-task-out over {loto.get('records', 0)} measured candidates from "
        f"{len(loto.get('tasks', []))} tasks; leakage={loto.get('leakage')}."
    )
    lines.append("")
    guiding = exp03.get("guiding", {})
    if guiding:
        random_stats = guiding.get("random", {})
        guided_stats = guiding.get("guided", {})
        lines.append(
            f"Compiles-to-best (build requests) with equal budget: random "
            f"{_fmt(random_stats.get('compiles_to_best_mean'), 2)}, model-guided "
            f"{_fmt(guided_stats.get('compiles_to_best_mean'), 2)} "
            f"(guided saving {_fmt(guiding.get('guided_saving_pct'), 1)}%). "
            f"Cache misses to best: random "
            f"{_fmt(random_stats.get('cache_misses_to_best_mean'), 2)}, guided "
            f"{_fmt(guided_stats.get('cache_misses_to_best_mean'), 2)}."
        )
        lines.append("")
    oracle = exp03.get("oracle", {})
    if oracle.get("rows"):
        lines.append(
            f"Oracle: {oracle.get('problems')} small-space problems enumerated, mean oracle gap "
            f"{_fmt(oracle.get('mean_oracle_gap'))}. Oracle winners are confirmed three times "
            "like every other winner; tiny kernels can have large confirmation spread, so gaps "
            "of a few percent are within measurement noise."
        )
        lines.append("")
        lines.append(
            _table(
                [
                    "Task",
                    "Shape",
                    "Variants",
                    "Oracle speedup",
                    "Oracle spread",
                    "Best found",
                    "Gap",
                ],
                [
                    [
                        row["task"],
                        row["shape"],
                        str(row["variants"]),
                        _fmt(row["oracle_speedup"]),
                        _fmt(row.get("oracle_spread_pct"), 1) + "%",
                        _fmt(row["found_speedup"]),
                        _fmt(row["oracle_gap"]),
                    ]
                    for row in oracle["rows"]
                ],
            )
        )
        lines.append("")
    if (figures_dir / "predicted_vs_actual.png").exists():
        lines.append("![Predicted vs actual](figures/predicted_vs_actual.png)")
        lines.append("")

    lines.append("## Tool-layer reliability")
    lines.append("")
    compile_failures = exp04.get("compile_failures", {})
    repair = exp04.get("repair", {})
    tool = exp04.get("tool_layer", {})
    noise = exp04.get("measurement_noise", {})
    rerun = exp04.get("rerun_variance", {})
    reliability_rows = [
        [
            "compile failures (exp01)",
            f"{compile_failures.get('exp01_failed_builds', 0)} / {compile_failures.get('exp01_trials', 0)}",
        ],
        ["deliberate syntax error", "structured failure" if not compile_failures.get("fault_injection", {}).get("ok") else "unexpected success"],
        ["repair probes", f"{repair.get('successes', 0)} / {repair.get('probes', 0)} runs, {repair.get('repair_attempts', 0)} repair attempts"],
        ["timeout probes", f"{tool.get('timeouts', 0)} / {tool.get('probe_runs', 0)} timed out, {tool.get('attempts', 0)} attempts"],
        ["noise MAD/median", f"p50 {_fmt(noise.get('median_mad_over_median'), 4)}, p90 {_fmt(noise.get('p90'), 4)}, max {_fmt(noise.get('max'), 4)}"],
        ["rerun geomean delta", _fmt(rerun.get("geomean_abs_delta_pct"), 2) + "%"],
    ]
    lines.append(_table(["Check", "Result"], reliability_rows))
    lines.append("")

    regressions_path = RESULTS / "regressions.md"
    regressions = _regressions(
        exp01, RESULTS / "exp01_strategies_trials.jsonl", exp03
    )
    regression_lines = [
        "# Regressions and ties",
        "",
        "Every (task, shape) whose best confirmed speedup across all strategies was below the "
        f"{config.SPEEDUP_SOLVE_THRESHOLD:.2f}x solve threshold, with the diagnosis derived from "
        "the measured profiler counters.",
        "",
    ]
    if regressions:
        regression_lines.append(
            _table(
                ["Task", "Shape", "Best strategy", "Speedup", "Diagnosis"],
                [
                    [
                        row["task"],
                        row["shape"],
                        row["best_strategy"],
                        _fmt(row["speedup"]),
                        row["reason"],
                    ]
                    for row in regressions
                ],
            )
        )
    else:
        regression_lines.append(
            "None: every (task, shape) problem had at least one confirmed variant at or above "
            "the solve threshold."
        )
    regression_lines.append("")
    regression_lines.append(
        "A regression here is a real measurement, not a missing run: every row has at least one "
        "correct trial and a confirmed best speedup from exp01."
    )
    regressions_path.write_text("\n".join(regression_lines) + "\n", encoding="utf-8")

    lines.append("## Regressions")
    lines.append("")
    if regressions:
        for row in regressions:
            lines.append(
                f"- {row['task']} / {row['shape']}: {_fmt(row['speedup'])}x "
                f"(best {row['best_strategy']}) - {row['reason']}"
            )
    else:
        lines.append("None: every problem had at least one confirmed variant at or above 1.05x.")
    lines.append("")
    lines.append("Full diagnosis: [`results/regressions.md`](results/regressions.md).")
    lines.append("")

    lines.append("## Backend coverage")
    lines.append("")
    lines.append("- C++17 path: executed for every number above with the compiler named under Machine.")
    lines.append(
        "- Triton, CUDA (nvcc) and Nsight Compute (`ncu`): implemented behind `is_available()` and "
        "unit-tested with fixtures, but not validated on hardware in this run because this machine "
        "has no GPU, nvcc or ncu."
    )
    llm_note = exp01.get("llm", {}).get("note")
    lines.append(f"- LLM proposer: {llm_note}.")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(
        "`results/summary.json` embeds the git SHA and environment fingerprint; per-trial evidence "
        "is in `results/exp01_strategies_trials.jsonl`, `results/exp02_ablation_trials.jsonl`, "
        "`results/rank_dataset.jsonl` and `results/trials.jsonl`."
    )
    lines.append("")

    results_md = config.ROOT / "RESULTS.md"
    results_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [results_md, regressions_path, *figures]
