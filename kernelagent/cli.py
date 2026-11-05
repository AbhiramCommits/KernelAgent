"""Command line entry point.

Commands are deliberately thin: parsing and rendering live here, measurement
lives in the tool/eval/agent layers so it can be unit-tested without a terminal.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__, config
from .env import capability_rows, probe_environment

app = typer.Typer(
    name="kernelagent",
    help="Agent loop for compiler-and-profiler-guided kernel optimisation.",
    no_args_is_help=True,
)
console = Console()


def _echo_json(payload: object) -> None:
    console.print_json(json.dumps(payload, sort_keys=True, default=str))


@app.command()
def doctor() -> None:
    """Print the capability table for this machine."""
    env = probe_environment()
    console.print(f"[bold]KernelAgent {__version__}[/bold] capability report")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Capability")
    table.add_column("Value")
    table.add_column("Status")
    for name, value, status in capability_rows():
        style = "green" if status == "ok" else "yellow"
        table.add_row(name, value, f"[{style}]{status}[/{style}]")
    console.print(table)
    console.print(f"fingerprint: {env['machine']} / compiler {env['compiler']}")
    console.print(f"results dir: {config.RESULTS_DIR}")


@app.command()
def tasks() -> None:
    """List the available kernel tasks."""
    try:
        from .tasks.registry import load_tasks

        loaded = load_tasks()
    except Exception as exc:  # pragma: no cover - only before the suite exists
        console.print(f"[yellow]no tasks available: {exc}[/yellow]")
        raise typer.Exit(0) from exc
    table = Table(show_header=True, header_style="bold")
    table.add_column("Task")
    table.add_column("Shapes")
    table.add_column("Knobs")
    table.add_column("Space")
    table.add_column("Character")
    for task in loaded.values():
        table.add_row(
            task.name,
            ", ".join(task.shapes),
            str(len(task.knobs)),
            str(task.knob_space_size()),
            str(task.notes.get("character", "")),
        )
    console.print(table)


@app.command("build")
def build_cmd(
    task: str = typer.Option(..., help="Task name"),
    variant: str = typer.Option("default", help="Variant name or JSON knob dict"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON"),
) -> None:
    """Compile one kernel variant."""
    from .tasks.registry import load_tasks
    from .tools.build import CxxBuilder

    loaded = load_tasks()
    if task not in loaded:
        console.print(f"[red]unknown task {task}[/red]")
        raise typer.Exit(2)
    kernel_task = loaded[task]
    chosen = kernel_task.default_variant()
    if variant != "default":
        chosen = chosen.model_copy(update={"knobs": json.loads(variant), "provenance": "cli"})
    result = CxxBuilder().build(kernel_task, chosen)
    if json_out:
        _echo_json(result.model_dump(mode="json"))
    else:
        console.print_json(result.model_dump_json())
    raise typer.Exit(0 if result.ok else 1)


@app.command()
def run(
    task: str = typer.Option(..., help="Task name"),
    shape: str = typer.Option(..., help="Shape name"),
    variant: str = typer.Option("default", help="Variant JSON"),
    reps: int = typer.Option(config.DEFAULT_REPS, help="Timed repetitions"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Build and time one variant."""
    from .eval.harness import run_variant

    result = run_variant(task, shape, variant, reps=reps, ledger=None)
    if json_out:
        _echo_json(result.model_dump(mode="json"))
    else:
        console.print_json(result.model_dump_json())
    raise typer.Exit(0 if result.run is not None else 1)


@app.command()
def check(
    task: str = typer.Option(..., help="Task name"),
    shape: str = typer.Option(..., help="Shape name"),
    variant: str = typer.Option("default", help="Variant JSON"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Check a variant against the NumPy reference."""
    from .eval.harness import check_variant

    result = check_variant(task, shape, variant)
    if json_out:
        _echo_json(result.model_dump(mode="json"))
    else:
        console.print_json(result.model_dump_json())
    raise typer.Exit(0 if (result.check and result.check.passed) else 1)


@app.command()
def search(
    task: str = typer.Option(..., help="Task name"),
    shape: str = typer.Option(..., help="Shape name"),
    strategy: str = typer.Option("mutation", help="random|greedy|mutation|evolution|guided|llm"),
    budget: int = typer.Option(config.DEFAULT_BUDGET),
    seed: int = typer.Option(0),
    patience: int = typer.Option(config.DEFAULT_PATIENCE),
    replay: str = typer.Option("", help="Replay a recorded LLM transcript"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Run one search episode and append its trials to the ledger."""
    from .agent.loop import run_search

    run = run_search(
        task=task,
        shape=shape,
        strategy=strategy,
        budget=budget,
        seed=seed,
        patience=patience,
        replay=replay or None,
    )
    if json_out:
        _echo_json(run.model_dump(mode="json"))
    else:
        best = run.best
        console.print(
            f"[bold]{run.strategy}[/bold] {task}/{shape}: {len(run.trials)} trials, "
            f"stop={run.stop_reason}, best speedup="
            f"{best.speedup if best and best.speedup else float('nan'):.3f}x"
        )
        console.print(f"wall: {run.wall_s:.2f}s, repairs: {run.repair_attempts}, tokens: {run.tokens}")
    raise typer.Exit(0 if run.best is not None else 1)


@app.command("run-suite")
def run_suite(
    strategy: str = typer.Option("default", help="default|random|greedy|mutation|evolution|guided"),
    budget: int = typer.Option(config.DEFAULT_BUDGET),
    out: Path = typer.Option(config.RESULTS_DIR / "baseline.json"),
    seeds: str = typer.Option("0"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Run every (task, shape) pair and write a baseline table."""
    from .eval.suite import run_suite as _run_suite

    summary = _run_suite(strategy=strategy, budget=budget, seeds=[int(s) for s in seeds.split(",")], out=out)
    if json_out:
        _echo_json(summary)
    else:
        console.print(f"wrote {out}: {summary['problems']} problems, {summary['trials']} trials")


@app.command("rank-eval")
def rank_eval(
    dataset: Path = typer.Option(config.RANK_DATASET),
    collect: bool = typer.Option(False, help="Collect a fresh measured dataset first"),
    per_task: int = typer.Option(10, help="Variants per task when collecting"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Leave-one-task-out ranking evaluation for the learned selector."""
    from .select.evaluate import run_loto_evaluation

    if collect:
        from .select.collect import collect_dataset

        collect_dataset(per_task=per_task, out=dataset)
    metrics = run_loto_evaluation(dataset_path=dataset)
    if json_out:
        _echo_json(metrics)
    else:
        console.print_json(json.dumps(metrics, default=str))


@app.command()
def experiment(
    name: str = typer.Option(..., help="exp01_strategies|exp02_ablation|exp03_cost_model|exp04_reliability"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Run one experiment module."""
    import importlib

    module = importlib.import_module(f"experiments.{name}")
    result = module.main()
    if json_out:
        _echo_json(result)


@app.command()
def report() -> None:
    """Generate RESULTS.md, figures and the regression list from results/."""
    from .report.generate import generate_report

    paths = generate_report()
    for path in paths:
        console.print(f"wrote {path}")


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
