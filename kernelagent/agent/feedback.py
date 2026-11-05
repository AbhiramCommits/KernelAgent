"""Structured feedback channel.

``format_feedback`` is the ablation knob: only the requested signal blocks are
rendered, so disabling compiler remarks or profiler counters genuinely removes
that information from the agent's next prompt.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..types import Trial

SIGNALS = ("runtime", "build_errors", "compiler_remarks", "profile_counters")
DEFAULT_SIGNALS = frozenset(SIGNALS)


def _runtime_block(trial: Trial) -> list[str]:
    if trial.run is None:
        return ["  unavailable"]
    run = trial.run
    speedup = f"{trial.speedup:.4f}" if trial.speedup is not None else "n/a"
    return [
        f"  ms_median={run.ms_median:.6g} ms_mad={run.ms_mad:.6g} "
        f"ms_min={run.ms_min:.6g} reps={run.reps}",
        f"  speedup={speedup} throughput_gflops={run.throughput:.4f}",
    ]


def _build_block(trial: Trial) -> list[str]:
    build = trial.build
    lines = [
        f"  ok={str(build.ok).lower()} error={trial.error or 'none'} "
        f"compile_s={build.compile_time_s:.4f} cached={str(build.cached).lower()}"
    ]
    errors = build.diagnostics.get("compile_errors") or []
    if errors:
        lines.append(f"  compile_errors={len(errors)}")
        for error in errors[:4]:
            lines.append(f"    {error[:240]}")
    if not build.ok and build.raw_stderr:
        for line in build.raw_stderr.splitlines()[:4]:
            if line.strip():
                lines.append(f"    {line.strip()[:240]}")
    return lines


def _compiler_block(trial: Trial) -> list[str]:
    diagnostics = trial.build.diagnostics
    if not diagnostics or diagnostics.get("source") != "opt-record":
        return ["  unavailable (no optimization record)"]
    unroll = diagnostics.get("unroll", {})
    widths = diagnostics.get("vector_widths", [])
    lines = [
        f"  vectorized={int(diagnostics.get('vectorized_loops', 0))} "
        f"widths={','.join(str(w) for w in widths) or 'n/a'} "
        f"missed={int(diagnostics.get('missed_count', 0))} "
        f"unroll_full={int(unroll.get('fully', 0))} "
        f"unroll_partial={int(unroll.get('partially', 0))} "
        f"unroll_not={int(unroll.get('not', 0))}",
    ]
    for item in (diagnostics.get("missed_vectorizations") or [])[:5]:
        lines.append(
            f"  missed line={item.get('line')} reason={str(item.get('reason', ''))[:200]}"
        )
    spills = diagnostics.get("spills")
    if spills:
        lines.append(
            f"  registers={diagnostics.get('registers')} spills={spills} "
            f"shared_bytes={diagnostics.get('shared_bytes')}"
        )
    return lines


def _profile_block(trial: Trial) -> list[str]:
    if trial.profile is None:
        return ["  unavailable"]
    counters = trial.profile.counters
    ordered = (
        "gflops",
        "gbps",
        "bandwidth_pct",
        "arithmetic_intensity",
        "occupancy_pct",
        "dram_throughput_pct",
        "sm_throughput_pct",
        "working_set_over_cache",
    )
    present = [f"{key}={counters[key]:.4g}" for key in ordered if key in counters]
    stalls = {
        key[len("warp_stall_") :]: value
        for key, value in counters.items()
        if key.startswith("warp_stall_")
    }
    if stalls:
        top = sorted(stalls.items(), key=lambda item: item[1], reverse=True)[:3]
        present.append("warp_stalls=" + ",".join(f"{name}:{value:.3g}" for name, value in top))
    lines = [f"  source={trial.profile.source} " + " ".join(present)]
    if trial.profile.notes:
        lines.append(f"  notes={trial.profile.notes[:200]}")
    return lines


def format_feedback(trial: Trial, signals: Iterable[str] | None = None) -> str:
    """Render compact structured feedback for the enabled signals only."""
    enabled = frozenset(signals) if signals is not None else DEFAULT_SIGNALS
    unknown = enabled - set(SIGNALS)
    if unknown:
        raise ValueError(f"unknown feedback signals: {sorted(unknown)}")
    lines = [
        f"task={trial.task} shape={trial.shape.key()} strategy={trial.strategy} "
        f"trial={trial.trial_id} variant={trial.variant.key()}"
    ]
    if "runtime" in enabled:
        lines.append("signal: runtime")
        lines.extend(_runtime_block(trial))
    if "build_errors" in enabled:
        lines.append("signal: build_errors")
        lines.extend(_build_block(trial))
    if "compiler_remarks" in enabled:
        lines.append("signal: compiler_remarks")
        lines.extend(_compiler_block(trial))
    if "profile_counters" in enabled:
        lines.append("signal: profile_counters")
        lines.extend(_profile_block(trial))
    return "\n".join(lines)


def repair_feedback(trial: Trial) -> str:
    """Feedback handed back to a proposer after a compile failure."""
    lines = [
        f"task={trial.task} shape={trial.shape.key()} strategy={trial.strategy}",
        "repair: the previous variant failed to compile; propose a corrected variant",
        "signal: build_errors",
    ]
    lines.extend(_build_block(trial))
    return "\n".join(lines)


def history_summary(trials: list[Trial], limit: int = 6) -> str:
    """Compact strongest-trials summary used in LLM prompts."""
    if not trials:
        return "no trials yet"
    correct = [t for t in trials if t.check is not None and t.check.passed]
    ordered = sorted(
        correct,
        key=lambda t: t.speedup if t.speedup is not None else -1.0,
        reverse=True,
    )[:limit]
    lines = []
    for trial in ordered:
        ms = f"{trial.run.ms_median:.5g}" if trial.run else "n/a"
        speedup = f"{trial.speedup:.4f}" if trial.speedup is not None else "n/a"
        lines.append(f"- {trial.variant.key()} ms={ms} speedup={speedup}")
    failed = [t for t in trials if t.error is not None][:3]
    for trial in failed:
        lines.append(f"- FAILED {trial.variant.key()} error={str(trial.error)[:160]}")
    return "\n".join(lines)
