"""Kernel variant proposers: random, greedy, feedback-driven and LLM.

A proposer only sees typed history and a structured feedback string; it never
sees raw process output. That keeps the feedback ablation honest: removing a
signal block removes the information from the proposer's input.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Protocol

from ..tasks.registry import default_variant
from ..types import KernelTask, KernelVariant, Shape, Trial

FLAG_POOL = ["-funroll-loops", "-fomit-frame-pointer", "-ffast-math"]


class Proposer(Protocol):
    """Protocol every proposer implements."""

    name: str

    def propose(
        self,
        task: KernelTask,
        shape: Shape,
        history: list[Trial],
        feedback: str,
    ) -> list[KernelVariant]: ...


def _correct(history: list[Trial]) -> list[Trial]:
    return [t for t in history if t.check is not None and t.check.passed and t.run is not None]


def best_so_far(history: list[Trial]) -> Trial | None:
    """Highest-speedup correct trial in the history."""
    candidates = [t for t in _correct(history) if t.speedup is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda t: t.speedup or 0.0)


def _remember(tried: set[str], variant: KernelVariant) -> KernelVariant:
    tried.add(variant.key())
    return variant


def sample_variant(
    task: KernelTask,
    rng: random.Random,
    tried: set[str],
    provenance: str = "random",
    allow_flags: bool = True,
) -> KernelVariant:
    """Uniform sample from the legal knob space, avoiding already-tried keys."""
    for _ in range(64):
        knobs = {name: rng.choice(spec.values) for name, spec in task.knobs.items()}
        cflags: list[str] = []
        if allow_flags and rng.random() < 0.2:
            cflags = [rng.choice(FLAG_POOL)]
        variant = KernelVariant(knobs=knobs, cflags=cflags, provenance=provenance)
        if variant.key() not in tried:
            return _remember(tried, variant)
    return _remember(
        tried,
        KernelVariant(knobs=dict(task.default_knobs), provenance=f"{provenance}-fallback"),
    )


class RandomProposer:
    """Uniform random search over the legal knob space."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.tried: set[str] = set()

    def propose(self, task, shape, history, feedback):  # type: ignore[no-untyped-def]
        return [sample_variant(task, self.rng, self.tried, "random")]


class GreedyHillClimb:
    """One-knob-at-a-time local search with random restarts."""

    name = "greedy"

    def __init__(self, seed: int = 0, max_restarts: int = 3) -> None:
        self.rng = random.Random(seed)
        self.tried: set[str] = set()
        self.current: KernelVariant | None = None
        self.queue: list[KernelVariant] = []
        self.max_restarts = max_restarts
        self.restarts = 0

    def propose(self, task, shape, history, feedback):  # type: ignore[no-untyped-def]
        best = best_so_far(history)
        if self.current is None and not history:
            self.current = default_variant(task)
            self.tried.add(self.current.key())
            self.queue = self._neighbours(task, self.current)
            return [self.current]
        if best is not None and (self.current is None or best.variant.key() != self.current.key()):
            self.current = best.variant
            self.queue = self._neighbours(task, best.variant)
        while self.queue:
            candidate = self.queue.pop(0)
            if candidate.key() not in self.tried:
                self.tried.add(candidate.key())
                return [candidate]
        self.restarts += 1
        restart = sample_variant(task, self.rng, self.tried, "greedy-restart")
        self.current = restart
        self.queue = self._neighbours(task, restart)
        return [restart]

    def _neighbours(self, task: KernelTask, variant: KernelVariant) -> list[KernelVariant]:
        neighbours: list[KernelVariant] = []
        for name, spec in task.knobs.items():
            for value in spec.values:
                if value == variant.knobs[name]:
                    continue
                neighbours.append(
                    KernelVariant(
                        knobs={**variant.knobs, name: value},
                        cflags=list(variant.cflags),
                        provenance=f"greedy:{name}",
                    )
                )
        for flag in (None, *FLAG_POOL):
            cflags = [] if flag is None else [flag]
            if cflags != list(variant.cflags):
                neighbours.append(
                    KernelVariant(
                        knobs=dict(variant.knobs),
                        cflags=cflags,
                        provenance="greedy:flags",
                    )
                )
        return neighbours


_NUMBER_RE = re.compile(r"([A-Za-z_]+)=(-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")


def _numbers(feedback: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for match in _NUMBER_RE.finditer(feedback):
        try:
            values[match.group(1)] = float(match.group(2))
        except ValueError:
            continue
    return values


def _other_values(task: KernelTask, name: str, current: int | str) -> list[int | str]:
    return [value for value in task.knobs[name].values if value != current]


def _step_up(task: KernelTask, knobs: dict[str, int | str], names: list[str]) -> str | None:
    for name in names:
        if name not in knobs:
            continue
        spec = task.knobs[name]
        values = spec.values
        if not all(isinstance(v, int) for v in values):
            continue
        current = knobs[name]
        larger = [int(v) for v in values if int(v) > int(current)]
        if larger:
            knobs[name] = min(larger)
            return name
    return None


def _step_down(task: KernelTask, knobs: dict[str, int | str], names: list[str]) -> str | None:
    for name in names:
        if name not in knobs:
            continue
        values = [int(v) for v in task.knobs[name].values if isinstance(v, int)]
        current = knobs[name]
        smaller = [v for v in values if v < int(current)]
        if smaller:
            knobs[name] = max(smaller)
            return name
    return None


class MutationProposer:
    """Deterministic, feedback-driven mutation rules.

    Each rule targets one diagnosed bottleneck: missed vectorisation, measured
    bandwidth, arithmetic intensity, register spills/occupancy, compile errors,
    or a fallback systematic knob sweep.
    """

    name = "mutation"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.tried: set[str] = set()
        self._sweep_cursor = 0
        self._last_rule = ""

    def propose(self, task, shape, history, feedback):  # type: ignore[no-untyped-def]
        base = best_so_far(history)
        base_variant = base.variant if base is not None else default_variant(task)
        numbers = _numbers(feedback)

        candidate: KernelVariant | None = None
        rule = ""

        if "repair:" in feedback:
            knobs = dict(task.default_knobs)
            candidate = KernelVariant(knobs=knobs, cflags=[], provenance="mutation:repair")
            if candidate.key() in self.tried:
                for name, _spec in task.knobs.items():
                    alternatives = _other_values(task, name, knobs[name])
                    if alternatives:
                        candidate = KernelVariant(
                            knobs={**knobs, name: alternatives[0]},
                            cflags=[],
                            provenance="mutation:repair",
                        )
                        break
            rule = "repair"
        else:
            candidates = self._candidates(task, base_variant, numbers)
            for option, rule_name in candidates:
                if option.key() not in self.tried:
                    candidate, rule = option, rule_name
                    break

        if candidate is None or candidate.key() in self.tried:
            candidate = self._sweep(task, base_variant)
            rule = rule or "sweep"
        if candidate.key() in self.tried:
            candidate = sample_variant(task, self.rng, self.tried, "mutation-fallback")
            rule = "fallback"
        self._last_rule = rule
        candidate = candidate.model_copy(update={"provenance": f"mutation:{rule}"})
        return [_remember(self.tried, candidate)]

    def _candidates(
        self, task: KernelTask, base: KernelVariant, numbers: dict[str, float]
    ) -> list[tuple[KernelVariant, str]]:
        knobs = dict(base.knobs)
        cflags = list(base.cflags)
        options: list[tuple[KernelVariant, str]] = []

        missed = int(numbers.get("missed", 0))
        vectorized = int(numbers.get("vectorized", 0))
        bandwidth_pct = numbers.get("bandwidth_pct")
        intensity = numbers.get("arithmetic_intensity")
        spills = numbers.get("spills", 0.0)
        occupancy = numbers.get("occupancy_pct")
        unroll_not = int(numbers.get("unroll_not", 0))

        if missed > 0 and "PRAGMA_VECTORIZE" in knobs and int(knobs["PRAGMA_VECTORIZE"]) == 0:
            options.append(
                (
                    KernelVariant(knobs={**knobs, "PRAGMA_VECTORIZE": 1}, cflags=cflags),
                    "vectorize-pragma",
                )
            )
        if missed > 0 and "LOOP_ORDER" in knobs and int(knobs["LOOP_ORDER"]) != 1:
            options.append(
                (KernelVariant(knobs={**knobs, "LOOP_ORDER": 1}, cflags=cflags), "loop-order")
            )
        if (occupancy is not None and occupancy < 30.0) or spills > 0:
            shrunk = dict(knobs)
            changed = _step_down(task, shrunk, ["TILE_M", "TILE_N", "TILE_K", "BLOCK", "BLOCK_L"])
            if changed:
                options.append((KernelVariant(knobs=shrunk, cflags=cflags), f"shrink-{changed}"))
        blocking = ["BLOCK", "TILE_M", "TILE_N", "TILE_K", "BLOCK_L", "ACCUMULATORS"]
        if bandwidth_pct is None or bandwidth_pct < 40.0:
            for name in blocking:
                if name not in knobs:
                    continue
                stepped = dict(knobs)
                if _step_up(task, stepped, [name]) is not None:
                    options.append(
                        (KernelVariant(knobs=stepped, cflags=cflags), f"increase-{name}")
                    )
        if intensity is not None and intensity < 3.0:
            for name in ("ACCUMULATORS", "UNROLL"):
                if name not in knobs:
                    continue
                stepped = dict(knobs)
                if _step_up(task, stepped, [name]) is not None:
                    options.append((KernelVariant(knobs=stepped, cflags=cflags), f"ilp-{name}"))
        if unroll_not > 0 and "UNROLL" in knobs:
            stepped = dict(knobs)
            if _step_up(task, stepped, ["UNROLL"]) is not None:
                options.append((KernelVariant(knobs=stepped, cflags=cflags), "unroll"))
        if missed > 0 and vectorized == 0 and "UNROLL" in knobs:
            options.append(
                (
                    KernelVariant(
                        knobs={
                            **knobs,
                            "UNROLL": max(int(v) for v in task.knobs["UNROLL"].values),
                        },
                        cflags=cflags,
                    ),
                    "max-unroll",
                )
            )
        return options

    def _sweep(self, task: KernelTask, base: KernelVariant) -> KernelVariant:
        options: list[tuple[str, int | str]] = []
        for name, spec in task.knobs.items():
            for value in spec.values:
                if value != base.knobs[name]:
                    options.append((name, value))
        for offset in range(len(options)):
            index = (self._sweep_cursor + offset) % len(options)
            name, value = options[index]
            candidate = KernelVariant(
                knobs={**base.knobs, name: value},
                cflags=list(base.cflags),
                provenance="mutation:sweep",
            )
            if candidate.key() not in self.tried:
                self._sweep_cursor = (index + 1) % len(options)
                return candidate
        for flag in FLAG_POOL:
            if flag not in base.cflags:
                candidate = KernelVariant(
                    knobs=dict(base.knobs), cflags=[*base.cflags, flag], provenance="mutation:flags"
                )
                if candidate.key() not in self.tried:
                    return candidate
        return sample_variant(task, self.rng, self.tried, "mutation-sweep-random")


class LLMProposer:
    """Tool-calling LLM proposer with transcript replay support."""

    name = "llm"

    def __init__(
        self,
        seed: int = 0,
        replay: str | Path | None = None,
        provider: str | None = None,
        model: str | None = None,
        max_tool_calls: int = 8,
        transcript_dir: Path | None = None,
    ) -> None:
        from .llm_agent import load_transcript

        self.seed = seed
        self.replay = Path(replay) if replay else None
        self.transcript_dir = transcript_dir
        self.max_tool_calls = max_tool_calls
        self.tokens = 0
        self._replay_proposals: list[KernelVariant] = []
        self._replay_cursor = 0
        if self.replay is not None:
            payload = load_transcript(self.replay)
            self._replay_proposals = [
                KernelVariant.model_validate(item) for item in payload.get("proposals", [])
            ]
            self.tokens = int(payload.get("tokens", 0))
            self.model = str(payload.get("model", "replay"))
            self.transport = None
        else:
            from .llm_agent import build_transport

            chosen_provider = provider or self._detect_provider()
            if chosen_provider is None:
                raise RuntimeError(
                    "no LLM provider configured: set ANTHROPIC_API_KEY or OPENAI_API_KEY, "
                    "or pass --replay <transcript>"
                )
            self.transport, self.model = build_transport(chosen_provider, model)

    @staticmethod
    def available() -> bool:
        """True when an API key is configured (replay is always available)."""
        import os

        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))

    def _detect_provider(self) -> str | None:
        import os

        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        return None

    def propose(self, task, shape, history, feedback):  # type: ignore[no-untyped-def]
        if self.replay is not None:
            if self._replay_cursor >= len(self._replay_proposals):
                return []
            variant = self._replay_proposals[self._replay_cursor]
            self._replay_cursor += 1
            return [variant]
        from .llm_agent import ToolAgent, save_transcript

        agent = ToolAgent(
            task=task,
            shape=shape,
            transport=self.transport,  # type: ignore[arg-type]
            max_tool_calls=self.max_tool_calls,
            seed=self.seed,
            model=self.model,
        )
        variant, transcript = agent.run(feedback, history)
        self.tokens += agent.tokens
        save_transcript(transcript, self.transcript_dir)
        if variant is None:
            return []
        return [variant]


def make_proposer(strategy: str, seed: int = 0, replay: str | None = None) -> Proposer:
    """Construct a proposer for a CLI strategy name."""
    if strategy == "random":
        return RandomProposer(seed)
    if strategy == "greedy":
        return GreedyHillClimb(seed)
    if strategy == "mutation":
        return MutationProposer(seed)
    if strategy == "llm":
        return LLMProposer(seed=seed, replay=replay)
    if strategy == "evolution":
        from ..select.evolution import EvolutionProposer

        return EvolutionProposer(seed=seed)
    if strategy == "guided":
        from ..select.guided import GuidedProposer

        return GuidedProposer(seed=seed)
    raise ValueError(f"unknown strategy {strategy!r}")
