"""Evolutionary search over knob dictionaries.

A steady-state genetic algorithm with population cap 12, tournament selection,
uniform knob crossover, a decaying mutation rate, elitism (the best measured
variants are retained in every generation) and deduplication against every
variant already measured.
"""

from __future__ import annotations

import random

from ..tasks.registry import default_variant
from ..types import KernelTask, KernelVariant, Trial


def _population(history: list[Trial], size: int) -> list[Trial]:
    best: dict[str, Trial] = {}
    for trial in history:
        if trial.check is None or not trial.check.passed or trial.speedup is None:
            continue
        current = best.get(trial.variant.key())
        if current is None or (trial.speedup or 0.0) > (current.speedup or 0.0):
            best[trial.variant.key()] = trial
    return sorted(best.values(), key=lambda trial: trial.speedup or 0.0, reverse=True)[:size]


class EvolutionProposer:
    """Proposer adapter around the evolutionary schedule."""

    name = "evolution"

    def __init__(
        self,
        seed: int = 0,
        population_size: int = 12,
        tournament_size: int = 3,
        immigration: float = 0.25,
    ) -> None:
        self.rng = random.Random(seed)
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.immigration = immigration
        self.tried: set[str] = set()
        self.generation = 0

    def propose(self, task, shape, history, feedback):  # type: ignore[no-untyped-def]
        measured = {trial.variant.key() for trial in history}
        population = _population(history, self.population_size)
        if not population:
            variant = default_variant(task)
            if variant.key() in measured:
                return [self._random(task, "evolution:seed", measured)]
            return [self._remember(variant)]
        if len(population) < 2:
            return [self._random(task, "evolution:seed", measured)]

        # Immigrants keep the population from collapsing early.
        if len(population) < self.population_size and self.rng.random() < self.immigration:
            return [self._random(task, "evolution:immigrant", measured)]

        first = self._tournament(population)
        second = self._tournament(population)
        knobs = {
            name: (
                first.variant.knobs[name]
                if self.rng.random() < 0.5
                else second.variant.knobs[name]
            )
            for name in task.knobs
        }
        rate = max(0.05, 0.40 * (0.85 ** (self.generation / max(1, self.population_size))))
        for name, spec in task.knobs.items():
            if self.rng.random() < rate:
                alternatives = [value for value in spec.values if value != knobs[name]]
                if alternatives:
                    knobs[name] = self.rng.choice(alternatives)
        cflags = list(first.variant.cflags if self.rng.random() < 0.5 else second.variant.cflags)
        if self.rng.random() < rate / 2:
            from ..config import CANDIDATE_FLAGS

            cflags = [self.rng.choice(CANDIDATE_FLAGS)]
        child = KernelVariant(knobs=knobs, cflags=cflags, provenance="evolution")
        self.generation += 1
        if child.key() in self.tried or any(
            child.key() == trial.variant.key() for trial in history
        ):
            for _ in range(32):
                child = self._crossover_or_mutate(task, first, second, rate)
                if child.key() not in self.tried and not any(
                    child.key() == trial.variant.key() for trial in history
                ):
                    break
            else:
                child = self._random(task, "evolution:dedup-fallback", measured)
        return [self._remember(child)]

    def _crossover_or_mutate(
        self, task: KernelTask, first: Trial, second: Trial, rate: float
    ) -> KernelVariant:
        knobs = {
            name: (
                first.variant.knobs[name]
                if self.rng.random() < 0.5
                else second.variant.knobs[name]
            )
            for name in task.knobs
        }
        for name, spec in task.knobs.items():
            if self.rng.random() < rate:
                alternatives = [value for value in spec.values if value != knobs[name]]
                if alternatives:
                    knobs[name] = self.rng.choice(alternatives)
        return KernelVariant(knobs=knobs, cflags=list(first.variant.cflags), provenance="evolution")

    def _tournament(self, population: list[Trial]) -> Trial:
        contenders = self.rng.sample(population, min(self.tournament_size, len(population)))
        return max(contenders, key=lambda trial: trial.speedup or 0.0)

    def _random(
        self, task: KernelTask, provenance: str, measured: set[str] | None = None
    ) -> KernelVariant:
        from ..agent.proposer import sample_variant

        blocked = measured or set()
        for _ in range(64):
            candidate = sample_variant(task, self.rng, self.tried, provenance)
            if candidate.key() not in blocked:
                return candidate
        return KernelVariant(knobs=dict(task.default_knobs), provenance=provenance)

    def _remember(self, variant: KernelVariant) -> KernelVariant:
        self.tried.add(variant.key())
        return variant
