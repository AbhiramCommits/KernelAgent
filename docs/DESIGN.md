# KernelAgent design

This document describes the contracts and decisions behind KernelAgent. Measured
results live in `RESULTS.md`; the experiment discussion lives in `REPORT.md`.

## 1. Layers and data flow

```
                    +-----------------------------+
                    |  proposer (random / greedy  |
                    |  mutation / evolution /     |
                    |  guided / LLM tool caller)  |
                    +--------------+--------------+
                                   | KernelVariant
                    +--------------v--------------+
                    |  AgentLoop.solve             |
                    |  propose -> build -> repair  |
                    |  -> check -> time -> profile |
                    |  -> feedback -> propose      |
                    +--------------+--------------+
                                   | typed calls only
                    +--------------v--------------+
                    |  tools/ (build, runner,      |
                    |  checker, profiler, gpu)     |
                    +--------------+--------------+
                                   |
             +---------------------+----------------------+
             |                     |                      |
       clang++/g++ C++17     Triton / nvcc / ncu     NumPy / scipy
       (always available)    (availability-gated)    (references)
                                   |
                    +--------------v--------------+
                    | TrialLedger (JSONL) + env   |
                    | fingerprint + git SHA       |
                    +--------------+--------------+
                                   |
                    +--------------v--------------+
                    | experiments -> summary.json |
                    | report -> RESULTS.md,       |
                    | figures/, regressions.md     |
                    +-----------------------------+
```

## 2. The typed tool contract

Every backend interaction returns a pydantic object from `kernelagent/types.py`:
`BuildResult`, `RunResult`, `ProfileResult`, `CheckResult`, and the composite
`Trial`. Raw compiler or profiler output is captured only inside the tool layer.
`run_subprocess` redacts secrets and the home directory, truncates output, kills
the whole process group on timeout, retries only transient failures, and records
the wall duration. The agent and proposers never receive a `Popen` object, a raw
stderr stream, or a shell string.

Why this matters:

- **Schema stability.** Transcripts, the ledger and report code consume the same
  models, so a parser change cannot silently change the report's shape.
- **Safety.** A proposed compiler flag is validated against a whitelist before it
  reaches the compiler; a hallucinated `-o /somewhere` is rejected.
- **Reproducibility.** A build cache key is a content hash over the kernel source,
  the shared harness, the compiler identity, all `-D` macros and extra flags.
  A cache hit never recompiles and is marked `cached=True`, so aggregate compile
  time is not double counted.
- **Failure is a value, not an exception.** Compile errors become
  `BuildResult(ok=False, ...)` with the first real `error:` lines; correctness
  mismatches become `CheckResult(passed=False, ...)`; no speedup field is set.

`@tool(description)` attaches a JSON schema (`name`, `description`,
`input_schema`) derived from each tool's pydantic argument model. The LLM
proposer exposes exactly five tools:

| Tool | Effect |
| --- | --- |
| `build_variant` | compile one complete knob assignment |
| `run_variant` | compile, check and time |
| `profile_variant` | compile, check, time and profile |
| `check_variant` | compile and compare to the NumPy reference |
| `submit_best` | record the final answer and end the episode |

Arguments are validated with `pydantic` on every call. Tool-call and token
budgets are enforced by the tool agent; every episode is written to
`results/transcripts/` and can be replayed without a network call.

## 3. Benchmarks and diagnostics

A task is a directory with:

- `kernel.cpp` — one compile unit including `../harness.hpp`; every tuning
  decision is a `-D` macro.
- `meta.yaml` — shapes, legal knobs, default baseline, FLOP/byte model, tolerance.
- `reference.py` — NumPy ground truth that reads the exact inputs the C++ binary
  dumped. The checker dumps those inputs once per (task, shape), then compares
  every variant's `out.bin` against the cached reference, so all variants see
  byte-identical data.

The harness self-calibrates an inner iteration count to about 6 ms, then reports
median, MAD and minimum of the timed samples. The runner rejects malformed JSON,
requires strictly positive times, and re-runs once when `MAD / median > 0.15`.

Compiler diagnostics come from three real sources:

1. clang optimization records (`-fsave-optimization-record`), parsed with a fast
   line parser into vectorized-loop counts/widths, missed vectorisations with the
   compiler's own reasons, unroll decisions and inlining notes. A real record can
   exceed 3 MB, so the parser reads only the fields it needs instead of loading
   the whole YAML document.
2. `-Rpass=loop-vectorize` / `-Rpass-missed` / `-Rpass-analysis` remarks, kept in
   `BuildResult.raw_stderr` for the repair and feedback paths.
3. `nvcc -Xptxas -v` or Triton metadata, parsed into registers, spill bytes and
   shared memory.

`DerivedProfiler` is the CPU fallback. It combines the task's FLOP/byte model with
a **measured** STREAM-style copy bandwidth ceiling for the same working-set size,
and reports achieved GFLOP/s, GB/s, arithmetic intensity and the fraction of the
measured ceiling. `ProfileResult.source` is `"derived"` or `"ncu"`; the report
never presents derived numbers as hardware counters. `NcuProfiler` runs real
`ncu --csv` and normalises occupancy, DRAM/SM throughput and warp-stall reasons.

## 4. Feedback schema and ablation

`format_feedback(trial, signals)` emits compact structured text for only the
enabled signal blocks:

```
task=sgemm_tiled shape=M=1024,N=1024,K=512 strategy=mutation trial=...
signal: runtime
  ms_median=... ms_mad=... speedup=...
signal: build_errors
  ok=true error=none compile_s=...
signal: compiler_remarks
  vectorized=... widths=... missed=... unroll_full=... unroll_partial=...
  missed line=... reason=...
signal: profile_counters
  source=derived gflops=... gbps=... bandwidth_pct=... arithmetic_intensity=...
```

The ablation in `exp02` runs the same mutation proposer with
`{runtime}`, `{runtime, build_errors}`,
`{runtime, build_errors, compiler_remarks}` and all four blocks. Because the
proposer only reads this text (plus the typed history), removing a block removes
the information from its decisions; the ablation is not cosmetic.

## 5. Agent loop and repair

`AgentLoop` is a small, explicit state machine:

1. Ask the proposer for candidates.
2. Build the first candidate.
3. On a compile failure, hand the real compiler diagnostics back to the proposer
   and retry up to two times. Repair attempts count against the trial budget.
4. Gate on correctness. An incorrect variant is recorded with no speedup and can
   never become the best trial.
5. Time the variant, profile it, format feedback, repeat.
6. Stop on budget, `patience` consecutive non-improving trials, or a wall-clock
   cap.

`confirm_best` re-runs the winning binary three times after a fresh correctness
check and reports the median, spread and confirmed speedup. The report uses the
confirmed value, not the single-shot number.

## 6. Selection layer

`featurize(task, shape, variant, build)` produces a global feature schema so a
model trained on seven tasks can be applied to the eighth. It includes normalised
knob values and min/max indicators, compiler-flag indicators, vectorisation and
unroll counts, register/spill statistics when present, shape sizes, FLOP/byte
ratio, arithmetic intensity and working-set-over-cache ratio. When `build` is
`None`, build-dependent slots are zero; this is how model-guided search scores
candidates it has not compiled yet.

- `CostModel` is a gradient-boosted regressor on `log(speedup)`.
- `PairwiseRanker` is a logistic model over feature differences from
  within-problem pairs; training always uses both directions of a pair.
- `evaluate.py` performs strict leave-one-task-out evaluation: no task appears in
  both training and test. It reports Spearman, Kendall tau, top-1 regret against
  the measured pool best, NDCG@5 and log-space MAE for the regressor.
- `EvolutionProposer` is a steady-state genetic algorithm with population cap 12,
  tournament selection, uniform knob crossover, a decaying mutation rate,
  elitism (best measured variants are always parents) and deduplication against
  every measured variant.
- `ModelGuidedSearch` builds a large pool (all variants when the space is small,
  otherwise default + knob extremes + seeded random), scores the *unbuilt* pool
  and proposes candidates in ranked order. The top-k proposal order is the
  "spend fewer compiles" strategy; exp03 measures compiles-to-best against random
  search at the same budget.
- `ExhaustiveOracle` enumerates the full legal space for tasks with at most 256
  variants, giving a true upper bound and an oracle gap.

## 7. Optional backends

Triton and CUDA builders are implemented behind `is_available()`, import their
dependencies lazily and are unit-tested with parser fixtures. The LLM proposer
supports Anthropic and OpenAI transports plus an offline `ScriptedTransport` and
transcript replay. On a CPU-only machine without API keys, none of these paths is
entered by the default CLI, and the report states that they were not validated on
hardware.

## 8. Reproducibility and evidence

Every trial is appended to `results/trials.jsonl` with a run id, git SHA and
environment fingerprint. Experiment runs write per-trial JSONL plus a JSON
summary. `kernelagent report` reads only those JSON files, so RESULTS.md cannot
contain a number that was not measured. `results/regressions.md` is generated
from the same data and lists every problem whose best confirmed speedup stayed
below the 1.05x solve threshold, with the diagnosis derived from measured
profiler counters.
