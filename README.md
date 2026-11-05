# KernelAgent

KernelAgent is an agent loop that optimises compute kernels from **real** compiler
diagnostics, real benchmark timings and real performance counters, together with
the evaluation harness that proves whether the loop works. It proposes kernel
variants (tile sizes, loop order, unrolling, vectorisation pragmas, blocking and
compiler flags), compiles them, checks them against a NumPy reference, times them,
reads back structured compiler remarks and profiler counters, and feeds that back
into the next proposal. On top of the loop sits a learned cost/ranking model and an
evolutionary search, scored head-to-head against random, greedy, model-guided and
(optionally) LLM proposers, with an exhaustive oracle for small knob spaces.

The default end-to-end path runs on CPU with `clang++`/`g++`, without a GPU or an
API key, and produces real measured numbers. GPU (Triton, CUDA, Nsight Compute)
and LLM backends are implemented and tested with fixtures, and are skipped
automatically when unavailable.

## Architecture

```
                    +--------------------------------------+
                    | proposer: random | greedy |         |
                    | mutation | evolution | guided | LLM |
                    +------------------+-------------------+
                                       | KernelVariant
                    +------------------v-------------------+
                    | AgentLoop.solve                      |
                    | propose -> build -> repair           |
                    | -> correctness gate -> time          |
                    | -> profile -> structured feedback    |
                    +------------------+-------------------+
                                       | typed ToolResults only
                    +------------------v-------------------+
                    | typed tool layer                     |
                    | build | runner | checker | profiler  |
                    +------------------+-------------------+
                                       |
          +----------------------------+----------------------------+
          |                            |                            |
   clang++/g++ C++17           Triton / nvcc / ncu          NumPy / SciPy
   (default, always on)        (availability gated)         (references and
                                                            library baselines)
                                       |
                    +------------------v-------------------+
                    | results/trials.jsonl (run id, git    |
                    | SHA, environment fingerprint)        |
                    +------------------+-------------------+
                                       |
                    +------------------v-------------------+
                    | experiments -> results/summary.json  |
                    | report -> RESULTS.md, figures/,      |
                    | results/regressions.md               |
                    +--------------------------------------+
```

### The typed tool contract

The agent never sees raw shell output. `build`, `run`, `check` and `profile` each
return a pydantic model (`BuildResult`, `RunResult`, `CheckResult`,
`ProfileResult`) and every measurement is captured in a `Trial`. A compile error
is a `BuildResult(ok=False)` with parsed `error:` lines, not an exception and not a
string of shell noise. `run_subprocess` enforces hard timeouts with process-group
kills, retries only transient failures, redacts secrets and truncates output, and
records duration.

This boundary is what makes the loop safe and auditable: a proposed compiler flag
is checked against a whitelist before it reaches the compiler, builds are cached
by a content hash of `(source, harness, compiler, -D macros, flags)`, and the
ledger records the machine and git SHA for every trial. The LLM proposer exposes
the same five operations as JSON-schema tools (`build_variant`, `run_variant`,
`profile_variant`, `check_variant`, `submit_best`), with pydantic validation on
every argument and a hard tool-call budget.

## Task suite

Eight parameterised C++17 tasks, each with a cache-resident small shape and an
out-of-cache large shape (16 problems total). Knob spaces range from 32 to 1728
legal variants.

| Task | Shapes | Knobs (space) | Workload character |
| --- | --- | --- | --- |
| `sgemm_tiled` | 256×256×256, 1024×1024×512 | 7 (1728) | compute-bound dense GEMM |
| `spmv_csr` | 20k×20k, 200k×200k (8 nnz/row) | 4 (36) | memory-bound sparse matvec |
| `softmax_rows` | 512×1024, 256×32768 | 4 (32) | memory-bound row reductions/exp |
| `layernorm_fwd` | 1024×512, 256×16384 | 5 (72) | memory-bound two-pass reduction |
| `conv1d_direct` | 8→16 ch, L=2048, K=5; 4→2 ch, L=4.2M, K=3 | 5 (144) | compute-bound small, bandwidth-bound large |
| `transpose_blocked` | 1024², 4096² | 4 (32) | pure data movement |
| `reduce_sum` | 1024², 2048×16384 | 4 (48) | memory-bound row reduction |
| `bias_relu_fused` | 1024², 4096² | 4 (48) | memory-bound fused elementwise |

Each task directory contains `kernel.cpp` (parameterised entirely by `-D` macros),
`meta.yaml` (shapes, legal knobs, FLOP/byte model, tolerance), `reference.py`
(NumPy ground truth) and shares `tasks/harness.hpp` (deterministic input
generation, self-calibrated timing, MAD/min, checksum, JSON output). Compiler
remarks come from real clang optimization records and `-Rpass` output; the CPU
profiler computes achieved GFLOP/s, GB/s and arithmetic intensity against a
measured STREAM-style bandwidth ceiling. `ncu` counters are used when present.

## How to run

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"

.venv/bin/kernelagent doctor        # capability table for this machine
.venv/bin/kernelagent tasks         # task suite and knob-space sizes
.venv/bin/kernelagent run-suite --strategy default   # baseline table, real timings
.venv/bin/kernelagent search --task sgemm_tiled --shape large \
    --strategy mutation --budget 24                  # one confirmed search episode
.venv/bin/kernelagent rank-eval --collect            # measured LOTO ranking report
.venv/bin/kernelagent run-suite --strategy guided --budget 8 --seeds 0,1,2
make experiments                                   # all four experiments
make report                                        # RESULTS.md + figures + regressions
make smoke                                         # 2 tasks, budget 4, CI smoke
```

Tests and lint:

```bash
make test      # pytest -q -m "not gpu and not llm"
make lint      # ruff + mypy
```

### Enabling the optional backends

GPU:

```bash
.venv/bin/pip install -e ".[gpu]"     # torch + triton
# CUDA C++ and profiling additionally need nvcc and ncu on PATH.
```

The CPU path never imports torch or triton. `CudaBuilder` and `TritonBuilder`
report unavailability through `is_available()`; `NcuProfiler` is selected only
when `ncu` and a CUDA device are genuinely present. GPU tests are marked `gpu`
and skipped otherwise.

LLM:

```bash
.venv/bin/pip install -e ".[llm]"
export ANTHROPIC_API_KEY=...          # or OPENAI_API_KEY
.venv/bin/kernelagent search --task conv1d_direct --shape large \
    --strategy llm --budget 8
# Replay a recorded transcript without any network access:
.venv/bin/kernelagent search --task conv1d_direct --shape large \
    --strategy llm --budget 8 --replay results/transcripts/<file>.json
```

Every live episode is written to `results/transcripts/`; replay returns the
recorded proposal sequence with zero API calls.

## Results

These are measured numbers from `results/summary.json`, produced by
`make experiments && make report` on an Apple M3 (8 cores, 16 GiB) with Apple
clang 21.0.0. The main sweep used **budget 8** per search and **3 seeds**; the
ablation used budget 8 and 2 seeds. Every winning variant was confirmed by three
re-measurements; failures and ties are reported, not hidden. Full tables and the
per-task breakdown are in [`RESULTS.md`](RESULTS.md).

**Scale of evidence.** 16 (task, shape) problems, **2,944 search trials** plus
**824 exhaustive-oracle variants** (3,768 measured variants in total), 402
cache-miss compiles, 216.3 s of compiler wall time and 1,051.1 s of search wall
time. No GPU, `nvcc`, `ncu` or LLM API key was present.

| Strategy | Solve rate | Geomean | Best task | Worst task | Trials to best |
| --- | --- | --- | --- | --- | --- |
| random | 75.0% | 1.835x | 7.694x | 1.001x | 3.9 |
| greedy | 68.8% | 1.326x | 1.973x | 0.999x | 4.8 |
| mutation | 62.5% | 2.020x | 13.606x | 1.000x | 6.5 |
| evolution | 68.8% | **2.111x** | 13.962x | 0.999x | 5.1 |
| guided | 68.8% | 2.096x | 13.634x | 0.987x | 5.4 |

Solve rate is the fraction of problems with a confirmed speedup of at least
1.05x. The large geomeans come from real schedule changes: layernorm reaches
**7.8x** and row-reduction **17.1x** over their defaults; dense GEMM reaches
**2.35x**; convolution, transpose and SpMV improve by 1.16–1.47x. Softmax and
fused bias+ReLU stay at 0.99–1.05x, and the exhaustive oracle shows why (below).

**Learned selection.** Leave-one-task-out over 160 measured candidates gives a
cost-model Kendall tau of **0.121**, top-1 regret **0.234** and NDCG@5 **0.721**
(pairwise ranker: 0.100, 0.225, 0.786). This is weak generalisation, and it costs
the guided strategy: at the same budget it needs **5.38 build requests-to-best
versus 3.94 for random** (cache misses: 0.50 vs 1.50). The model-guided search
still reaches 2.096x geomean, but it does **not** spend fewer compiles here. The
mean oracle gap over 14 small-space problems is **1.5%**; softmax is effectively
optimal in its space (oracle 0.997–0.999x), bias+ReLU small is a genuine search
miss (oracle 1.210x vs 1.046x found), and transpose small has 7.0% headroom.

**Feedback ablation.** Enabling compiler remarks on top of runtime and build
errors adds **+0.558x geomean and +6.2 pp solve rate**. Adding profiler counters
on top *hurts*: **−0.075x geomean and −12.5 pp solve rate** with this deterministic
mutation policy. That is a measured negative result, not a missing run.

**Cost.** Per solved problem, mutation costs 2.47 s of compilation and 12.0 s of
wall time; random costs 5.88 s and 19.4 s; evolution 5.43 s and 17.6 s. No LLM
tokens were spent because no API key was present.

**Reliability.** 0 compile failures in 1,920 exp01 trials; a deliberately broken
syntax probe produced a structured failure; 3/3 repair probes recovered; 3/3
timeout probes were killed and retried (9 attempts, 0.21 s mean wall). Measurement
noise (MAD/median) was p50 0.0084, p90 0.0408, max 0.2419. Repeating the same
strategy/seed/budget twice changed the headline geomean by **1.43%**.

**Regressions (ties/losses).** Four problems stayed below 1.05x: `softmax_rows`
small and large (oracle: 0.999x and 0.997x — no headroom in the space),
`bias_relu_fused` large (oracle 0.963x with 13.4% confirmation spread), and
`bias_relu_fused` small (searched 1.046x vs oracle 1.210x — the budget missed the
best tile/unroll combination). Full diagnoses are in
[`results/regressions.md`](results/regressions.md).

**Backends.** Triton, CUDA/nvcc and Nsight Compute are implemented behind
`is_available()` and fixture-tested, but this machine has no GPU, nvcc or ncu, so
they did not execute. The LLM proposer is implemented, exposed as JSON-schema
tools, and tested fully offline via a scripted transport and transcript replay;
there was no API key, so no live LLM number is reported. No GPU or LLM figure is
substituted.


## Limitations and next steps

- The default path is single-threaded CPU code. It deliberately does not use
  OpenMP or vendor BLAS, because the point is to measure the agent's schedules,
  not a library's; the NumPy/BLAS "library reference" column is reported
  separately and shows where the tuned kernels still lose.
- CPU profiling is derived from the FLOP/byte model and a measured copy
  bandwidth ceiling, not from hardware performance counters. `ncu` counters are
  wired up but were not exercised on this machine.
- The Triton/CUDA builders and the LLM proposer are implemented and
  fixture-tested, but this run had no GPU, no `nvcc`/`ncu`, and no API key, so
  they are unvalidated on hardware/against a live model here. RESULTS.md says so
  explicitly rather than substituting numbers.
- Leave-one-task-out ranking is hard with eight tasks and twenty measured
  candidates per task; the cost model's absolute ranking quality is modest. More
  candidates per problem and richer features (e.g. real ncu counters) are the
  obvious next step.
- Search budgets are intentionally small (a few dozen trials per problem) so the
  full sweep finishes on a laptop. The oracle gap quantifies how much headroom
  the small-space tasks still have.
- The feedback ablation changes what the deterministic mutation proposer reads;
  the LLM proposer was not run, so the ablation is not an LLM prompt-engineering
  study.
