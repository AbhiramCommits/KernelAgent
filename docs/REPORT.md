# KernelAgent experiment report

## 1. Setup

All experiments run through `make experiments`, which executes `exp01`–`exp04` in
order and then assembles `results/summary.json`. Every search uses the same
`AgentLoop`, the same typed tool layer, the same task suite and the same machine.
The machine, compiler, git SHA and environment fingerprint are embedded in
`results/summary.json` and repeated in `RESULTS.md`.

**Protocol.**

- **Equal budget.** Every strategy gets the same number of evaluated variants per
  (task, shape) problem. A failed build or a correctness failure consumes budget;
  it is a real cost of the agent.
- **Confirmed winners.** The best variant each strategy finds per problem is
  re-measured with `confirm_best(repeats=3)` after a fresh correctness check.
  Reported speedups are the confirmed median, with spread recorded.
- **Correctness gate.** A variant that fails `CorrectnessChecker` is recorded with
  `speedup=None` and can never become the best trial.
- **Seeds.** The main strategy sweep uses three seeds; the feedback ablation uses
  two. Each seed is an independent proposer random stream (the schedule and task
  definitions are deterministic).
- **Costs.** "Compiles" counts cache misses only, so a cached rebuild is not
  charged as compile time. Wall time includes builds, correctness checks, timings
  and profiling.
- **Leave-one-task-out.** `rank-eval` fits the cost model and pairwise ranker on
  all tasks except one and ranks the held-out task's measured candidate pool. The
  code asserts that the held-out task never appears in the training set.

The exact budgets are recorded in `results/summary.json` (`budgets`).

## 2. What is measured

- **Solve rate:** fraction of problems whose confirmed best speedup is at least
  1.05x over the measured default variant.
- **Geomean speedup:** geometric mean of confirmed speedups over problems with a
  correct trial.
- **Trials-to-best / compiles-to-best:** trial index and cache-miss compile count
  up to the first trial attaining the run's best speedup.
- **Cost:** compile seconds and wall seconds per solved problem, and tokens per
  solved problem for LLM runs.
- **Ablation:** solve rate and geomean as each feedback signal set is enabled.
- **Ranking:** Spearman, Kendall tau, top-1 regret against the measured pool best,
  NDCG@5 and log-space MAE under leave-one-task-out.
- **Reliability:** compile failure rate, repair success, deliberate timeout/retry
  probes, MAD/median noise distribution and a fixed-seed rerun of the headline
  metric.

## 3. Results discussion

The complete tables are in [`RESULTS.md`](../RESULTS.md). The discussion below
refers only to measured values in `results/summary.json` and the per-experiment
JSON/JSONL files.

### Strategy comparison

At budget 8 with three seeds, evolution has the best geomean (**2.111x**),
followed closely by guided (2.096x), mutation (2.020x), random (1.835x) and
greedy (1.326x). Random has the highest solve rate (**75.0%**) because it happens
to solve `transpose_blocked` small, which the other strategies miss by 1–4%; it
pays for that with a much lower geomean. Evolution, guided and mutation solve
68.8%, and greedy 68.8%.

The geomean is dominated by two genuine schedule wins:

- `layernorm_fwd`: **7.8x** small and **7.1–7.6x** large. The default does three
  scalar reduction passes; the winners combine `TWO_PASS=1`, multiple
  accumulators and vectorisation.
- `reduce_sum`: **17.1x** small and **11.5x** large. The default row loop is
  scalar; independent accumulators plus vectorisation turn a ~22 ms large-shape
  baseline into ~1.9 ms, close to the measured copy-bandwidth ceiling.

Dense GEMM gains **2.35x** from larger tiles and unrolling, convolution
**1.47x** and **1.18x**, transpose **1.39x** large, and SpMV **1.17x** large.
`softmax_rows` and `bias_relu_fused` are the regressions: their best confirmed
speedups stay at ~1.0x.

Greedy is the clearest failure of a simple baseline: it finds at most 1.97x on
layernorm and reduce, because one-knob-at-a-time moves cannot combine the
independent changes (pass fusion, accumulator count, vectorisation) required for
the big wins. Mutation and evolution make multi-knob jumps and find them.

### Learned selection and oracle

Leave-one-task-out ranking is weak: the cost model reaches Kendall tau 0.121,
top-1 regret 0.234, NDCG@5 0.721 and log MAE 0.460; the pairwise ranker reaches
Kendall 0.100, regret 0.225 and NDCG 0.786. With eight tasks and twenty measured
candidates per task, static features do not generalise well across tasks. This
shows up directly in the equal-budget guided comparison: guided needs 5.38 build
requests to reach its best versus 3.94 for random (cache misses 0.50 vs 1.50).
Guided's geomean is still high (2.096x) because its candidate pool contains the
same high-value regions, but the claim that the model saves compiles is **not
supported by this experiment**.

The exhaustive oracle evaluated 824 variants across the 14 problems with at most
256 legal variants. The mean gap between the best found and the oracle is 1.48%,
but the distribution matters:

- `softmax_rows` small/large: oracle 0.999x / 0.997x. There is no headroom in
  this knob space; the default is effectively optimal. This is why softmax is a
  regression rather than a search failure.
- `bias_relu_fused` small: oracle 1.210x versus 1.046x found (12.9% gap). Here
  the search genuinely missed a better tile/unroll combination within budget.
- `transpose_blocked` small: oracle 1.134x versus 1.055x found (7.0% gap).
- On ten problems the confirmed best found matched or exceeded the confirmed
  oracle within noise.

Oracle winners are confirmed three times like any other winner, but tiny kernels
are noisy: bias+ReLU large had a 13.4% confirmation spread and its confirmed
oracle (0.963x) fell below the baseline. Gaps of a few percent should therefore be
read as "within measurement noise", not as precise headroom.

### Feedback ablation

With the deterministic mutation proposer and two seeds:

- `runtime` -> `runtime + build_errors`: no solve-rate change, geomean −0.011x.
- adding `compiler_remarks`: **+6.2 pp solve rate, +0.558x geomean**. This is the
  largest measured benefit from a feedback signal; the proposer uses the real
  missed-vectorisation reasons to change the vectorisation/loop-order decisions.
- adding `profile_counters` on top: **−12.5 pp solve rate, −0.075x geomean**. The
  bandwidth-driven mutation rule raises blocking on problems where that is not the
  bottleneck, and the extra signal changes the proposal sequence enough to lose
  two solves. With only two seeds this negative delta is itself uncertain, but it
  is a real measured outcome of this policy and is reported as such.

### Reliability and variance

Across 1,920 exp01 trials there were **zero compile failures**; the deliberate
syntax-error probe returned a structured failure and the repair probes recovered
in **3/3** runs. Three deliberate sleep probes all timed out and were killed after
9 total attempts (mean wall 0.21 s for a 2 s sleep), confirming process-group
timeout behaviour rather than just unit-testing it.

Measurement noise (MAD/median over all trials) was p50 0.0084, p90 0.0408, max
0.2419; only 2 of 1,920 trials crossed the 0.15 re-run threshold. Repeating the
same strategy, seed and budget on four representative problems changed the
headline geomean by **1.43%**. That is the right scale for interpreting the
1.0–1.2x differences in the regression table: they are close to the noise floor,
while the 2–17x wins are far outside it.

### Failure modes and interpretation

- **No-headroom problems.** Softmax is compute/exp-bound and the legal knobs do
  not change that; the oracle proves the default is at the top of the space.
- **Search misses.** Bias+ReLU small has an oracle result 15.7% faster than the best found that the
  budget of 8 did not locate. A larger budget or a better warm start would be the fix.
- **Noisy tiny kernels.** At ~0.06 ms per call, process-level frequency and
  scheduling effects can move a measured speedup by more than 10%; confirmation
  spread exposes this.
- **Cache-sharing in compile counts.** Experiments share a content-hash build
  cache, so a strategy that runs later can reuse an earlier strategy's binaries.
  The compiles-to-best analysis therefore counts build requests (one per trial)
  as the fair equal-budget measure and reports cache misses separately.
- **Model-guided search is not automatically cheaper.** The learned ranker's
  top-1 regret (~23%) is large enough that ranking unbuilt candidates is worse
  than uniform sampling at this data scale.


## 4. Failure modes observed

- **Compile failures.** Every compiler error is captured as a failed
  `BuildResult` with real `error:` lines. The repair path hands those lines back
  to the proposer; the dedicated repair probe in `exp04` measures whether a
  deliberately broken variant is recovered. The main sweep's compile-failure
  count is reported in RESULTS.md rather than assumed to be zero.
- **Correctness failures.** Out-of-range/illegal knob values are rejected before
  compilation; every legal point is checked against the NumPy reference with
  per-task `rtol`/`atol`, NaN/Inf detection, and a second run for determinism.
  Fast-math flags can change results; when they do, the variant is discarded.
- **Noise.** The harness self-calibrates to roughly 6 ms per timed sample and
  reports MAD/median. The runner re-runs once when MAD/median exceeds 0.15.
  `exp04` reports the observed distribution and a fixed-seed rerun; the rerun is
  the honest estimate of the headline metric's run-to-run variance.
- **Slow variants.** A bad schedule (e.g. an axpy convolution order with too many
  passes over a large tensor) can be tens of times slower than the baseline. This
  is a real property of the space, not a measurement artifact; it also means
  wall-clock cost per trial varies substantially by strategy.
- **Model error.** Leave-one-task-out ranking with eight tasks is a small-sample
  problem. The cost model's top-1 regret is measured, not hidden; the oracle and
  the equal-budget random comparison bound how much that error costs in practice.
- **Cache and compilation reuse.** Content-hash caching means a variant measured
  by an earlier strategy is not recompiled. Compile-time totals therefore reflect
  new code only, while wall time and trial counts reflect the full search.

## 5. Threats to validity

- **Single machine, single compiler.** All numbers come from one Apple M3 with
  Apple clang. Apple's clang differs from upstream LLVM in code generation and
  remark text; absolute speedups may not transfer.
- **Synthetic tasks.** The tasks are small, self-contained and single-threaded.
  They exercise real compiler behaviour, but they are not a production workload.
  The sparse matrix uses a deterministic pattern rather than a real matrix, so
  cache behaviour is representative but not universal.
- **Small budgets.** Budgets of a few dozen trials are enough to compare
  strategies but not to saturate the space. Oracle gaps quantify the remaining
  headroom for the small-space tasks; large spaces have no oracle.
- **Derived CPU counters.** Achieved GFLOP/s and bandwidth percentages are model
  based (FLOP/byte model plus a measured copy bandwidth ceiling), not hardware
  counters. They are labelled `source=derived` throughout; ncu is the only path
  that produces hardware counters.
- **Library comparisons.** NumPy/BLAS timings are a separate column and are not
  the scoring baseline. BLAS is multi-threaded while the C++ kernels are
  single-threaded, so the comparison is intentionally unfavourable to the agent.
- **LLM and GPU backends.** No API key, GPU, nvcc or ncu was present. Those paths
  were exercised only by offline fixtures and unit tests; they are not part of
  any measured claim in RESULTS.md.
- **LLM ablation.** The feedback ablation uses the deterministic mutation
  proposer, so it measures the value of the signal schema, not prompt quality.
