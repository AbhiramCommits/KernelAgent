# Regressions and ties

Every (task, shape) whose best confirmed speedup across all strategies was below the 1.05x solve threshold, with the diagnosis derived from the measured profiler counters.

| Task | Shape | Best strategy | Speedup | Diagnosis |
|---|---|---|---|---|
| bias_relu_fused | large | mutation | 1.030 | no headroom in the legal space: the exhaustive oracle over 48 variants reached only 0.963x (confirmed spread 13.4%) |
| bias_relu_fused | small | mutation | 1.046 | search miss: the exhaustive oracle over 48 legal variants reached 1.210x, 12.9% above the best found; the budget did not locate it |
| softmax_rows | large | random | 1.001 | no headroom in the legal space: the exhaustive oracle over 32 variants reached only 0.997x (confirmed spread 1.2%) |
| softmax_rows | small | evolution | 1.001 | no headroom in the legal space: the exhaustive oracle over 32 variants reached only 0.999x (confirmed spread 0.1%) |

A regression here is a real measurement, not a missing run: every row has at least one correct trial and a confirmed best speedup from exp01.
