# KernelAgent

KernelAgent is an agent loop that optimises compute kernels using real compiler
diagnostics, real benchmark timings and real performance counters, plus the
evaluation harness that scores the loop against random, greedy, evolutionary and
model-guided search.

The default path runs on CPU with `clang++`/`g++` and needs no GPU or API key.
GPU (Triton, CUDA, Nsight Compute) and LLM backends are implemented but gated on
availability.

Full documentation and measured results are added in later steps.
