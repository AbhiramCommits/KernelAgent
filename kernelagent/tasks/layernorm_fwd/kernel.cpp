// layernorm_fwd: row-wise layer normalisation followed by an affine transform.
#include "../harness.hpp"

#ifndef UNROLL
#define UNROLL 2
#endif
#ifndef TWO_PASS
#define TWO_PASS 0
#endif
#ifndef ACCUMULATORS
#define ACCUMULATORS 1
#endif
#ifndef USE_RESTRICT
#define USE_RESTRICT 1
#endif
#ifndef PRAGMA_VECTORIZE
#define PRAGMA_VECTORIZE 0
#endif

#if USE_RESTRICT
#define KA_RESTRICT __restrict
#else
#define KA_RESTRICT
#endif

#if PRAGMA_VECTORIZE
#define KA_VEC _Pragma("clang loop vectorize(enable)")
#else
#define KA_VEC
#endif

#if UNROLL > 1
#define KA_UNROLL _Pragma("clang loop unroll_count(UNROLL)")
#else
#define KA_UNROLL _Pragma("clang loop unroll(disable)")
#endif

static inline float row_sum(const float* KA_RESTRICT p, int n) {
#if ACCUMULATORS == 1
  float s0 = 0.0f;
  KA_UNROLL
  KA_VEC
  for (int i = 0; i < n; ++i) s0 += p[i];
  return s0;
#elif ACCUMULATORS == 2
  float s0 = 0.0f, s1 = 0.0f;
  int i = 0;
  for (; i + 1 < n; i += 2) { s0 += p[i]; s1 += p[i + 1]; }
  if (i < n) s0 += p[i];
  return s0 + s1;
#else
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  int i = 0;
  for (; i + 3 < n; i += 4) {
    s0 += p[i]; s1 += p[i + 1]; s2 += p[i + 2]; s3 += p[i + 3];
  }
  for (; i < n; ++i) s0 += p[i];
  return (s0 + s1) + (s2 + s3);
#endif
}

static inline float row_sumsq(const float* KA_RESTRICT p, int n, float mean) {
#if ACCUMULATORS == 1
  float s0 = 0.0f;
  KA_UNROLL
  KA_VEC
  for (int i = 0; i < n; ++i) {
    const float d = p[i] - mean;
    s0 += d * d;
  }
  return s0;
#else
  float s0 = 0.0f, s1 = 0.0f;
  int i = 0;
  for (; i + 1 < n; i += 2) {
    const float d0 = p[i] - mean;
    const float d1 = p[i + 1] - mean;
    s0 += d0 * d0;
    s1 += d1 * d1;
  }
  if (i < n) {
    const float d = p[i] - mean;
    s0 += d * d;
  }
  return s0 + s1;
#endif
}

static void layernorm(int rows, int cols, float eps, const float* KA_RESTRICT x,
                      const float* KA_RESTRICT gamma, const float* KA_RESTRICT beta,
                      float* KA_RESTRICT out) {
  for (int r = 0; r < rows; ++r) {
    const float* KA_RESTRICT row = x + static_cast<size_t>(r) * cols;
    float* KA_RESTRICT dst = out + static_cast<size_t>(r) * cols;
    const float mean = row_sum(row, cols) / static_cast<float>(cols);
#if TWO_PASS == 1
    // One variance pass (biased), then normalise: two reads of the row.
    float sq = 0.0f;
    KA_UNROLL
    KA_VEC
    for (int c = 0; c < cols; ++c) {
      const float d = row[c] - mean;
      sq += d * d;
    }
    const float var = sq / static_cast<float>(cols);
#else
    const float var = row_sumsq(row, cols, mean) / static_cast<float>(cols);
#endif
    const float inv = 1.0f / std::sqrt(var + eps);
    KA_UNROLL
    KA_VEC
    for (int c = 0; c < cols; ++c) {
      dst[c] = (row[c] - mean) * inv * gamma[c] + beta[c];
    }
  }
}

int main(int argc, char** argv) {
  ka::Args args = ka::parse_args(argc, argv);
  const int ROWS = static_cast<int>(args.dim("ROWS", 1024));
  const int COLS = static_cast<int>(args.dim("COLS", 512));
  const float EPS = 1e-5f;

  std::vector<float> x(static_cast<size_t>(ROWS) * COLS);
  std::vector<float> gamma(static_cast<size_t>(COLS));
  std::vector<float> beta(static_cast<size_t>(COLS));
  std::vector<float> out(x.size(), 0.0f);

  ka::load_or_random(x.data(), x.size(), args.input_dir, "x.bin", 3141);
  ka::load_or_random(gamma.data(), gamma.size(), args.input_dir, "gamma.bin", 2718);
  ka::load_or_random(beta.data(), beta.size(), args.input_dir, "beta.bin", 1618);
  // Dump the raw affine parameters; the kernel and the reference both apply the
  // same deterministic transform to them.
  ka::maybe_dump_input(args, "x.bin", x.data(), x.size() * sizeof(float));
  ka::maybe_dump_input(args, "gamma.bin", gamma.data(), gamma.size() * sizeof(float));
  ka::maybe_dump_input(args, "beta.bin", beta.data(), beta.size() * sizeof(float));
  for (int c = 0; c < COLS; ++c) {
    gamma[c] = 1.0f + 0.1f * gamma[c];
    beta[c] = 0.05f * beta[c];
  }

  auto fn = [&]() -> double {
    layernorm(ROWS, COLS, EPS, x.data(), gamma.data(), beta.data(), out.data());
    return static_cast<double>(out[0]);
  };

  ka::BenchResult result = ka::time_kernel(fn, args.warmup, args.reps);
  ka::emit_result(result);
  ka::dump_output(args, "out.bin", out.data(), out.size());
  return 0;
}
