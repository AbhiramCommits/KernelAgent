// reduce_sum: row sums of a dense float32 matrix.
#include "../harness.hpp"

#ifndef UNROLL
#define UNROLL 4
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

static void reduce_rows(int rows, int cols, const float* KA_RESTRICT x, float* KA_RESTRICT y) {
  for (int r = 0; r < rows; ++r) {
    const float* KA_RESTRICT row = x + static_cast<size_t>(r) * cols;
#if ACCUMULATORS == 1
    float s0 = 0.0f;
    KA_UNROLL
    KA_VEC
    for (int c = 0; c < cols; ++c) s0 += row[c];
    y[r] = s0;
#elif ACCUMULATORS == 2
    float s0 = 0.0f, s1 = 0.0f;
    int c = 0;
    for (; c + 1 < cols; c += 2) { s0 += row[c]; s1 += row[c + 1]; }
    if (c < cols) s0 += row[c];
    y[r] = s0 + s1;
#else
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int c = 0;
    for (; c + 3 < cols; c += 4) {
      s0 += row[c]; s1 += row[c + 1]; s2 += row[c + 2]; s3 += row[c + 3];
    }
    for (; c < cols; ++c) s0 += row[c];
    y[r] = (s0 + s1) + (s2 + s3);
#endif
  }
}

int main(int argc, char** argv) {
  ka::Args args = ka::parse_args(argc, argv);
  const int M = static_cast<int>(args.dim("M", 1024));
  const int N = static_cast<int>(args.dim("N", 1024));

  std::vector<float> x(static_cast<size_t>(M) * N);
  std::vector<float> y(static_cast<size_t>(M), 0.0f);
  ka::load_or_random(x.data(), x.size(), args.input_dir, "x.bin", 8080);
  ka::maybe_dump_input(args, "x.bin", x.data(), x.size() * sizeof(float));

  auto fn = [&]() -> double {
    reduce_rows(M, N, x.data(), y.data());
    return static_cast<double>(y[0]);
  };

  ka::BenchResult result = ka::time_kernel(fn, args.warmup, args.reps);
  ka::emit_result(result);
  ka::dump_output(args, "out.bin", y.data(), y.size());
  return 0;
}
