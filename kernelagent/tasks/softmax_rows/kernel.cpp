// softmax_rows: numerically stable row-wise softmax over float32 rows.
//
// TWO_PASS=1 uses an online max/sum pass (rescaling the running sum) followed by
// a normalisation pass; TWO_PASS=0 uses three straightforward passes.
#include "../harness.hpp"

#ifndef UNROLL
#define UNROLL 2
#endif
#ifndef TWO_PASS
#define TWO_PASS 0
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

static void softmax(int rows, int cols, const float* KA_RESTRICT x, float* KA_RESTRICT out) {
  for (int r = 0; r < rows; ++r) {
    const float* KA_RESTRICT row = x + static_cast<size_t>(r) * cols;
    float* KA_RESTRICT dst = out + static_cast<size_t>(r) * cols;
#if TWO_PASS == 1
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    KA_UNROLL
    for (int c = 0; c < cols; ++c) {
      const float v = row[c];
      if (v > running_max) {
        running_sum = running_sum * std::exp(running_max - v);
        running_max = v;
      }
      running_sum += std::exp(v - running_max);
    }
    const float inv = 1.0f / running_sum;
    KA_UNROLL
    KA_VEC
    for (int c = 0; c < cols; ++c) dst[c] = std::exp(row[c] - running_max) * inv;
#else
    float maximum = -INFINITY;
    KA_UNROLL
    KA_VEC
    for (int c = 0; c < cols; ++c) maximum = std::fmax(maximum, row[c]);
    float sum = 0.0f;
    KA_UNROLL
    KA_VEC
    for (int c = 0; c < cols; ++c) sum += std::exp(row[c] - maximum);
    const float inv = 1.0f / sum;
    KA_UNROLL
    KA_VEC
    for (int c = 0; c < cols; ++c) dst[c] = std::exp(row[c] - maximum) * inv;
#endif
  }
}

int main(int argc, char** argv) {
  ka::Args args = ka::parse_args(argc, argv);
  const int ROWS = static_cast<int>(args.dim("ROWS", 512));
  const int COLS = static_cast<int>(args.dim("COLS", 1024));

  std::vector<float> x(static_cast<size_t>(ROWS) * COLS);
  std::vector<float> out(x.size(), 0.0f);
  ka::load_or_random(x.data(), x.size(), args.input_dir, "x.bin", 1111);
  ka::maybe_dump_input(args, "x.bin", x.data(), x.size() * sizeof(float));

  auto fn = [&]() -> double {
    softmax(ROWS, COLS, x.data(), out.data());
    return static_cast<double>(out[0]);
  };

  ka::BenchResult result = ka::time_kernel(fn, args.warmup, args.reps);
  ka::emit_result(result);
  ka::dump_output(args, "out.bin", out.data(), out.size());
  return 0;
}
