// bias_relu_fused: out = max(x + bias, 0), one bias value per row.
#include "../harness.hpp"

#ifndef UNROLL
#define UNROLL 2
#endif
#ifndef BLOCK
#define BLOCK 4096
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

static void bias_relu(int rows, int cols, const float* KA_RESTRICT x,
                      const float* KA_RESTRICT bias, float* KA_RESTRICT out) {
  for (int r = 0; r < rows; ++r) {
    const float b = bias[r];
    const float* KA_RESTRICT src = x + static_cast<size_t>(r) * cols;
    float* KA_RESTRICT dst = out + static_cast<size_t>(r) * cols;
    for (int base = 0; base < cols; base += BLOCK) {
      const int end = (base + BLOCK < cols) ? base + BLOCK : cols;
      KA_UNROLL
      KA_VEC
      for (int c = base; c < end; ++c) dst[c] = std::fmax(src[c] + b, 0.0f);
    }
  }
}

int main(int argc, char** argv) {
  ka::Args args = ka::parse_args(argc, argv);
  const int ROWS = static_cast<int>(args.dim("ROWS", 1024));
  const int COLS = static_cast<int>(args.dim("COLS", 1024));

  std::vector<float> x(static_cast<size_t>(ROWS) * COLS);
  std::vector<float> bias(static_cast<size_t>(ROWS));
  std::vector<float> out(x.size(), 0.0f);
  ka::load_or_random(x.data(), x.size(), args.input_dir, "x.bin", 6060);
  ka::load_or_random(bias.data(), bias.size(), args.input_dir, "bias.bin", 5050);
  ka::maybe_dump_input(args, "x.bin", x.data(), x.size() * sizeof(float));
  ka::maybe_dump_input(args, "bias.bin", bias.data(), bias.size() * sizeof(float));

  auto fn = [&]() -> double {
    bias_relu(ROWS, COLS, x.data(), bias.data(), out.data());
    return static_cast<double>(out[0]);
  };

  ka::BenchResult result = ka::time_kernel(fn, args.warmup, args.reps);
  ka::emit_result(result);
  ka::dump_output(args, "out.bin", out.data(), out.size());
  return 0;
}
