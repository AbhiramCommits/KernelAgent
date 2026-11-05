// transpose_blocked: out[N, M] = in[M, N]^T with optional padded tiles.
#include "../harness.hpp"

#ifndef BLOCK
#define BLOCK 16
#endif
#ifndef USE_RESTRICT
#define USE_RESTRICT 1
#endif
#ifndef PAD
#define PAD 0
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

static void transpose(int m, int n, const float* KA_RESTRICT in, float* KA_RESTRICT out) {
  for (int i0 = 0; i0 < m; i0 += BLOCK) {
    const int i1 = (i0 + BLOCK < m) ? i0 + BLOCK : m;
    for (int j0 = 0; j0 < n; j0 += BLOCK) {
      const int j1 = (j0 + BLOCK < n) ? j0 + BLOCK : n;
#if PAD == 1
      float tile[BLOCK][BLOCK + 1];
      for (int i = i0; i < i1; ++i) {
        for (int j = j0; j < j1; ++j) tile[i - i0][j - j0] = in[static_cast<size_t>(i) * n + j];
      }
      for (int j = j0; j < j1; ++j) {
        for (int i = i0; i < i1; ++i) {
          out[static_cast<size_t>(j) * m + i] = tile[i - i0][j - j0];
        }
      }
#else
      for (int j = j0; j < j1; ++j) {
        float* KA_RESTRICT dst = out + static_cast<size_t>(j) * m;
        KA_VEC
        for (int i = i0; i < i1; ++i) dst[i] = in[static_cast<size_t>(i) * n + j];
      }
#endif
    }
  }
}

int main(int argc, char** argv) {
  ka::Args args = ka::parse_args(argc, argv);
  const int M = static_cast<int>(args.dim("M", 1024));
  const int N = static_cast<int>(args.dim("N", 1024));

  std::vector<float> in(static_cast<size_t>(M) * N);
  std::vector<float> out(in.size(), 0.0f);
  ka::load_or_random(in.data(), in.size(), args.input_dir, "x.bin", 5150);
  ka::maybe_dump_input(args, "x.bin", in.data(), in.size() * sizeof(float));

  auto fn = [&]() -> double {
    transpose(M, N, in.data(), out.data());
    return static_cast<double>(out[0]);
  };

  ka::BenchResult result = ka::time_kernel(fn, args.warmup, args.reps);
  ka::emit_result(result);
  ka::dump_output(args, "out.bin", out.data(), out.size());
  return 0;
}
