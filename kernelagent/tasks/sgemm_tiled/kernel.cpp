// sgemm_tiled: C = A * B with row-major float32 matrices.
//
// Every tuning decision is a -D macro so the agent can enumerate it. The
// default is a modest blocked kernel; the search space contains materially
// faster tile/order combinations.
#include "../harness.hpp"

#ifndef TILE_M
#define TILE_M 16
#endif
#ifndef TILE_N
#define TILE_N 16
#endif
#ifndef TILE_K
#define TILE_K 16
#endif
#ifndef LOOP_ORDER
#define LOOP_ORDER 2
#endif
#ifndef UNROLL
#define UNROLL 2
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

static void sgemm(int M, int N, int K, const float* KA_RESTRICT A,
                  const float* KA_RESTRICT B, float* KA_RESTRICT C) {
#if LOOP_ORDER == 0
  // i-j-k: dot-product form, the weakest order for row-major operands.
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      KA_UNROLL
      for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
      C[i * N + j] = acc;
    }
  }
#elif LOOP_ORDER == 1
  // i-k-j: stream B rows, the inner j loop is vectorisable.
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) C[i * N + j] = 0.0f;
    for (int k = 0; k < K; ++k) {
      const float a = A[i * K + k];
      KA_UNROLL
      KA_VEC
      for (int j = 0; j < N; ++j) C[i * N + j] += a * B[k * N + j];
    }
  }
#else
  // Blocked i-j-k over tiles of M, N and K.
  for (int ii = 0; ii < M; ii += TILE_M) {
    const int i_end = (ii + TILE_M < M) ? ii + TILE_M : M;
    for (int jj = 0; jj < N; jj += TILE_N) {
      const int j_end = (jj + TILE_N < N) ? jj + TILE_N : N;
      for (int i = ii; i < i_end; ++i) {
        for (int j = jj; j < j_end; ++j) C[i * N + j] = 0.0f;
      }
      for (int kk = 0; kk < K; kk += TILE_K) {
        const int k_end = (kk + TILE_K < K) ? kk + TILE_K : K;
        for (int i = ii; i < i_end; ++i) {
          for (int k = kk; k < k_end; ++k) {
            const float a = A[i * K + k];
            KA_UNROLL
            KA_VEC
            for (int j = jj; j < j_end; ++j) C[i * N + j] += a * B[k * N + j];
          }
        }
      }
    }
  }
#endif
}

int main(int argc, char** argv) {
  ka::Args args = ka::parse_args(argc, argv);
  const int M = static_cast<int>(args.dim("M", 256));
  const int N = static_cast<int>(args.dim("N", 256));
  const int K = static_cast<int>(args.dim("K", 256));

  std::vector<float> A(static_cast<size_t>(M) * K);
  std::vector<float> B(static_cast<size_t>(K) * N);
  std::vector<float> C(static_cast<size_t>(M) * N, 0.0f);

  ka::load_or_random(A.data(), A.size(), args.input_dir, "a.bin", 1234);
  ka::load_or_random(B.data(), B.size(), args.input_dir, "b.bin", 5678);
  ka::maybe_dump_input(args, "a.bin", A.data(), A.size() * sizeof(float));
  ka::maybe_dump_input(args, "b.bin", B.data(), B.size() * sizeof(float));

  auto fn = [&]() -> double {
    std::fill(C.begin(), C.end(), 0.0f);
    sgemm(M, N, K, A.data(), B.data(), C.data());
    return static_cast<double>(C[0]);
  };

  ka::BenchResult result = ka::time_kernel(fn, args.warmup, args.reps);
  ka::emit_result(result);
  ka::dump_output(args, "out.bin", C.data(), C.size());
  return 0;
}
