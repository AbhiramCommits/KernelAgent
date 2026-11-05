// conv1d_direct: valid 1-D convolution, single batch, no padding, stride 1.
//
// LOOP_ORDER=0 is output-stationary (dot product over channels/kernel).
// LOOP_ORDER=1 is a blocked axpy nest writing contiguous output.
// LOOP_ORDER=2 is the same nest without cache blocking over the output length.
#include "../harness.hpp"

#ifndef LOOP_ORDER
#define LOOP_ORDER 1
#endif
#ifndef BLOCK_L
#define BLOCK_L 64
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

static void conv1d(int c_in, int c_out, int length, int kernel, const float* KA_RESTRICT x,
                   const float* KA_RESTRICT w, float* KA_RESTRICT out) {
  const int l_out = length - kernel + 1;
#if LOOP_ORDER == 0
  for (int oc = 0; oc < c_out; ++oc) {
    for (int l = 0; l < l_out; ++l) {
      float acc = 0.0f;
      for (int ci = 0; ci < c_in; ++ci) {
        const float* KA_RESTRICT xp = x + static_cast<size_t>(ci) * length + l;
        const float* KA_RESTRICT wp = w + (static_cast<size_t>(oc) * c_in + ci) * kernel;
        KA_UNROLL
        KA_VEC
        for (int k = 0; k < kernel; ++k) acc += xp[k] * wp[k];
      }
      out[static_cast<size_t>(oc) * l_out + l] = acc;
    }
  }
#elif LOOP_ORDER == 1
  for (int oc = 0; oc < c_out; ++oc) {
    float* KA_RESTRICT op = out + static_cast<size_t>(oc) * l_out;
    for (int l0 = 0; l0 < l_out; l0 += BLOCK_L) {
      const int l1 = (l0 + BLOCK_L < l_out) ? l0 + BLOCK_L : l_out;
      for (int l = l0; l < l1; ++l) op[l] = 0.0f;
      for (int ci = 0; ci < c_in; ++ci) {
        const float* KA_RESTRICT xp = x + static_cast<size_t>(ci) * length;
        const float* KA_RESTRICT wp = w + (static_cast<size_t>(oc) * c_in + ci) * kernel;
        for (int k = 0; k < kernel; ++k) {
          const float wv = wp[k];
          KA_UNROLL
          KA_VEC
          for (int l = l0; l < l1; ++l) op[l] += xp[l + k] * wv;
        }
      }
    }
  }
#else
  for (int oc = 0; oc < c_out; ++oc) {
    float* KA_RESTRICT op = out + static_cast<size_t>(oc) * l_out;
    for (int l = 0; l < l_out; ++l) op[l] = 0.0f;
    for (int ci = 0; ci < c_in; ++ci) {
      const float* KA_RESTRICT xp = x + static_cast<size_t>(ci) * length;
      const float* KA_RESTRICT wp = w + (static_cast<size_t>(oc) * c_in + ci) * kernel;
      for (int k = 0; k < kernel; ++k) {
        const float wv = wp[k];
        KA_UNROLL
        KA_VEC
        for (int l = 0; l < l_out; ++l) op[l] += xp[l + k] * wv;
      }
    }
  }
#endif
}

int main(int argc, char** argv) {
  ka::Args args = ka::parse_args(argc, argv);
  const int C_IN = static_cast<int>(args.dim("C_IN", 8));
  const int C_OUT = static_cast<int>(args.dim("C_OUT", 16));
  const int L = static_cast<int>(args.dim("L", 2048));
  const int K = static_cast<int>(args.dim("K", 5));
  const int L_OUT = L - K + 1;

  std::vector<float> x(static_cast<size_t>(C_IN) * L);
  std::vector<float> w(static_cast<size_t>(C_OUT) * C_IN * K);
  std::vector<float> out(static_cast<size_t>(C_OUT) * L_OUT, 0.0f);
  ka::load_or_random(x.data(), x.size(), args.input_dir, "x.bin", 909);
  ka::load_or_random(w.data(), w.size(), args.input_dir, "w.bin", 707);
  ka::maybe_dump_input(args, "x.bin", x.data(), x.size() * sizeof(float));
  ka::maybe_dump_input(args, "w.bin", w.data(), w.size() * sizeof(float));

  auto fn = [&]() -> double {
    std::fill(out.begin(), out.end(), 0.0f);
    conv1d(C_IN, C_OUT, L, K, x.data(), w.data(), out.data());
    return static_cast<double>(out[0]);
  };

  ka::BenchResult result = ka::time_kernel(fn, args.warmup, args.reps);
  ka::emit_result(result);
  ka::dump_output(args, "out.bin", out.data(), out.size());
  return 0;
}
