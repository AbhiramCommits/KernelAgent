// spmv_csr: y = A * x for a fixed per-row-nnz CSR pattern.
//
// The matrix pattern and values are generated deterministically by a closed-form
// formula (not an RNG) so the NumPy reference can rebuild the same operator from
// the dumped CSR arrays.
#include "../harness.hpp"

#ifndef ACCUMULATORS
#define ACCUMULATORS 1
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

static inline int32_t pattern_column(int row, int slot, int n) {
  return (row * 7 + slot * 13 + (row * 17 + slot * 3) % 5) % n;
}

static inline float pattern_value(int row, int slot) {
  return static_cast<float>((row * 131 + slot * 17) % 97) / 97.0f - 0.5f;
}

static void spmv(int m, int nnz, const int32_t* KA_RESTRICT rowptr,
                 const int32_t* KA_RESTRICT cols, const float* KA_RESTRICT vals,
                 const float* KA_RESTRICT x, float* KA_RESTRICT y) {
  for (int row = 0; row < m; ++row) {
    const int begin = rowptr[row];
    const int end = rowptr[row + 1];
#if ACCUMULATORS == 1
    float acc = 0.0f;
    KA_UNROLL
    KA_VEC
    for (int j = begin; j < end; ++j) acc += vals[j] * x[cols[j]];
    y[row] = acc;
#elif ACCUMULATORS == 2
    float a0 = 0.0f, a1 = 0.0f;
    int j = begin;
    for (; j + 1 < end; j += 2) {
      a0 += vals[j] * x[cols[j]];
      a1 += vals[j + 1] * x[cols[j + 1]];
    }
    if (j < end) a0 += vals[j] * x[cols[j]];
    y[row] = a0 + a1;
#else
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    int j = begin;
    for (; j + 3 < end; j += 4) {
      a0 += vals[j] * x[cols[j]];
      a1 += vals[j + 1] * x[cols[j + 1]];
      a2 += vals[j + 2] * x[cols[j + 2]];
      a3 += vals[j + 3] * x[cols[j + 3]];
    }
    for (; j < end; ++j) a0 += vals[j] * x[cols[j]];
    y[row] = (a0 + a1) + (a2 + a3);
#endif
  }
}

int main(int argc, char** argv) {
  ka::Args args = ka::parse_args(argc, argv);
  const int M = static_cast<int>(args.dim("M", 20000));
  const int N = static_cast<int>(args.dim("N", 20000));
  const int NNZ = static_cast<int>(args.dim("NNZ", 8));

  std::vector<int32_t> rowptr(static_cast<size_t>(M) + 1);
  std::vector<int32_t> cols(static_cast<size_t>(M) * NNZ);
  std::vector<float> vals(static_cast<size_t>(M) * NNZ);
  std::vector<float> x(static_cast<size_t>(N));
  std::vector<float> y(static_cast<size_t>(M), 0.0f);

  for (int i = 0; i <= M; ++i) rowptr[i] = i * NNZ;

  const bool have_csr =
      !args.input_dir.empty() && ka::file_exists(ka::join_path(args.input_dir, "cols.bin"));
  if (have_csr) {
    ka::read_bytes(ka::join_path(args.input_dir, "rowptr.bin"), rowptr.data(),
                   rowptr.size() * sizeof(int32_t));
    ka::read_bytes(ka::join_path(args.input_dir, "cols.bin"), cols.data(),
                   cols.size() * sizeof(int32_t));
    ka::read_bytes(ka::join_path(args.input_dir, "vals.bin"), vals.data(),
                   vals.size() * sizeof(float));
  } else {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < NNZ; ++j) {
        cols[static_cast<size_t>(i) * NNZ + j] = pattern_column(i, j, N);
        vals[static_cast<size_t>(i) * NNZ + j] = pattern_value(i, j);
      }
    }
  }
  ka::load_or_random(x.data(), x.size(), args.input_dir, "x.bin", 2468);
  ka::maybe_dump_input_i32(args, "rowptr.bin", rowptr.data(), rowptr.size());
  ka::maybe_dump_input_i32(args, "cols.bin", cols.data(), cols.size());
  ka::maybe_dump_input(args, "vals.bin", vals.data(), vals.size() * sizeof(float));
  ka::maybe_dump_input(args, "x.bin", x.data(), x.size() * sizeof(float));

  auto fn = [&]() -> double {
    std::fill(y.begin(), y.end(), 0.0f);
    spmv(M, NNZ, rowptr.data(), cols.data(), vals.data(), x.data(), y.data());
    return static_cast<double>(y[0]);
  };

  ka::BenchResult result = ka::time_kernel(fn, args.warmup, args.reps);
  ka::emit_result(result);
  ka::dump_output(args, "out.bin", y.data(), y.size());
  return 0;
}
