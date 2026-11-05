// Shared benchmark harness for every KernelAgent C++ task.
//
// A task kernel.cpp only provides input generation, one kernel call and an
// output dump. This header owns argument parsing, deterministic input seeding,
// self-calibrated timing, MAD/min statistics, checksums and JSON output so all
// tasks are measured identically.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace ka {

using Clock = std::chrono::steady_clock;

inline double now_s() {
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

struct Args {
  std::map<std::string, long> dims;
  int reps = 5;
  int warmup = 2;
  std::string dump_dir;   // write inputs and outputs here
  std::string input_dir;  // load inputs from here
  std::string dump;       // write only the output array here

  long dim(const std::string& key, long fallback = 0) const {
    auto it = dims.find(key);
    return it == dims.end() ? fallback : it->second;
  }
};

inline bool starts_with(const std::string& text, const std::string& prefix) {
  return text.rfind(prefix, 0) == 0;
}

inline std::string trim(const std::string& s) {
  size_t b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  size_t e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

inline Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", name);
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--shape") {
      std::string spec = value("--shape");
      size_t pos = 0;
      while (pos < spec.size()) {
        size_t comma = spec.find(',', pos);
        std::string item = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        size_t eq = item.find('=');
        if (eq != std::string::npos) {
          args.dims[trim(item.substr(0, eq))] = std::stol(item.substr(eq + 1));
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
      }
    } else if (arg == "--reps") {
      args.reps = std::max(1, std::atoi(value("--reps").c_str()));
    } else if (arg == "--warmup") {
      args.warmup = std::max(0, std::atoi(value("--warmup").c_str()));
    } else if (arg == "--dump-dir") {
      args.dump_dir = value("--dump-dir");
    } else if (arg == "--input-dir") {
      args.input_dir = value("--input-dir");
    } else if (arg == "--dump") {
      args.dump = value("--dump");
    }
  }
  return args;
}

// xorshift64*: deterministic, identical for every task and every variant.
struct Rng {
  uint64_t state;
  explicit Rng(uint64_t seed) : state(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
  inline uint64_t next() {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * 0x2545F4914F6CDD1DULL;
  }
  // Uniform in [-1, 1).
  inline float uniform() {
    return static_cast<float>((next() >> 40) * (1.0 / 8388608.0) - 1.0);
  }
};

inline void fill_random(float* data, size_t n, uint64_t seed) {
  Rng rng(seed);
  for (size_t i = 0; i < n; ++i) data[i] = rng.uniform();
}

inline std::string join_path(const std::string& dir, const std::string& name) {
  if (dir.empty()) return name;
  if (dir.back() == '/') return dir + name;
  return dir + "/" + name;
}

inline void write_bytes(const std::string& path, const void* data, size_t bytes) {
  std::FILE* fp = std::fopen(path.c_str(), "wb");
  if (!fp) {
    std::fprintf(stderr, "cannot write %s\n", path.c_str());
    std::exit(3);
  }
  if (bytes && std::fwrite(data, 1, bytes, fp) != bytes) {
    std::fprintf(stderr, "short write %s\n", path.c_str());
    std::exit(3);
  }
  std::fclose(fp);
}

inline bool file_exists(const std::string& path) {
  std::FILE* fp = std::fopen(path.c_str(), "rb");
  if (!fp) return false;
  std::fclose(fp);
  return true;
}

inline void read_bytes(const std::string& path, void* data, size_t bytes) {
  std::FILE* fp = std::fopen(path.c_str(), "rb");
  if (!fp) {
    std::fprintf(stderr, "cannot read %s\n", path.c_str());
    std::exit(3);
  }
  if (bytes && std::fread(data, 1, bytes, fp) != bytes) {
    std::fprintf(stderr, "short read %s\n", path.c_str());
    std::exit(3);
  }
  std::fclose(fp);
}

// Read an input when an input directory is supplied, otherwise generate it.
// This keeps every variant on exactly the same data without re-dumping large
// arrays for every trial.
inline void load_or_random(float* data, size_t n, const std::string& input_dir,
                           const std::string& name, uint64_t seed) {
  if (!input_dir.empty()) {
    std::string path = join_path(input_dir, name);
    if (file_exists(path)) {
      read_bytes(path, data, n * sizeof(float));
      return;
    }
  }
  fill_random(data, n, seed);
}

inline void load_or_random_i32(int32_t* data, size_t n, const std::string& input_dir,
                               const std::string& name) {
  if (!input_dir.empty()) {
    std::string path = join_path(input_dir, name);
    if (file_exists(path)) {
      read_bytes(path, data, n * sizeof(int32_t));
      return;
    }
  }
  for (size_t i = 0; i < n; ++i) data[i] = 0;
}

inline void maybe_dump_input(const Args& args, const std::string& name, const void* data,
                             size_t bytes) {
  if (args.dump_dir.empty()) return;
  write_bytes(join_path(args.dump_dir, name), data, bytes);
}

inline void maybe_dump_input_i32(const Args& args, const std::string& name, const int32_t* data,
                                 size_t count) {
  if (args.dump_dir.empty()) return;
  write_bytes(join_path(args.dump_dir, name), data, count * sizeof(int32_t));
}

inline double median_of(std::vector<double> values) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  size_t n = values.size();
  if (n % 2 == 1) return values[n / 2];
  return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

struct BenchResult {
  double ms_median = 0.0;
  double ms_mad = 0.0;
  double ms_min = 0.0;
  int reps = 0;
  int inner = 1;
  double checksum = 0.0;
};

// Calibrate the inner iteration count so one timed sample lasts roughly
// `target_ms`; this gives sub-microsecond kernels a resolvable measurement
// without changing the reported per-call time.
template <typename F>
inline BenchResult time_kernel(F fn, int warmup, int reps, double target_ms = 6.0) {
  double t0 = now_s();
  volatile double sink = fn();
  double elapsed_ms = (now_s() - t0) * 1000.0;
  (void)sink;
  if (!(elapsed_ms > 0.0) || !std::isfinite(elapsed_ms)) elapsed_ms = 0.001;
  int inner = static_cast<int>(std::ceil(target_ms / elapsed_ms));
  inner = std::max(1, std::min(inner, 10000));

  for (int w = 0; w < warmup; ++w) {
    for (int j = 0; j < inner; ++j) sink = fn();
  }

  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(reps));
  double checksum = 0.0;
  for (int r = 0; r < reps; ++r) {
    double s0 = now_s();
    double acc = 0.0;
    for (int j = 0; j < inner; ++j) acc += fn();
    double s1 = now_s();
    samples.push_back((s1 - s0) * 1000.0 / static_cast<double>(inner));
    checksum += acc;
  }

  BenchResult out;
  out.ms_median = median_of(samples);
  std::vector<double> deviations;
  deviations.reserve(samples.size());
  for (double s : samples) deviations.push_back(std::fabs(s - out.ms_median));
  out.ms_mad = median_of(deviations);
  out.ms_min = *std::min_element(samples.begin(), samples.end());
  out.reps = reps;
  out.inner = inner;
  out.checksum = checksum;
  return out;
}

inline void emit_result(const BenchResult& result) {
  std::printf(
      "{\"ms_median\": %.9g, \"ms_mad\": %.9g, \"ms_min\": %.9g, "
      "\"reps\": %d, \"inner\": %d, \"checksum\": %.9g}\n",
      result.ms_median, result.ms_mad, result.ms_min, result.reps, result.inner,
      result.checksum);
  std::fflush(stdout);
}

// Write the kernel's output (float32, row-major) for the correctness checker.
inline void dump_output(const Args& args, const std::string& name, const float* data,
                        size_t count) {
  if (!args.dump.empty()) write_bytes(args.dump, data, count * sizeof(float));
  if (!args.dump_dir.empty()) write_bytes(join_path(args.dump_dir, name), data,
                                          count * sizeof(float));
}

}  // namespace ka
