// gemm_cpu.cpp - optimised, verified sequential GEMM baseline (C = A * B, FP32).
//
// Every GPU speedup in P4 is measured against `cpu_seq_blocked`.
//   cpu_seq_naive   : textbook i-j-k loop (only for small sizes, shows why blocking matters)
//   cpu_seq_blocked : cache-blocked i-k-j loop; the innermost loop is unit-stride over
//                     B and C so -O3 -march=native vectorises it with AVX/AVX2 FMA.
//
// Build: g++ -O3 -march=native -std=c++14 src/gemm_cpu.cpp -o bin/gemm_cpu
#include <algorithm>
#include <chrono>

#include "common.h"

#ifndef BLOCK_I
#define BLOCK_I 32
#endif
#ifndef BLOCK_K
#define BLOCK_K 128
#endif
#ifndef BLOCK_J
#define BLOCK_J 256
#endif

static const double MIN_RUN_MS = 20.0;

static void gemm_naive(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < n; ++k) acc += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = acc;
        }
}

// The BLOCK_K x BLOCK_J panel of B (128 KB) stays in L2 while BLOCK_I rows of A
// stream past it; the BLOCK_I x BLOCK_J tile of C (32 KB) stays in L1/L2.
static void gemm_blocked(const float* __restrict__ A, const float* __restrict__ B,
                         float* __restrict__ C, int n) {
    std::fill(C, C + (size_t)n * n, 0.0f);
    for (int ii = 0; ii < n; ii += BLOCK_I) {
        const int i_end = std::min(ii + BLOCK_I, n);
        for (int kk = 0; kk < n; kk += BLOCK_K) {
            const int k_end = std::min(kk + BLOCK_K, n);
            for (int jj = 0; jj < n; jj += BLOCK_J) {
                const int j_end = std::min(jj + BLOCK_J, n);
                for (int i = ii; i < i_end; ++i) {
                    float* __restrict__ c = C + (size_t)i * n;
                    for (int k = kk; k < k_end; ++k) {
                        const float a = A[(size_t)i * n + k];
                        const float* __restrict__ b = B + (size_t)k * n;
                        for (int j = jj; j < j_end; ++j) c[j] += a * b[j];
                    }
                }
            }
        }
    }
}

struct CpuImpl {
    const char* name;
    void (*fn)(const float*, const float*, float*, int);
    int max_n;  // skip sizes above this (naive is too slow to be useful there)
};

static const CpuImpl IMPLS[] = {
    {"cpu_seq_naive", gemm_naive, 512},
    {"cpu_seq_blocked", gemm_blocked, 1 << 30},
};

int main(int argc, char** argv) {
    Options o = parse_options(argc, argv, {64, 128, 256, 512, 1024, 2048},
                              "results/raw/p4_gemm_cpu.csv", "");
    const std::string platform = o.platform.empty() ? "CPU" : sanitize(o.platform);
    FILE* csv = open_csv(o.out, CSV_HEADER);

    std::printf("CPU sequential GEMM  blocks I=%d K=%d J=%d  runs=%d (+1 warm-up)\n",
                BLOCK_I, BLOCK_K, BLOCK_J, o.runs);
    std::printf("%-16s %6s %12s %10s %10s %10s\n", "impl", "N", "mean_ms", "std_ms", "GFLOP/s", "error");

    for (int n : o.sizes) {
        const size_t count = (size_t)n * n;
        std::vector<float> A(count), B(count), C(count);
        init_matrix(A.data(), count, o.seed);
        init_matrix(B.data(), count, o.seed + 1);

        for (const CpuImpl& impl : IMPLS) {
            if (!impl_selected(o, impl.name) || n > impl.max_n) continue;

            // Warm-up (not recorded). Small matrices finish faster than the clock
            // resolution, so each recorded run repeats the call `reps` times
            // (at least MIN_RUN_MS in total) and reports the time per call.
            int reps = 1;
            for (;;) {
                auto t0 = std::chrono::steady_clock::now();
                for (int k = 0; k < reps; ++k) impl.fn(A.data(), B.data(), C.data(), n);
                auto t1 = std::chrono::steady_clock::now();
                if (std::chrono::duration<double, std::milli>(t1 - t0).count() >= MIN_RUN_MS) break;
                reps *= 2;
            }
            const double err = sampled_error(A.data(), B.data(), C.data(), n, o.samples);
            if (err > ERROR_TOLERANCE)
                std::fprintf(stderr, "VERIFY FAIL %s N=%d error=%.3e\n", impl.name, n, err);

            std::vector<double> times;
            for (int r = 1; r <= o.runs; ++r) {
                auto t0 = std::chrono::steady_clock::now();
                for (int k = 0; k < reps; ++k) impl.fn(A.data(), B.data(), C.data(), n);
                auto t1 = std::chrono::steady_clock::now();
                const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
                times.push_back(ms);

                Row row;
                row.implementation = impl.name;
                row.size = n;
                row.threads = 1;
                row.run_id = r;
                row.total_ms = ms;
                row.kernel_ms = ms;
                row.transfer_ms = 0.0;
                row.throughput = gemm_gflops(n, ms);
                row.error = err;
                write_row(csv, o, platform, row);
            }
            std::printf("%-16s %6d %12.3f %10.3f %10.2f %10.2e\n", impl.name, n, mean_of(times),
                        std_of(times), gemm_gflops(n, mean_of(times)), err);
        }
    }
    std::fclose(csv);
    return 0;
}
