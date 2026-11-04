// common.h - helpers shared by the CPU and GPU GEMM benchmarks (P4).
//
// Everything that has to be identical across implementations lives here:
// the common CSV schema, command-line options, matrix initialisation and the
// verification metric. All matrices are square N x N, row-major, FP32.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// Column order required by the course brief so every team's CSV merges.
static const char* CSV_HEADER =
    "cluster,problem,platform,implementation,dataset,size,threads_or_ranks,"
    "run_id,total_ms,kernel_ms,transfer_ms,throughput,error,power_w";

struct Options {
    std::vector<int> sizes;
    int runs = 10;                  // recorded runs (one extra warm-up run is never recorded)
    std::string cluster = "C1";
    std::string problem = "P4";
    std::string platform;           // empty -> detected automatically
    std::string out;                // common-schema CSV
    std::string detail;             // extra per-run columns (GPU only)
    std::string impls;              // comma-separated filter, empty -> all
    bool pageable = false;          // GPU: use pageable instead of pinned host memory
    int profile_n = 0;              // GPU: run every kernel once at this size (for ncu)
    unsigned seed = 42;
    int samples = 4096;             // entries checked against the FP64 reference
};

inline std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> v;
    size_t start = 0;
    while (start < s.size()) {
        size_t end = s.find(',', start);
        if (end == std::string::npos) end = s.size();
        if (end > start) v.push_back(std::atoi(s.substr(start, end - start).c_str()));
        start = end + 1;
    }
    return v;
}

inline bool impl_selected(const Options& o, const std::string& name) {
    if (o.impls.empty()) return true;
    std::string list = "," + o.impls + ",";
    return list.find("," + name + ",") != std::string::npos;
}

inline void usage(const char* prog) {
    std::printf(
        "usage: %s [--sizes 64,128,...] [--runs R] [--cluster C1] [--platform NAME]\n"
        "          [--out file.csv] [--detail file.csv] [--impl a,b,...] [--pageable]\n"
        "          [--profile N] [--seed S] [--samples K]\n",
        prog);
}

inline Options parse_options(int argc, char** argv, const std::vector<int>& default_sizes,
                             const std::string& default_out, const std::string& default_detail) {
    Options o;
    o.sizes = default_sizes;
    o.out = default_out;
    o.detail = default_detail;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(argv[0]); std::exit(1); }
            return argv[++i];
        };
        if (a == "--sizes") o.sizes = parse_int_list(next());
        else if (a == "--runs") o.runs = std::atoi(next().c_str());
        else if (a == "--cluster") o.cluster = next();
        else if (a == "--platform") o.platform = next();
        else if (a == "--out") o.out = next();
        else if (a == "--detail") o.detail = next();
        else if (a == "--impl") o.impls = next();
        else if (a == "--pageable") o.pageable = true;
        else if (a == "--profile") o.profile_n = std::atoi(next().c_str());
        else if (a == "--seed") o.seed = (unsigned)std::atoi(next().c_str());
        else if (a == "--samples") o.samples = std::atoi(next().c_str());
        else { usage(argv[0]); std::exit(a == "--help" || a == "-h" ? 0 : 1); }
    }
    return o;
}

// Replace characters that would break a CSV field or a file name.
inline std::string sanitize(std::string s) {
    for (char& c : s)
        if (c == ' ' || c == ',' || c == '(' || c == ')' || c == '@') c = '_';
    return s;
}

inline std::string dataset_name(const Options& o) {
    return "uniform_m1_1_seed" + std::to_string(o.seed);
}

// Opens a CSV for appending and writes the header only if the file is new/empty.
inline FILE* open_csv(const std::string& path, const char* header) {
    bool has_content = false;
    if (FILE* f = std::fopen(path.c_str(), "r")) {
        has_content = std::fgetc(f) != EOF;
        std::fclose(f);
    }
    FILE* f = std::fopen(path.c_str(), "a");
    if (!f) { std::perror(path.c_str()); std::exit(1); }
    if (!has_content) std::fprintf(f, "%s\n", header);
    return f;
}

struct Row {
    std::string implementation;
    int size = 0;
    int threads = 1;
    int run_id = 0;
    double total_ms = 0, kernel_ms = 0, transfer_ms = 0;
    double throughput = 0;          // GFLOP/s computed from kernel_ms
    double error = 0;
    double power_w = NAN;           // empty in the CSV when not measured
};

inline void write_row(FILE* f, const Options& o, const std::string& platform, const Row& r) {
    std::fprintf(f, "%s,%s,%s,%s,%s,%d,%d,%d,%.6f,%.6f,%.6f,%.4f,%.3e,",
                 o.cluster.c_str(), o.problem.c_str(), platform.c_str(), r.implementation.c_str(),
                 dataset_name(o).c_str(), r.size, r.threads, r.run_id, r.total_ms, r.kernel_ms,
                 r.transfer_ms, r.throughput, r.error);
    if (std::isnan(r.power_w)) std::fprintf(f, "\n");
    else std::fprintf(f, "%.2f\n", r.power_w);
    std::fflush(f);
}

inline double gemm_gflops(int n, double ms) {
    return 2.0 * (double)n * n * n / (ms * 1e-3) / 1e9;
}

// Deterministic uniform values in [-1, 1] so every implementation sees the same data.
inline void init_matrix(float* p, size_t count, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < count; ++i) p[i] = dist(rng);
}

// Verification metric used by every implementation: pick `samples` random
// entries of C, recompute them in double precision and return
//   max |C_ij - ref_ij| / max |ref_ij|.
inline double sampled_error(const float* A, const float* B, const float* C, int n, int samples,
                            unsigned seed = 7) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, n - 1);
    double max_err = 0.0, max_ref = 0.0;
    for (int s = 0; s < samples; ++s) {
        int i = pick(rng), j = pick(rng);
        double ref = 0.0;
        for (int k = 0; k < n; ++k) ref += (double)A[(size_t)i * n + k] * (double)B[(size_t)k * n + j];
        max_err = std::fmax(max_err, std::fabs((double)C[(size_t)i * n + j] - ref));
        max_ref = std::fmax(max_ref, std::fabs(ref));
    }
    return max_ref > 0 ? max_err / max_ref : max_err;
}

// Full-matrix comparison against another result (e.g. cuBLAS), same normalisation.
inline double full_relative_diff(const float* C, const float* R, size_t count) {
    double max_err = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < count; ++i) {
        max_err = std::fmax(max_err, std::fabs((double)C[i] - (double)R[i]));
        max_ref = std::fmax(max_ref, std::fabs((double)R[i]));
    }
    return max_ref > 0 ? max_err / max_ref : max_err;
}

static const double ERROR_TOLERANCE = 1e-4;

inline double mean_of(const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x;
    return v.empty() ? 0 : s / v.size();
}

inline double std_of(const std::vector<double>& v) {
    if (v.size() < 2) return 0;
    double m = mean_of(v), s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / (v.size() - 1));
}
