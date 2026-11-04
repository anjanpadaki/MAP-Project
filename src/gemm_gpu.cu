// gemm_gpu.cu - P4: CUDA GEMM kernels (C = A * B, FP32, row-major, square N x N).
//
//   gpu_naive_b16        one thread per C element, 16x16 blocks, operands read from global memory
//   gpu_tiled_t{8,16,32} shared-memory tiling with __syncthreads(); tile width = block width
//   gpu_coarse2x2_t16    16x16 threads, each thread computes a 2x2 block of C (32x32 C tile)
//   gpu_coarse4x4_t16    16x16 threads, each thread computes a 4x4 block of C (64x64 C tile)
//   cublas_sgemm         cublasSgemm, the vendor-library reference
//
// Every run times host-to-device copy, kernel and device-to-host copy separately
// with cudaEvent. Results go to the common CSV (results/raw/p4_gemm_gpu.csv) and a
// detail CSV with the separate H2D/D2H times and launch/occupancy data.
//
// Build: nvcc -O3 -arch=native -std=c++14 src/gemm_gpu.cu -lcublas -o bin/gemm_gpu
// Profile: ./bin/gemm_gpu --profile 2048   (runs each kernel once, used by scripts/profile_ncu.sh)
#include <cublas_v2.h>
#include <cuda_runtime.h>

#ifdef USE_NVML
#include <nvml.h>
#endif

#include "common.h"

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err_ = (call);                                                         \
        if (err_ != cudaSuccess) {                                                         \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                         __FILE__, __LINE__);                                              \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)

#define CUBLAS_CHECK(call)                                                                 \
    do {                                                                                   \
        cublasStatus_t st_ = (call);                                                       \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                                \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__,         \
                         __LINE__);                                                        \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Each thread reads a full row of A and column of B from global memory:
// 2N loads per 2N FLOPs, so arithmetic intensity is only 0.25 FLOP/byte
// (before caches help). threadIdx.x walks along a row of C, so the B and C
// accesses of a warp are coalesced and the A access is a broadcast.
__global__ void gemm_naive(const float* __restrict__ A, const float* __restrict__ B,
                           float* __restrict__ C, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float acc = 0.0f;
        for (int k = 0; k < n; ++k) acc += A[(size_t)row * n + k] * B[(size_t)k * n + col];
        C[(size_t)row * n + col] = acc;
    }
}

// Kirk & Hwu tiled kernel. The block cooperatively stages a TILE x TILE tile of
// A and of B in shared memory, then every element staged is reused TILE times,
// cutting global traffic by a factor of TILE (intensity ~ TILE/4 FLOP/byte).
// Out-of-range elements are loaded as zero so any N works.
template <int TILE>
__global__ void gemm_tiled(const float* __restrict__ A, const float* __restrict__ B,
                           float* __restrict__ C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;
    float acc = 0.0f;

    for (int t = 0; t < (n + TILE - 1) / TILE; ++t) {
        const int a_col = t * TILE + tx;
        const int b_row = t * TILE + ty;
        As[ty][tx] = (row < n && a_col < n) ? A[(size_t)row * n + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < n && col < n) ? B[(size_t)b_row * n + col] : 0.0f;
        __syncthreads();  // tile fully loaded before anyone reads it

#pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();  // everyone done reading before the tile is overwritten
    }
    if (row < n && col < n) C[(size_t)row * n + col] = acc;
}

// Thread coarsening: a TILE x TILE thread block computes a (TILE*CF) x (TILE*CF)
// tile of C, each thread owning CF x CF outputs spaced TILE apart (keeps global
// loads coalesced and shared-memory reads conflict-free). Each value read from
// shared memory is reused CF times from registers, so shared-memory traffic per
// FLOP drops by CF and global traffic per FLOP drops by TILE*CF.
template <int TILE, int CF>
__global__ void gemm_coarse(const float* __restrict__ A, const float* __restrict__ B,
                            float* __restrict__ C, int n) {
    constexpr int BM = TILE * CF;
    __shared__ float As[BM][TILE];
    __shared__ float Bs[TILE][BM];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row0 = blockIdx.y * BM + ty;
    const int col0 = blockIdx.x * BM + tx;

    float acc[CF][CF];
#pragma unroll
    for (int i = 0; i < CF; ++i)
#pragma unroll
        for (int j = 0; j < CF; ++j) acc[i][j] = 0.0f;

    for (int t = 0; t < (n + TILE - 1) / TILE; ++t) {
        const int a_col = t * TILE + tx;
        const int b_row = t * TILE + ty;
#pragma unroll
        for (int i = 0; i < CF; ++i) {
            const int r = row0 + i * TILE;
            As[ty + i * TILE][tx] = (r < n && a_col < n) ? A[(size_t)r * n + a_col] : 0.0f;
        }
#pragma unroll
        for (int j = 0; j < CF; ++j) {
            const int c = col0 + j * TILE;
            Bs[ty][tx + j * TILE] = (b_row < n && c < n) ? B[(size_t)b_row * n + c] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            float a[CF], b[CF];
#pragma unroll
            for (int i = 0; i < CF; ++i) a[i] = As[ty + i * TILE][k];
#pragma unroll
            for (int j = 0; j < CF; ++j) b[j] = Bs[k][tx + j * TILE];
#pragma unroll
            for (int i = 0; i < CF; ++i)
#pragma unroll
                for (int j = 0; j < CF; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < CF; ++i)
#pragma unroll
        for (int j = 0; j < CF; ++j) {
            const int r = row0 + i * TILE, c = col0 + j * TILE;
            if (r < n && c < n) C[(size_t)r * n + c] = acc[i][j];
        }
}

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------

static cublasHandle_t g_cublas;

static void launch_naive(const float* A, const float* B, float* C, int n) {
    dim3 block(16, 16), grid((n + 15) / 16, (n + 15) / 16);
    gemm_naive<<<grid, block>>>(A, B, C, n);
}

template <int TILE>
static void launch_tiled(const float* A, const float* B, float* C, int n) {
    dim3 block(TILE, TILE), grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    gemm_tiled<TILE><<<grid, block>>>(A, B, C, n);
}

template <int TILE, int CF>
static void launch_coarse(const float* A, const float* B, float* C, int n) {
    constexpr int BM = TILE * CF;
    dim3 block(TILE, TILE), grid((n + BM - 1) / BM, (n + BM - 1) / BM);
    gemm_coarse<TILE, CF><<<grid, block>>>(A, B, C, n);
}

// cuBLAS is column-major. A row-major matrix is its transpose in column-major,
// so computing C^T = B^T * A^T in column-major yields row-major C = A * B.
static void launch_cublas(const float* A, const float* B, float* C, int n) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, B, n, A, n,
                             &beta, C, n));
}

struct GpuImpl {
    const char* name;
    void (*launch)(const float*, const float*, float*, int);
    const void* kernel;  // for occupancy queries; nullptr for cuBLAS
    int block_threads;
    int outputs_per_thread;
};

static const GpuImpl IMPLS[] = {
    {"gpu_naive_b16", launch_naive, (const void*)gemm_naive, 256, 1},
    {"gpu_tiled_t8", launch_tiled<8>, (const void*)gemm_tiled<8>, 64, 1},
    {"gpu_tiled_t16", launch_tiled<16>, (const void*)gemm_tiled<16>, 256, 1},
    {"gpu_tiled_t32", launch_tiled<32>, (const void*)gemm_tiled<32>, 1024, 1},
    {"gpu_coarse2x2_t16", launch_coarse<16, 2>, (const void*)gemm_coarse<16, 2>, 256, 4},
    {"gpu_coarse4x4_t16", launch_coarse<16, 4>, (const void*)gemm_coarse<16, 4>, 256, 16},
    {"cublas_sgemm", launch_cublas, nullptr, 0, 0},
};

// ---------------------------------------------------------------------------
// Power (optional, build with `make NVML=1`)
// ---------------------------------------------------------------------------
#ifdef USE_NVML
static nvmlDevice_t g_nvml_dev;
static bool g_nvml_ok = false;
static void power_init() {
    g_nvml_ok = nvmlInit() == NVML_SUCCESS && nvmlDeviceGetHandleByIndex(0, &g_nvml_dev) == NVML_SUCCESS;
}
static double power_w() {
    unsigned int mw = 0;
    if (g_nvml_ok && nvmlDeviceGetPowerUsage(g_nvml_dev, &mw) == NVML_SUCCESS) return mw / 1000.0;
    return NAN;
}
#else
static void power_init() {}
static double power_w() { return NAN; }
#endif

// ---------------------------------------------------------------------------

static const char* DETAIL_HEADER =
    "implementation,size,run_id,host_memory,h2d_ms,kernel_ms,d2h_ms,total_ms,gflops_kernel,"
    "gflops_total,block_threads,outputs_per_thread,regs_per_thread,static_smem_bytes,"
    "blocks_per_sm,theoretical_occupancy";

struct Timing { float h2d, kernel, d2h; };

struct Buffers {
    float *hA, *hB, *hC, *dA, *dB, *dC;
    size_t bytes;
};

static Timing run_once(const GpuImpl& impl, const Buffers& b, int n, cudaEvent_t ev[4]) {
    CUDA_CHECK(cudaMemset(b.dC, 0, b.bytes));  // a stale C can never pass verification
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev[0]));
    CUDA_CHECK(cudaMemcpy(b.dA, b.hA, b.bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.dB, b.hB, b.bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev[1]));
    impl.launch(b.dA, b.dB, b.dC, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev[2]));
    CUDA_CHECK(cudaMemcpy(b.hC, b.dC, b.bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev[3]));
    CUDA_CHECK(cudaEventSynchronize(ev[3]));

    Timing t;
    CUDA_CHECK(cudaEventElapsedTime(&t.h2d, ev[0], ev[1]));
    CUDA_CHECK(cudaEventElapsedTime(&t.kernel, ev[1], ev[2]));
    CUDA_CHECK(cudaEventElapsedTime(&t.d2h, ev[2], ev[3]));
    return t;
}

static Buffers alloc_buffers(int n, bool pageable) {
    Buffers b;
    b.bytes = (size_t)n * n * sizeof(float);
    if (pageable) {
        b.hA = (float*)std::malloc(b.bytes);
        b.hB = (float*)std::malloc(b.bytes);
        b.hC = (float*)std::malloc(b.bytes);
        if (!b.hA || !b.hB || !b.hC) { std::fprintf(stderr, "host malloc failed\n"); std::exit(1); }
    } else {
        CUDA_CHECK(cudaMallocHost(&b.hA, b.bytes));
        CUDA_CHECK(cudaMallocHost(&b.hB, b.bytes));
        CUDA_CHECK(cudaMallocHost(&b.hC, b.bytes));
    }
    CUDA_CHECK(cudaMalloc(&b.dA, b.bytes));
    CUDA_CHECK(cudaMalloc(&b.dB, b.bytes));
    CUDA_CHECK(cudaMalloc(&b.dC, b.bytes));
    return b;
}

static void free_buffers(Buffers& b, bool pageable) {
    if (pageable) { std::free(b.hA); std::free(b.hB); std::free(b.hC); }
    else { cudaFreeHost(b.hA); cudaFreeHost(b.hB); cudaFreeHost(b.hC); }
    cudaFree(b.dA); cudaFree(b.dB); cudaFree(b.dC);
}

// Runs every selected kernel exactly once so Nsight Compute profiles one launch each.
static int profile_mode(const Options& o, int n) {
    Buffers b = alloc_buffers(n, false);
    init_matrix(b.hA, (size_t)n * n, o.seed);
    init_matrix(b.hB, (size_t)n * n, o.seed + 1);
    CUDA_CHECK(cudaMemcpy(b.dA, b.hA, b.bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.dB, b.hB, b.bytes, cudaMemcpyHostToDevice));
    for (const GpuImpl& impl : IMPLS) {
        if (!impl_selected(o, impl.name)) continue;
        std::printf("profiling %s N=%d\n", impl.name, n);
        impl.launch(b.dA, b.dB, b.dC, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    free_buffers(b, false);
    return 0;
}

int main(int argc, char** argv) {
    Options o = parse_options(argc, argv, {64, 128, 256, 512, 1024, 2048, 4096, 8192},
                              "results/raw/p4_gemm_gpu.csv", "results/raw/p4_gemm_gpu_detail.csv");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const std::string platform = o.platform.empty() ? "GPU-" + sanitize(prop.name) : sanitize(o.platform);
    std::printf("Device: %s  (cc %d.%d, %d SMs, %.1f GB)\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount, prop.totalGlobalMem / 1e9);

    CUBLAS_CHECK(cublasCreate(&g_cublas));
    // Plain FP32 maths, no TF32 tensor-core shortcut, so the comparison is fair.
    CUBLAS_CHECK(cublasSetMathMode(g_cublas, CUBLAS_DEFAULT_MATH));
    power_init();

    if (o.profile_n > 0) return profile_mode(o, o.profile_n);

    FILE* csv = open_csv(o.out, CSV_HEADER);
    FILE* det = open_csv(o.detail, DETAIL_HEADER);
    const char* host_mem = o.pageable ? "pageable" : "pinned";

    cudaEvent_t ev[4];
    for (auto& e : ev) CUDA_CHECK(cudaEventCreate(&e));

    std::printf("host memory: %s, runs=%d (+1 warm-up)\n", host_mem, o.runs);
    std::printf("%-18s %6s %10s %10s %10s %10s %10s %9s %s\n", "impl", "N", "h2d_ms", "kernel_ms",
                "d2h_ms", "kern_std", "GFLOP/s", "occ", "verify");

    for (int n : o.sizes) {
        Buffers b = alloc_buffers(n, o.pageable);
        const size_t count = (size_t)n * n;
        init_matrix(b.hA, count, o.seed);
        init_matrix(b.hB, count, o.seed + 1);

        // cuBLAS result, used for a full-matrix cross-check of the custom kernels.
        std::vector<float> ref(count);
        run_once(IMPLS[6], b, n, ev);
        std::memcpy(ref.data(), b.hC, b.bytes);

        for (const GpuImpl& impl : IMPLS) {
            if (!impl_selected(o, impl.name)) continue;

            run_once(impl, b, n, ev);  // warm-up, not recorded
            const double err = sampled_error(b.hA, b.hB, b.hC, n, o.samples);
            const double diff = full_relative_diff(b.hC, ref.data(), count);
            const bool ok = err <= ERROR_TOLERANCE && diff <= ERROR_TOLERANCE;
            if (!ok)
                std::fprintf(stderr, "VERIFY FAIL %s N=%d sampled=%.3e vs_cublas=%.3e\n", impl.name,
                             n, err, diff);

            int regs = 0, smem = 0, blocks_per_sm = 0;
            double occupancy = NAN;
            if (impl.kernel) {
                cudaFuncAttributes attr;
                CUDA_CHECK(cudaFuncGetAttributes(&attr, impl.kernel));
                regs = attr.numRegs;
                smem = (int)attr.sharedSizeBytes;
                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, impl.kernel,
                                                                         impl.block_threads, 0));
                occupancy = (double)blocks_per_sm * impl.block_threads / prop.maxThreadsPerMultiProcessor;
            }

            std::vector<double> h2d, ker, d2h;
            for (int r = 1; r <= o.runs; ++r) {
                Timing t = run_once(impl, b, n, ev);
                h2d.push_back(t.h2d);
                ker.push_back(t.kernel);
                d2h.push_back(t.d2h);
                const double total = (double)t.h2d + t.kernel + t.d2h;

                Row row;
                row.implementation = impl.name;
                row.size = n;
                row.threads = impl.block_threads;  // threads per block (0 = chosen by cuBLAS)
                row.run_id = r;
                row.total_ms = total;
                row.kernel_ms = t.kernel;
                row.transfer_ms = (double)t.h2d + t.d2h;
                row.throughput = gemm_gflops(n, t.kernel);
                row.error = err;
                row.power_w = power_w();
                write_row(csv, o, platform, row);

                std::fprintf(det, "%s,%d,%d,%s,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%d,%d,%d,%d,%d,%.4f\n",
                             impl.name, n, r, host_mem, t.h2d, t.kernel, t.d2h, total,
                             gemm_gflops(n, t.kernel), gemm_gflops(n, total), impl.block_threads,
                             impl.outputs_per_thread, regs, smem, blocks_per_sm, occupancy);
            }
            std::fflush(det);
            std::printf("%-18s %6d %10.4f %10.4f %10.4f %10.4f %10.1f %9.2f %s\n", impl.name, n,
                        mean_of(h2d), mean_of(ker), mean_of(d2h), std_of(ker),
                        gemm_gflops(n, mean_of(ker)), occupancy, ok ? "PASS" : "FAIL");
        }
        free_buffers(b, o.pageable);
    }

    for (auto& e : ev) cudaEventDestroy(e);
    cublasDestroy(g_cublas);
    std::fclose(csv);
    std::fclose(det);
    return 0;
}
