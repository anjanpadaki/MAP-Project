// gpu_stream.cu - STREAM-style device-memory bandwidth test plus peak-FP32 estimate.
//
// Gives the two roofs of the GPU Roofline plot:
//   * memory roof  = best measured Copy / Triad bandwidth (GB/s)
//   * compute roof = SMs x FP32 cores/SM x 2 FLOP (FMA) x boost clock
// Writes results/raw/gpu_machine.csv (overwritten on every run).
//
// Build: nvcc -O3 -arch=native src/gpu_stream.cu -o bin/gpu_stream
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err_ = (call);                                                         \
        if (err_ != cudaSuccess) {                                                         \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                         __FILE__, __LINE__);                                              \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)

__global__ void k_copy(const float* __restrict__ a, float* __restrict__ c, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        c[i] = a[i];
}

__global__ void k_triad(float* __restrict__ a, const float* __restrict__ b,
                        const float* __restrict__ c, float s, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        a[i] = b[i] + s * c[i];
}

__global__ void k_init(float* p, float v, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = v;
}

// FP32 lanes per SM for each architecture (NVIDIA CUDA C++ Programming Guide, arithmetic throughput table).
static int fp32_cores_per_sm(int major, int minor) {
    switch (major) {
        case 3: return 192;
        case 5: return 128;
        case 6: return minor == 0 ? 64 : 128;
        case 7: return 64;                     // Volta, Turing (T4)
        case 8: return minor == 0 ? 64 : 128;  // A100 : RTX 30xx / L4 / Ada
        default: return 128;                   // Hopper, Blackwell
    }
}

struct Stat { double best, mean, std; };

static Stat stats(const std::vector<double>& gbs) {
    Stat s{0, 0, 0};
    for (double g : gbs) { s.best = std::fmax(s.best, g); s.mean += g; }
    s.mean /= gbs.size();
    for (double g : gbs) s.std += (g - s.mean) * (g - s.mean);
    s.std = gbs.size() > 1 ? std::sqrt(s.std / (gbs.size() - 1)) : 0;
    return s;
}

int main(int argc, char** argv) {
    const size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : (size_t)1 << 26;  // 256 MB / array
    const int runs = 10;
    const char* out = argc > 2 ? argv[2] : "results/raw/gpu_machine.csv";

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int clock_khz = 0, mem_clock_khz = 0, bus_bits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0));

    const int cores = fp32_cores_per_sm(prop.major, prop.minor);
    const double peak_gflops = (double)prop.multiProcessorCount * cores * 2.0 * clock_khz * 1e3 / 1e9;
    const double theo_bw = 2.0 * mem_clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;  // DDR: 2 transfers/clock

    float *a, *b, *c;
    CUDA_CHECK(cudaMalloc(&a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c, n * sizeof(float)));
    const int threads = 256, blocks = prop.multiProcessorCount * 32;
    k_init<<<blocks, threads>>>(a, 1.0f, n);
    k_init<<<blocks, threads>>>(b, 2.0f, n);
    k_init<<<blocks, threads>>>(c, 0.5f, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    std::vector<double> copy_gbs, triad_gbs;
    for (int r = 0; r <= runs; ++r) {  // r == 0 is the warm-up
        float ms;
        CUDA_CHECK(cudaEventRecord(e0));
        k_copy<<<blocks, threads>>>(a, c, n);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (r > 0) copy_gbs.push_back(2.0 * n * sizeof(float) / (ms * 1e-3) / 1e9);

        CUDA_CHECK(cudaEventRecord(e0));
        k_triad<<<blocks, threads>>>(a, b, c, 3.0f, n);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (r > 0) triad_gbs.push_back(3.0 * n * sizeof(float) / (ms * 1e-3) / 1e9);
    }
    CUDA_CHECK(cudaGetLastError());
    const Stat sc = stats(copy_gbs), st = stats(triad_gbs);

    std::string name = prop.name;
    for (char& ch : name) if (ch == ',' || ch == ' ') ch = '_';
    std::printf("Device %s  cc %d.%d  SMs %d  FP32 cores/SM %d  clock %.0f MHz\n", prop.name, prop.major,
                prop.minor, prop.multiProcessorCount, cores, clock_khz / 1e3);
    std::printf("Peak FP32            : %.1f GFLOP/s\n", peak_gflops);
    std::printf("Theoretical bandwidth: %.1f GB/s (%d-bit bus, %.0f MHz)\n", theo_bw, bus_bits, mem_clock_khz / 1e3);
    std::printf("Copy  bandwidth      : best %.1f  mean %.1f +- %.1f GB/s\n", sc.best, sc.mean, sc.std);
    std::printf("Triad bandwidth      : best %.1f  mean %.1f +- %.1f GB/s\n", st.best, st.mean, st.std);

    FILE* f = std::fopen(out, "w");
    if (!f) { std::perror(out); return 1; }
    std::fprintf(f, "device,compute_capability,sms,fp32_cores_per_sm,clock_mhz,peak_fp32_gflops,"
                    "mem_clock_mhz,bus_width_bits,theoretical_bw_gbs,copy_best_gbs,copy_mean_gbs,"
                    "copy_std_gbs,triad_best_gbs,triad_mean_gbs,triad_std_gbs,global_mem_gb,"
                    "l2_cache_kb,smem_per_sm_kb,max_threads_per_sm\n");
    std::fprintf(f, "%s,%d.%d,%d,%d,%.0f,%.1f,%.0f,%d,%.1f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%zu,%d\n",
                 name.c_str(), prop.major, prop.minor, prop.multiProcessorCount, cores, clock_khz / 1e3,
                 peak_gflops, mem_clock_khz / 1e3, bus_bits, theo_bw, sc.best, sc.mean, sc.std, st.best,
                 st.mean, st.std, prop.totalGlobalMem / 1e9, prop.l2CacheSize / 1024,
                 prop.sharedMemPerMultiprocessor / 1024, prop.maxThreadsPerMultiProcessor);
    std::fclose(f);

    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}
