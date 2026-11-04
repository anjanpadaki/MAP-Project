# P4 — CUDA GEMM and Cluster Integration

**Team 4:** Aditya Karan, Shivansh Agrawal, Amay Ranjan, Anjan Padaki

> Wherever the text says **[fill]**, copy the number from `results/summary.md` or `results/raw/hardware.txt` after the
> measured run. Do not keep numbers from anywhere else.

## 1. Problem

Single-precision matrix multiplication `C = A·B` with square N × N row-major matrices costs 2N³ FLOP and touches 3N² values.
In principle it is compute-bound: arithmetic intensity can reach O(N). But a naive kernel re-reads A and B from global memory
for every output, which makes it memory-bound. The project measures how much shared-memory tiling, thread coarsening and
finally cuBLAS recover, and whether the GPU still wins once host↔device transfers are counted.

## 2. Hardware and software (common baseline)

| Item | Value |
|---|---|
| CPU model, cores / threads | **[fill from hardware.txt]** |
| Caches (L1d / L2 / L3) | **[fill]** |
| Memory | **[fill]** |
| GPU, memory, SMs, compute capability | **[fill]** (e.g. Tesla T4, 16 GB GDDR6, 40 SMs, cc 7.5) |
| Driver / CUDA version | **[fill]** |
| Compilers and flags | g++ `-O3 -march=native -std=c++14`; nvcc `-O3 -arch=native -lineinfo`; cuBLAS |
| MPI | not used by P4 |
| CPU STREAM triad (1 thread / all threads) | **[fill]** GB/s |
| GPU STREAM copy / triad, theoretical | **[fill]** GB/s |
| CPU peak FP32 (cores × GHz × FLOP/cycle) | **[fill]** GFLOP/s |
| GPU peak FP32 (SMs × 64 lanes × 2 × boost clock) | **[fill]** GFLOP/s |

## 3. Methodology

- **Data:** A and B are filled with uniform values in [-1, 1] from fixed seeds (42, 43). Every implementation gets
  identical inputs.
- **Verification:** 4096 random entries of C are recomputed in FP64, and we report `max|C − ref| / max|ref|` in the
  `error` column. Every GPU kernel is also compared element by element with the cuBLAS result. Tolerance is 1e-4.
- **Baseline:** `cpu_seq_blocked` is a cache-blocked i-k-j loop (blocks 32 × 128 × 256) whose unit-stride inner loop
  `-O3 -march=native` auto-vectorises. All speedups use it as the reference.
- **Timing:** one warm-up run (never recorded), then 10 recorded runs. We report mean ± standard deviation and draw error bars on every graph.
  On the GPU, `cudaEvent`s bracket H2D (A and B), the kernel and D2H (C) separately. Host buffers are pinned
  (`cudaMallocHost`). On the CPU, sizes below the clock resolution repeat the call until a run lasts at least 20 ms.
- **Profiling:** Nsight Compute (`--set full`) at N = 2048, one launch per kernel.
- **Sizes:** GPU 64 to 8192, CPU 64 to 4096 (naive CPU only up to 512).

## 4. Implementations

| Name | Idea | Global-memory traffic (model) | Intensity (FLOP/B) |
|---|---|---|---|
| `gpu_naive_b16` | one thread per C element, 16×16 blocks | 8N³ B | 0.25 |
| `gpu_tiled_tT` | TILE×TILE tiles of A and B staged in shared memory, two `__syncthreads()` per tile | 8N³/T B | T/4 |
| `gpu_coarse2x2_t16` | 16×16 threads, 2×2 outputs each (32×32 C tile) | 8N³/32 B | 8 |
| `gpu_coarse4x4_t16` | 16×16 threads, 4×4 outputs each (64×64 C tile) | 8N³/64 B | 16 |
| `cublas_sgemm` | vendor library (register-blocked, double-buffered) | — | measured |

Why each step should help:
- **Tiling** reuses each staged value T times, cutting DRAM/L2 traffic by T.
- **Larger tiles** cut traffic further, but T = 32 means 1024 threads per block and at most one or two resident blocks per SM.
  T = 8 has only 2 warps per block and hits the blocks-per-SM limit. Look for this in the occupancy panel of Fig. 5.
- **Coarsening** keeps CF² accumulators in registers. Each shared-memory load then feeds CF FMAs, so the kernel stops being
  limited by shared-memory bandwidth, the main limit of the plain tiled kernel. The cost is more registers per thread, which can lower occupancy.

## 5. Results

### 5.1 Time and throughput vs problem size
![](../results/plots/fig01_time_and_gflops_vs_size.png)

**[fill: 2–3 sentences. Where does each kernel stop improving with N? Peak GFLOP/s of each.]**

### 5.2 Speedup over the sequential CPU
![](../results/plots/fig02_speedup_vs_size.png)

**[fill: the largest kernel-only speedup and the speedup with transfer included. Why small N gains little.]**

### 5.3 Efficiency (% of peak FP32)
![](../results/plots/fig03_efficiency_vs_size.png)

### 5.4 Main tuning parameter: tile width
![](../results/plots/fig04_tile_width_tuning.png)

**[fill: best tile width, and the occupancy / synchronisation argument.]**

### 5.5 Waterfall — speedup contributed by each optimisation
![](../results/plots/fig06_waterfall_speedup.png)

### 5.6 Transfer vs kernel time
![](../results/plots/fig07_transfer_vs_kernel.png)

Transfers grow as N² (3·4N² bytes over PCIe) while compute grows as N³. Below some size, H2D + D2H dominate.
**[fill: the size where kernel time overtakes transfer time for the tiled kernel and for cuBLAS.]**

## 6. Profiling (Nsight Compute)

![](../results/plots/fig05_bandwidth_occupancy_vs_tile.png)

![](../results/plots/fig08_roofline_gpu.png)

**[fill: table from summary.md. Achieved vs theoretical occupancy per kernel, DRAM GB/s, shared-memory efficiency.
Place each kernel on the Roofline: memory-bound or compute-bound?]**

CPU baseline on its own (single-core) Roofline:

![](../results/plots/fig09_roofline_cpu.png)

## 7. Cluster comparison (P1, P2, P3, P4)

![](../results/plots/fig10_final_comparison.png)

| Problem | Team | Implementation | Platform |
|---|---|---|---|
| P1 | **[fill]** | | |
| P2 | **[fill]** | | |
| P3 | **[fill]** | | |
| P4 | Team 4 | CUDA kernels, cuBLAS | **[fill GPU]** |

**Size at which the GPU wins when transfer time is counted:** **[fill from summary.md]**.
Note that other teams may have measured on different machines. Say which ones.

## 8. Graphs that do not apply

- **Speedup / efficiency vs threads or ranks:** a CUDA kernel's "thread count" is set by the problem size, so
  we report speedup vs N and % of peak instead.
- **Weak scaling:** P4 uses one GPU, so the resources are fixed. Strong scaling over N is shown in Fig. 1 and Fig. 2.

## 9. Conclusions

**[fill: 4–6 bullet points]**

## References

1. D. Kirk, W. Hwu, *Programming Massively Parallel Processors*, 4th ed. — tiled matrix multiplication and thread coarsening chapters.
2. NVIDIA, *CUDA C++ Programming Guide* — shared memory, arithmetic instruction throughput.
3. NVIDIA, *cuBLAS Library User Guide* — `cublasSgemm`.
4. NVIDIA, *Nsight Compute Kernel Profiling Guide* — metrics reference.
5. J. McCalpin, *STREAM: Sustainable Memory Bandwidth in High Performance Computers*.
6. S. Williams, A. Waterman, D. Patterson, "Roofline: an insightful visual performance model", CACM 2009.
