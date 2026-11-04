# P4 — CUDA GEMM and Cluster Integration

**Team 4:** Aditya Karan, Shivansh Agrawal, Amay Ranjan, Anjan Padaki
Multicore Architecture & Programming, MSRIT, Semester 7

Dense single-precision matrix multiplication `C = A × B` on an NVIDIA GPU, from a naive kernel through
shared-memory tiling and thread coarsening to cuBLAS. The kernels are timed separately for host-to-device copy, kernel and
device-to-host copy, profiled with Nsight Compute, and compared with an optimised sequential CPU baseline.
This team also merges the P1–P3 results into the cluster's final CPU–MPI–GPU–library comparison.

## What the brief asks for, and where it is

| Brief item | Where |
|---|---|
| Hardware description (CPU, caches, memory, GPU, compilers, flags, CUDA/MPI versions) | `scripts/hwinfo.sh` → `results/raw/hardware.txt` |
| STREAM bandwidth + peak GFLOP/s estimate (for the Roofline) | `src/stream_cpu.c`, `src/gpu_stream.cu`, `scripts/hwinfo.sh` |
| Optimised, verified sequential version compiled with `-O3` | `src/gemm_cpu.cpp` (`cpu_seq_blocked`) |
| ≥ 5 runs (we use 10), warm-up ignored, mean ± std, error bars | every benchmark; every plot |
| ≥ 3 problem sizes | GPU: 64 … 8192, CPU: 64 … 4096 |
| 1. Naive kernel, one thread per C element, 16×16 blocks | `gemm_naive` in `src/gemm_gpu.cu` |
| 2. Shared-memory tiled kernel, tile width 8, 16, 32, `__syncthreads()` | `gemm_tiled<TILE>` |
| 3. Coarsened kernel, 2×2 and 4×4 outputs per thread | `gemm_coarse<16, CF>` |
| 4. H2D, kernel, D2H timed separately with `cudaEvent` | `run_once()` in `src/gemm_gpu.cu` |
| 5. cuBLAS comparison; occupancy, achieved bandwidth, shared-memory efficiency with Nsight | `launch_cublas`, `scripts/profile_ncu.sh` |
| 6. Merge P1, P2, P3 CSVs; final comparison; GPU-wins-with-transfer size | `scripts/merge_results.py`, `plot_final()` in `scripts/plot_results.py` |
| Submit: bandwidth and occupancy vs tile size | `results/plots/fig05_bandwidth_occupancy_vs_tile.png` |
| Submit: waterfall chart of speedup per optimisation | `results/plots/fig06_waterfall_speedup.png` |
| Submit: transfer-vs-kernel chart | `results/plots/fig07_transfer_vs_kernel.png` |
| Submit: final comparison chart | `results/plots/fig10_final_comparison.png` |
| Standard graphs: time vs size, speedup, efficiency, Roofline, tuning parameter | `fig01`–`fig04`, `fig08`, `fig09` |

Weak scaling and speedup vs threads/ranks don't really apply to a single-GPU kernel. The report explains why and uses speedup
vs problem size and % of peak instead.

## Repository layout

```
src/                 gemm_gpu.cu (kernels + cuBLAS), gemm_cpu.cpp (baseline), gpu_stream.cu, stream_cpu.c, common.h
scripts/             run_all.sh, hwinfo.sh, profile_ncu.sh, merge_results.py, plot_results.py
notebooks/           P4_colab.ipynb — one-click run on Google Colab (Tesla T4)
results/raw/         CSVs written by the benchmarks (+ hardware.txt, ncu export)
results/external/    drop the P1, P2, P3 team CSVs here
results/merged/      cluster_results.csv (all teams, one schema)
results/plots/       every graph
results/summary.md   generated tables: times, GFLOP/s, speedups, ncu metrics, crossover size
report/report.md     the report
LOG.md               work log
```

## Running it

You need an NVIDIA GPU with the CUDA toolkit (nvcc, cuBLAS), g++, and Python 3 with pandas, numpy and matplotlib.
If you don't have an NVIDIA GPU locally, use the Colab notebook (Runtime → Change runtime type → T4 GPU).

```bash
git clone https://github.com/anjanpadaki/MAP-Project.git && cd MAP-Project
bash scripts/run_all.sh            # build, hardware info, STREAM, CPU + GPU benchmarks, ncu, merge, plots
```

Running the steps one at a time:

```bash
make                                         # ARCH=sm_75 if -arch=native is not supported by your nvcc
bash scripts/hwinfo.sh
./bin/stream_cpu && OMP_NUM_THREADS=1 ./bin/stream_cpu
./bin/gpu_stream
./bin/gemm_cpu --sizes 64,128,256,512,1024,2048 --runs 10
./bin/gemm_gpu --sizes 64,128,256,512,1024,2048,4096,8192 --runs 10
bash scripts/profile_ncu.sh 2048
python3 scripts/merge_results.py
python3 scripts/plot_results.py
```

Useful options: `--impl gpu_tiled_t16,cublas_sgemm` (subset), `--pageable` (pageable instead of pinned host memory,
with `--out`/`--detail` set to other files), `make NVML=1` (records GPU power into `power_w`).
`run_all.sh` takes `RUNS`, `GPU_SIZES`, `CPU_SIZES`, `SKIP_CPU`, `SKIP_GPU`, `SKIP_NCU` from the environment.

## Common CSV schema

Every row is one recorded run (warm-up runs are never written):

| column | P4 meaning |
|---|---|
| `cluster`, `problem` | `C1` (set with `--cluster`), `P4` |
| `platform` | `GPU-<device name>` or `CPU-<cpu model>` |
| `implementation` | `cpu_seq_naive`, `cpu_seq_blocked`, `gpu_naive_b16`, `gpu_tiled_t8/16/32`, `gpu_coarse2x2_t16`, `gpu_coarse4x4_t16`, `cublas_sgemm` |
| `dataset` | `uniform_m1_1_seed42`: A and B uniform in [-1, 1], fixed seeds |
| `size` | N (matrices are N × N) |
| `threads_or_ranks` | CPU: 1. GPU: threads per block (0 = chosen by cuBLAS) |
| `run_id` | 1 … R |
| `total_ms` | H2D + kernel + D2H (CPU: compute time) |
| `kernel_ms` | kernel time from `cudaEvent` (CPU: compute time) |
| `transfer_ms` | H2D + D2H |
| `throughput` | GFLOP/s = 2N³ / kernel time |
| `error` | max \|C − C_ref\| / max \|C_ref\| over 4096 random entries recomputed in FP64 |
| `power_w` | GPU board power (only with `make NVML=1`), otherwise empty |

The H2D/D2H split, register count, shared memory per block and theoretical occupancy go in
`results/raw/p4_gemm_gpu_detail.csv`.

## Merging the other teams' results (task 6)

1. Ask P1, P2 and P3 for their CSVs in the common schema and copy them into `results/external/`
   (for example `P1_openmp.csv`, `P2_mpi.csv`, `P3_simd.csv`).
2. Run `python3 scripts/merge_results.py`, then `python3 scripts/plot_results.py`.
3. The merge script classifies each row as CPU sequential / CPU parallel / MPI / CPU library / GPU kernel / GPU library,
   based on `implementation`, `platform` and `threads_or_ranks`. Check the printed table. If a team used unusual names, add a keyword to
   `categorise()`.
4. `results/summary.md` gives the size from which the GPU (custom kernels, and cuBLAS) beats the best CPU or MPI result
   **with transfer time included**. It interpolates on a log-log scale between measured sizes.
