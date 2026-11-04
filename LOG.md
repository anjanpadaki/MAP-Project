# Work log — Team 4, P4 (CUDA GEMM and Cluster Integration)

Add an entry for every work session: date, who, what was done, what was measured, problems, next steps.

## 2026-11-04 — repository set up
- Created the repository structure (`src/`, `scripts/`, `results/raw/`, `results/plots/`, `report/`, `LOG.md`).
- Implemented the CPU baseline (`cpu_seq_naive`, cache-blocked `cpu_seq_blocked`, `-O3 -march=native`) with
  FP64 sampled verification.
- Implemented the CUDA kernels: naive (16×16), tiled T = 8/16/32, coarsened 2×2 and 4×4 (T = 16), plus cuBLAS Sgemm.
  Separate cudaEvent timing of H2D, kernel and D2H; full-matrix cross-check against cuBLAS.
- Added the STREAM-style CPU and GPU bandwidth benchmarks, the peak-FLOP estimate, the Nsight Compute profiling script,
  the CSV merge script and the plotting script.
- Tested the CPU code and the plotting pipeline locally (no NVIDIA GPU on the laptop: AMD Ryzen 7 5700U with Radeon graphics).
- **Next:** run `notebooks/P4_colab.ipynb` on a Colab T4, commit `results/`, collect the P1–P3 CSVs.

## YYYY-MM-DD — first GPU run
- Platform:
- Commands run:
- Observations:
- Problems / fixes:
