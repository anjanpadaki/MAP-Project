#!/usr/bin/env bash
# Full P4 pipeline: build -> hardware info -> STREAM (CPU + GPU) -> CPU baseline
# -> GPU kernels -> Nsight Compute profile -> merge cluster CSVs -> plots.
#
# Tunables (environment variables):
#   RUNS=10  GPU_SIZES=...  CPU_SIZES=...  CPU_BIG_SIZES=4096  CPU_BIG_RUNS=5
#   CLUSTER=C1  PROFILE_N=2048  SKIP_CPU=1  SKIP_GPU=1  SKIP_NCU=1
set -euo pipefail
cd "$(dirname "$0")/.."

RUNS=${RUNS:-10}
GPU_SIZES=${GPU_SIZES:-64,128,256,512,1024,2048,4096,8192}
CPU_SIZES=${CPU_SIZES:-64,128,256,512,1024,2048}
CPU_BIG_SIZES=${CPU_BIG_SIZES:-4096}
CPU_BIG_RUNS=${CPU_BIG_RUNS:-5}
CLUSTER=${CLUSTER:-C1}
PROFILE_N=${PROFILE_N:-2048}

mkdir -p results/raw results/plots results/merged

# Benchmarks append to their CSVs, so archive the previous results first.
stamp=$(date +%Y%m%d_%H%M%S)
shopt -s nullglob
old=(results/raw/p4_*.csv results/raw/stream_cpu.csv)
if [ ${#old[@]} -gt 0 ]; then
    mkdir -p "results/raw/archive/$stamp"
    mv "${old[@]}" "results/raw/archive/$stamp/"
    echo "Archived previous results to results/raw/archive/$stamp"
fi
shopt -u nullglob

echo "== build"
if [ "${SKIP_GPU:-0}" = 1 ]; then make cpu; else make all; fi

echo "== hardware description"
bash scripts/hwinfo.sh

if [ "${SKIP_CPU:-0}" != 1 ]; then
    echo "== STREAM (CPU): all threads, then one thread"
    ./bin/stream_cpu
    OMP_NUM_THREADS=1 ./bin/stream_cpu

    cpu_name=$(awk -F, 'NR==2 {gsub(/"/, "", $1); gsub(/[ ()@]/, "_", $1); print $1}' results/raw/cpu_peak.csv)
    echo "== CPU sequential baseline"
    ./bin/gemm_cpu --cluster "$CLUSTER" --platform "CPU-$cpu_name" --sizes "$CPU_SIZES" --runs "$RUNS"
    [ -n "$CPU_BIG_SIZES" ] && ./bin/gemm_cpu --cluster "$CLUSTER" --platform "CPU-$cpu_name" \
        --impl cpu_seq_blocked --sizes "$CPU_BIG_SIZES" --runs "$CPU_BIG_RUNS"
fi

if [ "${SKIP_GPU:-0}" != 1 ]; then
    echo "== STREAM (GPU) and peak FP32"
    ./bin/gpu_stream
    echo "== GPU GEMM kernels"
    ./bin/gemm_gpu --cluster "$CLUSTER" --sizes "$GPU_SIZES" --runs "$RUNS"
    if [ "${SKIP_NCU:-0}" != 1 ]; then
        echo "== Nsight Compute profile at N=$PROFILE_N"
        bash scripts/profile_ncu.sh "$PROFILE_N" || echo "profiling failed - continuing without it"
    fi
fi

echo "== merge + plots"
python3 scripts/merge_results.py
python3 scripts/plot_results.py --profile-n "$PROFILE_N"
echo "Done. Graphs in results/plots/, summary in results/summary.md"
