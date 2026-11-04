#!/usr/bin/env bash
# Profiles every GEMM kernel once with Nsight Compute (occupancy, DRAM bandwidth,
# shared-memory wavefronts/bank conflicts, FLOP counts) and exports a CSV that
# scripts/plot_results.py turns into the bandwidth/occupancy-vs-tile-size graph.
#   usage: bash scripts/profile_ncu.sh [N]       (default N = 2048)
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:-2048}
mkdir -p results/raw

NCU=$(command -v ncu || true)
for cand in /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu /usr/local/cuda/nsight-compute*/ncu; do
    [ -z "$NCU" ] && [ -x "$cand" ] && NCU=$cand
done
if [ -z "$NCU" ]; then
    echo "ncu (Nsight Compute) not found. On Colab/Ubuntu install it with:"
    echo "  apt-get install -y nsight-compute-\$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | tr . -)"
    echo "Skipping profiling; plots will fall back to theoretical occupancy."
    exit 0
fi
echo "Using $NCU"

REP=results/raw/ncu_gemm_N${N}
# --set full collects every section (SpeedOfLight, MemoryWorkloadAnalysis,
# Occupancy, LaunchStats, ...) so no metric name has to be hard-coded here.
"$NCU" --set full --target-processes all -f -o "$REP" ./bin/gemm_gpu --profile "$N"
"$NCU" --import "$REP.ncu-rep" --csv --page raw --print-units base > results/raw/ncu_metrics.csv
"$NCU" --import "$REP.ncu-rep" --page details > "results/raw/ncu_details_N${N}.txt"
echo "Wrote results/raw/ncu_metrics.csv and results/raw/ncu_details_N${N}.txt"
