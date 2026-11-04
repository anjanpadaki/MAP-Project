#!/usr/bin/env bash
# Collects the hardware/software description required by the common baseline
# and estimates the CPU peak GFLOP/s for the Roofline plot.
#   results/raw/hardware.txt  - human-readable description (copy into the report)
#   results/raw/cpu_peak.csv  - numbers read by scripts/plot_results.py
# Override the estimate with CPU_GHZ=... FLOPS_PER_CYCLE=... if lscpu is incomplete.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/raw
OUT=results/raw/hardware.txt

section() { printf '\n===== %s =====\n' "$1"; }
{
    echo "Generated: $(date -u '+%Y-%m-%d %H:%M UTC')  host: $(hostname)"
    section "CPU (lscpu)"
    lscpu 2>/dev/null || echo "lscpu not available"
    section "Caches"
    lscpu -C 2>/dev/null || grep -H . /sys/devices/system/cpu/cpu0/cache/index*/size 2>/dev/null
    section "Memory"
    free -h 2>/dev/null
    section "OS"
    uname -a
    section "GPU (nvidia-smi)"
    nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
    nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem,power.limit,driver_version \
        --format=csv 2>/dev/null
    section "Compilers"
    g++ --version 2>/dev/null | head -1
    gcc --version 2>/dev/null | head -1
    nvcc --version 2>/dev/null | tail -2
    section "MPI"
    (mpirun --version 2>/dev/null || echo "MPI not installed (not used by P4)") | head -2
    section "Build flags"
    make -s flags 2>/dev/null
} > "$OUT"

# ---- CPU peak estimate: cores x GHz x FLOP/cycle/core (FP32) ----
model=$(lscpu 2>/dev/null | awk -F: '/Model name/ {gsub(/^ +/, "", $2); print $2; exit}')
sockets=$(lscpu 2>/dev/null | awk -F: '/^Socket\(s\)/ {gsub(/ /, "", $2); print $2; exit}')
cps=$(lscpu 2>/dev/null | awk -F: '/Core\(s\) per socket/ {gsub(/ /, "", $2); print $2; exit}')
threads=$(nproc 2>/dev/null || echo 1)
cores=$(( ${sockets:-1} * ${cps:-1} ))
mhz=$(lscpu 2>/dev/null | awk -F: '/CPU max MHz/ {gsub(/ /, "", $2); print $2; exit}')
[ -z "$mhz" ] && mhz=$(awk -F: '/cpu MHz/ {gsub(/ /, "", $2); print $2; exit}' /proc/cpuinfo 2>/dev/null)
ghz=${CPU_GHZ:-$(awk -v m="${mhz:-2000}" 'BEGIN {printf "%.3f", m / 1000}')}

flags=$(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null)
if [[ $flags == *avx512f* ]]; then simd=AVX-512; fpc=64      # 2 FMA units x 16 lanes x 2
elif [[ $flags == *avx2* && $flags == *fma* ]]; then simd=AVX2+FMA; fpc=32  # 2 x 8 x 2
elif [[ $flags == *avx* ]]; then simd=AVX; fpc=16
else simd=SSE; fpc=8; fi
fpc=${FLOPS_PER_CYCLE:-$fpc}

peak1=$(awk -v g="$ghz" -v f="$fpc" 'BEGIN {printf "%.1f", g * f}')
peakall=$(awk -v g="$ghz" -v f="$fpc" -v c="$cores" 'BEGIN {printf "%.1f", g * f * c}')
{
    echo "cpu_model,cores,threads,ghz,simd,flops_per_cycle_per_core,peak_gflops_1core,peak_gflops_all"
    echo "\"${model:-unknown}\",$cores,$threads,$ghz,$simd,$fpc,$peak1,$peakall"
} > results/raw/cpu_peak.csv
{
    section "Peak FP32 estimate"
    echo "$cores cores x $ghz GHz x $fpc FLOP/cycle ($simd) = $peakall GFLOP/s ($peak1 GFLOP/s per core)"
} >> "$OUT"

echo "Wrote $OUT and results/raw/cpu_peak.csv"
echo "CPU peak estimate: $peakall GFLOP/s all cores, $peak1 GFLOP/s single core ($simd)"
