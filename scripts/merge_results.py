#!/usr/bin/env python3
"""Merge every team's common-schema CSV into results/merged/cluster_results.csv.

Inputs
  results/raw/p4_gemm_cpu.csv, results/raw/p4_gemm_gpu.csv   (this team, P4)
  results/external/*.csv                                      (P1, P2, P3 - copy their files here)

Each file must have the 14 columns of the common schema. Rows from run_id 0
(warm-up) are dropped, numeric columns are coerced, and each row gets a
`category` used by the final comparison chart:
  CPU sequential | CPU parallel | MPI | GPU kernel | GPU library | CPU library
Run from the repository root:  python scripts/merge_results.py
"""
import argparse
import glob
import os
import sys

import pandas as pd

COLUMNS = ["cluster", "problem", "platform", "implementation", "dataset", "size",
           "threads_or_ranks", "run_id", "total_ms", "kernel_ms", "transfer_ms",
           "throughput", "error", "power_w"]
NUMERIC = ["size", "threads_or_ranks", "run_id", "total_ms", "kernel_ms", "transfer_ms",
           "throughput", "error", "power_w"]
LIBRARIES = ("cublas", "mkl", "openblas", "blis", "blas", "eigen")


def categorise(row):
    impl = str(row["implementation"]).lower()
    plat = str(row["platform"]).lower()
    on_gpu = "gpu" in plat or "cuda" in plat or impl.startswith("gpu")
    if any(lib in impl for lib in LIBRARIES):
        return "GPU library" if on_gpu else "CPU library"
    if on_gpu:
        return "GPU kernel"
    if "mpi" in impl or "mpi" in plat:
        return "MPI"
    if row["threads_or_ranks"] > 1:
        return "CPU parallel"
    return "CPU sequential"


def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        print(f"  SKIP {path}: missing columns {missing}")
        return None
    df = df[COLUMNS].copy()
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["run_id"] > 0]  # warm-up runs are never reported
    df["source_file"] = os.path.relpath(path)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="*", default=None, help="CSV files (default: P4 raw + results/external)")
    ap.add_argument("--out", default="results/merged/cluster_results.csv")
    args = ap.parse_args()

    files = args.inputs or (sorted(glob.glob("results/raw/p4_gemm_cpu.csv"))
                            + sorted(glob.glob("results/raw/p4_gemm_gpu.csv"))
                            + sorted(glob.glob("results/external/*.csv")))
    frames = []
    for f in files:
        df = load(f)
        if df is not None and len(df):
            print(f"  {f}: {len(df)} rows, problems {sorted(df['problem'].unique())}")
            frames.append(df)
    if not frames:
        print("No result files found.")
        sys.exit(1)

    merged = pd.concat(frames, ignore_index=True)
    merged["category"] = merged.apply(categorise, axis=1)
    bad = merged[merged["total_ms"].isna() | (merged["total_ms"] <= 0)]
    if len(bad):
        print(f"  WARNING: {len(bad)} rows with missing/non-positive total_ms dropped")
        merged = merged.drop(bad.index)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"Wrote {args.out}: {len(merged)} rows")
    summary = (merged.groupby(["problem", "category", "implementation"])["size"]
               .agg(lambda s: ",".join(str(int(x)) for x in sorted(s.unique()))))
    print(summary.to_string())


if __name__ == "__main__":
    main()
