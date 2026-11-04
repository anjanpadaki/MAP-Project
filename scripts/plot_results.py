#!/usr/bin/env python3
"""Generate every P4 graph and results/summary.md from the measured CSVs.

Run from the repository root after the benchmarks (scripts/run_all.sh does it):
    python scripts/plot_results.py [--profile-n 2048] [--waterfall-n 2048]

Inputs (missing ones are skipped with a message, nothing is invented):
  results/raw/p4_gemm_cpu.csv          CPU sequential baseline   (common schema)
  results/raw/p4_gemm_gpu.csv          GPU kernels + cuBLAS      (common schema)
  results/raw/p4_gemm_gpu_detail.csv   H2D / D2H split, occupancy
  results/raw/gpu_machine.csv          GPU peak FP32 + STREAM bandwidth
  results/raw/cpu_peak.csv, stream_cpu.csv
  results/raw/ncu_metrics.csv          Nsight Compute export (scripts/profile_ncu.sh)
  results/merged/cluster_results.csv   P1-P4 merged (scripts/merge_results.py)

Every point is the mean of the recorded runs (warm-up excluded); error bars are
one standard deviation.
"""
import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAW, PLOTS = "results/raw", "results/plots"

# Validated categorical palette (fixed order, light mode) + neutral greys for the CPU reference.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

GPU_IMPLS = ["gpu_naive_b16", "gpu_tiled_t8", "gpu_tiled_t16", "gpu_tiled_t32",
             "gpu_coarse2x2_t16", "gpu_coarse4x4_t16", "cublas_sgemm"]
LABEL = {
    "cpu_seq_naive": "CPU seq naive (ijk)",
    "cpu_seq_blocked": "CPU seq blocked (-O3)",
    "gpu_naive_b16": "GPU naive 16×16",
    "gpu_tiled_t8": "Tiled T=8",
    "gpu_tiled_t16": "Tiled T=16",
    "gpu_tiled_t32": "Tiled T=32",
    "gpu_coarse2x2_t16": "Coarsened 2×2",
    "gpu_coarse4x4_t16": "Coarsened 4×4",
    "cublas_sgemm": "cuBLAS Sgemm",
}
COLOR = {impl: SERIES[i] for i, impl in enumerate(GPU_IMPLS)}
COLOR["cpu_seq_blocked"] = INK2
COLOR["cpu_seq_naive"] = MUTED
BASELINE = "cpu_seq_blocked"
TILE_OF = {"gpu_tiled_t8": 8, "gpu_tiled_t16": 16, "gpu_tiled_t32": 32}
# DRAM-traffic model (no cache reuse beyond shared memory): FLOP per byte.
MODEL_AI = {"gpu_naive_b16": 0.25, "gpu_tiled_t8": 2.0, "gpu_tiled_t16": 4.0, "gpu_tiled_t32": 8.0,
            "gpu_coarse2x2_t16": 8.0, "gpu_coarse4x4_t16": 16.0}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.labelsize": 10, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 1.6, "lines.markersize": 6,
    "font.size": 9.5, "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
})
ERR = dict(capsize=3, elinewidth=1, capthick=1)

summary = []  # markdown lines for results/summary.md


def md(obj):
    """DataFrame/Series -> GitHub markdown table (no tabulate dependency)."""
    df = obj.to_frame() if isinstance(obj, pd.Series) else obj
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        out.append("| " + " | ".join([str(idx)] + ["" if pd.isna(v) else str(v) for v in row]) + " |")
    return "\n".join(out)


def note(msg):
    print("  " + msg)


def read(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def agg(df, col, keys=("implementation", "size")):
    g = df.groupby(list(keys))[col].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g


def save(fig, name):
    os.makedirs(PLOTS, exist_ok=True)
    path = os.path.join(PLOTS, name)
    fig.savefig(path)
    plt.close(fig)
    note(f"wrote {path}")


def log2_sizes(ax, sizes):
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(sizes))
    ax.set_xticklabels([str(int(s)) for s in sorted(sizes)])
    ax.minorticks_off()


def ratio_std(a, sa, b, sb):
    """std of a/b from independent means/stds (first-order propagation)."""
    r = a / b
    return r * np.sqrt((sa / a) ** 2 + (sb / b) ** 2)


# ---------------------------------------------------------------------------
# Standard graphs
# ---------------------------------------------------------------------------

def plot_time_and_gflops(p4):
    t = agg(p4, "kernel_ms")
    impls = [i for i in [BASELINE, "cpu_seq_naive"] + GPU_IMPLS if i in set(t.implementation)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for impl in impls:
        d = t[t.implementation == impl].sort_values("size")
        style = "--" if impl.startswith("cpu") else "-"
        axes[0].errorbar(d["size"], d["mean"], yerr=d["std"], fmt="o" + style, color=COLOR[impl],
                         label=LABEL.get(impl, impl), **ERR)
        gf = 2 * d["size"].astype(float) ** 3 / (d["mean"] * 1e-3) / 1e9
        gf_std = gf * d["std"] / d["mean"]
        axes[1].errorbar(d["size"], gf, yerr=gf_std, fmt="o" + style, color=COLOR[impl],
                         label=LABEL.get(impl, impl), **ERR)
    for ax in axes:
        log2_sizes(ax, t["size"].unique())
        ax.set_xlabel("Matrix size N (N×N)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Compute time (ms, log scale)")
    axes[0].set_title("Time vs problem size (kernel only)")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("GFLOP/s (log scale)")
    axes[1].set_title("Throughput vs problem size")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig01_time_and_gflops_vs_size.png")


def plot_speedup(p4):
    t = agg(p4, "kernel_ms")
    tot = agg(p4, "total_ms")
    base = t[t.implementation == BASELINE].set_index("size")
    if base.empty:
        note("no CPU baseline - skipping speedup plot")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    rows = []
    for ax, src, title in [(axes[0], t, "kernel time only"), (axes[1], tot, "including H2D + D2H transfer")]:
        for impl in GPU_IMPLS:
            d = src[src.implementation == impl].set_index("size")
            common = sorted(set(d.index) & set(base.index))
            if not common:
                continue
            s = base.loc[common, "mean"] / d.loc[common, "mean"]
            ss = ratio_std(base.loc[common, "mean"], base.loc[common, "std"], d.loc[common, "mean"],
                           d.loc[common, "std"])
            ax.errorbar(common, s, yerr=ss, fmt="o-", color=COLOR[impl], label=LABEL[impl], **ERR)
            if src is t:
                rows += [(impl, n, v) for n, v in zip(common, s)]
        ax.axhline(1, color=AXIS, lw=1, ls="--")
        ax.set_yscale("log")
        log2_sizes(ax, base.index)
        ax.set_xlabel("Matrix size N")
        ax.set_title(f"Speedup over optimised sequential CPU\n({title})")
    axes[0].set_ylabel("Speedup = T_seq / T_gpu (log scale)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig02_speedup_vs_size.png")

    if rows:
        sp = pd.DataFrame(rows, columns=["implementation", "size", "speedup"]).pivot(
            index="implementation", columns="size", values="speedup")
        sp = sp.reindex([i for i in GPU_IMPLS if i in sp.index])
        summary.append("## Speedup over `cpu_seq_blocked` (kernel time)\n")
        summary.append(md(sp.round(1)) + "\n")


def plot_efficiency(p4, gpu_machine):
    if gpu_machine is None:
        note("no gpu_machine.csv - skipping efficiency plot")
        return
    peak = float(gpu_machine["peak_fp32_gflops"].iloc[0])
    t = agg(p4[p4.implementation.isin(GPU_IMPLS)], "throughput")
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for impl in GPU_IMPLS:
        d = t[t.implementation == impl].sort_values("size")
        if len(d):
            ax.errorbar(d["size"], 100 * d["mean"] / peak, yerr=100 * d["std"] / peak, fmt="o-",
                        color=COLOR[impl], label=LABEL[impl], **ERR)
    log2_sizes(ax, t["size"].unique())
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Matrix size N")
    ax.set_ylabel("% of peak FP32")
    ax.set_title(f"GPU efficiency: achieved / peak ({peak:,.0f} GFLOP/s)")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig03_efficiency_vs_size.png")


def plot_tile_tuning(p4):
    t = agg(p4[p4.implementation.isin(TILE_OF)], "throughput")
    if t.empty:
        return
    t["tile"] = t.implementation.map(TILE_OF)
    sizes = sorted(s for s in t["size"].unique() if s >= 256)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for k, n in enumerate(sizes):
        d = t[t["size"] == n].sort_values("tile")
        color = BLUE_RAMP[min(len(BLUE_RAMP) - 1, k + max(0, len(BLUE_RAMP) - len(sizes)))]
        ax.errorbar(d["tile"], d["mean"], yerr=d["std"], fmt="o-", color=color, label=f"N = {n}", **ERR)
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32])
    ax.set_xticklabels(["8 (64 thr)", "16 (256 thr)", "32 (1024 thr)"])
    ax.minorticks_off()
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Tile width T (threads per block)")
    ax.set_ylabel("GFLOP/s (kernel)")
    ax.set_title("Main tuning parameter: shared-memory tile width")
    ax.legend(title="matrix size", loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "fig04_tile_width_tuning.png")


# ---------------------------------------------------------------------------
# Nsight Compute: bandwidth / occupancy / shared-memory efficiency vs tile size
# ---------------------------------------------------------------------------

UNIT_SCALE = {"byte": 1, "kbyte": 1e3, "mbyte": 1e6, "gbyte": 1e9, "tbyte": 1e12,
              "nsecond": 1e-9, "usecond": 1e-6, "msecond": 1e-3, "second": 1,
              "byte/second": 1, "kbyte/second": 1e3, "mbyte/second": 1e6,
              "gbyte/second": 1e9, "tbyte/second": 1e12, "%": 1, "": 1}


def kernel_to_impl(name):
    name = str(name)
    m = re.search(r"gemm_tiled<\s*(\d+)\s*>", name)
    if m:
        return f"gpu_tiled_t{m.group(1)}"
    m = re.search(r"gemm_coarse<\s*16\s*,\s*(\d)\s*>", name)
    if m:
        return f"gpu_coarse{m.group(1)}x{m.group(1)}_t16"
    if "gemm_naive" in name:
        return "gpu_naive_b16"
    if "gemm" in name.lower() or "sgemm" in name.lower() or "xmma" in name.lower():
        return "cublas_sgemm"
    return None


def load_ncu(path):
    if not os.path.exists(path):
        return None
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
    if start is None:
        return None
    from io import StringIO
    df = pd.read_csv(StringIO("\n".join(lines[start:])), dtype=str)
    units = df.iloc[0]
    df = df[pd.to_numeric(df["ID"], errors="coerce").notna()].copy()

    def metric(col):
        if col not in df.columns:
            return pd.Series(np.nan, index=df.index)
        scale = UNIT_SCALE.get(str(units.get(col, "")).strip().lower(), 1)
        vals = pd.to_numeric(df[col].str.replace(",", "", regex=False), errors="coerce")
        return vals * scale

    out = pd.DataFrame({"implementation": df["Kernel Name"].map(kernel_to_impl)})
    out["duration_s"] = metric("gpu__time_duration.sum")
    out["occ_achieved"] = metric("sm__warps_active.avg.pct_of_peak_sustained_active")
    out["occ_theoretical"] = metric("sm__maximum_warps_per_active_cycle_pct")
    rd, wr, tot = metric("dram__bytes_read.sum"), metric("dram__bytes_write.sum"), metric("dram__bytes.sum")
    out["dram_bytes"] = tot.where(tot.notna(), rd + wr)
    bw = metric("dram__bytes.sum.per_second")
    out["dram_gbs"] = bw.where(bw.notna(), out["dram_bytes"] / out["duration_s"]) / 1e9
    wav = metric("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum").fillna(0) + \
        metric("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum").fillna(0)
    conf = metric("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum").fillna(0) + \
        metric("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum").fillna(0)
    out["smem_efficiency"] = np.where(wav > 0, 100 * (1 - conf / wav.replace(0, np.nan)), np.nan)
    out["sm_throughput_pct"] = metric("sm__throughput.avg.pct_of_peak_sustained_elapsed")
    out = out.dropna(subset=["implementation"])
    # cuBLAS may launch helper kernels: keep the longest launch per implementation.
    out = out.sort_values("duration_s").groupby("implementation").tail(1)
    return out.set_index("implementation").reindex([i for i in GPU_IMPLS if i in set(out.implementation)])


def plot_bw_occupancy(ncu, detail, profile_n):
    impls = [i for i in GPU_IMPLS if i != "cublas_sgemm"]
    theo = None
    if detail is not None:
        theo = detail.groupby("implementation")["theoretical_occupancy"].first() * 100
    if ncu is None and theo is None:
        note("no ncu metrics or detail CSV - skipping bandwidth/occupancy plot")
        return
    show = impls + (["cublas_sgemm"] if ncu is not None and "cublas_sgemm" in ncu.index else [])
    x = np.arange(len(show))
    labels = [LABEL[i].replace(" ", "\n", 1) for i in show]
    colors = [COLOR[i] for i in show]

    panels = 3 if ncu is not None else 1
    fig, axes = plt.subplots(1, panels, figsize=(5.2 * panels, 4.4), squeeze=False)
    axes = axes[0]

    ax = axes[0]
    if ncu is not None:
        ach = [ncu["occ_achieved"].get(i, np.nan) for i in show]
        ax.bar(x, ach, color=colors, width=0.62, edgecolor=SURFACE, linewidth=2, label="achieved (ncu)")
    th = [ncu["occ_theoretical"].get(i, np.nan) if ncu is not None and not np.isnan(ncu["occ_theoretical"].get(i, np.nan))
          else (theo.get(i, np.nan) if theo is not None else np.nan) for i in show]
    ax.scatter(x, th, marker="_", s=600, color=INK, linewidths=2, zorder=3, label="theoretical")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Occupancy (% of max warps per SM)")
    ax.set_title("Occupancy" + ("" if ncu is not None else " (theoretical only - run ncu)"))
    ax.legend(loc="upper right")

    if ncu is not None:
        ax = axes[1]
        ax.bar(x, [ncu["dram_gbs"].get(i, np.nan) for i in show], color=colors, width=0.62,
               edgecolor=SURFACE, linewidth=2)
        ax.set_ylabel("Achieved DRAM bandwidth (GB/s)")
        ax.set_title("Achieved DRAM bandwidth")
        ax = axes[2]
        ax.bar(x, [ncu["smem_efficiency"].get(i, np.nan) for i in show], color=colors, width=0.62,
               edgecolor=SURFACE, linewidth=2)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Shared-memory efficiency (%)\n= 1 − bank conflicts / wavefronts")
        ax.set_title("Shared-memory efficiency")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"Profiling vs tile size / coarsening (N = {profile_n}, Nsight Compute)",
                 color=INK, fontweight="bold", y=1.02)
    save(fig, "fig05_bandwidth_occupancy_vs_tile.png")

    if ncu is not None:
        tab = ncu[["occ_theoretical", "occ_achieved", "dram_gbs", "smem_efficiency", "sm_throughput_pct"]]
        summary.append(f"## Nsight Compute metrics (N = {profile_n})\n")
        summary.append(md(tab.round(1)) + "\n")


# ---------------------------------------------------------------------------
# Waterfall, transfer-vs-kernel, roofline
# ---------------------------------------------------------------------------

def plot_waterfall(p4, n_req):
    t = agg(p4, "kernel_ms")
    have = lambda impl: set(t[t.implementation == impl]["size"])  # noqa: E731
    common = have(BASELINE) & set.intersection(*[have(i) for i in GPU_IMPLS if have(i)] or [set()])
    if not common:
        note("waterfall needs the CPU baseline and GPU runs at a common size - skipped")
        return
    n = n_req if n_req in common else max(common)
    tn = t[t["size"] == n].set_index("implementation")
    steps = [BASELINE] + [i for i in GPU_IMPLS if i in tn.index]
    sp = [tn.loc[BASELINE, "mean"] / tn.loc[i, "mean"] for i in steps]
    sp_std = [0.0] + [ratio_std(tn.loc[BASELINE, "mean"], tn.loc[BASELINE, "std"], tn.loc[i, "mean"],
                                tn.loc[i, "std"]) for i in steps[1:]]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    prev = 0.0
    for k, impl in enumerate(steps):
        lo, hi = (0, sp[k]) if k == 0 else (min(prev, sp[k]), max(prev, sp[k]))
        if k == 0:
            color = INK2
        else:
            color = COLOR[impl] if sp[k] >= prev else "#d03b3b"
        ax.bar(k, hi - lo, bottom=lo, color=color, width=0.62, edgecolor=SURFACE, linewidth=2)
        ax.errorbar(k, sp[k], yerr=sp_std[k], fmt="none", color=INK, **ERR)
        if k:
            ax.plot([k - 1 + 0.31, k - 0.31], [prev, prev], color=MUTED, lw=1, ls=":")
            step = sp[k] / prev
            ax.annotate(f"{sp[k]:,.0f}×\n(step ×{step:.2f})", (k, hi), textcoords="offset points",
                        xytext=(12, 4), ha="left", fontsize=8, color=INK2)
        prev = sp[k]
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([LABEL[s].replace(" ", "\n", 1) for s in steps], fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Cumulative speedup over CPU seq")
    ax.set_title(f"Waterfall: speedup contributed by each optimisation (N = {n}, kernel time)\n"
                 "red = this step is slower than the previous one")
    ax.set_ylim(0, max(sp) * 1.18)
    save(fig, "fig06_waterfall_speedup.png")


def plot_transfer_vs_kernel(detail):
    if detail is None:
        note("no detail CSV - skipping transfer-vs-kernel plot")
        return
    d = detail[detail.run_id > 0]
    impls = [i for i in ["gpu_tiled_t16", "gpu_coarse4x4_t16", "cublas_sgemm"] if i in set(d.implementation)]
    if not impls:
        return
    fig, axes = plt.subplots(2, len(impls), figsize=(5.2 * len(impls), 8.4), squeeze=False)
    parts = [("h2d_ms", "Host→Device", SERIES[0]), ("kernel_ms", "Kernel", SERIES[1]),
             ("d2h_ms", "Device→Host", SERIES[2])]
    for c, impl in enumerate(impls):
        di = d[d.implementation == impl]
        g = di.groupby("size")[[p[0] for p in parts] + ["total_ms"]].agg(["mean", "std"]).sort_index()
        sizes = g.index.values
        ax = axes[0][c]
        for col, lab, colr in parts:
            ax.errorbar(sizes, g[(col, "mean")], yerr=g[(col, "std")].fillna(0), fmt="o-", color=colr,
                        label=lab, **ERR)
        ax.set_yscale("log")
        log2_sizes(ax, sizes)
        ax.set_xlabel("Matrix size N")
        ax.set_ylabel("Time (ms, log scale)")
        ax.set_title(f"{LABEL[impl]}: transfer vs kernel")
        ax.legend(loc="upper left")

        ax = axes[1][c]
        xs = np.arange(len(sizes))
        bottom = np.zeros(len(sizes))
        tot = g[("total_ms", "mean")].values
        for col, lab, colr in parts:
            frac = 100 * g[(col, "mean")].values / tot
            ax.bar(xs, frac, bottom=bottom, color=colr, width=0.68, edgecolor=SURFACE, linewidth=2, label=lab)
            bottom += frac
        for x, (m, s) in zip(xs, zip(tot, g[("total_ms", "std")].fillna(0).values)):
            ax.annotate(f"{m:.3g} ms\n±{s:.2g}", (x, 100), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7, color=INK2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(s)) for s in sizes])
        ax.set_ylim(0, 118)
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("Matrix size N")
        ax.set_ylabel("Share of end-to-end time (%)")
        ax.set_title("Breakdown (total mean ± std above bars)")
    axes[1][-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    host = d["host_memory"].iloc[0] if "host_memory" in d else "pinned"
    fig.suptitle(f"Transfer vs kernel time ({host} host memory, cudaEvent timing)", color=INK,
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "fig07_transfer_vs_kernel.png")


def plot_roofline_gpu(p4, gpu_machine, ncu, profile_n):
    if gpu_machine is None:
        note("no gpu_machine.csv - skipping GPU roofline")
        return
    m = gpu_machine.iloc[0]
    peak, bw, theo_bw = float(m["peak_fp32_gflops"]), float(m["triad_best_gbs"]), float(m["theoretical_bw_gbs"])
    t = agg(p4[p4.implementation.isin(GPU_IMPLS)], "throughput")
    n = profile_n if profile_n in set(t["size"]) else t["size"].max()
    tn = t[t["size"] == n].set_index("implementation")

    ai = np.logspace(-2, 3, 300)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(ai, np.minimum(peak, ai * bw), color=INK, lw=2, label=f"Roof: STREAM triad {bw:.0f} GB/s, peak {peak:,.0f} GFLOP/s")
    ax.plot(ai, np.minimum(peak, ai * theo_bw), color=MUTED, lw=1, ls="--", label=f"Theoretical bandwidth {theo_bw:.0f} GB/s")
    measured = ncu is not None
    for impl in GPU_IMPLS:
        if impl not in tn.index:
            continue
        if measured and impl in ncu.index and ncu.loc[impl, "dram_bytes"] > 0:
            x = 2.0 * n ** 3 / ncu.loc[impl, "dram_bytes"]
        elif impl in MODEL_AI:
            x = MODEL_AI[impl]
        else:
            continue
        ax.errorbar(x, tn.loc[impl, "mean"], yerr=tn.loc[impl, "std"], fmt="o", color=COLOR[impl],
                    markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABEL[impl], **ERR)
    ax.axvline(peak / bw, color=GRID, lw=1)
    ax.annotate(f"ridge {peak / bw:.1f} FLOP/B", (peak / bw, peak * 1.05), fontsize=8, color=MUTED, ha="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP / DRAM byte, log)")
    ax.set_ylabel("GFLOP/s (log)")
    src = "measured DRAM bytes from Nsight Compute" if measured else "DRAM-traffic model (no ncu data)"
    ax.set_title(f"GPU Roofline ({m['device']}, N = {n})\nintensity: {src}")
    ax.legend(loc="lower right", fontsize=7.5)
    save(fig, "fig08_roofline_gpu.png")


def plot_roofline_cpu(p4, cpu_peak, stream):
    if cpu_peak is None or stream is None:
        note("no cpu_peak.csv / stream_cpu.csv - skipping CPU roofline")
        return
    peak1 = float(cpu_peak["peak_gflops_1core"].iloc[0])
    one = stream[(stream.kernel == "Triad") & (stream.threads == 1)]
    if one.empty:
        return
    bw = float(one["best_gbs"].iloc[0])
    t = agg(p4[p4.implementation.isin([BASELINE, "cpu_seq_naive"])], "throughput")
    ai = np.logspace(-2, 2.5, 200)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(ai, np.minimum(peak1, ai * bw), color=INK, lw=2,
            label=f"1 core: triad {bw:.1f} GB/s, peak {peak1:.0f} GFLOP/s")
    # Model DRAM intensity: blocked (I=32,K=128) ~ 2/(4*(1/32+2/128)) FLOP/B; naive ijk ~ 0.25.
    for impl, x in [(BASELINE, 2 / (4 * (1 / 32 + 2 / 128))), ("cpu_seq_naive", 0.25)]:
        d = t[t.implementation == impl]
        if len(d):
            best = d.loc[d["mean"].idxmax()]
            ax.errorbar(x, best["mean"], yerr=best["std"], fmt="o", color=COLOR[impl], markersize=8,
                        label=f"{LABEL[impl]} (N={int(best['size'])})", **ERR)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP / byte, cache-blocking model)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("CPU Roofline (single core, sequential baseline)")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "fig09_roofline_cpu.png")


# ---------------------------------------------------------------------------
# Cluster comparison (P1-P4) and crossover
# ---------------------------------------------------------------------------

CPU_SIDE = ["CPU sequential", "CPU parallel", "MPI", "CPU library"]


def crossover(sizes_a, ta, sizes_b, tb):
    """Smallest N from which curve b (GPU) stays below curve a (CPU), log-log interpolation."""
    lo, hi = max(min(sizes_a), min(sizes_b)), min(max(sizes_a), max(sizes_b))
    grid = sorted(s for s in set(sizes_a) | set(sizes_b) if lo <= s <= hi)
    if len(grid) < 1:
        return None, "no overlapping sizes"
    la = np.interp(np.log(grid), np.log(sizes_a), np.log(ta))
    lb = np.interp(np.log(grid), np.log(sizes_b), np.log(tb))
    diff = lb - la  # < 0 means GPU faster
    if np.all(diff < 0):
        return grid[0], f"GPU already faster at the smallest common size ({grid[0]})"
    if diff[-1] >= 0:
        return None, f"GPU never faster up to N = {grid[-1]}"
    k = max(i for i in range(len(diff)) if diff[i] >= 0)  # last size where CPU still wins
    x0, x1 = np.log(grid[k]), np.log(grid[k + 1])
    xc = x0 + (x1 - x0) * diff[k] / (diff[k] - diff[k + 1])
    return float(np.exp(xc)), f"between N = {grid[k]} and N = {grid[k + 1]}"


def plot_final(merged):
    if merged is None:
        note("no merged CSV - run scripts/merge_results.py first")
        return
    m = merged.copy()
    m["series"] = m["problem"] + " · " + m["implementation"]
    g = m.groupby(["category", "series", "size"]).agg(total=("total_ms", "mean"), total_std=("total_ms", "std"),
                                                      kernel=("kernel_ms", "mean"),
                                                      kernel_std=("kernel_ms", "std")).reset_index().fillna(0)
    # Best implementation of each category at each size.
    best = g.loc[g.groupby(["category", "size"])["total"].idxmin()]

    cats = [c for c in CPU_SIDE + ["GPU kernel", "GPU library"] if c in set(best.category)]
    cat_color = {c: SERIES[i] for i, c in enumerate(cats)}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for c in cats:
        d = best[best.category == c].sort_values("size")
        ax.errorbar(d["size"], d["total"], yerr=d["total_std"], fmt="o-", color=cat_color[c],
                    label=f"{c} (incl. transfer)" if c.startswith("GPU") else c, **ERR)
        if c.startswith("GPU"):
            gk = g[g.category == c]
            bk = gk.loc[gk.groupby("size")["kernel"].idxmin()].sort_values("size")
            ax.errorbar(bk["size"], bk["kernel"], yerr=bk["kernel_std"], fmt="o--", color=cat_color[c],
                        alpha=0.55, label=f"{c} (kernel only)", **ERR)

    cpu = best[best.category.isin(CPU_SIDE)].groupby("size")["total"].min()
    lines, boxed = [], []
    for c in [c for c in cats if c.startswith("GPU")]:
        gpu = best[best.category == c].set_index("size")["total"]
        if len(cpu) and len(gpu):
            xc, why = crossover(cpu.index.values, cpu.values, gpu.index.values, gpu.values)
            lines.append(f"- **{c}** (with transfer) vs best CPU/MPI: "
                         + (f"crossover at N ≈ {xc:,.0f} ({why})" if xc else why))
            if xc:
                ax.axvline(xc, color=cat_color[c], lw=1, ls=":")
            boxed.append(f"{c}: " + (f"GPU wins from N≈{xc:,.0f}" if xc else why))
    if boxed:
        ax.text(0.98, 0.03, "With transfer, vs best CPU/MPI\n" + "\n".join(boxed), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5, color=INK2,
                bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=AXIS))
    log2_sizes(ax, best["size"].unique())
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size N")
    ax.set_ylabel("End-to-end time (ms, log)")
    ax.set_title("CPU – MPI – GPU – library: best of each category")
    ax.legend(fontsize=7.5, loc="upper left")

    ax = axes[1]
    n_cmp = max(set.intersection(*[set(best[best.category == c]["size"]) for c in cats]) or {best["size"].max()})
    bn = best[best["size"] == n_cmp].set_index("category").reindex(cats).dropna(subset=["total"])
    gf = 2 * float(n_cmp) ** 3 / (bn["total"] * 1e-3) / 1e9
    gf_std = gf * bn["total_std"] / bn["total"]
    ax.bar(range(len(bn)), gf, yerr=gf_std, color=[cat_color[c] for c in bn.index], width=0.62,
           edgecolor=SURFACE, linewidth=2, error_kw=ERR)
    for k, (c, row) in enumerate(bn.iterrows()):
        ax.annotate(row["series"], (k, gf.iloc[k]), textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=7, color=INK2, rotation=0)
    ax.set_yscale("log")
    ax.set_xticks(range(len(bn)))
    ax.set_xticklabels([c.replace(" ", "\n") for c in bn.index], fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("GFLOP/s incl. transfer (log)")
    ax.set_title(f"Best per category at N = {n_cmp}")
    fig.tight_layout()
    save(fig, "fig10_final_comparison.png")

    summary.append("## Cluster comparison (best implementation per category, end-to-end time)\n")
    piv = best.pivot(index="size", columns="category", values="total")[cats]
    summary.append(md(piv.round(3)) + "\n")
    summary.append("### Where the GPU wins when transfer time is counted\n")
    summary.extend(lines or ["- not enough overlapping data"])
    summary.append("")


# ---------------------------------------------------------------------------

def p4_tables(p4, detail, gpu_machine, cpu_peak, stream):
    summary.append("# P4 results summary (generated by scripts/plot_results.py)\n")
    summary.append("Mean ± standard deviation over the recorded runs (warm-up excluded).\n")
    if gpu_machine is not None:
        m = gpu_machine.iloc[0]
        summary.append(f"- GPU: **{m['device']}** (cc {m['compute_capability']}, {m['sms']} SMs), "
                       f"peak FP32 ≈ {m['peak_fp32_gflops']:,.0f} GFLOP/s, theoretical bandwidth "
                       f"{m['theoretical_bw_gbs']:.0f} GB/s, STREAM triad {m['triad_best_gbs']:.0f} GB/s "
                       f"(copy {m['copy_best_gbs']:.0f} GB/s)")
    if cpu_peak is not None:
        c = cpu_peak.iloc[0]
        summary.append(f"- CPU: **{c['cpu_model']}**, {c['cores']} cores / {c['threads']} threads, "
                       f"peak FP32 ≈ {c['peak_gflops_all']} GFLOP/s ({c['peak_gflops_1core']} per core, {c['simd']})")
    if stream is not None:
        for th, d in stream[stream.kernel == "Triad"].groupby("threads"):
            summary.append(f"- CPU STREAM triad, {th} thread(s): best {d['best_gbs'].iloc[-1]:.1f} GB/s "
                           f"(mean {d['mean_gbs'].iloc[-1]:.1f} ± {d['std_gbs'].iloc[-1]:.1f})")
    summary.append("")
    t = agg(p4, "kernel_ms")
    t["cell"] = t.apply(lambda r: f"{r['mean']:.4g} ± {r['std']:.2g}", axis=1)
    tab = t.pivot(index="implementation", columns="size", values="cell")
    order = [i for i in ["cpu_seq_naive", BASELINE] + GPU_IMPLS if i in tab.index]
    summary.append("## Kernel / compute time (ms)\n")
    summary.append(md(tab.reindex(order).fillna("–")) + "\n")
    gf = agg(p4, "throughput").pivot(index="implementation", columns="size", values="mean")
    summary.append("## Throughput (GFLOP/s, kernel time)\n")
    summary.append(md(gf.reindex(order).round(1)) + "\n")
    err = p4.groupby("implementation")["error"].max().reindex(order)
    summary.append("## Verification (max sampled relative error vs FP64 reference, all sizes)\n")
    summary.append(md(err.map(lambda e: f"{e:.2e}")) + "\n")
    if detail is not None:
        occ = detail.groupby("implementation")[["block_threads", "regs_per_thread", "static_smem_bytes",
                                                 "blocks_per_sm", "theoretical_occupancy"]].first()
        summary.append("## Launch configuration (from cudaFuncGetAttributes / occupancy API)\n")
        summary.append(md(occ.reindex([i for i in GPU_IMPLS if i in occ.index])) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile-n", type=int, default=2048, help="size used for the ncu profile")
    ap.add_argument("--waterfall-n", type=int, default=2048, help="size used for the waterfall chart")
    args = ap.parse_args()

    frames = [f for f in (read(f"{RAW}/p4_gemm_cpu.csv"), read(f"{RAW}/p4_gemm_gpu.csv")) if f is not None]
    if not frames:
        raise SystemExit("No P4 result CSVs in results/raw - run the benchmarks first.")
    p4 = pd.concat(frames, ignore_index=True)
    p4 = p4[p4.run_id > 0]
    detail = read(f"{RAW}/p4_gemm_gpu_detail.csv")
    gpu_machine = read(f"{RAW}/gpu_machine.csv")
    cpu_peak = read(f"{RAW}/cpu_peak.csv")
    stream = read(f"{RAW}/stream_cpu.csv")
    ncu = load_ncu(f"{RAW}/ncu_metrics.csv")
    merged = read("results/merged/cluster_results.csv")

    p4_tables(p4, detail, gpu_machine, cpu_peak, stream)
    plot_time_and_gflops(p4)
    plot_speedup(p4)
    plot_efficiency(p4, gpu_machine)
    plot_tile_tuning(p4)
    plot_bw_occupancy(ncu, detail, args.profile_n)
    plot_waterfall(p4, args.waterfall_n)
    plot_transfer_vs_kernel(detail)
    plot_roofline_gpu(p4, gpu_machine, ncu, args.profile_n)
    plot_roofline_cpu(p4, cpu_peak, stream)
    plot_final(merged)

    with open("results/summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")
    note("wrote results/summary.md")


if __name__ == "__main__":
    main()
