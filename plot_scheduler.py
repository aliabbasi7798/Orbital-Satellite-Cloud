
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ALGORITHMS = ["step1" , "step2", "khop1", "khop2", "khop3", "kxk2", "kxk4", "kxk8"]
RATES = [100, 500, 1000, 2000]
MIXES = ["m8b", "mix"]

# Consistent color per algorithm
COLORS = {
    "step2": "#d62728",   # red — baseline
    "khop1": "#1f77b4",   # blue
    "khop2": "#4a90d9",
    "khop3": "#76abe0",
    "kxk2":  "#2ca02c",   # green family
    "kxk4":  "#4fb94f",
    "kxk8":  "#7fcc7f",
}
MARKERS = {
    "step2": "o", "khop1": "s", "khop2": "^", "khop3": "v",
    "kxk2":  "D", "kxk4":  "P", "kxk8":  "X",
}
# Human-readable labels
PRETTY = {
    "step2": "step2 (baseline)",
    "khop1": "K-hop, K=1",
    "khop2": "K-hop, K=2",
    "khop3": "K-hop, K=3",
    "kxk2":  "K×K, K=2",
    "kxk4":  "K×K, K=4",
    "kxk8":  "K×K, K=8",
}


def scenario_name(algo: str, mix: str, rate: int) -> str:
    return f"GS20_{algo}_{mix}_r{rate}"


def load_scenario(results_dir: str, algo: str, mix: str, rate: int,
                  warmup_s: float) -> pd.DataFrame | None:
    path = os.path.join(results_dir, f"{scenario_name(algo,mix,rate)}_requests.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df[df["arrival_s"] >= warmup_s].reset_index(drop=True)
    return df


# ============================================================
# BATCH 1 — queue time per request rate (algorithms compared)
# ============================================================
def plot_queue_by_rate(results_dir: str, warmup_s: float, out_dir: str):
    """One plot per (rate, mix): grouped bars of median/p95/p99 queue delay
    for all 7 algorithms."""
    for mix in MIXES:
        for rate in RATES:
            med, p95, p99, labs = [], [], [], []
            for algo in ALGORITHMS:
                df = load_scenario(results_dir, algo, mix, rate, warmup_s)
                if df is None or len(df) == 0:
                    med.append(0); p95.append(0); p99.append(0)
                    labs.append(PRETTY[algo]); continue
                done = df[~df["dropped"].astype(bool)]
                q = done["queue_s"].to_numpy() * 1000
                if len(q) == 0:
                    med.append(0); p95.append(0); p99.append(0)
                else:
                    med.append(float(np.median(q)))
                    p95.append(float(np.percentile(q, 95)))
                    p99.append(float(np.percentile(q, 99)))
                labs.append(PRETTY[algo])

            fig, ax = plt.subplots(figsize=(11, 5))
            x = np.arange(len(labs))
            w = 0.27
            ax.bar(x - w, med, w, label="median", color="#4C72B0")
            ax.bar(x,     p95, w, label="p95",    color="#DD8452")
            ax.bar(x + w, p99, w, label="p99",    color="#55A868")
            ax.set_xticks(x)
            ax.set_xticklabels(labs, rotation=20, ha="right", fontsize=9)
            ax.set_ylabel("Queue delay (ms)")
            ax.set_title(f"Queue delay @ {rate} req/s, {mix}")
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend()
            # Annotate p99 with exact values (the interesting number)
            ymax = max(p99) if any(p99) else 1
            for i, v in enumerate(p99):
                ax.text(i + w, v + ymax * 0.015, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=7)
            fig.tight_layout()
            out = os.path.join(out_dir, f"gs20_queue_r{rate}_{mix}.png")
            fig.savefig(out, dpi=130)
            plt.close(fig)
            print(f"  wrote {out}")


# ============================================================
# BATCH 2 — per-satellite load distribution
# ============================================================
def plot_sat_load(results_dir: str, warmup_s: float, out_dir: str,
                  rate: int = 2000, mix: str = "m8b"):
    """One plot per algorithm. Shows: number of requests computed at each sat,
    sats ranked from most-loaded to least. Y-axis and x-range are normalized
    across all plots so they can be visually compared.

    Only plots for satellite-compute modes (step2, khop*, kxk*) — all of
    these use compute_node = sat id.
    """
    # First pass: compute max y-axis across all algos so plots are comparable
    loads_per_algo = {}
    max_y = 0
    N_sats = 0
    for algo in ALGORITHMS:
        df = load_scenario(results_dir, algo, mix, rate, warmup_s)
        if df is None: continue
        done = df[(~df["dropped"].astype(bool)) & (df["compute_node"] >= 0)]
        if len(done) == 0: continue
        counts = done.groupby("compute_node").size()
        # Pad missing sats with 0
        N = int(df["compute_node"].max()) + 1
        full = np.zeros(N, dtype=np.int64)
        full[counts.index.values] = counts.values
        # Sort descending
        sorted_counts = np.sort(full)[::-1]
        loads_per_algo[algo] = sorted_counts
        max_y = max(max_y, sorted_counts[0])
        N_sats = max(N_sats, N)

    # Second pass: draw each plot with shared y-axis
    for algo, counts in loads_per_algo.items():
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(len(counts))
        ax.fill_between(x, counts, color=COLORS[algo], alpha=0.7)
        ax.plot(x, counts, color=COLORS[algo], lw=1)
        ax.set_xlim(0, N_sats)
        ax.set_ylim(0, max_y * 1.05)
        ax.set_xlabel("Satellite rank (most-loaded first)")
        ax.set_ylabel("Requests computed")
        # Annotate a few key stats
        active = np.sum(counts > 0)
        p95_load = np.percentile(counts[counts > 0], 95) if active > 0 else 0
        p50_load = np.percentile(counts[counts > 0], 50) if active > 0 else 0
        ax.set_title(f"{PRETTY[algo]} @ {rate} req/s, {mix}  —  "
                     f"{active} active sats, "
                     f"top load={counts[0]}, median={p50_load:.0f}, p95={p95_load:.0f}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = os.path.join(out_dir, f"load_r{rate}_{algo}_{mix}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"  wrote {out}")


def plot_sat_load_overlay(results_dir: str, warmup_s: float, out_dir: str,
                          rate: int = 2000, mix: str = "m8b"):
    """One overlay plot: all 7 algorithms' sat-load curves on one axis.
    Good single comparison view (the 'how evenly is load spread' plot)."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for algo in ALGORITHMS:
        df = load_scenario(results_dir, algo, mix, rate, warmup_s)
        if df is None: continue
        done = df[(~df["dropped"].astype(bool)) & (df["compute_node"] >= 0)]
        if len(done) == 0: continue
        counts = done.groupby("compute_node").size()
        N = int(df["compute_node"].max()) + 1
        full = np.zeros(N, dtype=np.int64)
        full[counts.index.values] = counts.values
        sorted_counts = np.sort(full)[::-1]
        x = np.arange(len(sorted_counts))
        ax.plot(x, sorted_counts, label=PRETTY[algo],
                color=COLORS[algo], linewidth=1.8)
    ax.set_xlabel("Satellite rank (most-loaded first)")
    ax.set_ylabel("Requests computed")
    ax.set_yscale("log")
    ax.set_title(f"Per-satellite load — all algorithms @ {rate} req/s, {mix}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(out_dir, f"load_r{rate}_overlay_{mix}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")


# ============================================================
# BATCH 3 — sweep plot (rate on x-axis, line per algorithm)
# ============================================================
def plot_sweep(results_dir: str, warmup_s: float, out_dir: str):
    """Produce headline sweep plots: x=req/s, y=latency, line per algorithm."""
    for mix in MIXES:
        # Collect data
        data = {algo: {"rate": [], "med_q": [], "p95_q": [], "p99_q": [],
                       "med_t": [], "p99_t": [], "drop": []}
                for algo in ALGORITHMS}
        for algo in ALGORITHMS:
            for rate in RATES:
                df = load_scenario(results_dir, algo, mix, rate, warmup_s)
                if df is None: continue
                total = len(df)
                done = df[~df["dropped"].astype(bool)]
                n = len(done)
                if n == 0: continue
                q = done["queue_s"].to_numpy() * 1000
                t = done["total_s"].to_numpy() * 1000
                data[algo]["rate"].append(rate)
                data[algo]["med_q"].append(float(np.median(q)))
                data[algo]["p95_q"].append(float(np.percentile(q, 95)))
                data[algo]["p99_q"].append(float(np.percentile(q, 99)))
                data[algo]["med_t"].append(float(np.median(t)))
                data[algo]["p99_t"].append(float(np.percentile(t, 99)))
                data[algo]["drop"].append((total - n) / total if total else 0.0)

        # Plot A: p99 queue
        _sweep_line_plot(data, "p99_q",
                         f"p99 queue delay vs request rate  —  {mix}",
                         "Queue delay p99 (ms)",
                         os.path.join(out_dir, f"sweep_p99_queue_{mix}.png"))
        # Plot B: median queue
        _sweep_line_plot(data, "med_q",
                         f"Median queue delay vs request rate  —  {mix}",
                         "Queue delay median (ms)",
                         os.path.join(out_dir, f"sweep_median_queue_{mix}.png"),
                         log_y=True)
        # Plot C: p99 total latency
        _sweep_line_plot(data, "p99_t",
                         f"p99 end-to-end latency vs request rate  —  {mix}",
                         "Total latency p99 (ms)",
                         os.path.join(out_dir, f"sweep_p99_total_{mix}.png"))
        # Plot D: drop rate
        _sweep_line_plot(data, "drop",
                         f"Drop rate vs request rate  —  {mix}",
                         "Drop rate",
                         os.path.join(out_dir, f"sweep_drop_{mix}.png"),
                         log_y=False)


def _sweep_line_plot(data, field, title, ylabel, out_path, log_y=True):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for algo, d in data.items():
        if not d["rate"]: continue
        ax.plot(d["rate"], d[field],
                marker=MARKERS[algo], color=COLORS[algo],
                label=PRETTY[algo], lw=1.8, markersize=7)
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ============================================================
# main
# ============================================================
def main():
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    warmup = cfg["simulation"]["warmup_s"]
    results_dir = cfg["output"]["results_dir"]
    out_dir = os.path.join(cfg["output"]["plots_dir"], "scheduler")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading from {results_dir}, warmup={warmup}s")
    print(f"Output: {out_dir}\n")

    print("BATCH 1: queue time by rate…")
    plot_queue_by_rate(results_dir, warmup, out_dir)

    print("\nBATCH 2: per-sat load distribution (rate=2000, m8b)…")
    plot_sat_load(results_dir, warmup, out_dir, rate=2000, mix="m8b")
    plot_sat_load_overlay(results_dir, warmup, out_dir, rate=2000, mix="m8b")
    print("\n         per-sat load overlay (rate=2000, mix)…")
    plot_sat_load_overlay(results_dir, warmup, out_dir, rate=2000, mix="mix")

    print("\nBATCH 3: sweep plots…")
    plot_sweep(results_dir, warmup, out_dir)

    print(f"\nAll plots in: {out_dir}/")


if __name__ == "__main__":
    main()