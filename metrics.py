

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STAGES = ["uplink_s", "out_path_s", "queue_s", "compute_s", "back_path_s", "downlink_s"]


def summarize(df: pd.DataFrame) -> dict:
    """Return a dict of summary metrics over completed (non-dropped) requests."""
    done = df[~df["dropped"].astype(bool)]
    total = len(df)
    n = len(done)
    out = {
        "requests_total": total,
        "requests_completed": n,
        "drop_rate": 1 - n / total if total else 0.0,
    }
    if n == 0:
        return out
    total_lat = done["total_s"].to_numpy()
    out["latency_mean_s"] = float(total_lat.mean())
    out["latency_median_s"] = float(np.median(total_lat))
    out["latency_p95_s"] = float(np.percentile(total_lat, 95))
    out["latency_p99_s"] = float(np.percentile(total_lat, 99))
    out["latency_max_s"] = float(total_lat.max())
    for s in STAGES:
        out[f"{s}_mean"] = float(done[s].mean())
    t_min, t_max = done["arrival_s"].min(), done["arrival_s"].max()
    out["throughput_rps"] = n / (t_max - t_min) if t_max > t_min else 0.0
    out["hops_out_mean"] = float(done["hops_out"].mean())
    out["hops_back_mean"] = float(done["hops_back"].mean())

    # Hybrid decision stats (fields present only for hybrid_* runs)
    if "chose_gs" in done.columns:
        chose_gs = done["chose_gs"].to_numpy()
        # chose_gs is -1 for non-hybrid runs
        if (chose_gs >= 0).any():
            used = chose_gs[chose_gs >= 0]
            out["hybrid_chose_gs_frac"] = float(used.mean())

    # Per-model percentiles
    if "model_id" in done.columns:
        for mid in done["model_id"].dropna().unique():
            sub = done[done["model_id"] == mid]["total_s"].to_numpy()
            if len(sub) == 0:
                continue
            out[f"median_{mid}_s"] = float(np.median(sub))
            out[f"p95_{mid}_s"] = float(np.percentile(sub, 95))
            out[f"p99_{mid}_s"] = float(np.percentile(sub, 99))

    return out


def trim_warmup(df: pd.DataFrame, warmup_s: float) -> pd.DataFrame:
    return df[df["arrival_s"] >= warmup_s].copy()


def plot_latency_cdf(scenarios: dict, out_path: str):
    """scenarios: {label: df}"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, df in scenarios.items():
        done = df[~df["dropped"].astype(bool)]
        if len(done) == 0:
            continue
        v = np.sort(done["total_s"].to_numpy())
        y = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v * 1000, y, label=label, linewidth=1.5)
    ax.set_xlabel("End-to-end latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("End-to-end latency CDF")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_stage_breakdown(scenarios: dict, out_path: str):
    """Stacked bar of mean stage times per scenario."""
    labels = list(scenarios.keys())
    stage_means = {s: [] for s in STAGES}
    for lbl in labels:
        df = scenarios[lbl]
        done = df[~df["dropped"].astype(bool)]
        for s in STAGES:
            stage_means[s].append(done[s].mean() * 1000 if len(done) else 0)
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(labels)), 5))
    bottoms = np.zeros(len(labels))
    for s in STAGES:
        vals = np.array(stage_means[s])
        ax.bar(labels, vals, bottom=bottoms, label=s.replace("_s", ""))
        bottoms += vals
    ax.set_ylabel("Mean latency per stage (ms)")
    ax.set_title("Per-stage latency breakdown")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_percentile_bars(scenarios: dict, out_path: str):
    """Grouped bars: median, p95, p99 per scenario."""
    labels = list(scenarios.keys())
    med = []; p95 = []; p99 = []
    for lbl in labels:
        df = scenarios[lbl]
        done = df[~df["dropped"].astype(bool)]
        v = done["total_s"].to_numpy() if len(done) else np.array([0])
        med.append(np.median(v) * 1000)
        p95.append(np.percentile(v, 95) * 1000)
        p99.append(np.percentile(v, 99) * 1000)
    x = np.arange(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(labels)), 5))
    ax.bar(x - w, med, w, label="median")
    ax.bar(x, p95, w, label="p95")
    ax.bar(x + w, p99, w, label="p99")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency percentiles by scenario")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_load_distribution(df: pd.DataFrame, out_path: str, which="gs"):
    done = df[~df["dropped"].astype(bool)]
    if len(done) == 0:
        return
    col = "gs_id" if which == "gs" else "compute_node"
    counts = done[col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_xlabel(col)
    ax.set_ylabel("Requests served")
    ax.set_title(f"Load distribution across {col}")
    if len(counts) > 30:
        ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_hop_distribution(scenarios: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, df in scenarios.items():
        done = df[~df["dropped"].astype(bool)]
        if len(done) == 0:
            continue
        hops = (done["hops_out"] + done["hops_back"]).to_numpy()
        ax.hist(hops, bins=range(0, int(hops.max()) + 2),
                alpha=0.5, label=label, edgecolor="black")
    ax.set_xlabel("Total ISL hops (out + back)")
    ax.set_ylabel("Requests")
    ax.set_title("ISL hop-count distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def build_summary_table(scenarios: dict) -> pd.DataFrame:
    rows = []
    for label, df in scenarios.items():
        s = summarize(df)
        s["scenario"] = label
        rows.append(s)
    base_cols = ["scenario", "requests_total", "requests_completed", "drop_rate",
                 "latency_median_s", "latency_p95_s", "latency_p99_s",
                 "latency_mean_s", "throughput_rps",
                 "hops_out_mean", "hops_back_mean",
                 "hybrid_chose_gs_frac",
                 "median_llama3_1b_s", "p99_llama3_1b_s",
                 "median_llama3_8b_s", "p99_llama3_8b_s"]
    df = pd.DataFrame(rows)
    return df[[c for c in base_cols if c in df.columns]]
