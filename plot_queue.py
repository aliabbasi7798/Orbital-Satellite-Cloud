
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCENARIOS = [
    "GS20_step1_low_m1b",
    "GS20_step1_low_m8b",
    "GS20_step1_low_mix",
    "GS20_step2_sat50_m1b",
    "GS20_step2_sat50_m8b",
    "GS20_step2_sat50_mix",
    "GS20_hybrid_queue_m1b",
    "GS20_hybrid_queue_m8b",
    "GS20_hybrid_queue_mix",
]

SHORT = {
    "GS20_step1_low_m1b":     "step1·low / 1B",
    "GS20_step1_low_m8b":     "step1·low / 8B",
    "GS20_step1_low_mix":     "step1·low / mix",
    "GS20_step2_sat50_m1b":   "step2·sat / 1B",
    "GS20_step2_sat50_m8b":   "step2·sat / 8B",
    "GS20_step2_sat50_mix":   "step2·sat / mix",
    "GS20_hybrid_queue_m1b":  "hybrid·q / 1B",
    "GS20_hybrid_queue_m8b":  "hybrid·q / 8B",
    "GS20_hybrid_queue_mix":  "hybrid·q / mix",
}

def color_for(lab):
    if "step1_low" in lab:    return "#1f77b4"
    if "step2_sat50" in lab:  return "#d62728"
    if "hybrid_queue" in lab: return "#2ca02c"
    return "#777"


def load_all(results_dir, warmup_s):
    out = {}
    for lab in SCENARIOS:
        p = os.path.join(results_dir, f"{lab}_requests.csv")
        if not os.path.exists(p):
            print(f"  MISSING: {p}")
            continue
        df = pd.read_csv(p)
        df = df[df["arrival_s"] >= warmup_s].reset_index(drop=True)
        out[lab] = df
    return out


def q_stats(q_ms):
    if len(q_ms) == 0:
        return dict(n=0, mean_ms=0.0, median_ms=0.0, p95_ms=0.0, p99_ms=0.0, max_ms=0.0)
    return dict(
        n=len(q_ms),
        mean_ms=float(np.mean(q_ms)),
        median_ms=float(np.median(q_ms)),
        p95_ms=float(np.percentile(q_ms, 95)),
        p99_ms=float(np.percentile(q_ms, 99)),
        max_ms=float(np.max(q_ms)),
    )


def build_table(dfs):
    rows = []
    for lab, df in dfs.items():
        done = df[~df["dropped"].astype(bool)]
        q = done["queue_s"].to_numpy() * 1000
        row = {"scenario": SHORT.get(lab, lab)}
        row.update(q_stats(q))
        # Per-model breakdown for mix scenarios
        if lab.endswith("_mix") and "model_id" in done.columns:
            for mid in ["llama3_1b", "llama3_8b"]:
                sub = done[done["model_id"] == mid]
                qs = sub["queue_s"].to_numpy() * 1000
                s = q_stats(qs)
                row[f"{mid}_median_ms"] = s["median_ms"]
                row[f"{mid}_p99_ms"]    = s["p99_ms"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_queue_bars(dfs, out_path, table):
    fig, ax = plt.subplots(figsize=(12, 5))
    labs = [SHORT[l] for l in dfs]
    med, p95, p99 = [], [], []
    for df in dfs.values():
        q = df[~df["dropped"].astype(bool)]["queue_s"].to_numpy() * 1000
        if len(q) == 0:
            med.append(0); p95.append(0); p99.append(0); continue
        med.append(np.median(q))
        p95.append(np.percentile(q, 95))
        p99.append(np.percentile(q, 99))
    x = np.arange(len(labs))
    w = 0.27
    ax.bar(x - w, med, w, label="median", color="#4C72B0")
    ax.bar(x,     p95, w, label="p95",    color="#DD8452")
    ax.bar(x + w, p99, w, label="p99",    color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=25, ha="right")
    ax.set_ylabel("Queue delay (ms)")
    ax.set_title("Queue-wait time — GS20")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    # Value annotations on the p99 bars
    for i, v in enumerate(p99):
        ax.text(i + w, v + max(p99) * 0.02, f"{v:.1f}",
                ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_queue_cdf(dfs, out_path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    LS = {"_m1b": "-", "_m8b": "--", "_mix": ":"}
    for lab, df in dfs.items():
        q = df[~df["dropped"].astype(bool)]["queue_s"].to_numpy() * 1000
        if len(q) == 0: continue
        q = np.sort(q)
        y = np.arange(1, len(q) + 1) / len(q)
        # Replace exact-zero values with tiny epsilon for log-x plotting
        q_plot = np.where(q < 1e-3, 1e-3, q)
        ls = next((s for suf, s in LS.items() if lab.endswith(suf)), "-")
        ax.plot(q_plot, y, lw=1.8, color=color_for(lab), linestyle=ls,
                label=SHORT[lab])
    ax.set_xscale("log")
    ax.set_xlabel("Queue delay (ms, log)")
    ax.set_ylabel("CDF")
    ax.set_title("Queue-delay CDF — GS20")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_mix_by_model(dfs, out_path):
    """For *_mix scenarios only: queue delay broken out by 1B vs 8B."""
    mix_scenarios = [l for l in dfs if l.endswith("_mix")]
    if not mix_scenarios:
        return
    fig, axes = plt.subplots(1, len(mix_scenarios),
                             figsize=(4.5 * len(mix_scenarios), 5),
                             sharey=True)
    if len(mix_scenarios) == 1:
        axes = [axes]
    for ax, lab in zip(axes, mix_scenarios):
        done = dfs[lab][~dfs[lab]["dropped"].astype(bool)]
        if "model_id" not in done.columns: continue
        medians, p95s, p99s, names = [], [], [], []
        for mid in ["llama3_1b", "llama3_8b"]:
            sub = done[done["model_id"] == mid]
            q = sub["queue_s"].to_numpy() * 1000
            if len(q) == 0: continue
            medians.append(np.median(q))
            p95s.append(np.percentile(q, 95))
            p99s.append(np.percentile(q, 99))
            names.append(mid.replace("llama3_", ""))
        x = np.arange(len(names))
        w = 0.27
        ax.bar(x - w, medians, w, label="median", color="#4C72B0")
        ax.bar(x,     p95s,    w, label="p95",    color="#DD8452")
        ax.bar(x + w, p99s,    w, label="p99",    color="#55A868")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_title(SHORT[lab])
        ax.grid(True, axis="y", alpha=0.3)
        # annotate p99
        for i, v in enumerate(p99s):
            ax.text(i + w, v + max(p99s) * 0.02, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7)
    axes[0].set_ylabel("Queue delay (ms)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Queue delay split by model — mix scenarios only", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    warmup = cfg["simulation"]["warmup_s"]
    results_dir = cfg["output"]["results_dir"]
    out_dir = os.path.join(cfg["output"]["plots_dir"], "paper")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading from {results_dir} (warmup={warmup}s)…")
    dfs = load_all(results_dir, warmup)
    if not dfs:
        print("No scenarios found. Run the sim first.")
        return

    table = build_table(dfs)
    tab_path = os.path.join(out_dir, "table_queue_gs20.csv")
    table.to_csv(tab_path, index=False)
    print(f"\nQueue-delay summary:")
    print(table.to_string(index=False))
    print(f"\nSaved: {tab_path}")

    plot_queue_bars(dfs, os.path.join(out_dir, "gs20_queue_bars.png"), table)
    plot_queue_cdf(dfs,  os.path.join(out_dir, "gs20_queue_cdf.png"))
    plot_mix_by_model(dfs, os.path.join(out_dir, "gs20_queue_mix_split.png"))

    print(f"\nPlots in: {out_dir}/")
    for f in ["gs20_queue_bars.png", "gs20_queue_cdf.png",
              "gs20_queue_mix_split.png", "table_queue_gs20.csv"]:
        p = os.path.join(out_dir, f)
        if os.path.exists(p):
            print(f"  {f}")


if __name__ == "__main__":
    main()