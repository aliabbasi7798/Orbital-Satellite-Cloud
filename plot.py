
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
PLOTS_DIR = "plots_gs20"
WARMUP_S = 300          # match your config

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

# Friendlier display labels for the plots
LABEL_MAP = {
    "GS20_step1_low_m1b":       "step1 low (1B)",
    "GS20_step1_low_m8b":       "step1 low (8B)",
    "GS20_step1_low_mix":       "step1 low (mix)",
    "GS20_step2_sat50_m1b":     "step2 sat (1B)",
    "GS20_step2_sat50_m8b":     "step2 sat (8B)",
    "GS20_step2_sat50_mix":     "step2 sat (mix)",
    "GS20_hybrid_queue_m1b":    "hybrid-Q (1B)",
    "GS20_hybrid_queue_m8b":    "hybrid-Q (8B)",
    "GS20_hybrid_queue_mix":    "hybrid-Q (mix)",
}

# Color per scenario (grouped by compute mode)
COLOR_MAP = {
    "GS20_step1_low_m1b":       "#1f77b4",
    "GS20_step1_low_m8b":       "#1f77b4",
    "GS20_step1_low_mix":       "#1f77b4",
    "GS20_step2_sat50_m1b":     "#d62728",
    "GS20_step2_sat50_m8b":     "#d62728",
    "GS20_step2_sat50_mix":     "#d62728",
    "GS20_hybrid_queue_m1b":    "#2ca02c",
    "GS20_hybrid_queue_m8b":    "#2ca02c",
    "GS20_hybrid_queue_mix":    "#2ca02c",
}


def load(label):
    path = os.path.join(RESULTS_DIR, f"{label}_requests.csv")
    if not os.path.exists(path):
        print(f"[MISSING] {path}")
        return None
    df = pd.read_csv(path)
    df = df[df["arrival_s"] >= WARMUP_S].copy()
    return df


def stats(df):
    done = df[~df["dropped"].astype(bool)]
    if len(done) == 0:
        return None
    lat = done["total_s"].to_numpy() * 1000.0
    n_total = len(df)
    n_done = len(done)
    frac_on_gs = None
    if "chose_gs" in done.columns:
        used = done["chose_gs"].to_numpy()
        used = used[used >= 0]
        if len(used) > 0:
            frac_on_gs = float(used.mean())
    return {
        "n_total": n_total,
        "n_done": n_done,
        "drop_rate": 1 - n_done / n_total,
        "median_ms": float(np.median(lat)),
        "mean_ms": float(np.mean(lat)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "frac_on_gs": frac_on_gs,
        "lat_ms": lat,
    }


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load everything
    data = {}
    for lab in SCENARIOS:
        df = load(lab)
        if df is None:
            continue
        s = stats(df)
        if s is None:
            print(f"[EMPTY] {lab}")
            continue
        data[lab] = s

    if not data:
        print("No data loaded.")
        return

    labels = [LABEL_MAP[l] for l in data.keys()]
    colors = [COLOR_MAP[l] for l in data.keys()]

    # ---------- plot 1: median comparison ----------
    fig, ax = plt.subplots(figsize=(11, 5))
    vals = [data[l]["median_ms"] for l in data]
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Median latency (ms)")
    ax.set_title("GS20 — Median latency by scenario")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "gs20_median.png"), dpi=140)
    plt.close(fig)

    # ---------- plot 2: percentiles (median/p95/p99) grouped bars ----------
    fig, ax = plt.subplots(figsize=(12, 5))
    med = [data[l]["median_ms"] for l in data]
    p95 = [data[l]["p95_ms"] for l in data]
    p99 = [data[l]["p99_ms"] for l in data]
    x = np.arange(len(data))
    w = 0.27
    ax.bar(x - w, med, w, label="median", color="#4c72b0")
    ax.bar(x,     p95, w, label="p95",    color="#dd8452")
    ax.bar(x + w, p99, w, label="p99",    color="#55a868")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("GS20 — Latency percentiles by scenario")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "gs20_percentiles.png"), dpi=140)
    plt.close(fig)

    # ---------- plot 3: p95 only ----------
    fig, ax = plt.subplots(figsize=(11, 5))
    vals = [data[l]["p95_ms"] for l in data]
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("p95 latency (ms)")
    ax.set_title("GS20 — p95 latency by scenario")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "gs20_p95.png"), dpi=140)
    plt.close(fig)

    # ---------- plot 4: p99 only ----------
    fig, ax = plt.subplots(figsize=(11, 5))
    vals = [data[l]["p99_ms"] for l in data]
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("p99 latency (ms)")
    ax.set_title("GS20 — p99 latency by scenario")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "gs20_p99.png"), dpi=140)
    plt.close(fig)

    # ---------- plot 5: CDF ----------
    fig, ax = plt.subplots(figsize=(9, 6))
    for lab in data:
        v = np.sort(data[lab]["lat_ms"])
        y = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, y, label=LABEL_MAP[lab], color=COLOR_MAP[lab], lw=1.6,
                linestyle="-" if "m1b" in lab else ("--" if "m8b" in lab else ":"))
    ax.set_xscale("log")
    ax.set_xlabel("End-to-end latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("GS20 — Latency CDF")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "gs20_cdf.png"), dpi=140)
    plt.close(fig)

    # ---------- summary table ----------
    rows = []
    for lab in data:
        s = data[lab]
        row = {
            "scenario":   LABEL_MAP[lab],
            "n_total":    s["n_total"],
            "n_done":     s["n_done"],
            "drop_rate":  round(s["drop_rate"] * 100, 2),
            "median_ms":  round(s["median_ms"], 2),
            "mean_ms":    round(s["mean_ms"], 2),
            "p95_ms":     round(s["p95_ms"], 2),
            "p99_ms":     round(s["p99_ms"], 2),
            "pct_on_GS":  round(s["frac_on_gs"] * 100, 1) if s["frac_on_gs"] is not None else "n/a",
        }
        rows.append(row)
    tab = pd.DataFrame(rows)
    table_csv = os.path.join(PLOTS_DIR, "gs20_summary.csv")
    tab.to_csv(table_csv, index=False)
    print("\n=== GS20 summary ===")
    print(tab.to_string(index=False))
    print(f"\nSaved:")
    for f in ["gs20_median.png", "gs20_percentiles.png", "gs20_p95.png",
              "gs20_p99.png", "gs20_cdf.png", "gs20_summary.csv"]:
        print(f"  {PLOTS_DIR}/{f}")

    # ---------- companion plot: for mix scenarios, show where the request was computed ----------
    mix_scenarios = [l for l in data if l.endswith("_mix")]
    if mix_scenarios:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        xs = []
        on_gs_counts = []
        on_sat_counts = []
        total_counts = []
        for lab in mix_scenarios:
            df = load(lab)
            done = df[~df["dropped"].astype(bool)]
            if "chose_gs" in done.columns:
                chose = done["chose_gs"].to_numpy()
                valid = chose[chose >= 0]
                if len(valid) > 0:
                    on_gs = int(valid.sum())
                    on_sat = len(valid) - on_gs
                else:
                    # step1 scenario — everyone on GS; step2 scenario — everyone on sat
                    if "step1" in lab:
                        on_gs = len(done); on_sat = 0
                    else:
                        on_gs = 0; on_sat = len(done)
            else:
                if "step1" in lab:
                    on_gs = len(done); on_sat = 0
                else:
                    on_gs = 0; on_sat = len(done)
            xs.append(LABEL_MAP[lab])
            on_gs_counts.append(on_gs)
            on_sat_counts.append(on_sat)
            total_counts.append(on_gs + on_sat)

        xpos = np.arange(len(xs))
        ax.bar(xpos, on_gs_counts, label="computed at GS", color="#4c72b0")
        ax.bar(xpos, on_sat_counts, bottom=on_gs_counts,
               label="computed on sat", color="#dd8452")
        # annotate GS percentages
        for i, (gs_n, tot) in enumerate(zip(on_gs_counts, total_counts)):
            pct = 100 * gs_n / tot if tot > 0 else 0
            ax.text(i, tot, f"{pct:.1f}% GS", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(xpos)
        ax.set_xticklabels(xs, rotation=15, ha="right")
        ax.set_ylabel("Requests")
        ax.set_title("GS20 mix scenarios — where the inference ran")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "gs20_mix_gs_vs_sat.png"), dpi=140)
        plt.close(fig)
        print(f"  {PLOTS_DIR}/gs20_mix_gs_vs_sat.png")


if __name__ == "__main__":
    main()