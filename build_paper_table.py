"""
build_paper_table.py
--------------------
Build the comparison table for GS20 / 8B / 2000 req/s across 4 scenarios:

  1. step1_low   — GS compute only (baseline)
  2. step2       — on-satellite, ingress only (baseline)
  3. khop K=4    — on-satellite, local K-hop offload
  4. kxk  K=4    — on-satellite, K×K hierarchical offload

For each scenario:
  * median, p95, p99 of total latency
  * mean and std of queue delay
  * Stage breakdown at median / p95 / p99 using Option B:
      Take requests in the percentile band and average each stage.
      - median band = [45th, 55th] percentile of total latency
      - p95 band    = [92.5, 97.5]
      - p99 band    = [98.5, 99.5]

Output:
  * console-printed human-readable table
  * results/paper_table.csv — wide CSV with every number

Usage:
  python3 build_paper_table.py
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import yaml


SCENARIOS = [
    ("step1·medium (GS only)",       "GS20_step1_medium_m8b_r2000"),
    ("step2 (sat ingress only)",  "GS20_step2_m8b_r2000"),
    ("K-hop, K=4",                "GS20_khop4_m8b_r2000"),
    ("K×K, K=4",                  "GS20_kxk4_m8b_r2000"),
]

STAGES = ["uplink_s", "out_path_s", "queue_s", "compute_s", "back_path_s", "downlink_s"]
STAGE_LABELS = {
    "uplink_s":    "uplink",
    "out_path_s":  "fwd/out-ISL",
    "queue_s":     "queue",
    "compute_s":   "compute",
    "back_path_s": "return-ISL",
    "downlink_s":  "downlink",
}

# Percentile bands (Option B)
BANDS = {
    "median": (45.0, 55.0),
    "p95":    (92.5, 97.5),
    "p99":    (98.5, 99.5),
}


def load(results_dir, label, warmup_s):
    path = os.path.join(results_dir, f"{label}_requests.csv")
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return None
    df = pd.read_csv(path)
    df = df[df["arrival_s"] >= warmup_s].reset_index(drop=True)
    done = df[~df["dropped"].astype(bool)].reset_index(drop=True)
    return done


def band_stage_means(done: pd.DataFrame, lo: float, hi: float) -> dict:
    """Average each stage within the percentile band [lo, hi] of total_s."""
    t = done["total_s"].to_numpy()
    lo_v = np.percentile(t, lo)
    hi_v = np.percentile(t, hi)
    mask = (t >= lo_v) & (t <= hi_v)
    sub = done[mask]
    out = {}
    for s in STAGES:
        out[s] = float(sub[s].mean()) * 1000 if len(sub) else 0.0  # ms
    out["total_ms"] = float(sub["total_s"].mean()) * 1000 if len(sub) else 0.0
    out["n_in_band"] = int(len(sub))
    return out


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    warmup = cfg["simulation"]["warmup_s"]
    results_dir = cfg["output"]["results_dir"]

    rows = []
    for pretty, label in SCENARIOS:
        done = load(results_dir, label, warmup)
        if done is None or len(done) == 0:
            print(f"[SKIP] {pretty} (no data)")
            continue
        t_ms = done["total_s"].to_numpy() * 1000
        q_ms = done["queue_s"].to_numpy() * 1000

        row = {
            "scenario": pretty,
            "label": label,
            "n": len(done),
            "median_ms":  float(np.median(t_ms)),
            "p95_ms":     float(np.percentile(t_ms, 95)),
            "p99_ms":     float(np.percentile(t_ms, 99)),
            "queue_mean_ms": float(np.mean(q_ms)),
            "queue_std_ms":  float(np.std(q_ms)),
        }
        for band_name, (lo, hi) in BANDS.items():
            b = band_stage_means(done, lo, hi)
            for s in STAGES:
                row[f"{band_name}_{s}"] = b[s]
            row[f"{band_name}_total_ms_avg_in_band"] = b["total_ms"]
            row[f"{band_name}_n_in_band"] = b["n_in_band"]
        rows.append(row)

    if not rows:
        print("No scenarios found.")
        return

    df = pd.DataFrame(rows)

    # --- CSV dump (wide, every number) ---
    out_path = os.path.join(results_dir, "paper_table.csv")
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}\n")

    # --- Console-readable table 1: headline metrics ---
    print("=" * 90)
    print("  TABLE 1 — Headline metrics (GS20, 8B, 2000 req/s)")
    print("=" * 90)
    print(f"{'Scenario':<28s}  {'n':>7s}  {'median':>8s}  {'p95':>8s}  {'p99':>8s}  "
          f"{'queue μ':>9s}  {'queue σ':>9s}")
    print("-" * 90)
    for r in rows:
        print(f"{r['scenario']:<28s}  {r['n']:>7d}  "
              f"{r['median_ms']:>7.1f}ms  {r['p95_ms']:>7.1f}ms  {r['p99_ms']:>7.1f}ms  "
              f"{r['queue_mean_ms']:>8.1f}ms  {r['queue_std_ms']:>8.1f}ms")
    print()

    # --- Console-readable table 2: per-band stage breakdown ---
    for band_name, (lo, hi) in BANDS.items():
        print("=" * 100)
        print(f"  TABLE 2·{band_name} — Stage breakdown for requests in "
              f"[{lo:.1f}%, {hi:.1f}%] band of total latency  (values in ms)")
        print("=" * 100)
        hdr = f"{'Scenario':<28s}  "
        for s in STAGES:
            hdr += f"{STAGE_LABELS[s]:>11s}  "
        hdr += f"{'Σ total':>9s}"
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            line = f"{r['scenario']:<28s}  "
            s_sum = 0.0
            for s in STAGES:
                v = r[f"{band_name}_{s}"]
                s_sum += v
                line += f"{v:>10.2f}   "
            line += f"{s_sum:>8.1f}"
            print(line)
        print()

    # --- Console-readable table 3: normalized share (% of total per stage) ---
    print("=" * 100)
    print("  TABLE 3 — Stage share (% of total latency, averaged in p99 band)")
    print("=" * 100)
    hdr = f"{'Scenario':<28s}  "
    for s in STAGES:
        hdr += f"{STAGE_LABELS[s]:>11s}  "
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        band_name = "p99"
        total = sum(r[f"{band_name}_{s}"] for s in STAGES)
        if total == 0:
            continue
        line = f"{r['scenario']:<28s}  "
        for s in STAGES:
            share = 100.0 * r[f"{band_name}_{s}"] / total
            line += f"{share:>9.1f}%   "
        print(line)
    print()

    print(f"Full CSV (every number) -> {out_path}")


if __name__ == "__main__":
    main()
