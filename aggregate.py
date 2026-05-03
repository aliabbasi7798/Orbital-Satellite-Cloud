"""
aggregate.py
------------
Run after a Slurm array job to assemble the final summary + plots from the
individual per-scenario CSVs in results/.

Usage:
    python3 aggregate.py
"""
from __future__ import annotations
import os
import sys
import pandas as pd
import yaml
import metrics as M


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    warmup = cfg["simulation"]["warmup_s"]
    results_dir = cfg["output"]["results_dir"]
    plots_dir = cfg["output"]["plots_dir"]
    os.makedirs(plots_dir, exist_ok=True)

    # Expected scenario order (matches build_matrix_configs enumeration)
    from run import build_matrix_configs
    order = [lab for lab, _ in build_matrix_configs(cfg)]

    scenarios = {}
    missing = []
    for label in order:
        path = os.path.join(results_dir, f"{label}_requests.csv")
        if not os.path.exists(path):
            missing.append(label)
            continue
        df = pd.read_csv(path)
        df = M.trim_warmup(df, warmup)
        scenarios[label] = df

    if missing:
        print(f"WARNING: {len(missing)} scenarios missing:")
        for m in missing[:10]:
            print(f"  - {m}")
        if len(missing) > 10:
            print(f"  ...and {len(missing)-10} more")

    if not scenarios:
        print("No results found. Run the array job first.")
        sys.exit(1)

    # Summary
    summary = M.build_summary_table(scenarios)
    summary_path = os.path.join(results_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}")
    print(summary.to_string(index=False))

    # Plots
    M.plot_latency_cdf(scenarios, os.path.join(plots_dir, "matrix_cdf.png"))
    M.plot_stage_breakdown(scenarios, os.path.join(plots_dir, "matrix_stages.png"))
    M.plot_percentile_bars(scenarios, os.path.join(plots_dir, "matrix_percentiles.png"))
    M.plot_hop_distribution(scenarios, os.path.join(plots_dir, "matrix_hops.png"))
    for label, df in scenarios.items():
        which = "gs" if ("step1" in label or "hybrid" in label) else "sat"
        M.plot_load_distribution(df, os.path.join(plots_dir, f"{label}_load.png"),
                                 which=which)
    print(f"\nPlots: {plots_dir}/")


if __name__ == "__main__":
    main()
