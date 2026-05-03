"""
run_sweep.py
------------
Run the full GS20 scheduler sweep serially, organized by (rate, mix) batches.

What it does
  * Runs all 152 scenarios (4 rates × 2 mixes × 19 methods) in the scheduler
    matrix.
  * Saves results to `results_sweep/` (keeps your existing `results/` clean).
  * Resumes gracefully: skips any scenario whose CSV already exists.
  * After finishing each (rate, mix) batch (19 scenarios), prints the full
    comparison table to console AND saves it as a CSV.
  * Uses a shared constellation across all scenarios — one-time ~30s build.

Output
  results_sweep/<scenario>_requests.csv    — per-request traces
  results_sweep/paper_table_r<R>_<mix>.csv — summary table per batch (8 total)

Usage
  python3 run_sweep.py                  # full run (resume-friendly)
  python3 run_sweep.py --force          # delete existing results and redo
  python3 run_sweep.py --only-rate 2000 # only run rate=2000 batches
  python3 run_sweep.py --only-mix m8b   # only run m8b (both rates)
  python3 run_sweep.py --smoke          # 60s horizon per scenario
"""

from __future__ import annotations
import os
import sys
import time
import argparse
import shutil
import copy
import yaml
import numpy as np
import pandas as pd

from constellation import build_constellation
from ground_stations import load_ground_stations
from population import load_or_build_population
from traces import make_sampler
from routing import Router
from simulator import SimEngine
from run import build_scheduler_matrix


STAGES = ["uplink_s", "out_path_s", "queue_s", "compute_s", "back_path_s", "downlink_s"]
STAGE_LABELS = {
    "uplink_s":    "uplink",
    "out_path_s":  "fwd/out-ISL",
    "queue_s":     "queue",
    "compute_s":   "compute",
    "back_path_s": "return-ISL",
    "downlink_s":  "downlink",
}
BANDS = {
    "median": (45.0, 55.0),
    "p95":    (92.5, 97.5),
    "p99":    (98.5, 99.5),
}

# Scenario method order within a batch (19 methods per batch)
METHOD_ORDER = [
    ("step1_low",    "step1·low (GS, 500 TF)"),
    ("step1_medium", "step1·medium (GS, 2.5 PF)"),
    ("step2",        "step2 (sat ingress only)"),
    ("khop1",        "K-hop K=1"),
    ("khop2",        "K-hop K=2"),
    ("khop3",        "K-hop K=3"),
    ("khop4",        "K-hop K=4"),
    ("khop5",        "K-hop K=5"),
    ("khop6",        "K-hop K=6"),
    ("khop7",        "K-hop K=7"),
    ("khop8",        "K-hop K=8"),
    ("kxk2",         "K×K K=2"),
    ("kxk3",         "K×K K=3"),
    ("kxk4",         "K×K K=4"),
    ("kxk5",         "K×K K=5"),
    ("kxk6",         "K×K K=6"),
    ("kxk7",         "K×K K=7"),
    ("kxk8",         "K×K K=8"),
]


def fmt_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h: return f"{h}h{m:02d}m{s:02d}s"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"


def run_scenario(cfg: dict, label: str, state, pop, results_dir: str):
    """Run one scenario, save CSV, return (elapsed_seconds, num_requests)."""
    t0 = time.time()
    gs_path = cfg["ground_stations"]["scenarios"][cfg["simulation"]["gs_scenario"]]["file"]
    gs = load_ground_stations(gs_path)
    gs_ll = np.array([[g.lat, g.lon] for g in gs])
    router = Router(state, cfg, gs_ll)
    rng = np.random.default_rng(cfg["traffic"]["trace_seed"])
    sampler = make_sampler(cfg, rng)
    eng = SimEngine(cfg, router, sampler, pop)
    eng.run()
    csv_path = os.path.join(results_dir, f"{label}_requests.csv")
    eng.save_request_log(csv_path)
    return time.time() - t0, len(eng.logs)


def band_stage_means(done: pd.DataFrame, lo: float, hi: float) -> dict:
    t = done["total_s"].to_numpy()
    lo_v = np.percentile(t, lo)
    hi_v = np.percentile(t, hi)
    mask = (t >= lo_v) & (t <= hi_v)
    sub = done[mask]
    out = {}
    for s in STAGES:
        out[s] = float(sub[s].mean()) * 1000 if len(sub) else 0.0
    out["total_ms"] = float(sub["total_s"].mean()) * 1000 if len(sub) else 0.0
    out["n_in_band"] = int(len(sub))
    return out


def build_batch_table(results_dir: str, rate: int, mix: str,
                      warmup_s: float) -> pd.DataFrame:
    """Build the full table for a (rate, mix) batch. One row per method."""
    rows = []
    for method_key, pretty in METHOD_ORDER:
        label = f"GS20_{method_key}_{mix}_r{rate}"
        path = os.path.join(results_dir, f"{label}_requests.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df = df[df["arrival_s"] >= warmup_s].reset_index(drop=True)
        done = df[~df["dropped"].astype(bool)].reset_index(drop=True)
        if len(done) == 0:
            continue
        t_ms = done["total_s"].to_numpy() * 1000
        q_ms = done["queue_s"].to_numpy() * 1000
        row = {
            "method": pretty,
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
    return pd.DataFrame(rows)


def print_batch_tables(df: pd.DataFrame, rate: int, mix: str):
    if len(df) == 0:
        print(f"  (empty batch for rate={rate}, mix={mix})")
        return
    print()
    print("=" * 110)
    print(f"  TABLE 1 — Headline metrics (GS20, {mix}, {rate} req/s)")
    print("=" * 110)
    print(f"{'Method':<32s}  {'n':>7s}  {'median':>9s}  {'p95':>10s}  "
          f"{'p99':>10s}  {'queue μ':>10s}  {'queue σ':>10s}")
    print("-" * 110)
    for _, r in df.iterrows():
        print(f"{r['method']:<32s}  {r['n']:>7d}  "
              f"{r['median_ms']:>7.1f}ms  {r['p95_ms']:>8.1f}ms  {r['p99_ms']:>8.1f}ms  "
              f"{r['queue_mean_ms']:>8.1f}ms  {r['queue_std_ms']:>8.1f}ms")

    for band_name in BANDS:
        print()
        print("=" * 120)
        lo, hi = BANDS[band_name]
        print(f"  TABLE 2·{band_name} — Stage breakdown in [{lo:.1f}%, {hi:.1f}%] band "
              f"(ms)  —  GS20, {mix}, {rate} req/s")
        print("=" * 120)
        hdr = f"{'Method':<32s}  "
        for s in STAGES:
            hdr += f"{STAGE_LABELS[s]:>11s}  "
        hdr += f"{'Σ total':>9s}"
        print(hdr)
        print("-" * len(hdr))
        for _, r in df.iterrows():
            line = f"{r['method']:<32s}  "
            s_sum = 0.0
            for s in STAGES:
                v = r[f"{band_name}_{s}"]
                s_sum += v
                line += f"{v:>10.2f}   "
            line += f"{s_sum:>8.1f}"
            print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Delete existing results_sweep/ and redo everything")
    ap.add_argument("--smoke", action="store_true",
                    help="60 s horizon per scenario (quick sanity run)")
    ap.add_argument("--only-rate", type=int, default=None,
                    help="Only run this rate (e.g. 2000)")
    ap.add_argument("--only-mix", default=None,
                    help="Only run this mix: m8b or mix")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    if args.smoke:
        base_cfg["simulation"]["duration_s"] = 60
        base_cfg["simulation"]["warmup_s"] = 5
        print("[SMOKE] duration_s=60, warmup_s=5\n")

    # New sandbox directory — keeps existing results/ untouched
    results_dir = "results_sweep"
    if args.force and os.path.isdir(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # Get the full 152-scenario list
    all_scenarios = build_scheduler_matrix(base_cfg)

    # Group by (rate, mix) in desired order
    batches = {}
    for label, cfg in all_scenarios:
        rate = int(cfg["traffic"]["global_rate_req_per_s"])
        mix = "mix" if len(cfg["traffic"]["active_models"]) > 1 else "m8b"
        if args.only_rate and rate != args.only_rate:
            continue
        if args.only_mix and mix != args.only_mix:
            continue
        batches.setdefault((rate, mix), []).append((label, cfg))

    rate_order = [100, 500, 1000, 2000]
    mix_order  = ["m8b", "mix"]
    ordered_keys = [(r, m) for r in rate_order for m in mix_order if (r, m) in batches]

    n_total_scenarios = sum(len(batches[k]) for k in ordered_keys)
    print(f"Scheduled {n_total_scenarios} scenarios across {len(ordered_keys)} batches.\n")

    # ---------- build constellation once (shared across ALL scenarios) ----------
    print("Building shared constellation (once)…")
    t0 = time.time()
    state = build_constellation(all_scenarios[0][1])
    pop = load_or_build_population(all_scenarios[0][1])
    print(f"  T = {len(state.times_s)}, N = {state.sat_positions_ecef.shape[1]}, "
          f"built in {fmt_hms(time.time() - t0)}\n")

    t_master = time.time()
    done_count = 0
    for (rate, mix) in ordered_keys:
        batch = batches[(rate, mix)]
        print()
        print("#" * 80)
        print(f"#  BATCH: rate={rate} req/s, mix={mix}   ({len(batch)} scenarios)")
        print("#" * 80)

        for i, (label, cfg) in enumerate(batch, 1):
            csv_path = os.path.join(results_dir, f"{label}_requests.csv")
            done_count += 1
            prefix = f"[{done_count}/{n_total_scenarios}] {label}"
            if os.path.exists(csv_path) and not args.force:
                print(f"{prefix}   (exists, skip)")
                continue
            t_sc = time.time()
            print(f"{prefix}   running…", end=" ", flush=True)
            try:
                elapsed, n_req = run_scenario(cfg, label, state, pop, results_dir)
                print(f"{n_req} req, {fmt_hms(elapsed)}")
            except Exception as e:
                print(f"FAILED: {e}")
                import traceback; traceback.print_exc()

        # After the batch, build + print + save the table
        df = build_batch_table(results_dir, rate, mix, base_cfg["simulation"]["warmup_s"])
        table_path = os.path.join(results_dir, f"paper_table_r{rate}_{mix}.csv")
        df.to_csv(table_path, index=False)
        print_batch_tables(df, rate, mix)
        print(f"\n  ==> saved {table_path}")

    print()
    print("=" * 80)
    print(f"ALL DONE — total wallclock {fmt_hms(time.time() - t_master)}")
    print(f"Results in: {results_dir}/")
    print(f"Tables: {results_dir}/paper_table_r*_*.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
