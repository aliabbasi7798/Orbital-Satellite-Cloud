"""
run.py
------
Main entry point.

Usage:
    python3 run.py                       # runs single scenario from config.yaml
    python3 run.py --matrix              # runs all 8 comparison cells
    python3 run.py --smoke               # 60s horizon, quick smoke test

Outputs per scenario:
    results/<label>_requests.csv
    plots/<combined>_cdf.png
    plots/<combined>_stages.png
    plots/<combined>_percentiles.png
    plots/<combined>_hops.png
    plots/<label>_load.png
    results/summary.csv
"""

from __future__ import annotations
import os
import sys
import argparse
import copy
import time
import yaml
import numpy as np
import pandas as pd

from constellation import build_constellation
from ground_stations import load_ground_stations
from population import load_or_build_population
from traces import make_sampler
from routing import Router
from simulator import SimEngine
import metrics as M


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def run_single(cfg: dict, label: str, state=None, pop=None):
    """Run one scenario. If state/pop provided, skip rebuild."""
    print(f"\n==== Scenario: {label} ====")
    t0 = time.time()

    if state is None:
        print("  building constellation…")
        state = build_constellation(cfg)
        print(f"    T ticks = {len(state.times_s)}, N sats = {state.sat_positions_ecef.shape[1]}")

    gs_path = cfg["ground_stations"]["scenarios"][cfg["simulation"]["gs_scenario"]]["file"]
    gs = load_ground_stations(gs_path)
    gs_ll = np.array([[g.lat, g.lon] for g in gs])
    print(f"    GS scenario: {cfg['simulation']['gs_scenario']} ({len(gs)} stations)")
    print(f"    step: {cfg['simulation']['step']}  tier: {cfg['simulation']['gs_compute_tier']}  "
          f"models: {cfg['traffic']['active_models']}")

    if pop is None:
        pop = load_or_build_population(cfg)

    router = Router(state, cfg, gs_ll)
    rng = np.random.default_rng(cfg["traffic"]["trace_seed"])
    sampler = make_sampler(cfg, rng)

    eng = SimEngine(cfg, router, sampler, pop)
    print("  running simulation…")
    eng.run()
    print(f"    {len(eng.logs)} requests processed in {time.time()-t0:.1f}s wallclock")

    csv_path = os.path.join(cfg["output"]["results_dir"],
                            f"{label}_requests.csv")
    eng.save_request_log(csv_path)
    print(f"    wrote {csv_path}")

    df = pd.read_csv(csv_path)
    df = M.trim_warmup(df, cfg["simulation"]["warmup_s"])
    return df


def build_matrix_configs(base: dict):
    """Comparison matrix.

    Per paper spec:
      GS counts: {20, 100}
      Compute modes:
        - step1_low, step1_medium, step1_high   (GS compute at 3 tiers)
        - step2                                  (on-sat compute)
        - hybrid_noqueue                         (per-req decision, queue-blind)
        - hybrid_queue                           (per-req decision, queue-aware)
      Model mixes:
        - m_1b    : [llama3_1b]
        - m_8b    : [llama3_8b]
        - m_mix   : [llama3_1b, llama3_8b]

    Total: 2 * 6 * 3 = 36 scenarios.
    """
    model_mixes = {
        "m1b":  ["llama3_1b"],
        "m8b":  ["llama3_8b"],
        "mix":  ["llama3_1b", "llama3_8b"],
    }
    compute_modes = [
        ("step1", "low"),
        ("step1", "medium"),
        ("step1", "high"),
        ("step2", "low"),              # tier unused but must be valid
        ("hybrid_noqueue", "low"),     # hybrids use low-tier GS by default
        ("hybrid_queue", "low"),
    ]
    scenarios = []
    for gs_sc in ["low", "high"]:
        for step, tier in compute_modes:
            for mix_tag, mix_ids in model_mixes.items():
                cfg = copy.deepcopy(base)
                cfg["simulation"]["gs_scenario"] = gs_sc
                cfg["simulation"]["step"] = step
                cfg["simulation"]["gs_compute_tier"] = tier
                cfg["traffic"]["active_models"] = mix_ids
                gs_ct = cfg["ground_stations"]["scenarios"][gs_sc]["count"]
                if step == "step1":
                    mode_tag = f"step1_{tier}"
                elif step == "step2":
                    mode_tag = "step2_sat50"
                else:
                    mode_tag = step   # hybrid_noqueue / hybrid_queue
                label = f"GS{gs_ct}_{mode_tag}_{mix_tag}"
                scenarios.append((label, cfg))
    return scenarios


def build_scheduler_matrix(base: dict):
    """Scheduler experiment matrix (GS20 only, m8b + mix, rate-swept).

    Compute methods (19):
      step1_low          — GS compute, 500 TFLOPs
      step1_medium       — GS compute, 2500 TFLOPs
      step2              — sat compute, ingress only (baseline)
      khop K in {1..8}   — sat compute, local K-hop offload
      kxk  K in {2..8}   — sat compute, K×K regional (K=1 excluded ≡ step2)

    Model mixes: m8b, mix
    Rates: 100, 500, 1000, 2000 req/s
    Total: 2 * 4 * 19 = 152 scenarios.
    """
    model_mixes = {
        "m8b": ["llama3_8b"],
        "mix": ["llama3_1b", "llama3_8b"],
    }
    rates = [100, 500, 1000, 2000]
    khop_values = [1, 2, 3, 4, 5, 6, 7, 8]
    kxk_values  = [2, 3, 4, 5, 6, 7, 8]

    scenarios = []
    for mix_tag, mix_ids in model_mixes.items():
        for rate in rates:
            # step1_low (GS compute low tier)
            cfg = copy.deepcopy(base)
            cfg["simulation"]["gs_scenario"] = "low"
            cfg["simulation"]["step"] = "step1"
            cfg["simulation"]["gs_compute_tier"] = "low"
            cfg["traffic"]["active_models"] = mix_ids
            cfg["traffic"]["global_rate_req_per_s"] = float(rate)
            scenarios.append((f"GS20_step1_low_{mix_tag}_r{rate}", cfg))

            # step1_medium (GS compute medium tier)
            cfg = copy.deepcopy(base)
            cfg["simulation"]["gs_scenario"] = "low"
            cfg["simulation"]["step"] = "step1"
            cfg["simulation"]["gs_compute_tier"] = "medium"
            cfg["traffic"]["active_models"] = mix_ids
            cfg["traffic"]["global_rate_req_per_s"] = float(rate)
            scenarios.append((f"GS20_step1_medium_{mix_tag}_r{rate}", cfg))

            # step2 (sat ingress only)
            cfg = copy.deepcopy(base)
            cfg["simulation"]["gs_scenario"] = "low"
            cfg["simulation"]["step"] = "step2"
            cfg["simulation"]["gs_compute_tier"] = "low"
            cfg["traffic"]["active_models"] = mix_ids
            cfg["traffic"]["global_rate_req_per_s"] = float(rate)
            scenarios.append((f"GS20_step2_{mix_tag}_r{rate}", cfg))

            # K-hop variants
            for K in khop_values:
                cfg = copy.deepcopy(base)
                cfg["simulation"]["gs_scenario"] = "low"
                cfg["simulation"]["step"] = "khop"
                cfg["simulation"]["gs_compute_tier"] = "low"
                cfg["traffic"]["active_models"] = mix_ids
                cfg["traffic"]["global_rate_req_per_s"] = float(rate)
                cfg.setdefault("scheduling", {})
                cfg["scheduling"]["k_hop"] = K
                scenarios.append((f"GS20_khop{K}_{mix_tag}_r{rate}", cfg))

            # K×K variants
            for K in kxk_values:
                cfg = copy.deepcopy(base)
                cfg["simulation"]["gs_scenario"] = "low"
                cfg["simulation"]["step"] = "kxk"
                cfg["simulation"]["gs_compute_tier"] = "low"
                cfg["traffic"]["active_models"] = mix_ids
                cfg["traffic"]["global_rate_req_per_s"] = float(rate)
                cfg.setdefault("scheduling", {})
                cfg["scheduling"]["k_region"] = K
                scenarios.append((f"GS20_kxk{K}_{mix_tag}_r{rate}", cfg))
    return scenarios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", action="store_true",
                    help="Run the full 8-cell comparison")
    ap.add_argument("--scheduler-matrix", action="store_true",
                    help="Run the scheduler experiment matrix (khop/kxk rate-sweep)")
    ap.add_argument("--smoke", action="store_true",
                    help="Run a 60s smoke test only")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--scenario",
                    help="Run only the named scenario (for Slurm array jobs). "
                         "Use --list-scenarios to see labels.")
    ap.add_argument("--list-scenarios", action="store_true",
                    help="Print matrix scenario labels and exit")
    ap.add_argument("--list-scheduler-scenarios", action="store_true",
                    help="Print scheduler-matrix scenario labels and exit")
    args = ap.parse_args()

    base = load_config(args.config)
    if args.smoke:
        base["simulation"]["duration_s"] = 60
        base["simulation"]["warmup_s"] = 5

    os.makedirs(base["output"]["results_dir"], exist_ok=True)
    os.makedirs(base["output"]["plots_dir"], exist_ok=True)

    if args.list_scenarios:
        for label, _ in build_matrix_configs(base):
            print(label)
        return
    if args.list_scheduler_scenarios:
        for label, _ in build_scheduler_matrix(base):
            print(label)
        return

    if args.scenario:
        # Single scenario from either matrix by label (for array jobs)
        all_scen = build_matrix_configs(base) + build_scheduler_matrix(base)
        match = [(lab, cfg) for lab, cfg in all_scen if lab == args.scenario]
        if not match:
            raise SystemExit(f"No scenario named {args.scenario}. "
                             f"Use --list-scenarios or --list-scheduler-scenarios to see valid labels.")
        scenarios_cfg = match
    elif args.scheduler_matrix:
        scenarios_cfg = build_scheduler_matrix(base)
    elif args.matrix:
        scenarios_cfg = build_matrix_configs(base)
    else:
        gs_ct = base["ground_stations"]["scenarios"][base["simulation"]["gs_scenario"]]["count"]
        label = (f"GS{gs_ct}_{base['simulation']['step']}"
                 f"_{base['simulation']['gs_compute_tier']}")
        scenarios_cfg = [(label, base)]

    # Build constellation ONCE if all scenarios share the same duration/tick
    # (always true for the matrix). This saves minutes on full-orbit runs.
    state = None
    pop = None
    if len(scenarios_cfg) > 1:
        print("Building shared constellation (once for all scenarios)…")
        state = build_constellation(scenarios_cfg[0][1])
        pop = load_or_build_population(scenarios_cfg[0][1])
        print(f"  T={len(state.times_s)}, N={state.sat_positions_ecef.shape[1]}")

    dfs = {}
    for label, cfg in scenarios_cfg:
        df = run_single(cfg, label, state=state, pop=pop)
        dfs[label] = df

    summary = M.build_summary_table(dfs)
    print("\n" + summary.to_string(index=False))
    summary.to_csv(os.path.join(base["output"]["results_dir"], "summary.csv"),
                   index=False)

    plots_dir = base["output"]["plots_dir"]
    combined_tag = "matrix" if args.matrix else next(iter(dfs))
    M.plot_latency_cdf(dfs, os.path.join(plots_dir, f"{combined_tag}_cdf.png"))
    M.plot_stage_breakdown(dfs, os.path.join(plots_dir, f"{combined_tag}_stages.png"))
    M.plot_percentile_bars(dfs, os.path.join(plots_dir, f"{combined_tag}_percentiles.png"))
    M.plot_hop_distribution(dfs, os.path.join(plots_dir, f"{combined_tag}_hops.png"))
    for label, df in dfs.items():
        which = "gs" if ("step1" in label or "hybrid" in label) else "sat"
        M.plot_load_distribution(df, os.path.join(plots_dir, f"{label}_load.png"),
                                 which=which)

    print(f"\nPlots: {plots_dir}/")
    print(f"Results: {base['output']['results_dir']}/")


if __name__ == "__main__":
    main()