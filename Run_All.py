
from __future__ import annotations
import os
import sys
import time
import argparse
import yaml
import numpy as np

from constellation import build_constellation
from ground_stations import load_ground_stations
from population import load_or_build_population
from traces import make_sampler
from routing import Router
from simulator import SimEngine
from run import build_matrix_configs


def fmt_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h: return f"{h}h{m:02d}m{s:02d}s"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="Use 60s horizon (fast smoke test)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run all scenarios even if results already exist")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Only run the named scenarios")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="Skip final summary + plot step")
    args = ap.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    if args.smoke:
        base_cfg["simulation"]["duration_s"] = 60
        base_cfg["simulation"]["warmup_s"] = 5
        print(f"[SMOKE MODE] duration_s=60, warmup_s=5")

    os.makedirs(base_cfg["output"]["results_dir"], exist_ok=True)
    os.makedirs(base_cfg["output"]["plots_dir"], exist_ok=True)

    # Build the full scenario list
    all_scenarios = build_matrix_configs(base_cfg)
    if args.only:
        wanted = set(args.only)
        all_scenarios = [(lab, cfg) for lab, cfg in all_scenarios if lab in wanted]
        missing = wanted - {lab for lab, _ in all_scenarios}
        if missing:
            print(f"WARNING: unknown scenarios: {sorted(missing)}")
        if not all_scenarios:
            print("No matching scenarios.")
            sys.exit(1)

    # Decide what to actually run (skip existing unless --force)
    results_dir = base_cfg["output"]["results_dir"]
    to_run = []
    for lab, cfg in all_scenarios:
        csv_path = os.path.join(results_dir, f"{lab}_requests.csv")
        if os.path.exists(csv_path) and not args.force:
            print(f"[SKIP] {lab} (already exists)")
            continue
        to_run.append((lab, cfg))

    if not to_run:
        print("\nAll scenarios already complete. Running aggregator…")
        if not args.no_aggregate:
            os.system(f"{sys.executable} aggregate.py")
        return

    total = len(to_run)
    print(f"\n{total} scenario(s) to run out of {len(all_scenarios)} total.")
    est_per = 10 * 60 if not args.smoke else 5
    print(f"Estimated wallclock: {fmt_hms(total * est_per)} "
          f"(~{est_per//60}min each)\n")

    # Build the constellation ONCE — all scenarios share it
    print("=" * 70)
    print("Building constellation (shared across all scenarios)…")
    t0 = time.time()
    state = build_constellation(to_run[0][1])
    pop = load_or_build_population(to_run[0][1])
    print(f"  T ticks = {len(state.times_s)}, N sats = {state.sat_positions_ecef.shape[1]}")
    print(f"  Built in {fmt_hms(time.time() - t0)}")
    print("=" * 70)

    # Run each scenario
    t_start = time.time()
    for i, (lab, cfg) in enumerate(to_run, 1):
        t_sc = time.time()
        elapsed = time.time() - t_start
        print(f"\n[{i}/{total}] {lab}")
        print(f"       elapsed={fmt_hms(elapsed)}  "
              f"gs={cfg['simulation']['gs_scenario']}  "
              f"step={cfg['simulation']['step']}  "
              f"models={cfg['traffic']['active_models']}")
        try:
            gs_path = cfg["ground_stations"]["scenarios"][cfg["simulation"]["gs_scenario"]]["file"]
            gs = load_ground_stations(gs_path)
            gs_ll = np.array([[g.lat, g.lon] for g in gs])
            router = Router(state, cfg, gs_ll)
            rng = np.random.default_rng(cfg["traffic"]["trace_seed"])
            sampler = make_sampler(cfg, rng)
            eng = SimEngine(cfg, router, sampler, pop)
            eng.run()
            csv_path = os.path.join(results_dir, f"{lab}_requests.csv")
            eng.save_request_log(csv_path)
            done = [L for L in eng.logs if not L.dropped]
            if done:
                lat_ms = np.array([L.total_s for L in done]) * 1000
                p50 = np.median(lat_ms)
                p99 = np.percentile(lat_ms, 99)
                print(f"       ✓ {len(eng.logs)} requests, "
                      f"p50={p50:.1f}ms p99={p99:.1f}ms  "
                      f"({fmt_hms(time.time()-t_sc)})")
            else:
                print(f"       ✓ {len(eng.logs)} requests (all dropped!)  "
                      f"({fmt_hms(time.time()-t_sc)})")
        except Exception as e:
            print(f"       ✗ FAILED: {e}")
            import traceback; traceback.print_exc()

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"All scenarios done in {fmt_hms(total_time)}")
    print("=" * 70)

    if not args.no_aggregate:
        print("\nRunning aggregator to build summary + plots…")
        os.system(f"{sys.executable} aggregate.py")


if __name__ == "__main__":
    main()