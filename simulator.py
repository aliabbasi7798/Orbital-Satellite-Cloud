
from __future__ import annotations
import os
import csv
import numpy as np
import simpy
from dataclasses import dataclass, field
from typing import List, Optional

from routing import Router
from schedulers import khop_choose, kxk_choose, KxKRegions

GBPS = 1e9


@dataclass
class RequestLog:
    req_id: int
    arrival_s: float = 0.0
    uplink_s: float = 0.0
    out_path_s: float = 0.0
    queue_s: float = 0.0
    compute_s: float = 0.0
    back_path_s: float = 0.0
    downlink_s: float = 0.0
    total_s: float = 0.0
    ingress_sat: int = -1
    compute_node: int = -1       # GS id if GS compute, sat id if sat compute
    egress_sat: int = -1
    gs_id: int = -1
    hops_out: int = 0
    hops_back: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    flops: float = 0.0
    model_idx: int = 0
    model_id: str = ""
    chose_gs: int = -1           # 1/0 for hybrid, else -1
    est_t_sat_s: float = -1.0
    est_t_gs_s: float = -1.0
    dropped: bool = False
    drop_reason: str = ""


class QueueTracker:
    """
    Tracks remaining work (sum of service times of queued + in-progress jobs)
    at each compute node for queue-aware hybrid decisions.

    Maintained explicitly rather than inspecting SimPy Resource internals —
    decrement on compute-start, decrement on compute-end is handled by the
    simulator loop calling these methods.

    Only used by hybrid_queue; cheap for other modes.
    """
    def __init__(self, n_nodes: int):
        self.remaining = np.zeros(n_nodes, dtype=np.float64)

    def enqueue(self, node_id: int, service_time: float):
        self.remaining[node_id] += service_time

    def dequeue(self, node_id: int, service_time: float):
        self.remaining[node_id] -= service_time
        if self.remaining[node_id] < 0:
            self.remaining[node_id] = 0.0


class SimEngine:
    def __init__(self, cfg: dict, router: Router, sampler, population: np.ndarray):
        self.cfg = cfg
        self.router = router
        self.sampler = sampler
        self.pop = population

        self.env = simpy.Environment()
        self.rng = np.random.default_rng(cfg["simulation"]["seed"])
        self.step = cfg["simulation"]["step"]
        self.duration = float(cfg["simulation"]["duration_s"])
        self.warmup = float(cfg["simulation"]["warmup_s"])

        self.bw_isl = cfg["links"]["isl_bandwidth_gbps"] * GBPS
        self.bw_up = cfg["links"]["uplink_bandwidth_gbps"] * GBPS
        self.bw_dn = cfg["links"]["downlink_bandwidth_gbps"] * GBPS

        tier = cfg["simulation"]["gs_compute_tier"]
        self.gs_tflops = cfg["compute"][f"gs_tflops_{tier}"] * 1e12
        self.sat_tflops = cfg["compute"]["sat_tflops"] * 1e12

        self.c = cfg["physics"]["speed_of_light_km_s"]
        self.user_min_el = cfg["links"]["user_min_elevation_deg"]

        self.G = router.G
        self.N = router.N
        self.gs_queue = [simpy.Resource(self.env, capacity=1) for _ in range(self.G)]
        self.sat_queue = [simpy.Resource(self.env, capacity=1) for _ in range(self.N)]
        self.gs_qt = QueueTracker(self.G)
        self.sat_qt = QueueTracker(self.N)

        # K-hop & K×K config
        sched_cfg = cfg.get("scheduling", {})
        self.K_hop = int(sched_cfg.get("k_hop", 1))
        self.K_region = int(sched_cfg.get("k_region", 4))
        # Representative per-hop delay used inside the scheduler's cost estimate.
        # Serialization of a typical request (800 bytes) over ISL + median ISL prop.
        self.hop_delay_s = sched_cfg.get("hop_delay_s", 0.006)  # ~6 ms
        # Precompute K×K regions if the mode needs it
        if self.step == "kxk":
            self.regions = KxKRegions(
                num_planes=router.state.num_planes,
                sats_per_plane=router.state.sats_per_plane,
                K=self.K_region,
            )
        else:
            self.regions = None

        # Model id lookup — for logging and for hybrid decision
        self.active_models = self.sampler.active_models
        self.model_ids = [m.model_id for m in self.active_models]

        self.logs: List[RequestLog] = []
        self._next_req_id = 0
        self._prepare_cell_rates()

    def _prepare_cell_rates(self):
        total = float(self.pop.sum())
        rate = self.cfg["traffic"]["global_rate_req_per_s"]
        self.cell_rates = (self.pop / total) * rate
        self.lat_centers = np.arange(89.5, -90.0, -1.0)
        self.lon_centers = np.arange(-179.5, 180.0, 1.0)

    # ---- arrivals ----
    def gen_arrivals(self):
        total_rate = float(self.cell_rates.sum())
        flat = self.cell_rates.flatten()
        probs = flat / flat.sum()
        idxs = np.arange(flat.size)
        BATCH = 5000
        while True:
            iats = self.rng.exponential(1.0 / total_rate, BATCH)
            cell_ids = self.rng.choice(idxs, size=BATCH, p=probs)
            in_t, out_t, flops, in_b, out_b, midx = self.sampler.sample_batch(BATCH)
            u_lat = self.rng.random(BATCH)
            u_lon = self.rng.random(BATCH)
            for k in range(BATCH):
                yield self.env.timeout(iats[k])
                if self.env.now > self.duration:
                    return
                ci = int(cell_ids[k])
                lat_i, lon_i = ci // 360, ci % 360
                lat = self.lat_centers[lat_i] + (u_lat[k] - 0.5)
                lon = self.lon_centers[lon_i] + (u_lon[k] - 0.5)
                rid = self._next_req_id
                self._next_req_id += 1
                mi = int(midx[k])
                log = RequestLog(
                    req_id=rid,
                    arrival_s=self.env.now,
                    in_tokens=int(in_t[k]),
                    out_tokens=int(out_t[k]),
                    flops=float(flops[k]),
                    model_idx=mi,
                    model_id=self.model_ids[mi],
                )
                self.env.process(self._dispatch(log, lat, lon,
                                                int(in_b[k]), int(out_b[k])))

    def _dispatch(self, log, lat, lon, in_bytes, out_bytes):
        s = self.step
        if s == "step1":
            yield from self._run_step1(log, lat, lon, in_bytes, out_bytes)
        elif s == "step2":
            yield from self._run_step2(log, lat, lon, in_bytes, out_bytes)
        elif s in ("hybrid_noqueue", "hybrid_queue"):
            yield from self._run_hybrid(log, lat, lon, in_bytes, out_bytes,
                                        queue_aware=(s == "hybrid_queue"))
        elif s == "khop":
            yield from self._run_khop(log, lat, lon, in_bytes, out_bytes)
        elif s == "kxk":
            yield from self._run_kxk(log, lat, lon, in_bytes, out_bytes)
        else:
            raise ValueError(s)
        self.logs.append(log)

    # ---------- shared primitives ----------
    def _uplink(self, log, lat, lon, in_bytes):
        """Do uplink; return (ingress_sat, slant_km) or None if dropped."""
        env, r, c = self.env, self.router, self.c
        t0 = env.now
        tick = r.tick_of(t0)
        v = r.nearest_sat_to_ground(lat, lon, tick, self.user_min_el)
        if v is None:
            log.dropped = True; log.drop_reason = "no_ingress_sat"
            return None
        ingress, slant = v
        log.ingress_sat = ingress
        yield env.timeout((in_bytes * 8) / self.bw_up + slant / c)
        log.uplink_s = env.now - t0
        return ingress, slant

    def _out_to_gs(self, log, ingress, in_bytes):
        """Route ingress -> closest GS via ISL + downlink to GS. Fill log fields."""
        env, r, c = self.env, self.router, self.c
        tick = r.tick_of(env.now)
        route = r.route_to_closest_gs(ingress, tick)
        if route is None:
            log.dropped = True; log.drop_reason = "no_gs_out"
            return None
        log.gs_id = route["gs_id"]
        log.egress_sat = route["egress_sat"]
        log.hops_out = route["hops"]
        t_out = env.now
        for i in range(len(route["path"]) - 1):
            a, b = route["path"][i], route["path"][i + 1]
            tk = r.tick_of(env.now)
            d = r.isl_km(a, b, tk)
            yield env.timeout((in_bytes * 8) / self.bw_isl + d / c)
        gs_ecef = r.gs_ecef[log.gs_id]
        tk = r.tick_of(env.now)
        dk = r.slant_km(log.egress_sat, tk, gs_ecef)
        yield env.timeout((in_bytes * 8) / self.bw_dn + dk / c)
        log.out_path_s = env.now - t_out
        return route

    def _compute_at(self, log, kind: str, node_id: int):
        """Run FIFO queue + compute at (gs|sat, node_id). Maintains QueueTracker."""
        env = self.env
        queues = self.gs_queue if kind == "gs" else self.sat_queue
        tracker = self.gs_qt if kind == "gs" else self.sat_qt
        tflops = self.gs_tflops if kind == "gs" else self.sat_tflops
        svc = log.flops / tflops
        tracker.enqueue(node_id, svc)
        t_q = env.now
        try:
            with queues[node_id].request() as req:
                yield req
                log.queue_s = env.now - t_q
                yield env.timeout(svc)
                log.compute_s = svc
        finally:
            tracker.dequeue(node_id, svc)

    def _return_from_gs(self, log, lat, lon, out_bytes):
        env, r, c = self.env, self.router, self.c
        t_b = env.now
        tick = r.tick_of(env.now)
        rr = r.nearest_sat_to_ground(lat, lon, tick, self.user_min_el)
        if rr is None:
            log.dropped = True; log.drop_reason = "no_user_sat_return"; return
        user_sat, user_slant = rr
        hops_arr, pred_arr, _ = None, None, None
        hops_arr, pred_arr = r.gs_bfs(log.gs_id, tick)
        if hops_arr[user_sat] < 0:
            log.dropped = True; log.drop_reason = "return_unreach"; return
        path_back = r.reconstruct_path(pred_arr, user_sat)
        gw = int(path_back[0])
        gs_ecef = r.gs_ecef[log.gs_id]
        gw_slant = float(np.linalg.norm(r.state.sat_positions_ecef[tick, gw] - gs_ecef))
        log.hops_back = int(hops_arr[user_sat])
        yield env.timeout((out_bytes * 8) / self.bw_up + gw_slant / c)
        for i in range(len(path_back) - 1):
            a, b = path_back[i], path_back[i + 1]
            tk = r.tick_of(env.now)
            d = r.isl_km(a, b, tk)
            yield env.timeout((out_bytes * 8) / self.bw_isl + d / c)
        ser = (out_bytes * 8) / self.bw_dn
        prop = user_slant / c
        yield env.timeout(ser + prop)
        log.downlink_s = ser + prop
        log.back_path_s = env.now - t_b - log.downlink_s

    def _return_from_sat(self, log, ingress, lat, lon, out_bytes):
        """From compute-done sat back to user."""
        env, r, c = self.env, self.router, self.c
        t_b = env.now
        tick = r.tick_of(env.now)
        rr = r.nearest_sat_to_ground(lat, lon, tick, self.user_min_el)
        if rr is None:
            log.dropped = True; log.drop_reason = "no_user_sat_return"; return
        return_sat, user_slant = rr
        log.egress_sat = return_sat
        if return_sat != ingress:
            r2 = r.route_sat_to_sat(ingress, return_sat, tick)
            if r2 is None:
                log.dropped = True; log.drop_reason = "sat_return_unreach"; return
            log.hops_back = r2["hops"]
            for i in range(len(r2["path"]) - 1):
                a, b = r2["path"][i], r2["path"][i + 1]
                tk = r.tick_of(env.now)
                d = r.isl_km(a, b, tk)
                yield env.timeout((out_bytes * 8) / self.bw_isl + d / c)
        ser = (out_bytes * 8) / self.bw_dn
        prop = user_slant / c
        yield env.timeout(ser + prop)
        log.downlink_s = ser + prop
        log.back_path_s = env.now - t_b - log.downlink_s

    # ---------- step 1 ----------
    def _run_step1(self, log, lat, lon, in_bytes, out_bytes):
        t0 = self.env.now
        v = yield from self._uplink(log, lat, lon, in_bytes)
        if v is None: return
        ingress, _ = v
        r = yield from self._out_to_gs(log, ingress, in_bytes)
        if r is None: return
        log.compute_node = log.gs_id
        yield from self._compute_at(log, "gs", log.gs_id)
        yield from self._return_from_gs(log, lat, lon, out_bytes)
        log.total_s = self.env.now - t0

    # ---------- step 2 ----------
    def _run_step2(self, log, lat, lon, in_bytes, out_bytes):
        t0 = self.env.now
        v = yield from self._uplink(log, lat, lon, in_bytes)
        if v is None: return
        ingress, _ = v
        log.compute_node = ingress
        yield from self._compute_at(log, "sat", ingress)
        yield from self._return_from_sat(log, ingress, lat, lon, out_bytes)
        log.total_s = self.env.now - t0

    # ---------- K-hop (cost-aware) ----------
    def _run_khop(self, log, lat, lon, in_bytes, out_bytes):
        """Ingress sat looks K hops around itself for the lowest-cost sat,
        forwards request via ISL, that sat computes, then return-to-user.
        Cost = hops*hop_delay + queue_remaining + service_time."""
        env, r, c = self.env, self.router, self.c
        t0 = env.now

        v = yield from self._uplink(log, lat, lon, in_bytes)
        if v is None: return
        ingress, _ = v
        tick = r.tick_of(env.now)

        compute_sat, hops_fwd = khop_choose(
            router=r, ingress_sat=ingress, tick=tick, K=self.K_hop,
            flops=log.flops, sat_tflops=self.sat_tflops,
            queue_remaining=self.sat_qt.remaining,
            hop_delay_s=self.hop_delay_s,
        )
        log.compute_node = compute_sat
        log.hops_out = hops_fwd

        # Forward ingress -> compute_sat via BFS
        if compute_sat != ingress:
            r2 = r.route_sat_to_sat(ingress, compute_sat, tick)
            if r2 is None:
                log.dropped = True; log.drop_reason = "khop_forward_unreach"; return
            t_fwd = env.now
            for i in range(len(r2["path"]) - 1):
                a, b = r2["path"][i], r2["path"][i + 1]
                tk = r.tick_of(env.now)
                d = r.isl_km(a, b, tk)
                yield env.timeout((in_bytes * 8) / self.bw_isl + d / c)
            log.out_path_s = env.now - t_fwd

        yield from self._compute_at(log, "sat", compute_sat)
        yield from self._return_from_sat(log, compute_sat, lat, lon, out_bytes)
        log.total_s = env.now - t0

    # ---------- K×K hierarchical ----------
    def _run_kxk(self, log, lat, lon, in_bytes, out_bytes):
        """Ingress sat forwards request to its region leader; leader picks
        min-queue worker in its K×K region; worker computes; return-to-user
        from worker."""
        env, r, c = self.env, self.router, self.c
        t0 = env.now
        assert self.regions is not None, "kxk mode requires regions"

        v = yield from self._uplink(log, lat, lon, in_bytes)
        if v is None: return
        ingress, _ = v
        tick = r.tick_of(env.now)

        leader, compute_sat, _, _ = kxk_choose(
            regions=self.regions, ingress_sat=ingress,
            queue_remaining=self.sat_qt.remaining,
            flops=log.flops, sat_tflops=self.sat_tflops,
        )
        log.compute_node = compute_sat

        total_fwd_hops = 0
        t_fwd = env.now

        # Hop 1: ingress -> leader (if different)
        if leader != ingress:
            r1 = r.route_sat_to_sat(ingress, leader, tick)
            if r1 is None:
                log.dropped = True; log.drop_reason = "kxk_ingress_to_leader_unreach"; return
            total_fwd_hops += r1["hops"]
            for i in range(len(r1["path"]) - 1):
                a, b = r1["path"][i], r1["path"][i + 1]
                tk = r.tick_of(env.now)
                d = r.isl_km(a, b, tk)
                yield env.timeout((in_bytes * 8) / self.bw_isl + d / c)

        # Hop 2: leader -> compute (if different)
        if compute_sat != leader:
            tick = r.tick_of(env.now)
            r2 = r.route_sat_to_sat(leader, compute_sat, tick)
            if r2 is None:
                log.dropped = True; log.drop_reason = "kxk_leader_to_compute_unreach"; return
            total_fwd_hops += r2["hops"]
            for i in range(len(r2["path"]) - 1):
                a, b = r2["path"][i], r2["path"][i + 1]
                tk = r.tick_of(env.now)
                d = r.isl_km(a, b, tk)
                yield env.timeout((in_bytes * 8) / self.bw_isl + d / c)

        log.hops_out = total_fwd_hops
        if total_fwd_hops > 0:
            log.out_path_s = env.now - t_fwd

        yield from self._compute_at(log, "sat", compute_sat)
        yield from self._return_from_sat(log, compute_sat, lat, lon, out_bytes)
        log.total_s = env.now - t0


    # ---------- hybrid ----------
    def _run_hybrid(self, log, lat, lon, in_bytes, out_bytes, queue_aware: bool):
        """Per-request decision: sat vs GS.

        Decision cost estimates (fixed cost parts that *differ* between paths):
          sat path extra cost vs GS path:
            t_sat_est = [queue_sat] + flops/sat_tflops                  + isl_back_hops * HOP_EST
            t_gs_est  = hops_out*HOP_EST + [queue_gs] + flops/gs_tflops + isl_back_hops_gs * HOP_EST
        We compare only the differentiating parts (uplink/downlink cancel).
        HOP_EST = approximate per-hop delay (serialization + mean prop).
        """
        env, r, c = self.env, self.router, self.c
        t0 = env.now
        v = yield from self._uplink(log, lat, lon, in_bytes)
        if v is None: return
        ingress, _ = v

        # ---- decision ----
        # Pre-estimate the hop count and candidate GS (closest from ingress).
        tick = r.tick_of(env.now)
        route = r.route_to_closest_gs(ingress, tick)
        if route is None:
            # GS unreachable; forced to compute on sat
            log.chose_gs = 0
            log.est_t_sat_s = log.flops / self.sat_tflops
            log.est_t_gs_s = float("inf")
            log.compute_node = ingress
            yield from self._compute_at(log, "sat", ingress)
            yield from self._return_from_sat(log, ingress, lat, lon, out_bytes)
            log.total_s = env.now - t0
            return

        # Per-hop delay estimate: serialization of in_bytes over ISL + avg ISL prop
        # Use a representative ISL distance (~median slant between connected sats ~1500 km)
        HOP_PROP_S = 1500.0 / c
        hop_est_isl_s = (in_bytes * 8) / self.bw_isl + HOP_PROP_S
        # For the return-path hops in both options, assume same HOP_EST (cancels)
        hops_out = route["hops"]
        cand_gs = route["gs_id"]

        svc_sat = log.flops / self.sat_tflops
        svc_gs = log.flops / self.gs_tflops
        q_sat = self.sat_qt.remaining[ingress] if queue_aware else 0.0
        q_gs = self.gs_qt.remaining[cand_gs] if queue_aware else 0.0

        # Sat path: queue + compute (return hops symmetric, drop)
        est_sat = q_sat + svc_sat
        # GS path: out hops + queue + compute (return hops symmetric, drop)
        est_gs = hops_out * hop_est_isl_s + q_gs + svc_gs

        log.est_t_sat_s = est_sat
        log.est_t_gs_s = est_gs
        chose_gs = est_gs <= est_sat
        log.chose_gs = 1 if chose_gs else 0

        # ---- execute chosen path ----
        if chose_gs:
            log.gs_id = cand_gs
            log.egress_sat = route["egress_sat"]
            log.hops_out = hops_out
            # Re-traverse out-path (we already have it)
            t_out = env.now
            for i in range(len(route["path"]) - 1):
                a, b = route["path"][i], route["path"][i + 1]
                tk = r.tick_of(env.now)
                d = r.isl_km(a, b, tk)
                yield env.timeout((in_bytes * 8) / self.bw_isl + d / c)
            gs_ecef = r.gs_ecef[log.gs_id]
            tk = r.tick_of(env.now)
            dk = r.slant_km(log.egress_sat, tk, gs_ecef)
            yield env.timeout((in_bytes * 8) / self.bw_dn + dk / c)
            log.out_path_s = env.now - t_out
            log.compute_node = log.gs_id
            yield from self._compute_at(log, "gs", log.gs_id)
            yield from self._return_from_gs(log, lat, lon, out_bytes)
        else:
            log.compute_node = ingress
            yield from self._compute_at(log, "sat", ingress)
            yield from self._return_from_sat(log, ingress, lat, lon, out_bytes)
        log.total_s = env.now - t0

    # ---- run ----
    def run(self):
        self.env.process(self.gen_arrivals())
        self.env.run(until=self.duration)

    def save_request_log(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fields = ["req_id", "arrival_s", "uplink_s", "out_path_s", "queue_s",
                  "compute_s", "back_path_s", "downlink_s", "total_s",
                  "ingress_sat", "compute_node", "egress_sat", "gs_id",
                  "hops_out", "hops_back", "in_tokens", "out_tokens", "flops",
                  "model_idx", "model_id",
                  "chose_gs", "est_t_sat_s", "est_t_gs_s",
                  "dropped", "drop_reason"]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(fields)
            for L in self.logs:
                w.writerow([getattr(L, k) for k in fields])