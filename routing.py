
from __future__ import annotations
import numpy as np
from collections import deque
from functools import lru_cache
from typing import List, Optional, Tuple

from constellation import ConstellationState, lla_to_ecef, elevation_deg


class Router:
    """
    Wraps a ConstellationState and provides routing queries.

    Key design: caches per-tick artifacts (adjacency, GS gateways) and
    per-(tick, srcs) BFS results. Cache is bounded by LRU.
    """

    def __init__(self, state: ConstellationState, cfg: dict,
                 gs_lat_lon: np.ndarray):
        """
        gs_lat_lon: (G, 2) array of (lat, lon) for each GS.
        """
        self.state = state
        self.cfg = cfg
        self.N = state.sat_positions_ecef.shape[1]
        self.T = state.sat_positions_ecef.shape[0]
        self.user_min_el = cfg["links"]["user_min_elevation_deg"]
        self.gs_min_el = cfg["links"]["gs_min_elevation_deg"]
        self.R_earth = cfg["physics"]["earth_radius_km"]
        self.c_km_s = cfg["physics"]["speed_of_light_km_s"]

        self.gs_lat_lon = gs_lat_lon                     # (G, 2)
        self.G = gs_lat_lon.shape[0]
        self.gs_ecef = np.array([
            lla_to_ecef(lat, lon, 0.0, self.R_earth)
            for lat, lon in gs_lat_lon
        ])                                               # (G, 3)

        # Per-tick adjacency (dict[int, list[int]]) lazily built
        self._adj_cache: dict = {}
        # Per-tick GS gateways: list of G arrays of sat ids visible to each GS
        self._gs_gateways_cache: dict = {}
        # BFS cache: (tick, tuple(srcs)) -> (hops np.ndarray, pred np.ndarray)
        self._bfs_cache: dict = {}
        # Per-(tick, gs_id) multi-source BFS from all gateways of that GS
        self._gs_bfs_cache: dict = {}

    # ---------- basic helpers ----------
    def tick_of(self, t_s: float) -> int:
        """Snap simulation time to a tick index."""
        dt = self.cfg["simulation"]["tick_s"]
        idx = int(round(t_s / dt))
        return max(0, min(self.T - 1, idx))

    def adjacency(self, tick: int) -> dict:
        if tick in self._adj_cache:
            return self._adj_cache[tick]
        adj = {i: [] for i in range(self.N)}
        for a, b in self.state.isl_edges_per_tick[tick]:
            adj[a].append(b)
            adj[b].append(a)
        self._adj_cache[tick] = adj
        return adj

    def gs_gateways(self, tick: int) -> List[np.ndarray]:
        """For each GS, array of sat ids visible with elevation ≥ min."""
        if tick in self._gs_gateways_cache:
            return self._gs_gateways_cache[tick]
        pos = self.state.sat_positions_ecef[tick]        # (N,3)
        out = []
        for g in range(self.G):
            ge = self.gs_ecef[g]
            # Vectorized elevation computation
            up = ge / np.linalg.norm(ge)
            los = pos - ge[None, :]                      # (N,3)
            rng = np.linalg.norm(los, axis=1)
            los_hat = los / rng[:, None]
            cos_zen = np.clip(los_hat @ up, -1.0, 1.0)
            el = np.rad2deg(np.pi / 2 - np.arccos(cos_zen))
            visible = np.where(el >= self.gs_min_el)[0]
            out.append(visible)
        self._gs_gateways_cache[tick] = out
        return out

    # ---------- visibility ----------
    def nearest_sat_to_ground(self, lat: float, lon: float,
                              tick: int, min_el: float) -> Optional[Tuple[int, float]]:
        """Return (sat_id, slant_km) of nearest visible sat, or None."""
        ground = lla_to_ecef(lat, lon, 0.0, self.R_earth)
        pos = self.state.sat_positions_ecef[tick]
        up = ground / np.linalg.norm(ground)
        los = pos - ground[None, :]
        rng = np.linalg.norm(los, axis=1)
        los_hat = los / rng[:, None]
        cos_zen = np.clip(los_hat @ up, -1.0, 1.0)
        el = np.rad2deg(np.pi / 2 - np.arccos(cos_zen))
        mask = el >= min_el
        if not mask.any():
            return None
        rng_masked = np.where(mask, rng, np.inf)
        best = int(np.argmin(rng_masked))
        return best, float(rng_masked[best])

    # ---------- BFS ----------
    def bfs_multi_source(self, srcs: np.ndarray, tick: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Multi-source BFS on the ISL graph at `tick`.
        Returns (hops (N,), pred (N,)) arrays. hops[i]=-1 if unreachable.
        pred[i] = predecessor sat id on shortest path back to some source; sources have pred = i.
        """
        if len(srcs) == 0:
            hops = -np.ones(self.N, dtype=np.int32)
            pred = -np.ones(self.N, dtype=np.int32)
            return hops, pred
        key = (tick, tuple(sorted(srcs.tolist())))
        if key in self._bfs_cache:
            return self._bfs_cache[key]

        adj = self.adjacency(tick)
        hops = -np.ones(self.N, dtype=np.int32)
        pred = -np.ones(self.N, dtype=np.int32)
        dq = deque()
        for s in srcs:
            s = int(s)
            if hops[s] == -1:
                hops[s] = 0
                pred[s] = s  # self-predecessor marks source
                dq.append(s)
        while dq:
            u = dq.popleft()
            h_u = hops[u]
            for v in adj[u]:
                if hops[v] == -1:
                    hops[v] = h_u + 1
                    pred[v] = u
                    dq.append(v)

        # Bounded LRU
        if len(self._bfs_cache) > 512:
            # Drop oldest-ish entry (dict iteration order)
            old = next(iter(self._bfs_cache))
            self._bfs_cache.pop(old)
        self._bfs_cache[key] = (hops, pred)
        return hops, pred

    def reconstruct_path(self, pred: np.ndarray, dst: int) -> List[int]:
        """Reconstruct path from BFS pred array. Path ends at dst, starts at a source
        (self-predecessor). Returns list from source -> dst."""
        path = [dst]
        cur = dst
        while pred[cur] != cur:
            if pred[cur] == -1:
                return []  # unreachable
            cur = int(pred[cur])
            path.append(cur)
        return list(reversed(path))

    # ---------- high-level: to GS ----------
    def gs_bfs(self, gs_id: int, tick: int) -> Tuple[np.ndarray, np.ndarray]:
        """BFS sourced from all gateways of GS `gs_id` at this tick."""
        key = (tick, gs_id)
        if key in self._gs_bfs_cache:
            return self._gs_bfs_cache[key]
        gateways = self.gs_gateways(tick)[gs_id]
        hops, pred = self.bfs_multi_source(gateways, tick)
        if len(self._gs_bfs_cache) > 2048:
            old = next(iter(self._gs_bfs_cache))
            self._gs_bfs_cache.pop(old)
        self._gs_bfs_cache[key] = (hops, pred)
        return hops, pred

    def route_to_closest_gs(self, src_sat: int, tick: int) -> Optional[dict]:
        """From src_sat at tick, find closest GS (fewest hops) via a single
        global multi-source BFS where every gateway of every GS is a source,
        labelled by its GS id. O(N+E) per tick, regardless of G."""
        hops, pred, gs_label = self._global_gs_bfs(tick)
        if hops[src_sat] < 0:
            return None
        gs_id = int(gs_label[src_sat])
        # Reconstruct path src -> gateway by walking pred chain
        path = [src_sat]
        cur = src_sat
        while pred[cur] != cur:
            cur = int(pred[cur])
            path.append(cur)
        egress = path[-1]
        return {
            "gs_id": gs_id,
            "hops": int(hops[src_sat]),
            "path": path,
            "egress_sat": int(egress),
        }

    def _global_gs_bfs(self, tick: int) -> tuple:
        """Multi-source BFS from ALL gateways of ALL GSes, each tagged with its
        GS id. Returns (hops (N,), pred (N,), gs_label (N,))."""
        key = ("__global__", tick)
        if key in self._gs_bfs_cache:
            return self._gs_bfs_cache[key]
        adj = self.adjacency(tick)
        hops = -np.ones(self.N, dtype=np.int32)
        pred = -np.ones(self.N, dtype=np.int32)
        gs_label = -np.ones(self.N, dtype=np.int32)
        dq = deque()
        gateways_per_gs = self.gs_gateways(tick)
        for gs_id in range(self.G):
            for s in gateways_per_gs[gs_id]:
                s = int(s)
                # Multiple GSes could share a gateway sat; first write wins (closest label by BFS order,
                # but since all these are at hops=0, we just pick the first GS id deterministically).
                if hops[s] == -1:
                    hops[s] = 0
                    pred[s] = s
                    gs_label[s] = gs_id
                    dq.append(s)
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if hops[v] == -1:
                    hops[v] = hops[u] + 1
                    pred[v] = u
                    gs_label[v] = gs_label[u]
                    dq.append(v)
        if len(self._gs_bfs_cache) > 2048:
            old = next(iter(self._gs_bfs_cache))
            self._gs_bfs_cache.pop(old)
        self._gs_bfs_cache[key] = (hops, pred, gs_label)
        return hops, pred, gs_label

    def route_to_specific_gs(self, src_sat: int, gs_id: int, tick: int) -> Optional[dict]:
        """Path from src_sat to the closest gateway of a fixed GS (needed for return)."""
        gateways = self.gs_gateways(tick)[gs_id]
        if len(gateways) == 0:
            return None
        hops, pred = self.gs_bfs(gs_id, tick)
        h = hops[src_sat]
        if h < 0:
            return None
        path = list(reversed(self.reconstruct_path(pred, src_sat)))
        return {
            "gs_id": gs_id,
            "hops": int(h),
            "path": path,
            "egress_sat": int(path[-1]),
        }

    # ---------- return-path helper: best gateway for a given user-sat ----------
    def route_gs_to_sat(self, gs_id: int, user_sat: int,
                        tick: int) -> Optional[dict]:
        """From GS gs_id, find the gateway sat that minimizes ISL hops to
        user_sat, and return the path gw -> ... -> user_sat."""
        gateways = self.gs_gateways(tick)[gs_id]
        if len(gateways) == 0:
            return None
        # BFS from user_sat once, look up hops to every gateway
        srcs = np.array([user_sat], dtype=np.int32)
        hops, pred = self.bfs_multi_source(srcs, tick)
        best_gw = None
        best_h = 10**9
        for gw in gateways:
            h = int(hops[int(gw)])
            if h >= 0 and h < best_h:
                best_h = h
                best_gw = int(gw)
        if best_gw is None:
            return None
        # pred chain goes best_gw -> ... -> user_sat (back via preds)
        path = self.reconstruct_path(pred, best_gw)
        # path is user_sat -> ... -> best_gw; reverse for gw -> user
        path = list(reversed(path))
        return {"gs_id": gs_id, "hops": best_h, "path": path,
                "gateway_sat": best_gw}

    # ---------- sat-to-sat (for Step 2 return path) ----------
    def route_sat_to_sat(self, src_sat: int, dst_sat: int,
                         tick: int) -> Optional[dict]:
        """Shortest path from src_sat to dst_sat at this tick."""
        srcs = np.array([src_sat], dtype=np.int32)
        hops, pred = self.bfs_multi_source(srcs, tick)
        if hops[dst_sat] < 0:
            return None
        path = self.reconstruct_path(pred, dst_sat)
        return {"hops": int(hops[dst_sat]), "path": path}

    # ---------- physics helpers ----------
    def slant_km(self, sat_id: int, tick: int, ground_ecef: np.ndarray) -> float:
        return float(np.linalg.norm(
            self.state.sat_positions_ecef[tick, sat_id] - ground_ecef))

    def isl_km(self, a: int, b: int, tick: int) -> float:
        p = self.state.sat_positions_ecef[tick]
        return float(np.linalg.norm(p[a] - p[b]))

    def prop_delay_s(self, km: float) -> float:
        return km / self.c_km_s
