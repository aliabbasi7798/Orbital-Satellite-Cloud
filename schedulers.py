"""
schedulers.py
-------------
K-hop and K×K scheduling algorithms for satellite compute.

Both take in: ingress sat id, request FLOPs, current queue state, routing.
Both return: (compute_sat_id, num_forward_hops, list_of_path_sats)

Oracle model: the scheduler sees exact current queue remaining-work across
all relevant sats at decision time. Signaling cost is ignored (per v1 spec).
"""

from __future__ import annotations
import numpy as np
from collections import deque
from typing import Tuple, List, Optional


# ============================================================
# K-hop
# ============================================================
def khop_candidates(router, ingress_sat: int, tick: int,
                    K: int) -> List[Tuple[int, int]]:
    """
    BFS from ingress_sat over the ISL graph at this tick.
    Returns list of (sat_id, hop_distance) for sats reachable in <= K hops.
    """
    adj = router.adjacency(tick)
    N = router.N
    hops = -np.ones(N, dtype=np.int32)
    hops[ingress_sat] = 0
    dq = deque([ingress_sat])
    out = [(ingress_sat, 0)]
    while dq:
        u = dq.popleft()
        if hops[u] >= K:
            continue
        for v in adj[u]:
            if hops[v] == -1:
                hops[v] = hops[u] + 1
                out.append((int(v), int(hops[v])))
                dq.append(v)
    return out


def khop_choose(router, ingress_sat: int, tick: int, K: int,
                flops: float, sat_tflops: float,
                queue_remaining: np.ndarray,
                hop_delay_s: float) -> Tuple[int, int]:
    """
    Cost-aware K-hop scheduler.
    For each candidate c within K hops of ingress_sat, compute:
        cost(c) = hops(ingress, c) * hop_delay_s
                + queue_remaining[c]
                + flops / sat_tflops
    Return (compute_sat_id, hops_to_it).

    Service time term is the same for all candidates (cancels in the argmin),
    but we include it so the returned cost is interpretable if you want it
    later.
    """
    cands = khop_candidates(router, ingress_sat, tick, K)
    svc = flops / sat_tflops
    best_sat = ingress_sat
    best_hops = 0
    best_cost = float("inf")
    for sat_id, h in cands:
        cost = h * hop_delay_s + queue_remaining[sat_id] + svc
        if cost < best_cost:
            best_cost = cost
            best_sat = sat_id
            best_hops = h
    return best_sat, best_hops


# ============================================================
# K×K hierarchical
# ============================================================
class KxKRegions:
    """
    Partition the Walker-Delta grid (P planes × S sats/plane) into rectangular
    regions of K × K sats (K planes × K sats-per-plane).

    Ragged edges at the end are allowed — the last row/column of regions may
    be smaller than K×K.

    Leader sat = top-left corner of each region (lowest plane index, lowest
    within-plane index). The leader is also a worker (computes its share).
    """
    def __init__(self, num_planes: int, sats_per_plane: int, K: int):
        self.P = num_planes
        self.S = sats_per_plane
        self.K = K
        self.N = num_planes * sats_per_plane

        # region_id[sat] -> index of the sat's region
        # region_members[rid] -> list of sat ids in that region
        # region_leader[rid] -> sat id of the leader
        # sat_leader[sat] -> leader sat id (for fast lookup from ingress)
        self.region_id = np.zeros(self.N, dtype=np.int32)
        self.region_members: List[List[int]] = []
        self.region_leader: List[int] = []
        self.sat_leader = np.zeros(self.N, dtype=np.int32)

        num_plane_groups = (num_planes + K - 1) // K
        num_sat_groups   = (sats_per_plane + K - 1) // K
        rid = 0
        for pg in range(num_plane_groups):
            for sg in range(num_sat_groups):
                members = []
                for dp in range(K):
                    p = pg * K + dp
                    if p >= num_planes: continue
                    for ds in range(K):
                        m = sg * K + ds
                        if m >= sats_per_plane: continue
                        sat_id = p * sats_per_plane + m
                        members.append(sat_id)
                        self.region_id[sat_id] = rid
                if not members: continue
                self.region_members.append(members)
                leader = members[0]  # top-left corner
                self.region_leader.append(leader)
                for sat_id in members:
                    self.sat_leader[sat_id] = leader
                rid += 1

        self.num_regions = rid


def kxk_choose(regions: KxKRegions, ingress_sat: int,
               queue_remaining: np.ndarray,
               flops: float, sat_tflops: float) -> Tuple[int, int, int, int]:
    """
    Two-level hierarchical scheduling:
      1. Ingress sat forwards the request to its region leader (0 hops if
         ingress == leader).
      2. Leader picks the min-queue-remaining worker in its region.

    Returns (leader_sat, compute_sat, hops_ingress_to_leader,
             hops_leader_to_compute).

    NOTE: the leader picks by pure queue_remaining (not cost-aware).
    Adding hops-inside-region to the cost is easy but changes the algorithm
    character — keep the "region-bounded scheduler" semantics clean for v1.
    """
    leader = int(regions.sat_leader[ingress_sat])
    region = regions.region_members[int(regions.region_id[ingress_sat])]
    # Pick worker with min queue_remaining in the region
    q = np.array([queue_remaining[s] for s in region])
    idx = int(np.argmin(q))
    compute = int(region[idx])
    # Hops inside region (Walker grid): we use simple Manhattan on (plane, in_plane)
    # Ingress->leader and leader->compute. Leader routing sat uses BFS at the
    # decision tick; we return a placeholder count for now and let the
    # simulator resolve exact paths via BFS.
    return leader, compute, -1, -1
    # (hops unused for now; simulator will call router.route_sat_to_sat for
    # actual ISL traversal)
