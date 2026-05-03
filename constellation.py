"""
constellation.py
----------------
Analytic Keplerian propagation for a Walker-Delta shell, plus a +Grid ISL
topology that gates seam links by geometric slant range.

Why Kepler (not SGP4): we define the shell synthetically (not from TLEs), so
J2/drag perturbations are not part of the model we're studying. Kepler is
exact for circular orbits and ~10x faster. The Hypatia satgen tool does the
same thing.

Output:
    sat_positions_ecef: np.ndarray shape (T, N, 3)   km, ECEF (rotates with Earth)
    isl_edges_per_tick: list of T sets, each set of (i,j) active ISL pairs
    plane_of_sat:       np.ndarray shape (N,)        which plane each sat belongs to
    idx_in_plane:       np.ndarray shape (N,)        index within plane

Convention: sat id s = plane * sats_per_plane + idx_in_plane.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class ConstellationState:
    sat_positions_ecef: np.ndarray   # (T, N, 3) km
    isl_edges_per_tick: list         # list[T] of list of (i,j) tuples (i<j)
    plane_of_sat: np.ndarray         # (N,)
    idx_in_plane: np.ndarray         # (N,)
    times_s: np.ndarray              # (T,)
    num_planes: int
    sats_per_plane: int


def build_constellation(cfg: dict) -> ConstellationState:
    """Build full constellation state over the simulation horizon."""
    con = cfg["constellation"]
    phys = cfg["physics"]
    sim = cfg["simulation"]

    N = con["num_sats"]
    P = con["num_planes"]
    S = con["sats_per_plane"]
    assert N == P * S, "num_sats must equal num_planes * sats_per_plane"

    R_earth = phys["earth_radius_km"]
    mu = phys["mu_earth_km3_s2"]
    alt = con["altitude_km"]
    inc = np.deg2rad(con["inclination_deg"])
    F = con["walker_phase_factor"]
    M0_offset = np.deg2rad(con["epoch_offset_deg"])

    a = R_earth + alt                       # semi-major axis km
    n = np.sqrt(mu / a ** 3)                # mean motion rad/s
    T_orbit = 2 * np.pi / n
    earth_omega = 7.2921159e-5              # rad/s (sidereal)

    dt = sim["tick_s"]
    duration = sim["duration_s"]
    times = np.arange(0.0, duration + dt, dt)
    T = len(times)

    # --- Satellite identifiers ---
    plane_of_sat = np.repeat(np.arange(P), S)
    idx_in_plane = np.tile(np.arange(S), P)

    # --- RAAN per plane (evenly spread over 360°) ---
    raan_per_plane = np.linspace(0.0, 2 * np.pi, P, endpoint=False)

    # --- Mean anomaly per sat (Walker-Delta phasing) ---
    # M(p, m) = m * 2π/S + p * F * 2π/N + M0_offset
    p_arr = plane_of_sat
    m_arr = idx_in_plane
    M0 = (m_arr * (2 * np.pi / S)
          + p_arr * F * (2 * np.pi / N)
          + M0_offset)

    # --- Propagate ---
    # r in ECI orbital plane: (a cos E, a sin E, 0) — for circular e=0, E = M
    positions = np.zeros((T, N, 3), dtype=np.float64)
    for t_idx, t in enumerate(times):
        M = M0 + n * t                                          # (N,)
        # Position in orbital plane (argument of periapsis = 0 for circular)
        x_op = a * np.cos(M)
        y_op = a * np.sin(M)
        # Rotate: first by inclination about x-axis, then by RAAN about z-axis
        cos_i, sin_i = np.cos(inc), np.sin(inc)
        raan = raan_per_plane[p_arr]                            # (N,)
        cos_O, sin_O = np.cos(raan), np.sin(raan)
        # After inclination rotation: (x_op, y_op*cos_i, y_op*sin_i)
        x1 = x_op
        y1 = y_op * cos_i
        z1 = y_op * sin_i
        # Then RAAN rotation about z:
        x_eci = x1 * cos_O - y1 * sin_O
        y_eci = x1 * sin_O + y1 * cos_O
        z_eci = z1
        # ECI -> ECEF by rotating about z by -theta (GMST advance)
        theta = earth_omega * t
        cos_th, sin_th = np.cos(theta), np.sin(theta)
        x_ecef = x_eci * cos_th + y_eci * sin_th
        y_ecef = -x_eci * sin_th + y_eci * cos_th
        z_ecef = z_eci
        positions[t_idx, :, 0] = x_ecef
        positions[t_idx, :, 1] = y_ecef
        positions[t_idx, :, 2] = z_ecef

    # --- Build +Grid ISL topology ---
    # 2 intra-plane (fore/aft): (plane, m) <-> (plane, m+1 mod S)
    # 2 inter-plane (port/starboard): (plane, m) <-> (plane+1 mod P, m)
    # Seam: between plane P-1 and plane 0 (cross-seam) — gated by range.
    isl_max_range = con["isl_max_range_km"]

    base_edges = []  # (i, j) where i<j, plus a flag: "intra" | "inter" | "seam"
    for p in range(P):
        for m in range(S):
            i = p * S + m
            # Fore (intra): next sat in plane
            m_next = (m + 1) % S
            j = p * S + m_next
            a_, b_ = (i, j) if i < j else (j, i)
            base_edges.append((a_, b_, "intra"))
            # Starboard (inter): next plane, same m
            p_next = (p + 1) % P
            j2 = p_next * S + m
            a_, b_ = (i, j2) if i < j2 else (j2, i)
            kind = "seam" if p_next == 0 and p == P - 1 else "inter"
            base_edges.append((a_, b_, kind))

    # Deduplicate (each edge generated twice by the loop)
    seen = set()
    edges_unique = []
    for a_, b_, k in base_edges:
        if (a_, b_) in seen:
            continue
        seen.add((a_, b_))
        edges_unique.append((a_, b_, k))

    # Gate edges by range per tick
    isl_edges_per_tick = []
    for t_idx in range(T):
        p_t = positions[t_idx]                              # (N, 3)
        active = []
        for a_, b_, kind in edges_unique:
            d = np.linalg.norm(p_t[a_] - p_t[b_])
            if d <= isl_max_range:
                active.append((a_, b_))
            # Intra-plane links should always satisfy this at 540 km; kept
            # for safety.
        isl_edges_per_tick.append(active)

    return ConstellationState(
        sat_positions_ecef=positions,
        isl_edges_per_tick=isl_edges_per_tick,
        plane_of_sat=plane_of_sat,
        idx_in_plane=idx_in_plane,
        times_s=times,
        num_planes=P,
        sats_per_plane=S,
    )


def lla_to_ecef(lat_deg: float, lon_deg: float, alt_km: float,
                earth_radius_km: float = 6371.0) -> np.ndarray:
    """Spherical-Earth LLA -> ECEF. Adequate for geometry; not WGS84."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    r = earth_radius_km + alt_km
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.array([x, y, z])


def elevation_deg(ground_ecef: np.ndarray, sat_ecef: np.ndarray,
                  earth_radius_km: float = 6371.0) -> float:
    """Elevation angle from ground point to satellite (spherical Earth)."""
    # Local up vector at ground point
    up = ground_ecef / np.linalg.norm(ground_ecef)
    los = sat_ecef - ground_ecef
    range_km = np.linalg.norm(los)
    los_hat = los / range_km
    # elevation = 90 - angle(up, los)
    cos_zenith = float(np.dot(up, los_hat))
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith = np.arccos(cos_zenith)
    return float(np.rad2deg(np.pi / 2 - zenith))
