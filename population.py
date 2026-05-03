
from __future__ import annotations
import os
import numpy as np


# (lat, lon, metro_pop_millions)  -- order-of-magnitude city/metro populations
URBAN_CENTERS = [
    # Asia
    (35.68, 139.69, 37.0),   # Tokyo
    (28.61,  77.21, 32.0),   # Delhi
    (31.23, 121.47, 28.0),   # Shanghai
    (23.13, 113.26, 26.0),   # Guangzhou
    (19.08,  72.88, 24.0),   # Mumbai
    (39.90, 116.41, 22.0),   # Beijing
    (23.81,  90.41, 22.0),   # Dhaka
    (30.57, 104.07, 21.0),   # Chengdu
    (13.76, 100.50, 17.0),   # Bangkok
    (14.60, 120.98, 17.0),   # Manila
    (37.57, 126.98, 25.0),   # Seoul
    (22.32, 114.17,  8.0),   # Hong Kong
    (25.03, 121.57,  7.0),   # Taipei
    ( 1.35, 103.82,  6.0),   # Singapore
    (22.54, 114.06, 17.0),   # Shenzhen
    (13.08,  80.27, 11.0),   # Chennai
    (12.97,  77.59, 13.0),   # Bangalore
    (17.38,  78.49, 10.0),   # Hyderabad
    (22.57,  88.36, 15.0),   # Kolkata
    (18.52,  73.85,  6.5),   # Pune
    (24.86,  67.01, 16.0),   # Karachi
    (33.69,  73.05,  2.3),   # Islamabad
    (31.55,  74.36, 13.0),   # Lahore
    (10.82, 106.63,  9.0),   # HCMC
    (21.03, 105.85,  9.0),   # Hanoi
    ( 3.14, 101.69,  8.0),   # Kuala Lumpur
    (-6.21, 106.85, 11.0),   # Jakarta
    (34.69, 135.50, 19.0),   # Osaka
    (35.18, 136.91,  9.5),   # Nagoya
    (41.01,  28.98, 15.5),   # Istanbul
    (39.93,  32.87,  5.7),   # Ankara
    (30.30,  77.30,  5.0),   # Dehradun region
    (29.66,  52.58,  1.6),   # Shiraz
    (35.69,  51.39,  9.3),   # Tehran
    (24.47,  39.61,  1.5),   # Medina
    (24.71,  46.68,  8.0),   # Riyadh
    (25.20,  55.27,  3.5),   # Dubai
    (29.38,  47.99,  4.1),   # Kuwait
    (33.51,  36.29,  2.5),   # Damascus
    (31.77,  35.22,  1.0),   # Jerusalem
    (32.08,  34.78,  4.4),   # Tel Aviv
    (33.31,  44.36,  8.0),   # Baghdad
    (51.17,  71.45,  1.2),   # Astana
    (43.25,  76.95,  2.0),   # Almaty

    # Europe
    (51.51,  -0.13, 14.5),   # London
    (48.86,   2.35, 13.0),   # Paris
    (55.75,  37.62, 17.3),   # Moscow
    (59.93,  30.34,  5.4),   # St Petersburg
    (52.52,  13.40,  6.1),   # Berlin
    (50.11,   8.68,  3.0),   # Frankfurt
    (48.14,  11.58,  6.2),   # Munich
    (52.37,   4.90,  2.5),   # Amsterdam
    (41.90,  12.50,  4.3),   # Rome
    (45.46,   9.19,  4.3),   # Milan
    (40.42,  -3.70,  6.7),   # Madrid
    (41.39,   2.17,  5.6),   # Barcelona
    (38.72,  -9.14,  2.9),   # Lisbon
    (50.85,   4.35,  2.5),   # Brussels
    (48.21,  16.37,  2.9),   # Vienna
    (52.23,  21.01,  3.1),   # Warsaw
    (53.35,  -6.26,  1.9),   # Dublin
    (47.37,   8.54,  1.4),   # Zurich
    (59.33,  18.07,  2.4),   # Stockholm
    (60.17,  24.94,  1.3),   # Helsinki
    (59.91,  10.75,  1.6),   # Oslo
    (55.68,  12.57,  2.0),   # Copenhagen
    (37.98,  23.73,  3.7),   # Athens
    (44.43,  26.10,  2.3),   # Bucharest
    (50.08,  14.44,  1.3),   # Prague
    (47.50,  19.04,  1.8),   # Budapest
    (53.42,  -2.98,  2.8),   # Liverpool/Manchester area
    (55.86,  -4.25,  1.8),   # Glasgow
    (43.61,   3.88,  0.6),   # Montpellier
    (45.76,   4.83,  2.3),   # Lyon
    (43.30,   5.37,  1.9),   # Marseille
    (50.45,  30.52,  3.0),   # Kyiv
    (46.47,  30.74,  1.0),   # Odesa

    # Africa
    (30.04,  31.24, 22.0),   # Cairo
    ( 6.52,   3.38, 15.0),   # Lagos
    ( 9.03,  38.74,  5.0),   # Addis Ababa
    (-1.29,  36.82,  5.0),   # Nairobi
    (-26.20, 28.04,  6.2),   # Johannesburg
    (-33.92, 18.42,  4.7),   # Cape Town
    ( 3.84,  11.50,  4.0),   # Yaoundé
    ( 4.05,   9.77,  3.7),   # Douala
    (-4.44,  15.27, 15.0),   # Kinshasa
    ( 5.60,  -0.19,  4.5),   # Accra
    (14.69, -17.45,  3.1),   # Dakar
    (33.57,  -7.59,  4.4),   # Casablanca
    (36.80,  10.18,  2.7),   # Tunis
    (36.75,   3.06,  3.4),   # Algiers
    (-8.84,  13.23,  8.7),   # Luanda
    (-15.41, 28.29,  3.3),   # Lusaka
    (-25.75, 28.19,  2.5),   # Pretoria
    ( 6.13,   1.22,  1.8),   # Lome
    (31.63,  -8.01,  1.3),   # Marrakech

    # North America
    (40.71, -74.01, 20.1),   # NYC
    (34.05,-118.24, 13.2),   # LA
    (41.88, -87.63,  9.5),   # Chicago
    (29.76, -95.37,  7.1),   # Houston
    (33.45,-112.07,  4.9),   # Phoenix
    (32.78, -96.80,  7.6),   # Dallas
    (39.74,-104.99,  2.9),   # Denver
    (47.61,-122.33,  4.0),   # Seattle
    (37.77,-122.42,  4.7),   # SF
    (38.90, -77.04,  6.3),   # DC
    (25.76, -80.19,  6.2),   # Miami
    (33.75, -84.39,  6.1),   # Atlanta
    (42.36, -71.06,  4.9),   # Boston
    (39.95, -75.17,  6.2),   # Philadelphia
    (42.33, -83.05,  4.3),   # Detroit
    (45.50, -73.57,  4.3),   # Montreal
    (43.65, -79.38,  6.4),   # Toronto
    (49.28,-123.12,  2.6),   # Vancouver
    (51.05,-114.07,  1.6),   # Calgary
    (19.43, -99.13, 22.0),   # Mexico City
    (20.67,-103.35,  5.3),   # Guadalajara
    (25.67,-100.32,  5.3),   # Monterrey
    (21.16, -86.85,  0.9),   # Cancun
    (14.63, -90.51,  3.0),   # Guatemala City
    ( 8.98, -79.52,  1.9),   # Panama City
    (22.24, -80.07,  0.5),   # central Cuba
    (23.13, -82.37,  2.1),   # Havana
    (18.47, -69.95,  3.3),   # Santo Domingo
    (18.40, -66.06,  2.5),   # San Juan

    # South America
    (-23.55, -46.63, 22.0),  # Sao Paulo
    (-22.91, -43.17, 13.5),  # Rio
    (-15.79, -47.88,  4.6),  # Brasilia
    (-12.97, -38.50,  3.9),  # Salvador
    ( -8.05, -34.88,  4.0),  # Recife
    ( -3.73, -38.52,  4.1),  # Fortaleza
    ( -3.10, -60.02,  2.2),  # Manaus
    (-30.03, -51.23,  4.3),  # Porto Alegre
    (-25.43, -49.27,  3.7),  # Curitiba
    (-34.61, -58.38, 15.0),  # Buenos Aires
    (-31.41, -64.18,  1.6),  # Cordoba
    (-33.45, -70.67,  7.0),  # Santiago
    (-12.05, -77.04, 10.8),  # Lima
    ( 4.71, -74.07, 11.0),   # Bogota
    ( 6.24, -75.57,  4.0),   # Medellin
    (10.48, -66.90,  2.9),   # Caracas
    ( 3.44, -76.52,  2.8),   # Cali
    (-0.22, -78.51,  2.0),   # Quito
    (-2.17, -79.92,  3.1),   # Guayaquil
    (-17.78, -63.18,  2.0),  # Santa Cruz BOL
    (-16.50, -68.15,  1.9),  # La Paz
    (-25.30, -57.58,  2.5),  # Asuncion
    (-34.90, -56.19,  1.9),  # Montevideo

    # Oceania
    (-33.87, 151.21,  5.4),  # Sydney
    (-37.81, 144.96,  5.1),  # Melbourne
    (-27.47, 153.03,  2.6),  # Brisbane
    (-31.95, 115.86,  2.2),  # Perth
    (-34.93, 138.60,  1.4),  # Adelaide
    (-36.85, 174.76,  1.7),  # Auckland
    (-41.29, 174.78,  0.4),  # Wellington
    (-43.53, 172.64,  0.4),  # Christchurch

    # Northern high-latitude
    (64.13, -21.90,  0.2),   # Reykjavik
    (61.22,-149.90,  0.4),   # Anchorage
    (64.84,-147.72,  0.1),   # Fairbanks
    (60.47,  8.47,   0.4),   # Central Norway
    (69.65, 18.96,   0.1),   # Tromso
]


def _great_circle_km(lat1, lon1, lat2, lon2, R=6371.0):
    """Vectorized great-circle distance. lat*/lon* in degrees."""
    lat1r = np.deg2rad(lat1)
    lon1r = np.deg2rad(lon1)
    lat2r = np.deg2rad(lat2)
    lon2r = np.deg2rad(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c


def build_synthetic_population(floor: float = 1000.0,
                               sigma_km: float = 120.0,
                               rural_baseline: float = 800.0) -> np.ndarray:
    """Return (180, 360) array, people/cell, lat index 0 = +89.5, lon index 0 = -179.5."""
    # Cell center latitudes: 89.5 .. -89.5  (north-to-south for convenient plotting)
    lat_centers = np.arange(89.5, -90.0, -1.0)    # (180,)
    lon_centers = np.arange(-179.5, 180.0, 1.0)   # (360,)
    lat_grid, lon_grid = np.meshgrid(lat_centers, lon_centers, indexing="ij")

    pop = np.zeros_like(lat_grid, dtype=np.float64)

    for lat_c, lon_c, pop_m in URBAN_CENTERS:
        d = _great_circle_km(lat_grid, lon_grid, lat_c, lon_c)
        # Gaussian smear; the kernel integral approximately equals pop_m million
        # Normalization is rough — we only need a plausible map shape.
        kernel = np.exp(-(d ** 2) / (2 * sigma_km ** 2))
        # Normalize kernel so sum matches city population (in people)
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            pop += kernel * (pop_m * 1e6 / kernel_sum)

    # Rural baseline — weakly modulated by |latitude| (less at poles)
    lat_factor = np.cos(np.deg2rad(lat_grid))
    rural = rural_baseline * np.clip(lat_factor, 0.05, 1.0)
    pop += rural

    # Apply floor (ships, remote-user users, etc.)
    pop = np.maximum(pop, floor)
    return pop


def load_or_build_population(cfg: dict) -> np.ndarray:
    """Load cached population grid, or build & cache."""
    popcfg = cfg["population"]
    path = popcfg["data_file"]
    if os.path.exists(path):
        return np.load(path)

    if popcfg["source"] == "synthetic":
        pop = build_synthetic_population(
            floor=popcfg["ocean_floor_people_per_cell"],
            sigma_km=120.0,
            rural_baseline=800.0,
        )
    elif popcfg["source"] == "gpwv4":
        raise NotImplementedError(
            "GPWv4 loader: drop a (180,360) .npy at the configured data_file path, "
            "or implement loader here."
        )
    else:
        raise ValueError(f"Unknown population source: {popcfg['source']}")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, pop)
    return pop


def cell_centers():
    """Return (lat_centers (180,), lon_centers (360,))."""
    return np.arange(89.5, -90.0, -1.0), np.arange(-179.5, 180.0, 1.0)


if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    pop = load_or_build_population(cfg)
    print(f"Population grid: {pop.shape}, total = {pop.sum()/1e9:.2f} billion")
    print(f"Min/Max: {pop.min():.0f} / {pop.max():.0f}  people/cell")
    print(f"Median: {np.median(pop):.0f}")
