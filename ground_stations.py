
from __future__ import annotations
import os
import csv
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class GroundStation:
    gs_id: int
    name: str
    lat: float
    lon: float
    country: str


# ---- 20 GS: regional backbones ----
GS_20 = [
    ("Hawthorne",        33.92,  -118.33, "USA"),
    ("Redmond",          47.67,  -122.12, "USA"),
    ("McAllen",          26.20,   -98.23, "USA"),
    ("Buffalo",          42.89,   -78.86, "USA"),
    ("Quincy",           47.23,  -119.86, "USA"),
    ("Merrillan",        44.45,   -90.83, "USA"),
    ("Sao_Paulo",       -23.55,   -46.63, "BRA"),
    ("London",           51.51,    -0.13, "GBR"),
    ("Frankfurt",        50.11,     8.68, "DEU"),
    ("Madrid",           40.42,    -3.70, "ESP"),
    ("Johannesburg",    -26.20,    28.04, "ZAF"),
    ("Lagos",             6.52,     3.38, "NGA"),
    ("Nairobi",          -1.29,    36.82, "KEN"),
    ("Dubai",            25.20,    55.27, "ARE"),
    ("Mumbai",           19.08,    72.88, "IND"),
    ("Singapore",         1.35,   103.82, "SGP"),
    ("Tokyo",            35.68,   139.69, "JPN"),
    ("Sydney",          -33.87,   151.21, "AUS"),
    ("Perth",           -31.95,   115.86, "AUS"),
    ("Santiago",        -33.45,   -70.67, "CHL"),
]

# ---- 100 GS: approximated Starlink gateway + major metro footprint ----
# Sources: publicly catalogued Starlink gateway cities + large population centers
# to fill coverage gaps. (These are not exact Starlink coordinates — they are
# plausible city-level proxies.)
GS_100 = [
    # North America (30)
    ("Hawthorne_CA",     33.92, -118.33, "USA"),
    ("Redmond_WA",       47.67, -122.12, "USA"),
    ("Quincy_WA",        47.23, -119.86, "USA"),
    ("McAllen_TX",       26.20,  -98.23, "USA"),
    ("Boca_Chica_TX",    25.99,  -97.19, "USA"),
    ("Merrillan_WI",     44.45,  -90.83, "USA"),
    ("Buffalo_NY",       42.89,  -78.86, "USA"),
    ("Hibbing_MN",       47.43,  -92.94, "USA"),
    ("Butte_MT",         46.00, -112.53, "USA"),
    ("Gaylord_MI",       45.03,  -84.67, "USA"),
    ("Greenville_PA",    41.40,  -80.39, "USA"),
    ("Litchfield_ME",    44.20,  -69.97, "USA"),
    ("Charleston_SC",    32.78,  -79.93, "USA"),
    ("Miami_FL",         25.76,  -80.19, "USA"),
    ("Atlanta_GA",       33.75,  -84.39, "USA"),
    ("Dallas_TX",        32.78,  -96.80, "USA"),
    ("Houston_TX",       29.76,  -95.37, "USA"),
    ("Phoenix_AZ",       33.45, -112.07, "USA"),
    ("Denver_CO",        39.74, -104.99, "USA"),
    ("Chicago_IL",       41.88,  -87.63, "USA"),
    ("New_York_NY",      40.71,  -74.01, "USA"),
    ("Los_Angeles_CA",   34.05, -118.24, "USA"),
    ("San_Francisco_CA", 37.77, -122.42, "USA"),
    ("Seattle_WA",       47.61, -122.33, "USA"),
    ("Toronto_ON",       43.65,  -79.38, "CAN"),
    ("Montreal_QC",      45.50,  -73.57, "CAN"),
    ("Vancouver_BC",     49.28, -123.12, "CAN"),
    ("Calgary_AB",       51.05, -114.07, "CAN"),
    ("Mexico_City",      19.43,  -99.13, "MEX"),
    ("Guadalajara",      20.67, -103.35, "MEX"),
    # South America (8)
    ("Sao_Paulo",       -23.55,  -46.63, "BRA"),
    ("Rio",             -22.91,  -43.17, "BRA"),
    ("Brasilia",        -15.79,  -47.88, "BRA"),
    ("Buenos_Aires",    -34.61,  -58.38, "ARG"),
    ("Santiago",        -33.45,  -70.67, "CHL"),
    ("Lima",            -12.05,  -77.04, "PER"),
    ("Bogota",            4.71,  -74.07, "COL"),
    ("Caracas",          10.48,  -66.90, "VEN"),
    # Europe (20)
    ("London",           51.51,   -0.13, "GBR"),
    ("Dublin",           53.35,   -6.26, "IRL"),
    ("Paris",            48.86,    2.35, "FRA"),
    ("Madrid",           40.42,   -3.70, "ESP"),
    ("Barcelona",        41.39,    2.17, "ESP"),
    ("Lisbon",           38.72,   -9.14, "PRT"),
    ("Milan",            45.46,    9.19, "ITA"),
    ("Rome",             41.90,   12.50, "ITA"),
    ("Frankfurt",        50.11,    8.68, "DEU"),
    ("Berlin",           52.52,   13.40, "DEU"),
    ("Munich",           48.14,   11.58, "DEU"),
    ("Amsterdam",        52.37,    4.90, "NLD"),
    ("Brussels",         50.85,    4.35, "BEL"),
    ("Zurich",           47.37,    8.54, "CHE"),
    ("Vienna",           48.21,   16.37, "AUT"),
    ("Warsaw",           52.23,   21.01, "POL"),
    ("Stockholm",        59.33,   18.07, "SWE"),
    ("Oslo",             59.91,   10.75, "NOR"),
    ("Helsinki",         60.17,   24.94, "FIN"),
    ("Athens",           37.98,   23.73, "GRC"),
    # Africa (10)
    ("Johannesburg",    -26.20,   28.04, "ZAF"),
    ("Cape_Town",       -33.92,   18.42, "ZAF"),
    ("Lagos",             6.52,    3.38, "NGA"),
    ("Nairobi",          -1.29,   36.82, "KEN"),
    ("Cairo",            30.04,   31.24, "EGY"),
    ("Casablanca",       33.57,   -7.59, "MAR"),
    ("Addis_Ababa",       9.03,   38.74, "ETH"),
    ("Accra",             5.60,   -0.19, "GHA"),
    ("Kinshasa",         -4.44,   15.27, "COD"),
    ("Dakar",            14.69,  -17.45, "SEN"),
    # Middle East (5)
    ("Dubai",            25.20,   55.27, "ARE"),
    ("Riyadh",           24.71,   46.68, "SAU"),
    ("Istanbul",         41.01,   28.98, "TUR"),
    ("Tel_Aviv",         32.08,   34.78, "ISR"),
    ("Tehran",           35.69,   51.39, "IRN"),
    # Asia (18)
    ("Mumbai",           19.08,   72.88, "IND"),
    ("Delhi",            28.61,   77.21, "IND"),
    ("Bangalore",        12.97,   77.59, "IND"),
    ("Chennai",          13.08,   80.27, "IND"),
    ("Karachi",          24.86,   67.01, "PAK"),
    ("Dhaka",            23.81,   90.41, "BGD"),
    ("Bangkok",          13.76,  100.50, "THA"),
    ("Ho_Chi_Minh",      10.82,  106.63, "VNM"),
    ("Manila",           14.60,  120.98, "PHL"),
    ("Jakarta",          -6.21,  106.85, "IDN"),
    ("Singapore",         1.35,  103.82, "SGP"),
    ("Kuala_Lumpur",      3.14,  101.69, "MYS"),
    ("Hong_Kong",        22.32,  114.17, "HKG"),
    ("Taipei",           25.03,  121.57, "TWN"),
    ("Seoul",            37.57,  126.98, "KOR"),
    ("Tokyo",            35.68,  139.69, "JPN"),
    ("Osaka",            34.69,  135.50, "JPN"),
    ("Beijing",          39.90,  116.41, "CHN"),
    # Oceania (5)
    ("Sydney",          -33.87,  151.21, "AUS"),
    ("Melbourne",       -37.81,  144.96, "AUS"),
    ("Perth",           -31.95,  115.86, "AUS"),
    ("Brisbane",        -27.47,  153.03, "AUS"),
    ("Auckland",        -36.85,  174.76, "NZL"),
    # High-latitude / polar fill (4)
    ("Reykjavik",        64.13,  -21.90, "ISL"),
    ("Anchorage",        61.22, -149.90, "USA"),
    ("Nuuk",             64.18,  -51.72, "GRL"),
    ("Punta_Arenas",    -53.16,  -70.92, "CHL"),
]


def write_gs_files(data_dir: str = "data"):
    """Write the two GS CSVs to disk."""
    os.makedirs(data_dir, exist_ok=True)
    for name, stations in [("gs_20.csv", GS_20), ("gs_100.csv", GS_100)]:
        path = os.path.join(data_dir, name)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gs_id", "name", "lat", "lon", "country"])
            for i, (nm, lat, lon, c) in enumerate(stations):
                w.writerow([i, nm, lat, lon, c])
        print(f"Wrote {path}  ({len(stations)} stations)")


def load_ground_stations(path: str) -> List[GroundStation]:
    out = []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(GroundStation(
                gs_id=int(row["gs_id"]),
                name=row["name"],
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                country=row["country"],
            ))
    return out


if __name__ == "__main__":
    write_gs_files()
