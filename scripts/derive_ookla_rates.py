#!/usr/bin/env python3
"""Fixed and mobile download rates from Ookla Speedtest open data tiles.

Input: the two quarterly tile parquet files in data/raw/ookla/ (type=fixed and
type=mobile). Each row is one zoom-16 web-mercator tile (about 610 m on a side
at the equator) with the average download speed of the Speedtest results in
that tile, and the number of tests and devices behind that average.

The paper uses the ratio of the local (depot or dock, fixed broadband) delivery
rate to the wide-area (cellular) backhaul rate. This script reports, per region,
the test-weighted median of tile-average download speed for fixed and for
mobile, and the ratio. The test-weighted median is the closest reproducible
analogue of a per-test median from tile aggregates. The unweighted tile median
is reported beside it.

Caveats to carry into the paper
  - User-initiated tests on consumer devices, not vehicle or dock modems.
  - Fixed tiles include satellite subscribers; Ookla does not split them out.
  - License CC BY-NC-SA 4.0, attribution: "Ookla Speedtest Open Data".

Outputs (data/derived/ookla/)
  ookla_rates.json   per region: n tiles, n tests, weighted and unweighted medians, ratio
  ookla_rates.csv    same, flat

Run with `python -I` since the inputs are downloaded files.
"""
import datetime as dt
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ookla")
OUT = os.path.join(ROOT, "data", "derived", "ookla")

# lon_min, lat_min, lon_max, lat_max
REGIONS = {
    "conus": (-125.0, 24.5, -66.9, 49.5),
    "chicago_metro": (-88.3, 41.45, -87.5, 42.2),
    "cincinnati": (-84.9, 38.95, -84.2, 39.40),
    "wichita": (-97.6, 37.50, -97.1, 37.85),
    "denver": (-105.25, 39.55, -104.7, 39.95),
    "colorado_springs": (-104.95, 38.70, -104.6, 39.00),
    "castle_rock": (-104.95, 39.30, -104.75, 39.45),
    "san_francisco": (-122.55, 37.70, -122.35, 37.85),
    # Rural proxy for a remote dock site: the state of Wyoming, which has no
    # metro area above 70,000 people. Cheyenne and Casper are included.
    "rural_proxy_wyoming": (-111.05, 41.0, -104.05, 45.0),
}


def quadkey_to_lonlat(qk):
    """Tile-centre lon/lat for an array of equal-length quadkey strings."""
    z = len(qk[0])
    digits = np.frombuffer("".join(qk).encode("ascii"), dtype=np.uint8).reshape(-1, z) - 48
    weights = (1 << np.arange(z - 1, -1, -1)).astype(np.int64)
    x = ((digits & 1).astype(np.int64) * weights).sum(1)
    y = (((digits >> 1) & 1).astype(np.int64) * weights).sum(1)
    n = float(1 << z)
    lon = (x + 0.5) / n * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + 0.5) / n))))
    return lon, lat


def wmedian(v, w):
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, 0.5 * c[-1])])


def load(kind):
    path = sorted(glob.glob(os.path.join(RAW, f"*_performance_{kind}_tiles.parquet")))[-1]
    t = pq.read_table(path, columns=["quadkey", "avg_d_kbps", "avg_u_kbps", "tests", "devices"])
    qk = t.column("quadkey").to_pylist()
    lon, lat = quadkey_to_lonlat(qk)
    return os.path.basename(path), {
        "lon": lon, "lat": lat,
        "d": t.column("avg_d_kbps").to_numpy().astype(float) / 1000.0,   # Mbps
        "u": t.column("avg_u_kbps").to_numpy().astype(float) / 1000.0,
        "tests": t.column("tests").to_numpy().astype(float),
        "devices": t.column("devices").to_numpy().astype(float),
    }


def stats(d, m):
    return {
        "tiles": int(m.sum()),
        "tests": int(d["tests"][m].sum()),
        "down_mbps_median_tile": round(float(np.median(d["d"][m])), 1),
        "down_mbps_median_test_weighted": round(wmedian(d["d"][m], d["tests"][m]), 1),
        "up_mbps_median_test_weighted": round(wmedian(d["u"][m], d["tests"][m]), 1),
        "down_mbps_p10_test_weighted": round(float(np.quantile(np.repeat(d["d"][m], np.minimum(d["tests"][m], 50).astype(int)), 0.10)), 1),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    files, data = {}, {}
    for kind in ("fixed", "mobile"):
        files[kind], data[kind] = load(kind)
        print(f"{kind}: {files[kind]}  {len(data[kind]['d']):,} tiles")
    out = {"generated_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
           "files": files, "license": "CC BY-NC-SA 4.0, Ookla Speedtest Open Data",
           "regions": {}}
    rows = []
    for name, (x0, y0, x1, y1) in REGIONS.items():
        r = {}
        for kind in ("fixed", "mobile"):
            d = data[kind]
            m = (d["lon"] >= x0) & (d["lon"] <= x1) & (d["lat"] >= y0) & (d["lat"] <= y1)
            r[kind] = stats(d, m)
        f, mo = r["fixed"]["down_mbps_median_test_weighted"], r["mobile"]["down_mbps_median_test_weighted"]
        r["ratio_fixed_to_mobile_down_test_weighted"] = round(f / mo, 2)
        r["bbox"] = [x0, y0, x1, y1]
        out["regions"][name] = r
        rows.append((name, r["fixed"]["tiles"], r["fixed"]["tests"], f, r["mobile"]["tiles"], r["mobile"]["tests"], mo,
                     r["ratio_fixed_to_mobile_down_test_weighted"]))
        print(f"{name:18s} fixed {f:7.1f} Mbps ({r['fixed']['tests']:>9,} tests)   mobile {mo:7.1f} Mbps "
              f"({r['mobile']['tests']:>9,} tests)   ratio {f / mo:5.2f}")
    with open(os.path.join(OUT, "ookla_rates.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    with open(os.path.join(OUT, "ookla_rates.csv"), "w", encoding="utf-8") as fh:
        fh.write("region,fixed_tiles,fixed_tests,fixed_down_mbps_wmedian,mobile_tiles,mobile_tests,mobile_down_mbps_wmedian,ratio\n")
        for row in rows:
            fh.write(",".join(str(v) for v in row) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
