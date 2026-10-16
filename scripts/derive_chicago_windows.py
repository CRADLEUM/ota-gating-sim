#!/usr/bin/env python3
"""Derive managed-fleet (AV proxy) operation and stationary windows from Chicago taxi trips.

Input: data/raw/chicago_taxi/taxi_<date>.csv, one file per local day, pulled by
pull_data.py from dataset ajtu-isnz with columns taxi_id, trip_start_timestamp,
trip_end_timestamp, trip_seconds, trip_miles, pickup_community_area,
dropoff_community_area.

Human-driven taxis stand in for a robotaxi fleet: each vehicle alternates
between trips (operation) and gaps between trips (stationary). The publisher
rounds trip_start_timestamp and trip_end_timestamp to 15 minutes but reports
trip_seconds exactly, so

    operation window  = trip_seconds / 60                      (exact)
    stationary window = next_start - (start + trip_seconds)    (start rounded, so +-7.5 min)

Gaps that are negative after rounding, or that exceed MAX_GAP_H, are dropped
from the window set (the latter are off-shift periods, and the trace keeps them
anyway). A gap shorter than SHORT_GAP_MIN is a curbside wait between fares,
where only a cellular link reaches the vehicle; a gap of at least DEPOT_GAP_MIN
is treated as a candidate depot or garage stop. Both thresholds are reported so
the reach parameter in gating_sim v4 can be set from the data rather than
assumed.

Outputs (data/derived/chicago_taxi/)
  taxi_windows.csv        one row per window: taxi, kind, start_local, minutes
  taxi_trace.csv          one row per trip: taxi, start_local, end_local, minutes (for replay)
  taxi_summary.json       counts and quantiles, daytime and all-hours, reach shares

Run with `python -I` since the inputs are downloaded files.
"""
import datetime as dt
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "chicago_taxi")
OUT = os.path.join(ROOT, "data", "derived", "chicago_taxi")

MAX_TRIP_MIN = 240.0
MAX_GAP_H = 24.0
SHORT_GAP_MIN = 15.0
DEPOT_GAP_MIN = 120.0
DAY_START, DAY_END = 6, 22
QS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def q(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    out = {f"q{int(p * 100):02d}": round(float(np.quantile(x, p)), 2) for p in QS}
    out.update(n=int(len(x)), mean=round(float(x.mean()), 2))
    return out


def main():
    files = sorted(glob.glob(os.path.join(RAW, "taxi_*.csv")))
    parts = []
    for f in files:
        if os.path.getsize(f) == 0:
            continue
        parts.append(pd.read_csv(f, dtype={"taxi_id": str}))
    d = pd.concat(parts, ignore_index=True)
    n0 = len(d)
    d = d.dropna(subset=["taxi_id", "trip_start_timestamp", "trip_seconds"])
    d["t0"] = pd.to_datetime(d["trip_start_timestamp"], errors="coerce")   # local wall clock
    d["dur"] = d["trip_seconds"].astype(float) / 60.0
    d = d[(d["dur"] > 0) & (d["dur"] <= MAX_TRIP_MIN)].dropna(subset=["t0"])
    d["t1"] = d["t0"] + pd.to_timedelta(d["dur"], unit="m")
    d = d.sort_values(["taxi_id", "t0", "t1"]).reset_index(drop=True)

    # per-taxi gaps
    same = d["taxi_id"].values[1:] == d["taxi_id"].values[:-1]
    gap = (d["t0"].values[1:] - d["t1"].values[:-1]) / np.timedelta64(1, "m")
    gap_start = d["t1"].values[:-1]
    gap_taxi = d["taxi_id"].values[:-1]
    keep = same & (gap >= 0) & (gap <= MAX_GAP_H * 60)
    st = pd.DataFrame({"taxi": gap_taxi[keep], "kind": "stationary",
                       "start_local": gap_start[keep], "minutes": gap[keep]})
    op = pd.DataFrame({"taxi": d["taxi_id"], "kind": "operation",
                       "start_local": d["t0"], "minutes": d["dur"]})
    w = pd.concat([op, st], ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    w.to_csv(os.path.join(OUT, "taxi_windows.csv"), index=False)
    tr = d[["taxi_id", "t0", "t1", "dur"]].copy()
    tr.columns = ["taxi", "start_local", "end_local", "minutes"]
    tr.to_csv(os.path.join(OUT, "taxi_trace.csv"), index=False)

    hour = pd.to_datetime(st["start_local"]).dt.hour
    day = st[(hour >= DAY_START) & (hour < DAY_END)]
    s = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "files": [os.path.basename(f) for f in files],
        "date_first": str(d["t0"].min().date()), "date_last": str(d["t0"].max().date()),
        "rows_raw": int(n0), "trips_used": int(len(d)), "taxis": int(d["taxi_id"].nunique()),
        "trips_per_taxi_day_median": float(d.groupby(["taxi_id", d["t0"].dt.date]).size().median()),
        "thresholds": {"max_trip_min": MAX_TRIP_MIN, "max_gap_h": MAX_GAP_H,
                       "short_gap_min": SHORT_GAP_MIN, "depot_gap_min": DEPOT_GAP_MIN,
                       "daytime_hours": [DAY_START, DAY_END]},
        "operation_min": q(op["minutes"]),
        "stationary_all_min": q(st["minutes"]),
        "stationary_daytime_min": q(day["minutes"]),
        "daytime_gap_share_lt_short": round(float((day["minutes"] < SHORT_GAP_MIN).mean()), 3),
        "daytime_gap_share_ge_depot": round(float((day["minutes"] >= DEPOT_GAP_MIN).mean()), 3),
        "daytime_gap_time_share_ge_depot": round(float(day.loc[day["minutes"] >= DEPOT_GAP_MIN, "minutes"].sum()
                                                       / day["minutes"].sum()), 3),
        "daytime_gap_share_ge_30": round(float((day["minutes"] >= 30).mean()), 3),
        "daytime_gap_share_ge_60": round(float((day["minutes"] >= 60).mean()), 3),
        "note": "trip starts are rounded to 15 min by the publisher; trip_seconds is exact",
    }
    with open(os.path.join(OUT, "taxi_summary.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)
    print(f"trips={s['trips_used']:,} taxis={s['taxis']} {s['date_first']}..{s['date_last']}")
    print("operation  ", s["operation_min"])
    print("stat (day) ", s["stationary_daytime_min"])
    print("stat (all) ", s["stationary_all_min"])
    print(f"daytime gaps: <{SHORT_GAP_MIN:.0f} min {s['daytime_gap_share_lt_short']:.2f}, "
          f">=30 {s['daytime_gap_share_ge_30']:.2f}, >=60 {s['daytime_gap_share_ge_60']:.2f}, "
          f">={DEPOT_GAP_MIN:.0f} {s['daytime_gap_share_ge_depot']:.2f} "
          f"(time share {s['daytime_gap_time_share_ge_depot']:.2f})")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
