#!/usr/bin/env python3
"""Derive drone operation and dock (stationary) windows from DFR flight logs.

Input: per-agency CSVs from AH Datalytics in data/raw/dfr/ with columns
takeoff_utc, landing_utc, duration_min, purpose, source. There is no aircraft or
dock identifier, and in the busy agencies a third or more of consecutive flights
overlap in time, so the raw agency-level gap between one landing and the next
takeoff is not a per-aircraft window.

Method
  1. Clean: drop flights with missing times, non-positive duration, or duration
     above MAX_FLIGHT_MIN (battery-limited airframes; longer rows are data errors
     or tethered operations).
  2. Fleet lower bound: K_min = maximum number of flights in progress at any
     instant (the depth of the interval set). No agency can have flown these
     logs with fewer aircraft.
  3. Assignment: walk flights in takeoff order and give each to the free virtual
     aircraft that has been idle longest (earliest landing). With K_min aircraft
     this is the busiest schedule consistent with the log, so the per-aircraft
     gaps it yields are a LOWER bound on true dock windows. A second pass with
     2 x K_min aircraft shows how much the distribution moves when the fleet is
     larger than the minimum; the truth lies between the two and above the
     agency-level gap.
  4. Windows: operation window = flight duration; stationary window = gap from an
     aircraft's landing to its next takeoff. Gaps are also split into daytime
     (window starts 06:00 to 22:00 local) and all-hours, because overnight gaps
     dominate the upper tail and the sim horizon is six hours.
  5. Replay traces: per aircraft, the ordered (takeoff, landing) pairs in local
     time, for the deterministic replay in gating_sim v4.

Outputs (data/derived/dfr/)
  <agency>_windows.csv     one row per window: aircraft, kind, start_local, minutes, fleet
  <agency>_trace.csv       one row per flight: aircraft, takeoff_local, landing_local (K_min fleet)
  dfr_summary.json         per-agency counts, K_min, and quantiles for every window set

Run with `python -I` since the inputs are downloaded files.
"""
import argparse
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "dfr")
OUT = os.path.join(ROOT, "data", "derived", "dfr")

MAX_FLIGHT_MIN = 90.0
DAY_START, DAY_END = 6, 22   # local hours for the daytime cut
QS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

# Local zones for the agencies pulled. Add a row when adding an agency.
TZ = {
    "cincinnati-pd": "America/New_York",
    "wichita-pd": "America/Chicago",
    "denver-pd": "America/Denver",
    "castle-rock-pd-co": "America/Denver",
    "cspd-real-time-crime-center": "America/Denver",
}


def load(slug):
    d = pd.read_csv(os.path.join(RAW, f"{slug}.csv"))
    n0 = len(d)
    d["t0"] = pd.to_datetime(d["takeoff_utc"], utc=True, errors="coerce")
    d["t1"] = pd.to_datetime(d["landing_utc"], utc=True, errors="coerce")
    d = d.dropna(subset=["t0", "t1"])
    d["dur"] = (d["t1"] - d["t0"]).dt.total_seconds() / 60.0
    d = d[(d["dur"] > 0) & (d["dur"] <= MAX_FLIGHT_MIN)]
    d = d.sort_values(["t0", "t1"]).reset_index(drop=True)
    d["t0_local"] = d["t0"].dt.tz_convert(TZ[slug])
    d["t1_local"] = d["t1"].dt.tz_convert(TZ[slug])
    return d, n0


def max_concurrency(d):
    ev = np.concatenate([
        np.stack([d["t0"].values.astype("datetime64[ns]").astype(np.int64), np.ones(len(d))], 1),
        np.stack([d["t1"].values.astype("datetime64[ns]").astype(np.int64), -np.ones(len(d))], 1),
    ])
    # landings before takeoffs at the same instant
    ev = ev[np.lexsort((ev[:, 1], ev[:, 0]))]
    return int(np.cumsum(ev[:, 1]).max())


def assign(d, k):
    """Give each flight to the free virtual aircraft idle the longest."""
    free_at = np.full(k, np.datetime64("1970-01-01T00:00:00.000000000"), dtype="datetime64[ns]")
    t0 = d["t0"].values.astype("datetime64[ns]")
    t1 = d["t1"].values.astype("datetime64[ns]")
    who = np.empty(len(d), dtype=int)
    for i in range(len(d)):
        cand = np.where(free_at <= t0[i])[0]
        if len(cand) == 0:          # cannot happen when k >= max concurrency
            j = int(np.argmin(free_at))
        else:
            j = int(cand[np.argmin(free_at[cand])])
        who[i] = j
        free_at[j] = t1[i]
    return who


def windows(d, who, fleet_label):
    rows = []
    d = d.assign(aircraft=who)
    for a, g in d.groupby("aircraft"):
        g = g.sort_values("t0")
        for _, r in g.iterrows():
            rows.append((a, "operation", r["t0_local"], r["dur"], fleet_label))
        gap = (g["t0"].values[1:] - g["t1"].values[:-1]) / np.timedelta64(1, "m")
        for s, m in zip(g["t1_local"].values[:-1], gap):
            rows.append((a, "stationary", pd.Timestamp(s), float(m), fleet_label))
    w = pd.DataFrame(rows, columns=["aircraft", "kind", "start_local", "minutes", "fleet"])
    return w


def q(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    out = {f"q{int(p * 100):02d}": round(float(np.quantile(x, p)), 2) for p in QS}
    out.update(n=int(len(x)), mean=round(float(x.mean()), 2))
    return out


def summarize(w, label):
    s = {}
    op = w[w.kind == "operation"]["minutes"]
    st = w[w.kind == "stationary"]
    hour = pd.to_datetime(st["start_local"]).dt.hour
    day = st[(hour >= DAY_START) & (hour < DAY_END)]["minutes"]
    s["operation_min"] = q(op)
    s["stationary_all_min"] = q(st["minutes"])
    s["stationary_daytime_min"] = q(day)
    # how many stationary windows would already fit a 30-minute session
    s["stationary_daytime_share_ge_30min"] = round(float((day >= 30).mean()), 3) if len(day) else None
    s["stationary_daytime_share_ge_60min"] = round(float((day >= 60).mean()), 3) if len(day) else None
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agencies", nargs="+", default=list(TZ))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    summary = {"generated_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
               "max_flight_min": MAX_FLIGHT_MIN, "daytime_hours": [DAY_START, DAY_END],
               "quantiles": QS, "agencies": {}}
    for slug in a.agencies:
        d, n0 = load(slug)
        k = max_concurrency(d)
        agency_gap = (d["t0"].values[1:] - d["t1"].values[:-1]) / np.timedelta64(1, "m")
        who_min = assign(d, k)
        who_2k = assign(d, 2 * k)
        w_min = windows(d, who_min, "kmin")
        w_2k = windows(d, who_2k, "2kmin")
        w = pd.concat([w_min, w_2k], ignore_index=True)
        w.to_csv(os.path.join(OUT, f"{slug}_windows.csv"), index=False)
        tr = d.assign(aircraft=who_min)[["aircraft", "t0_local", "t1_local", "dur", "purpose"]]
        tr.columns = ["aircraft", "takeoff_local", "landing_local", "minutes", "purpose"]
        tr.to_csv(os.path.join(OUT, f"{slug}_trace.csv"), index=False)
        entry = {
            "rows_raw": int(n0), "rows_used": int(len(d)),
            "date_first": str(d["t0_local"].min().date()), "date_last": str(d["t0_local"].max().date()),
            "source": d["source"].value_counts().to_dict(),
            "fleet_lower_bound_kmin": k,
            "share_consecutive_flights_overlapping": round(float((agency_gap < 0).mean()), 3),
            "agency_level_gap_min": q(agency_gap[agency_gap >= 0]),
            "kmin_fleet": summarize(w_min, "kmin"),
            "two_kmin_fleet": summarize(w_2k, "2kmin"),
        }
        summary["agencies"][slug] = entry
        km, k2 = entry["kmin_fleet"], entry["two_kmin_fleet"]
        print(f"{slug:30s} n={len(d):6d} Kmin={k:3d} overlap={entry['share_consecutive_flights_overlapping']:.2f} "
              f"op med={km['operation_min']['q50']:5.1f}  stat(day) med Kmin={km['stationary_daytime_min']['q50']:6.1f} "
              f"2Kmin={k2['stationary_daytime_min']['q50']:6.1f}  P(>=30|Kmin)={km['stationary_daytime_share_ge_30min']:.2f}")
    with open(os.path.join(OUT, "dfr_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
