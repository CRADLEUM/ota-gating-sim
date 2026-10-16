#!/usr/bin/env python3
"""Build the compact inputs that gating_sim.py (v4) reads, from data/derived/.

The per-window and per-trip derived files are large (the taxi set is 270 MB),
so they stay out of git. This script reduces them to what the simulator needs
and writes small CSVs into data/sim_inputs/, which is committed. A reader can
regenerate them by running the derive_*.py scripts and then this one.

Outputs (data/sim_inputs/)
  windows_drone_<agency>.csv   kind,minutes   operation = flight durations (all hours);
                               stationary = dock windows, K_min fleet, daytime start
  windows_av_chicago.csv       kind,minutes   operation = trip durations; stationary =
                               gaps between trips, daytime start
  replay_drone_<agency>.csv    device,start_min,dur_min   every flight, K_min aircraft index,
                               minutes since local midnight of the first day
  replay_av_chicago.csv        device,start_min,dur_min   trips of a 500-taxi random sample
  contacts_cabspotting.csv     range_m,duration_bin_start_s,events   measured contact-length histogram
  contacts_cabspotting.json    contact rates per range and the histogram's provenance
  README.json                  counts, seeds, and the source summaries' generation stamps

Window sets larger than MAX_WINDOWS are subsampled without replacement with a
fixed seed. Daytime means the window starts between DAY_START and DAY_END local.
"""
import datetime as dt
import json
import os
import shutil

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DER = os.path.join(ROOT, "data", "derived")
OUT = os.path.join(ROOT, "data", "sim_inputs")

SEED = 20260705
MAX_WINDOWS = 50_000
AV_SAMPLE_TAXIS = 500
AGENCIES = ["cincinnati-pd", "wichita-pd", "denver-pd", "castle-rock-pd-co", "cspd-real-time-crime-center"]
DAY_START, DAY_END = 6, 22


def hour_of(s):
    """Local hour from a 'YYYY-MM-DD HH:MM:SS[.ffffff][+zz:zz]' string."""
    return s.str.slice(11, 13).astype(int)


def subsample(rng, x):
    x = np.asarray(x, dtype=float)
    if len(x) > MAX_WINDOWS:
        x = rng.choice(x, MAX_WINDOWS, replace=False)
    return np.sort(x)


def positive(op, st):
    """A window is a positive interval. A zero-length gap is two adjoining trips,
    not a stop, and a zero-length flight or trip is a record with no duration;
    neither is a window. Returns the filtered sets and the number dropped."""
    op, st = np.asarray(op, dtype=float), np.asarray(st, dtype=float)
    dropped = int((op <= 0).sum() + (st <= 0).sum())
    return op[op > 0], st[st > 0], dropped


def write_windows(path, op, st):
    with open(path, "w", encoding="utf-8") as f:
        f.write("kind,minutes\n")
        for v in op:
            f.write(f"operation,{v:.2f}\n")
        for v in st:
            f.write(f"stationary,{v:.2f}\n")


def abs_minutes(s):
    """Minutes since local midnight of the earliest day, from naive local timestamps."""
    t = pd.to_datetime(s.str.slice(0, 19), format="%Y-%m-%d %H:%M:%S")
    day0 = t.min().normalize()
    return ((t - day0).dt.total_seconds() / 60.0).values, str(day0.date())


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    info = {"generated_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
            "seed": SEED, "max_windows": MAX_WINDOWS, "daytime_hours": [DAY_START, DAY_END],
            "av_sample_taxis": AV_SAMPLE_TAXIS, "sets": {}}

    for slug in AGENCIES:
        w = pd.read_csv(os.path.join(DER, "dfr", f"{slug}_windows.csv"))
        w = w[w["fleet"] == "kmin"]
        op = w.loc[w["kind"] == "operation", "minutes"].values
        st = w[w["kind"] == "stationary"]
        h = hour_of(st["start_local"].astype(str))
        st = st.loc[(h >= DAY_START) & (h < DAY_END), "minutes"].values
        op, st, dropped = positive(op, st)
        op_s, st_s = subsample(rng, op), subsample(rng, st)
        write_windows(os.path.join(OUT, f"windows_drone_{slug}.csv"), op_s, st_s)
        tr = pd.read_csv(os.path.join(DER, "dfr", f"{slug}_trace.csv"))
        start, day0 = abs_minutes(tr["takeoff_local"].astype(str))
        rp = pd.DataFrame({"device": tr["aircraft"].astype(int), "start_min": np.round(start, 2),
                           "dur_min": np.round(tr["minutes"].values, 2)}).sort_values(["device", "start_min"])
        rp.to_csv(os.path.join(OUT, f"replay_drone_{slug}.csv"), index=False)
        info["sets"][f"drone_{slug}"] = {
            "nonpositive_dropped": dropped,
            "operation_n": int(len(op)), "operation_kept": int(len(op_s)),
            "stationary_daytime_n": int(len(st)), "stationary_kept": int(len(st_s)),
            "op_median": round(float(np.median(op)), 2), "stat_median": round(float(np.median(st)), 2),
            "replay_flights": int(len(rp)), "replay_devices": int(rp["device"].nunique()),
            "replay_day0": day0, "replay_days": int(np.ceil(rp["start_min"].max() / 1440.0))}
        print(f"{slug:30s} op {len(op):6d} -> {len(op_s):6d}   stat(day) {len(st):6d} -> {len(st_s):6d}   "
              f"replay {len(rp):6d} flights / {rp['device'].nunique()} aircraft")

    w = pd.read_csv(os.path.join(DER, "chicago_taxi", "taxi_windows.csv"), dtype={"taxi": str})
    op = w.loc[w["kind"] == "operation", "minutes"].values
    st = w[w["kind"] == "stationary"]
    h = hour_of(st["start_local"].astype(str))
    st = st.loc[(h >= DAY_START) & (h < DAY_END), "minutes"].values
    op, st, dropped = positive(op, st)
    op_s, st_s = subsample(rng, op), subsample(rng, st)
    write_windows(os.path.join(OUT, "windows_av_chicago.csv"), op_s, st_s)
    tr = pd.read_csv(os.path.join(DER, "chicago_taxi", "taxi_trace.csv"), dtype={"taxi": str})
    taxis = np.sort(tr["taxi"].unique())
    pick = rng.choice(taxis, min(AV_SAMPLE_TAXIS, len(taxis)), replace=False)
    idx = {t: i for i, t in enumerate(np.sort(pick))}
    tr = tr[tr["taxi"].isin(idx)].copy()
    start, day0 = abs_minutes(tr["start_local"].astype(str))
    rp = pd.DataFrame({"device": tr["taxi"].map(idx).astype(int), "start_min": np.round(start, 2),
                       "dur_min": np.round(tr["minutes"].values, 2)}).sort_values(["device", "start_min"])
    rp.to_csv(os.path.join(OUT, "replay_av_chicago.csv"), index=False)
    info["sets"]["av_chicago"] = {
        "nonpositive_dropped": dropped,
        "operation_n": int(len(op)), "operation_kept": int(len(op_s)),
        "stationary_daytime_n": int(len(st)), "stationary_kept": int(len(st_s)),
        "op_median": round(float(np.median(op)), 2), "stat_median": round(float(np.median(st)), 2),
        "replay_trips": int(len(rp)), "replay_devices": int(rp["device"].nunique()),
        "replay_day0": day0, "replay_days": int(np.ceil(rp["start_min"].max() / 1440.0))}
    print(f"{'av_chicago':30s} op {len(op):6d} -> {len(op_s):6d}   stat(day) {len(st):6d} -> {len(st_s):6d}   "
          f"replay {len(rp):6d} trips / {rp['device'].nunique()} taxis")

    # Measured contact-length histogram and contact rates (Cabspotting), for the gossip
    # path: the simulator gates the contact rate by the contact length a transfer needs.
    cab_dir = os.path.join(DER, "cabspotting")
    if os.path.exists(os.path.join(cab_dir, "contact_rates.json")):
        with open(os.path.join(cab_dir, "contact_rates.json"), encoding="utf-8") as f:
            cab = json.load(f)
        shutil.copyfile(os.path.join(cab_dir, "contact_durations.csv"), os.path.join(OUT, "contacts_cabspotting.csv"))
        meta = {"source": cab["source"], "generated_utc": cab["generated_utc"], "cabs": cab["cabs"],
                "days": round(cab["days"], 2), "step_s": cab["step_s"], "max_gap_s": cab["max_gap_s"],
                "mean_online_cabs_per_step": round(cab["mean_online_cabs_per_step"], 1),
                "beta_definition": cab["beta_definition"],
                "beta_per_min_daytime": {r: e["beta_per_min_daytime"] for r, e in cab["ranges"].items()},
                "beta_per_min_all_hours": {r: e["beta_per_min_all_hours"] for r, e in cab["ranges"].items()},
                "histogram": "contacts_cabspotting.csv: range_m, duration_bin_start_s (bin covers [start, start + step_s); the last bin is overflow), events"}
        with open(os.path.join(OUT, "contacts_cabspotting.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        info["sets"]["contacts_cabspotting"] = {"cabs": cab["cabs"], "days": round(cab["days"], 1),
                                                "ranges": list(cab["ranges"]), "contact_events": {r: e["contact_events"] for r, e in cab["ranges"].items()}}
        print(f"{'contacts_cabspotting':30s} {cab['cabs']} cabs, {cab['days']:.1f} days, ranges {', '.join(cab['ranges'])}")
    else:
        print("contacts_cabspotting: derivation not run, skipped")

    with open(os.path.join(OUT, "README.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
