#!/usr/bin/env python3
"""Derive the gossip contact rate beta from the Cabspotting taxi traces.

Input: data/raw/cabspotting/cabspottingdata.tar.gz (CRAWDAD epfl/mobility,
doi:10.15783/C7J010; about 500 San Francisco taxis, 17 May to 10 June 2008,
one GPS fix every few seconds). Each cab file new_<id>.txt has lines
"lat lon occupancy unixtime". The archive is read in place; nothing is
extracted to disk.

What beta means in gating_sim. The gossip architecture uses the mean-field SI
curve I(t) = I0 / (I0 + (1 - I0) e^{-beta t}). With N nodes and a pairwise
contact rate lambda per pair, dI/dt = lambda N I (1 - I), so beta = lambda N is
the rate at which one node meets any other node, in contacts per minute, with
transmission assumed on every contact. That makes the value below an upper
bound for a fleet of this size and density; the sim sweeps beta downward.

Method
  1. Resample every cab to a common grid of STEP seconds by holding the last
     fix (gaps longer than MAX_GAP seconds count as off-line).
  2. At each grid time, find every pair of on-line cabs within range R using a
     flat-earth projection; a contact event starts when a pair comes within R
     and ends when it leaves.
  3. beta(R) = 2 x (contact events) / (sum over steps of on-line cabs) / STEP
     in minutes, i.e. contacts per on-line node per minute. Also report the
     contact-duration quantiles, the daytime-only rate, and the per-day spread.

Ranges: 100 m and 300 m bracket DSRC and C-V2X sidelink line-of-sight use in
dense streets; 1,000 m is an optimistic bound.

Outputs (data/derived/cabspotting/)
  contact_rates.json     beta per range, duration quantiles, per-day values, and the
                         duration-gated daytime rates (contacts of at least 1 to 20 min)
  contact_durations.csv  contact-duration histogram per range (STEP-second bins, overflow at 1 h)

Run with `python -I` since the input is a downloaded archive. Takes a few
minutes for the full trace set.
"""
import argparse
import datetime as dt
import json
import os
import tarfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "cabspotting", "cabspottingdata.tar.gz")
OUT = os.path.join(ROOT, "data", "derived", "cabspotting")

STEP = 30          # seconds
MAX_GAP = 300      # seconds without a fix = off-line
RANGES_M = (100.0, 300.0, 1000.0)
GATE_MINUTES = (1, 2, 5, 10, 20)   # minimum contact length for the duration-gated rates
TZ_OFFSET_H = -7   # San Francisco, PDT in May/June 2008
DAY_START, DAY_END = 6, 22
QS = [0.1, 0.25, 0.5, 0.75, 0.9]


def iter_traces(path):
    """Yield (cab_id, array[lat, lon, occ, t]) one cab at a time, sorted by time.
    Parsed with numpy's C-level reader on the raw bytes; nothing is kept."""
    with tarfile.open(path, "r:gz") as tf:
        for m in tf.getmembers():
            name = os.path.basename(m.name)
            if not (m.isfile() and name.startswith("new_") and name.endswith(".txt")):
                continue
            raw = tf.extractfile(m).read()
            flat = np.fromstring(raw.decode("ascii", "ignore"), dtype=float, sep=" ")
            if flat.size < 4:
                continue
            a = flat[: (flat.size // 4) * 4].reshape(-1, 4)
            a = a[np.argsort(a[:, 3])]
            yield name[4:-4], a


def time_span(path):
    t_min, t_max, lat, lon, n = np.inf, -np.inf, [], [], 0
    for _, a in iter_traces(path):
        t_min, t_max = min(t_min, a[0, 3]), max(t_max, a[-1, 3])
        lat.append(np.median(a[:, 0])); lon.append(np.median(a[:, 1]))
        n += 1
    return t_min, t_max, float(np.median(lat)), float(np.median(lon)), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=RAW)
    ap.add_argument("--max-cabs", type=int, default=None, help="subsample for a quick run")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t_min, t_max, lat0, lon0, n_all = time_span(a.path)
    n = min(n_all, a.max_cabs) if a.max_cabs else n_all
    grid = np.arange(np.floor(t_min / STEP) * STEP, t_max + STEP, STEP)
    T = len(grid)
    print(f"{n} cabs, grid {T:,} steps of {STEP}s, {(t_max - t_min) / 86400:.1f} days", flush=True)

    # flat-earth projection around the median position
    kx = 111_320.0 * np.cos(np.radians(lat0))
    ky = 110_540.0

    # one pass over the archive: resample each cab onto the grid and drop its trace
    X = np.full((n, T), np.nan, dtype=np.float32)
    Y = np.full((n, T), np.nan, dtype=np.float32)
    ids = []
    for k, (cab, tr) in enumerate(iter_traces(a.path)):
        if k >= n:
            break
        ids.append(cab)
        idx = np.searchsorted(tr[:, 3], grid, side="right") - 1
        ok = idx >= 0
        idx = np.clip(idx, 0, len(tr) - 1)
        ok &= (grid - tr[idx, 3]) <= MAX_GAP
        X[k, ok] = ((tr[idx, 1] - lon0) * kx)[ok]
        Y[k, ok] = ((tr[idx, 0] - lat0) * ky)[ok]
        del tr
    online = ~np.isnan(X)
    online_steps = online.sum(0)                     # cabs on-line per step
    print(f"mean on-line cabs per step: {online_steps.mean():.0f}")

    local_hour = ((grid + TZ_OFFSET_H * 3600) % 86400) / 3600.0
    is_day = (local_hour >= DAY_START) & (local_hour < DAY_END)
    day_index = ((grid + TZ_OFFSET_H * 3600) // 86400).astype(int)
    day_index -= day_index.min()

    results = {"generated_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
               "source": "CRAWDAD epfl/mobility (Cabspotting), doi:10.15783/C7J010",
               "cabs": n, "step_s": STEP, "max_gap_s": MAX_GAP, "days": float((t_max - t_min) / 86400),
               "mean_online_cabs_per_step": float(online_steps.mean()),
               "beta_definition": "contacts per on-line node per minute; transmission on every contact",
               "ranges": {}}
    # Contact durations go into a histogram (STEP-second bins up to one hour, plus
    # overflow) rather than a per-event list, which is what blew the memory before.
    bins = np.arange(0, 3600 + STEP, STEP)
    iu = np.triu_indices(n, 1)
    hist_rows = []
    for R in RANGES_M:
        prev = np.zeros(len(iu[0]), dtype=bool)
        start = np.zeros(len(iu[0]), dtype=np.int32)
        hist = np.zeros(len(bins), dtype=np.int64)      # last bin = overflow
        events_total = 0
        events_day = 0
        node_steps_day = 0
        per_day_events = {}
        per_day_nodes = {}
        for t in range(T):
            x, y = X[:, t], Y[:, t]
            dx = x[iu[0]] - x[iu[1]]
            dy = y[iu[0]] - y[iu[1]]
            close = (dx * dx + dy * dy) <= R * R        # NaN compares False
            new = close & ~prev
            ended = prev & ~close
            cnt = int(new.sum())
            events_total += cnt
            d = int(day_index[t])
            per_day_events[d] = per_day_events.get(d, 0) + cnt
            per_day_nodes[d] = per_day_nodes.get(d, 0) + int(online_steps[t])
            if is_day[t]:
                events_day += cnt
                node_steps_day += int(online_steps[t])
            start[new] = t
            if ended.any():
                secs = (t - start[ended]) * STEP
                hist += np.bincount(np.minimum(secs // STEP, len(bins) - 1), minlength=len(bins))
            prev = close
        node_steps = int(online_steps.sum())
        beta_all = 2.0 * events_total / node_steps / (STEP / 60.0) if node_steps else float("nan")
        beta_day = 2.0 * events_day / node_steps_day / (STEP / 60.0) if node_steps_day else float("nan")
        per_day = [2.0 * per_day_events[d] / per_day_nodes[d] / (STEP / 60.0)
                   for d in sorted(per_day_events) if per_day_nodes[d] > 0]
        total = int(hist.sum())
        cum = np.cumsum(hist) / max(total, 1)
        def hq(p):
            return float(bins[min(int(np.searchsorted(cum, p)), len(bins) - 1)])
        # Duration gate: a whole-image transfer needs a contact at least as long as the
        # transfer, so the usable rate is beta times the share of contacts that long.
        gated = {f"ge_{m}min": (round(beta_day * float(hist[bins >= 60 * m].sum() / total), 5) if total else None)
                 for m in GATE_MINUTES}
        results["ranges"][f"{R:.0f}m"] = {
            "contact_events": events_total,
            "beta_per_min_all_hours": round(beta_all, 5),
            "beta_per_min_daytime": round(beta_day, 5),
            "beta_per_min_per_day_q10_q50_q90": [round(float(v), 5) for v in np.quantile(per_day, [0.1, 0.5, 0.9])] if per_day else None,
            "contact_duration_s": ({f"q{int(p * 100)}": hq(p) for p in QS} if total else None),
            "share_contacts_ge_60s": (round(float(hist[bins >= 60].sum() / total), 3) if total else None),
            "beta_per_min_daytime_gated": gated,
            "duration_bin_s": STEP, "duration_overflow_s": int(bins[-1]),
        }
        for b, c in zip(bins, hist):
            hist_rows.append((R, int(b), int(c)))
        print(f"R={R:5.0f} m  events={events_total:,}  beta(all)={beta_all:.4f}/min  beta(day)={beta_day:.4f}/min  "
              f"dur q50={hq(0.5) if total else 'n/a'} s", flush=True)
    with open(os.path.join(OUT, "contact_rates.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUT, "contact_durations.csv"), "w", encoding="utf-8") as f:
        f.write("range_m,duration_bin_start_s,events\n")
        for r, b, c in hist_rows:
            f.write(f"{r:.0f},{b},{c}\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
