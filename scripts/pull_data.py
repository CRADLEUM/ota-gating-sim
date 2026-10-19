#!/usr/bin/env python3
"""Pull the public datasets behind the v4 measured inputs.

Each source lands in data/raw/<source>/ (ignored by git) and every file gets a
row in data/raw/manifest.json with its URL, byte count, SHA-256, and pull time in
UTC. The derive_*.py scripts turn these raw files into the values that go into
the paper, so a reader can regenerate every measured Table I entry from a fresh
pull and compare hashes.

Sources
    dfr      AH Datalytics law-enforcement drone flight logs, one CSV per agency
    chicago  City of Chicago taxi trips (dataset ajtu-isnz, 2024 onward), paged
             one day at a time through the Socrata API
    ookla    Ookla Speedtest open data, fixed and mobile tile parquet for one
             quarter (CC BY-NC-SA 4.0, attribution required)
    nhtsa    NHTSA recall flat file (post-2010), quarterly completion reports,
             and the field specification

Cabspotting (CRAWDAD epfl/mobility) is behind an IEEE DataPort login and is
downloaded by hand into data/raw/cabspotting/; register it with `manifest`.

Usage
    python scripts/pull_data.py dfr --agencies cincinnati-pd wichita-pd
    python scripts/pull_data.py chicago --from 2025-03-03 --to 2025-03-17
    python scripts/pull_data.py ookla --year 2026 --quarter 2
    python scripts/pull_data.py nhtsa
    python scripts/pull_data.py manifest --source cabspotting --note "..."

Standard library only.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw")
MANIFEST = os.path.join(RAW, "manifest.json")
UA = "ota-gating-sim/v4 (research data pull; github.com/CRADLEUM/ota-gating-sim)"

DFR_URL = "https://drones.ahdatalytics.com/data/csv/{slug}.csv"
CHICAGO_URL = "https://data.cityofchicago.org/resource/ajtu-isnz.csv"
CHICAGO_COLS = ("taxi_id,trip_start_timestamp,trip_end_timestamp,trip_seconds,"
                "trip_miles,pickup_community_area,dropoff_community_area")
CHICAGO_PAGE = 50000
OOKLA_URL = ("https://ookla-open-data.s3.amazonaws.com/parquet/performance/"
             "type={kind}/year={year}/quarter={quarter}/"
             "{year}-{month}-01_performance_{kind}_tiles.parquet")
NHTSA_URLS = [
    "https://static.nhtsa.gov/odi/ffdd/rcl/FLAT_RCL_POST_2010.zip",
    "https://static.nhtsa.gov/odi/ffdd/rcl/FLAT_RCL_Qrtly_Rpts.zip",
    "https://static.nhtsa.gov/odi/ffdd/rcl/RCL.txt",
    "https://static.nhtsa.gov/odi/ffdd/rcl/RCL_Qtrly_Rpts.txt",
    "https://static.nhtsa.gov/odi/ffdd/rcl/FLAT_RCL_Annual_Rpts.zip",
    "https://static.nhtsa.gov/odi/ffdd/rcl/RCL_Annual_Rpts.txt",
    "https://static.nhtsa.gov/odi/ffdd/rcl/Import_Instructions_Recalls.pdf",
]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest():
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as f:
            return json.load(f)
    return {"note": "raw data provenance; files themselves are not in git",
            "entries": []}


def _record(source, path, url, note="", extra=None):
    """Add or replace the manifest row for one raw file."""
    m = _load_manifest()
    rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
    row = {
        "source": source,
        "file": rel,
        "url": url,
        "bytes": os.path.getsize(path),
        "sha256": _sha256(path),
        "pulled_utc": _now(),
        "note": note,
    }
    if extra:
        row.update(extra)
    m["entries"] = [e for e in m["entries"] if e["file"] != rel] + [row]
    m["entries"].sort(key=lambda e: (e["source"], e["file"]))
    os.makedirs(RAW, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    print(f"  recorded {rel}  {row['bytes']:,} B  sha256 {row['sha256'][:12]}...")
    return row


def _get(url, retries=4, timeout=120):
    """GET with a few retries. Returns bytes."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last = e
            wait = 3 * (attempt + 1)
            print(f"  retry {attempt + 1}/{retries} after error: {e} (sleep {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"failed after {retries} attempts: {url}: {last}")


def _download(url, path, timeout=600):
    """Stream a large file to disk. Returns bytes written."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    tmp = path + ".part"
    n = 0
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        total = r.headers.get("Content-Length")
        total = int(total) if total else None
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            n += len(chunk)
            if total and n % (50 << 20) < (1 << 20):
                print(f"    {n / 1e6:,.0f} / {total / 1e6:,.0f} MB")
    os.replace(tmp, path)
    print(f"  {os.path.basename(path)}: {n:,} B in {time.time() - t0:.0f}s")
    return n


# ----------------------------------------------------------------------------
# sources
# ----------------------------------------------------------------------------
def pull_dfr(agencies):
    outdir = os.path.join(RAW, "dfr")
    for slug in agencies:
        url = DFR_URL.format(slug=slug)
        path = os.path.join(outdir, f"{slug}.csv")
        print(f"[dfr] {slug}")
        _download(url, path)
        with open(path, encoding="utf-8", errors="replace") as f:
            header = f.readline().strip()
            rows = sum(1 for _ in f)
        _record("dfr", path, url, note="AH Datalytics per-agency flight log",
                extra={"rows": rows, "columns": header})


def pull_chicago(d_from, d_to):
    """One CSV per day, paged with $limit/$offset on a stable $order."""
    outdir = os.path.join(RAW, "chicago_taxi")
    os.makedirs(outdir, exist_ok=True)
    day = d_from
    while day < d_to:
        nxt = day + dt.timedelta(days=1)
        path = os.path.join(outdir, f"taxi_{day.isoformat()}.csv")
        print(f"[chicago] {day.isoformat()}")
        where = (f"trip_start_timestamp >= '{day.isoformat()}T00:00:00' AND "
                 f"trip_start_timestamp < '{nxt.isoformat()}T00:00:00'")
        offset, total, header_written = 0, 0, False
        with open(path, "w", encoding="utf-8", newline="") as f:
            while True:
                q = {"$select": CHICAGO_COLS, "$where": where,
                     "$order": "trip_start_timestamp,taxi_id,trip_end_timestamp",
                     "$limit": CHICAGO_PAGE, "$offset": offset}
                url = CHICAGO_URL + "?" + urllib.parse.urlencode(q)
                text = _get(url).decode("utf-8")
                lines = text.splitlines()
                if not lines:
                    break
                header, rows = lines[0], lines[1:]
                if not header_written:
                    f.write(header + "\n")
                    header_written = True
                if rows:
                    f.write("\n".join(rows) + "\n")
                total += len(rows)
                if len(rows) < CHICAGO_PAGE:
                    break
                offset += CHICAGO_PAGE
                time.sleep(0.5)
        print(f"  {total:,} trips")
        _record("chicago_taxi", path, CHICAGO_URL + "?" + urllib.parse.urlencode(
            {"$select": CHICAGO_COLS, "$where": where}),
            note="one local day of trips, timestamps rounded to 15 min by the publisher",
            extra={"rows": total})
        day = nxt


def pull_ookla(year, quarter, kinds=("fixed", "mobile")):
    month = {1: "01", 2: "04", 3: "07", 4: "10"}[quarter]
    outdir = os.path.join(RAW, "ookla")
    for kind in kinds:
        url = OOKLA_URL.format(kind=kind, year=year, quarter=quarter, month=month)
        path = os.path.join(outdir, os.path.basename(url))
        print(f"[ookla] {kind} {year} Q{quarter}")
        _download(url, path)
        _record("ookla", path, url,
                note="Ookla Speedtest open data, CC BY-NC-SA 4.0, attribution required",
                extra={"year": year, "quarter": quarter, "kind": kind})


def pull_nhtsa():
    outdir = os.path.join(RAW, "nhtsa")
    for url in NHTSA_URLS:
        path = os.path.join(outdir, os.path.basename(url))
        print(f"[nhtsa] {os.path.basename(url)}")
        _download(url, path)
        _record("nhtsa", path, url, note="NHTSA ODI recall data, US public domain")


def register(source, note):
    """Register files that were downloaded by hand (Cabspotting)."""
    outdir = os.path.join(RAW, source)
    if not os.path.isdir(outdir):
        sys.exit(f"no directory {outdir}")
    for name in sorted(os.listdir(outdir)):
        path = os.path.join(outdir, name)
        if os.path.isfile(path) and not name.endswith(".part"):
            _record(source, path, "manual download, see note", note=note)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("dfr")
    p.add_argument("--agencies", nargs="+", required=True)

    p = sub.add_parser("chicago")
    p.add_argument("--from", dest="d_from", required=True, type=dt.date.fromisoformat)
    p.add_argument("--to", dest="d_to", required=True, type=dt.date.fromisoformat,
                   help="exclusive")

    p = sub.add_parser("ookla")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--quarter", type=int, required=True, choices=[1, 2, 3, 4])
    p.add_argument("--kinds", nargs="+", default=["fixed", "mobile"])

    sub.add_parser("nhtsa")

    p = sub.add_parser("manifest")
    p.add_argument("--source", required=True)
    p.add_argument("--note", default="")

    a = ap.parse_args()
    if a.cmd == "dfr":
        pull_dfr(a.agencies)
    elif a.cmd == "chicago":
        pull_chicago(a.d_from, a.d_to)
    elif a.cmd == "ookla":
        pull_ookla(a.year, a.quarter, tuple(a.kinds))
    elif a.cmd == "nhtsa":
        pull_nhtsa()
    elif a.cmd == "manifest":
        register(a.source, a.note)


if __name__ == "__main__":
    main()
