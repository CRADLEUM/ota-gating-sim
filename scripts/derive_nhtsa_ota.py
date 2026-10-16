#!/usr/bin/env python3
"""Extract remotely delivered software recall campaigns from the NHTSA flat
files and join their quarterly completion reports.

Inputs (data/raw/nhtsa/, unzipped)
  FLAT_RCL_POST_2010.txt   one row per campaign x make x model x year, 29 tab-separated
                           fields per RCL.txt (field 2 CAMPNO, 8 MFGNAME, 11 RCLTYPECD,
                           12 POTAFF, 16 RCDATE, 20 DESC_DEFECT, 22 CORRECTIVE_ACTION)
  FLAT_RCL_Qrtly_Rpts.txt  one row per campaign x quarterly report, 13 fields per
                           RCL_Qtrly_Rpts.txt (MFGTXT, CAMPNO, MFGCAMPNO, RCLSUBJ, ODATE,
                           ODATEEND, RPTNO, RPTQTR, INVOLVED, TTLREMEDIED, TTLUNREACH,
                           TTLREMOVED, SUBMDATE); covers reports from 2015-1 onward.

Step 1 (this script, mechanical). Two detection rules on the corrective-action
text, both reported as separate flags so the author can include or exclude
either:
  ota_explicit      the text says over-the-air or OTA
  ads_fleet_remote  the text says the manufacturer updated the ADS software of
                    its own fleet (Waymo, Zoox, Cruise, Pony.ai, Avride file this
                    way and do not name the delivery path)
A campaign is in the study set when either flag is true. A third flag,
dual_path, marks remedy text that also offers a dealer or service visit, so the
author can separate remote-only campaigns from dealer-or-OTA ones. The script also
parses "all affected vehicles were repaired by <date>" into completion_at_filing,
collapses to one row per campaign (makes and models listed for the spot check),
joins the quarterly rows, and computes the remedied share at the first, fourth,
sixth and latest report plus the number of quarterly reports needed to pass 95
percent.

Step 2 (author decision D4). Ownership class is proposed from the manufacturer
name (OPERATOR_MFRS, FLEET_MFRS, MIXED_MFRS) and marked as a proposal. The
author confirms or overrides in ownership_override and records the spot check
in spot_check before the falsification test runs. Nothing in the paper cites
this file until that column is signed off.

Outputs (data/derived/nhtsa/)
  ota_campaigns.csv       one row per study campaign
  ota_quarterly.csv       the joined quarterly rows for study campaigns
  nhtsa_summary.json      counts and the per-class completion summary

Run with `python -I` since the inputs are downloaded files.
"""
import csv
import datetime as dt
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "nhtsa")
OUT = os.path.join(ROOT, "data", "derived", "nhtsa")

RCL_COLS = ["RECORD_ID", "CAMPNO", "MAKETXT", "MODELTXT", "YEARTXT", "MFGCAMPNO", "COMPNAME",
            "MFGNAME", "BGMAN", "ENDMAN", "RCLTYPECD", "POTAFF", "ODATE", "INFLUENCED_BY",
            "MFGTXT", "RCDATE", "DATEA", "RPNO", "FMVSS", "DESC_DEFECT", "CONEQUENCE_DEFECT",
            "CORRECTIVE_ACTION", "NOTES", "RCL_CMPT_ID", "MFR_COMP_NAME", "MFR_COMP_DESC",
            "MFR_COMP_PTNO", "DO_NOT_DRIVE", "PARK_OUTSIDE"]
QTR_COLS = ["MFGTXT", "CAMPNO", "MFGCAMPNO", "RCLSUBJ", "ODATE", "ODATEEND", "RPTNO",
            "RPTQTR", "INVOLVED", "TTLREMEDIED", "TTLUNREACH", "TTLREMOVED", "SUBMDATE"]

# Word boundaries on both ends: "cover the air gap" (a trim-piece recall) is not an OTA remedy.
OTA_RE = re.compile(r"\bover[\s-]*the[\s-]*air\b|\bOTA\b", re.I)
ADS_RE = re.compile(r"\bADS\b|automated driving system", re.I)
SOFTWARE_RE = re.compile(r"software|firmware", re.I)
DEALER_RE = re.compile(r"dealer|service cent(?:er|re)|service facilit|technician|appointment|workshop|retailer", re.I)
DONE_RE = re.compile(r"all (?:affected |of the )?(?:driverless |ads-equipped )?(?:fleet )?vehicles"
                     r"(?: operating on public roads)? (?:were|have been|had been|are) (?:repaired|remedied|updated)"
                     r"(?: (?:by|on|in|as of) ([A-Za-z]+ \d{1,2}, \d{4}|[A-Za-z]+ \d{4}))?", re.I)

# Manufacturers that own and operate every affected unit (robotaxi and driverless fleets).
OPERATOR_MFRS = ("WAYMO", "CRUISE", "ZOOX", "NURO", "MOTIONAL", "AURORA", "KODIAK", "TUSIMPLE",
                 "MAY MOBILITY", "PONY", "ARGO", "GATIK", "BEEP", "EINRIDE", "AVRIDE", "TORC")
# Manufacturers whose vehicles are bought by commercial or public fleets, not consumers.
FLEET_MFRS = ("MACK", "VOLVO TRUCKS", "DAIMLER TRUCK", "FREIGHTLINER", "NAVISTAR", "INTERNATIONAL",
              "PACCAR", "KENWORTH", "PETERBILT", "MOTIV", "PROTERRA", "LION ELECTRIC", "NIKOLA",
              "WORKHORSE", "XOS", "LIGHTNING EMOTORS", "BLUE BIRD", "GREENPOWER", "THOMAS BUILT",
              "IC BUS", "GILLIG", "NEW FLYER", "NOVA BUS", "BYD", "HINO", "ARRIVAL", "SEA ELECTRIC",
              "SHYFT", "BOLLINGER", "HARBINGER", "ZEUS", "REE AUTOMOTIVE", "ELDORADO", "ENC",
              "PIERCE", "E-ONE", "FERRARA", "SPARTAN", "WELDON", "BRAUN", "HALCORE", "TESLA SEMI",
              "CUSTOM TRUCK", "NORTH CENTRAL", "OSHKOSH", "ROSENBAUER", "SUTPHEN")
# Products sold to both consumers and fleets; the author decides per campaign.
MIXED_MFRS = ("DAIMLER VANS", "MERCEDES-BENZ VANS", "RIVIAN", "FORD", "GENERAL MOTORS", "CHRYSLER",
              "STELLANTIS", "RAM")


def word_re(keys):
    """Whole-word match on the manufacturer name. A plain substring test put Zero
    Motorcycles in the operator class ("TORC" inside "MOTORCYCLES") and Polestar and
    Rivian in the fleet class ("MOTIV" inside "AUTOMOTIVE")."""
    return re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keys) + r")\b", re.I)


OPERATOR_RE, FLEET_RE, MIXED_RE = word_re(OPERATOR_MFRS), word_re(FLEET_MFRS), word_re(MIXED_MFRS)


def read_rcl():
    rows = []
    with open(os.path.join(RAW, "FLAT_RCL_POST_2010.txt"), encoding="latin-1", newline="") as f:
        for rec in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(rec) < 22:
                continue
            rows.append((rec + [""] * 29)[:29])
    return pd.DataFrame(rows, columns=RCL_COLS)


def read_qtr():
    q = pd.read_csv(os.path.join(RAW, "FLAT_RCL_Qrtly_Rpts.txt"), sep="\t", header=None,
                    names=QTR_COLS, dtype=str, quoting=csv.QUOTE_NONE, encoding="latin-1")
    for c in ("INVOLVED", "TTLREMEDIED", "TTLUNREACH", "TTLREMOVED", "RPTNO"):
        q[c] = pd.to_numeric(q[c], errors="coerce")
    q["share"] = q["TTLREMEDIED"] / q["INVOLVED"]
    return q


def classify(mfg):
    m = mfg or ""
    if OPERATOR_RE.search(m):
        return "manufacturer_operated", "manufacturer name (operator list)"
    if FLEET_RE.search(m):
        return "third_party_fleet", "manufacturer name (fleet list)"
    if MIXED_RE.search(m):
        return "consumer", "manufacturer name (mixed market, author to confirm)"
    return "consumer", "default"


def share_at(g, n):
    r = g[g["RPTNO"] == n]
    return float(r["share"].iloc[0]) if len(r) else np.nan


def main():
    os.makedirs(OUT, exist_ok=True)
    r = read_rcl()
    r["POTAFF"] = pd.to_numeric(r["POTAFF"], errors="coerce")
    text = r["CORRECTIVE_ACTION"].fillna("") + " " + r["NOTES"].fillna("")
    r["software"] = text.str.contains(SOFTWARE_RE)
    r["ota_explicit"] = text.str.contains(OTA_RE)
    # dual_path: the remedy text also offers a dealer or service visit, so completion mixes
    # the two delivery paths. Reported as a flag; the per-class summary splits on it.
    r["dual_path"] = text.str.contains(DEALER_RE)
    is_operator = r["MFGNAME"].fillna("").str.contains(OPERATOR_RE)
    r["ads_fleet_remote"] = is_operator & text.str.contains(ADS_RE) & r["software"] & ~r["dual_path"]

    g = (r.groupby("CAMPNO")
          .agg(MFGNAME=("MFGNAME", "first"), RCLTYPECD=("RCLTYPECD", "first"),
               POTAFF=("POTAFF", "max"), RCDATE=("RCDATE", "first"), ODATE=("ODATE", "first"),
               makes=("MAKETXT", lambda s: "|".join(sorted(set(s)))),
               models=("MODELTXT", lambda s: "|".join(sorted(set(s)))[:200]),
               n_models=("MODELTXT", "nunique"),
               n_rows=("RECORD_ID", "size"), software=("software", "any"),
               ota_explicit=("ota_explicit", "any"), ads_fleet_remote=("ads_fleet_remote", "any"),
               dual_path=("dual_path", "any"),
               corrective=("CORRECTIVE_ACTION", "first"))
          .reset_index())
    o = g[g["ota_explicit"] | g["ads_fleet_remote"]].copy()

    done = o["corrective"].fillna("").str.extract(DONE_RE, expand=True)
    o["completion_at_filing"] = o["corrective"].fillna("").apply(lambda s: bool(DONE_RE.search(s)))
    o["completion_date_text"] = done[0].fillna("")

    q = read_qtr()
    qo = q[q["CAMPNO"].isin(set(o["CAMPNO"]))].sort_values(["CAMPNO", "RPTNO"]).copy()
    by = qo.groupby("CAMPNO")
    o = o.set_index("CAMPNO")
    o["n_quarterly_reports"] = by.size()
    o["q1_quarter"] = by["RPTQTR"].first()
    o["q1_involved"] = by["INVOLVED"].first()
    o["q1_remedied"] = by["TTLREMEDIED"].first()
    o["q1_share"] = by.apply(lambda g: share_at(g, 1))
    o["q4_share"] = by.apply(lambda g: share_at(g, 4))
    o["q6_share"] = by.apply(lambda g: share_at(g, 6))
    o["last_quarter"] = by["RPTQTR"].last()
    o["last_share"] = by["share"].last()
    o["last_unreachable_share"] = (by["TTLUNREACH"].last() / by["INVOLVED"].last())
    o["quarters_to_95"] = by.apply(lambda g: float(g.loc[g["share"] >= 0.95, "RPTNO"].min()))
    o = o.reset_index()
    for c in ("q1_share", "q4_share", "q6_share", "last_share", "last_unreachable_share"):
        o[c] = o[c].astype(float).round(4)

    cls = o["MFGNAME"].apply(classify)
    o["ownership_proposed"] = [c[0] for c in cls]
    o["ownership_basis"] = [c[1] for c in cls]
    o["ownership_override"] = ""      # author: manufacturer_operated | consumer | third_party_fleet | exclude
    o["spot_check"] = ""              # author: ok | not_remote | note
    o["corrective"] = o["corrective"].fillna("").str.replace(r"\s+", " ", regex=True).str.slice(0, 500)
    o = o.sort_values(["ownership_proposed", "RCDATE"])
    cols = ["CAMPNO", "MFGNAME", "RCLTYPECD", "POTAFF", "RCDATE", "ODATE", "makes", "models", "n_models",
            "ota_explicit", "ads_fleet_remote", "dual_path", "completion_at_filing", "completion_date_text",
            "ownership_proposed", "ownership_basis", "n_quarterly_reports", "q1_quarter", "q1_involved", "q1_remedied",
            "q1_share", "q4_share", "q6_share", "last_quarter", "last_share", "last_unreachable_share",
            "quarters_to_95", "ownership_override", "spot_check", "corrective"]
    o[cols].to_csv(os.path.join(OUT, "ota_campaigns.csv"), index=False)
    qo.to_csv(os.path.join(OUT, "ota_quarterly.csv"), index=False)

    def cls_summary(sub):
        hasq = sub[sub["n_quarterly_reports"].notna()]
        return {
            "campaigns": int(len(sub)),
            "units": int(sub["POTAFF"].fillna(0).sum()),
            "dual_path_share": round(float(sub["dual_path"].mean()), 3) if len(sub) else None,
            "completion_at_filing_share": round(float(sub["completion_at_filing"].mean()), 3) if len(sub) else None,
            "with_quarterly": int(len(hasq)),
            "q1_share_median": round(float(hasq["q1_share"].median()), 3) if len(hasq) else None,
            "q4_share_median": round(float(hasq["q4_share"].median()), 3) if hasq["q4_share"].notna().any() else None,
            "q6_share_median": round(float(hasq["q6_share"].median()), 3) if hasq["q6_share"].notna().any() else None,
            "last_share_median": round(float(hasq["last_share"].median()), 3) if len(hasq) else None,
            "share_reaching_95_by_q6": round(float((hasq["quarters_to_95"] <= 6).mean()), 3) if len(hasq) else None,
            "quarters_to_95_median": round(float(hasq["quarters_to_95"].median()), 1) if hasq["quarters_to_95"].notna().any() else None,
        }

    s = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "campaigns_post_2010": int(len(g)),
        "campaigns_software_remedy": int(g["software"].sum()),
        "study_set": int(len(o)),
        "ota_explicit": int(o["ota_explicit"].sum()),
        "ads_fleet_remote_only": int((o["ads_fleet_remote"] & ~o["ota_explicit"]).sum()),
        "by_type": o["RCLTYPECD"].value_counts().to_dict(),
        "by_ownership_proposed": {k: cls_summary(v) for k, v in o.groupby("ownership_proposed")},
        # Same classes with the dual-path (dealer or OTA) campaigns removed, so a remote-only
        # consumer trajectory can be compared with the operator class on equal terms.
        "by_ownership_proposed_remote_only": {k: cls_summary(v) for k, v in o[~o["dual_path"]].groupby("ownership_proposed")},
        "by_manufacturer_top": o["MFGNAME"].value_counts().head(12).to_dict(),
        "classifier": "whole-word manufacturer-name match (v2, 2026-10-09); OTA regex bounded on both ends",
        "status": "proposed classification; author sign-off (D4) required before use",
    }
    with open(os.path.join(OUT, "nhtsa_summary.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)
    print(json.dumps(s, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
