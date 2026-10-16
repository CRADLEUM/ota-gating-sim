#!/usr/bin/env python3
"""Reproduction check for gating_sim.py (v4).

Runs the simulation in a scratch directory and compares its outputs against the
reference artifacts in expected/:

    stdout            vs  expected/run_v4.log
    results_v4.csv    vs  expected/results_v4.csv
    sweeps_v4.csv     vs  expected/sweeps_v4.csv
    replay_v4.csv     vs  expected/replay_v4.csv

Those reference artifacts are the ones behind the numbers in the paper, so a
clean run here means the published results regenerate from source on this
machine. A mismatch means either the model changed or the random stream moved
under a library upgrade, and both are worth knowing before anyone cites it.

The simulation reads data/sim_inputs/ relative to its own location, so the
scratch copy is given a link-free copy of that directory as well.

Figures are not byte-compared. Matplotlib output is not stable across versions,
and a pixel difference in a plot is not a reproduction failure. The CSVs are
the data; figures/ holds the renders that went into the paper.

Runs the full sweep set, which takes about five minutes.

Usage:
    python verify.py [--outdir DIR]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "gating_sim.py")
INPUTS = os.path.join(HERE, "data", "sim_inputs")
EXPECTED_DIR = os.path.join(HERE, "expected")
ARTIFACTS = ("results_v4.csv", "sweeps_v4.csv", "replay_v4.csv")


def normalize(text):
    """Compare on content, not on which platform wrote the newlines."""
    return [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]


def read_text(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def report_diff(label, expected, actual, limit=12):
    exp, act = normalize(expected), normalize(actual)
    if exp == act:
        print("  ok  %s matches" % label)
        return True
    print("  FAIL  %s differs" % label)
    shown = 0
    for i in range(max(len(exp), len(act))):
        e = exp[i] if i < len(exp) else "<missing>"
        a = act[i] if i < len(act) else "<missing>"
        if e != a:
            print("        line %d" % (i + 1))
            print("          expected: %s" % e)
            print("          actual:   %s" % a)
            shown += 1
            if shown >= limit:
                print("        ... further differences suppressed")
                break
    return False


def environment():
    versions = {"python": sys.version.split()[0]}
    for mod in ("numpy", "matplotlib"):
        try:
            versions[mod] = __import__(mod).__version__
        except ImportError:
            versions[mod] = "MISSING"
    return versions


def main():
    ap = argparse.ArgumentParser(description="Reproduction check for gating_sim.py")
    ap.add_argument("--outdir", help="keep the run in this directory instead of a temp dir")
    args = ap.parse_args()

    versions = environment()
    print("environment")
    for name, version in versions.items():
        print("  %-11s %s" % (name, version))
    if "MISSING" in versions.values():
        print("\nInstall dependencies first: pip install -r requirements.txt")
        return 2

    workdir = args.outdir or tempfile.mkdtemp(prefix="gating-sim-verify-")
    os.makedirs(workdir, exist_ok=True)
    shutil.copy2(MODEL, os.path.join(workdir, "gating_sim.py"))
    dst_inputs = os.path.join(workdir, "data", "sim_inputs")
    if os.path.isdir(dst_inputs):
        shutil.rmtree(dst_inputs)
    shutil.copytree(INPUTS, dst_inputs)

    print("\nrunning gating_sim.py (about five minutes)")
    proc = subprocess.run(
        [sys.executable, "-P", "gating_sim.py"],
        cwd=workdir, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print("  FAIL  simulation exited %d" % proc.returncode)
        print(proc.stderr)
        return 1

    print("\ncomparing against expected/")
    ok = report_diff("stdout vs run_v4.log", read_text(os.path.join(EXPECTED_DIR, "run_v4.log")), proc.stdout)
    for name in ARTIFACTS:
        path = os.path.join(workdir, name)
        if not os.path.exists(path):
            print("  FAIL  the run produced no %s" % name)
            ok = False
        else:
            ok &= report_diff(name, read_text(os.path.join(EXPECTED_DIR, name)), read_text(path))

    if args.outdir:
        print("\nrun kept in %s" % workdir)
    else:
        shutil.rmtree(workdir, ignore_errors=True)

    print("")
    if ok:
        print("Reproduced. The published results regenerate from source in this environment.")
        return 0
    print("Did NOT reproduce. See the differences above before relying on these results.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
