#!/usr/bin/env python3
"""Print-ready figures for the paper, rendered from the frozen reference run.

Reads only the byte-compared artifacts in expected/ (results_v4.csv,
sweeps_v4.csv) so every curve in the manuscript traces to data that verify.py
checks. gating_sim.py is not touched; this script only re-renders its outputs at
IEEE column width with print-sized type.

Figures (PDF for the LaTeX build, PNG for review):

    fig1_t95      completion against deadline, 4 GB, four architectures, both classes
    fig2_drivers  four panels: install time, backhaul rate, gate isolation, reach
    fig3_sources  centralized push under exponential, bootstrap and replay windows

Colour is the Okabe-Ito colourblind-safe set; architecture is colour plus marker,
device class is line style, so no series is identified by colour alone.

Usage:
    python scripts/make_paper_figures.py [--outdir DIR] [--expected DIR]
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Okabe-Ito. Architecture colours (fixed order) and the two class colours used
# where every series on a panel is the same architecture.
ARCH = {
    "centralized": dict(color="#0072B2", marker="o", label="centralized push"),
    "gossip":      dict(color="#E69F00", marker="s", label="P2P gossip"),
    "depot":       dict(color="#009E73", marker="^", label="depot staging"),
    "hybrid":      dict(color="#CC79A7", marker="D", label="hybrid (best link)"),
}
CLASS = {
    "drone": dict(color="#0072B2", ls="-",  marker="o", label="drone (Cincinnati docks)"),
    "av":    dict(color="#D55E00", ls="--", marker="s", label="AV (Chicago taxi)"),
}
CLASS_LS = {"drone": "-", "av": "--"}
TARGET = 0.95
H_MAX = 360.0
NOT_REACHED_Y = 385.0  # where a "not reached within six hours" point is drawn

COLW = 3.5  # IEEE single-column width, inches


def style():
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "lines.linewidth": 1.2, "lines.markersize": 3.2,
        "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.5,
        "axes.edgecolor": "#444444", "axes.linewidth": 0.6,
        "legend.frameon": True, "legend.framealpha": 0.95, "legend.edgecolor": "#CCCCCC",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    })


def read_results(path):
    out = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[(r["class"], r["source"], r["arch"], float(r["S_gb"]))].append(
                (float(r["H_min"]), float(r["mean"]), float(r["std"])))
    for k in out:
        out[k].sort()
    return out


def read_sweeps(path):
    out = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["metric"] != "T95_min_at_4GB":
                continue
            out[(r["sweep"], r["class"], r["arch"])].append(
                (r["value"], float(r["result"]) if r["result"] != "" else None))
    return out


def t95_series(sweeps, sweep, cls, arch, numeric=True):
    rows = sweeps[(sweep, cls, arch)]
    xs = [float(v) if numeric else v for v, _ in rows]
    ys = [y for _, y in rows]
    return xs, ys


def plot_t95(ax, xs, ys, color, ls, marker, label, log=False):
    """Draw a T95 curve; points that never reached the target sit above the axis
    as open markers at NOT_REACHED_Y."""
    reached = [(x, y) for x, y in zip(xs, ys) if y is not None]
    missing = [x for x, y in zip(xs, ys) if y is None]
    if reached:
        ax.plot([x for x, _ in reached], [y for _, y in reached], color=color, ls=ls,
                marker=marker, label=label, markeredgecolor="white", markeredgewidth=0.4)
    if missing:
        ax.plot(missing, [NOT_REACHED_Y] * len(missing), ls="none", marker=marker,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=0.9,
                markersize=3.6, clip_on=False)
    ax.set_ylim(0, 400)
    ax.set_yticks([0, 100, 200, 300])
    ax.axhline(H_MAX, color="#999999", lw=0.5, ls=":")


def fig1(results, sweeps, outdir):
    fig, ax = plt.subplots(figsize=(COLW, 2.2))
    for cls in ("drone", "av"):
        for arch, spec in ARCH.items():
            pts = [(h, m, s) for h, m, s in results[(cls, "bootstrap", arch, 4.0)] if h <= H_MAX]
            hs = [p[0] for p in pts]; ms = [p[1] for p in pts]; ss = [p[2] for p in pts]
            ax.plot(hs, ms, color=spec["color"], ls=CLASS_LS[cls], marker=spec["marker"],
                    markevery=3, markeredgecolor="white", markeredgewidth=0.4)
            ax.fill_between(hs, [m - 1.96 * s for m, s in zip(ms, ss)],
                            [min(1.0, m + 1.96 * s) for m, s in zip(ms, ss)],
                            color=spec["color"], alpha=0.12, lw=0)
    ax.axhline(TARGET, color="#666666", lw=0.6, ls=":")
    ax.text(4, TARGET + 0.012, "target 0.95", fontsize=6.5, color="#444444", va="bottom")
    # T95 markers for push, read from the sweep file (not typed in).
    for cls, ytxt in (("drone", 0.30), ("av", 0.40)):
        t = dict(sweeps[("main", cls, "centralized")])["bootstrap"]
        ax.axvline(t, color="#666666", lw=0.6, ls=":")
        ax.text(t + 4, ytxt, "%s push\n$T_{95}$ = %d min" % (cls.upper(), t),
                fontsize=6.5, color="#333333", va="bottom")
    ax.set_xlim(0, H_MAX); ax.set_ylim(0, 1.02)
    ax.set_xlabel("deadline after release (min)")
    ax.set_ylabel("completion rate, 4 GB update")
    arch_handles = [Line2D([0], [0], color=s["color"], marker=s["marker"], ls="-",
                           markeredgecolor="white", markeredgewidth=0.4, label=s["label"])
                    for s in ARCH.values()]
    class_handles = [Line2D([0], [0], color="#333333", ls=CLASS_LS[c], label=CLASS[c]["label"])
                     for c in ("drone", "av")]
    leg1 = ax.legend(handles=arch_handles, loc="lower right", bbox_to_anchor=(1.0, 0.02))
    ax.add_artist(leg1)
    ax.legend(handles=class_handles, loc="center right", bbox_to_anchor=(1.0, 0.66))
    fig.tight_layout(pad=0.3)
    save(fig, outdir, "fig1_t95")


def fig2(sweeps, outdir):
    fig, axs = plt.subplots(2, 2, figsize=(COLW, 2.85), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.06)
    (a, b), (c, d) = axs
    small = dict(fontsize=7)

    # (a) install time, centralized push
    for cls, spec in CLASS.items():
        xs, ys = t95_series(sweeps, "install", cls, "centralized")
        plot_t95(a, xs, ys, spec["color"], spec["ls"], spec["marker"], cls.upper() if cls == "av" else "drone")
    a.set_title("(a) install time", loc="left", **small)
    a.set_xlabel("install time (min)", **small)
    a.set_ylabel("$T_{95}$ (min), 4 GB", **small)
    a.set_xticks([5, 20, 40, 60])
    a.legend(loc="upper left", fontsize=6, handlelength=1.6)

    # (b) backhaul rate, centralized push
    for cls, spec in CLASS.items():
        xs, ys = t95_series(sweeps, "backhaul", cls, "centralized")
        plot_t95(b, xs, ys, spec["color"], spec["ls"], spec["marker"], cls)
    b.set_xscale("log")
    b.set_xticks([5, 28, 118, 297]); b.set_xticklabels(["5", "28", "118", "297"])
    b.minorticks_off()
    for x, lab, y, va in ((28, "p10", 10, "bottom"), (118, "sat", 10, "bottom"),
                          (137, "rural", 330, "top"), (297, "metro", 10, "bottom")):
        b.axvline(x, color="#AAAAAA", lw=0.5, ls=":")
        b.text(x, y, lab, rotation=90, fontsize=5.2, color="#666666", ha="right", va=va)
    b.set_title("(b) backhaul rate", loc="left", **small)
    b.set_xlabel("backhaul (Mbps)", **small)

    # (c) gate isolation: stationary windows scaled, rates fixed
    for cls, spec in CLASS.items():
        xs, ys = t95_series(sweeps, "scale", cls, "centralized")
        plot_t95(c, xs, ys, spec["color"], spec["ls"], spec["marker"], cls)
    c.set_title("(c) window scale, rates fixed", loc="left", **small)
    c.set_xlabel("stationary windows (x measured)", **small)
    c.set_ylabel("$T_{95}$ (min), 4 GB", **small)
    c.set_xticks([0.25, 1.0, 2.0])

    # (d) reach (AV): depot only, hybrid at the metro and stress rates
    xs, ys = t95_series(sweeps, "reach_metro", "av", "depot")
    plot_t95(d, xs, ys, ARCH["depot"]["color"], "-", "^", "depot only")
    xs, ys = t95_series(sweeps, "reach_metro", "av", "hybrid")
    plot_t95(d, xs, ys, ARCH["hybrid"]["color"], "-", "D", "hybrid, 297 Mbps")
    xs, ys = t95_series(sweeps, "reach_p10", "av", "hybrid")
    plot_t95(d, xs, ys, ARCH["hybrid"]["color"], "--", "d", "hybrid, 28 Mbps")
    d.set_title("(d) reach, AV", loc="left", **small)
    d.set_xlabel("share of stops with staging", **small)
    d.set_xticks([0, 0.5, 1.0])
    d.legend(loc="lower left", fontsize=6, handlelength=1.6)

    for ax in (a, b, c, d):
        ax.tick_params(labelsize=6.5)
    save(fig, outdir, "fig2_drivers")


def fig3(results, outdir):
    fig, axs = plt.subplots(1, 2, figsize=(COLW, 1.6), sharey=True, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.03)
    sources = (("exp", "#888888", ":", "exponential (assumed)"),
               ("bootstrap", "#0072B2", "-", "bootstrap (measured)"),
               ("replay12", "#D55E00", "--", "replay, 12:00 start"))
    for ax, cls, title in ((axs[0], "drone", "drone, push"), (axs[1], "av", "AV, push")):
        for src, color, ls, label in sources:
            pts = [(h, m) for h, m, _ in results[(cls, src, "centralized", 4.0)] if h <= H_MAX]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, ls=ls, label=label)
        ax.axhline(TARGET, color="#666666", lw=0.6, ls=":")
        ax.set_title(title, loc="left", fontsize=7)
        ax.set_xlim(0, H_MAX); ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 120, 240, 360])
        ax.tick_params(labelsize=6.5)
    axs[0].set_ylabel("completion, 4 GB", fontsize=7)
    fig.supxlabel("deadline after release (min)", fontsize=7)
    axs[0].legend(loc="lower right", fontsize=5.8, handlelength=1.6)
    save(fig, outdir, "fig3_sources")


def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        path = os.path.join(outdir, "%s.%s" % (name, ext))
        fig.savefig(path, **kw)
        print("wrote", path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expected", default=os.path.join(ROOT, "expected"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "figures", "paper"))
    args = ap.parse_args()
    style()
    results = read_results(os.path.join(args.expected, "results_v4.csv"))
    sweeps = read_sweeps(os.path.join(args.expected, "sweeps_v4.csv"))
    print("T95 at 4 GB (min) from the main sweep, as annotated:")
    for cls in ("drone", "av"):
        for arch in ARCH:
            print("  %-5s %-11s %s" % (cls, arch, dict(sweeps[("main", cls, arch)])))
    fig1(results, sweeps, args.outdir)
    fig2(sweeps, args.outdir)
    fig3(results, args.outdir)


if __name__ == "__main__":
    main()
