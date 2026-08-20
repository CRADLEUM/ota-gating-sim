"""
Gating-first OTA delivery simulation v2 (IPCCC 2026) -- WITHIN-WINDOW ATOMIC COMPLETION.

Revision after Stage-3 review: the v1 model let received bytes accumulate across windows,
so "current" reduced to throughput x H >= S (a bandwidth condition) and the install gate was
non-binding. Reviewers (Methods + Devil's Advocate) correctly showed gating then did no
independent work: the cross-class flip was driven by the backhaul/local rate ratio, not by
the gate.

v2 fix: model deploy-on-complete as an ATOMIC MAINTENANCE SESSION. A device becomes current
only when a SINGLE stationary window is long enough to (a) transfer the whole update at the
in-window delivery rate and (b) complete the atomic install, contiguously. No cross-window
carryover: a partial session is lost when the window ends. This is the semantics under which
update GATING (the stationary-window length / duty cycle), not aggregate link capacity,
governs feasibility -- because over the horizon H a slow backhaul may deliver enough total
bytes yet never enough WITHIN ONE window.

Gate-isolation experiment (the point): hold ALL transfer rates fixed and vary the stationary-
window length alone. If the preferred topology changes, gating is an independent driver.
"""

import numpy as np
import csv, os

SEED = 20260705
H = 360            # patch deadline (min) = 6 h
DT = 1
N = 500
INSTALL_T = 5      # atomic install duration (min), within one stationary window

CLASSES = {
    "drone": dict(
        op_mean=30, stat_mean=60,
        cen_r_stat=0.30,     # dock WAN backhaul: constrained (remote cellular/satellite)
        depot_r=4.0,         # local dock staging (backhaul-independent)
        gossip_beta=0.0015, gossip_seed=0.05,
    ),
    "av": dict(
        op_mean=180, stat_mean=120,
        cen_r_stat=1.10,     # curb/garage backhaul: good (urban)
        depot_r=4.0,
        gossip_beta=0.006, gossip_seed=0.05,
    ),
}


def _win(rng, mean, n):
    return np.maximum(1.0, rng.exponential(mean, size=n))


def simulate(topology, cls, S, rng, stat_mean=None, cen_r_stat=None):
    """WITHIN-WINDOW completion. Optional overrides let the gate/backhaul sweeps vary one
    parameter while holding the rest fixed. Returns fraction current by H."""
    p = CLASSES[cls]
    sm = p["stat_mean"] if stat_mean is None else stat_mean
    crs = p["cen_r_stat"] if cen_r_stat is None else cen_r_stat
    om = p["op_mean"]
    depot_r = p["depot_r"]

    in_op = rng.random(N) < (om / (om + sm))
    rem = np.where(in_op, rng.random(N) * _win(rng, om, N), rng.random(N) * _win(rng, sm, N))
    sess_tx = np.zeros(N)      # transfer accrued IN THE CURRENT stationary window
    sess_in = np.zeros(N)      # install min accrued in the current window (after tx complete)
    done = np.zeros(N, dtype=bool)
    completed = np.zeros(N, dtype=bool)

    if topology == "gossip":
        I0, beta = p["gossip_seed"], p["gossip_beta"]
        u = rng.random(N)
        ratio = (1 - I0) / I0
        with np.errstate(divide="ignore"):
            infect_t = np.clip(-np.log((1 / np.clip(u, 1e-9, 1 - 1e-9) - 1) / ratio) / beta, 0, None)
    else:
        infect_t = None

    t = 0
    while t < H:
        rem -= DT
        in_stat = ~in_op
        active = in_stat & ~done

        # --- transfer, only within the current stationary window (no carryover) ---
        if topology == "depot":
            sess_tx = np.where(active, np.minimum(S, sess_tx + depot_r * DT), sess_tx)
        elif topology == "centralized":
            sess_tx = np.where(active, np.minimum(S, sess_tx + crs * DT), sess_tx)
        elif topology == "gossip":
            infected = infect_t <= t
            sess_tx = np.where(active & infected, S, sess_tx)  # holds full update once infected

        # --- atomic install: contiguous minutes in-window after transfer complete ---
        installing = active & (sess_tx >= S - 1e-9)
        sess_in = np.where(installing, sess_in + DT, sess_in)
        newly = (~done) & (sess_in >= INSTALL_T)
        completed = completed | newly
        done = done | newly

        # --- window transitions: leaving a stationary window loses the partial session ---
        flipped = rem <= 0
        if flipped.any():
            to_op = flipped & (~in_op)     # stationary -> operation: reset session
            to_stat = flipped & in_op      # operation -> stationary: fresh session (already 0)
            if to_op.any():
                rem[to_op] = _win(rng, om, int(to_op.sum()))
                sess_tx[to_op] = 0.0
                sess_in[to_op] = 0.0
            if to_stat.any():
                rem[to_stat] = _win(rng, sm, int(to_stat.sum()))
            in_op = np.where(flipped, ~in_op, in_op)
        t += DT

    return float(np.mean(completed))


R = 60
R_SENS = 30
TOP_IDX = {"centralized": 0, "gossip": 1, "depot": 2}
CLS_IDX = {"drone": 0, "av": 1}


def _seed(cls, top, S, r, tag=0):
    return SEED + r * 1_000_003 + CLS_IDX[cls] * 10_007 + TOP_IDX[top] * 101 + int(S) + tag * 7_919


def reps(top, cls, S, R_=R, stat_mean=None, cen_r_stat=None, tag=0):
    return np.array([
        simulate(top, cls, S, np.random.default_rng(_seed(cls, top, S, r, tag)),
                 stat_mean=stat_mean, cen_r_stat=cen_r_stat)
        for r in range(R_)
    ])


def crossover(unaware, depot, S_values, thr=0.15):
    for i, S in enumerate(S_values):
        if depot[i] - unaware[i] > thr:
            return S
    return None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    S_values = list(range(5, 205, 10))

    # --- main: completion vs S per class ---
    results = {}
    for cls in CLASSES:
        results[cls] = {}
        for top in ("centralized", "gossip", "depot"):
            m = np.array([reps(top, cls, S).mean() for S in S_values])
            s = np.array([reps(top, cls, S).std() for S in S_values])
            results[cls][top] = (m, s)

    max_std = max(results[c][t][1].max() for c in CLASSES for t in results[c])
    print(f"WITHIN-WINDOW model. R={R}, N={N}, H={H}, seed={SEED}. Max std={max_std:.3f}.\n")
    for cls in CLASSES:
        print(f"[{cls}]  S   cen    gossip  depot")
        for i, S in enumerate(S_values):
            if S in (25, 65, 105, 155):
                c, g, d = (results[cls][t][0][i] for t in ("centralized", "gossip", "depot"))
                print(f"      {S:4d}  {c:.2f}   {g:.2f}    {d:.2f}")
        best = np.maximum(results[cls]["centralized"][0], results[cls]["gossip"][0])
        cx = crossover(best, results[cls]["depot"][0], S_values)
        print(f"   crossover S* (depot beats best gating-unaware by >0.15): {cx}\n")

    # --- GATE-ISOLATION: fix all rates, vary drone stationary-window length ---
    print("GATE-ISOLATION (drone, all rates fixed): crossover S* vs stationary-window mean:")
    win_values = [20, 40, 60, 90, 120, 160, 200, 260, 320]
    gate_Sstar = []
    S_FIXED_REPORT = 105
    for wm in win_values:
        mc = np.array([reps("centralized", "drone", S, R_=R_SENS, stat_mean=wm, tag=1).mean() for S in S_values])
        md = np.array([reps("depot", "drone", S, R_=R_SENS, stat_mean=wm, tag=1).mean() for S in S_values])
        cx = crossover(mc, md, S_values)
        gate_Sstar.append(cx)
        # also the depot-minus-centralized gap at a fixed representative S
        idx = S_values.index(S_FIXED_REPORT)
        gap = md[idx] - mc[idx]
        print(f"   window~{wm:3d} min -> S*={cx}   (gap at S={S_FIXED_REPORT}: {gap:+.2f})")

    # --- backhaul sensitivity (secondary robustness), within-window model ---
    print("\nBACKHAUL (drone, window fixed): crossover S* vs dock backhaul:")
    bh_values = [round(x, 2) for x in np.arange(0.10, 1.61, 0.15)]
    bh_Sstar = []
    for bh in bh_values:
        mc = np.array([reps("centralized", "drone", S, R_=R_SENS, cen_r_stat=bh, tag=2).mean() for S in S_values])
        md = np.array([reps("depot", "drone", S, R_=R_SENS, cen_r_stat=bh, tag=2).mean() for S in S_values])
        cx = crossover(mc, md, S_values)
        bh_Sstar.append(cx)
        print(f"   backhaul {bh:.2f} -> S*={cx}")

    # --- write CSV ---
    with open(os.path.join(here, "results_v2.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "S", "cen_mean", "cen_std", "gossip_mean", "gossip_std", "depot_mean", "depot_std"])
        for cls in CLASSES:
            for i, S in enumerate(S_values):
                row = [cls, S]
                for top in ("centralized", "gossip", "depot"):
                    row += [f"{results[cls][top][0][i]:.3f}", f"{results[cls][top][1][i]:.3f}"]
                w.writerow(row)

    # --- figures ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"centralized": "tab:blue", "gossip": "tab:orange", "depot": "tab:green"}
    mk = {"centralized": "o", "gossip": "s", "depot": "^"}
    lab = {"centralized": "centralized push", "gossip": "P2P gossip", "depot": "depot-staged"}

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, cls in zip(axes, CLASSES):
        for top in ("centralized", "gossip", "depot"):
            m, s = results[cls][top]
            ax.plot(S_values, m, marker=mk[top], color=colors[top], label=lab[top], ms=4, lw=1.7)
            ax.fill_between(S_values, m - 1.96 * s, m + 1.96 * s, color=colors[top], alpha=0.18)
        cx = crossover(np.maximum(results[cls]["centralized"][0], results[cls]["gossip"][0]),
                       results[cls]["depot"][0], S_values)
        if cx:
            ax.axvline(cx, ls="--", c="0.4", lw=1.1)
            ax.annotate(f"$S^*\\approx {cx}$", xy=(cx, 0.4), xytext=(cx + 6, 0.45), fontsize=11, color="0.2")
        ax.set_title(cls.upper(), fontsize=12)
        ax.set_xlabel("update size $S$ (delivery-minutes)", fontsize=11)
        ax.grid(alpha=0.3); ax.tick_params(labelsize=10)
    axes[0].set_ylabel("current by redeployment (fraction)", fontsize=11)
    axes[0].legend(fontsize=9, loc="center right")
    fig.tight_layout(); fig.savefig(os.path.join(here, "crossover.png"), dpi=150)

    # Fig 2: what sets the boundary -- gate (window, rates fixed) beside backhaul.
    fig2, (axg, axb) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    gx = [w for w, cx in zip(win_values, gate_Sstar) if cx is not None]
    gy = [cx for cx in gate_Sstar if cx is not None]
    axg.plot(gx, gy, "o-", color="tab:purple", ms=5)
    axg.set_xlabel("stationary-window mean (min)", fontsize=10)
    axg.set_ylabel("crossover $S^*$ (delivery-min)", fontsize=10)
    axg.set_title("(a) gate isolation: all rates fixed", fontsize=10)
    axg.grid(alpha=0.3); axg.tick_params(labelsize=9)
    bx = [b for b, cx in zip(bh_values, bh_Sstar) if cx is not None]
    by = [cx for cx in bh_Sstar if cx is not None]
    axb.plot(bx, by, "s-", color="tab:red", ms=5)
    axb.set_xlabel("dock backhaul rate (deliv-min/min)", fontsize=10)
    axb.set_title("(b) backhaul: window fixed", fontsize=10)
    axb.grid(alpha=0.3); axb.tick_params(labelsize=9)
    fig2.tight_layout(); fig2.savefig(os.path.join(here, "drivers.png"), dpi=150)

    print("\nwrote results_v2.csv, crossover.png, drivers.png")


if __name__ == "__main__":
    main()
