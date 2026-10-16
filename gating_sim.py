"""
Gating-first OTA delivery simulation v4 (IPCCC 2026 short paper) -- MEASURED INPUTS.

v4 (2026-10-08) keeps the v3 semantics (a device becomes current only when ONE
stationary window is long enough to transfer the whole update and finish an
atomic install, contiguously) and changes what feeds it, and what is reported.

  units          update size S in GB, link rates in Mbps, time in minutes;
                 transfer minutes = S * 8000 / Mbps / 60
  windows        three sources per device class
                   exp        exponential windows with the v3 means (continuity)
                   bootstrap  windows resampled from measured sets (DFR dock windows
                              per agency, K_min fleet, daytime; Chicago taxi gaps,
                              daytime) with the length-biased residual for the first
                              window
                   replay     deterministic: every device replays its real schedule
                              from a chosen campaign start time; no random draws
  rates          measured anchors (Ookla 2026 Q2 tiles; Starlink documented)
  install        documented 20 min (dock) and 30 min (vehicle), swept
  reach          a depot window is one spent where local staging exists; rule per
                 class ("all" for docked drones, gaps >= 120 min for AVs, or a
                 probability p that is swept)
  architectures  centralized push (backhaul, every window), depot staging (local
                 rate, depot windows only), P2P gossip, and a hybrid that uses the
                 best link available in each window
  baselines      the window-aware non-atomic selector (transfer may span windows,
                 install still atomic) and the bandwidth-only selector; the report
                 lists where each one picks centralized push but atomic completion
                 misses the target
  metrics        completion-before-redeployment rate, as in v3, now reported along
                 two axes: against update size at the six-hour deadline (S95 = the
                 largest swept S that still meets TARGET, margin-free) and against
                 the deadline at anchor sizes (T95 = the earliest deadline at which
                 completion reaches TARGET). At measured 2026 rates transfer takes
                 minutes, so S95 sits beyond the documented sizes for most settings
                 and T95 is where the install gate, the window length, and reach
                 show up. The v3 margin crossover is printed once for continuity.

Engine. For rho = 0, completion is evaluated exactly on each device's window
sequence (vectorised, no time stepping). The resumability sweep uses an exact
window-walking engine with the carried fraction rho; at rho = 0 it reproduces
the main engine and at rho = 1 the window-aware non-atomic baseline, and the run
prints both differences. The v3 one-minute stepped engine is kept only as an
independent cross-check (own random stream, so the comparison is statistical).

Inputs: data/sim_inputs/ (built by scripts/make_sim_inputs.py from data/derived/).
Usage:  python gating_sim.py [--quick] [--outdir DIR]
        --quick runs every sweep with a few replications as a smoke test.
Tests:  python -m pytest tests/
"""

import csv
import json
import os

import numpy as np

SEED = 20260705
H_MAX = 360.0       # longest deadline considered (min) = 6 h, the v3 horizon
N = 500             # devices per class
R = 60              # replications, main sweep
R_SENS = 30         # replications, sensitivity sweeps
TARGET = 0.95       # completion target behind S95 and T95
MARGIN_V3 = 0.15    # v3 crossover margin, printed once for continuity
DT = 1              # step of the stepped engine (min)
K = 200             # window pairs generated per device (asserted to cover H_MAX)

S_GB = np.round(np.arange(0.25, 10.01, 0.25), 2)
S_ANCHORS = (0.5, 2.0, 4.0, 8.0)   # GB; 0.42 GB documented dock firmware, 3.75 GB documented upper order
S_HEADLINE = 4.0
H_GRID = np.arange(10.0, 361.0, 10.0)
H_REPORT = (60.0, 120.0, 360.0)
REPLAY_HOURS = (8, 12, 16, 20)
REPLAY_HEADLINE_HOUR = 12

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(HERE, "data", "sim_inputs")

# Mbps. Measured values: data/derived/ookla/ookla_rates.json (test-weighted medians,
# 2026 Q2). Documented: data/derived/documented_rates.json.
RATES = {
    "fixed_conus": 372.0,          # Ookla fixed, CONUS
    "dock_ethernet": 1000.0,       # DJI dock spec, wired uplink
    "mobile_chicago": 297.0,       # Ookla mobile, Chicago metro
    "mobile_conus": 229.0,         # Ookla mobile, CONUS
    "mobile_rural_wy": 137.0,      # Ookla mobile, Wyoming (rural proxy)
    "satellite_starlink": 118.0,   # Ookla 2025 satellite report, Starlink US median
    "mobile_p10_conus": 28.0,      # Ookla mobile, CONUS 10th percentile tile
}
BACKHAUL_SWEEP = (5.0, 10.0, 28.0, 50.0, 118.0, 137.0, 297.0)
INSTALL_SWEEP = (5.0, 10.0, 20.0, 30.0, 45.0, 60.0)
SCALE_SWEEP = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
REACH_SWEEP = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
RHO_SWEEP = (0.0, 0.25, 0.5, 0.75, 1.0)
AGENCIES = ("cincinnati-pd", "wichita-pd", "denver-pd", "castle-rock-pd-co", "cspd-real-time-crime-center")

# Gossip contact rate beta, contacts per on-line node per minute, measured from the
# Cabspotting taxi traces (data/derived/cabspotting/contact_rates.json: 536 San
# Francisco cabs, 24 days, 30 s grid, 100 m range, daytime). "every_contact" assumes
# transmission on every contact, the upper bound in the SI model; the gated values
# count only contacts at least that long (contact_durations.csv), which is what a
# whole-image transfer needs at a given sidelink rate. The v3 assumptions (0.0015
# drones, 0.006 AVs) are kept in the sweep. No contact data exists for docked drone
# fleets, so the drone default stays the v3 assumption and is labelled as such.
CONTACT_BETA = {
    "every_contact": 0.822, "ge_1min": 0.523, "ge_2min": 0.204, "ge_5min": 0.103,
    "ge_10min": 0.042, "ge_20min": 0.013,
}
BETA_V3 = {"drone": 0.0015, "av": 0.006}
BETA_SWEEP = tuple(sorted(set(BETA_V3.values()) | set(CONTACT_BETA.values())))
BETA_LABEL = {v: k for k, v in CONTACT_BETA.items()}
BETA_LABEL.update({0.0015: "v3 assumption (drone)", 0.006: "v3 assumption (av)"})

# Documented peer-link rates, Mbps (data/derived/documented_rates.json, "peer_links").
# Ground vehicles: IEEE 802.11p physical-layer rates on a 10 MHz channel (6 Mbps default,
# 27 Mbps top of the rate set; Kenney 2011, Molina-Masegosa and Gozalvez 2017) and the
# field-measured LTE-V2X PC5 effective throughput with commercial radios (0.72 Mbps mean,
# about 2.6 Mbps peak per pair, four nodes on a 20 MHz channel; Wu et al. 2026). No field
# figure exists for 5G NR sidelink, so none is used. Drones: vendor "up to" figures for
# MANET mesh radios flown on UAVs (Silvus SC4200E+ 100, Doodle Labs 80); the deployed
# dock systems have no aircraft-to-aircraft link, so these are the radios that would
# have to be added. The contact gate below turns each rate into the contact length a
# whole-image transfer needs and the share of measured contacts that long.
PEER_RATES = {
    "av": {"LTE-V2X PC5, field mean": 0.72, "LTE-V2X PC5, field peak": 2.6,
           "802.11p default": 6.0, "802.11p top rate": 27.0},
    "drone": {"UAV mesh radio, Doodle Labs": 80.0, "UAV mesh radio, Silvus": 100.0},
}
CONTACT_RANGE = "100m"      # Cabspotting contact range used for the gate
S_FIRMWARE = 0.42           # GB, documented DJI dock aircraft firmware package

CLASSES = {
    "drone": dict(agency="cincinnati-pd", install_t=20.0,
                  r_bh=RATES["mobile_rural_wy"], r_local=RATES["fixed_conus"],
                  depot_rule=("all", None), gossip_beta=BETA_V3["drone"], gossip_seed=0.05,
                  exp_op=30.0, exp_stat=60.0),
    "av": dict(agency="chicago", install_t=30.0,
               r_bh=RATES["mobile_chicago"], r_local=RATES["fixed_conus"],
               depot_rule=("long", 120.0), gossip_beta=CONTACT_BETA["every_contact"], gossip_seed=0.05,
               exp_op=180.0, exp_stat=120.0),
}
ARCHS = ("centralized", "gossip", "depot", "hybrid")
BASELINES = ("nonatomic", "bandwidth")
ALL = ARCHS + BASELINES
MAIN3 = ("centralized", "depot", "hybrid")
CLS_IDX = {"drone": 0, "av": 1}
SRC_IDX = {"exp": 0, "bootstrap": 1}


def tx_min(s_gb, mbps):
    return s_gb * 8000.0 / mbps / 60.0


# ----------------------------------------------------------------------------
# measured contacts (Cabspotting) and the contact gate for the peer path
# ----------------------------------------------------------------------------
_CONTACTS = None


def contacts():
    """Measured contact-length histogram per range and the contact rates
    (data/sim_inputs/contacts_cabspotting.*). Returns ({range: {bin_start_s: events}}, meta)."""
    global _CONTACTS
    if _CONTACTS is None:
        hist = {}
        with open(os.path.join(INPUTS, "contacts_cabspotting.csv"), encoding="utf-8") as f:
            for row in csv.DictReader(f):
                hist.setdefault(row["range_m"] + "m", {})[int(row["duration_bin_start_s"])] = int(row["events"])
        with open(os.path.join(INPUTS, "contacts_cabspotting.json"), encoding="utf-8") as f:
            meta = json.load(f)
        _CONTACTS = (hist, meta)
    return _CONTACTS


def contact_share_ge(minutes, rng_m=CONTACT_RANGE):
    """Share of measured contacts at least `minutes` long. Bins are 30 s wide and a bin
    counts only when it starts at or after the threshold, so the share is conservative
    by less than one bin; the last bin is the overflow at one hour."""
    hist, _ = contacts()
    h = hist[rng_m]
    total = sum(h.values())
    thr = minutes * 60.0
    return sum(c for b, c in h.items() if b >= thr) / total if total else float("nan")


def beta_effective(S, peer_mbps, rng_m=CONTACT_RANGE):
    """Contact rate that can carry the whole image: the measured daytime rate times the
    share of contacts at least as long as the transfer at the peer-link rate.
    Returns (beta_eff per minute, share)."""
    _, meta = contacts()
    share = contact_share_ge(tx_min(S, peer_mbps), rng_m)
    return meta["beta_per_min_daytime"][rng_m] * share, share


# ----------------------------------------------------------------------------
# window sources
# ----------------------------------------------------------------------------
class EmpiricalWindows:
    """Bootstrap resampling from measured window sets."""

    def __init__(self, op, stat, stat_scale=1.0):
        self.op = np.asarray(op, float)
        self.stat = np.asarray(stat, float) * stat_scale
        self.mean_op, self.mean_stat = float(self.op.mean()), float(self.stat.mean())
        self._pop = self.op / self.op.sum()
        self._pst = self.stat / self.stat.sum()

    def draw_op(self, rng, size):
        return rng.choice(self.op, size)

    def draw_stat(self, rng, size):
        return rng.choice(self.stat, size)

    # window in progress at t = 0: length-biased pick, uniform position within it.
    # Returns (full length, remaining part); the full length decides the depot rule.
    def residual_op(self, rng, n):
        full = rng.choice(self.op, n, p=self._pop)
        return full, full * rng.random(n)

    def residual_stat(self, rng, n):
        full = rng.choice(self.stat, n, p=self._pst)
        return full, full * rng.random(n)


class ExpWindows:
    """v3 exponential windows, floored at one minute; memoryless residuals."""

    def __init__(self, op_mean, stat_mean, stat_scale=1.0):
        self.mean_op, self.mean_stat = float(op_mean), float(stat_mean) * stat_scale

    def draw_op(self, rng, size):
        return np.maximum(1.0, rng.exponential(self.mean_op, size))

    def draw_stat(self, rng, size):
        return np.maximum(1.0, rng.exponential(self.mean_stat, size))

    # memoryless: the remaining part is exponential, and so is the elapsed part,
    # so the full length of the window in progress is their sum (length-biased).
    def residual_op(self, rng, n):
        rem = self.draw_op(rng, n)
        return rem + rng.exponential(self.mean_op, n), rem

    def residual_stat(self, rng, n):
        rem = self.draw_stat(rng, n)
        return rem + rng.exponential(self.mean_stat, n), rem


def load_windows(name):
    op, st = [], []
    with open(os.path.join(INPUTS, f"windows_{name}.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            (op if row["kind"] == "operation" else st).append(float(row["minutes"]))
    return np.array(op), np.array(st)


def load_replay(name):
    dev = {}
    with open(os.path.join(INPUTS, f"replay_{name}.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = int(row["device"])
            s = float(row["start_min"])
            dev.setdefault(d, []).append((s, s + float(row["dur_min"])))
    out = {}
    for d, iv in dev.items():
        iv.sort()
        out[d] = (np.array([x[0] for x in iv]), np.array([x[1] for x in iv]))
    return out


_WIN_CACHE = {}


def windows_for(cls, source, agency=None, stat_scale=1.0):
    p = CLASSES[cls]
    if source == "exp":
        return ExpWindows(p["exp_op"], p["exp_stat"], stat_scale)
    name = "av_chicago" if cls == "av" else f"drone_{agency or p['agency']}"
    if name not in _WIN_CACHE:
        _WIN_CACHE[name] = load_windows(name)
    op, st = _WIN_CACHE[name]
    return EmpiricalWindows(op, st, stat_scale)


# ----------------------------------------------------------------------------
# exact engine (rho = 0)
# ----------------------------------------------------------------------------
def depot_flags(rng, rule, st):
    kind, val = rule
    if kind == "all":
        return np.ones(st.shape, dtype=bool)
    if kind == "long":
        return st >= val
    if kind == "prob":
        return rng.random(st.shape) < val
    raise ValueError(rule)


def sequences(rng, win, n, depot_rule):
    """Stationary windows [a, b] (unclipped, n x K) of n devices, and depot flags.
    Devices start in operation with the renewal-process probability; the first
    segment is the remaining part of the window in progress. The depot rule is
    applied to full window lengths, so a long stop that is already under way at
    release still counts as a depot stop."""
    p_op = win.mean_op / (win.mean_op + win.mean_stat)
    in_op = rng.random(n) < p_op
    st = win.draw_stat(rng, (n, K))
    op = win.draw_op(rng, (n, K))
    st_full0, st_rem0 = win.residual_stat(rng, n)
    _, op_rem0 = win.residual_op(rng, n)
    st_full = st.copy()
    st[:, 0] = np.where(in_op, st[:, 0], st_rem0)
    st_full[:, 0] = np.where(in_op, st_full[:, 0], st_full0)
    op[:, 0] = np.where(in_op, op_rem0, op[:, 0])
    cop, cst = np.cumsum(op, 1), np.cumsum(st, 1)
    cop_prev, cst_prev = cop - op, cst - st
    # starting in operation: op0 st0 op1 st1 ...   starting stationary: st0 op0 st1 op1 ...
    a = np.where(in_op[:, None], cop + cst_prev, cst_prev + cop_prev)
    b = a + st
    assert (a[:, -1] >= H_MAX).all(), "K too small to cover the horizon"
    return a, b, st, depot_flags(rng, depot_rule, st_full)


def infection_times(rng, n, I0, beta, deterministic=False):
    u = (np.arange(n) + 0.5) / n if deterministic else rng.random(n)
    ratio = (1 - I0) / I0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -np.log((1 / np.clip(u, 1e-9, 1 - 1e-9) - 1) / ratio) / beta   # beta = 0: seeded devices at 0, the rest never
    return np.clip(np.nan_to_num(t, nan=0.0, posinf=np.inf, neginf=-np.inf), 0, None)


def completion_at(a, b, depot, infect, H, S_values, install, r_bh, r_local, which=ALL):
    """Completion of every architecture in `which` at deadline H over S_values,
    for one set of device window sequences. Returns {name: array over S}."""
    L = np.clip(np.minimum(b, H) - np.maximum(a, 0.0), 0.0, None)
    out = {k: np.empty(len(S_values)) for k in which}
    lmax_all = L.max(1)
    lmax_dep = np.where(depot, L, 0.0).max(1)
    lmax_non = np.where(depot, 0.0, L).max(1)
    if "gossip" in which:
        lo = np.maximum(np.maximum(a, 0.0), infect[:, None])
        g_max = (np.minimum(b, H) - lo).max(1)
        out["gossip"][:] = float((g_max >= install).mean())
    if "nonatomic" in which:
        C = np.cumsum(L, 1) - L
    if "bandwidth" in which:
        L_sum = L.sum(1)
    for i, S in enumerate(S_values):
        t_bh, t_loc, t_best = tx_min(S, r_bh), tx_min(S, r_local), tx_min(S, max(r_local, r_bh))
        if "centralized" in which:
            out["centralized"][i] = (lmax_all >= t_bh + install).mean()
        if "depot" in which:
            out["depot"][i] = (lmax_dep >= t_loc + install).mean()
        if "hybrid" in which:
            out["hybrid"][i] = ((lmax_dep >= t_best + install) | (lmax_non >= t_bh + install)).mean()
        if "nonatomic" in which:
            finish = (C < t_bh) & (C + L >= t_bh)
            rem = L - (t_bh - C)
            out["nonatomic"][i] = ((finish & (rem >= install)) | ((C >= t_bh) & (L >= install))).any(1).mean()
        if "bandwidth" in which:
            out["bandwidth"][i] = (L_sum >= t_bh + install).mean()
    return out


def completion_resumable(a, b, depot, H, S, install, r_bh, r_local, rho, which=MAIN3):
    """Exact completion at deadline H for one update size when transfer progress
    carries across stationary windows in proportion rho (install still atomic).
    Walks the windows in order, vectorised over devices. rho = 0 reproduces
    completion_at for the three architectures; rho = 1 with centralized push
    reproduces the window-aware non-atomic baseline. Returns {name: completion}."""
    L = np.clip(np.minimum(b, H) - np.maximum(a, 0.0), 0.0, None)
    gb_per_min = lambda mbps: mbps * 60.0 / 8000.0
    out = {}
    for arch in which:
        if arch == "centralized":
            rate = np.full(L.shape, gb_per_min(r_bh))
        elif arch == "depot":
            rate = np.where(depot, gb_per_min(r_local), 0.0)
        elif arch == "hybrid":
            rate = np.where(depot, gb_per_min(max(r_local, r_bh)), gb_per_min(r_bh))
        else:
            raise ValueError(arch)
        carry = np.zeros(L.shape[0])
        done = np.zeros(L.shape[0], bool)
        for k in range(L.shape[1]):
            Lk, rk = L[:, k], rate[:, k]
            with np.errstate(divide="ignore", invalid="ignore"):
                t_need = np.where(rk > 0, (S - carry) / rk, np.inf)   # minutes to finish the transfer
            t_need = np.where(carry >= S, 0.0, t_need)
            full = t_need <= Lk                                         # update complete inside this window
            done |= full & (Lk - t_need >= install)
            carry = np.where(full, S, np.minimum(S, carry + rk * Lk)) * rho   # leaving the window
        out[arch] = float(done.mean())
    return out


def _seed(cls, source, r, tag):
    return SEED + r * 1_000_003 + CLS_IDX[cls] * 10_007 + SRC_IDX[source] * 101 + tag * 7_919


def reps(cls, source, R_=None, tag=0, agency=None, install=None, r_bh=None, r_local=None,
         depot_rule=None, beta=None, stat_scale=1.0, S_values=S_GB, S_anchors=S_ANCHORS,
         H_grid=H_GRID, which=ALL):
    """Mean and std over replications of
       vsS: completion at H_MAX over S_values            {arch: (mean[S], std[S])}
       vsH: completion over H_grid at S_anchors           {arch: (mean[H, S], std[H, S])}"""
    R_ = R if R_ is None else R_
    p = CLASSES[cls]
    install = p["install_t"] if install is None else install
    r_bh = p["r_bh"] if r_bh is None else r_bh
    r_local = p["r_local"] if r_local is None else r_local
    depot_rule = p["depot_rule"] if depot_rule is None else depot_rule
    beta = p["gossip_beta"] if beta is None else beta
    win = windows_for(cls, source, agency, stat_scale)
    accS = {k: [] for k in which}
    accH = {k: [] for k in which}
    anchors = np.array(S_anchors)
    for r in range(R_):
        rng = np.random.default_rng(_seed(cls, source, r, tag))
        a, b, st, depot = sequences(rng, win, N, depot_rule)
        inf = infection_times(rng, N, p["gossip_seed"], beta)
        res = completion_at(a, b, depot, inf, H_MAX, S_values, install, r_bh, r_local, which)
        for k in which:
            accS[k].append(res[k])
        curves = {k: np.empty((len(H_grid), len(anchors))) for k in which}
        for hi, H in enumerate(H_grid):
            resH = completion_at(a, b, depot, inf, H, anchors, install, r_bh, r_local, which)
            for k in which:
                curves[k][hi] = resH[k]
        for k in which:
            accH[k].append(curves[k])
    vsS = {k: (np.mean(v, 0), np.std(v, 0)) for k, v in accS.items()}
    vsH = {k: (np.mean(v, 0), np.std(v, 0)) for k, v in accH.items()}
    return vsS, vsH


def s95(S_values, means, target=TARGET):
    ok = np.where(means >= target)[0]
    return float(S_values[ok.max()]) if len(ok) else None


def t95(H_grid, means, target=TARGET):
    ok = np.where(means >= target)[0]
    return float(H_grid[ok.min()]) if len(ok) else None


def fmt(x, unit=""):
    return "none" if x is None else f"{x:.0f}{unit}" if unit == "min" else f"{x:.2f}{unit}"


def margin_crossover(unaware, depot, S_values, thr=MARGIN_V3):
    for i, S in enumerate(S_values):
        if depot[i] - unaware[i] > thr:
            return float(S)
    return None


# ----------------------------------------------------------------------------
# deterministic replay
# ----------------------------------------------------------------------------
def replay(cls, hours=REPLAY_HOURS, install=None, r_bh=None, r_local=None, agency=None,
           S_values=S_GB, S_anchors=S_ANCHORS, H_grid=H_GRID):
    """Every device replays its real schedule from each (day, hour) campaign start.
    Returns {hour: {"vsS": {arch: (all[S], active[S])}, "vsH": {arch: (all[H,S], active[H,S])},
                    "_n": (device_days, active_device_days)}}"""
    p = CLASSES[cls]
    install = p["install_t"] if install is None else install
    r_bh = p["r_bh"] if r_bh is None else r_bh
    r_local = p["r_local"] if r_local is None else r_local
    name = "av_chicago" if cls == "av" else f"drone_{agency or p['agency']}"
    trace = load_replay(name)
    devices = sorted(trace)
    n_dev = len(devices)
    days = int(np.ceil(max(trace[d][1].max() for d in devices) / 1440.0))
    infect = infection_times(None, n_dev, p["gossip_seed"], p["gossip_beta"], deterministic=True)
    kind, val = p["depot_rule"]
    anchors = np.array(S_anchors)
    out = {}
    for hour in hours:
        t0 = np.arange(days) * 1440.0 + hour * 60.0
        gaps = []
        active = np.zeros((n_dev, days), bool)
        for j, d in enumerate(devices):
            fa, fb = trace[d]
            ga = np.concatenate([[-np.inf], fb])
            gb = np.concatenate([fa, [np.inf]])
            dep = np.ones(len(ga), bool) if kind == "all" else ((gb - ga) >= val)
            gaps.append((ga, gb, dep))
            active[j] = ((fa[None, :] < (t0 + H_MAX)[:, None]) & (fb[None, :] > t0[:, None])).any(1)

        def evaluate(H, S_list):
            res = {k: np.zeros((n_dev, days, len(S_list)), bool) for k in ALL}
            for j, (ga, gb, dep) in enumerate(gaps):
                lo = np.maximum(ga[None, :], t0[:, None])
                hi = np.minimum(gb[None, :], (t0 + H)[:, None])
                L = np.clip(hi - lo, 0.0, None)
                lmax_all = L.max(1)
                lmax_dep = np.where(dep[None, :], L, 0.0).max(1)
                lmax_non = np.where(dep[None, :], 0.0, L).max(1)
                g_max = np.clip(hi - np.maximum(lo, (t0 + infect[j])[:, None]), 0.0, None).max(1)
                C = np.cumsum(L, 1) - L
                L_sum = L.sum(1)
                for i, S in enumerate(S_list):
                    t_bh, t_loc = tx_min(S, r_bh), tx_min(S, r_local)
                    res["centralized"][j, :, i] = lmax_all >= t_bh + install
                    res["depot"][j, :, i] = lmax_dep >= t_loc + install
                    res["hybrid"][j, :, i] = (lmax_dep >= t_loc + install) | (lmax_non >= t_bh + install)
                    res["gossip"][j, :, i] = g_max >= install
                    finish = (C < t_bh) & (C + L >= t_bh)
                    rem = L - (t_bh - C)
                    res["nonatomic"][j, :, i] = ((finish & (rem >= install)) | ((C >= t_bh) & (L >= install))).any(1)
                    res["bandwidth"][j, :, i] = L_sum >= t_bh + install
            return res

        resS = evaluate(H_MAX, S_values)
        vsS = {k: (resS[k].mean((0, 1)), resS[k][active].mean(0)) for k in ALL}
        allH = {k: np.empty((len(H_grid), len(anchors))) for k in ALL}
        actH = {k: np.empty((len(H_grid), len(anchors))) for k in ALL}
        for hi, H in enumerate(H_grid):
            r = evaluate(H, anchors)
            for k in ALL:
                allH[k][hi] = r[k].mean((0, 1))
                actH[k][hi] = r[k][active].mean(0)
        out[hour] = {"vsS": vsS, "vsH": {k: (allH[k], actH[k]) for k in ALL},
                     "_n": (n_dev * days, int(active.sum()))}
    return out


# ----------------------------------------------------------------------------
# stepped engine (v3 port) for the resumability sweep; returns the completion curve
# ----------------------------------------------------------------------------
def simulate_stepped(topology, cls, S, rng, win, resume=0.0, install=None, r_bh=None,
                     r_local=None, depot_rule=None):
    p = CLASSES[cls]
    install = p["install_t"] if install is None else install
    r_bh = p["r_bh"] if r_bh is None else r_bh
    r_local = p["r_local"] if r_local is None else r_local
    depot_rule = p["depot_rule"] if depot_rule is None else depot_rule
    rate_bh, rate_loc = S / tx_min(S, r_bh), S / tx_min(S, r_local)   # GB per minute
    p_op = win.mean_op / (win.mean_op + win.mean_stat)
    in_op = rng.random(N) < p_op
    st_full0, st_rem0 = win.residual_stat(rng, N)
    _, op_rem0 = win.residual_op(rng, N)
    rem = np.where(in_op, op_rem0, st_rem0)
    depot_now = depot_flags(rng, depot_rule, st_full0)
    sess_tx = np.zeros(N)
    sess_in = np.zeros(N)
    done = np.zeros(N, bool)
    steps = int(H_MAX / DT)
    curve = np.empty(steps)
    for k in range(steps):
        rem = rem - DT
        active = (~in_op) & ~done
        if topology == "depot":
            rate = np.where(depot_now, rate_loc, 0.0)
        elif topology == "centralized":
            rate = np.full(N, rate_bh)
        else:
            rate = np.where(depot_now, max(rate_loc, rate_bh), rate_bh)
        sess_tx = np.where(active, np.minimum(S, sess_tx + rate * DT), sess_tx)
        installing = active & (sess_tx >= S - 1e-9)
        sess_in = np.where(installing, sess_in + DT, sess_in)
        done = done | ((~done) & (sess_in >= install))
        curve[k] = done.mean()
        flipped = rem <= 0
        if flipped.any():
            to_op = flipped & ~in_op
            to_stat = flipped & in_op
            if to_op.any():
                rem[to_op] = win.draw_op(rng, int(to_op.sum()))
                sess_tx[to_op] = resume * sess_tx[to_op]
                sess_in[to_op] = 0.0
            if to_stat.any():
                new = win.draw_stat(rng, int(to_stat.sum()))
                rem[to_stat] = new
                depot_now[to_stat] = depot_flags(rng, depot_rule, new)
            in_op = np.where(flipped, ~in_op, in_op)
    return curve


def curve_stepped(top, cls, S, R_=None, tag=5, resume=0.0):
    R_ = R_SENS if R_ is None else R_
    win = windows_for(cls, "bootstrap")
    curves = [simulate_stepped(top, cls, S, np.random.default_rng(_seed(cls, "bootstrap", r, tag) + int(S * 100)),
                               win, resume=resume) for r in range(R_)]
    return np.mean(curves, 0)


def t95_curve(curve, target=TARGET):
    ok = np.where(curve >= target)[0]
    return float((ok.min() + 1) * DT) if len(ok) else None


def resumable_curves(cls, rho, S=S_HEADLINE, R_=None, tag=8, which=("centralized", "depot"), H_grid=H_GRID):
    """Completion over H_grid at size S under resumable fraction rho, exact engine,
    bootstrap windows, class defaults. Seeds match reps(cls, "bootstrap", tag=tag),
    so the two engines see the same window sequences. Returns {arch: curve[H]}."""
    R_ = R_SENS if R_ is None else R_
    p = CLASSES[cls]
    win = windows_for(cls, "bootstrap")
    acc = {k: [] for k in which}
    for r in range(R_):
        rng = np.random.default_rng(_seed(cls, "bootstrap", r, tag))
        a, b, _, depot = sequences(rng, win, N, p["depot_rule"])
        kmax = int((a < H_grid[-1]).any(0).sum())          # windows that can start before the last deadline
        curve = {k: np.empty(len(H_grid)) for k in which}
        for hi, H in enumerate(H_grid):
            res = completion_resumable(a[:, :kmax], b[:, :kmax], depot[:, :kmax], H, S,
                                       p["install_t"], p["r_bh"], p["r_local"], rho, which)
            for k in which:
                curve[k][hi] = res[k]
        for k in which:
            acc[k].append(curve[k])
    return {k: np.mean(v, 0) for k, v in acc.items()}


def renewal_pred(S, r_tx, om, sm, install, H):
    tau = tx_min(S, r_tx) + install
    n = H / (om + sm)
    return 1.0 - (1.0 - np.exp(-tau / sm)) ** n


# ----------------------------------------------------------------------------
def main(argv=None):
    import argparse
    global R, R_SENS
    ap = argparse.ArgumentParser(description="Gating-first OTA delivery simulation v4 (measured inputs)")
    ap.add_argument("--quick", action="store_true", help="smoke test: 3 and 2 replications instead of %d and %d" % (R, R_SENS))
    ap.add_argument("--outdir", default=HERE, help="directory for the CSVs and figures (default: next to this file)")
    args = ap.parse_args(argv)
    if args.quick:
        R, R_SENS = 3, 2
    here = args.outdir
    os.makedirs(here, exist_ok=True)
    i_head = list(S_ANCHORS).index(S_HEADLINE)
    i_rep = [int(np.where(H_GRID == h)[0][0]) for h in H_REPORT]
    sweep_rows = []     # sweep, class, param, value, arch, metric, value
    result_rows = []    # class, source, arch, S_gb, H_min, mean, std
    replay_rows = []    # class, hour, arch, S_gb, H_min, all, active, n, n_active

    print(f"v4 MEASURED-INPUT model. R={R}, N={N}, H_max={H_MAX:.0f} min, seed={SEED}, target={TARGET}.")
    print("Rates (Mbps): " + ", ".join(f"{k}={v:.0f}" for k, v in RATES.items()))
    for cls, p in CLASSES.items():
        print(f"  {cls}: agency={p['agency']} install={p['install_t']:.0f} min r_bh={p['r_bh']:.0f} "
              f"r_local={p['r_local']:.0f} depot_rule={p['depot_rule']} beta={p['gossip_beta']}")
    print()

    # --- transfer time by link type ----------------------------------------------------------
    # What each documented or measured link does with the two anchor sizes: the transfer in
    # minutes, and for the links used at a stop the share of measured stationary windows long
    # enough for transfer plus install (one window's chance of completing), for the peer links
    # the share of measured contacts long enough for the transfer alone.
    _, cmeta = contacts()
    print(f"TRANSFER TIME BY LINK TYPE at {S_FIRMWARE:.2f} GB (documented firmware package) and {S_HEADLINE:.0f} GB (headline): "
          f"minutes, then 'windows ok' = share of measured daytime stationary windows >= transfer + install, "
          f"'contacts ok' = share of measured {CONTACT_RANGE} contacts >= transfer (Cabspotting, daytime beta "
          f"{cmeta['beta_per_min_daytime'][CONTACT_RANGE]:.3f}/min)")
    for cls in CLASSES:
        p = CLASSES[cls]
        win = windows_for(cls, "bootstrap")
        links = [("cellular backhaul", p["r_bh"]), ("satellite, documented", RATES["satellite_starlink"]),
                 ("poor cellular, p10", RATES["mobile_p10_conus"]), ("local staging, fixed", p["r_local"])]
        if cls == "drone":
            links.append(("dock Ethernet, spec", RATES["dock_ethernet"]))
        for S in (S_FIRMWARE, S_HEADLINE):
            print(f"   {cls:5s} S={S:.2f} GB, install {p['install_t']:.0f} min:")
            for name, r in links:
                t = tx_min(S, r)
                share = float((win.stat >= t + p["install_t"]).mean())
                sweep_rows.append(("transfer", cls, f"link_S{S:g}", name, f"{r:g} Mbps", "transfer_min", t))
                sweep_rows.append(("transfer", cls, f"link_S{S:g}", name, f"{r:g} Mbps", "share_windows_ok", share))
                print(f"      {name:24s} {r:7.2f} Mbps  transfer {t:6.2f} min  windows ok {share:.2f}")
            for name, r in PEER_RATES[cls].items():
                t = tx_min(S, r)
                sweep_rows.append(("transfer", cls, f"peer_S{S:g}", name, f"{r:g} Mbps", "transfer_min", t))
                if cls == "av":
                    b_eff, share = beta_effective(S, r)
                    sweep_rows.append(("transfer", cls, f"peer_S{S:g}", name, f"{r:g} Mbps", "share_contacts_ok", share))
                    sweep_rows.append(("transfer", cls, f"peer_S{S:g}", name, f"{r:g} Mbps", "beta_eff_per_min", b_eff))
                    print(f"      {name:24s} {r:7.2f} Mbps  transfer {t:6.2f} min  contacts ok {share:.3f}  beta_eff {b_eff:.4f}/min")
                else:
                    print(f"      {name:24s} {r:7.2f} Mbps  transfer {t:6.2f} min  (no contact data for docked fleets)")
    print()

    # --- main --------------------------------------------------------------------------------
    main_res = {}
    for cls in CLASSES:
        main_res[cls] = {}
        for source in ("exp", "bootstrap"):
            vsS, vsH = reps(cls, source)
            main_res[cls][source] = (vsS, vsH)
            for arch in ALL:
                for i, S in enumerate(S_GB):
                    result_rows.append((cls, source, arch, S, H_MAX, vsS[arch][0][i], vsS[arch][1][i]))
                for hi, H in enumerate(H_GRID):
                    for si, S in enumerate(S_ANCHORS):
                        result_rows.append((cls, source, arch, S, H, vsH[arch][0][hi, si], vsH[arch][1][hi, si]))
        rp = replay(cls)
        main_res[cls]["replay"] = rp
        for hour in REPLAY_HOURS:
            r = rp[hour]
            for arch in ALL:
                for i, S in enumerate(S_GB):
                    replay_rows.append((cls, hour, arch, S, H_MAX, r["vsS"][arch][0][i], r["vsS"][arch][1][i], *r["_n"]))
                for hi, H in enumerate(H_GRID):
                    for si, S in enumerate(S_ANCHORS):
                        replay_rows.append((cls, hour, arch, S, H, r["vsH"][arch][0][hi, si], r["vsH"][arch][1][hi, si], *r["_n"]))
        hd = rp[REPLAY_HEADLINE_HOUR]
        for arch in ALL:
            for i, S in enumerate(S_GB):
                result_rows.append((cls, f"replay{REPLAY_HEADLINE_HOUR:02d}", arch, S, H_MAX, hd["vsS"][arch][0][i], 0.0))
            for hi, H in enumerate(H_GRID):
                for si, S in enumerate(S_ANCHORS):
                    result_rows.append((cls, f"replay{REPLAY_HEADLINE_HOUR:02d}", arch, S, H, hd["vsH"][arch][0][hi, si], 0.0))

    def get_curves(cls, source):
        """{arch: (vsS_mean[S], vsH_mean[H, S])} for a window source."""
        if source == "replay":
            r = main_res[cls]["replay"][REPLAY_HEADLINE_HOUR]
            return {k: (r["vsS"][k][0], r["vsH"][k][0]) for k in ALL}
        vsS, vsH = main_res[cls][source]
        return {k: (vsS[k][0], vsH[k][0]) for k in ALL}

    for cls in CLASSES:
        print(f"[{cls}] completion at S={S_HEADLINE:.0f} GB by deadline ({' / '.join(ARCHS)}), "
              f"then T95 at {S_HEADLINE:.0f} GB and S95 at {H_MAX:.0f} min")
        for source in ("exp", "bootstrap", "replay"):
            c = get_curves(cls, source)
            label = f"replay {REPLAY_HEADLINE_HOUR:02d}:00" if source == "replay" else source
            cells = [f"H={H:.0f}: " + "/".join(f"{c[a][1][ih, i_head]:.2f}" for a in ARCHS) for H, ih in zip(H_REPORT, i_rep)]
            print(f"   {label:14s} " + "   ".join(cells))
            print(f"   {'':14s} T95: " + ", ".join(f"{a}={fmt(t95(H_GRID, c[a][1][:, i_head]), 'min')}" for a in ARCHS) +
                  f"   S95: " + ", ".join(f"{a}={fmt(s95(S_GB, c[a][0]))}" for a in ARCHS))
            print(f"   {'':14s} baselines T95: nonatomic={fmt(t95(H_GRID, c['nonatomic'][1][:, i_head]), 'min')} "
                  f"bandwidth={fmt(t95(H_GRID, c['bandwidth'][1][:, i_head]), 'min')}")
            for a in ALL:
                sweep_rows.append(("main", cls, "source", source, a, "T95_min_at_4GB", t95(H_GRID, c[a][1][:, i_head])))
                sweep_rows.append(("main", cls, "source", source, a, "S95_gb_at_360min", s95(S_GB, c[a][0])))
        c = get_curves(cls, "bootstrap")
        best = np.maximum(c["centralized"][0], c["gossip"][0])
        print(f"   v3-style margin crossover at {H_MAX:.0f} min (depot beats best gating-unaware by >{MARGIN_V3}): "
              f"{fmt(margin_crossover(best, c['depot'][0], S_GB))} GB")
        # selector disagreement: deadlines at which the selector's model meets the target but atomic completion does not
        cen, non, bw = c["centralized"][1][:, i_head], c["nonatomic"][1][:, i_head], c["bandwidth"][1][:, i_head]
        dis_non = [float(H) for H, x, y in zip(H_GRID, cen, non) if y >= TARGET and x < TARGET]
        dis_bw = [float(H) for H, x, y in zip(H_GRID, cen, bw) if y >= TARGET and x < TARGET]
        def span(xs):
            if not xs:
                return "none"
            return f"{min(xs):.0f} min" if min(xs) == max(xs) else f"{min(xs):.0f} to {max(xs):.0f} min"
        print(f"   selector disagreement at {S_HEADLINE:.0f} GB (model says push meets {TARGET}, atomic completion does not): "
              f"window-aware non-atomic {span(dis_non)}; bandwidth-only {span(dis_bw)}")
        print()

    # --- replay by campaign start hour -------------------------------------------------------
    print(f"REPLAY by campaign start hour: T95 at {S_HEADLINE:.0f} GB (centralized / depot / hybrid), all device-days [active only]:")
    for cls in CLASSES:
        rp = main_res[cls]["replay"]
        for hour in REPLAY_HOURS:
            r = rp[hour]
            cells = []
            for a in MAIN3:
                cells.append(f"{a}={fmt(t95(H_GRID, r['vsH'][a][0][:, i_head]), 'min')}"
                             f"[{fmt(t95(H_GRID, r['vsH'][a][1][:, i_head]), 'min')}]")
            print(f"   {cls:5s} {hour:02d}:00  " + "  ".join(cells) + f"   device-days={r['_n'][0]} active={r['_n'][1]}")
            for a in MAIN3:
                sweep_rows.append(("replay_hour", cls, "start_hour", hour, a, "T95_min_at_4GB", t95(H_GRID, r["vsH"][a][0][:, i_head])))
    print()

    def sweep(name, cls, param, values, tag, kw_of, which=MAIN3):
        """Run a one-parameter sweep on bootstrap windows; report T95 at the headline size.
        kw_of(value) gives the keyword arguments for reps()."""
        out = {}
        for v in values:
            vsS, vsH = reps(cls, "bootstrap", R_=R_SENS, tag=tag, which=which, S_values=np.array(S_ANCHORS), **kw_of(v))
            out[v] = {a: t95(H_GRID, vsH[a][0][:, i_head]) for a in which}
            for a in which:
                sweep_rows.append((name, cls, param, v, a, "T95_min_at_4GB", out[v][a]))
                sweep_rows.append((name, cls, param, v, a, "completion_at_360min_4GB", float(vsH[a][0][-1, i_head])))
        return out

    print(f"INSTALL TIME (bootstrap windows): T95 at {S_HEADLINE:.0f} GB vs install minutes:")
    install_T = {}
    for cls in CLASSES:
        install_T[cls] = sweep("install", cls, "install_min", INSTALL_SWEEP, 1, kw_of=lambda v: dict(install=v))
        for v in INSTALL_SWEEP:
            print(f"   {cls:5s} install={v:3.0f} min -> " + "  ".join(f"{a}={fmt(install_T[cls][v][a], 'min')}" for a in MAIN3))
    print()

    print(f"BACKHAUL (bootstrap windows): T95 at {S_HEADLINE:.0f} GB vs backhaul Mbps (centralized push; hybrid):")
    bh_T = {}
    for cls in CLASSES:
        bh_T[cls] = sweep("backhaul", cls, "r_bh_mbps", BACKHAUL_SWEEP, 2, which=("centralized", "hybrid"), kw_of=lambda v: dict(r_bh=v))
        for v in BACKHAUL_SWEEP:
            print(f"   {cls:5s} backhaul={v:5.0f} Mbps -> centralized={fmt(bh_T[cls][v]['centralized'], 'min')}  hybrid={fmt(bh_T[cls][v]['hybrid'], 'min')}")
    print()

    print(f"GATE ISOLATION (bootstrap windows scaled, all rates fixed): T95 at {S_HEADLINE:.0f} GB vs stationary-window scale:")
    scale_T = {}
    for cls in CLASSES:
        scale_T[cls] = sweep("scale", cls, "stat_scale", SCALE_SWEEP, 3, kw_of=lambda v: dict(stat_scale=v))
        for v in SCALE_SWEEP:
            print(f"   {cls:5s} scale={v:4.2f} -> " + "  ".join(f"{a}={fmt(scale_T[cls][v][a], 'min')}" for a in MAIN3))
    print()

    print(f"REACH (av, bootstrap windows): T95 at {S_HEADLINE:.0f} GB vs probability that a stationary window has local staging, "
          f"at the metro cellular backhaul ({CLASSES['av']['r_bh']:.0f} Mbps) and at the poor-cellular stress rate "
          f"({RATES['mobile_p10_conus']:.0f} Mbps, CONUS mobile 10th percentile):")
    reach_T = {}
    for label, bh, tag in (("metro", CLASSES["av"]["r_bh"], 4), ("p10", RATES["mobile_p10_conus"], 9)):
        reach_T[label] = sweep("reach_" + label, "av", "p_depot", REACH_SWEEP, tag,
                               kw_of=lambda v, bh=bh: dict(depot_rule=("prob", v), r_bh=bh))
        for v in REACH_SWEEP:
            print(f"   backhaul {bh:3.0f} Mbps  p={v:.1f} -> " + "  ".join(f"{a}={fmt(reach_T[label][v][a], 'min')}" for a in MAIN3))
    print()

    print(f"GOSSIP CONTACT RATE (bootstrap windows): T95 and completion at {H_MAX:.0f} min vs beta "
          f"(Cabspotting, 100 m, daytime; gated values count only contacts at least that long):")
    for cls in CLASSES:
        for bv in BETA_SWEEP:
            vsS, vsH = reps(cls, "bootstrap", R_=R_SENS, tag=6, beta=bv, which=("gossip",), S_values=np.array(S_ANCHORS))
            tv, cv = t95(H_GRID, vsH["gossip"][0][:, i_head]), float(vsH["gossip"][0][-1, i_head])
            sweep_rows.append(("gossip", cls, "beta_per_min", bv, "gossip", "T95_min_at_4GB", tv))
            sweep_rows.append(("gossip", cls, "beta_per_min", bv, "gossip", "completion_at_360min_4GB", cv))
            print(f"   {cls:5s} beta={bv:.4f}/min ({BETA_LABEL.get(bv, ''):22s}) -> T95={fmt(tv, 'min')}  completion={cv:.2f}")
    print()

    print(f"PEER LINK (av, bootstrap windows): gossip T95 when the contact rate counts only measured contacts long enough "
          f"to carry the whole image at a documented vehicle-to-vehicle rate (Cabspotting, {CONTACT_RANGE}, daytime):")
    for S in (S_FIRMWARE, S_HEADLINE):
        for name, r in PEER_RATES["av"].items():
            b_eff, share = beta_effective(S, r)
            vsS, vsH = reps("av", "bootstrap", R_=R_SENS, tag=10, beta=b_eff, which=("gossip",),
                            S_values=np.array([S]), S_anchors=(S,))
            tv, cv = t95(H_GRID, vsH["gossip"][0][:, 0]), float(vsH["gossip"][0][-1, 0])
            for metric, val in (("beta_eff_per_min", b_eff), ("share_contacts_ok", share), ("T95_min", tv), ("completion_at_360min", cv)):
                sweep_rows.append(("peer", "av", f"peer_rate_mbps_S{S:g}", r, "gossip", metric, val))
            print(f"   S={S:.2f} GB  {name:24s} {r:6.2f} Mbps: transfer {tx_min(S, r):6.1f} min, contacts ok {share:.3f}, "
                  f"beta_eff={b_eff:.4f}/min -> T95={fmt(tv, 'min')}  completion={cv:.2f}")
    print(f"   drone: no contact data for docked fleets; at the documented mesh-radio rates a {S_HEADLINE:.0f} GB image needs "
          + ", ".join(f"{tx_min(S_HEADLINE, r):.1f} min at {r:.0f} Mbps" for r in PEER_RATES["drone"].values())
          + " of continuous contact, and the deployed dock systems have no aircraft-to-aircraft link")
    print()

    print(f"AGENCY SPREAD (drone, bootstrap windows per agency): T95 at {S_HEADLINE:.0f} GB:")
    agency_T = sweep("agency", "drone", "agency", AGENCIES, 7, kw_of=lambda v: dict(agency=v))
    for v in AGENCIES:
        print(f"   {v:30s} -> " + "  ".join(f"{a}={fmt(agency_T[v][a], 'min')}" for a in MAIN3))
    print()

    print(f"RESUMABILITY (exact engine, bootstrap windows, install atomic): T95 at {S_HEADLINE:.0f} GB vs resumable fraction rho:")
    rho_T = {cls: {} for cls in CLASSES}
    for cls in CLASSES:
        for rho in RHO_SWEEP:
            curves_r = resumable_curves(cls, rho)
            vals = {a: t95(H_GRID, curves_r[a]) for a in ("centralized", "depot")}
            rho_T[cls][rho] = vals
            for a, v in vals.items():
                sweep_rows.append(("resume", cls, "rho", rho, a, "T95_min_at_4GB", v))
                sweep_rows.append(("resume", cls, "rho", rho, a, "completion_at_360min_4GB", float(curves_r[a][-1])))
            print(f"   {cls:5s} rho={rho:.2f} -> centralized={fmt(vals['centralized'], 'min')}  depot={fmt(vals['depot'], 'min')}")
    print()

    print("CHECKS (engine consistency; the first two are exact and should be 0, the others are statistical):")
    for cls in CLASSES:
        vsS, vsH = reps(cls, "bootstrap", R_=R_SENS, tag=8, which=("centralized", "depot", "nonatomic"),
                        S_values=np.array(S_ANCHORS))
        ex_c, ex_d, ex_n = (vsH[k][0][:, i_head] for k in ("centralized", "depot", "nonatomic"))
        c0 = resumable_curves(cls, 0.0)
        c1 = resumable_curves(cls, 1.0, which=("centralized",))
        d0 = max(np.max(np.abs(c0["centralized"] - ex_c)), np.max(np.abs(c0["depot"] - ex_d)))
        d1 = np.max(np.abs(c1["centralized"] - ex_n))
        print(f"   {cls:5s} resumable engine at rho=0 vs main engine (centralized, depot, {S_HEADLINE:.0f} GB): max |diff| = {d0:.1e}")
        print(f"   {cls:5s} resumable engine at rho=1 vs window-aware non-atomic baseline (centralized): max |diff| = {d1:.1e}")
        stc = curve_stepped("centralized", cls, S_HEADLINE, resume=0.0)
        st = np.array([stc[int(H / DT) - 1] for H in H_GRID])
        print(f"   {cls:5s} v3 stepped engine ({DT:.0f}-min steps, own random stream) vs main engine at rho=0 "
              f"(centralized): max |diff| = {np.max(np.abs(ex_c - st)):.3f}")
    for cls in CLASSES:
        p = CLASSES[cls]
        sim = main_res[cls]["exp"][1]["centralized"][0][:, i_head]
        pred = np.array([renewal_pred(S_HEADLINE, p["r_bh"], p["exp_op"], p["exp_stat"], p["install_t"], H) for H in H_GRID])
        late = H_GRID >= 2 * (p["exp_op"] + p["exp_stat"])
        print(f"   {cls:5s} renewal approximation 1-(1-e^(-tau/m_s))^(H/(m_o+m_s)) vs exp-window simulation "
              f"(centralized, {S_HEADLINE:.0f} GB): |diff| at {H_MAX:.0f} min = {abs(pred[-1] - sim[-1]):.3f}; "
              f"max |diff| beyond two cycles = {('%.3f' % np.max(np.abs(pred - sim)[late])) if late.any() else 'n/a (two cycles exceed the horizon)'}; "
              f"max |diff| overall = {np.max(np.abs(pred - sim)):.3f} (the form ignores the window under way at release "
              f"and the deadline cut, so it is a long-horizon approximation)")
    print()

    # --- CSVs --------------------------------------------------------------------------------
    with open(os.path.join(here, "results_v4.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "source", "arch", "S_gb", "H_min", "mean", "std"])
        for row in result_rows:
            w.writerow([row[0], row[1], row[2], f"{row[3]:.2f}", f"{row[4]:.0f}", f"{row[5]:.3f}", f"{row[6]:.3f}"])
    with open(os.path.join(here, "sweeps_v4.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sweep", "class", "param", "value", "arch", "metric", "result"])
        for row in sweep_rows:
            w.writerow([row[0], row[1], row[2], row[3], row[4], row[5], "" if row[6] is None else f"{row[6]:.3f}"])
    with open(os.path.join(here, "replay_v4.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "start_hour", "arch", "S_gb", "H_min", "completion_all", "completion_active", "device_days", "active_device_days"])
        for row in replay_rows:
            w.writerow([row[0], row[1], row[2], f"{row[3]:.2f}", f"{row[4]:.0f}", f"{row[5]:.3f}", f"{row[6]:.3f}", row[7], row[8]])

    # --- figures -----------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = {"centralized": "tab:blue", "gossip": "tab:orange", "depot": "tab:green", "hybrid": "tab:purple"}
    mk = {"centralized": "o", "gossip": "s", "depot": "^", "hybrid": "D"}
    lab = {"centralized": "centralized push", "gossip": "P2P gossip", "depot": "depot-staged", "hybrid": "hybrid (best link)"}
    cls_ls = {"drone": "-", "av": "--"}
    cls_name = {"drone": "drone (Cincinnati docks)", "av": "AV (Chicago taxi)"}

    # Fig 1: completion vs deadline at the headline size, bootstrap windows, both classes
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for cls in CLASSES:
        vsS, vsH = main_res[cls]["bootstrap"]
        for arch in ARCHS:
            m, s = vsH[arch][0][:, i_head], vsH[arch][1][:, i_head]
            ax.plot(H_GRID, m, ls=cls_ls[cls], marker=mk[arch], color=colors[arch], ms=3.0, lw=1.5, markevery=3,
                    markerfacecolor=(colors[arch] if cls == "drone" else "white"), markeredgecolor=colors[arch])
            ax.fill_between(H_GRID, m - 1.96 * s, m + 1.96 * s, color=colors[arch], alpha=0.08)
    ax.axhline(TARGET, ls=":", c="0.4", lw=0.9)
    ax.text(4, TARGET + 0.012, f"target {TARGET}", ha="left", va="bottom", fontsize=8, color="0.3")
    for cls, y in (("drone", 0.30), ("av", 0.15)):
        v = t95(H_GRID, main_res[cls]["bootstrap"][1]["centralized"][0][:, i_head])
        if v is not None:
            ax.axvline(v, ls=":", c="0.5", lw=0.9)
            ax.annotate(f"{cls.upper()} push\n$T_{{95}}$={v:.0f} min", xy=(v, y), xytext=(v + 6, y), fontsize=8, color="0.25")
    ax.set_xlabel("deadline after release (min)", fontsize=11)
    ax.set_ylabel(f"completion rate, {S_HEADLINE:.0f} GB update", fontsize=11)
    ax.set_ylim(0, 1.02); ax.set_xlim(0, H_MAX); ax.grid(alpha=0.3); ax.tick_params(labelsize=9)
    h1 = [Line2D([], [], color=colors[a], marker=mk[a], ms=4, lw=1.5, label=lab[a]) for a in ARCHS]
    h2 = [Line2D([], [], color="0.2", ls=cls_ls[c], lw=1.5, label=cls_name[c]) for c in CLASSES]
    leg1 = ax.legend(handles=h1, fontsize=7.5, loc="lower right", framealpha=0.9); ax.add_artist(leg1)
    ax.legend(handles=h2, fontsize=7.5, loc="center right", framealpha=0.9)
    fig.tight_layout(); fig.savefig(os.path.join(here, "crossover.png"), dpi=150)

    # Fig 2: drivers, 2 x 2, T95 at the headline size
    fig2, axs = plt.subplots(2, 2, figsize=(5.6, 5.0))
    cap = H_MAX + 20

    def plot_T(ax_, xs, ys, style, color, label):
        yy = [v if v is not None else cap for v in ys]
        ax_.plot(xs, yy, style, color=color, ms=4, label=label)
        for x, v in zip(xs, ys):
            if v is None:
                ax_.annotate(">6 h", xy=(x, cap), fontsize=6.5, color=color, ha="center", va="bottom")

    axa, axb, axc, axd = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    for cls, (sty, col) in zip(CLASSES, (("o-", "tab:purple"), ("s--", "tab:red"))):
        plot_T(axa, INSTALL_SWEEP, [install_T[cls][v]["centralized"] for v in INSTALL_SWEEP], sty, col, cls.upper())
        plot_T(axb, BACKHAUL_SWEEP, [bh_T[cls][v]["centralized"] for v in BACKHAUL_SWEEP], sty, col, cls.upper())
        plot_T(axc, SCALE_SWEEP, [scale_T[cls][v]["centralized"] for v in SCALE_SWEEP], sty, col, cls.upper())
    plot_T(axd, REACH_SWEEP, [reach_T["metro"][v]["depot"] for v in REACH_SWEEP], "^-", "tab:green", "depot only")
    plot_T(axd, REACH_SWEEP, [reach_T["metro"][v]["hybrid"] for v in REACH_SWEEP], "D-", "tab:purple",
           f"hybrid, {CLASSES['av']['r_bh']:.0f} Mbps cell")
    plot_T(axd, REACH_SWEEP, [reach_T["p10"][v]["hybrid"] for v in REACH_SWEEP], "D--", "tab:red",
           f"hybrid, {RATES['mobile_p10_conus']:.0f} Mbps cell")
    axa.set_xlabel("atomic install time (min)", fontsize=9); axa.set_title("(a) install time, push", fontsize=9.5)
    axb.set_xlabel("backhaul rate (Mbps)", fontsize=9); axb.set_title("(b) backhaul, push", fontsize=9.5); axb.set_xscale("log")
    for name, key in (("sat", "satellite_starlink"), ("rural", "mobile_rural_wy"), ("metro", "mobile_chicago"), ("p10", "mobile_p10_conus")):
        axb.axvline(RATES[key], ls=":", c="0.6", lw=0.8)
        axb.text(RATES[key], 5, name, fontsize=6.5, rotation=90, va="bottom", ha="right", color="0.4")
    axc.set_xlabel("stationary-window scale (x measured)", fontsize=9); axc.set_title("(c) gate isolation, rates fixed", fontsize=9.5)
    axd.set_xlabel("share of AV windows with local staging", fontsize=9); axd.set_title("(d) reach (AV)", fontsize=9.5)
    for ax_ in (axa, axb, axc, axd):
        ax_.set_ylabel(f"$T_{{95}}$ (min), {S_HEADLINE:.0f} GB", fontsize=9); ax_.grid(alpha=0.3); ax_.tick_params(labelsize=8)
        ax_.set_ylim(0, cap + 30); ax_.legend(fontsize=7, loc="best")
    fig2.tight_layout(); fig2.savefig(os.path.join(here, "drivers.png"), dpi=150)

    # Fig 3: window-source comparison, centralized push, completion vs deadline
    fig3, axs3 = plt.subplots(1, 2, figsize=(5.6, 2.8), sharey=True)
    src_style = {"exp": (":", "0.5", "exponential (v3 means)"), "bootstrap": ("-", "tab:blue", "bootstrap (measured)"),
                 "replay": ("--", "tab:red", f"replay, {REPLAY_HEADLINE_HOUR:02d}:00 start")}
    for ax_, cls in zip(axs3, CLASSES):
        for src, (ls, col, lb) in src_style.items():
            m = get_curves(cls, src)["centralized"][1][:, i_head]
            ax_.plot(H_GRID, m, ls=ls, color=col, lw=1.5, label=lb)
        ax_.axhline(TARGET, ls=":", c="0.4", lw=0.8)
        ax_.set_title(cls_name[cls] + ", push", fontsize=9); ax_.set_xlabel("deadline after release (min)", fontsize=9)
        ax_.grid(alpha=0.3); ax_.tick_params(labelsize=8); ax_.set_ylim(0, 1.02); ax_.set_xlim(0, H_MAX)
    axs3[0].set_ylabel(f"completion rate, {S_HEADLINE:.0f} GB", fontsize=9); axs3[0].legend(fontsize=7, loc="lower right")
    fig3.tight_layout(); fig3.savefig(os.path.join(here, "sources.png"), dpi=150)

    print("wrote results_v4.csv, sweeps_v4.csv, replay_v4.csv, crossover.png, drivers.png, sources.png")


if __name__ == "__main__":
    main()
