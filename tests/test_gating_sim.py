"""Tests for gating_sim.py (v4).

Run:  python -m pytest tests/        or        python tests/test_gating_sim.py

What is checked
  units            the GB / Mbps / minute conversion
  windows          sequence geometry, the length-biased window in progress, depot flags
  engine           every architecture and baseline against a plain per-device reference
                   implementation, on random sequences
  invariants       monotone in S and in the deadline; bandwidth >= non-atomic >= centralized;
                   hybrid >= centralized and >= depot
  resumability     the exact window-walking engine equals the main engine at rho = 0 and the
                   non-atomic baseline at rho = 1, matches a per-device reference in between,
                   and is monotone in rho
  stepped engine   the v3 one-minute engine agrees with the exact engine within its
                   discretisation bias (statistical check on the measured windows)
  replay           a two-device synthetic trace with hand-computed completion
  summaries        S95 and T95 edge cases, the SI inversion, determinism of the seeds
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import gating_sim as gs  # noqa: E402

RNG = np.random.default_rng(7)


# ----------------------------------------------------------------------------
# plain per-device reference implementation (independent of the vectorised code)
# ----------------------------------------------------------------------------
def ref_windows(a, b, dep, H):
    """(start, end, depot) of the windows that have positive length inside [0, H]."""
    out = []
    for x, y, d in zip(a, b, dep):
        lo, hi = max(x, 0.0), min(y, H)
        if hi > lo:
            out.append((lo, hi, bool(d)))
    return out


def ref_device(a, b, dep, infect, H, S, install, r_bh, r_local):
    t_bh, t_loc, t_best = gs.tx_min(S, r_bh), gs.tx_min(S, r_local), gs.tx_min(S, max(r_local, r_bh))
    wins = ref_windows(a, b, dep, H)
    out = {}
    out["centralized"] = any(hi - lo >= t_bh + install for lo, hi, d in wins)
    out["depot"] = any(d and hi - lo >= t_loc + install for lo, hi, d in wins)
    out["hybrid"] = any((hi - lo >= t_best + install) if d else (hi - lo >= t_bh + install) for lo, hi, d in wins)
    out["gossip"] = any(hi - max(lo, infect) >= install for lo, hi, d in wins)
    prog, non = 0.0, False
    for lo, hi, d in wins:
        L = hi - lo
        if prog >= t_bh:
            if L >= install:
                non = True
                break
        elif prog + L >= t_bh:
            if L - (t_bh - prog) >= install:
                non = True
                break
        prog += L
    out["nonatomic"] = non
    out["bandwidth"] = sum(hi - lo for lo, hi, d in wins) >= t_bh + install
    return out


def ref_resumable(a, b, dep, H, S, install, r_bh, r_local, rho, arch):
    """Per-device walk: transfer at the window's rate, carry rho x progress across windows."""
    gbm = lambda mbps: mbps * 60.0 / 8000.0
    carry = 0.0
    for lo, hi, d in ref_windows(a, b, dep, H):
        L = hi - lo
        if arch == "centralized":
            r = gbm(r_bh)
        elif arch == "depot":
            r = gbm(r_local) if d else 0.0
        else:
            r = gbm(max(r_local, r_bh)) if d else gbm(r_bh)
        if carry >= S:                      # holds the whole image: the install needs no link
            if L >= install:
                return True
        elif r > 0 and carry + r * L >= S:
            t_fin = (S - carry) / r
            if L - t_fin >= install:
                return True
            carry = S
        else:
            carry = min(S, carry + r * L)
        carry *= rho
    return False


def random_sequences(n=300, k=10, p_long=0.3, seed=0):
    """Random stationary windows: some tiny, some long; some start before 0, some after H_MAX."""
    rng = np.random.default_rng(seed)
    st = np.where(rng.random((n, k)) < p_long, rng.uniform(40, 200, (n, k)), rng.uniform(0.1, 40, (n, k)))
    op = rng.uniform(0.1, 60, (n, k))
    start = rng.uniform(-100, 50, n)
    a = start[:, None] + np.cumsum(op, 1) + np.cumsum(st, 1) - st
    b = a + st
    dep = rng.random((n, k)) < 0.4
    infect = rng.uniform(0, 120, n)
    return a, b, dep, infect


# ----------------------------------------------------------------------------
class TestUnits(unittest.TestCase):
    def test_transfer_minutes(self):
        self.assertAlmostEqual(gs.tx_min(4.0, 137.0), 4.0 * 8000 / 137 / 60, places=9)   # 3.89 min
        self.assertAlmostEqual(gs.tx_min(1.0, 1000.0), 8.0 / 60.0, places=12)          # 8 s
        self.assertAlmostEqual(gs.tx_min(10.0, 28.0), 47.619047619, places=6)
        self.assertAlmostEqual(gs.tx_min(0.5, 372.0) * 60.0, 10.7526881720, places=6)   # seconds

    def test_rates_and_classes_consistent(self):
        for cls, p in gs.CLASSES.items():
            self.assertIn(p["r_bh"], gs.RATES.values(), cls)
            self.assertIn(p["r_local"], gs.RATES.values(), cls)
            self.assertGreater(p["install_t"], 0)
            self.assertIn(p["depot_rule"][0], ("all", "long", "prob"))
        self.assertIn(gs.S_HEADLINE, gs.S_ANCHORS)
        self.assertTrue(all(h in gs.H_GRID for h in gs.H_REPORT))


class TestWindows(unittest.TestCase):
    def _check_geometry(self, win, rule):
        rng = np.random.default_rng(3)
        n = 400
        a, b, st, dep = gs.sequences(rng, win, n, rule)
        self.assertEqual(a.shape, (n, gs.K))
        self.assertTrue((a[:, 0] >= 0).all())
        self.assertTrue(np.allclose(b, a + st))
        self.assertTrue((a[:, 1:] >= b[:, :-1]).all(), "stationary windows overlap")
        self.assertTrue((a[:, -1] >= gs.H_MAX).all())
        self.assertEqual(dep.shape, a.shape)
        starts_stat = a[:, 0] == 0.0
        self.assertTrue(0.05 < starts_stat.mean() < 0.95)
        return a, b, st, dep, starts_stat

    def test_empirical_geometry_and_rules(self):
        win = gs.windows_for("av", "bootstrap")
        a, b, st, dep, starts_stat = self._check_geometry(win, ("long", 120.0))
        # columns after the first: flag equals the full-length rule
        self.assertTrue((dep[:, 1:] == (st[:, 1:] >= 120.0)).all())
        # first column: devices that start in operation draw a full window
        self.assertTrue((dep[~starts_stat, 0] == (st[~starts_stat, 0] >= 120.0)).all())
        # devices that start stationary: a flag of False implies the remaining part is short too
        self.assertTrue((st[starts_stat & ~dep[:, 0], 0] < 120.0).all())
        # and some of them are long stops already under way whose remaining part is under 120
        self.assertTrue((starts_stat & dep[:, 0] & (st[:, 0] < 120.0)).any())
        _, _, _, dep_all, _ = self._check_geometry(win, ("all", None))
        self.assertTrue(dep_all.all())
        _, _, _, dep_p, _ = self._check_geometry(win, ("prob", 0.3))
        self.assertTrue(0.25 < dep_p.mean() < 0.35)

    def test_exponential_geometry(self):
        win = gs.windows_for("drone", "exp")
        self._check_geometry(win, ("all", None))

    def test_length_biased_window_in_progress(self):
        win = gs.EmpiricalWindows(op=[10.0, 20.0], stat=[10.0, 100.0])
        rng = np.random.default_rng(11)
        full, rem = win.residual_stat(rng, 40000)
        self.assertAlmostEqual((full == 100.0).mean(), 100.0 / 110.0, delta=0.01)   # by length, not by count
        self.assertTrue((rem <= full).all() and (rem >= 0).all())
        self.assertAlmostEqual((rem / full).mean(), 0.5, delta=0.01)                # uniform position
        full_o, rem_o = win.residual_op(rng, 40000)
        self.assertAlmostEqual((full_o == 20.0).mean(), 20.0 / 30.0, delta=0.01)

    def test_exponential_window_in_progress(self):
        win = gs.ExpWindows(op_mean=60.0, stat_mean=90.0)
        rng = np.random.default_rng(5)
        full, rem = win.residual_stat(rng, 60000)
        self.assertTrue((full >= rem).all())
        self.assertAlmostEqual(rem.mean(), 90.0, delta=2.0)      # memoryless remaining part
        self.assertAlmostEqual(full.mean(), 180.0, delta=3.0)    # length-biased: twice the mean

    def test_stat_scale(self):
        base = gs.windows_for("drone", "bootstrap")
        half = gs.windows_for("drone", "bootstrap", stat_scale=0.5)
        self.assertAlmostEqual(half.mean_stat, 0.5 * base.mean_stat)
        self.assertAlmostEqual(half.mean_op, base.mean_op)

    def test_inputs_are_measured_sets(self):
        for name in ("av_chicago",) + tuple(f"drone_{a}" for a in gs.AGENCIES):
            op, st = gs.load_windows(name)
            self.assertGreater(len(op), 500, name)
            self.assertGreater(len(st), 500, name)
            self.assertTrue((op > 0).all() and (st > 0).all(), name)


class TestEngine(unittest.TestCase):
    def test_all_architectures_match_reference(self):
        a, b, dep, infect = random_sequences()
        install, r_bh, r_local = 20.0, 137.0, 372.0
        S_values = np.array([0.5, 4.0, 10.0, 40.0])
        for H in (30.0, 120.0, 360.0):
            out = gs.completion_at(a, b, dep, infect, H, S_values, install, r_bh, r_local)
            for i, S in enumerate(S_values):
                ref = [ref_device(a[j], b[j], dep[j], infect[j], H, S, install, r_bh, r_local) for j in range(len(a))]
                for arch in gs.ALL:
                    want = np.mean([r[arch] for r in ref])
                    self.assertAlmostEqual(out[arch][i], want, places=12, msg=f"{arch} S={S} H={H}")

    def test_reference_is_not_trivial(self):
        a, b, dep, infect = random_sequences()
        out = gs.completion_at(a, b, dep, infect, 120.0, np.array([4.0]), 20.0, 137.0, 372.0)
        for arch in gs.ALL:
            self.assertTrue(0.02 < out[arch][0] < 0.98, f"{arch} completion {out[arch][0]} is at a bound")

    def test_backhaul_faster_than_local_still_best_link(self):
        a, b, dep, infect = random_sequences(seed=4)
        out = gs.completion_at(a, b, dep, infect, 120.0, np.array([4.0]), 20.0, 1000.0, 100.0, which=("centralized", "hybrid"))
        self.assertGreaterEqual(out["hybrid"][0], out["centralized"][0])

    def test_invariants_on_measured_windows(self):
        for cls in gs.CLASSES:
            vsS, vsH = gs.reps(cls, "bootstrap", R_=2, S_values=np.array([0.5, 2.0, 4.0, 8.0, 40.0, 200.0]))
            for arch in gs.ALL:
                mS, mH = vsS[arch][0], vsH[arch][0]
                self.assertTrue((np.diff(mS) <= 1e-12).all(), f"{cls} {arch} not monotone in S")
                self.assertTrue((np.diff(mH, axis=0) >= -1e-12).all(), f"{cls} {arch} not monotone in H")
                self.assertTrue((np.diff(mH, axis=1) <= 1e-12).all(), f"{cls} {arch} not monotone in anchor size")
            for m in (vsS, vsH):
                self.assertTrue((m["bandwidth"][0] >= m["nonatomic"][0] - 1e-12).all())
                self.assertTrue((m["nonatomic"][0] >= m["centralized"][0] - 1e-12).all())
                self.assertTrue((m["hybrid"][0] >= m["centralized"][0] - 1e-12).all())
                self.assertTrue((m["hybrid"][0] >= m["depot"][0] - 1e-12).all())

    def test_which_subset(self):
        a, b, dep, infect = random_sequences()
        full = gs.completion_at(a, b, dep, infect, 120.0, np.array([4.0]), 20.0, 137.0, 372.0)
        part = gs.completion_at(a, b, dep, infect, 120.0, np.array([4.0]), 20.0, 137.0, 372.0, which=("depot", "bandwidth"))
        self.assertEqual(set(part), {"depot", "bandwidth"})
        for k in part:
            self.assertEqual(part[k][0], full[k][0])


class TestResumable(unittest.TestCase):
    def test_rho_zero_equals_main_engine_and_rho_one_equals_nonatomic(self):
        a, b, dep, infect = random_sequences()
        install, r_bh, r_local = 20.0, 137.0, 372.0
        for H in (60.0, 200.0, 360.0):
            for S in (0.5, 4.0, 20.0):
                main_ = gs.completion_at(a, b, dep, infect, H, np.array([S]), install, r_bh, r_local)
                r0 = gs.completion_resumable(a, b, dep, H, S, install, r_bh, r_local, 0.0)
                for arch in gs.MAIN3:
                    self.assertAlmostEqual(r0[arch], main_[arch][0], places=12, msg=f"rho=0 {arch} S={S} H={H}")
                r1 = gs.completion_resumable(a, b, dep, H, S, install, r_bh, r_local, 1.0, which=("centralized",))
                self.assertAlmostEqual(r1["centralized"], main_["nonatomic"][0], places=12, msg=f"rho=1 S={S} H={H}")

    def test_matches_reference_and_monotone_in_rho(self):
        a, b, dep, infect = random_sequences(seed=9)
        install, r_bh, r_local = 30.0, 50.0, 372.0
        prev = {arch: 0.0 for arch in gs.MAIN3}
        for rho in (0.0, 0.25, 0.5, 0.75, 1.0):
            out = gs.completion_resumable(a, b, dep, 240.0, 12.0, install, r_bh, r_local, rho)
            for arch in gs.MAIN3:
                want = np.mean([ref_resumable(a[j], b[j], dep[j], 240.0, 12.0, install, r_bh, r_local, rho, arch)
                                for j in range(len(a))])
                self.assertAlmostEqual(out[arch], want, places=12, msg=f"{arch} rho={rho}")
                self.assertGreaterEqual(out[arch] + 1e-12, prev[arch], f"{arch} not monotone in rho")
                prev[arch] = out[arch]
        self.assertGreater(prev["centralized"], 0.0)

    def test_resumable_curves_match_reps_at_rho_zero(self):
        for cls in gs.CLASSES:
            vsS, vsH = gs.reps(cls, "bootstrap", R_=2, tag=8, which=("centralized", "depot", "nonatomic"),
                               S_values=np.array(gs.S_ANCHORS))
            i = list(gs.S_ANCHORS).index(gs.S_HEADLINE)
            c0 = gs.resumable_curves(cls, 0.0, R_=2)
            c1 = gs.resumable_curves(cls, 1.0, R_=2, which=("centralized",))
            self.assertTrue(np.allclose(c0["centralized"], vsH["centralized"][0][:, i], atol=1e-12), cls)
            self.assertTrue(np.allclose(c0["depot"], vsH["depot"][0][:, i], atol=1e-12), cls)
            self.assertTrue(np.allclose(c1["centralized"], vsH["nonatomic"][0][:, i], atol=1e-12), cls)


class TestSteppedEngine(unittest.TestCase):
    def test_v3_stepped_engine_agrees_within_discretisation(self):
        # The v3 engine credits whole one-minute steps, so a window earns up to one extra
        # minute and the transfer-completion step also counts towards the install: it runs
        # up to about two minutes optimistic per window. On its own random stream the
        # comparison is statistical; 6 replications of 500 devices put the Monte Carlo
        # spread near 0.01, so 0.06 leaves room for the bias where the curve is steep.
        i = list(gs.S_ANCHORS).index(gs.S_HEADLINE)
        for cls in gs.CLASSES:
            _, vsH = gs.reps(cls, "bootstrap", R_=6, tag=5, which=("centralized",), S_values=np.array(gs.S_ANCHORS))
            exact = vsH["centralized"][0][:, i]
            stc = gs.curve_stepped("centralized", cls, gs.S_HEADLINE, R_=6, resume=0.0)
            stepped = np.array([stc[int(H / gs.DT) - 1] for H in gs.H_GRID])
            self.assertLess(np.max(np.abs(exact - stepped)), 0.06, cls)
            self.assertGreaterEqual(stepped.mean(), exact.mean() - 0.02, f"{cls}: stepped engine should not be pessimistic")


class TestReplay(unittest.TestCase):
    def test_synthetic_two_device_trace(self):
        # device 0 flies 08:30 to 08:45 on day 0; device 1 flies 08:20 to 08:40 on day 1
        fake = {0: (np.array([510.0]), np.array([525.0])),
                1: (np.array([1440.0 + 500.0]), np.array([1440.0 + 520.0]))}
        saved = gs.load_replay
        gs.load_replay = lambda name: fake
        try:
            out = gs.replay("drone", hours=(8,), install=20.0, r_bh=137.0, r_local=372.0,
                            S_values=np.array([4.0]), S_anchors=(4.0,), H_grid=np.array([20.0, 60.0, 120.0]))
        finally:
            gs.load_replay = saved
        r = out[8]
        self.assertEqual(r["_n"], (4, 2))                      # 2 devices x 2 days, 2 active device-days
        need = gs.tx_min(4.0, 137.0) + 20.0                     # 23.9 min
        self.assertTrue(23.0 < need < 24.0)
        allH, actH = r["vsH"]["centralized"]
        # H=20:  a 20-min deadline is shorter than the 23.9 min needed -> nobody
        # H=60:  dev0 day0 completes in the 30 min before its flight; dev0 day1 and dev1 day0 sit all day;
        #        dev1 day1 has 20 min before and 20 min after its flight -> 3 of 4, and 1 of the 2 active
        # H=120: dev1 day1 has 80 min after the flight -> 4 of 4, 2 of 2
        self.assertTrue(np.allclose(allH[:, 0], [0.0, 0.75, 1.0]), allH[:, 0])
        self.assertTrue(np.allclose(actH[:, 0], [0.0, 0.5, 1.0]), actH[:, 0])
        self.assertTrue(np.allclose(r["vsH"]["depot"][0][:, 0], [0.0, 0.75, 1.0]))      # drones: every window is a dock
        self.assertTrue(np.allclose(r["vsS"]["centralized"][0], [1.0]))                 # at 360 min everyone completes
        # bandwidth-only baseline: dev1 day1 at H=60 has 40 min of stationary time in total -> says yes
        self.assertTrue(np.allclose(r["vsH"]["bandwidth"][0][:, 0], [0.0, 1.0, 1.0]))

    def test_replay_is_deterministic_and_covers_days(self):
        out1 = gs.replay("drone", hours=(12,), S_values=np.array([4.0]), S_anchors=(4.0,), H_grid=np.array([60.0, 360.0]))
        out2 = gs.replay("drone", hours=(12,), S_values=np.array([4.0]), S_anchors=(4.0,), H_grid=np.array([60.0, 360.0]))
        for k in gs.ALL:
            self.assertTrue(np.array_equal(out1[12]["vsH"][k][0], out2[12]["vsH"][k][0]))
        n, n_active = out1[12]["_n"]
        self.assertGreater(n_active, 0)
        self.assertLess(n_active, n)


class TestContactGate(unittest.TestCase):
    def test_histogram_loads_and_shares_are_monotone(self):
        hist, meta = gs.contacts()
        self.assertIn(gs.CONTACT_RANGE, hist)
        self.assertGreater(sum(hist[gs.CONTACT_RANGE].values()), 1_000_000)
        self.assertAlmostEqual(gs.contact_share_ge(0.0), 1.0)
        prev = 1.0
        for m in (0.5, 1, 2, 5, 10, 20, 60, 120):
            s = gs.contact_share_ge(m)
            self.assertLessEqual(s, prev + 1e-12, f"share not monotone at {m} min")
            self.assertGreaterEqual(s, 0.0)
            prev = s
        self.assertEqual(gs.contact_share_ge(120.0), 0.0)     # beyond the one-hour overflow bin
        # the module constants were copied from the measured file, to three decimals
        self.assertAlmostEqual(meta["beta_per_min_daytime"][gs.CONTACT_RANGE], gs.CONTACT_BETA["every_contact"], places=3)
        for m, key in ((1, "ge_1min"), (2, "ge_2min"), (5, "ge_5min"), (10, "ge_10min"), (20, "ge_20min")):
            self.assertAlmostEqual(meta["beta_per_min_daytime"][gs.CONTACT_RANGE] * gs.contact_share_ge(m), gs.CONTACT_BETA[key], places=3)

    def test_beta_effective_tracks_rate_and_size(self):
        b_small, s_small = gs.beta_effective(gs.S_FIRMWARE, 27.0)
        b_big, s_big = gs.beta_effective(gs.S_HEADLINE, 27.0)
        self.assertGreater(b_small, b_big)
        self.assertGreater(s_small, s_big)
        b_slow, _ = gs.beta_effective(gs.S_HEADLINE, 6.0)
        self.assertLessEqual(b_slow, b_big)
        for cls, rates in gs.PEER_RATES.items():
            for r in rates.values():
                b, s = gs.beta_effective(gs.S_HEADLINE, r)
                self.assertLessEqual(b, gs.CONTACT_BETA["every_contact"] + 1e-9)
                self.assertTrue(0.0 <= s <= 1.0)

    def test_zero_contact_rate_means_only_seeds_are_infected(self):
        t = gs.infection_times(None, 100, 0.05, 0.0, deterministic=True)
        self.assertEqual(int((t == 0).sum()), 5)
        self.assertTrue(np.isinf(t[5:]).all())
        a, b, dep, _ = random_sequences()
        t = gs.infection_times(None, len(a), 0.05, 0.0, deterministic=True)
        out = gs.completion_at(a, b, dep, t, 360.0, np.array([4.0]), 20.0, 137.0, 372.0, which=("gossip",))
        self.assertLessEqual(out["gossip"][0], 0.05 + 1e-12)     # only the seeded 5 percent can ever complete


class TestSummaries(unittest.TestCase):
    def test_t95_and_s95(self):
        H = np.array([10.0, 20.0, 30.0, 40.0])
        self.assertEqual(gs.t95(H, np.array([0.1, 0.5, 0.95, 1.0])), 30.0)
        self.assertEqual(gs.t95(H, np.array([0.96, 0.97, 0.98, 1.0])), 10.0)
        self.assertIsNone(gs.t95(H, np.array([0.1, 0.5, 0.9, 0.94])))
        S = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(gs.s95(S, np.array([1.0, 0.97, 0.95, 0.8])), 3.0)
        self.assertIsNone(gs.s95(S, np.array([0.9, 0.8, 0.7, 0.6])))
        self.assertEqual(gs.margin_crossover(np.array([0.5, 0.5, 0.5]), np.array([0.6, 0.64, 0.9]), S[:3]), 3.0)
        self.assertIsNone(gs.margin_crossover(np.array([0.5, 0.5]), np.array([0.6, 0.6]), S[:2]))
        self.assertEqual(gs.fmt(None), "none")
        self.assertEqual(gs.fmt(52.4, "min"), "52min")
        self.assertEqual(gs.fmt(3.456), "3.46")

    def test_infection_times_invert_the_si_curve(self):
        I0, beta, n = 0.05, 0.02, 1000
        t = gs.infection_times(None, n, I0, beta, deterministic=True)
        self.assertTrue((np.diff(t) >= 0).all())
        u = (np.arange(n) + 0.5) / n
        I_t = I0 / (I0 + (1 - I0) * np.exp(-beta * t))
        pos = t > 0
        self.assertTrue(np.allclose(I_t[pos], u[pos], atol=1e-9))
        self.assertTrue((t[~pos] == 0).all() and (u[~pos] <= I0).all())   # the seeded share holds it at release
        t2 = gs.infection_times(np.random.default_rng(1), n, I0, beta)
        self.assertTrue((t2 >= 0).all())
        self.assertAlmostEqual((t2 == 0).mean(), I0, delta=0.02)

    def test_renewal_approximation_converges_at_long_horizons(self):
        # Exponential windows, centralized push. The closed form fixes the number of
        # windows at H / (m_o + m_s) and ignores both the window under way at release
        # and the deadline cut, so it is a long-horizon approximation: it is wrong by
        # up to 0.2 inside the first cycle and agrees within about 0.01 at six hours.
        i = list(gs.S_ANCHORS).index(gs.S_HEADLINE)
        for cls in gs.CLASSES:
            p = gs.CLASSES[cls]
            _, vsH = gs.reps(cls, "exp", R_=6, S_values=np.array(gs.S_ANCHORS))
            sim = vsH["centralized"][0][:, i]
            pred = np.array([gs.renewal_pred(gs.S_HEADLINE, p["r_bh"], p["exp_op"], p["exp_stat"], p["install_t"], H)
                             for H in gs.H_GRID])
            self.assertLess(abs(pred[-1] - sim[-1]), 0.03, cls)
            self.assertTrue((np.diff(pred) >= 0).all())
            self.assertTrue((pred >= 0).all() and (pred <= 1).all())

    def test_seeds_are_deterministic_and_distinct(self):
        S = np.array([0.5, 4.0])
        a1 = gs.reps("av", "bootstrap", R_=2, S_values=S, H_grid=np.array([60.0, 360.0]))
        a2 = gs.reps("av", "bootstrap", R_=2, S_values=S, H_grid=np.array([60.0, 360.0]))
        a3 = gs.reps("av", "bootstrap", R_=2, S_values=S, H_grid=np.array([60.0, 360.0]), tag=1)
        for k in gs.ALL:
            self.assertTrue(np.array_equal(a1[0][k][0], a2[0][k][0]))
            self.assertTrue(np.array_equal(a1[1][k][0], a2[1][k][0]))
        self.assertFalse(all(np.array_equal(a1[1][k][0], a3[1][k][0]) for k in gs.ALL))
        seeds = {gs._seed(c, s, r, t) for c in gs.CLASSES for s in gs.SRC_IDX for r in range(gs.R) for t in range(10)}
        self.assertEqual(len(seeds), len(gs.CLASSES) * len(gs.SRC_IDX) * gs.R * 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
