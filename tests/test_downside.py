import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from portopt import downside as dn


def _returns(seed=0, T=400, N=5):
    rng = np.random.default_rng(seed)
    mu = np.array([0.0002, 0.0004, 0.0006, 0.0003, 0.0005])
    vol = np.array([0.01, 0.015, 0.02, 0.008, 0.012])
    R = rng.standard_t(4, size=(T, N)) / np.sqrt(2) * vol + mu   # t(4) has variance 2
    return pd.DataFrame(R, columns=[f"A{i}" for i in range(N)])


def _brute_dd(R, w):
    """Additive drawdown path M_t - S_t for t = 1..T, computed directly."""
    S = np.concatenate([[0.0], np.cumsum(R @ w)])
    M = np.maximum.accumulate(S)
    return (M - S)[1:]


def _valid(w, n):
    return w.shape == (n,) and np.all(np.isfinite(w)) and w.min() >= -1e-12 and abs(w.sum() - 1) < 1e-9


def test_cvar_lp_matches_bruteforce():
    R = _returns().values
    w, val = dn._cvar_lp(R, 0.95, None, R.mean(0))
    assert abs(dn.cvar(-R @ w, 0.95) - val) < 1e-6
    rng = np.random.default_rng(1)
    for _ in range(300):   # no random long-only portfolio beats the LP optimum
        assert dn.cvar(-R @ rng.dirichlet(np.ones(R.shape[1])), 0.95) >= val - 1e-6


def test_cvar_lp_min_return_floor():
    R = _returns().values
    mu = R.mean(0)
    floor = mu @ np.ones(5) / 5
    w, _ = dn._cvar_lp(R, 0.95, floor, mu)
    assert mu @ w >= floor - 1e-8


def test_drawdown_lp_matches_bruteforce():
    R = _returns().values
    w, val = dn._drawdown_cvar_lp(R, 0.95)
    assert abs(dn.cvar(_brute_dd(R, w), 0.95) - val) < 1e-6
    assert dn.cvar(_brute_dd(R, w), 0.95) <= dn.cvar(_brute_dd(R, np.ones(5) / 5), 0.95) + 1e-9


def test_min_cvar_beats_equal_weight_insample():
    R = _returns().values
    ew = np.ones(5) / 5
    assert dn.cvar(-R @ dn.min_cvar(R), 0.95) <= dn.cvar(-R @ ew, 0.95) + 1e-6


def test_mean_cvar_beats_equal_weight_and_keeps_return():
    R = _returns().values
    ew = np.ones(5) / 5
    w = dn.mean_cvar(R)
    assert _valid(w, 5)
    assert R.mean(0) @ w >= R.mean(0) @ ew - 1e-8
    assert dn.cvar(-R @ w, 0.95) <= dn.cvar(-R @ ew, 0.95) + 1e-6


def test_semivariance_beats_equal_weight_and_random_search():
    R = _returns().values
    ew = np.ones(5) / 5
    def dsv(w):
        d = np.minimum(R @ w, 0.0)
        return d @ d / len(d)
    w = dn.downside_semivariance(R)
    assert _valid(w, 5)
    assert dsv(w) < dsv(ew) * 0.99   # the optimiser must actually move off equal weight
    rng = np.random.default_rng(2)
    best = min(dsv(v) for v in rng.dirichlet(np.ones(5), 3000))
    assert dsv(w) <= best * (1 + 1e-3)


def test_calmar_beats_equal_weight_ratio():
    R = _returns().values
    ew = np.ones(5) / 5
    def ratio(w):
        return (R.mean(0) @ w) / _brute_dd(R, w).max()
    assert ratio(dn.calmar(R)) >= ratio(ew) * (1 - 1e-4)


def test_monte_carlo_cvar_beats_equal_weight_on_same_paths():
    R = _returns().values
    ew = np.ones(5) / 5
    scen = dn._mc_paths(R, 300, 21, 0).sum(axis=1)
    w = dn.monte_carlo(R, n_paths=300, seed=0)
    assert dn.cvar(-scen @ w, 0.95) <= dn.cvar(-scen @ ew, 0.95) + 1e-6


def test_all_weight_functions_valid():
    R = pd.DataFrame(_returns().values)
    fns = [dn.min_cvar, dn.mean_cvar, dn.downside_semivariance, dn.drawdown_aware, dn.calmar,
           lambda h: dn.monte_carlo(h, n_paths=300),
           lambda h: dn.monte_carlo(h, objective="utility", n_paths=200)]
    for fn in fns:
        assert _valid(np.asarray(fn(R), float), 5)


def test_mc_fan_deterministic_on_constant_returns():
    hist = pd.DataFrame(np.full((300, 2), 0.001))
    out = dn.mc_fan(np.array([0.5, 0.5]), hist, horizon=252, n=100)
    expected = 1.001 ** 252
    assert all(abs(v - expected) < 1e-9 for v in out["terminal_wealth"].values())
    assert out["prob_dd_gt_20"] == 0.0


def test_mc_fan_outputs_sane():
    R = _returns().values
    out = dn.mc_fan(np.ones(5) / 5, R, horizon=252, n=500)
    pct = list(out["terminal_wealth"].values())
    assert all(a <= b for a, b in zip(pct, pct[1:]))
    assert 0.0 <= out["prob_dd_gt_20"] <= 1.0
