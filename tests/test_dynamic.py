import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from portopt.dynamic import (ewma_cov, garch_cov, minvar, risk_parity, regime_allocation, vol_target,
                             factor_tilt, pca_risk_parity, strategies)


def _ret(T, vols, means=None, seed=0):
    rng = np.random.default_rng(seed)
    vols = np.asarray(vols, float)
    means = np.zeros(len(vols)) if means is None else np.asarray(means, float)
    X = rng.normal(0, 1, (T, len(vols))) * vols + means
    return pd.DataFrame(X, index=pd.bdate_range("2015-01-01", periods=T), columns=[f"A{i}" for i in range(len(vols))])


def _valid(w, n):
    w = np.asarray(w, float)
    assert w.shape == (n,)
    assert np.all(np.isfinite(w))
    assert np.all(w >= -1e-12)
    assert abs(w.sum() - 1) < 1e-8


def test_ewma_cov_symmetric_psd():
    C = ewma_cov(_ret(300, [.01, .02, .03]))
    assert np.allclose(C, C.T)
    assert np.linalg.eigvalsh(C).min() > -1e-12


def test_garch_cov_psd():
    h = _ret(500, [.008, .012, .02])
    C = garch_cov(h)
    assert C.shape == (3, 3)
    assert np.allclose(C, C.T)
    assert np.linalg.eigvalsh(C).min() > -1e-12
    # diagonal should track the per-asset vol scale
    assert C[0, 0] < C[1, 1] < C[2, 2]


def test_minvar_moves_toward_low_vol():
    h = _ret(400, [.004, .02, .02, .02])
    w = minvar(ewma_cov)(h)
    _valid(w, 4)
    # guards against a silent SLSQP no-op that just returns equal weight
    assert w[0] > .4
    assert np.abs(w - .25).max() > .1


def test_risk_parity_valid_and_balanced():
    h = _ret(400, [.005, .01, .02, .04], seed=1)
    w = risk_parity(ewma_cov)(h)
    _valid(w, 4)
    C = ewma_cov(h)
    rc = w * (C @ w)
    rc = rc / rc.sum()
    assert np.abs(rc - .25).max() < .05


def test_garch_weight_fns_valid():
    h = _ret(300, [.005, .01, .02])
    _valid(minvar(garch_cov)(h), 3)
    _valid(risk_parity(garch_cov)(h), 3)


def test_regime_two_regimes_sane():
    rng = np.random.default_rng(3)
    # 4 blocks of 100 days alternating calm/stress. Asset 0 is defensive (steady), asset 1 is risky.
    means_calm, vols_calm = np.array([.0002, .0012]), np.array([.004, .010])
    means_stress, vols_stress = np.array([.0002, -.003]), np.array([.004, .030])

    def make(states):
        rows = []
        for s in states:
            m, v = (means_calm, vols_calm) if s == 0 else (means_stress, vols_stress)
            rows.append(rng.normal(m, v, size=(100, 2)))  # 100 days per block
        X = np.vstack(rows)
        return pd.DataFrame(X, index=pd.bdate_range("2015-01-01", periods=len(X)), columns=["DEF", "RISK"])

    states_end_calm = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]   # ends calm (last block 0)
    states_end_stress = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # ends stress (last block 1)
    w_calm = regime_allocation(make(states_end_calm))
    w_stress = regime_allocation(make(states_end_stress))
    _valid(w_calm, 2)
    _valid(w_stress, 2)
    # risk-off window should put more weight on the defensive asset than the calm window does
    assert w_stress[0] > w_calm[0]


def test_regime_fallback_equal_weight_on_short_or_constant_input():
    h = _ret(10, [.01, .02, .03])
    assert np.allclose(regime_allocation(h), 1 / 3)
    flat = pd.DataFrame(np.zeros((300, 3)))
    _valid(regime_allocation(flat), 3)


def test_vol_target_scales_down_to_target():
    h = _ret(400, [.003, .02, .03])  # lowest-vol asset ~4.8% annual, so a 10% target is reachable
    base = lambda hist: np.ones(3) / 3
    C = ewma_cov(h)
    w = vol_target(base, target=.10)(h)
    _valid(w, 3)
    vol = np.sqrt(w @ C @ w * 252)
    assert abs(vol - .10) < 1e-3
    # the remainder went to the lowest-variance asset
    assert w[0] > 1 / 3


def test_vol_target_no_leverage_when_below_target():
    h = _ret(400, [.002, .002, .002])
    base = lambda hist: np.ones(3) / 3
    assert np.allclose(vol_target(base, target=.10)(h), base(h))


def test_factor_tilt_valid_and_fallback():
    h = _ret(600, [.005, .01, .015, .02, .025], means=[.0002, .0004, .0001, .0006, .0003], seed=4)
    _valid(factor_tilt(h), 5)
    _valid(factor_tilt(h.iloc[:100]), 5)
    assert np.allclose(factor_tilt(h.iloc[:100]), .2)


def test_pca_risk_parity_valid():
    h = _ret(400, [.005, .01, .02, .03, .04], seed=5)
    w = pca_risk_parity(h, k=3)
    _valid(w, 5)


def test_strategies_all_valid_on_synthetic():
    h = _ret(520, [.004, .008, .012, .02, .03, .005], means=[.0002, .0003, .0002, .0005, .0004, .0001], seed=6)
    for name, fn in strategies().items():
        _valid(fn(h), 6)
