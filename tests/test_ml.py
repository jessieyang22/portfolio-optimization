import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from portopt.ml import (hrp, herc, erc, min_var_lw, kmeans_labels, kmeans_cluster_allocation,
                        ensemble, ml_return_forecast)


def _factor_returns(seed=0, T=750):
    """Two blocks (A: assets 0-2 on factor 1, B: assets 3-5 on factor 2). Asset 5 has 3x idiosyncratic vol."""
    rng = np.random.default_rng(seed)
    f = rng.normal(0, .01, (T, 2))
    load = np.zeros((6, 2)); load[:3, 0] = 1; load[3:, 1] = 1
    idio = np.full(6, .005); idio[5] = .015
    r = f @ load.T + rng.normal(0, 1, (T, 6)) * idio
    return pd.DataFrame(r, columns=list("ABCDEF"), index=pd.bdate_range("2015-01-01", periods=T))


def _valid(w, n):
    w = np.asarray(w, float)
    return w.shape == (n,) and np.all(np.isfinite(w)) and np.all(w >= -1e-12) and abs(w.sum() - 1) < 1e-8


def test_hrp_valid_and_penalises_high_vol():
    r = _factor_returns()
    w = hrp(r)
    assert _valid(w, 6)
    assert w[5] < w[3] and w[5] < w[4]


def test_hrp_inverse_variance_for_independent_assets():
    rng = np.random.default_rng(1)
    r = pd.DataFrame(rng.normal(0, [.01, .01, .03], (800, 3)))
    w = hrp(r)
    assert _valid(w, 3) and w[2] < w[0] and w[2] < w[1]


def test_hrp_single_asset():
    r = pd.DataFrame(np.random.default_rng(2).normal(0, .01, (100, 1)))
    assert np.allclose(hrp(r), [1.0])


def test_kmeans_recovers_two_blocks():
    labels = kmeans_labels(_factor_returns())
    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:])) == 1
    assert labels[0] != labels[3]
    assert len(set(labels)) == 2


def test_kmeans_allocation_valid():
    r = _factor_returns()
    assert _valid(kmeans_cluster_allocation(r), 6)


def test_herc_and_erc_valid():
    r = _factor_returns()
    assert _valid(herc(r), 6)
    w = erc(r)
    assert _valid(w, 6)
    # equal risk contributions: every asset's share of portfolio variance should be ~1/6
    cov = np.cov(r.values, rowvar=False); rc = w * (cov @ w); rc = rc / rc.sum()
    assert np.max(rc) / np.min(rc) < 1.05


def test_min_var_lw_valid():
    assert _valid(min_var_lw(_factor_returns()), 6)


def test_ensemble_valid():
    r = _factor_returns()
    w = ensemble(r)
    assert _valid(w, 6)


def test_ml_return_forecast_valid():
    r = _factor_returns(T=800)
    assert _valid(ml_return_forecast(r), 6)
    assert _valid(ml_return_forecast(r, model="ridge"), 6)
