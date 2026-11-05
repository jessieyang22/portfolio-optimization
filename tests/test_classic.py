import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from portopt.classic import (markowitz, max_sharpe, min_volatility, risk_parity, black_litterman,
                             multi_objective, stateful, bl_posterior, make_strategies)
from portopt.common import walk_forward


def _synth(T=1000, seed=0):
    rng = np.random.default_rng(seed)
    vols = np.array([.010, .014, .008, .020, .012])
    mus = np.array([.0004, .0003, .0002, .0008, .0001])
    A = rng.normal(size=(5, 5)) * .3
    corr = np.clip(np.eye(5) + (A + A.T) / 2, -.6, .6)
    np.fill_diagonal(corr, 1.0)
    L = np.linalg.cholesky(corr * np.outer(vols, vols))
    r = mus + rng.normal(size=(T, 5)) @ L.T
    return pd.DataFrame(r, columns=list("ABCDE"), index=pd.bdate_range("2015-01-01", periods=T))


def _valid(w, n):
    return w.shape == (n,) and np.all(w >= -1e-9) and abs(w.sum() - 1) < 1e-6


def test_all_weights_valid():
    h = _synth()
    for name, fn in make_strategies().items():
        w = fn(h)
        assert _valid(w, 5), name


def test_risk_parity_equal_risk_contributions():
    h = _synth()
    w = risk_parity(h)
    from sklearn.covariance import LedoitWolf
    S = LedoitWolf().fit(h.values).covariance_  # same matrix risk_parity optimises on
    share = w * (S @ w) / (w @ S @ w)
    assert np.allclose(share, 1 / 5, atol=1e-4), share


def test_max_sharpe_beats_equal_in_sample():
    h = _synth()
    mu, S = h.mean().values, h.cov().values
    sr = lambda w: (mu @ w) / np.sqrt(w @ S @ w)
    w_ms = max_sharpe(h)
    w_eq = np.ones(5) / 5
    assert sr(w_ms) > sr(w_eq)


def test_black_litterman_no_views_returns_prior():
    h = _synth()
    mkt = np.array([.4, .25, .15, .1, .1])
    pi, post, _ = bl_posterior(h, views={"P": np.zeros((0, 5)), "Q": []}, mkt_w=mkt)
    assert np.allclose(post, pi)
    w = black_litterman(h, views={"P": np.zeros((0, 5)), "Q": []}, mkt_w=mkt)
    assert np.allclose(w, mkt, atol=1e-4), w


def test_black_litterman_default_momentum_views_valid():
    h = _synth()
    w = black_litterman(h)
    assert _valid(w, 5)


def test_multi_objective_turnover_pulls_toward_prev():
    h = _synth()
    prev = np.array([1, 0, 0, 0, 0.0])
    w_free = multi_objective(h, w_turn=0.0)
    w_sticky = multi_objective(h, w_turn=5.0, prev=prev)
    assert _valid(w_sticky, 5)
    assert np.abs(w_sticky - prev).sum() < np.abs(w_free - prev).sum()


def test_stateful_feeds_previous_weights():
    h = _synth()
    fn = stateful(multi_objective, w_turn=5.0)
    w1 = fn(h)
    w2 = fn(h)
    assert np.abs(w2 - w1).sum() < 1e-2  # with a heavy turnover penalty the second call stays put


def test_walk_forward_runs_with_classic_strategies():
    h = _synth(T=800)
    s = walk_forward(h, min_volatility, lookback=252, rebal=63)
    assert len(s) > 300 and np.isfinite(s.values).all()
