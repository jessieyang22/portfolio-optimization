"""Classic portfolio construction: each strategy is fn(hist_returns: DataFrame) -> np.ndarray (long-only, sums to 1).

Run the out-of-sample comparison from ~/quant-lab with:  .venv/bin/python -m portopt.classic
"""
import numpy as np, pandas as pd
from sklearn.covariance import LedoitWolf
from portopt.common import solve, load_returns, compare, ETF_UNIVERSE

ANN = 252
MOM_LOOKBACK, MOM_SKIP = 252, 21   # 12-1 month momentum
MOM_VIEW_SCALE = 0.5               # shrink raw momentum before using it as an absolute view
OBJ_SCALE = 1e4                    # SLSQP stops on absolute ftol; scaling the objective makes weights converge tightly


def _solve(obj, n, **kw):
    return solve(lambda w: OBJ_SCALE * obj(w), n, **kw)


def _mu(h):
    return h.mean().values * ANN


def _cov(h):
    return h.cov().values * ANN


def _lw_cov(h):
    return LedoitWolf().fit(h.values).covariance_ * ANN


def markowitz(hist, risk_aversion=3.0):
    """Mean-variance: maximise mu'w - (gamma/2) w'Sigma w (sample covariance, annualised)."""
    mu, S = _mu(hist), _cov(hist)
    return _solve(lambda w: -mu @ w + risk_aversion / 2 * w @ S @ w, hist.shape[1])


def max_sharpe(hist):
    """Tangency portfolio with rf = 0: maximise mu'w / sqrt(w'Sigma w)."""
    mu, S = _mu(hist), _cov(hist)
    return _solve(lambda w: -(mu @ w) / np.sqrt(max(w @ S @ w, 1e-12)), hist.shape[1])


def min_volatility(hist):
    """Global minimum variance on a Ledoit-Wolf shrunk covariance."""
    S = _lw_cov(hist)
    return _solve(lambda w: w @ S @ w, hist.shape[1])


def risk_parity(hist, lb=1e-6):
    """Equal risk contribution: minimise sum_i (RC_i/sigma_p - 1/n)^2 on Ledoit-Wolf covariance.

    RC_i / sigma_p = w_i (Sigma w)_i / (w' Sigma w), which sums to 1 across assets; at the optimum every share is 1/n.
    """
    S = _lw_cov(hist)
    n = hist.shape[1]

    def f(w):
        share = w * (S @ w) / max(w @ S @ w, 1e-12)
        return ((share - 1 / n) ** 2).sum()

    return _solve(f, n, lb=lb)


def momentum_views(hist):
    """Default absolute views: each asset's 12-1 month momentum, annualised and shrunk by MOM_VIEW_SCALE."""
    window = hist.iloc[-(MOM_LOOKBACK + MOM_SKIP):-MOM_SKIP] if len(hist) > MOM_LOOKBACK + MOM_SKIP else hist.iloc[:-MOM_SKIP]
    m = (1 + window).prod().values
    ann = m ** (ANN / len(window)) - 1
    n = hist.shape[1]
    return {"P": np.eye(n), "Q": MOM_VIEW_SCALE * ann}


def bl_posterior(hist, views=None, mkt_w=None, delta=2.5, tau=0.05, omega=None):
    """Black-Litterman posterior mean. Returns (prior_mu, posterior_mu, Sigma).

    views: dict {"P": k x n array, "Q": length-k array} of annualised view returns. None -> momentum_views(hist).
           An empty dict/zero-row P means no views (posterior == prior).
    omega: k x k view-uncertainty matrix; default He-Litterman diag(P tau Sigma P').
    """
    n = hist.shape[1]
    S = _cov(hist)
    w_mkt = np.ones(n) / n if mkt_w is None else np.asarray(mkt_w, float)
    pi = delta * S @ w_mkt
    if views is None:
        views = momentum_views(hist)
    P = np.asarray(views.get("P", np.zeros((0, n))), float).reshape(-1, n)
    Q = np.asarray(views.get("Q", np.zeros(0)), float).reshape(-1)
    if len(Q) == 0:
        return pi, pi, S
    tS = tau * S
    if omega is None:
        omega = np.diag(np.diag(P @ tS @ P.T))
    inv_tS, inv_om = np.linalg.inv(tS), np.linalg.inv(omega)
    A = inv_tS + P.T @ inv_om @ P
    b = inv_tS @ pi + P.T @ inv_om @ Q
    return pi, np.linalg.solve(A, b), S


def black_litterman(hist, views=None, mkt_w=None, delta=2.5, tau=0.05, omega=None):
    """Black-Litterman: prior Pi = delta Sigma w_mkt (equal weights if mkt_w is None), blended with views,
    then the long-only mean-variance optimum under the posterior mean. With no views this returns the prior weights."""
    n = hist.shape[1]
    _, mu, S = bl_posterior(hist, views, mkt_w, delta, tau, omega)
    return _solve(lambda w: -(mu @ w) + delta / 2 * w @ S @ w, n)


def multi_objective(hist, w_ret=1.0, w_var=2.5, w_cvar=0.05, w_turn=0.01, prev=None, alpha=0.05):
    """Weighted sum to minimise:  -w_ret*mu'w + w_var*w'Sigma w + w_cvar*CVaR_alpha(w) + w_turn*|w - prev|_1.

    CVaR is the mean of the worst alpha share of historical daily portfolio returns, annualised (x252) so it
    sits on the same scale as the annual return and variance terms. prev=None drops the turnover term.
    Use the `stateful` wrapper to carry prev across walk-forward rebalances.
    """
    mu, S = _mu(hist), _cov(hist)
    R = hist.values
    k = max(1, int(np.ceil(alpha * len(R))))
    n = hist.shape[1]

    def f(w):
        port = R @ w
        cvar = -np.sort(port)[:k].mean() * ANN
        obj = -w_ret * (mu @ w) + w_var * (w @ S @ w) + w_cvar * cvar
        if prev is not None and w_turn:
            obj += w_turn * np.abs(w - prev).sum()
        return obj

    return _solve(f, n)


def stateful(fn, **kw):
    """Wrap a strategy that takes prev=... so each call feeds the previous rebalance's weights back in."""
    state = {"prev": None}

    def wrapped(hist):
        w = fn(hist, prev=state["prev"], **kw)
        state["prev"] = w
        return w

    return wrapped


def make_strategies():
    """Fresh instances per run; multi_objective is stateful so its turnover term sees the last rebalance."""
    return {
        "markowitz": markowitz,
        "max_sharpe": max_sharpe,
        "min_volatility_lw": min_volatility,
        "risk_parity": risk_parity,
        "black_litterman": black_litterman,
        "multi_objective": stateful(multi_objective),
    }


def main():
    ret = load_returns(ETF_UNIVERSE)
    print(f"{ret.shape[1]} ETFs, {ret.index[0].date()}..{ret.index[-1].date()}, walk-forward (504d fit, 21d hold, 5bp cost)")
    table = compare(ret, make_strategies()).astype(float)
    pct = ["ann_ret", "ann_vol", "max_dd"]
    out = table.copy()
    out[pct] = out[pct].map(lambda x: f"{x:.1%}")
    out[["sharpe"]] = out[["sharpe"]].map(lambda x: f"{x:.2f}")
    out[["cvar95_daily"]] = out[["cvar95_daily"]].map(lambda x: f"{x:.4f}")
    print(out.to_string())
    return table


if __name__ == "__main__":
    main()
