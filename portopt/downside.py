"""Downside-risk, drawdown and Monte Carlo portfolio weights.

Every strategy is fn(hist_returns: DataFrame) -> np.ndarray (long-only, sums to 1), so it plugs into
common.walk_forward / common.compare. CVaR, drawdown and Calmar strategies are linear programs solved with
scipy.optimize.linprog (HiGHS). Drawdowns are additive (cumulative sum of daily returns), which keeps them linear.
"""
from functools import partial

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog, minimize

from . import common

BETA = 0.95


def _arr(hist):
    return np.asarray(hist, dtype=float)


def _equal(n):
    return np.ones(n) / n


def _clean(w):
    w = np.clip(np.asarray(w, float), 0, None)
    return w / w.sum()


def _lp(c, A_ub, b_ub, A_eq, b_eq, bounds):
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"linprog failed: {res.message}")
    return res.x, res.fun


def cvar(losses, beta=BETA):
    """Empirical CVaR (expected shortfall) of a loss sample, equal weights, fractional boundary tail.
    Equals the Rockafellar-Uryasev minimum over alpha."""
    L = np.sort(np.asarray(losses, float))[::-1]
    k = (1 - beta) * len(L)
    m = int(np.floor(k))
    tail = L[:m].sum() + ((k - m) * L[m] if k > m else 0.0)
    return tail / k


def _cvar_lp(scen, beta=BETA, min_return=None, mu=None):
    """Min CVaR_beta of the loss -scen @ w over long-only, fully-invested w.

    Rockafellar-Uryasev LP with x = [w (N), alpha, u (S)]:
        min alpha + sum(u) / ((1 - beta) S)
        s.t. u_s >= -scen_s . w - alpha,  u >= 0,  sum(w) = 1,  [mu . w >= min_return].
    Returns (w, lp_objective)."""
    S, N = scen.shape
    nv = N + 1 + S
    c = np.zeros(nv)
    c[N] = 1.0
    c[N + 1:] = 1.0 / ((1 - beta) * S)
    A = sparse.hstack([sparse.csr_matrix(-scen), sparse.csr_matrix(-np.ones((S, 1))),
                       -sparse.identity(S, format="csr")], format="csr")
    b = np.zeros(S)
    if min_return is not None:
        row = np.zeros((1, nv))
        row[0, :N] = -mu
        A = sparse.vstack([A, sparse.csr_matrix(row)], format="csr")
        b = np.append(b, -min_return)
    Aeq = np.zeros((1, nv))
    Aeq[0, :N] = 1.0
    bounds = [(0, None)] * N + [(None, None)] + [(0, None)] * S
    x, val = _lp(c, A, b, Aeq, [1.0], bounds)
    return _clean(x[:N]), val


def min_cvar(hist, beta=BETA, min_return=None):
    """Minimum-CVaR portfolio on historical daily returns, optional expected-return floor."""
    R = _arr(hist)
    w, _ = _cvar_lp(R, beta, min_return, R.mean(0))
    return w


def mean_cvar(hist, beta=BETA):
    """Minimum CVaR subject to expected return >= the equal-weight portfolio's expected return."""
    R = _arr(hist)
    mu = R.mean(0)
    w, _ = _cvar_lp(R, beta, min_return=float(mu @ _equal(R.shape[1])), mu=mu)
    return w


def downside_semivariance(hist):
    """Minimise mean squared downside deviation below a target of 0 (Sortino-style denominator)."""
    R = _arr(hist)
    N = R.shape[1]

    def raw(w):
        d = np.minimum(R @ w, 0.0)
        return float(d @ d) / len(d)

    # SLSQP's stopping tolerance is absolute; the raw objective is ~1e-5, so rescale it to O(1).
    scale = max(raw(_equal(N)), 1e-12)
    return common.solve(lambda w: raw(w) / scale, N)


def _path_rows(C):
    """Constraint rows on [w (N), M (T)] that force M_t = running max of S_t = C_t @ w, with S_0 = 0 and M_0 = 0.
    S_t <= M_t and M_{t-1} <= M_t (M >= 0 comes from variable bounds). Right-hand side is zero."""
    T, N = C.shape
    I = sparse.identity(T, format="csr")
    D = sparse.diags([np.ones(T - 1), -np.ones(T - 1)], [0, 1], shape=(T - 1, T), format="csr")
    top = sparse.hstack([sparse.csr_matrix(C), -I], format="csr")
    mono = sparse.hstack([sparse.csr_matrix((T - 1, N)), D], format="csr")
    return sparse.vstack([top, mono], format="csr")


def _drawdown_cvar_lp(R, beta=BETA):
    """Min CVaR_beta of the additive drawdown path dd_t = M_t - S_t (t = 1..T).
    Variables x = [w (N), M (T), alpha, u (T)]. Returns (w, lp_objective)."""
    T, N = R.shape
    C = np.cumsum(R, axis=0)
    I = sparse.identity(T, format="csr")
    top = sparse.hstack([_path_rows(C), sparse.csr_matrix((2 * T - 1, 1 + T))], format="csr")
    # dd_t <= alpha + u_t  ->  M_t - C_t w - alpha - u_t <= 0
    cv = sparse.hstack([sparse.csr_matrix(-C), I, sparse.csr_matrix(-np.ones((T, 1))), -I], format="csr")
    A = sparse.vstack([top, cv], format="csr")
    b = np.zeros(A.shape[0])
    nv = N + T + 1 + T
    c = np.zeros(nv)
    c[N + T] = 1.0
    c[N + T + 1:] = 1.0 / ((1 - beta) * T)
    Aeq = np.zeros((1, nv))
    Aeq[0, :N] = 1.0
    bounds = [(0, None)] * (N + T) + [(None, None)] + [(0, None)] * T
    x, val = _lp(c, A, b, Aeq, [1.0], bounds)
    return _clean(x[:N]), val


def drawdown_aware(hist, beta=BETA):
    """Minimum CVaR of the additive drawdown path over the lookback window (LP, no compounding)."""
    w, _ = _drawdown_cvar_lp(_arr(hist), beta)
    return w


def _calmar_lp(R, mu, cap=1e3):
    """Max mu.w / maxdrawdown(w) over the simplex, via the Charnes-Cooper scaling: maximise mu.y subject to
    maxdrawdown(y) <= 1, y >= 0, then w = y / sum(y). The cap on sum(y) only guards against unboundedness."""
    T, N = R.shape
    C = np.cumsum(R, axis=0)
    I = sparse.identity(T, format="csr")
    dd = sparse.hstack([sparse.csr_matrix(-C), I], format="csr")          # M_t - C_t y <= 1
    cap_row = sparse.hstack([sparse.csr_matrix(np.ones((1, N))), sparse.csr_matrix((1, T))], format="csr")
    A = sparse.vstack([dd, _path_rows(C), cap_row], format="csr")
    b = np.concatenate([np.ones(T), np.zeros(2 * T - 1), [cap]])
    c = np.concatenate([-np.asarray(mu, float), np.zeros(T)])
    x, _ = _lp(c, A, b, None, None, [(0, None)] * (N + T))
    y = x[:N]
    return _equal(N) if y.sum() < 1e-12 else y / y.sum()


def calmar(hist):
    """Calmar-style portfolio: maximise mean daily return over additive max drawdown in the lookback window."""
    R = _arr(hist)
    return _calmar_lp(R, R.mean(0))


def _block_idx(T, n, horizon, block, rng):
    """Circular block-bootstrap row indices, shape (n, horizon). block=1 is the iid bootstrap."""
    nb = -(-horizon // block)
    starts = rng.integers(0, T, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)) % T
    return idx.reshape(n, nb * block)[:, :horizon]


def _mc_paths(R, n_paths, block, seed):
    """n_paths block-bootstrap resamples of the history, each the same length as the history: (n, T, N)."""
    rng = np.random.default_rng(seed)
    return R[_block_idx(R.shape[0], n_paths, R.shape[0], block, rng)]


def monte_carlo(hist, objective="cvar", n_paths=2000, block=21, gamma=3.0, seed=0, beta=BETA):
    """Optimise on resampled histories.
    objective="cvar": CVaR LP over each path's total (additive) return, one scenario per path.
    objective="utility": maximise mean CRRA utility of compounded terminal wealth, gamma=1 means log utility."""
    R = _arr(hist)
    N = R.shape[1]
    G = _mc_paths(R, n_paths, block, seed)
    if objective == "cvar":
        w, _ = _cvar_lp(G.sum(axis=1), beta)
        return w
    if objective != "utility":
        raise ValueError(f"unknown objective: {objective}")

    def fg(w):
        X = np.maximum(1.0 + G @ w, 1e-8)                 # (n, T) gross daily returns of each path
        L = np.log(X).sum(axis=1)                         # log terminal wealth per path
        g = np.einsum("pt,ptn->pn", 1.0 / X, G)           # dL_p / dw
        if abs(gamma - 1.0) < 1e-9:
            return -L.mean(), -g.mean(axis=0)
        a = np.exp((1.0 - gamma) * L)
        return -a.mean() / (1.0 - gamma), -(a[:, None] * g).mean(axis=0)

    r = minimize(fg, _equal(N), jac=True, method="SLSQP", bounds=[(0, 1)] * N,
                 constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(N)}],
                 options={"maxiter": 300})
    return _clean(r.x)


def mc_fan(weights, hist, horizon=252, n=5000, block=21, seed=0, dd_threshold=0.20):
    """Simulate n block-bootstrap futures of the portfolio over `horizon` days.
    Returns terminal-wealth percentiles (start = 1), the probability that the path's peak-to-trough
    drawdown exceeds dd_threshold at any point in the horizon, and the median of that max drawdown."""
    R = _arr(hist)
    rp = R @ np.asarray(weights, float)
    rng = np.random.default_rng(seed)
    paths = rp[_block_idx(len(rp), n, horizon, block, rng)]
    cum = np.cumprod(1.0 + paths, axis=1)
    peak = np.maximum.accumulate(np.maximum(cum, 1.0), axis=1)
    maxdd = (cum / peak - 1.0).min(axis=1)
    pct = np.percentile(cum[:, -1], [5, 25, 50, 75, 95])
    return {
        "terminal_wealth": dict(zip(["p5", "p25", "p50", "p75", "p95"], map(float, pct))),
        f"prob_dd_gt_{int(round(dd_threshold * 100))}": float((maxdd < -dd_threshold).mean()),
        "median_max_dd": float(np.median(maxdd)),
    }


def main():
    ret = common.load_returns(common.ETF_UNIVERSE)
    strategies = {
        "min_cvar": min_cvar,
        "mean_cvar": mean_cvar,
        "downside_semivar": downside_semivariance,
        "drawdown_aware": drawdown_aware,
        "calmar": calmar,
        "mc_cvar": monte_carlo,
        "mc_utility": partial(monte_carlo, objective="utility"),
    }
    table = common.compare(ret, strategies)
    print(table.round(4).to_string())

    hist = ret.iloc[-504:]
    w = min_cvar(hist)
    print("\nmin_cvar weights (last 504 days):")
    print(pd.Series(w, index=ret.columns).round(4).to_string())
    fan = mc_fan(w, hist, horizon=252, n=5000)
    print("\nmc_fan, 252-day horizon, min_cvar weights:")
    for k, v in fan.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
