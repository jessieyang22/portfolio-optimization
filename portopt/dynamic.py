"""Dynamic / regime-aware allocation strategies on the ETF universe.

Every function here has the shape weights(hist_returns: DataFrame) -> np.ndarray (long-only, sums to 1),
so it plugs straight into common.walk_forward / common.compare.

Contents
  ewma_cov, garch_cov          covariance estimators (inputs to the risk-based weights)
  minvar, risk_parity          factories: cov_fn -> weight function
  regime_allocation            2-3 state Gaussian HMM on a market proxy, blended per-regime mean-variance weights
  vol_target                   overlay: scale a base portfolio down to a target vol
  factor_tilt                  equal weight tilted by momentum (12-1), low-vol and a value proxy
  pca_risk_parity              equal risk across the top-k principal components of the covariance

No-lookahead: every function only sees the history window it is handed. The HMM regime probability is the
last row of predict_proba on that window; the last day has no future days inside the window, so that row is
the forward-filtered posterior (avoids Viterbi lookahead; same idea as a standalone regime filter
and smoothed paths for decisions).
"""
import logging
from functools import lru_cache

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from .common import solve, compare, load_returns, ETF_UNIVERSE

logging.getLogger("hmmlearn").setLevel(logging.ERROR)

PERIODS = 252


def _equal(n):
    return np.ones(n) / n


def _clean_weights(w, n):
    """Final guard: long-only, finite, sums to 1; otherwise equal weight."""
    w = np.asarray(w, float)
    if w.shape != (n,) or not np.all(np.isfinite(w)):
        return _equal(n)
    w = np.clip(w, 0, None)
    s = w.sum()
    return w / s if s > 0 else _equal(n)


def _to_matrix(hist):
    return np.asarray(hist, float)


# ---------------------------------------------------------------- covariance estimators

def ewma_cov(hist, lam=0.94):
    """RiskMetrics-style EWMA covariance of daily returns (decay lam per day)."""
    X = _to_matrix(hist)
    T = len(X)
    w = lam ** np.arange(T)[::-1]
    w = w / w.sum()
    D = X - w @ X
    C = (D * w[:, None]).T @ D
    return (C + C.T) / 2


@lru_cache(maxsize=512)
def _garch_cov_cached(key, shape, cols):
    X = np.frombuffer(key, dtype=float).reshape(shape)
    return _garch_cov_impl(X)


def _garch_cov_impl(X):
    """Per-asset GARCH(1,1) variances (1-step-ahead) + constant correlation of standardized residuals.

    Returns covariance in the units of X (daily returns). Eigenvalues are clipped so the result is PSD.
    """
    from arch import arch_model

    Xp = X * 100.0  # percent returns: arch's optimiser is happier at this scale
    T, n = Xp.shape
    var = np.empty(n)
    Z = np.empty((T, n))
    for i in range(n):
        x = Xp[:, i]
        sample_var = max(float(np.var(x)), 1e-8)
        try:
            res = arch_model(x, mean="Zero", vol="GARCH", p=1, q=1, dist="normal",
                             rescale=False).fit(disp="off", show_warning=False)
            v = float(res.forecast(horizon=1, reindex=False).variance.values[-1, 0])
            z = np.asarray(res.std_resid, float)
            if res.convergence_flag != 0 or not np.isfinite(v) or v <= 0 or not np.all(np.isfinite(z)):
                raise ValueError("bad GARCH fit")
        except Exception:
            v = sample_var
            z = x / np.sqrt(sample_var)
        var[i] = v
        Z[:, i] = z

    R = np.corrcoef(Z, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)
    R = (R + R.T) / 2
    sd = np.sqrt(var)
    C = (sd[:, None] * R * sd[None, :]) / 1e4  # back to return units
    return _psd(C)


def _psd(C, floor=1e-12):
    vals, vecs = np.linalg.eigh((C + C.T) / 2)
    vals = np.clip(vals, floor, None)
    return (vecs * vals) @ vecs.T


def garch_cov(hist):
    """Constant-correlation GARCH(1,1) covariance (per-asset GARCH vols, correlation from standardized residuals)."""
    X = np.ascontiguousarray(_to_matrix(hist))
    return _garch_cov_cached(X.tobytes(), X.shape, tuple(getattr(hist, "columns", range(X.shape[1]))))


# ---------------------------------------------------------------- min-variance / risk parity

def minvar(cov_fn):
    """Long-only global minimum variance using cov_fn. Covariance is rescaled to O(1) before SLSQP."""
    def fn(hist):
        C = cov_fn(hist)
        n = C.shape[0]
        Cs = C / np.mean(np.diag(C))  # SLSQP ftol is absolute; daily variances ~1e-4 would stop early
        return _clean_weights(solve(lambda w: w @ Cs @ w, n, x0=_equal(n)), n)
    return fn


def risk_parity(cov_fn):
    """Equal risk contribution: minimise sum_i (RC_i/total - 1/n)^2 over long-only fully-invested weights."""
    def fn(hist):
        C = cov_fn(hist)
        n = C.shape[0]
        Cs = C / np.mean(np.diag(C))

        def obj(w):
            rc = w * (Cs @ w)
            tot = rc.sum() + 1e-12
            return float(np.sum((rc / tot - 1.0 / n) ** 2)) * 1e3

        return _clean_weights(solve(obj, n, x0=_equal(n)), n)
    return fn


# ---------------------------------------------------------------- regime allocation (HMM)

def _market_features(hist, win=21):
    """Market proxy = equal-weight average return; features = [return, trailing 21d vol], in percent."""
    r_m = _to_matrix(hist).mean(axis=1)
    s = pd.Series(r_m)
    vol = s.rolling(win).std().values
    ok = ~np.isnan(vol)
    X = np.column_stack([r_m[ok], vol[ok]]) * 100.0
    return X, np.where(ok)[0]


def _fit_regime_hmm(X, states=(2, 3), restarts=3, seed=0):
    """Fit each state count with several restarts; choose by BIC. Returns the fitted model or None.

    BLAS is pinned to one thread inside the fit: on these small matrices, multi-threaded BLAS is ~4x slower.
    """
    from threadpoolctl import threadpool_limits

    best = None
    d = X.shape[1]
    with threadpool_limits(limits=1):
        for k in states:
            for j in range(restarts):
                try:
                    m = GaussianHMM(n_components=k, covariance_type="full", n_iter=100,
                                    random_state=seed + j, tol=1e-3)
                    m.fit(X)
                    ll = m.score(X)
                    if not np.isfinite(ll):
                        continue
                    p = (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) // 2
                    bic = -2 * ll + p * np.log(len(X))
                except Exception:
                    continue
                if best is None or bic < best[0]:
                    best = (bic, m)
    return None if best is None else best[1]


def regime_allocation(hist, min_obs=150, gamma=10.0, shrink_mu=0.0, shrink_cov=0.2):
    """Blend per-regime mean-variance weights by the current regime posterior.

    1. Market proxy = equal-weight return and its trailing 21d vol.
    2. Fit a 2 or 3 state Gaussian HMM (BIC choice). Posterior = last row of predict_proba (forward-filtered).
    3. For each state, posterior-weighted asset mean and covariance. shrink_mu / shrink_cov are the weights given to
       the full-sample mean / covariance (0 = pure state estimate). The mean is not shrunk by default: shrinking it
       erases the regime signal (a 50% shrink made the calm-state risky asset look worse than the defensive one).
    4. Per-state long-only mean-variance weights (risk aversion gamma), then w = sum_k p_k * w_k.
    Falls back to equal weight if there is too little data or the fit fails.
    """
    H = _to_matrix(hist)
    n = H.shape[1]
    if len(H) < min_obs:
        return _equal(n)
    try:
        X, rows = _market_features(hist)
        if len(X) < 60:
            return _equal(n)
        m = _fit_regime_hmm(X)
        if m is None:
            return _equal(n)
        post = m.predict_proba(X)  # last row = filtered posterior at the window end
        p = post[-1]
        R = H[rows]  # asset returns aligned with the HMM rows
        mu_all = R.mean(axis=0)
        C_all = np.cov(R, rowvar=False)
        scale = np.mean(np.diag(C_all))
        if not scale > 1e-16:  # flat returns: nothing to optimise
            return _equal(n)
        W = np.zeros((post.shape[1], n))
        for k in range(post.shape[1]):
            g = post[:, k]
            gs = g.sum()
            if gs < 30:  # too little mass for a stable state estimate: use the full-sample moments
                mu_k, C_k = mu_all, C_all
            else:
                mu_k = g @ R / gs
                D = R - mu_k
                C_k = (D * g[:, None]).T @ D / gs
            mu_k = (1 - shrink_mu) * mu_k + shrink_mu * mu_all
            C_k = (1 - shrink_cov) * C_k + shrink_cov * C_all
            C_k = _psd(C_k) / scale
            mu_s = mu_k / scale

            def obj(w, mu_s=mu_s, C_k=C_k):
                return -(w @ mu_s) + 0.5 * gamma * (w @ C_k @ w)

            W[k] = solve(obj, n, x0=_equal(n))
        w = p @ W
        return _clean_weights(w, n)
    except Exception:
        return _equal(n)


# ---------------------------------------------------------------- volatility-target overlay

def vol_target(base_fn, target=0.10, cov_fn=ewma_cov, periods=PERIODS):
    """Scale the base portfolio down to `target` annualised vol. No leverage (cap 1).

    The removed exposure goes to the lowest-variance asset, NOT to zero-return cash. That asset still carries
    risk, so this is a cash substitute only in name. The shift amount is found by bisection so realised
    (estimated) vol equals the target when reachable. If the base is already below target, it is returned
    unchanged. If even the lowest-vol asset alone is above target, the lowest-vol asset is returned.
    """
    def fn(hist):
        w = _clean_weights(base_fn(hist), _to_matrix(hist).shape[1])
        C = cov_fn(hist)
        n = len(w)

        def vol(x):
            return float(np.sqrt(max(x @ C @ x, 0.0) * periods))

        if vol(w) <= target:
            return w
        low = int(np.argmin(np.diag(C)))
        e = np.zeros(n)
        e[low] = 1.0
        if vol(e) >= target:
            return e
        lo, hi = 0.0, 1.0  # s = share kept in the base portfolio; vol(lo) < target < vol(hi)
        for _ in range(60):
            mid = (lo + hi) / 2
            if vol(mid * w + (1 - mid) * e) > target:
                hi = mid
            else:
                lo = mid
        return _clean_weights(lo * w + (1 - lo) * e, n)
    return fn


# ---------------------------------------------------------------- factor tilt

def factor_tilt(hist, tilt=0.5):
    """Equal weight tilted by three cross-sectional z-scores (averaged):
      momentum 12-1 : compounded return from t-252 to t-21
      low-vol       : negative 252d volatility
      value proxy   : negative trailing (whole window) return, i.e. short-term reversal; not a book-to-price value factor
    Weights = (1/n) * (1 + tilt * score), floored at 0 and renormalised.
    """
    H = _to_matrix(hist)
    T, n = H.shape
    if T < 253:
        return _equal(n)

    def z(x):
        return (x - x.mean()) / (x.std() + 1e-12)

    mom = np.prod(1 + H[-252:-21], axis=0) - 1
    lowvol = -H[-252:].std(axis=0, ddof=1)
    value = -(np.prod(1 + H, axis=0) - 1)
    score = np.clip((z(mom) + z(lowvol) + z(value)) / 3.0, -2, 2)
    w = (1.0 / n) * (1 + tilt * score)
    return _clean_weights(w, n)


# ---------------------------------------------------------------- PCA statistical-factor risk parity

def pca_risk_parity(hist, k=3, cov_fn=ewma_cov):
    """Equal risk across the top-k principal components plus a residual bucket (k+1 buckets, each 1/(k+1)).

    Variance from PC j is lambda_j * (v_j' w)^2 (eigenvectors are orthogonal). The residual bucket is
    w' (C - V_k L_k V_k') w, the variance the top-k factors do not explain. Without it the optimiser can pile
    weight into assets that load weakly on the top-k PCs (this happened on the ETF data: 90% in one ETF).
    """
    C = cov_fn(hist)
    n = C.shape[0]
    k = min(k, n - 1)
    vals, vecs = np.linalg.eigh(C)
    idx = np.argsort(vals)[::-1][:k]
    lam = np.clip(vals[idx], 1e-12, None)
    V = vecs[:, idx]
    resid = (C - (V * lam) @ V.T)
    resid = (resid + resid.T) / 2
    Cs_scale = np.mean(np.diag(C))
    nb = k + 1

    def obj(w):
        f = np.append(lam * (V.T @ w) ** 2, max(float(w @ resid @ w), 0.0)) / Cs_scale
        s = f / (f.sum() + 1e-12)
        return float(np.sum((s - 1.0 / nb) ** 2))

    # The objective is non-convex: SLSQP from equal weight stalls at a poor point (objective 0.086 vs 0.000 found
    # from a random start). So run SLSQP from equal weight plus the best few of a seeded random cloud.
    rng = np.random.default_rng(0)
    cloud = np.vstack([rng.dirichlet(np.ones(n) * 0.3, size=200), _equal(n)])
    starts = [cloud[i] for i in np.argsort([obj(x) for x in cloud])[:3]]
    best = None
    for x0 in starts:
        w = _clean_weights(solve(obj, n, x0=x0), n)
        if best is None or obj(w) < obj(best):
            best = w
    return _clean_weights(best, n)


# ---------------------------------------------------------------- strategy set and main

def strategies():
    rp_ewma = risk_parity(ewma_cov)
    return {
        "ewma_minvar": minvar(ewma_cov),
        "ewma_riskparity": rp_ewma,
        "garch_minvar": minvar(garch_cov),
        "garch_riskparity": risk_parity(garch_cov),
        "regime_hmm": regime_allocation,
        "vol_target_10pct_rp": vol_target(rp_ewma, target=0.10),
        "factor_tilt": factor_tilt,
        "pca_riskparity_k3": pca_risk_parity,
    }


def main():
    ret = load_returns(ETF_UNIVERSE)
    print(f"Universe: {ret.shape[1]} ETFs, {len(ret)} days, {ret.index[0].date()} to {ret.index[-1].date()}")
    table = compare(ret, strategies())
    pd.set_option("display.width", 120)
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
