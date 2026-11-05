"""Hierarchical, clustering and ML-based allocation rules.

Every rule is a weight function  fn(hist_returns: DataFrame) -> np.ndarray  (long-only, sums to 1)
so it plugs straight into common.walk_forward / common.compare.
"""
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from portopt.common import ETF_UNIVERSE, compare, load_returns, solve

ANN = 252


def _stats(h):
    """Daily returns array, sample covariance and correlation (NaN-safe, zero-variance safe)."""
    X = np.nan_to_num(np.asarray(h, float))
    cov = np.atleast_2d(np.cov(X, rowvar=False, ddof=1))
    sd = np.maximum(np.sqrt(np.diag(cov)), 1e-12)
    corr = np.clip(cov / np.outer(sd, sd), -1, 1)
    np.fill_diagonal(corr, 1.0)
    return X, cov, corr


def _dist(corr):
    """Lopez de Prado correlation distance d = sqrt(0.5 (1 - rho))."""
    d = np.sqrt(np.clip(0.5 * (1 - corr), 0, None))
    np.fill_diagonal(d, 0.0)
    return d


def _ivp(cov):
    """Inverse-variance portfolio weights for a covariance block."""
    iv = 1.0 / np.maximum(np.diag(cov), 1e-18)
    return iv / iv.sum()


def _cvar(cov, idx):
    """Variance of the inverse-variance portfolio over the assets in idx."""
    sub = cov[np.ix_(idx, idx)]
    w = _ivp(sub)
    return float(w @ sub @ w)


def hrp(h, method="single"):
    """Hierarchical Risk Parity: correlation-distance tree, quasi-diagonalisation, recursive bisection."""
    _, cov, corr = _stats(h)
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)
    order = leaves_list(linkage(squareform(_dist(corr), checks=False), method=method))
    c = cov[np.ix_(order, order)]  # covariance in quasi-diagonal order
    w = np.ones(n)
    stack = [np.arange(n)]
    while stack:
        idx = stack.pop()
        if len(idx) < 2:
            continue
        half = len(idx) // 2
        left, right = idx[:half], idx[half:]
        v_left, v_right = _cvar(c, left), _cvar(c, right)
        alpha = 1 - v_left / (v_left + v_right)  # more weight to the lower-variance half
        w[left] *= alpha
        w[right] *= 1 - alpha
        stack += [left, right]
    out = np.empty(n)
    out[order] = w
    return out / out.sum()


def herc(h, k=None, method="average"):
    """Simple HERC-style nested risk parity: cut the correlation tree into k clusters,
    inverse-variance weight clusters by their variance, and inverse-variance within each cluster."""
    _, cov, corr = _stats(h)
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)
    k = max(2, n // 3) if k is None else k
    lab = fcluster(linkage(squareform(_dist(corr), checks=False), method=method), t=min(k, n), criterion="maxclust")
    groups = [np.where(lab == g)[0] for g in np.unique(lab)]
    within = np.zeros(n)
    cvar = np.array([_cvar(cov, idx) for idx in groups])
    for idx in groups:
        within[idx] = _ivp(cov[np.ix_(idx, idx)])
    cw = (1 / cvar) / (1 / cvar).sum()
    w = np.zeros(n)
    for j, idx in enumerate(groups):
        w[idx] = cw[j] * within[idx]
    return w / w.sum()


def erc(h):
    """Equal risk contribution (Maillard-Roncalli-Teiletche convex form): minimise 0.5 y'Sy - sum(log y), w = y / sum(y).
    Uses annualised covariance so the objective is O(1) and the optimiser does not stall."""
    _, cov, _ = _stats(h)
    S = cov * ANN
    n = S.shape[0]

    def f(y):
        return 0.5 * y @ S @ y - np.log(y).sum()

    def g(y):
        return S @ y - 1 / y

    r = minimize(f, 1 / np.sqrt(np.diag(S)), jac=g, method="L-BFGS-B", bounds=[(1e-12, None)] * n)
    w = np.clip(r.x, 0, None)
    return w / w.sum()


def min_var_lw(h):
    """Long-only minimum variance using a Ledoit-Wolf shrunk covariance."""
    X, _, _ = _stats(h)
    S = LedoitWolf().fit(X).covariance_ * ANN
    return solve(lambda w: w @ S @ w, X.shape[1])


def _kmeans_labels(corr, k=None, kmax=6, seed=0):
    n = corr.shape[0]
    if n <= 2:
        return np.arange(n)
    if k is not None:
        return KMeans(k, n_init=10, random_state=seed).fit_predict(corr)
    best = None
    for kk in range(2, min(kmax, n - 1) + 1):
        lab = KMeans(kk, n_init=10, random_state=seed).fit_predict(corr)
        if len(np.unique(lab)) < 2:
            continue
        s = silhouette_score(corr, lab)
        if best is None or s > best[0]:
            best = (s, lab)
    return best[1] if best is not None else np.zeros(n, int)


def kmeans_labels(h, k=None, kmax=6, seed=0):
    """Cluster assets on rows of the correlation matrix. k=None picks k in 2..kmax by silhouette."""
    _, _, corr = _stats(h)
    return _kmeans_labels(corr, k, kmax, seed)


def kmeans_cluster_allocation(h, k=None, kmax=6, seed=0):
    """Equal weight across k-means clusters, inverse volatility within each cluster."""
    _, cov, corr = _stats(h)
    n = cov.shape[0]
    lab = _kmeans_labels(corr, k, kmax, seed)
    vol = np.maximum(np.sqrt(np.diag(cov)), 1e-12)
    groups = np.unique(lab)
    w = np.zeros(n)
    for g in groups:
        idx = np.where(lab == g)[0]
        iv = 1 / vol[idx]
        w[idx] = (iv / iv.sum()) / len(groups)
    return w / w.sum()


def ensemble(h):
    """Equal-weight average of HRP, Ledoit-Wolf min-variance and ERC weights."""
    w = np.mean([hrp(h), min_var_lw(h), erc(h)], axis=0)
    return w / w.sum()


def _panel(h):
    """Per-date features (T, N, F) and forward 21-day simple return target (T, N). Features use data up to t only."""
    c = (1 + h).cumprod()
    feats = [c / c.shift(k) - 1 for k in (21, 63, 126)]
    feats.append(h.rolling(63).std())
    feats.append(c / c.rolling(252, min_periods=1).max() - 1)  # drawdown from trailing high
    X = np.stack([f.values for f in feats], axis=-1)
    y = (c.shift(-21) / c - 1).values
    return X, y


def ml_return_forecast(h, model="rf", shrink=0.2, step=5, gamma=5.0, horizon=21):
    """Pooled cross-asset model predicts the next-21d return from lagged features. Forecasts are shrunk
    hard toward zero, then fed into long-only mean-variance (Ledoit-Wolf covariance, risk aversion gamma)."""
    h = pd.DataFrame(h)
    X, y = _panel(h)
    T, N, F = X.shape
    tr = np.arange(T - 1 - horizon, -1, -step)  # only dates whose target is fully inside the window
    Xtr, ytr = X[tr].reshape(-1, F), y[tr].reshape(-1)
    ok = np.isfinite(Xtr).all(1) & np.isfinite(ytr)
    Xtr, ytr = Xtr[ok], ytr[ok]
    xnow = X[-1]
    sc = StandardScaler().fit(Xtr)
    if model == "rf":
        m = RandomForestRegressor(n_estimators=50, max_depth=4, min_samples_leaf=20, random_state=0)
    else:
        m = Ridge(alpha=10.0)
    m.fit(sc.transform(Xtr), ytr)
    pred = m.predict(sc.transform(xnow))
    mu = shrink * pred * ANN / horizon  # annualised, shrunk
    S = LedoitWolf().fit(h.values).covariance_ * ANN
    return solve(lambda w: -(mu @ w) + 0.5 * gamma * (w @ S @ w), N)


def main():
    ret = load_returns(ETF_UNIVERSE)
    print(f"{ret.shape[0]} daily returns, {ret.shape[1]} assets, {ret.index[0].date()} to {ret.index[-1].date()}")
    fast = {"hrp": hrp, "herc": herc, "erc": erc, "min_var_lw": min_var_lw,
            "kmeans": kmeans_cluster_allocation, "ensemble": ensemble}
    table = compare(ret, fast)
    ml = compare(ret, {"ml_rf (rebal=63)": ml_return_forecast}, rebal=63).drop("equal_weight")
    print(pd.concat([table, ml]).round(4).to_string())


if __name__ == "__main__":
    main()
