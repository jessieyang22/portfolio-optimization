"""Shared helpers: price loading, walk-forward backtest, metrics, constrained solver."""
import os, warnings, numpy as np, pandas as pd
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
ETF_UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD", "DBC", "VNQ"]
CACHE = os.path.join(os.path.dirname(__file__), ".cache")


def load_returns(tickers=ETF_UNIVERSE, start="2008-01-01"):
    """Daily simple returns, cached to disk. Falls back to cache if offline."""
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, "px_" + "_".join(tickers)[:80] + ".csv")
    try:
        import yfinance as yf
        px = yf.download(tickers, start=start, progress=False, auto_adjust=True)["Close"]
        px.to_csv(f)
    except Exception:
        px = pd.read_csv(f, index_col=0, parse_dates=True)
    px = px[tickers].dropna(how="all")
    px = px.loc[:, px.notna().mean() > .95].ffill().dropna()
    return px.pct_change().dropna()


def solve(obj, n, ub=1.0, lb=0.0, x0=None, extra=()):
    """Long-only, fully-invested minimiser (SLSQP). Returns weights."""
    x0 = np.ones(n) / n if x0 is None else x0
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}, *extra]
    r = minimize(obj, x0, bounds=[(lb, ub)] * n, constraints=cons, method="SLSQP", options={"maxiter": 500})
    w = np.clip(r.x, 0, None)
    return w / w.sum()


def max_drawdown(r):
    c = (1 + np.asarray(r)).cumprod()
    return float((c / np.maximum.accumulate(c) - 1).min())


def metrics(r, periods=252):
    r = np.asarray(r)
    ann = (1 + r).prod() ** (periods / len(r)) - 1
    vol = r.std(ddof=1) * np.sqrt(periods)
    q = np.quantile(r, .05)
    return {"ann_ret": ann, "ann_vol": vol, "sharpe": r.mean() / r.std(ddof=1) * np.sqrt(periods),
            "max_dd": max_drawdown(r), "cvar95_daily": r[r <= q].mean()}


def walk_forward(ret, weight_fn, lookback=504, rebal=21, cost_bps=5):
    """Out-of-sample: fit on trailing `lookback` days, hold `rebal` days, charge turnover cost. Returns daily Series."""
    out, prev = [], None
    for t in range(lookback, len(ret) - 1, rebal):
        w = np.asarray(weight_fn(ret.iloc[t - lookback:t]), float)
        f = ret.iloc[t:t + rebal]
        turn = np.abs(w - prev).sum() if prev is not None else 1.0
        r = f.values @ w
        r[0] -= turn * cost_bps / 1e4
        out.append(pd.Series(r, index=f.index))
        prev = w
    return pd.concat(out)


def compare(ret, strategies: dict, **kw):
    """strategies: name -> weight_fn. Returns metrics DataFrame (+ equal-weight benchmark)."""
    strategies = {"equal_weight": lambda h: np.ones(h.shape[1]) / h.shape[1], **strategies}
    rows = {k: metrics(walk_forward(ret, fn, **kw)) for k, fn in strategies.items()}
    return pd.DataFrame(rows).T
