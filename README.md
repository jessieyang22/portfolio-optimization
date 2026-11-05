# portfolio-optimization

15+ portfolio construction methods, backtested out of sample and reported honestly.

Every strategy is `fn(hist_returns) -> weights` (long-only, sum to 1) and runs through one walk-forward backtest (`common.walk_forward`: 504d lookback, 21d rebalance, 5bp turnover cost) on 12 ETFs, 2008 to 2026. Run any module: `python -m portopt.<module>`. Tests: `python -m pytest tests`  (`pip install -r requirements.txt` first).

| Module | Methods |
|---|---|
| `classic.py` | Markowitz, max Sharpe, Ledoit-Wolf min vol, risk parity (ERC), Black-Litterman, multi-objective |
| `downside.py` | min/mean CVaR (LP), semivariance, drawdown-aware, Calmar, Monte Carlo bootstrap + wealth fan |
| `dynamic.py` | EWMA/GARCH covariance, HMM regime allocation, vol targeting, factor tilt, PCA risk parity |
| `ml.py` | HRP, HERC, random-forest forecasts, k-means clusters, ensemble |

## Honest results (single sample, no significance test; equal weight Sharpe 0.79, vol 10.5%, max DD -22%)
- Best Sharpe: mean_cvar 0.90, ERC 0.88-0.90, k-means 0.88, GARCH risk parity 0.88. Differences are mostly within noise (the Sharpe standard error over 18 years is roughly 0.23).
- Risk-based rules (min var, min CVaR, HRP) cut vol roughly in half and max DD to -15% to -20%, but return falls.
- Did not help: ML return forecasts (0.73), HMM regime (0.37), PCA risk parity (0.36), Calmar, Monte Carlo utility.
- Universe is ETFs, so no survivorship bias, but one 18-year path.
