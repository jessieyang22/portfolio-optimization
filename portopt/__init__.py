"""Portfolio optimization suite: every strategy is a function  weights(hist_returns: DataFrame) -> np.ndarray
(long-only, sums to 1) and plugs into backtest.walk_forward for an apples-to-apples out-of-sample comparison."""
