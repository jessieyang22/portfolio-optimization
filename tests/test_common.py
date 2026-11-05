import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from portopt.common import solve, walk_forward, metrics

def test_solve_minvar_two_assets():
    cov = np.diag([.04, .01]); w = solve(lambda w: w @ cov @ w, 2)
    assert abs(w.sum() - 1) < 1e-6 and w[1] > w[0]

def test_walk_forward_runs():
    r = pd.DataFrame(np.random.default_rng(0).normal(0, .01, (900, 4)), index=pd.bdate_range("2015-01-01", periods=900))
    s = walk_forward(r, lambda h: np.ones(4) / 4); assert len(s) > 300 and "sharpe" in metrics(s)
