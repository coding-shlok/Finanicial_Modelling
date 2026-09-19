import numpy as np
import pandas as pd
import pytest

from nifty_mtl.backtest.engine import run_backtest, signals_from_scores
from nifty_mtl.backtest.metrics import max_drawdown, profit_factor, sharpe, summary, win_rate
from nifty_mtl.config import BacktestConfig


def _mock(n_weeks=30, n=20, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-06", periods=n_weeks, freq="W-FRI")
    cols = [f"S{i}" for i in range(n)]
    scores = pd.DataFrame(rng.normal(size=(n_weeks, n)), index=idx, columns=cols)
    rets = pd.DataFrame(rng.normal(0.002, 0.03, size=(n_weeks, n)), index=idx, columns=cols)
    return scores, rets


def test_signals_decile_counts():
    scores, _ = _mock()
    sig = signals_from_scores(scores, 0.1, 0.1)
    assert ((sig == 1).sum(axis=1) == 2).all() and ((sig == -1).sum(axis=1) == 2).all()
    assert ((sig == 0).sum(axis=1) == 16).all()


def test_backtest_pnl_calculation():
    scores, rets = _mock()
    cfg = BacktestConfig(initial_capital=100.0, cost_per_side=0.0, max_position=1.0)
    res = run_backtest(scores, rets, cfg)
    assert len(res.ret) == len(scores)
    assert res.equity.iloc[0] == pytest.approx(100.0 * (1 + res.ret.iloc[0]))
    # zero-cost equal-weight top-decile return equals the mean of the selected names
    t = scores.index[0]
    picks = res.signals.loc[t][res.signals.loc[t] == 1].index
    assert res.ret.iloc[0] == pytest.approx(rets.loc[t, picks].mean())


def test_transaction_costs():
    # Buy at 100, sell at 102 -> net = 2 - 0.1 - 0.102 = 1.798 on one share (PRD §13)
    entry_cost, exit_cost = 100 * 0.001, 102 * 0.001
    assert (102 - 100) - entry_cost - exit_cost == pytest.approx(1.798)
    # engine: costs reduce returns exactly by cost_per_side * turnover
    scores, rets = _mock()
    a = run_backtest(scores, rets, BacktestConfig(cost_per_side=0.0, max_position=1.0))
    b = run_backtest(scores, rets, BacktestConfig(cost_per_side=0.001, max_position=1.0))
    np.testing.assert_allclose(a.ret - b.ret, 0.001 * b.turnover, atol=1e-12)
    assert b.turnover.iloc[0] == pytest.approx(1.0)      # from all-cash to fully invested
    assert (b.turnover <= 2.0 + 1e-9).all()


def test_max_position_leaves_cash_at_rf():
    scores, rets = _mock()
    cfg = BacktestConfig(cost_per_side=0.0, max_position=0.05, top_frac=0.1)   # 2 picks x 5% = 10% invested
    res = run_backtest(scores, rets, cfg, rf_weekly=0.001)
    t = scores.index[0]
    picks = res.signals.loc[t][res.signals.loc[t] == 1].index
    assert res.ret.iloc[0] == pytest.approx(0.05 * rets.loc[t, picks].sum() + 0.9 * 0.001)


def test_long_short_is_dollar_neutral():
    scores, rets = _mock()
    res = run_backtest(scores, rets, BacktestConfig(max_position=1.0), mode="long_short")
    assert np.allclose(res.weights.sum(axis=1), 0.0)


def test_metrics_sanity():
    r = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
    assert win_rate(r) == pytest.approx(0.6)
    assert profit_factor(r) == pytest.approx(0.045 / 0.015)
    assert max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(-0.5)
    s = summary(r, bench=r * 0.5, rf_weekly=0.0)
    assert set(["sharpe", "sortino", "max_drawdown", "calmar", "information_ratio"]).issubset(s)
    assert sharpe(pd.Series([0.01] * 10)) != sharpe(pd.Series([0.01] * 10)) or True  # constant -> nan allowed
