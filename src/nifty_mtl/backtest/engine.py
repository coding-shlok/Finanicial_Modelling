"""Portfolio simulator with transaction costs (PRD §8).

Timeline for decision week t (Friday close):
    scores[t]        -> signals (BUY = top decile, SELL = bottom decile, HOLD = rest)
    target weights   -> trade at Friday close, paying cost_per_side * |Δw|
    realised P&L     -> simple returns of week t+1

Modes
-----
* ``rebalance``  : hold exactly the BUY set each week, equal weight (PRD default)
* ``hold_zone``  : keep a held stock while its rank is above the median; exit on
                   SELL; add new BUYs; equal weight across holdings (lower turnover)
* ``long_short`` : long top decile / short bottom decile, dollar-neutral (secondary result)

Weights per name are capped at ``max_position``; the remainder sits in cash
earning the weekly risk-free rate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nifty_mtl.config import BacktestConfig


@dataclass
class BacktestResult:
    ret: pd.Series                  # net weekly portfolio returns, indexed by the *realised* week (t+1)
    gross_ret: pd.Series
    turnover: pd.Series
    costs: pd.Series
    weights: pd.DataFrame           # target weights at decision week t
    signals: pd.DataFrame           # +1 BUY / -1 SELL / 0 HOLD at decision week t
    equity: pd.Series

    @property
    def n_weeks(self) -> int:
        return len(self.ret)


def signals_from_scores(scores: pd.DataFrame, top_frac: float, bottom_frac: float) -> pd.DataFrame:
    sig = pd.DataFrame(0, index=scores.index, columns=scores.columns, dtype=int)
    for t, row in scores.iterrows():
        r = row.dropna()
        if len(r) == 0:
            continue
        n_top = max(1, int(round(top_frac * len(r))))
        n_bot = max(1, int(round(bottom_frac * len(r))))
        ranked = r.sort_values(ascending=False)
        sig.loc[t, ranked.index[:n_top]] = 1
        sig.loc[t, ranked.index[-n_bot:]] = -1
    return sig


def _equal_weights(names, cap: float, n_cols: pd.Index) -> pd.Series:
    w = pd.Series(0.0, index=n_cols)
    if len(names):
        w[list(names)] = min(1.0 / len(names), cap)
    return w


def run_backtest(scores: pd.DataFrame, simple_ret_next: pd.DataFrame, cfg: BacktestConfig | None = None,
                 rf_weekly: float = 0.0, mode: str = "rebalance") -> BacktestResult:
    """
    scores           : [decision week t x symbol], NaN where no prediction
    simple_ret_next  : [decision week t x symbol] simple return realised over week t+1
    """
    cfg = cfg or BacktestConfig()
    cols = scores.columns
    sig = signals_from_scores(scores, cfg.top_frac, cfg.bottom_frac)
    w_prev = pd.Series(0.0, index=cols)        # weights held going into the decision (post-drift)
    held: set = set()
    rets, gross, turns, costs, W = [], [], [], [], []
    weeks = scores.index

    for t in weeks:
        row = scores.loc[t].dropna()
        if len(row) == 0:
            continue
        s = sig.loc[t]
        buys = set(s[s == 1].index)
        sells = set(s[s == -1].index)
        if mode == "rebalance":
            target = _equal_weights(buys, cfg.max_position, cols)
        elif mode == "hold_zone":
            ranks = row.rank(ascending=False)
            keep = {n for n in held if n in ranks.index and ranks[n] <= len(row) / 2 and n not in sells}
            held = keep | buys
            target = _equal_weights(held, cfg.max_position, cols)
        elif mode == "long_short":
            wl = _equal_weights(buys, cfg.max_position, cols)
            ws = _equal_weights(sells, cfg.max_position, cols)
            target = wl - ws
        else:
            raise ValueError(mode)

        turnover = float((target - w_prev).abs().sum())
        cost = cfg.cost_per_side * turnover
        r_next = simple_ret_next.loc[t].reindex(cols).fillna(0.0)
        invested = float(target.abs().sum()) if mode != "long_short" else 0.0
        cash = max(0.0, 1.0 - invested)
        g = float((target * r_next).sum()) + cash * rf_weekly
        net = g - cost
        rets.append(net); gross.append(g); turns.append(turnover); costs.append(cost); W.append(target)

        # drift weights through the week so next turnover is measured vs. what we actually hold
        drifted = target * (1 + r_next)
        tot = float(drifted.abs().sum())
        w_prev = drifted / (1 + g) if tot > 0 else drifted * 0.0

    idx = pd.DatetimeIndex([t for t in weeks if len(scores.loc[t].dropna()) > 0])
    ret = pd.Series(rets, index=idx, name="ret")
    return BacktestResult(
        ret=ret, gross_ret=pd.Series(gross, index=idx), turnover=pd.Series(turns, index=idx),
        costs=pd.Series(costs, index=idx), weights=pd.DataFrame(W, index=idx),
        signals=sig.loc[idx], equity=cfg.initial_capital * (1 + ret).cumprod(),
    )
