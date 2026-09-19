"""Performance metrics on a weekly return series (PRD §8)."""
from __future__ import annotations

import numpy as np
import pandas as pd

WEEKS_PER_YEAR = 52


def sharpe(ret: pd.Series, rf_weekly: float = 0.0, annualise: bool = True) -> float:
    ex = ret - rf_weekly
    sd = ex.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    s = ex.mean() / sd
    return float(s * np.sqrt(WEEKS_PER_YEAR)) if annualise else float(s)


def sortino(ret: pd.Series, rf_weekly: float = 0.0) -> float:
    ex = ret - rf_weekly
    dn = ex[ex < 0].std(ddof=1)
    if dn == 0 or np.isnan(dn):
        return float("nan")
    return float(ex.mean() / dn * np.sqrt(WEEKS_PER_YEAR))


def max_drawdown(ret: pd.Series) -> float:
    eq = (1 + ret).cumprod()
    dd = eq / eq.cummax() - 1
    return float(dd.min())


def cagr(ret: pd.Series) -> float:
    n = len(ret)
    if n == 0:
        return float("nan")
    return float((1 + ret).prod() ** (WEEKS_PER_YEAR / n) - 1)


def win_rate(ret: pd.Series) -> float:
    return float((ret > 0).mean())


def profit_factor(ret: pd.Series) -> float:
    wins, losses = ret[ret > 0].sum(), -ret[ret < 0].sum()
    return float(wins / losses) if losses > 0 else float("inf")


def information_ratio(ret: pd.Series, bench: pd.Series) -> float:
    ex = (ret - bench.reindex(ret.index)).dropna()
    te = ex.std(ddof=1)
    if te == 0 or np.isnan(te):
        return float("nan")
    return float(ex.mean() / te * np.sqrt(WEEKS_PER_YEAR))


def calmar(ret: pd.Series) -> float:
    mdd = max_drawdown(ret)
    return float(cagr(ret) / abs(mdd)) if mdd < 0 else float("nan")


def summary(ret: pd.Series, bench: pd.Series | None = None, rf_weekly: float = 0.0,
            turnover: pd.Series | None = None) -> dict:
    out = {
        "n_weeks": int(len(ret)),
        "cagr": cagr(ret),
        "ann_vol": float(ret.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR)),
        "sharpe": sharpe(ret, rf_weekly),
        "sharpe_weekly": sharpe(ret, rf_weekly, annualise=False),
        "sortino": sortino(ret, rf_weekly),
        "max_drawdown": max_drawdown(ret),
        "calmar": calmar(ret),
        "win_rate": win_rate(ret),
        "profit_factor": profit_factor(ret),
        "total_return": float((1 + ret).prod() - 1),
    }
    if bench is not None:
        b = bench.reindex(ret.index)
        out["information_ratio"] = information_ratio(ret, b)
        out["bench_cagr"] = cagr(b.dropna())
        out["bench_sharpe"] = sharpe(b.dropna(), rf_weekly)
        out["excess_cagr"] = out["cagr"] - out["bench_cagr"]
    if turnover is not None:
        out["avg_weekly_turnover"] = float(turnover.mean())
    return out


def bootstrap_sharpe_ci(ret: pd.Series, rf_weekly: float = 0.0, n_boot: int = 2000,
                        block: int = 4, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Stationary block bootstrap CI for the annualised Sharpe ratio."""
    rng = np.random.default_rng(seed)
    x = ret.to_numpy()
    n = len(x)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = []
        while len(idx) < n:
            start = rng.integers(0, n)
            length = rng.geometric(1 / block)
            idx.extend(((start + np.arange(length)) % n).tolist())
        s = pd.Series(x[np.array(idx[:n])])
        stats[b] = sharpe(s, rf_weekly)
    return float(np.nanpercentile(stats, 100 * alpha / 2)), float(np.nanpercentile(stats, 100 * (1 - alpha / 2)))
