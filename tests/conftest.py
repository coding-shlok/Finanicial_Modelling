"""Shared fixtures: a small synthetic daily panel so tests never hit the network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_mtl.config import Universe
from nifty_mtl.data.preprocess import build_weekly
from nifty_mtl.features.build import build_features

SECTORS = ["Banks", "IT", "Energy", "FMCG"]
SYMS = [f"S{i:02d}" for i in range(12)]


@pytest.fixture(scope="session")
def universe() -> Universe:
    return Universe(
        symbols=SYMS,
        tickers={s: s + ".NS" for s in SYMS},
        sector_of={s: SECTORS[i % len(SECTORS)] for i, s in enumerate(SYMS)},
        sectors=SECTORS,
        names={s: s for s in SYMS},
        benchmark="^NSEI",
        risk_free_annual=0.065,
        spinoffs={},
    )


def make_daily(seed: int = 0, n_days: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    frames = []
    market = rng.normal(0.0003, 0.01, n_days)
    for i, s in enumerate(SYMS + ["BENCH"]):
        beta = 1.0 if s == "BENCH" else rng.uniform(0.5, 1.5)
        idio = 0.0 if s == "BENCH" else rng.normal(0, 0.015, n_days)
        r = beta * market + idio
        close = 100 * np.exp(np.cumsum(r))
        high = close * (1 + np.abs(rng.normal(0, 0.005, n_days)))
        low = close * (1 - np.abs(rng.normal(0, 0.005, n_days)))
        open_ = np.roll(close, 1); open_[0] = close[0]
        vol = rng.integers(500_000, 2_000_000, n_days).astype(float)
        frames.append(pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                                    "Volume": vol, "symbol": s}, index=dates))
    d = pd.concat(frames)
    d.index.name = "date"
    return d.reset_index().set_index(["date", "symbol"]).sort_index()


@pytest.fixture(scope="session")
def daily() -> pd.DataFrame:
    return make_daily()


@pytest.fixture(scope="session")
def panel(daily, universe):
    return build_weekly(daily, universe)


@pytest.fixture(scope="session")
def features(panel, universe):
    return build_features(panel, universe)
