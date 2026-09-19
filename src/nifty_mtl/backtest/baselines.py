"""Baseline strategies (PRD §8): each returns a score DataFrame [week x symbol]
that is fed through the same backtest engine as the MTL model.

1. Buy & hold Nifty 50      -> benchmark series, not a score
2. Simple momentum          -> score = trailing 4-week return
3. Inverse-volatility       -> score = -trailing 4-week vol (low-vol tilt)
4. Equal-weight random 10   -> random scores (seeded), averaged over seeds in eval
5. Logistic regression      -> P(next-week return > cross-sectional median) on the
                               flattened last-week features + graph features, fit on train weeks only
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nifty_mtl.features.build import FeatureSet


def _panel(fs: FeatureSet, name: str) -> pd.DataFrame:
    j = fs.seq_names.index(name)
    return pd.DataFrame(np.where(fs.sample_ok, fs.seq[..., j], np.nan), index=fs.weeks, columns=fs.symbols)


def momentum_scores(fs: FeatureSet, window: str = "ret_4w") -> pd.DataFrame:
    return _panel(fs, window)


def inverse_vol_scores(fs: FeatureSet) -> pd.DataFrame:
    return -_panel(fs, "vol_4w")


def random_scores(fs: FeatureSet, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(size=fs.y_ret.shape)
    return pd.DataFrame(np.where(fs.sample_ok, r, np.nan), index=fs.weeks, columns=fs.symbols)


def _flat_features(fs: FeatureSet, t: int, n_idx: np.ndarray) -> np.ndarray:
    # last week's own features + static (graph + sector); same information the MTL model sees at t
    return np.concatenate([fs.seq[t, n_idx], fs.static[t, n_idx]], axis=-1)


def logistic_scores(fs: FeatureSet, train_weeks: np.ndarray, pred_weeks: np.ndarray, C: float = 0.1) -> pd.DataFrame:
    Xs, ys = [], []
    for t in np.where(train_weeks)[0]:
        n_idx = np.where(fs.sample_ok[t])[0]
        if len(n_idx) < 10:
            continue
        y = fs.y_ret[t, n_idx]
        Xs.append(_flat_features(fs, t, n_idx)); ys.append((y > np.median(y)).astype(int))
    X, y = np.concatenate(Xs), np.concatenate(ys)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    clf.fit(X, y)
    out = np.full(fs.y_ret.shape, np.nan, dtype=np.float32)
    for t in np.where(pred_weeks)[0]:
        n_idx = np.where(fs.sample_ok[t])[0]
        if len(n_idx) == 0:
            continue
        out[t, n_idx] = clf.predict_proba(_flat_features(fs, t, n_idx))[:, 1]
    return pd.DataFrame(out, index=fs.weeks, columns=fs.symbols)


def benchmark_weekly_returns(fs: FeatureSet, bench_ret_log: pd.Series) -> pd.Series:
    """Nifty 50 simple return over week t+1 indexed by decision week t."""
    r = np.exp(bench_ret_log.shift(-1)) - 1
    return r.reindex(fs.weeks)
