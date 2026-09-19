"""Assemble the feature tensor.

Output (``FeatureSet``):
  seq    : float32 [T, N, F_seq]   per-week stock-own features (fed as a 60-token sequence)
  static : float32 [T, N, F_static] graph features + sector one-hot at decision week t
  y_ret  : float32 [T, N]          next-week log return
  y_vol  : float32 [T, N]          next-week realized vol (clipped)
  valid  : bool    [T, N]          bar is clean at week t
  sample_ok : bool [T, N]          full lookback valid AND targets available

The 60-week window ending at t is built lazily by the dataset: seq[t-59:t+1].
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd

from nifty_mtl.config import DATA_PROCESSED, DataConfig, Universe, load_universe
from nifty_mtl.data.preprocess import WeeklyPanel
from nifty_mtl.features.graph import GRAPH_NAMES, graph_features
from nifty_mtl.features.statistical import STATISTICAL_NAMES, statistical_features
from nifty_mtl.features.technical import TECHNICAL_NAMES, technical_features

log = logging.getLogger(__name__)

CANDLE_NAMES = ["log_open", "log_high", "log_low", "log_close", "log_volume_ratio", "rv"]
SEQ_NAMES = CANDLE_NAMES + TECHNICAL_NAMES + STATISTICAL_NAMES


@dataclass
class FeatureSet:
    weeks: pd.DatetimeIndex
    symbols: list[str]
    seq_names: list[str]
    static_names: list[str]
    seq: np.ndarray
    static: np.ndarray
    y_ret: np.ndarray
    y_vol: np.ndarray
    valid: np.ndarray
    sample_ok: np.ndarray
    lookback: int

    @property
    def n_graph(self) -> int:
        return len(GRAPH_NAMES)

    def week_index(self, date: str) -> int:
        return int(self.weeks.searchsorted(pd.Timestamp(date)))

    def save(self, path=None) -> None:
        path = path or DATA_PROCESSED / "features.h5"
        with h5py.File(path, "w") as f:
            for k in ["seq", "static", "y_ret", "y_vol", "valid", "sample_ok"]:
                f.create_dataset(k, data=getattr(self, k), compression="gzip", compression_opts=4)
            f.attrs["weeks"] = [str(w.date()) for w in self.weeks]
            f.attrs["symbols"] = self.symbols
            f.attrs["seq_names"] = self.seq_names
            f.attrs["static_names"] = self.static_names
            f.attrs["lookback"] = self.lookback
        log.info("saved features to %s: seq %s static %s", path, self.seq.shape, self.static.shape)

    @classmethod
    def load(cls, path=None) -> "FeatureSet":
        path = path or DATA_PROCESSED / "features.h5"
        with h5py.File(path, "r") as f:
            kw = {k: f[k][()] for k in ["seq", "static", "y_ret", "y_vol", "valid", "sample_ok"]}
            weeks = pd.DatetimeIndex([str(w) for w in f.attrs["weeks"]])
            return cls(weeks=weeks, symbols=list(f.attrs["symbols"]), seq_names=list(f.attrs["seq_names"]),
                       static_names=list(f.attrs["static_names"]), lookback=int(f.attrs["lookback"]), **kw)


def _stack(feats: dict[str, pd.DataFrame], names: list[str]) -> np.ndarray:
    return np.stack([feats[n].to_numpy(dtype=np.float32) for n in names], axis=-1)


def build_features(panel: WeeklyPanel, universe: Universe | None = None,
                   cfg: DataConfig | None = None) -> FeatureSet:
    universe = universe or load_universe()
    cfg = cfg or DataConfig()
    syms = panel.symbols
    prev_close = panel.close.shift(1)

    candle = {
        "log_open": np.log(panel.open / prev_close),
        "log_high": np.log(panel.high / prev_close),
        "log_low": np.log(panel.low / prev_close),
        "log_close": np.log(panel.close / prev_close),
        "log_volume_ratio": np.log(panel.volume / panel.volume.rolling(12).mean().replace(0, np.nan)),
        "rv": panel.rv,
    }
    tech = technical_features(panel.open, panel.high, panel.low, panel.close, panel.volume)
    stat = statistical_features(panel.ret, panel.volume)
    graph = graph_features(panel.ret, panel.daily_ret, panel.bench_daily_ret, panel.bench_close, universe)

    seq = _stack({**candle, **tech, **stat}, SEQ_NAMES)                     # [T, N, 31]
    sector_onehot = np.zeros((len(panel.weeks), len(syms), len(universe.sectors)), dtype=np.float32)
    for j, s in enumerate(syms):
        sector_onehot[:, j, universe.sector_index[universe.sector_of[s]]] = 1.0
    static = np.concatenate([_stack(graph, GRAPH_NAMES), sector_onehot], axis=-1)  # [T, N, 41]
    static_names = GRAPH_NAMES + [f"sector_{s}" for s in universe.sectors]

    y_ret_df, y_vol_df = panel.targets(cfg.vol_clip)
    y_ret, y_vol = y_ret_df.to_numpy(np.float32), y_vol_df.to_numpy(np.float32)
    valid = panel.valid.to_numpy(bool)

    # a sample at (t, n) is usable if the last `lookback` weeks are valid with finite features,
    # and both targets exist. Features that are NaN inside a valid window (indicator warm-up)
    # invalidate the window — no imputation of true unknowns.
    L = cfg.lookback_weeks
    feat_ok = valid & np.isfinite(seq).all(-1) & np.isfinite(static).all(-1)
    win_ok = pd.DataFrame(feat_ok).rolling(L).sum().to_numpy() == L
    sample_ok = win_ok & np.isfinite(y_ret) & np.isfinite(y_vol)

    # zero-fill NaNs *only* where the sample will never be used (keeps tensors finite)
    seq = np.where(np.isfinite(seq), seq, 0.0).astype(np.float32)
    static = np.where(np.isfinite(static), static, 0.0).astype(np.float32)
    y_ret = np.where(np.isfinite(y_ret), y_ret, 0.0).astype(np.float32)
    y_vol = np.where(np.isfinite(y_vol), y_vol, 0.0).astype(np.float32)

    fs = FeatureSet(weeks=panel.weeks, symbols=syms, seq_names=SEQ_NAMES, static_names=static_names,
                    seq=seq, static=static, y_ret=y_ret, y_vol=y_vol, valid=valid,
                    sample_ok=sample_ok, lookback=L)
    log.info("features: %d weeks, %d symbols, seq=%d static=%d, usable samples=%d (first usable week %s)",
             len(fs.weeks), len(syms), seq.shape[-1], static.shape[-1], int(sample_ok.sum()),
             fs.weeks[np.argmax(sample_ok.any(1))].date())
    return fs
