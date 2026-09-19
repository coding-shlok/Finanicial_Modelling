"""Datasets and scalers.

Samples are grouped by *week* so that every batch is one cross-section of the
market. This is required by (a) the transaction-cost loss term, which needs the
previous week's implied weights, and (b) validation on portfolio Sharpe. A
cross-section is ~45-50 stocks, close to the PRD's batch size of 32.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from nifty_mtl.features.build import FeatureSet


@dataclass
class Scalers:
    seq_mean: np.ndarray
    seq_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray
    ret_mean: float
    ret_std: float
    vol_mean: float
    vol_std: float

    @classmethod
    def fit(cls, fs: FeatureSet, week_mask: np.ndarray, target_mode: str = "raw") -> "Scalers":
        """Fit on all usable samples inside ``week_mask`` (train weeks only)."""
        ok = fs.sample_ok & week_mask[:, None]
        y_ret_all = target_returns(fs, target_mode)
        L = fs.lookback
        # every token of every training window contributes (windows overlap; fine for moments)
        t_idx, n_idx = np.where(ok)
        rows = np.concatenate([fs.seq[t - L + 1: t + 1, n] for t, n in zip(t_idx, n_idx)])
        seq_mean, seq_std = rows.mean(0), rows.std(0) + 1e-8
        st = fs.static[ok]
        static_mean, static_std = st.mean(0), st.std(0) + 1e-8
        # never rescale one-hot sector columns
        n_graph = fs.n_graph
        static_mean[n_graph:] = 0.0
        static_std[n_graph:] = 1.0
        yr, yv = y_ret_all[ok], fs.y_vol[ok]
        return cls(seq_mean, seq_std, static_mean, static_std,
                   float(yr.mean()), float(yr.std() + 1e-8), float(yv.mean()), float(yv.std() + 1e-8))

    def to_dict(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Scalers":
        return cls(**{k: (np.array(v, dtype=np.float32) if isinstance(v, list) else v) for k, v in d.items()})


def target_returns(fs: FeatureSet, mode: str) -> np.ndarray:
    """Training target for the return head. ``cs_demean`` subtracts each week's
    cross-sectional mean over usable stocks (relative return, which is what the
    decile ranking actually needs). Backtests always use raw realised returns."""
    if mode == "raw":
        return fs.y_ret
    if mode == "cs_demean":
        ok = fs.sample_ok
        cnt = np.maximum(ok.sum(1, keepdims=True), 1)
        mean = (fs.y_ret * ok).sum(1, keepdims=True) / cnt
        return np.where(ok, fs.y_ret - mean, 0.0).astype(np.float32)
    raise ValueError(mode)


class WeekDataset(Dataset):
    """One item = the cross-section of usable stocks at week t (plus the previous week's
    cross-section for the same stocks, used by the turnover term)."""

    def __init__(self, fs: FeatureSet, scalers: Scalers, weeks: np.ndarray,
                 min_stocks: int = 10, static_mask: np.ndarray | None = None, target_mode: str = "raw"):
        self.fs, self.sc = fs, scalers
        self.y_ret = target_returns(fs, target_mode)
        self.L = fs.lookback
        self.static_mask = static_mask  # optional 0/1 vector to ablate feature groups
        self.week_ids = [t for t in weeks if fs.sample_ok[t].sum() >= min_stocks and t >= self.L]

    def __len__(self) -> int:
        return len(self.week_ids)

    def _window(self, t: int, n_idx: np.ndarray) -> np.ndarray:
        L = self.L
        w = self.fs.seq[t - L + 1: t + 1][:, n_idx]                 # [L, n, F]
        w = (w - self.sc.seq_mean) / self.sc.seq_std
        return np.transpose(w, (1, 0, 2))                            # [n, L, F]

    def _static(self, t: int, n_idx: np.ndarray) -> np.ndarray:
        s = (self.fs.static[t, n_idx] - self.sc.static_mean) / self.sc.static_std
        if self.static_mask is not None:
            s = s * self.static_mask
        return s

    def cross_section(self, t: int) -> dict:
        n_idx = np.where(self.fs.sample_ok[t])[0]
        out = {
            "t": t,
            "n_idx": torch.as_tensor(n_idx),
            "seq": torch.as_tensor(self._window(t, n_idx), dtype=torch.float32),
            "static": torch.as_tensor(self._static(t, n_idx), dtype=torch.float32),
            "y_ret": torch.as_tensor(self.y_ret[t, n_idx], dtype=torch.float32),
            "y_vol": torch.as_tensor(self.fs.y_vol[t, n_idx], dtype=torch.float32),
        }
        # previous week's features for the same stocks (if all were usable then)
        tp = t - 1
        if tp >= self.L and self.fs.sample_ok[tp, n_idx].all():
            out["prev_seq"] = torch.as_tensor(self._window(tp, n_idx), dtype=torch.float32)
            out["prev_static"] = torch.as_tensor(self._static(tp, n_idx), dtype=torch.float32)
        return out

    def __getitem__(self, i: int) -> dict:
        return self.cross_section(self.week_ids[i])


def collate_identity(batch):
    assert len(batch) == 1
    return batch[0]


def week_masks(fs: FeatureSet, splits) -> dict[str, np.ndarray]:
    w = fs.weeks
    def m(a, b):
        return np.asarray((w >= pd.Timestamp(a)) & (w <= pd.Timestamp(b)))
    last = str(w[-1].date())
    return {
        "train": m(splits.train_start, splits.train_end),
        "val": m(splits.val_start, splits.val_end),
        "test": m(splits.test_start, splits.test_end),
        "holdout": m(splits.holdout_start, last),
    }
