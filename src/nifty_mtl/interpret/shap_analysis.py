"""SHAP feature attribution (PRD §9) with ``shap.GradientExplainer`` (expected
gradients), which works with the attention backbone where DeepExplainer's
op-by-op rules do not.

Attributions are computed for the *decision score* (predicted return /
predicted volatility) so that "why is this stock recommended?" is answered on
the quantity actually ranked. Sequence attributions [n, L, F_seq] are summed
over time to give per-feature importance, and over features to give per-lag
importance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn

from nifty_mtl.features.build import FeatureSet
from nifty_mtl.models.dataset import WeekDataset
from nifty_mtl.models.mtl import MTLModel, denormalise

GROUPS = {
    "candle": ["log_open", "log_high", "log_low", "log_close", "log_volume_ratio", "rv"],
    "technical": ["rsi14", "macd", "macd_signal", "macd_hist", "bb_pctb", "bb_width", "atr14", "obv12", "cci20",
                  "stoch_k", "stoch_d", "williams_r", "adx14", "plus_di", "minus_di"],
    "statistical": ["ret_1w", "ret_4w", "ret_12w", "ret_26w", "vol_4w", "vol_12w", "skew_12w", "kurt_12w",
                    "acf1_12w", "volume_ratio_12w"],
    "graph_pca": [f"pca_{i}" for i in range(5)],
    "graph_peer": ["peer_ret_1w", "peer_ret_4w", "peer_ret_12w", "rel_strength_4w", "sector_mom_10w"],
    "graph_corr": ["corr_banks", "corr_it", "corr_energy", "corr_mkt", "corr_mkt_chg", "corr_mkt_vol",
                   "corr_intra_sector", "corr_extra_sector", "beta_mkt", "beta_sector"],
    "graph_centrality": ["eig_centrality", "deg_centrality", "eig_centrality_chg", "clustering", "sector_centrality"],
    "graph_regime": ["mkt_vol_20d", "mkt_ret_12w", "mkt_ma26_dev", "mkt_avg_corr", "mkt_dispersion"],
}


class ScoreWrapper(nn.Module):
    """(seq, static) -> decision score, in raw units."""
    def __init__(self, model: MTLModel, scalers, target: str = "score"):
        super().__init__()
        self.m, self.sc, self.target = model, scalers, target

    def forward(self, seq, static):
        r, v = self.m(seq, static)
        r, v = denormalise(r, v, self.sc, self.m.cfg.vol_clip)
        return {"score": r / v, "ret": r, "vol": v}[self.target].unsqueeze(-1)


@dataclass
class ShapResult:
    seq_values: np.ndarray       # [n, L, F_seq]
    static_values: np.ndarray    # [n, F_static]
    seq_input: np.ndarray
    static_input: np.ndarray
    weeks: pd.DatetimeIndex
    symbols: np.ndarray
    seq_names: list[str]
    static_names: list[str]

    def feature_importance(self, mode: str = "signed") -> pd.Series:
        """Global mean |SHAP| per feature. Sequence features are aggregated over the 60 lags:
        ``signed`` = |sum over lags| (net effect), ``mass`` = sum over lags of |SHAP| (total mass)."""
        seq_imp = np.abs(self.seq_values.sum(1)).mean(0) if mode == "signed" else np.abs(self.seq_values).sum(1).mean(0)
        st_imp = np.abs(self.static_values).mean(0)
        return pd.Series(np.concatenate([seq_imp, st_imp]), index=self.seq_names + self.static_names).sort_values(ascending=False)

    def group_importance(self, mode: str = "signed") -> pd.Series:
        fi = self.feature_importance(mode)
        out = {g: float(fi.reindex(names).fillna(0).sum()) for g, names in GROUPS.items()}
        out["sector_onehot"] = float(fi[[n for n in fi.index if n.startswith("sector_")]].sum())
        return pd.Series(out).sort_values(ascending=False)

    def lag_importance(self) -> pd.Series:
        L = self.seq_values.shape[1]
        return pd.Series(np.abs(self.seq_values).sum(2).mean(0), index=np.arange(L)[::-1], name="mean_abs_shap").sort_index()

    def top5_share(self) -> float:
        fi = self.feature_importance()
        return float(fi.iloc[:5].sum() / fi.sum())

    def explain_sample(self, i: int, k: int = 5) -> pd.Series:
        """Signed top-k contributions for one sample (seq features summed over lags)."""
        seq_c = self.seq_values[i].sum(0)
        st_c = self.static_values[i]
        s = pd.Series(np.concatenate([seq_c, st_c]), index=self.seq_names + self.static_names)
        return s.reindex(s.abs().sort_values(ascending=False).index[:k])


def _stack_batches(ds: WeekDataset, week_ids: list[int], max_samples: int | None, seed: int = 0):
    seqs, sts, weeks, syms = [], [], [], []
    for t in week_ids:
        b = ds.cross_section(t)
        seqs.append(b["seq"]); sts.append(b["static"])
        weeks.extend([ds.fs.weeks[t]] * len(b["n_idx"])); syms.extend(np.array(ds.fs.symbols)[b["n_idx"].numpy()])
    seq, st = torch.cat(seqs), torch.cat(sts)
    weeks, syms = pd.DatetimeIndex(weeks), np.array(syms)
    if max_samples and len(seq) > max_samples:
        idx = np.random.default_rng(seed).choice(len(seq), max_samples, replace=False)
        idx.sort()
        seq, st, weeks, syms = seq[idx], st[idx], weeks[idx], syms[idx]
    return seq, st, weeks, syms


def compute_shap(model: MTLModel, scalers, ds_bg: WeekDataset, ds_explain: WeekDataset,
                 n_background: int = 300, n_explain: int = 600, target: str = "score", seed: int = 0) -> ShapResult:
    torch.manual_seed(seed)
    wrapper = ScoreWrapper(model, scalers, target).eval()
    bg_seq, bg_st, _, _ = _stack_batches(ds_bg, ds_bg.week_ids, n_background, seed)
    ex_seq, ex_st, weeks, syms = _stack_batches(ds_explain, ds_explain.week_ids, n_explain, seed)
    explainer = shap.GradientExplainer(wrapper, [bg_seq, bg_st])
    vals = explainer.shap_values([ex_seq, ex_st], nsamples=100)
    seq_v, st_v = vals[0], vals[1]
    if seq_v.ndim == 4:               # newer shap returns [..., n_outputs]
        seq_v, st_v = seq_v[..., 0], st_v[..., 0]
    return ShapResult(seq_v, st_v, ex_seq.numpy(), ex_st.numpy(), weeks, syms,
                      list(ds_explain.fs.seq_names), list(ds_explain.fs.static_names))
