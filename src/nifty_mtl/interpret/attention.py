"""Attention-weight analysis (PRD §9): which past weeks does the model look at,
and does that change with the market regime?"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from nifty_mtl.features.build import FeatureSet
from nifty_mtl.models.dataset import WeekDataset
from nifty_mtl.models.mtl import MTLModel


@torch.no_grad()
def collect_attention(model: MTLModel, ds: WeekDataset, device) -> dict:
    """Returns pooling weights [n_samples, L], last-layer self-attention averaged over
    heads and queries [n_samples, L] (how much each key week is attended overall), plus
    the regime variable (market vol) for each sample."""
    model.eval()
    pool, self_attn, mkt_vol, weeks, syms = [], [], [], [], []
    j_vol = ds.fs.static_names.index("mkt_vol_20d")
    for i in range(len(ds)):
        b = ds[i]
        model(b["seq"].to(device), b["static"].to(device), need_weights=True)
        pool.append(model.last_pool_weights.cpu().numpy())
        a = model.attention_maps()[-1]                      # [B, H, L, L]
        self_attn.append(a.mean(dim=(1, 2)).cpu().numpy())  # attention received by each key position
        t = b["t"]
        mkt_vol.append(ds.fs.static[t, b["n_idx"].numpy(), j_vol])
        weeks.extend([ds.fs.weeks[t]] * len(b["n_idx"]))
        syms.extend(np.array(ds.fs.symbols)[b["n_idx"].numpy()])
    return {"pool": np.concatenate(pool), "self_attn": np.concatenate(self_attn),
            "mkt_vol": np.concatenate(mkt_vol), "week": pd.DatetimeIndex(weeks), "symbol": np.array(syms)}


def attention_by_lag(att: dict) -> pd.DataFrame:
    """Mean attention per lag (0 = most recent week) overall and by volatility regime."""
    L = att["pool"].shape[1]
    lag = np.arange(L)[::-1]
    hi = att["mkt_vol"] > np.median(att["mkt_vol"])
    df = pd.DataFrame({
        "lag_weeks": lag,
        "pool_all": att["pool"].mean(0),
        "pool_high_vol": att["pool"][hi].mean(0),
        "pool_low_vol": att["pool"][~hi].mean(0),
        "self_attn_all": att["self_attn"].mean(0),
    }).set_index("lag_weeks").sort_index()
    return df


def attention_concentration(att: dict) -> dict:
    """How concentrated is the pooling attention: share on the last 4 / 12 weeks and entropy."""
    p = att["pool"]
    ent = -(p * np.log(p + 1e-12)).sum(1)
    return {
        "share_last_4w": float(p[:, -4:].sum(1).mean()),
        "share_last_12w": float(p[:, -12:].sum(1).mean()),
        "share_first_12w": float(p[:, :12].sum(1).mean()),
        "entropy_mean": float(ent.mean()),
        "entropy_uniform": float(np.log(p.shape[1])),
    }
