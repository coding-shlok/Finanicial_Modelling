"""Graph-based features (30 features, PRD §7).

The "graph" is the stock-stock correlation network re-estimated every week
from the trailing 63 trading days (~13 weeks) of daily log returns. All
quantities at week t use returns with date <= t (Friday close), so there is
no look-ahead.

Groups
------
1. Sector/market PCA embedding (5): loadings of each stock on the top-5
   eigenvectors of the correlation matrix (sign-aligned across weeks).
2. Peer momentum (5): sector-mean returns excluding self (1/4/12w), relative
   strength vs sector (4w), sector 10w momentum.
3. Correlation trends (10): corr with Banks/IT/Energy sector indices and the
   market, corr change (63d vs 126d), corr volatility, intra-sector vs
   extra-sector average corr, market beta, sector beta.
4. Network centrality (5): eigenvector centrality, degree centrality
   (|corr|>0.3), 13-week change in eigenvector centrality, clustering
   coefficient, sector-average centrality.
5. Market regime (5): market realized vol (20d), market 12w return, market
   price vs 26w MA, average pairwise correlation, cross-sectional dispersion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nifty_mtl.config import Universe

GRAPH_NAMES = (
    [f"pca_{i}" for i in range(5)]
    + ["peer_ret_1w", "peer_ret_4w", "peer_ret_12w", "rel_strength_4w", "sector_mom_10w"]
    + ["corr_banks", "corr_it", "corr_energy", "corr_mkt", "corr_mkt_chg", "corr_mkt_vol",
       "corr_intra_sector", "corr_extra_sector", "beta_mkt", "beta_sector"]
    + ["eig_centrality", "deg_centrality", "eig_centrality_chg", "clustering", "sector_centrality"]
    + ["mkt_vol_20d", "mkt_ret_12w", "mkt_ma26_dev", "mkt_avg_corr", "mkt_dispersion"]
)
assert len(GRAPH_NAMES) == 30

CORR_WINDOW = 63
LONG_WINDOW = 126
SHORT_WINDOW = 21
MIN_OBS = 40
DEG_THRESHOLD = 0.3
MAJOR_SECTORS = ["Banks", "IT", "Energy"]


def _safe_corr(x: np.ndarray) -> np.ndarray:
    """Correlation matrix with NaN for columns lacking enough observations."""
    n = x.shape[1]
    out = np.full((n, n), np.nan)
    ok = np.isfinite(x).sum(0) >= MIN_OBS
    if ok.sum() < 3:
        return out
    sub = x[:, ok]
    sub = sub[np.isfinite(sub).all(1)]
    if len(sub) < MIN_OBS:
        return out
    c = np.corrcoef(sub, rowvar=False)
    idx = np.where(ok)[0]
    out[np.ix_(idx, idx)] = c
    return out


def _pca_loadings(c: np.ndarray, k: int = 5) -> np.ndarray:
    n = c.shape[0]
    out = np.full((n, k), np.nan)
    ok = np.isfinite(np.diag(c))
    if ok.sum() < k + 1:
        return out
    sub = c[np.ix_(ok, ok)]
    w, v = np.linalg.eigh(sub)
    order = np.argsort(w)[::-1][:k]
    w, v = w[order], v[:, order]
    # sign convention: make each eigenvector's sum positive for stability over time
    v = v * np.sign(v.sum(0, keepdims=True) + 1e-12)
    out[ok] = v * np.sqrt(np.clip(w, 0, None))
    return out


def _eigvec_centrality(a: np.ndarray) -> np.ndarray:
    """Perron vector of a non-negative weighted adjacency, normalised to max 1."""
    w, v = np.linalg.eigh(a)
    vec = np.abs(v[:, np.argmax(w)])
    return vec / (vec.max() + 1e-12)


def _clustering(adj: np.ndarray) -> np.ndarray:
    """Local clustering coefficient of an unweighted graph (adjacency 0/1)."""
    deg = adj.sum(1)
    tri = np.diag(adj @ adj @ adj) / 2
    denom = deg * (deg - 1) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        cc = np.where(denom > 0, tri / denom, 0.0)
    return cc


def graph_features(ret_w: pd.DataFrame, daily_ret: pd.DataFrame, bench_daily_ret: pd.Series,
                   bench_close_w: pd.Series, universe: Universe) -> dict[str, pd.DataFrame]:
    weeks = ret_w.index
    syms = list(ret_w.columns)
    n = len(syms)
    sectors = np.array([universe.sector_of[s] for s in syms])
    sec_names = universe.sectors
    sec_mask = {sec: sectors == sec for sec in sec_names}

    daily_ret = daily_ret[syms]
    dr_np = daily_ret.to_numpy()
    d_index = daily_ret.index
    br = bench_daily_ret.reindex(d_index).to_numpy()

    # sector index daily returns (equal-weight mean of members)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sec_daily = {sec: np.nanmean(dr_np[:, sec_mask[sec]], axis=1) if sec_mask[sec].any()
                         else np.full(len(d_index), np.nan) for sec in sec_names}

    # containers (weeks x n)
    F = {k: np.full((len(weeks), n), np.nan) for k in GRAPH_NAMES}
    corr_mkt_21 = np.full((len(weeks), n), np.nan)

    for wi, t in enumerate(weeks):
        end = d_index.searchsorted(t, side="right")
        if end < MIN_OBS:
            continue
        x = dr_np[max(0, end - CORR_WINDOW):end]
        x_long = dr_np[max(0, end - LONG_WINDOW):end]
        x_short = dr_np[max(0, end - SHORT_WINDOW):end]
        b = br[max(0, end - CORR_WINDOW):end]
        b_long = br[max(0, end - LONG_WINDOW):end]
        b_short = br[max(0, end - SHORT_WINDOW):end]

        C = _safe_corr(x)
        ok = np.isfinite(np.diag(C))

        # 1. PCA embedding
        L = _pca_loadings(C, 5)
        for i in range(5):
            F[f"pca_{i}"][wi] = L[:, i]

        # 3. correlation trends & betas (per stock vs series)
        def _corr_vec(xs, s):
            out = np.full(n, np.nan)
            for j in range(n):
                xj = xs[:, j]
                m = np.isfinite(xj) & np.isfinite(s)
                if m.sum() >= min(MIN_OBS, len(xs) - 2) and m.sum() > 5:
                    out[j] = np.corrcoef(xj[m], s[m])[0, 1]
            return out

        def _beta_vec(xs, s):
            out = np.full(n, np.nan)
            var = np.nanvar(s)
            for j in range(n):
                xj = xs[:, j]
                m = np.isfinite(xj) & np.isfinite(s)
                if m.sum() > 5 and var > 0:
                    out[j] = np.cov(xj[m], s[m])[0, 1] / np.var(s[m])
            return out

        for sec, key in zip(MAJOR_SECTORS, ["corr_banks", "corr_it", "corr_energy"]):
            sd = sec_daily[sec][max(0, end - CORR_WINDOW):end]
            F[key][wi] = _corr_vec(x, sd)
        cm = _corr_vec(x, b)
        F["corr_mkt"][wi] = cm
        F["corr_mkt_chg"][wi] = cm - _corr_vec(x_long, b_long)
        corr_mkt_21[wi] = _corr_vec(x_short, b_short)
        F["beta_mkt"][wi] = _beta_vec(x, b)
        own_sec = np.full(n, np.nan)
        for sec in sec_names:
            sd = sec_daily[sec][max(0, end - CORR_WINDOW):end]
            bs = _beta_vec(x, sd)
            own_sec[sec_mask[sec]] = bs[sec_mask[sec]]
        F["beta_sector"][wi] = own_sec

        # intra/extra sector average correlation
        Cz = np.where(np.isfinite(C), C, np.nan)
        np.fill_diagonal(Cz, np.nan)
        for j in range(n):
            if not ok[j]:
                continue
            same = sec_mask[sectors[j]].copy(); same[j] = False
            other = ~sec_mask[sectors[j]]
            F["corr_intra_sector"][wi, j] = np.nanmean(Cz[j, same]) if same.any() else 0.0
            F["corr_extra_sector"][wi, j] = np.nanmean(Cz[j, other])

        # 4. network centrality
        if ok.sum() >= 3:
            sub = C[np.ix_(ok, ok)].copy()
            np.fill_diagonal(sub, 0.0)
            A = np.clip(sub, 0, None)
            ec = _eigvec_centrality(A)
            adj = (np.abs(sub) > DEG_THRESHOLD).astype(float)
            deg = adj.sum(1) / max(ok.sum() - 1, 1)
            cc = _clustering(adj)
            F["eig_centrality"][wi, ok] = ec
            F["deg_centrality"][wi, ok] = deg
            F["clustering"][wi, ok] = cc
            sec_c = np.full(n, np.nan)
            ecf = np.full(n, np.nan); ecf[ok] = ec
            for sec in sec_names:
                m = sec_mask[sec] & ok
                if m.any():
                    sec_c[sec_mask[sec]] = np.nanmean(ecf[m])
            F["sector_centrality"][wi] = sec_c
            # 5. market avg pairwise corr
            iu = np.triu_indices(ok.sum(), 1)
            F["mkt_avg_corr"][wi] = np.nanmean(sub[iu])

    # post-loop time-series transforms
    W = len(weeks)
    ec_df = pd.DataFrame(F["eig_centrality"], index=weeks)
    F["eig_centrality_chg"] = (ec_df - ec_df.shift(13)).to_numpy()
    F["corr_mkt_vol"] = pd.DataFrame(corr_mkt_21, index=weeks).rolling(12, min_periods=6).std().to_numpy()

    # 2. peer momentum on weekly returns
    r1, r4, r12 = ret_w, ret_w.rolling(4).sum(), ret_w.rolling(12).sum()
    r10 = ret_w.rolling(10).sum()
    for key, r in [("peer_ret_1w", r1), ("peer_ret_4w", r4), ("peer_ret_12w", r12)]:
        out = pd.DataFrame(np.nan, index=weeks, columns=syms)
        for sec in sec_names:
            cols = [s for s, m in zip(syms, sec_mask[sec]) if m]
            if len(cols) < 2:
                out[cols] = 0.0
                continue
            block = r[cols]
            tot = block.sum(axis=1, min_count=1)
            cnt = block.notna().sum(axis=1)
            peer = (tot.to_numpy()[:, None] - block.fillna(0).to_numpy()) / np.clip((cnt.to_numpy()[:, None] - block.notna().astype(int).to_numpy()), 1, None)
            out[cols] = np.where(block.notna(), peer, np.nan)
        F[key] = out.to_numpy()
    F["rel_strength_4w"] = (r4.to_numpy() - F["peer_ret_4w"])
    sec_mom = pd.DataFrame(np.nan, index=weeks, columns=syms)
    for sec in sec_names:
        cols = [s for s, m in zip(syms, sec_mask[sec]) if m]
        sec_mom.loc[:, cols] = np.repeat(r10[cols].mean(axis=1).to_numpy()[:, None], len(cols), axis=1)
    F["sector_mom_10w"] = sec_mom.to_numpy()

    # 5. market regime (broadcast to all stocks)
    b_daily = bench_daily_ret.dropna()
    mkt_vol = (b_daily.rolling(20).std() * np.sqrt(252)).reindex(weeks, method="ffill")
    bench_ret_w = np.log(bench_close_w / bench_close_w.shift(1))
    mkt_ret_12 = bench_ret_w.rolling(12).sum()
    mkt_ma_dev = bench_close_w / bench_close_w.rolling(26).mean() - 1
    disp = ret_w.std(axis=1).rolling(4).mean()
    for key, s in [("mkt_vol_20d", mkt_vol), ("mkt_ret_12w", mkt_ret_12),
                   ("mkt_ma26_dev", mkt_ma_dev), ("mkt_dispersion", disp)]:
        F[key] = np.repeat(s.to_numpy()[:, None], n, axis=1)
    F["mkt_avg_corr"] = np.repeat(np.nanmean(F["mkt_avg_corr"], axis=1, keepdims=True), n, axis=1)

    return {k: pd.DataFrame(F[k], index=weeks, columns=syms) for k in GRAPH_NAMES}
