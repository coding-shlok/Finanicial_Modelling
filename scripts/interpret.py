"""Interpretability + failure analysis (PRD Week 4, §9).

Uses the static `final` checkpoint for attention/SHAP (one model, test-period
samples) and the walk-forward `mtl` predictions for regime / characteristic
slices (the actual out-of-sample signal).
"""
from __future__ import annotations

import json
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nifty_mtl import plotting as P
from nifty_mtl.backtest.baselines import benchmark_weekly_returns
from nifty_mtl.backtest.engine import run_backtest
from nifty_mtl.config import FIGURES, RESULTS, TABLES, BacktestConfig, Config, load_universe
from nifty_mtl.data.preprocess import WeeklyPanel
from nifty_mtl.features.build import FeatureSet
from nifty_mtl.interpret.attention import attention_by_lag, attention_concentration, collect_attention
from nifty_mtl.interpret.failure import grouped_summary, regime_table, stock_characteristic_table
from nifty_mtl.interpret.shap_analysis import GROUPS, compute_shap
from nifty_mtl.models.dataset import WeekDataset, week_masks
from nifty_mtl.models.train import Predictions, load_checkpoint, pick_device, realised_simple_returns

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("numexpr").setLevel(logging.WARNING)
log = logging.getLogger("interpret")


def weekly_ic(score, rr):
    out = {}
    for t in score.index:
        a, b = score.loc[t], rr.loc[t]; m = a.notna() & b.notna()
        if m.sum() > 5:
            out[t] = a[m].rank().corr(b[m].rank())
    return pd.Series(out)


def main():
    P.setup()
    fs = FeatureSet.load(); u = load_universe(); panel = WeeklyPanel.load(); cfg = Config()
    raw = json.loads((RESULTS / "final_config.json").read_text()); cfg.backtest = BacktestConfig(**raw["backtest"])
    masks = week_masks(fs, cfg.splits); device = pick_device()
    model, sc, ck = load_checkpoint("final")
    tmode = ck.get("target_mode", "raw")
    ds_train = WeekDataset(fs, sc, np.where(masks["train"])[0], target_mode=tmode)
    ds_test = WeekDataset(fs, sc, np.where(masks["test"])[0], target_mode=tmode)

    # ---------------- attention
    att = collect_attention(model, ds_test, device)
    by_lag = attention_by_lag(att); by_lag.to_csv(TABLES / "attention_by_lag.csv")
    conc = attention_concentration(att); (TABLES / "attention_concentration.json").write_text(json.dumps(conc, indent=2))
    log.info("attention concentration: %s", conc)
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    axes[0].plot(by_lag.index, by_lag.pool_all, color=P.PALETTE["blue"], label="all weeks")
    axes[0].plot(by_lag.index, by_lag.pool_high_vol, color=P.PALETTE["orange"], lw=1.1, label="high-vol regime")
    axes[0].plot(by_lag.index, by_lag.pool_low_vol, color=P.PALETTE["aqua"], lw=1.1, label="low-vol regime")
    axes[0].axhline(1 / 60, color=P.TEXT2, lw=0.8, ls=":"); axes[0].set_xlabel("weeks before decision"); axes[0].set_ylabel("pooling attention")
    axes[0].invert_xaxis(); axes[0].legend(); axes[0].set_title("Which past weeks the model attends to")
    axes[1].plot(by_lag.index, by_lag.self_attn_all, color=P.PALETTE["blue"]); axes[1].invert_xaxis()
    axes[1].axhline(1 / 60, color=P.TEXT2, lw=0.8, ls=":"); axes[1].set_xlabel("weeks before decision"); axes[1].set_title("Self-attention received (last layer)")
    P.save(fig, FIGURES / "attention_by_lag.png")

    # ---------------- SHAP
    shap_res = compute_shap(model, sc, ds_train, ds_test, n_background=300, n_explain=600)
    fi = shap_res.feature_importance(); gi = shap_res.group_importance(); li = shap_res.lag_importance()
    fi.to_csv(TABLES / "shap_feature_importance.csv"); gi.to_csv(TABLES / "shap_group_importance.csv"); li.to_csv(TABLES / "shap_lag_importance.csv")
    top5 = shap_res.top5_share()
    kpi = {"top5_share_of_mean_abs_shap": top5, "top10_share": float(fi.iloc[:10].sum() / fi.sum()),
           "graph_share": float(gi[[g for g in gi.index if g.startswith("graph")]].sum() / gi.sum())}
    (TABLES / "shap_kpis.json").write_text(json.dumps(kpi, indent=2)); log.info("SHAP KPIs: %s", kpi)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
    top = fi.iloc[:15][::-1]
    graph_feats = {n for k, v in GROUPS.items() if k.startswith("graph") for n in v}
    is_graph = [n in graph_feats for n in top.index]
    axes[0].barh(top.index, top.values, color=[P.PALETTE["violet"] if g else P.PALETTE["blue"] for g in is_graph], height=0.7)
    axes[0].set_xlabel("mean |SHAP| on decision score"); axes[0].set_title("Top-15 features (violet = graph features)")
    g = gi[::-1]
    axes[1].barh(g.index, g.values, color=[P.PALETTE["violet"] if n.startswith("graph") else P.PALETTE["blue"] for n in g.index], height=0.7)
    axes[1].set_title("Importance by feature group")
    P.save(fig, FIGURES / "shap_importance.png")

    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.plot(li.index, li.values, color=P.PALETTE["blue"]); ax.invert_xaxis(); ax.set_xlabel("weeks before decision"); ax.set_ylabel("mean |SHAP|")
    ax.set_title("Attribution mass by lag"); P.save(fig, FIGURES / "shap_by_lag.png")

    # example explanations for the first test week's top picks (interpretable decision rules)
    examples = []
    wk = shap_res.weeks[0]
    m = shap_res.weeks == wk
    sub_idx = np.where(m)[0]
    ds_scores = {}
    for i in sub_idx:
        top_c = shap_res.explain_sample(i, 4)
        examples.append({"week": str(wk.date()), "symbol": shap_res.symbols[i],
                         **{f"factor{j+1}": f"{n} ({v:+.3f})" for j, (n, v) in enumerate(top_c.items())}})
    pd.DataFrame(examples).to_csv(TABLES / "example_explanations.csv", index=False)

    # ---------------- failure analysis on walk-forward predictions
    preds = Predictions.load(RESULTS / "preds" / "mtl.h5")
    rr = realised_simple_returns(fs); bench = benchmark_weekly_returns(fs, panel.bench_ret)
    ic = weekly_ic(preds.score, rr.loc[preds.score.index])
    res = run_backtest(preds.score, rr.loc[preds.score.index], cfg.backtest, u.rf_weekly)
    reg = regime_table(ic, res, fs, u.rf_weekly, bench)
    reg.to_csv(TABLES / "failure_regimes.csv"); (TABLES / "failure_regimes.md").write_text(reg.round(3).to_markdown())
    log.info("\n%s", reg.round(3).to_string())
    turnover_inr = panel.volume * panel.close
    chars = stock_characteristic_table(preds.score, rr, fs, u, turnover_inr)
    chars.to_csv(TABLES / "stock_characteristics.csv")
    for by in ["sector", "beta_bucket", "size_bucket"]:
        g = grouped_summary(chars, by); g.to_csv(TABLES / f"failure_by_{by}.csv")
        (TABLES / f"failure_by_{by}.md").write_text(g.round(3).to_markdown()); log.info("\n%s", g.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    r = reg.reset_index()
    for ax, col, title in [(axes[0], "mean_ic", "Mean weekly rank-IC by regime"), (axes[1], "port_sharpe", "Portfolio Sharpe by regime")]:
        labels = [f"{a}\n{b}" for a, b in zip(r.regime, r.bucket)]
        ax.bar(range(len(r)), r[col], color=[P.PALETTE["blue"] if b in ("high", "bench < -2%") else P.PALETTE["aqua"] for b in r.bucket], width=0.7)
        ax.set_xticks(range(len(r))); ax.set_xticklabels(labels, fontsize=6.5, rotation=0); ax.axhline(0, color=P.TEXT2, lw=0.8); ax.set_title(title)
    P.save(fig, FIGURES / "failure_regimes.png")
    log.info("interpretability outputs written")


if __name__ == "__main__":
    main()
