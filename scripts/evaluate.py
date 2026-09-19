"""Evaluate walk-forward predictions vs baselines (PRD Week 3, §8).

Inputs : results/preds/*.h5 from scripts/run_walkforward.py
Outputs: results/tables/*.csv|md  and  results/figures/*.png|pdf
"""
from __future__ import annotations

import json
import logging
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nifty_mtl import plotting as P
from nifty_mtl.backtest.baselines import (benchmark_weekly_returns, inverse_vol_scores, momentum_scores,
                                          random_scores)
from nifty_mtl.backtest.engine import run_backtest
from nifty_mtl.backtest.metrics import bootstrap_sharpe_ci, sharpe, summary
from nifty_mtl.config import FIGURES, RESULTS, TABLES, BacktestConfig, Config, load_universe
from nifty_mtl.data.preprocess import WeeklyPanel
from nifty_mtl.features.build import FeatureSet
from nifty_mtl.models.train import Predictions, realised_simple_returns

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("eval")
PREDS = RESULTS / "preds"

VARIANT_LABEL = {"mtl": "MTL (ours)", "mtl_nograph": "MTL no-graph", "mtl_ret_only": "MTL return-only",
                 "mtl_nocost": "MTL no-cost", "mtl_cost": "MTL +cost-term", "logistic": "Logistic"}


def weekly_ic(score: pd.DataFrame, rr: pd.DataFrame) -> pd.Series:
    out = {}
    for t in score.index:
        a, b = score.loc[t], rr.loc[t]
        m = a.notna() & b.notna()
        if m.sum() > 5:
            out[t] = a[m].rank().corr(b[m].rank())
    return pd.Series(out)


def main():
    P.setup()
    fs = FeatureSet.load(); u = load_universe(); panel = WeeklyPanel.load()
    cfg = Config()
    if (RESULTS / "final_config.json").exists():
        raw = json.loads((RESULTS / "final_config.json").read_text())
        cfg.backtest = BacktestConfig(**raw["backtest"])
    rf = u.rf_weekly
    rr = realised_simple_returns(fs)
    bench = benchmark_weekly_returns(fs, panel.bench_ret)

    # ---------- load score panels
    scores: dict[str, pd.DataFrame] = {}
    for v, label in VARIANT_LABEL.items():
        p = PREDS / f"{v}.h5"
        if not p.exists():
            continue
        scores[label] = pd.read_hdf(p, "score") if v == "logistic" else Predictions.load(p).score
    oos_index = scores["MTL (ours)"].index
    scores["Momentum"] = momentum_scores(fs).loc[oos_index]
    scores["Inverse-vol"] = inverse_vol_scores(fs).loc[oos_index]
    scores = {k: v.reindex(oos_index) for k, v in scores.items()}
    periods = {
        "test": oos_index[(oos_index >= cfg.splits.test_start) & (oos_index <= cfg.splits.test_end)],
        "holdout": oos_index[oos_index >= cfg.splits.holdout_start],
        "all_oos": oos_index,
    }

    # ---------- main table: every strategy x period
    rows, curves = [], {}
    for pname, idx in periods.items():
        if len(idx) == 0:
            continue
        for label, sc in scores.items():
            covered = sc.reindex(idx).notna().any(axis=1).mean()
            if covered < 0.9:          # ablations run on the test period only are not reported elsewhere
                continue
            res = run_backtest(sc.reindex(idx), rr.loc[idx], cfg.backtest, rf)
            s = summary(res.ret, bench.loc[idx], rf, res.turnover)
            s.update({"strategy": label, "period": pname, "mean_ic": float(weekly_ic(sc.loc[idx], rr.loc[idx]).mean())})
            if pname == "all_oos":
                lo, hi = bootstrap_sharpe_ci(res.ret, rf)
                s["sharpe_ci_lo"], s["sharpe_ci_hi"] = lo, hi
                curves[label] = res
            rows.append(s)
        # random 10-stock portfolio: average over 20 seeds
        rs = [summary(run_backtest(random_scores(fs, k).loc[idx], rr.loc[idx], cfg.backtest, rf).ret, bench.loc[idx], rf)
              for k in range(20)]
        r = pd.DataFrame(rs).mean(numeric_only=True).to_dict(); r.update({"strategy": "Random", "period": pname})
        rows.append(r)
        # buy & hold Nifty 50
        b = summary(bench.loc[idx].dropna(), None, rf); b.update({"strategy": "Nifty 50", "period": pname})
        if pname == "all_oos":
            lo, hi = bootstrap_sharpe_ci(bench.loc[idx].dropna(), rf); b["sharpe_ci_lo"], b["sharpe_ci_hi"] = lo, hi
        rows.append(b)
    main_tbl = pd.DataFrame(rows).set_index(["period", "strategy"])
    cols = ["n_weeks", "cagr", "ann_vol", "sharpe", "sharpe_weekly", "sortino", "max_drawdown", "calmar", "win_rate",
            "profit_factor", "information_ratio", "mean_ic", "avg_weekly_turnover", "sharpe_ci_lo", "sharpe_ci_hi"]
    main_tbl = main_tbl.reindex(columns=[c for c in cols if c in main_tbl.columns])
    main_tbl.to_csv(TABLES / "main_results.csv")
    (TABLES / "main_results.md").write_text(main_tbl.round(3).to_markdown())
    log.info("\n%s", main_tbl.loc["all_oos"].round(3).to_string())

    # ---------- sensitivity: transaction cost, position cap, portfolio mode
    sens = []
    mtl = scores["MTL (ours)"]
    for c in [0.0, 0.0005, 0.001, 0.0015, 0.0025]:
        bc = BacktestConfig(**{**cfg.backtest.__dict__, "cost_per_side": c})
        res = run_backtest(mtl, rr.loc[oos_index], bc, rf)
        sens.append({"setting": f"cost {c*100:.2f}%/side", "sharpe": sharpe(res.ret, rf), "cagr": summary(res.ret)["cagr"]})
    for cap in [0.05, 0.10, 0.20]:
        bc = BacktestConfig(**{**cfg.backtest.__dict__, "max_position": cap})
        res = run_backtest(mtl, rr.loc[oos_index], bc, rf)
        sens.append({"setting": f"max position {cap:.0%}", "sharpe": sharpe(res.ret, rf), "cagr": summary(res.ret)["cagr"]})
    for mode in ["rebalance", "hold_zone", "long_short"]:
        res = run_backtest(mtl, rr.loc[oos_index], cfg.backtest, rf, mode=mode)
        s = summary(res.ret, bench.loc[oos_index], rf, res.turnover)
        sens.append({"setting": f"mode {mode}", "sharpe": s["sharpe"], "cagr": s["cagr"], "max_drawdown": s["max_drawdown"],
                     "turnover": s["avg_weekly_turnover"]})
        if mode != "rebalance":
            curves[f"MTL {mode}"] = res
    for frac in [0.10, 0.20]:
        bc = BacktestConfig(**{**cfg.backtest.__dict__, "top_frac": frac, "max_position": 1.0 / max(1, round(frac * 48))})
        res = run_backtest(mtl, rr.loc[oos_index], bc, rf)
        sens.append({"setting": f"top {frac:.0%} of universe", "sharpe": sharpe(res.ret, rf), "cagr": summary(res.ret)["cagr"]})
    # post-hoc decision-layer variants (not used for any selection; reported for diagnosis)
    mtl_preds = Predictions.load(PREDS / "mtl.h5")
    for name, sc in [("decision: E[ret]/E[vol] (default)", mtl_preds.score), ("decision: E[ret] only", mtl_preds.ret),
                     ("decision: E[ret]/(E[vol]+0.03)", mtl_preds.ret / (mtl_preds.vol + 0.03)),
                     ("decision: -E[vol] only", -mtl_preds.vol)]:
        sc = sc.reindex(oos_index)
        for pname, idx in periods.items():
            res = run_backtest(sc.loc[idx], rr.loc[idx], cfg.backtest, rf)
            sens.append({"setting": f"{name} [{pname}]", "sharpe": sharpe(res.ret, rf), "cagr": summary(res.ret)["cagr"],
                         "ic": float(weekly_ic(sc.loc[idx], rr.loc[idx]).mean())})
    sens_tbl = pd.DataFrame(sens).set_index("setting")
    sens_tbl.to_csv(TABLES / "sensitivity.csv"); (TABLES / "sensitivity.md").write_text(sens_tbl.round(3).to_markdown())
    log.info("\n%s", sens_tbl.round(3).to_string())

    # ---------- decile analysis: mean next-week return by score decile (all OOS)
    dec = []
    for t in oos_index:
        s, r = mtl.loc[t].dropna(), rr.loc[t]
        m = s.index[r[s.index].notna()]
        if len(m) < 10:
            continue
        q = pd.qcut(s[m].rank(method="first"), 10, labels=False)
        dec.append(r[m].groupby(q.values).mean())
    dec_tbl = pd.DataFrame(dec).mean() * 100
    dec_tbl.index = [f"D{i+1}" for i in dec_tbl.index]
    dec_tbl.to_csv(TABLES / "decile_returns.csv")

    # ---------- yearly breakdown for MTL and Nifty
    yr = pd.DataFrame({"MTL (ours)": curves["MTL (ours)"].ret, "Nifty 50": bench.loc[oos_index]})
    yearly = yr.groupby(yr.index.year).apply(lambda d: pd.Series({
        "mtl_return": (1 + d["MTL (ours)"]).prod() - 1, "nifty_return": (1 + d["Nifty 50"]).prod() - 1,
        "mtl_sharpe": sharpe(d["MTL (ours)"], rf), "nifty_sharpe": sharpe(d["Nifty 50"].dropna(), rf),
        "weeks": len(d)}))
    yearly.to_csv(TABLES / "yearly.csv"); (TABLES / "yearly.md").write_text(yearly.round(3).to_markdown())

    # ================= FIGURES =================
    # 1. equity curves (log scale), all OOS
    fig, ax = plt.subplots(figsize=(7, 3.6))
    order = ["MTL (ours)", "MTL no-graph", "MTL return-only", "Momentum", "Inverse-vol", "Logistic", "Nifty 50"]
    for label in order:
        if label == "Nifty 50":
            eq = (1 + bench.loc[oos_index].fillna(0)).cumprod()
        elif label in curves:
            eq = (1 + curves[label].ret).cumprod()
        else:
            continue
        ax.plot(eq.index, eq.values, color=P.color(label), label=label, lw=2.0 if label == "MTL (ours)" else 1.2,
                ls="-" if label != "Nifty 50" else "--")
    ax.axvspan(pd.Timestamp(cfg.splits.holdout_start), oos_index[-1], color="#f2f1ec", zorder=0, lw=0)
    ax.set_yscale("log"); ax.set_ylabel("Growth of 1 (log)"); ax.set_title("Out-of-sample equity curves, net of 0.1%/side costs")
    ax.text(pd.Timestamp(cfg.splits.holdout_start), ax.get_ylim()[0] * 1.02, " untouched holdout", va="bottom", color=P.TEXT2, fontsize=8)
    ax.text(oos_index[0], ax.get_ylim()[0] * 1.02, " PRD test period", va="bottom", color=P.TEXT2, fontsize=8)
    ax.legend(ncol=2, loc="upper left")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    P.save(fig, FIGURES / "equity_curves.png")

    # 2. drawdowns
    fig, ax = plt.subplots(figsize=(7, 2.4))
    for label in ["MTL (ours)", "Nifty 50"]:
        r = curves[label].ret if label in curves else bench.loc[oos_index].fillna(0)
        eq = (1 + r).cumprod(); dd = eq / eq.cummax() - 1
        ax.fill_between(dd.index, dd.values, 0, color=P.color(label), alpha=0.25 if label == "MTL (ours)" else 0.12, lw=0)
        ax.plot(dd.index, dd.values, color=P.color(label), lw=1.2, label=label)
    ax.set_ylabel("Drawdown"); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}")); ax.legend(loc="lower left")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    P.save(fig, FIGURES / "drawdowns.png")

    # 3. decile bar chart
    fig, ax = plt.subplots(figsize=(5, 3))
    cols_ = [P.PALETTE["red"] if i < 1 else (P.PALETTE["blue"] if i >= 9 else "#b9b8b1") for i in range(10)]
    ax.bar(dec_tbl.index, dec_tbl.values, color=cols_, width=0.7)
    ax.axhline(0, color=P.TEXT2, lw=0.8); ax.set_ylabel("Mean next-week return (%)"); ax.set_title("Return by MTL score decile (out-of-sample)")
    P.save(fig, FIGURES / "decile_returns.png")

    # 4. rolling 26-week IC
    fig, ax = plt.subplots(figsize=(7, 2.6))
    for label in ["MTL (ours)", "Momentum", "Logistic"]:
        if label not in scores:
            continue
        ic = weekly_ic(scores[label].dropna(how="all"), rr.loc[oos_index]).rolling(26, min_periods=13).mean()
        ax.plot(ic.index, ic.values, color=P.color(label), label=label, lw=1.4)
    ax.axhline(0, color=P.TEXT2, lw=0.8); ax.set_ylabel("Rolling 26w rank-IC"); ax.legend(ncol=3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    P.save(fig, FIGURES / "rolling_ic.png")

    # 5. Sharpe with bootstrap CIs (all OOS)
    t = main_tbl.loc["all_oos"]
    t = t[t["sharpe_ci_lo"].notna()].sort_values("sharpe")
    fig, ax = plt.subplots(figsize=(5.5, 3))
    y = np.arange(len(t))
    ax.hlines(y, t["sharpe_ci_lo"], t["sharpe_ci_hi"], color="#b9b8b1", lw=2)
    ax.scatter(t["sharpe"], y, color=[P.color(n) for n in t.index], s=36, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(t.index); ax.set_xlabel("Annualised Sharpe (95% block-bootstrap CI)")
    ax.axvline(0, color=P.TEXT2, lw=0.8)
    P.save(fig, FIGURES / "sharpe_ci.png")

    # 6. cost sensitivity
    cs = sens_tbl[sens_tbl.index.str.startswith("cost")]
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    ax.plot([0, 0.05, 0.10, 0.15, 0.25], cs["sharpe"].values, marker="o", color=P.color("MTL (ours)"), ms=5)
    ax.set_xlabel("Transaction cost per side (%)"); ax.set_ylabel("Annualised Sharpe"); ax.axhline(0, color=P.TEXT2, lw=0.8)
    P.save(fig, FIGURES / "cost_sensitivity.png")

    # 7. training curves from a fold csv (if available) + val Sharpe noise illustration from a static run
    log.info("tables -> %s ; figures -> %s", TABLES, FIGURES)


if __name__ == "__main__":
    main()
