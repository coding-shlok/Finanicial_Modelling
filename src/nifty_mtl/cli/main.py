"""Command-line interface.

  nifty-mtl data            fetch + preprocess + build features
  nifty-mtl train           train the static model with configs/default.yaml (or best params)
  nifty-mtl recommend       rank the universe for the latest week with explanations
  nifty-mtl backtest        run a quick backtest of a saved prediction panel
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from nifty_mtl.config import RESULTS, TABLES, Config, load_universe

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console(width=200)
warnings.filterwarnings("ignore")


def _log():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("numexpr").setLevel(logging.WARNING)


@app.command()
def data(force: bool = typer.Option(False, help="re-download from yfinance")):
    """Fetch daily OHLCV, build the weekly panel and the feature tensor."""
    _log()
    from nifty_mtl.data.fetch import fetch_daily
    from nifty_mtl.data.preprocess import build_weekly, quality_report
    from nifty_mtl.features.build import build_features
    daily = fetch_daily(force=force)
    panel = build_weekly(daily); panel.save()
    quality_report(panel).to_csv(TABLES / "data_quality.csv")
    fs = build_features(panel); fs.save()
    console.print(f"[green]done[/] {len(fs.weeks)} weeks x {len(fs.symbols)} symbols, "
                  f"{int(fs.sample_ok.sum())} usable samples, latest week {fs.weeks[-1].date()}")


@app.command()
def train(name: str = "final", epochs: int = 200, patience: int = 20,
          use_best: bool = typer.Option(True, help="use results/best_params.json if present")):
    """Train the static MTL model on 2019-2022, early-stop on 2023-H1."""
    _log()
    from nifty_mtl.features.build import FeatureSet
    from nifty_mtl.models.dataset import week_masks
    from nifty_mtl.models.train import save_checkpoint, train_model
    fs = FeatureSet.load(); u = load_universe(); cfg = Config()
    bp = RESULTS / "best_params.json"
    if use_best and bp.exists():
        import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
        from run_walkforward import apply_best
        cfg = apply_best(cfg, json.loads(bp.read_text())["params"])
    cfg.train.epochs, cfg.train.patience = epochs, patience
    masks = week_masks(fs, cfg.splits)
    model, sc, hist = train_model(fs, cfg, masks["train"], masks["val"], u.rf_weekly, tag=name)
    p = save_checkpoint(model, sc, cfg, fs, hist, name)
    console.print(f"[green]saved[/] {p}  best epoch {hist.best_epoch}  val Sharpe {hist.best_val_sharpe:.2f}")


@app.command()
def recommend(checkpoint: str = "final", week: str = typer.Option(None, help="decision week (YYYY-MM-DD), default latest"),
              explain: bool = True, top: int = 10):
    """Rank all stocks for a decision week: BUY / SELL / HOLD with the top contributing factors."""
    _log()
    import torch
    from nifty_mtl.backtest.engine import signals_from_scores
    from nifty_mtl.features.build import FeatureSet
    from nifty_mtl.models.dataset import WeekDataset
    from nifty_mtl.models.train import load_checkpoint, pick_device, predict
    fs = FeatureSet.load(); u = load_universe(); cfg = Config()
    model, sc, ck = load_checkpoint(checkpoint)
    tmode = ck.get("target_mode", "raw")
    # the latest week has no realised target, so sample_ok is False there: relax it for inference
    fs_inf = fs
    if week is None:
        t = len(fs.weeks) - 1
    else:
        t = fs.week_index(week)
    L = fs.lookback
    feat_ok = fs.valid & np.isfinite(fs.seq).all(-1) & np.isfinite(fs.static).all(-1)
    win_ok = pd.DataFrame(feat_ok).rolling(L).sum().to_numpy() == L
    fs_inf.sample_ok = fs.sample_ok.copy(); fs_inf.sample_ok[t] = win_ok[t]
    ds = WeekDataset(fs_inf, sc, np.array([t]), target_mode=tmode)
    preds = predict(model, ds, sc, pick_device(), cfg.model.vol_clip)
    wk = fs.weeks[t]
    score = preds.score.loc[[wk]]
    sig = signals_from_scores(score, cfg.backtest.top_frac, cfg.backtest.bottom_frac).loc[wk]
    row = pd.DataFrame({"ret_pred": preds.ret.loc[wk], "vol_pred": preds.vol.loc[wk], "score": score.loc[wk]}).dropna()
    row["pct"] = row.score.rank(pct=True)
    row["signal"] = sig.reindex(row.index).map({1: "BUY", -1: "SELL", 0: "HOLD"})
    row = row.sort_values("score", ascending=False)

    factors = {}
    if explain:
        from nifty_mtl.interpret.shap_analysis import compute_shap
        from nifty_mtl.models.dataset import week_masks
        masks = week_masks(fs, cfg.splits)
        ds_bg = WeekDataset(fs, sc, np.where(masks["train"])[0], target_mode=tmode)
        res = compute_shap(model, sc, ds_bg, ds, n_background=200, n_explain=None)
        for i, s in enumerate(res.symbols):
            factors[s] = ", ".join(f"{n} {v:+.2f}" for n, v in res.explain_sample(i, 3).items())

    tbl = Table(title=f"Recommendations for week ending {wk.date()} (decision at Friday close, horizon = next week)")
    for c in ["Rank", "Symbol", "Sector", "Signal", "E[ret]", "E[vol]", "Score", "Pctl", "Top factors (SHAP on score)"]:
        tbl.add_column(c)
    for i, (s, r) in enumerate(row.iterrows(), 1):
        colour = {"BUY": "green", "SELL": "red", "HOLD": "white"}[r.signal]
        if i <= top or r.signal != "HOLD":
            tbl.add_row(str(i), s, u.sector_of[s], f"[{colour}]{r.signal}[/]", f"{r.ret_pred*100:+.2f}%",
                        f"{r.vol_pred*100:.2f}%", f"{r.score:+.3f}", f"{r.pct:.0%}", factors.get(s, ""))
    console.print(tbl)
    out = RESULTS / f"recommendations_{wk.date()}.csv"
    row.assign(factors=pd.Series(factors)).to_csv(out)
    console.print(f"saved {out}")


@app.command()
def backtest(preds: str = "results/preds/mtl.h5", mode: str = "rebalance"):
    """Backtest a saved prediction panel and print the metrics."""
    from nifty_mtl.backtest.baselines import benchmark_weekly_returns
    from nifty_mtl.backtest.engine import run_backtest
    from nifty_mtl.backtest.metrics import summary
    from nifty_mtl.data.preprocess import WeeklyPanel
    from nifty_mtl.features.build import FeatureSet
    from nifty_mtl.models.train import Predictions, realised_simple_returns
    fs = FeatureSet.load(); u = load_universe(); cfg = Config(); panel = WeeklyPanel.load()
    p = Predictions.load(Path(preds))
    rr = realised_simple_returns(fs).loc[p.score.index]
    res = run_backtest(p.score, rr, cfg.backtest, u.rf_weekly, mode=mode)
    s = summary(res.ret, benchmark_weekly_returns(fs, panel.bench_ret).loc[p.score.index], u.rf_weekly, res.turnover)
    for k, v in s.items():
        console.print(f"{k:24s} {v: .4f}" if isinstance(v, float) else f"{k:24s} {v}")


if __name__ == "__main__":
    app()
