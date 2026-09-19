"""Train the final static model (train 2019-2022, val 2023-H1) with the best
Optuna parameters, plus its ablation twins, and save checkpoints + curves.
Used for interpretability and the `recommend` CLI; walk-forward results come
from scripts/run_walkforward.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from run_walkforward import apply_best  # noqa: E402

from nifty_mtl import plotting as P
from nifty_mtl.config import FIGURES, RESULTS, TABLES, Config, load_universe
from nifty_mtl.features.build import FeatureSet
from nifty_mtl.models.dataset import WeekDataset, week_masks
from nifty_mtl.models.train import (pick_device, predict, rank_ic, save_checkpoint, static_mask_for, train_model)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("numexpr").setLevel(logging.WARNING)
log = logging.getLogger("final")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["final", "final_sharpe_stop", "final_nograph", "final_ret_only"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    args = ap.parse_args()
    P.setup()
    fs = FeatureSet.load(); u = load_universe()
    best = json.loads((RESULTS / "best_params.json").read_text())["params"]
    masks = week_masks(fs, Config().splits)
    device = pick_device()
    rows = []
    for v in args.variants:
        cfg = apply_best(Config(), best)
        cfg.train.epochs, cfg.train.patience = args.epochs, args.patience
        kw = {}
        if v == "final_sharpe_stop":
            cfg.train.early_stop_metric = "sharpe"
        if v == "final_nograph":
            kw["use_graph"] = False
        if v == "final_ret_only":
            kw["tasks"] = ("ret",)
        model, sc, hist = train_model(fs, cfg, masks["train"], masks["val"], u.rf_weekly, tag=v, **kw)
        save_checkpoint(model, sc, cfg, fs, hist, v, extra=kw)
        hist.to_frame().to_csv(TABLES / f"training_curve_{v}.csv", index=False)
        smask = static_mask_for(fs, kw.get("use_graph", True))
        for split in ["val", "test", "holdout"]:
            ds = WeekDataset(fs, sc, np.where(masks[split])[0], static_mask=smask, target_mode=cfg.train.target_mode)
            p = predict(model, ds, sc, device, cfg.model.vol_clip)
            rows.append({"variant": v, "split": split, "rank_ic": rank_ic(p, fs), "best_epoch": hist.best_epoch})
        log.info("%s done: best epoch %d", v, hist.best_epoch)
    tbl = pd.DataFrame(rows).pivot(index="variant", columns="split", values="rank_ic")
    tbl.to_csv(TABLES / "static_model_ic.csv"); log.info("\n%s", tbl.round(4).to_string())

    # training curves figure for the main model
    h = pd.read_csv(TABLES / "training_curve_final.csv")
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.6))
    axes[0].plot(h.epoch, h.train_loss, color=P.PALETTE["blue"], label="train"); axes[0].plot(h.epoch, h.val_loss, color=P.PALETTE["orange"], label="val")
    axes[0].set_title("MTL loss"); axes[0].legend()
    axes[1].plot(h.epoch, h.val_ic, color=P.PALETTE["blue"]); axes[1].axhline(0, color=P.TEXT2, lw=0.8); axes[1].set_title("Validation rank-IC")
    axes[2].plot(h.epoch, h.val_sharpe, color=P.PALETTE["blue"]); axes[2].axhline(0, color=P.TEXT2, lw=0.8); axes[2].set_title("Validation Sharpe (26 wk)")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.axvline(int(h.epoch[h.val_ic.idxmax()]), color=P.TEXT2, lw=0.8, ls=":")
    P.save(fig, FIGURES / "training_curves.png")


if __name__ == "__main__":
    main()
