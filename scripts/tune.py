"""Optuna hyper-parameter search (PRD Week 2, day 5-6).

Objective: validation rank-IC of the Sharpe score (mean weekly Spearman between
score and realised next-week return). Val Sharpe is logged as a user attribute.
Shorter epoch budget than the final run; the best trial is retrained in full by
scripts/train_final.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings

import numpy as np
import optuna

from nifty_mtl.config import RESULTS, Config, load_universe
from nifty_mtl.features.build import FeatureSet
from nifty_mtl.models.dataset import week_masks
from nifty_mtl.models.train import train_model

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("numexpr").setLevel(logging.WARNING)
log = logging.getLogger("tune")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--study", default="mtl_v1")
    args = ap.parse_args()

    fs = FeatureSet.load()
    u = load_universe()
    base = Config()
    masks = week_masks(fs, base.splits)

    def objective(trial: optuna.Trial) -> float:
        cfg = Config()
        cfg.model.d_model = trial.suggest_categorical("d_model", [32, 64, 96])
        cfg.model.n_layers = trial.suggest_int("n_layers", 1, 3)
        cfg.model.dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
        cfg.train.lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        cfg.train.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        cfg.train.w_return = trial.suggest_float("w_return", 0.3, 0.8, step=0.1)
        cfg.train.w_vol = 1.0 - cfg.train.w_return
        cfg.train.lambda_turnover = trial.suggest_categorical("lambda_turnover", [0.0, 0.1, 0.3, 1.0])
        cfg.train.target_mode = trial.suggest_categorical("target_mode", ["raw", "cs_demean"])
        cfg.train.epochs = args.epochs
        cfg.train.patience = 15
        model, sc, hist = train_model(fs, cfg, masks["train"], masks["val"], u.rf_weekly, verbose=False,
                                      tag=f"trial{trial.number}")
        h = hist.to_frame()
        best = h.iloc[hist.best_epoch - 1]
        trial.set_user_attr("val_sharpe", float(best.val_sharpe))
        trial.set_user_attr("best_epoch", int(hist.best_epoch))
        log.info("trial %d: IC %.4f  valSharpe %.3f  best_ep %d  params %s",
                 trial.number, best.val_ic, best.val_sharpe, hist.best_epoch, trial.params)
        return float(best.val_ic)

    storage = f"sqlite:///{RESULTS / 'optuna.db'}"
    study = optuna.create_study(study_name=args.study, storage=storage, direction="maximize",
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=args.trials)
    best = study.best_trial
    out = {"params": best.params, "val_ic": best.value, **best.user_attrs}
    (RESULTS / "best_params.json").write_text(json.dumps(out, indent=2))
    log.info("BEST: %s", out)


if __name__ == "__main__":
    main()
