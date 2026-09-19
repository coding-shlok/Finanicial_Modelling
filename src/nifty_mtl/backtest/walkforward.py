"""Walk-forward evaluation (PRD §8): retrain every 13 weeks on an expanding window.

Fold k covers prediction weeks [T_k, T_k + 13). Its validation set is the 26
weeks immediately before T_k and its training set is everything from
``train_start`` up to the validation start. Predictions from all folds are
concatenated into one out-of-sample score panel.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nifty_mtl.backtest.baselines import logistic_scores
from nifty_mtl.config import Config
from nifty_mtl.features.build import FeatureSet
from nifty_mtl.models.dataset import WeekDataset
from nifty_mtl.models.train import Predictions, pick_device, predict, static_mask_for, train_model

log = logging.getLogger(__name__)


@dataclass
class Fold:
    k: int
    train: np.ndarray
    val: np.ndarray
    pred: np.ndarray

    def describe(self, weeks: pd.DatetimeIndex) -> str:
        f = lambda m: f"{weeks[m][0].date()}..{weeks[m][-1].date()}" if m.any() else "-"
        return f"fold {self.k}: train {f(self.train)} | val {f(self.val)} | predict {f(self.pred)}"


def make_folds(fs: FeatureSet, cfg: Config, start: str, end: str | None = None,
               val_weeks: int = 26) -> list[Fold]:
    w = fs.weeks
    step = cfg.backtest.retrain_every_weeks
    t0 = int(w.searchsorted(pd.Timestamp(start)))
    t_end = int(w.searchsorted(pd.Timestamp(end), side="right")) if end else len(w) - 1  # last week has no target
    train_from = int(w.searchsorted(pd.Timestamp(cfg.splits.train_start)))
    folds = []
    k = 0
    for ts in range(t0, t_end, step):
        te = min(ts + step, t_end)
        pred = np.zeros(len(w), bool); pred[ts:te] = True
        val = np.zeros(len(w), bool); val[max(train_from, ts - val_weeks):ts] = True
        train = np.zeros(len(w), bool); train[train_from: max(train_from, ts - val_weeks)] = True
        folds.append(Fold(k, train, val, pred)); k += 1
    return folds


def _merge(preds: list[Predictions]) -> Predictions:
    return Predictions(pd.concat([p.ret for p in preds]).sort_index(),
                       pd.concat([p.vol for p in preds]).sort_index(),
                       pd.concat([p.score for p in preds]).sort_index())


def _average(preds: list[Predictions]) -> Predictions:
    ret = sum(p.ret for p in preds) / len(preds)
    vol = sum(p.vol for p in preds) / len(preds)
    return Predictions(ret, vol, ret / vol)


def walk_forward_mtl(fs: FeatureSet, cfg: Config, rf_weekly: float, start: str, end: str | None = None,
                     n_seeds: int = 1, use_graph: bool = True, tasks=("ret", "vol"), tag: str = "wf") -> tuple[Predictions, list[dict]]:
    folds = make_folds(fs, cfg, start, end)
    device = pick_device(cfg.train.device)
    smask = static_mask_for(fs, use_graph)
    out, info = [], []
    for f in folds:
        log.info("[%s] %s", tag, f.describe(fs.weeks))
        seed_preds = []
        for s in range(n_seeds):
            c = copy.deepcopy(cfg); c.train.seed = cfg.train.seed + s
            model, sc, hist = train_model(fs, c, f.train, f.val, rf_weekly, use_graph=use_graph,
                                          tasks=tasks, verbose=False, tag=f"{tag}-f{f.k}-s{s}")
            ds = WeekDataset(fs, sc, np.where(f.pred)[0], static_mask=smask, target_mode=c.train.target_mode)
            seed_preds.append(predict(model, ds, sc, device, c.model.vol_clip))
            info.append({"fold": f.k, "seed": s, "best_epoch": hist.best_epoch, "val_sharpe": hist.best_val_sharpe,
                         "val_ic": hist.to_frame().val_ic.iloc[hist.best_epoch - 1]})
        out.append(_average(seed_preds))
        log.info("[%s] fold %d done: best epochs %s", tag, f.k, [i["best_epoch"] for i in info if i["fold"] == f.k])
    return _merge(out), info


def walk_forward_logistic(fs: FeatureSet, cfg: Config, start: str, end: str | None = None) -> pd.DataFrame:
    folds = make_folds(fs, cfg, start, end)
    parts = []
    for f in folds:
        s = logistic_scores(fs, f.train | f.val, f.pred)
        parts.append(s[f.pred])
    return pd.concat(parts).sort_index()
