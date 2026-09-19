"""Walk-forward predictions for the MTL model and its ablations (PRD Week 3).

Writes results/preds/<variant>.h5 with ret/vol/score panels covering the test
period (2023-07 .. 2024-12) and the untouched holdout (2025-01 .. today).

Variants
--------
mtl            full model (best Optuna params)
mtl_nograph    graph features zeroed          (ablation: graph)
mtl_ret_only   single-task return head        (ablation: MTL)
mtl_nocost     lambda_turnover = 0            (ablation: cost-aware loss; only if best lambda > 0)
mtl_cost       lambda_turnover = 0.3          (ablation in the other direction; only if best lambda == 0)
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings

import pandas as pd

from nifty_mtl.backtest.walkforward import walk_forward_logistic, walk_forward_mtl
from nifty_mtl.config import RESULTS, Config, load_universe
from nifty_mtl.features.build import FeatureSet

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("numexpr").setLevel(logging.WARNING)
log = logging.getLogger("wf")
PREDS = RESULTS / "preds"
PREDS.mkdir(exist_ok=True)


def apply_best(cfg: Config, params: dict) -> Config:
    cfg.model.d_model = params.get("d_model", cfg.model.d_model); cfg.model.n_layers = params.get("n_layers", cfg.model.n_layers)
    cfg.model.dropout = params["dropout"]
    cfg.train.lr = params["lr"]; cfg.train.weight_decay = params["weight_decay"]
    cfg.train.w_return = params["w_return"]; cfg.train.w_vol = 1 - params["w_return"]
    cfg.train.lambda_turnover = params["lambda_turnover"]; cfg.train.target_mode = params["target_mode"]
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["mtl", "mtl_nograph", "mtl_ret_only", "mtl_nocost", "mtl_cost", "logistic"])
    ap.add_argument("--ablation-seeds", type=int, default=1, help="seeds for ablation variants (main uses --seeds)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--end", default=None, help="last prediction week (default: latest)")
    args = ap.parse_args()

    fs = FeatureSet.load(); u = load_universe()
    best = json.loads((RESULTS / "best_params.json").read_text())["params"]
    cfg = apply_best(Config(), best)
    cfg.train.epochs, cfg.train.patience = args.epochs, args.patience
    (RESULTS / "final_config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    start = cfg.splits.test_start
    log.info("walk-forward from %s with params %s", start, best)

    infos = {}
    for v in args.variants:
        if v == "logistic":
            s = walk_forward_logistic(fs, cfg, start, args.end)
            s.to_hdf(PREDS / "logistic.h5", key="score", mode="w")
            continue
        if v == "mtl_nocost" and cfg.train.lambda_turnover == 0:
            log.info("skip mtl_nocost: best lambda is already 0")
            continue
        if v == "mtl_cost" and cfg.train.lambda_turnover > 0:
            log.info("skip mtl_cost: best lambda is already > 0")
            continue
        c = apply_best(Config(), best)
        c.train.epochs, c.train.patience = args.epochs, args.patience
        kw = {}
        if v == "mtl_nograph":
            kw["use_graph"] = False
        if v == "mtl_ret_only":
            kw["tasks"] = ("ret",)
        if v == "mtl_nocost":
            c.train.lambda_turnover = 0.0
        if v == "mtl_cost":
            c.train.lambda_turnover = 0.3
        n_seeds = args.seeds if v == "mtl" else args.ablation_seeds
        preds, info = walk_forward_mtl(fs, c, u.rf_weekly, start, args.end, n_seeds=n_seeds, tag=v, **kw)
        preds.save(PREDS / f"{v}.h5")
        infos[v] = info
        pd.DataFrame(info).to_csv(PREDS / f"{v}_folds.csv", index=False)
        log.info("saved %s: %d weeks", v, len(preds.score))


if __name__ == "__main__":
    main()
