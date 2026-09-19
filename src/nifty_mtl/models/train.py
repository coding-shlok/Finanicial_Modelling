"""Training loop with early stopping on validation portfolio Sharpe (PRD §6)."""
from __future__ import annotations

import copy
import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from nifty_mtl.backtest.engine import run_backtest
from nifty_mtl.backtest.metrics import sharpe
from nifty_mtl.config import CHECKPOINTS, BacktestConfig, Config, ModelConfig, TrainConfig
from nifty_mtl.features.build import FeatureSet
from nifty_mtl.models.dataset import Scalers, WeekDataset, collate_identity, week_masks
from nifty_mtl.models.losses import mtl_loss
from nifty_mtl.models.mtl import MTLModel, denormalise, sharpe_score

log = logging.getLogger(__name__)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    # The model is tiny and batches are ~50 rows: CPU is usually faster than MPS launch overhead.
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def static_mask_for(fs: FeatureSet, use_graph: bool = True, use_sector: bool = True) -> np.ndarray:
    m = np.ones(fs.static.shape[-1], dtype=np.float32)
    if not use_graph:
        m[: fs.n_graph] = 0.0
    if not use_sector:
        m[fs.n_graph:] = 0.0
    return m


@dataclass
class Predictions:
    ret: pd.DataFrame
    vol: pd.DataFrame
    score: pd.DataFrame

    def save(self, path: Path) -> None:
        with pd.HDFStore(path, mode="w") as st:
            st.put("ret", self.ret); st.put("vol", self.vol); st.put("score", self.score)

    @classmethod
    def load(cls, path: Path) -> "Predictions":
        with pd.HDFStore(path, mode="r") as st:
            return cls(st["ret"], st["vol"], st["score"])


@torch.no_grad()
def predict(model: MTLModel, ds: WeekDataset, scalers: Scalers, device, vol_clip=(0.01, 0.10)) -> Predictions:
    model.eval()
    fs = ds.fs
    T, N = fs.y_ret.shape
    R = np.full((T, N), np.nan, dtype=np.float32); V = R.copy()
    for i in range(len(ds)):
        b = ds[i]
        r, v = model(b["seq"].to(device), b["static"].to(device))
        r, v = denormalise(r, v, scalers, vol_clip)
        R[b["t"], b["n_idx"].numpy()] = r.cpu().numpy()
        V[b["t"], b["n_idx"].numpy()] = v.cpu().numpy()
    weeks_used = [ds.week_ids[i] for i in range(len(ds))]
    idx = fs.weeks[weeks_used]
    ret = pd.DataFrame(R[weeks_used], index=idx, columns=fs.symbols)
    vol = pd.DataFrame(V[weeks_used], index=idx, columns=fs.symbols)
    return Predictions(ret, vol, ret / vol)


def realised_simple_returns(fs: FeatureSet) -> pd.DataFrame:
    """Simple return over week t+1, indexed by decision week t. NaN where not usable."""
    r = np.where(fs.sample_ok, np.exp(fs.y_ret) - 1, np.nan)
    return pd.DataFrame(r, index=fs.weeks, columns=fs.symbols)


def portfolio_sharpe(preds: Predictions, fs: FeatureSet, bt_cfg: BacktestConfig, rf_weekly: float) -> float:
    rr = realised_simple_returns(fs).loc[preds.score.index]
    res = run_backtest(preds.score, rr, bt_cfg, rf_weekly=rf_weekly, mode="rebalance")
    return sharpe(res.ret, rf_weekly)


@dataclass
class TrainHistory:
    epoch: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_sharpe: list[float] = field(default_factory=list)
    val_ic: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_sharpe: float = float("-inf")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"epoch": self.epoch, "train_loss": self.train_loss, "val_loss": self.val_loss,
                             "val_sharpe": self.val_sharpe, "val_ic": self.val_ic})


def _epoch_loss(model, loader, scalers, tcfg: TrainConfig, device, opt=None) -> float:
    train = opt is not None
    model.train(train)
    tot, n = 0.0, 0
    for b in loader:
        seq, st = b["seq"].to(device), b["static"].to(device)
        yr, yv = b["y_ret"].to(device), b["y_vol"].to(device)
        with torch.set_grad_enabled(train):
            r, v = model(seq, st)
            prev = None
            if tcfg.lambda_turnover > 0 and "prev_seq" in b:
                prev = model(b["prev_seq"].to(device), b["prev_static"].to(device))
            loss, _ = mtl_loss(r, v, yr, yv, scalers, tcfg.w_return, tcfg.w_vol,
                               lam=tcfg.lambda_turnover, cost=tcfg.cost_per_side, tau=tcfg.tau, prev=prev)
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                opt.step()
        tot += float(loss); n += 1
    return tot / max(n, 1)


def rank_ic(preds: Predictions, fs: FeatureSet) -> float:
    """Mean weekly Spearman correlation between score and realised return."""
    rr = realised_simple_returns(fs).loc[preds.score.index]
    ics = []
    for t in preds.score.index:
        a, b = preds.score.loc[t], rr.loc[t]
        m = a.notna() & b.notna()
        if m.sum() > 5:
            ics.append(a[m].rank().corr(b[m].rank()))
    return float(np.nanmean(ics)) if ics else float("nan")


def train_model(fs: FeatureSet, cfg: Config, train_weeks: np.ndarray, val_weeks: np.ndarray,
                rf_weekly: float, use_graph: bool = True, use_sector: bool = True,
                tasks: tuple[str, ...] = ("ret", "vol"), verbose: bool = True,
                tag: str = "mtl") -> tuple[MTLModel, Scalers, TrainHistory]:
    tcfg, mcfg = cfg.train, cfg.model
    set_seed(tcfg.seed)
    device = pick_device(tcfg.device)
    scalers = Scalers.fit(fs, train_weeks, tcfg.target_mode)
    smask = static_mask_for(fs, use_graph, use_sector)
    tr_ds = WeekDataset(fs, scalers, np.where(train_weeks)[0], static_mask=smask, target_mode=tcfg.target_mode)
    va_ds = WeekDataset(fs, scalers, np.where(val_weeks)[0], static_mask=smask, target_mode=tcfg.target_mode)
    tr_loader = DataLoader(tr_ds, batch_size=1, shuffle=True, collate_fn=collate_identity)
    va_loader = DataLoader(va_ds, batch_size=1, shuffle=False, collate_fn=collate_identity)

    model = MTLModel(fs.seq.shape[-1], fs.static.shape[-1], fs.lookback, mcfg, tasks).to(device)
    # single-task ablations: zero the other task's loss weight
    if tasks == ("ret",):
        tcfg = copy.copy(tcfg); tcfg.w_vol = 0.0; tcfg.w_return = 1.0
    elif tasks == ("vol",):
        tcfg = copy.copy(tcfg); tcfg.w_return = 0.0; tcfg.w_vol = 1.0; tcfg.early_stop_metric = "loss"
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr, betas=(0.9, 0.999), weight_decay=tcfg.weight_decay)

    hist = TrainHistory()
    best_state, bad, best_crit = copy.deepcopy(model.state_dict()), 0, float("-inf")
    t0 = time.time()
    for ep in range(1, tcfg.epochs + 1):
        tl = _epoch_loss(model, tr_loader, scalers, tcfg, device, opt)
        vl = _epoch_loss(model, va_loader, scalers, tcfg, device)
        preds = predict(model, va_ds, scalers, device, mcfg.vol_clip)
        vs = portfolio_sharpe(preds, fs, cfg.backtest, rf_weekly) if tasks != ("vol",) else -vl
        ic = rank_ic(preds, fs)
        hist.epoch.append(ep); hist.train_loss.append(tl); hist.val_loss.append(vl)
        hist.val_sharpe.append(vs); hist.val_ic.append(ic)
        crit = {"sharpe": vs, "ic": ic, "loss": -vl}[tcfg.early_stop_metric]
        improved = np.isfinite(crit) and crit > best_crit + 1e-6
        if improved:
            best_crit, hist.best_val_sharpe, hist.best_epoch, bad = float(crit), float(vs), ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if verbose and (ep % 5 == 0 or improved or ep == 1):
            log.info("[%s] ep %3d  train %.4f  val %.4f  valSharpe %+.3f  IC %+.4f  %s (%.0fs)",
                     tag, ep, tl, vl, vs, ic, "*" if improved else "", time.time() - t0)
        if bad >= tcfg.patience:
            if verbose:
                log.info("[%s] early stop at epoch %d (best %d, val Sharpe %.3f)", tag, ep, hist.best_epoch, hist.best_val_sharpe)
            break
    model.load_state_dict(best_state)
    return model, scalers, hist


def save_checkpoint(model: MTLModel, scalers: Scalers, cfg: Config, fs: FeatureSet, hist: TrainHistory,
                    name: str, extra: dict | None = None) -> Path:
    path = CHECKPOINTS / f"{name}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "scalers": scalers.to_dict(),
        "config": cfg.to_dict(),
        "f_seq": fs.seq.shape[-1], "f_static": fs.static.shape[-1], "lookback": fs.lookback,
        "tasks": model.tasks,
        "target_mode": cfg.train.target_mode,
        "history": hist.to_frame().to_dict(orient="list"),
        "best_epoch": hist.best_epoch, "best_val_sharpe": hist.best_val_sharpe,
        "extra": extra or {},
    }, path)
    return path


def load_checkpoint(name: str, device=None) -> tuple[MTLModel, Scalers, dict]:
    path = CHECKPOINTS / f"{name}.pt" if not str(name).endswith(".pt") else Path(name)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in ck["config"]["model"].items()})
    model = MTLModel(ck["f_seq"], ck["f_static"], ck["lookback"], mcfg, tuple(ck["tasks"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, Scalers.from_dict(ck["scalers"]), ck
