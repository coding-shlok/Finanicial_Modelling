"""Multi-task loss with an optional differentiable transaction-cost term.

L = w_ret * MSE(r_hat, r) + w_vol * MSE(v_hat, v)            (PRD §6, standardised targets)
    + lambda * L_port

L_port (only when lambda > 0 and a previous-week cross-section is available):
    soft long-only weights  w_t = softmax(score_t / tau)   over the cross-section
    net portfolio return    R_t = sum(w_t * r_{t+1}) - c * sum(|w_t - w_{t-1}|)
    L_port = -R_t / ret_std   (scaled so it is commensurate with the MSE terms)

This makes transaction costs part of the *training objective*, which is the
"realistic constraints built into training" contribution claimed in the PRD.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from nifty_mtl.models.mtl import denormalise, sharpe_score


def soft_weights(score: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.softmax(score / tau, dim=0)


def mtl_loss(r_hat, v_hat, y_ret, y_vol, scalers, w_ret=0.5, w_vol=0.5,
             lam=0.0, cost=0.001, tau=0.1, prev=None):
    """prev = (r_hat_prev, v_hat_prev) for the same stocks one week earlier, or None."""
    yr = (y_ret - scalers.ret_mean) / scalers.ret_std
    yv = (y_vol - scalers.vol_mean) / scalers.vol_std
    l_ret = F.mse_loss(r_hat, yr)
    l_vol = F.mse_loss(v_hat, yv)
    total = w_ret * l_ret + w_vol * l_vol
    parts = {"ret": l_ret.detach(), "vol": l_vol.detach()}

    if lam > 0 and prev is not None:
        r, v = denormalise(r_hat, v_hat, scalers)
        rp, vp = denormalise(*prev, scalers)
        w = soft_weights(sharpe_score(r, v), tau)
        w_prev = soft_weights(sharpe_score(rp, vp), tau)
        gross = (w * y_ret).sum()
        turnover = (w - w_prev).abs().sum()
        net = gross - cost * turnover
        l_port = -net / scalers.ret_std
        total = total + lam * l_port
        parts["port"] = l_port.detach()
        parts["turnover"] = turnover.detach()
    return total, parts
