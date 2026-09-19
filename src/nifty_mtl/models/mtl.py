"""Multi-task attention model (PRD §4/§6).

Architecture
------------
seq  [B, L=60, F_seq] --Linear--> [B, L, d]  (+ learned positional embedding)
     --> n_layers x (pre-LN multi-head self-attention + FFN)
     --> attention pooling with a learned query  -> [B, d]   (pool weights = "which weeks matter")
static [B, F_static]   --Linear--> [B, d]
concat [B, 2d] --> FC(d_hidden) + ReLU + Dropout   (shared backbone)
      --> head_ret: FC(d_head) -> ReLU -> 1
      --> head_vol: FC(d_head) -> ReLU -> 1

Both heads predict in *standardised* target space; ``denormalise`` maps back and
clips volatility to [0.01, 0.10] (PRD).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nifty_mtl.config import ModelConfig


class AttentionBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        # attention-prob dropout disabled: keeps the fused SDPA kernel; dropout is applied on residual + FFN paths
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=0.0, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, need_weights: bool = False) -> torch.Tensor:
        h = self.ln1(x)
        a, w = self.attn(h, h, h, need_weights=need_weights, average_attn_weights=False)
        if need_weights:
            self.last_attn = w.detach()
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class MTLModel(nn.Module):
    def __init__(self, f_seq: int, f_static: int, lookback: int, cfg: ModelConfig | None = None,
                 tasks: tuple[str, ...] = ("ret", "vol")):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        self.tasks = tasks
        d = cfg.d_model
        self.seq_proj = nn.Linear(f_seq, d)
        self.pos = nn.Parameter(torch.zeros(1, lookback, d))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([AttentionBlock(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)])
        self.ln_seq = nn.LayerNorm(d)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, 1, dropout=0.0, batch_first=True)
        self.static_proj = nn.Sequential(nn.Linear(f_static, d), nn.GELU())
        self.backbone = nn.Sequential(nn.Linear(2 * d, cfg.d_hidden), nn.ReLU(), nn.Dropout(cfg.dropout))
        self.head_ret = nn.Sequential(nn.Linear(cfg.d_hidden, cfg.d_head), nn.ReLU(), nn.Linear(cfg.d_head, 1))
        self.head_vol = nn.Sequential(nn.Linear(cfg.d_hidden, cfg.d_head), nn.ReLU(), nn.Linear(cfg.d_head, 1))
        self.last_pool_weights: torch.Tensor | None = None

    def encode(self, seq: torch.Tensor, static: torch.Tensor, need_weights: bool = False) -> torch.Tensor:
        x = self.seq_proj(seq) + self.pos[:, -seq.shape[1]:]
        for blk in self.blocks:
            x = blk(x, need_weights=need_weights)
        x = self.ln_seq(x)
        q = self.pool_query.expand(x.shape[0], -1, -1)
        pooled, pw = self.pool(q, x, x, need_weights=True)           # pw: [B, 1, L]
        self.last_pool_weights = pw.detach().squeeze(1)
        z = torch.cat([pooled.squeeze(1), self.static_proj(static)], dim=-1)
        return self.backbone(z)

    def forward(self, seq: torch.Tensor, static: torch.Tensor, need_weights: bool = False):
        h = self.encode(seq, static, need_weights)
        r = self.head_ret(h).squeeze(-1)
        v = self.head_vol(h).squeeze(-1)
        return r, v

    def attention_maps(self) -> list[torch.Tensor]:
        return [b.last_attn for b in self.blocks if b.last_attn is not None]


def denormalise(r_norm: torch.Tensor, v_norm: torch.Tensor, scalers, vol_clip=(0.01, 0.10)):
    r = r_norm * scalers.ret_std + scalers.ret_mean
    v = (v_norm * scalers.vol_std + scalers.vol_mean).clamp(*vol_clip)
    return r, v


def sharpe_score(r: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Decision layer: predicted return / predicted volatility (PRD §4)."""
    return r / v
