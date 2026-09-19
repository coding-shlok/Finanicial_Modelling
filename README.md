# nifty-mtl — Multi-Task Deep Learning with Graph Features for Risk-Adjusted Stock Selection (Nifty 50)

Weekly BUY / SELL / HOLD recommendations for Nifty 50 stocks from a multi-task
attention model that jointly predicts next-week **return** and **volatility**,
ranks stocks by a Sharpe-like score `E[ret] / E[vol]`, and is evaluated with a
walk-forward backtest that charges realistic transaction costs. Built from
[docs/PRD.md](docs/PRD.md).

## Quick start

```bash
uv venv --python 3.11 && uv pip install -r requirements.txt --python .venv/bin/python
uv pip install -e . --python .venv/bin/python
source .venv/bin/activate

nifty-mtl data                      # 1. yfinance daily -> weekly panel -> feature tensor (data/processed/*.h5)
python scripts/tune.py --trials 30  # 2. Optuna search on validation rank-IC   -> results/best_params.json
python scripts/train_final.py       # 3. static model + ablations             -> results/checkpoints/final*.pt
python scripts/run_walkforward.py   # 4. walk-forward (retrain every 13 wk)    -> results/preds/*.h5
python scripts/evaluate.py          # 5. metrics vs baselines, figures         -> results/tables, results/figures
python scripts/interpret.py         # 6. attention, SHAP, failure analysis     -> results/tables, results/figures
nifty-mtl recommend                 # 7. this week's ranked list with SHAP explanations
pytest -q                           # unit + integration tests (no network needed)
```

## What is in the box

| Path | What |
|---|---|
| `configs/universe.yaml` | Nifty 50 constituents, curated 11-sector map, corporate-action overrides |
| `configs/default.yaml` | PRD hyper-parameters |
| `src/nifty_mtl/data/` | `fetch.py` (yfinance daily, HDF5 cache), `preprocess.py` (W-FRI resample, quality screens, targets) |
| `src/nifty_mtl/features/` | 15 technical, 10 statistical, 30 graph features (`graph.py`: correlation-network PCA, peer momentum, correlation trends, centrality, regime) |
| `src/nifty_mtl/models/` | `mtl.py` (attention backbone + two heads), `losses.py` (MTL + differentiable cost term), `train.py` (early stopping on validation metric), `dataset.py` (week-batched cross-sections, train-only scalers) |
| `src/nifty_mtl/backtest/` | `engine.py` (costs, drift, cash at rf, 3 portfolio modes), `metrics.py` (Sharpe, Sortino, IR, MDD, Calmar, bootstrap CI), `baselines.py`, `walkforward.py` |
| `src/nifty_mtl/interpret/` | attention-by-lag, SHAP (GradientExplainer on the decision score), regime / sector / beta / size failure analysis |
| `scripts/` | the reproducible pipeline listed above |
| `paper/` | LaTeX (IEEE) manuscript with figures pulled from `results/figures` |
| `tests/` | look-ahead-bias test, feature NaN/normalisation, model shapes, cost-term direction, backtest P&L and cost arithmetic |

## Method in one paragraph

At every Friday close *t* each stock is a 60-week sequence of 31 own-history
features (candle geometry, technical and statistical indicators) plus 41 static
features at *t* (30 graph features from the trailing-63-day correlation network
and 11 sector one-hots). Two pre-LN self-attention layers (8 heads, d=64) and a
learned-query attention pooling turn the sequence into a vector; concatenated
with the projected static features it feeds a shared FC(128) backbone and two
heads. Targets are next-week log return (optionally cross-sectionally demeaned)
and next-week realized volatility (std of daily returns × √5, clipped to
[1%, 10%]). Loss = `w·MSE_ret + (1−w)·MSE_vol + λ·L_cost`, where `L_cost` is the
negative net return of a softmax-weighted portfolio after 0.1 %/side costs on
the change in weights from the previous week — transaction costs are part of
the training objective. The top decile by `E[ret]/E[vol]` is bought equal-weight,
the bottom decile is sold, the rest held; the walk-forward backtest retrains every
13 weeks on an expanding window.

## Design decisions that differ from the PRD text (and why)

| PRD says | Implemented | Why |
|---|---|---|
| flat 295-vector into attention | 60-token sequence + static vector | attention over a single vector is a linear layer; the sequence is what makes "which weeks matter" meaningful |
| early-stop on validation Sharpe | configurable; default rank-IC, Sharpe also run | 26-week / 5-stock Sharpe swings from −1.3 to +2.4 between epochs (see `results/figures/training_curves.png`); rank-IC uses all ~48 stocks each week |
| batch size 32 | one cross-section (~45–50 stocks) per batch | the cost term and the validation Sharpe need whole weeks |
| max position 5 % | 20 % (5 % kept as sensitivity row) | top decile of 50 = 5 names; a 5 % cap leaves 75 % in cash |
| weekly yfinance bars | daily bars resampled to W-FRI | liquidity screen (`avg daily volume > 100K`) and the realized-vol target need intraweek data |
| "include delisted stocks" (§5) vs "current constituents" (§12) | current constituents (§12) | historical membership is not available free; stated as a limitation |
| Nifty 50 *or* 100 | Nifty 50 | decision rule (Top 10 / Bottom 10 / Middle 30) only adds up for 50 |

Test period is 2023-07 → 2024-12 as in the PRD; 2025-01 → today is reported
separately as an untouched holdout that no design decision looked at.

## Results

Full tables in `results/tables/` (regenerated by `scripts/evaluate.py`); manuscript in `paper/main.pdf`.

### Headline numbers (walk-forward, retrain every 13 weeks, 3-seed average, 0.1%/side costs)

| Period | Strategy | Sharpe | CAGR | Max DD | Win rate | IR vs Nifty | Rank-IC |
|---|---|---|---|---|---|---|---|
| PRD test 2023-H2→2024 (78 wk) | MTL (ours) | 2.54 | 74.9% | -14.9% | 70.5% | 2.86 | 0.06 |
| PRD test 2023-H2→2024 (78 wk) | MTL no-graph | 2.01 | 61.5% | -8.9% | 65.4% | 2.19 | 0.04 |
| PRD test 2023-H2→2024 (78 wk) | Momentum | 0.20 | 8.7% | -22.2% | 56.4% | -0.43 | 0.02 |
| PRD test 2023-H2→2024 (78 wk) | Nifty 50 | 0.77 | 15.5% | -10.1% | 64.1% | — | — |
| Untouched holdout 2025→Sep 2026 (89 wk) | MTL (ours) | -1.12 | -12.4% | -25.9% | 43.8% | -0.88 | -0.04 |
| Untouched holdout 2025→Sep 2026 (89 wk) | MTL no-graph | -1.08 | -15.4% | -33.9% | 39.3% | -1.08 | -0.02 |
| Untouched holdout 2025→Sep 2026 (89 wk) | Momentum | -0.07 | 4.0% | -16.6% | 50.6% | 0.48 | -0.01 |
| Untouched holdout 2025→Sep 2026 (89 wk) | Nifty 50 | -0.56 | -1.6% | -13.7% | 47.2% | — | — |
| All OOS (167 wk) | MTL (ours) | 0.77 | 20.9% | -27.7% | 56.3% | 0.99 | 0.01 |
| All OOS (167 wk) | MTL no-graph | 0.44 | 14.4% | -33.9% | 51.5% | 0.61 | 0.01 |
| All OOS (167 wk) | Momentum | 0.06 | 6.2% | -27.6% | 53.3% | 0.06 | 0.00 |
| All OOS (167 wk) | Nifty 50 | 0.03 | 6.1% | -15.5% | 55.1% | — | — |

MTL Sharpe 95% block-bootstrap CI over all OOS weeks: [-0.43, 1.92] — includes zero.

**Read this honestly.** The PRD targets (Sharpe > 1.0, win rate > 52%, IR > 0.5, MDD < 15%) are all exceeded on the PRD test period, and all missed on the untouched 2025–26 holdout, where the return signal decayed to a significantly negative IC while the volatility head kept working (IC 0.18–0.38 every quarter). Ranking on E[ret] alone instead of E[ret]/E[vol] would have kept the holdout flat (Sharpe −0.11) — the Sharpe-style decision rule amplifies noise once the return forecast weakens. Ablations on the test period: return-only 1.21, no-graph 2.01, +cost-term 1.64, logistic 2.20 vs MTL 2.54. SHAP: 76% of decision-score attribution is on graph features (beta, market correlation lead); top-5 features explain 27% (PRD KPI of 60% missed). The Optuna-selected weight decay (5.7e-4) pruned the attention path to zero — the tuned model is effectively mean-pooled history + graph features. Full discussion in `paper/main.pdf` §VI.


## Reproducibility

* Data: `nifty-mtl data --force` re-downloads from Yahoo Finance (`yfinance`
  ≥ 1.7). The Tata Motors demerger (2025-10-14) is back-adjusted; see
  `configs/universe.yaml`.
* Seeds are fixed (`train.seed`); walk-forward predictions average 3 seeds.
* Every table and figure in the paper is produced by a script in `scripts/`.
