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

See `results/tables/main_results.md` (regenerated by `scripts/evaluate.py`) and
the paper in `paper/`. Summary numbers are filled in below by the final run.

<!-- RESULTS -->

## Reproducibility

* Data: `nifty-mtl data --force` re-downloads from Yahoo Finance (`yfinance`
  ≥ 1.7). The Tata Motors demerger (2025-10-14) is back-adjusted; see
  `configs/universe.yaml`.
* Seeds are fixed (`train.seed`); walk-forward predictions average 3 seeds.
* Every table and figure in the paper is produced by a script in `scripts/`.
