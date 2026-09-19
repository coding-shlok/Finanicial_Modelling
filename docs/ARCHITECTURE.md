# Architecture notes

Deeper technical reference for `nifty-mtl`, complementing the README. Read the
README first for the narrative; this document is for someone extending or
debugging the code.

## Data flow

```
yfinance (daily OHLCV)
   │  fetch.py: HDF5 cache (data/raw/daily.h5), corporate-action back-adjustment
   ▼
preprocess.py: W-FRI resample, quality screens, targets
   │  -> WeeklyPanel  (data/processed/weekly.h5)
   ▼
features/build.py: technical + statistical + graph features
   │  -> FeatureSet: seq [T,N,31], static [T,N,41], y_ret/y_vol [T,N], sample_ok [T,N]
   │     (data/processed/features.h5)
   ▼
models/dataset.py: WeekDataset (per-fold, train-only Scalers)
   ▼
models/train.py: train_model() -> MTLModel, Scalers, TrainHistory
   │  -> checkpoints (results/checkpoints/*.pt)
   ▼
backtest/walkforward.py: 13-week expanding-window folds, 3-seed average
   │  -> Predictions (results/preds/*.h5)
   ▼
backtest/engine.py + metrics.py: portfolio simulation, Sharpe/Sortino/IR/...
   │  -> results/tables/*.csv, results/figures/*.png
   ▼
interpret/{attention,shap_analysis,failure}.py -> more tables/figures
   ▼
scripts/make_paper.py -> paper/numbers.tex (macros) -> paper/main.pdf (tectonic)
```

Every arrow is a script in `scripts/` that can be re-run independently as
long as its upstream `.h5`/`.pt`/`.csv` inputs exist.

## `FeatureSet` — the central data structure

```python
@dataclass
class FeatureSet:
    weeks: pd.DatetimeIndex        # length T
    symbols: list[str]             # length N (Nifty 50 order from universe.yaml)
    seq: np.ndarray                # [T, N, 31] float32 — own-history features
    static: np.ndarray             # [T, N, 41] float32 — graph (30) + sector one-hot (11)
    y_ret: np.ndarray              # [T, N] float32 — next-week log return (0 where unusable)
    y_vol: np.ndarray              # [T, N] float32 — next-week realized vol, clipped [0.01,0.10]
    valid: np.ndarray              # [T, N] bool — bar passes quality screens at week t
    sample_ok: np.ndarray          # [T, N] bool — full 60-week window valid AND targets exist
    lookback: int                  # 60
```

`sample_ok[t, n]` is the single source of truth for "can we train or predict
on stock `n` at week `t`." It requires:
1. `valid[t-59..t, n]` all True (60 consecutive clean weeks),
2. every sequence and static feature finite over that window (indicator
   warm-up periods are NOT imputed — they simply make the sample unusable
   until enough history exists),
3. `y_ret[t, n]` and `y_vol[t, n]` both finite (i.e. week `t+1` exists and is
   itself valid).

NaNs are zero-filled **after** `sample_ok` is computed, purely so the tensors
are finite for PyTorch — the zero-fill never affects any sample that is
actually used, because `sample_ok` already excludes it.

## Model internals (`models/mtl.py`)

- `AttentionBlock`: pre-LayerNorm, `nn.MultiheadAttention` with **attention-prob
  dropout disabled** (`dropout=0.0` inside the attention op) to keep PyTorch's
  fused scaled-dot-product-attention kernel — this alone halved per-epoch time
  in profiling (11s → 5s on an M-series CPU). Residual dropout is applied
  separately on the attention output and the FFN output.
- Positional embedding: a learned `[1, 60, d]` parameter, initialized
  `N(0, 0.02²)`, added after the linear projection of the sequence.
- Pooling: a single learned query vector attends over the 60 encoded
  positions via a 1-head `nn.MultiheadAttention`; `model.last_pool_weights`
  and `model.attention_maps()` are populated whenever `forward(...,
  need_weights=True)` is called, and are what `interpret/attention.py` reads.
- `denormalise()` maps standardized head outputs back to raw return/volatility
  units and clips volatility to `[0.01, 0.10]`.

**Attention degeneracy finding.** The Optuna-selected weight decay (5.7e-4)
shrinks `blk.attn.in_proj_weight` to a Frobenius norm of ~0.001 (vs. ~10 for
the PRD-default weight decay of 1e-5). With near-zero projection weights, the
attention logits `q @ k.T` are near-zero everywhere, so softmax returns an
exactly uniform distribution regardless of input — this is verified directly
in `scripts/interpret.py` by comparing `attention_by_lag_tuned.png` (flat) to
`attention_by_lag_prd_default.png` (structured). This is a real, reported
finding, not a bug: the hyper-parameter search, optimizing purely for
validation IC, found that suppressing the attention path generalized better
than using it.

## Loss internals (`models/losses.py`)

```python
mtl_loss(r_hat, v_hat, y_ret, y_vol, scalers, w_ret, w_vol, lam, cost, tau, prev)
```

`prev` is `(r_hat_prev, v_hat_prev)` — the model's own predictions for the
**same stocks one week earlier**, computed with `torch.no_grad()` (see
`train.py::_epoch_loss`) so the cost term does not backpropagate through the
previous week's forward pass. This was a deliberate simplification once
profiling showed the naive double-backward roughly doubled epoch time; the
soft portfolio weights `ω_t` still receive full gradient from `score_t`.

The `WeekDataset` only attaches `prev_seq`/`prev_static` to a batch when
**all** stocks in the current week's cross-section were also `sample_ok` in
the previous week — this avoids survivorship-style artifacts from a set of
stocks that changes composition abruptly.

## Walk-forward fold construction (`backtest/walkforward.py`)

```python
make_folds(fs, cfg, start="2023-07-01") -> list[Fold]
```

Fold `k` covers 13 prediction weeks `[t_k, t_k+13)`. Its validation set is
the 26 weeks immediately before `t_k`; its training set is
`[train_start, t_k - 26)` — i.e. an **expanding window**, never a sliding
one, per PRD §8. `fold_range=(a, b)` lets a single walk-forward run be sharded
across parallel processes (used during development to use all CPU cores);
`scripts/run_walkforward.py --merge` recombines the shards afterward.

Each fold trains `n_seeds` independent models (different `train.seed`) and
**averages their predicted return and volatility** before computing the
decision score — not their scores, so that outlier volatility predictions
from one seed cannot dominate multiplicatively.

## Backtest engine (`backtest/engine.py`)

State carried between weeks is `w_prev`: the portfolio weights **after**
price drift over the week, renormalized by `(1 + gross_return)`. Turnover is
computed as `|target_weights - w_prev|.sum()`, so a stock that is held with an
unchanged target weight but whose price moved still contributes some turnover
if rebalancing pulls it back toward its target — this matches how a real
weekly-rebalanced fund operates and is validated in
`tests/test_backtesting.py::test_transaction_costs`.

Three `mode` values:
- `"rebalance"` (default): hold exactly this week's BUY set, equal-weighted.
- `"hold_zone"`: keep a previously-held stock as long as it stays in the
  top half of the current ranking; only add/replace at the edges. Lower
  turnover, reported as a sensitivity.
- `"long_short"`: BUY decile long, SELL decile short, weights sum to zero
  (verified in `tests/test_backtesting.py::test_long_short_is_dollar_neutral`).

## Interpretability internals

- `interpret/attention.py::collect_attention` records, for every usable
  test-period sample: the pooling-attention vector (length 60), the
  last-layer self-attention averaged over heads and query positions (how much
  attention each key position *receives*), and the concurrent market
  20-day-volatility regime variable, for later splitting into high/low-vol
  buckets.
- `interpret/shap_analysis.py` wraps the model in `ScoreWrapper`, which
  returns `r̂/σ̂` (or `r̂` / `σ̂` alone when `target="ret"`/`"vol"`) as a single
  scalar output — required because `shap.GradientExplainer` explains one
  scalar per sample. `GradientExplainer` (expected gradients) was chosen over
  `DeepExplainer` because the latter's op-by-op backward rules don't cover
  `nn.MultiheadAttention`.
  - `ShapResult.feature_importance(mode="signed")` sums sequence attributions
    over the 60 lags **before** taking the absolute value (net effect over the
    lookback); `mode="mass"` takes absolute value **before** summing (total
    attribution mass regardless of sign cancellation). Both are reported;
    they agree that graph features dominate (75% signed, 67% mass).
- `interpret/failure.py::regime_table` reads market-wide static features (all
  stocks share the same value in a given week) via the first `sample_ok`
  stock at that week — a convenience since e.g. `mkt_vol_20d` is identical
  across the cross-section by construction.

## Paper build (`scripts/make_paper.py`)

Every number that appears in `paper/main.tex` is a LaTeX macro defined in the
auto-generated `paper/numbers.tex`, built from `results/tables/*.csv` and
`results/*.json`. Macro names are sanitized (`_clean()` strips non-alphanumerics
and spells out digits, since LaTeX macro names cannot contain them) so that,
e.g., `main_results.csv` row `("test", "MTL (ours)")` column `sharpe` becomes
`\MtlTestSharpe`. This means **the paper cannot silently drift from the
results tables** — regenerating `results/` and re-running `make_paper.py`
always produces a consistent PDF. `paper/results_prose.tex` contains the
human-written discussion, referencing these macros inline
(e.g. `\MtlTestSharpe{}`) rather than typing numbers directly.

## Known rough edges

- `models/train.py::pick_device` always returns `"cpu"` — MPS was measured
  faster for a full training epoch (~5s vs ~9s) but the improvement was
  inconsistent across model sizes during development, and CPU determinism
  was preferred for reproducibility. Override with
  `TrainConfig(device="mps")` if you want to try it.
- The `mtl_cost` and `mtl_ret_only` walk-forward ablations were only run
  through the PRD test period (not the holdout) to fit within the compute
  budget of this build; `mtl_nograph` covers all 13 folds. Re-run
  `scripts/run_walkforward.py --variants mtl_cost mtl_ret_only` with no
  `--end` flag to extend them.
- `cli/main.py::recommend` recomputes SHAP for a single week on demand
  (fast enough interactively — a few seconds) rather than caching it.
