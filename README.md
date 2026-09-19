# nifty-mtl

**Multi-Task Deep Learning with Graph Features for Risk-Adjusted Stock Selection (Nifty 50)**

Weekly BUY / SELL / HOLD recommendations for Nifty 50 stocks from a multi-task
attention model that jointly predicts next-week **return** and **volatility**,
ranks stocks by a Sharpe-like score `E[ret] / E[vol]`, and is evaluated with a
walk-forward backtest that charges realistic transaction costs. Built end-to-end
— data pipeline, model, backtester, interpretability study, and an IEEE-format
research paper — from [docs/PRD.md](docs/PRD.md).

**Read this before the results table:** the system beats every baseline on the
period the PRD specified as its test set, and loses money on an untouched
2025–2026 holdout that no design decision ever looked at. Both numbers are
reported. See [§ Results](#results) and `paper/main.pdf` for the honest account.

---

## Table of contents

- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Method](#method)
  - [Data pipeline](#1-data-pipeline)
  - [Feature engineering](#2-feature-engineering)
  - [Model](#3-model)
  - [Training objective](#4-training-objective)
  - [Model selection](#5-model-selection)
  - [Portfolio construction & backtest](#6-portfolio-construction--backtest)
- [Results](#results)
- [Ablations](#ablations)
- [Interpretability](#interpretability)
- [Design decisions that differ from the PRD](#design-decisions-that-differ-from-the-prd-and-why)
- [Limitations](#limitations)
- [Reproducing everything](#reproducing-everything)
- [Testing](#testing)
- [CLI reference](#cli-reference)

---

## Quick start

```bash
git clone https://github.com/coding-shlok/Finanicial_Modelling.git
cd Finanicial_Modelling
uv venv --python 3.11
uv pip install -r requirements.txt --python .venv/bin/python
uv pip install -e . --python .venv/bin/python
source .venv/bin/activate

nifty-mtl data                      # 1. yfinance daily -> weekly panel -> feature tensor
python scripts/tune.py --trials 30  # 2. Optuna search on validation rank-IC
python scripts/train_final.py       # 3. static model + ablations
python scripts/run_walkforward.py   # 4. walk-forward backtest (retrain every 13 weeks)
python scripts/evaluate.py          # 5. metrics vs baselines, figures
python scripts/interpret.py         # 6. attention, SHAP, failure analysis
python scripts/make_paper.py        # 7. compile paper/main.pdf (needs `tectonic`)
nifty-mtl recommend                 # 8. this week's ranked list with SHAP explanations
pytest -q                           # unit + integration tests (no network needed)
```

Each script reads the previous stage's output from `data/` or `results/` and
writes its own; you can re-run any single stage without repeating the others.
Total pipeline time on a 2024-era Apple Silicon Mac (CPU, no GPU needed): data
fetch ~1 min, tuning ~1–2 hrs (30 trials, parallelizable), final training
~10 min, walk-forward backtest ~2–3 hrs (13 folds × 3 seeds), evaluation and
interpretability ~2 min each.

---

## Repository layout

```
nifty-mtl/
├── configs/
│   ├── universe.yaml          # Nifty 50 constituents, sector map, corporate actions
│   └── default.yaml           # hyper-parameters (PRD defaults; overridden by tuning)
├── docs/
│   └── PRD.md                 # the original product requirements document
├── src/nifty_mtl/
│   ├── config.py              # paths, dataclasses for Universe / Splits / Model / Train / Backtest config
│   ├── data/
│   │   ├── fetch.py           # yfinance daily download, HDF5 cache, corporate-action back-adjustment
│   │   └── preprocess.py      # daily -> weekly (W-FRI) resample, quality screens, prediction targets
│   ├── features/
│   │   ├── technical.py       # 15 technical indicators (RSI, MACD, Bollinger, ATR, ADX, ...)
│   │   ├── statistical.py     # 10 statistical features (multi-horizon returns, vol, skew, kurtosis, ACF)
│   │   ├── graph.py           # 30 graph features from a weekly-reestimated correlation network
│   │   └── build.py           # assembles the [T, N, F] tensors and usability masks
│   ├── models/
│   │   ├── mtl.py             # attention backbone + return/volatility heads
│   │   ├── losses.py          # MTL loss + differentiable transaction-cost term
│   │   ├── dataset.py         # week-batched cross-sections, train-only feature/target scalers
│   │   └── train.py           # training loop, early stopping, prediction, checkpointing
│   ├── backtest/
│   │   ├── engine.py          # portfolio simulator: costs, weight drift, 3 rebalancing modes
│   │   ├── metrics.py         # Sharpe, Sortino, IR, MDD, Calmar, win rate, profit factor, bootstrap CI
│   │   ├── baselines.py       # momentum, inverse-vol, random, logistic-regression baselines
│   │   └── walkforward.py     # expanding-window walk-forward fold construction and orchestration
│   ├── interpret/
│   │   ├── attention.py       # pooling/self-attention weight analysis by lag and regime
│   │   ├── shap_analysis.py   # SHAP (GradientExplainer) on the decision score, by feature and group
│   │   └── failure.py         # signal quality sliced by market regime, sector, beta, size
│   ├── plotting.py            # shared matplotlib style (fixed categorical palette, dataviz-skill compliant)
│   └── cli/main.py            # `nifty-mtl {data, train, recommend, backtest}`
├── scripts/                    # the reproducible pipeline (tune, train_final, run_walkforward,
│                                #  evaluate, interpret, make_paper) — each is a thin driver over src/
├── paper/
│   ├── main.tex                # IEEE-format manuscript
│   ├── results_prose.tex       # results/discussion text, written against the auto-generated macros
│   └── numbers.tex             # AUTO-GENERATED by scripts/make_paper.py — every number in the paper
│                                #  comes from results/tables/*.csv, never hand-typed
├── notebooks/
│   └── experiments.ipynb       # reproduces every table/figure inline from results/
├── tests/                      # look-ahead-bias test, feature/model/backtest unit tests (21 tests, no network)
├── data/                       # gitignored: raw daily (HDF5), processed weekly panel, feature tensor
└── results/                    # gitignored (except tables/figures): checkpoints, predictions, logs
    ├── tables/                  # CSV + Markdown tables (tracked — the numeric record of every run)
    └── figures/                 # PNG + PDF figures (tracked)
```

---

## Method

### 1. Data pipeline

**Source.** Daily OHLCV for the 50 current Nifty 50 constituents plus the
index (`^NSEI`) from Yahoo Finance (`yfinance`), January 2017 to the present.
Daily bars are fetched (not weekly) because the liquidity screen and the
realized-volatility target both need intraweek observations.

**Corporate actions.** `yfinance`'s split/dividend adjustment does not cover
demergers. The October 2025 Tata Motors demerger (into TMPV/TMCV) is
back-adjusted explicitly in `configs/universe.yaml` and applied by
`data/fetch.py`: prices before the ex-date are scaled by
`open(ex_date) / close(ex_date − 1)` so the corporate action does not appear
as a fake return.

**Weekly resampling.** Daily bars are resampled to Friday closes (`W-FRI`).
For each week: `open` = first, `high`/`low` = max/min, `close` = last,
`volume` = sum. Weekly log return `r_t = log(C_t / C_{t-1})`. Realized weekly
volatility `σ_t = std(daily log returns in week t) × √5`.

**Quality screens** (applied per stock per week, PRD §5): price > ₹10,
trailing-year average daily volume > 100K shares, less than 10% missing
sessions in the trailing year. A stock only enters the training/prediction set
once it passes all three; recently-listed names (Eternal, Jio Financial, Max
Healthcare) phase in once they qualify.

**Targets.** At decision week `t`, the model predicts `r_{t+1}` (next week's
log return) and `σ_{t+1}` (next week's realized volatility, clipped to
`[0.01, 0.10]` per the PRD). Both targets use only information realized in
week `t+1` — never available at decision time, and never used to build the
feature at `t`.

**No look-ahead, verified by test.** `tests/test_features.py::test_no_look_ahead_bias`
perturbs every price strictly after week `t` and asserts that every feature
and target at `≤ t` is bit-identical before and after — the strongest
practical guarantee against leakage that a unit test can offer.

### 2. Feature engineering

Each stock at each week carries **31 own-history ("sequence") features**
replicated over a 60-week lookback, and **41 static features** computed once
at the decision week:

| Group | Count | Examples |
|---|---|---|
| Candle geometry | 6 | log O/H/L/C relative to previous close, log volume ratio, realized vol |
| Technical indicators | 15 | RSI-14, MACD (line/signal/histogram), Bollinger %B/width, ATR, OBV, CCI, stochastic %K/%D, Williams %R, ADX, ±DI |
| Statistical | 10 | 1/4/12/26-week returns, 4/12-week volatility, 12-week skew/kurtosis, lag-1 autocorrelation, volume ratio |
| **Graph — PCA embedding** | 5 | loadings on the top-5 eigenvectors of the 63-day correlation matrix (sign-aligned across weeks) |
| **Graph — peer momentum** | 5 | sector-mean return excluding self (1/4/12w), relative strength vs. sector, 10w sector momentum |
| **Graph — correlation trends** | 10 | correlation with Banks/IT/Energy indices and the market, correlation change and volatility, intra-/extra-sector correlation, market and sector beta |
| **Graph — network centrality** | 5 | eigenvector and degree centrality of the correlation graph (`\|ρ\|>0.3`), 13-week centrality change, clustering coefficient, sector-average centrality |
| **Graph — market regime** | 5 | 20-day market volatility, 12-week market return, deviation from 26-week MA, average pairwise correlation, cross-sectional dispersion |
| Sector one-hot | 11 | Banks, Financial Services, IT, Energy, FMCG, Auto, Pharma & Healthcare, Metals & Mining, Infra & Cement, Telecom & Media, Consumer |

The **graph** is a weekly-reestimated stock–stock correlation network (63
trading days of daily log returns). All graph statistics are recomputed from
scratch every week — this is what lets the model see sector rotation,
correlation breakdowns, and network centrality shifts as they happen, rather
than as fixed sector labels.

Features are standardized (`StandardScaler`-equivalent, fit **only** on the
training split of each fold — never on validation or test data). Sector
one-hot columns are never rescaled.

### 3. Model

A 60-token sequence (one token per week of history) is embedded, given a
learned positional embedding, and passed through **two pre-LayerNorm
self-attention blocks** (8 heads, `d_model=64`, GELU feed-forward). A
**learned-query attention pooling** layer reduces the sequence to a single
vector — its attention weights are what the interpretability study reads to
answer "which past weeks matter." The pooled vector is concatenated with the
projected 41-dim static vector and fed through a shared `FC(128)` backbone
into two heads: `FC(64)→1` for return, `FC(64)→1` for volatility (clipped to
`[0.01, 0.10]` after de-standardization).

```
seq [60×31] --Linear+pos--> [60×d] --2×(MHA+FFN)--> [60×d] --attn-pool--> h_s [d]
static [41] --Linear+GELU--> h_z [d]
[h_s;h_z] --FC(128)+ReLU+Drop--> h [128] --> FC(64)→1  (return head)
                                        --> FC(64)→1  (volatility head)
```

~125K parameters. Trains on CPU in a few minutes per fold (no GPU required).

### 4. Training objective

```
L = w · MSE(r̂, r) + (1−w) · MSE(σ̂, σ) + λ · L_cost
```

`L_cost` is the negative net return of a **softmax-implied portfolio**:
`ω_t = softmax(score_t / τ)` over the weekly cross-section, with turnover
costs `c · Σ|ω_t − ω_{t-1}|` charged against the previous week's implied
weights (a second, no-grad forward pass). This is what lets the paper claim
"transaction costs are part of the training objective" rather than a
post-hoc backtest adjustment — though the ablation study finds it did not
transfer well to the hard top-decile portfolio actually traded (see
[Ablations](#ablations)).

Optimizer: Adam, gradient clipping at 1.0, up to 200 epochs, early stopping
with patience 20. Batches are **whole weekly cross-sections** (~45–50 stocks),
not the PRD's fixed batch size of 32, because the cost term and validation
Sharpe both need a complete week.

### 5. Model selection

Hyper-parameters (dropout, learning rate, weight decay, task weight `w`,
`λ`, and whether the return target is raw or cross-sectionally demeaned) were
tuned with **24 Optuna TPE trials** on validation-period rank-IC. Architecture
(`d=64`, 2 layers) was fixed at the PRD's spec.

**We deliberately deviated from the PRD's early-stopping criterion.** The PRD
specifies stopping on validation Sharpe ratio. With a 26-week validation
window and a 5-stock portfolio, that statistic swings from −1.3 to +2.4
between adjacent epochs while the loss moves smoothly (see
`results/figures/training_curves.png`) — it is dominated by noise. We select
on mean weekly rank-IC instead (uses all ~48 stocks per week) and report the
Sharpe-selected model as an ablation.

### 6. Portfolio construction & backtest

Every Friday close, stocks are ranked by `score = r̂ / σ̂`. Top decile = BUY
(equal-weight, capped at 20% each — five picks fully invested; PRD's 5% cap
reported as a sensitivity), bottom decile = SELL if held, middle 80% = HOLD.
Costs of 0.1% are charged **per side** on the change in portfolio weights
(including price drift during the week); idle capital earns the weekly
risk-free rate (6.5% annual). Two secondary modes are also implemented: a
lower-turnover **hold-zone** variant, and a dollar-neutral **long–short**
variant.

**Walk-forward evaluation:** prediction blocks of 13 weeks starting July 2023.
For each block, the validation set is the preceding 26 weeks and the training
set is everything from January 2019 up to that point (expanding window, per
PRD §8). Predictions are averaged over 3 random seeds per fold. Results are
split into the PRD's test period (2023-H2–2024) and an **untouched holdout**
(2025–September 2026) that no tuning, architecture, or model-selection
decision ever examined.

---

## Results

Full tables in [`results/tables/`](results/tables/) (regenerated by
`scripts/evaluate.py`); full discussion in [`paper/main.pdf`](paper/main.pdf).

### Headline numbers (walk-forward, retrain every 13 weeks, 3-seed average, 0.1%/side costs)

| Period | Strategy | Sharpe | CAGR | Max DD | Win rate | IR vs Nifty | Rank-IC |
|---|---|---|---|---|---|---|---|
| **PRD test 2023-H2→2024** (78 wk) | **MTL (ours)** | **2.54** | **74.9%** | −14.9% | **70.5%** | **2.86** | **0.064** |
| PRD test 2023-H2→2024 (78 wk) | MTL no-graph | 2.01 | 61.5% | −8.9% | 65.4% | 2.19 | 0.037 |
| PRD test 2023-H2→2024 (78 wk) | Momentum | 0.20 | 8.7% | −22.2% | 56.4% | −0.43 | 0.018 |
| PRD test 2023-H2→2024 (78 wk) | Nifty 50 | 0.77 | 15.5% | −10.1% | 64.1% | — | — |
| **Untouched holdout 2025→Sep 2026** (89 wk) | **MTL (ours)** | **−1.12** | **−12.4%** | −25.9% | 43.8% | −0.88 | **−0.041** |
| Untouched holdout 2025→Sep 2026 (89 wk) | MTL no-graph | −1.08 | −15.4% | −33.9% | 39.3% | −1.08 | −0.023 |
| Untouched holdout 2025→Sep 2026 (89 wk) | Momentum | −0.07 | 4.0% | −16.6% | 50.6% | 0.48 | −0.009 |
| Untouched holdout 2025→Sep 2026 (89 wk) | Nifty 50 | −0.56 | −1.6% | −13.7% | 47.2% | — | — |
| All out-of-sample (167 wk) | MTL (ours) | 0.77 | 20.9% | −27.7% | 56.3% | 0.99 | 0.008 |
| All out-of-sample (167 wk) | MTL no-graph | 0.44 | 14.4% | −33.9% | 51.5% | 0.61 | 0.005 |
| All out-of-sample (167 wk) | Momentum | 0.06 | 6.2% | −27.6% | 53.3% | 0.06 | 0.004 |
| All out-of-sample (167 wk) | Nifty 50 | 0.03 | 6.1% | −15.5% | 55.1% | — | — |

MTL Sharpe 95% stationary-block-bootstrap CI over all out-of-sample weeks:
**[−0.43, 1.92]** — includes zero. 167 weeks of a 5-stock portfolio is not
enough to establish statistical significance on its own.

### Read this honestly

The PRD success criteria (Sharpe > 1.0, win rate > 52%, information ratio >
0.5, max drawdown < 15%) are **all exceeded** on the PRD's own test period,
and **all missed** on the untouched holdout. The return signal's quarterly
rank-IC was 0.10–0.15 throughout the 2023–2024 rally (the model's picks
carried roughly twice the market's 12-week momentum) and decayed to zero or
negative from 2024-Q3 onward, while the volatility head's IC stayed strong
(0.18–0.38) in **every** quarter including the holdout. The decision layer
made the failure worse: dividing a near-zero return forecast by predicted
volatility manufactures a systematic, significantly-negative tilt; ranking on
`E[ret]` alone instead would have kept the holdout roughly flat (Sharpe
−0.11, IC +0.014) rather than −1.12. This is reported as a post-hoc
diagnostic, not as a design change — we did not re-tune anything after seeing
the holdout.

---

## Ablations

On the PRD test period (78 weeks, net of costs):

| Variant | Sharpe | Rank-IC | Turnover | What it isolates |
|---|---|---|---|---|
| **MTL (ours)** | **2.54** | **0.064** | 0.36 | full model |
| MTL, no graph features | 2.01 | 0.037 | 0.13 | value of the 30 graph features |
| MTL, return-only (no volatility task) | 1.21 | 0.017 | 0.92 | value of the auxiliary volatility task |
| MTL + cost-term (λ=0.3) | 1.64 | 0.031 | 0.69 | value of the differentiable cost penalty (did not help) |
| Logistic regression (same features) | 2.20 | 0.026 | 0.86 | non-deep baseline on identical inputs |

The volatility task is the single biggest lever: dropping it more than doubles
turnover and nearly halves the Sharpe — the auxiliary task regularizes the
shared representation. The transaction-cost loss term, evaluated in the
direction Optuna rejected it (`λ>0` on the tuned config), did not help: the
soft, temperature-scaled portfolio it optimizes is a weak proxy for the hard
top-decile portfolio that is actually traded.

---

## Interpretability

- **Attention.** In the tuned model (weight decay 5.7e-4, ~57× the PRD
  default), the attention projection weights shrink to norm ≈0.001 and the
  learned attention becomes **exactly uniform** — the sequence encoder
  degenerates to a mean-pool over 60 weeks. The PRD-default configuration
  (weight decay 1e-5) retains real structure: the last 3–4 weeks and the
  oldest positions receive above-uniform attention, and pooling tilts toward
  recent weeks in high-volatility regimes. Both are reported
  (`results/figures/attention_by_lag_tuned.png` /
  `_prd_default.png`) — an honest negative result about what the tuner found.
- **SHAP** (GradientExplainer on the decision score, 600 test-period samples):
  **76% of attribution mass is on graph features**, led by market beta and
  market correlation — the model learned to buy high-beta, market-sensitive
  names, which explains both why it won the 2023–2024 rally and why it lost
  the flat 2025–2026 market. The top 5 features explain only 27% of
  attribution (the PRD's 60% target is not met — importance is genuinely
  spread across the graph feature groups).
- **Failure analysis** by regime: rank-IC is higher when pairwise
  correlations are low (+0.039 vs. −0.024 when high) and in rising markets
  (+0.028 vs. −0.012 in falling ones). In the 19 weeks the index fell >2%,
  the model's cross-sectional ranking was actually *good* (IC +0.091) but the
  long-only portfolio still lost heavily (−2.5%/week) — it ranks correctly
  inside a crash but cannot avoid being long during one.

Full detail, tables, and figures: `paper/main.pdf` §V.

---

## Design decisions that differ from the PRD (and why)

| PRD says | Implemented | Why |
|---|---|---|
| flat 295-dim vector fed to attention | 60-token sequence + separate static vector | attention over a single vector is a linear layer; a real sequence is what makes "which weeks matter" a meaningful question |
| early-stop on validation Sharpe | early-stop on validation rank-IC (Sharpe stopping also run, reported as ablation) | 26-week / 5-stock Sharpe swings from −1.3 to +2.4 between epochs; IC uses all ~48 stocks per week and is far more stable |
| batch size 32 | one weekly cross-section (~45–50 stocks) per batch | the cost term and portfolio-level validation metrics need a whole week at once |
| max position 5% | 20% (5% kept as a sensitivity row) | the top decile of 50 stocks is 5 names — a 5% cap would leave 75% of capital in cash |
| weekly bars fetched directly | daily bars fetched and resampled to `W-FRI` | the liquidity screen (avg. *daily* volume) and the realized-volatility target both need intraweek data |
| "include delisted stocks" (§5) vs. "current constituents only" (§12) — PRD is internally inconsistent | current constituents only | historical index membership is not available from a free data source; documented as a limitation (survivorship bias) |
| Nifty 50 *or* Nifty 100 (PRD is ambiguous) | Nifty 50 | the PRD's own decision rule (Top 10 / Bottom 10 / Middle 30) only totals to 50 |

---

## Limitations

1. **Survivorship bias** — the universe is *today's* Nifty 50; stocks removed
   after poor performance are absent from both training and the benchmark
   comparison built on the same universe (though not from the actual Nifty
   index return itself).
2. **Small universe** — 50 names means a 5-stock top decile; portfolio
   statistics are noisy, as the wide bootstrap confidence interval shows.
3. **0.1%/side costs** are realistic for a retail discount broker on large
   caps but exclude market impact, which would matter at any real scale.
4. **Regime coverage** — training starts in 2019 and contains one crash
   (2020); the PRD's proposed 2008 stress test is not possible since most
   constituents were not listed then.
5. **Sector labels are hand-curated**, not sourced from an authoritative
   classification.
6. **The test period is under four years** — a Sharpe ratio over a span this
   short can be positive by chance, which the confidence intervals make
   explicit rather than hiding.

---

## Reproducing everything

```bash
nifty-mtl data --force               # re-download from Yahoo Finance
pytest -q                            # 21 tests: look-ahead bias, feature/model/backtest correctness
python scripts/tune.py --trials 30 --study my_run
python scripts/train_final.py
python scripts/run_walkforward.py --seeds 3
python scripts/evaluate.py
python scripts/interpret.py
python scripts/make_paper.py         # requires `tectonic` (brew install tectonic)
```

Every number in `paper/main.pdf` is produced by `scripts/make_paper.py` from
`results/tables/*.csv` via LaTeX macros in the auto-generated
`paper/numbers.tex` — nothing in the paper is hand-typed. Seeds are fixed
(`train.seed`, default 42); walk-forward predictions average 3 seeds per fold.

---

## Testing

21 tests, no network access required (a synthetic 12-stock panel is used for
fixtures):

- `tests/test_features.py` — weekly return/target definitions, no-NaN
  invariant on usable samples, feature-count and normalization checks, and
  the **look-ahead-bias test** (perturb the future, assert the past is
  unchanged).
- `tests/test_model.py` — model output shapes, volatility clipping, attention
  weights sum to 1, the cost-term loss penalizes turnover in the correct
  direction, scalers are fit only on training data.
- `tests/test_backtesting.py` — decile signal counts, P&L arithmetic against
  a hand-computed example, exact transaction-cost accounting, position-cap
  cash handling, long-short dollar neutrality, metric sanity checks.

```bash
pytest -q
```

---

## CLI reference

```bash
nifty-mtl data [--force]                          # build the data pipeline
nifty-mtl train [--name NAME] [--epochs N]         # train the static model
nifty-mtl recommend [--week YYYY-MM-DD] [--top N]  # ranked BUY/SELL/HOLD list with SHAP explanations
nifty-mtl backtest --preds results/preds/mtl.h5    # backtest a saved prediction panel
```

`nifty-mtl recommend` prints a table with predicted return, predicted
volatility, decision-score percentile, and the top-3 SHAP factors behind each
recommendation — e.g. *"BUY, r̂=+1.2%, σ̂=2.6%, score 0.46 (96th pct); factors:
corr_mkt +0.21, beta_mkt +0.18, sector_IT −0.09."*

---

## Further technical detail

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — data structures, model
internals, loss internals, fold construction, and known rough edges, for
anyone extending this code.

---

## Acknowledgment

Built from [`docs/PRD.md`](docs/PRD.md) using Claude Opus 5 / Sonnet 5
(Claude Code).
