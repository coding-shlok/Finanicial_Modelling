# AI-Driven Stock Recommendation Engine: Multi-Task Learning with Graph Features

**Project Owner:** Pratik Shah  
**Created:** 2026-09-18  
**Target Timeline:** 4 weeks  
**Status:** PRD (Ready for Development)

---

## 1. Executive Summary

We're building an **AI-driven stock recommendation engine** that uses Multi-Task Learning (MTL) with graph-aware features to predict weekly stock movements in the Indian market (Nifty 50/Top 100).

**Core idea:** Instead of predicting absolute stock prices (which is notoriously hard), we jointly predict:
1. **Expected 1-week ahead returns** for each stock
2. **Expected 1-week ahead volatility** (risk)

Then we generate actionable **buy/sell/hold signals** using a Sharpe-like ratio: `signal_score = predicted_return / predicted_volatility`, incorporating realistic constraints (transaction costs, slippage).

**Why this matters:**
- Most stock prediction models fail because they ignore risk or overfit to historical data
- Joint prediction of returns + risk creates more robust, risk-adjusted signals
- Graph features (sector relationships, peer performance) capture market dynamics better than isolated time-series models
- Realistic constraints ensure the model learns practically viable strategies

**Deliverables:**
- Trained MTL model with interpretability analysis
- Working backtesting system (with transaction costs)
- Publishable research paper (NeurIPS/ICML/IEEE)
- Codebase with experiments and results

---

## 2. Problem Statement & Motivation

### Market Inefficiency We're Exploiting

Indian equity markets (Nifty 50) show predictable patterns at weekly timescales due to:
- **Sector correlations:** Tech stocks move together, bank stocks move together
- **Momentum effects:** Stocks outperforming peers tend to continue outperforming
- **Volatility clustering:** High volatility periods are predictable
- **Market regimes:** Different strategies work in bull vs bear markets

### Why Existing Approaches Fail

1. **Pure return prediction:** Markets are semi-efficient; predicting absolute returns is hard
   - Traditional: Linear regression on technical indicators → low signal
   - Modern: LSTMs/Transformers on prices → overfitting risk

2. **Ignoring risk:** Many models predict returns but ignore volatility
   - Result: High-return stocks with extreme risk get recommended
   - Reality: Investors care about risk-adjusted returns (Sharpe ratio)

3. **Ignoring relationships:** Treating each stock as independent
   - Missing: Sector dynamics, correlation breakdowns during crises
   - Solution: Graph features capture peer relationships

4. **Unrealistic constraints:** Lab backtests don't account for:
   - Transaction costs (0.1-0.5% in India)
   - Market impact (large buys move prices)
   - Slippage (execution delay costs)

### Our Solution

**Multi-Task Learning + Graph Features:**
- Jointly optimize for return + volatility prediction (shared representation)
- Add graph features (sector PCA, peer momentum, correlation trends)
- Incorporate realistic constraints in training & backtesting
- Use attention mechanisms for interpretability

---

## 3. Objectives & Success Criteria

### Primary Objectives

1. **Build working recommendation engine**
   - Weekly buy/sell/hold signals for Nifty 50/Top 100
   - Incorporates transaction costs and realistic constraints
   - Interpretable (can explain *why* a stock is recommended)

2. **Rigorous backtesting**
   - Sharpe ratio > 1.0 in-sample (vs. 0.3-0.5 for buy-and-hold)
   - Win rate > 52% on hold-out test period
   - Profit factor > 1.3 (total wins / total losses)

3. **Publishable research**
   - Novel: Joint MTL + graph features for stock recommendation
   - Rigorous: Proper train/val/test splits, no look-ahead bias
   - Reproducible: Code, data, results all available

### Success Metrics (KPIs)

| Metric | Target | Why |
|--------|--------|-----|
| **Sharpe Ratio (weekly)** | > 1.0 | Risk-adjusted returns |
| **Win Rate (% profitable weeks)** | > 52% | Baseline is 50% random |
| **Information Ratio** | > 0.5 | Excess return vs benchmark |
| **Max Drawdown** | < 15% | Risk management |
| **Model Interpretability** | Top 5 features explain 60%+ variance | Explainability |
| **Generalization Gap** | Val Sharpe ≈ Test Sharpe (within 0.1) | Overfitting check |

### Research Novelty

- **Novel combination:** MTL (returns + volatility) + graph features (sector relationships)
- **Realistic modeling:** Transaction costs built into training, not post-hoc
- **Interpretability:** Attention-based feature importance for investor trust
- **Market-specific:** First application to Indian equities with weekly rebalancing

---

## 4. Technical Architecture

### System Overview

```
┌─────────────────────────────────────┐
│     Data Pipeline                   │
│  (Price → Features → Preprocessing) │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Feature Engineering                │
│  - Technical indicators             │
│  - Statistical features             │
│  - Graph-based features             │
│  - Normalization                    │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Multi-Task Learning Model          │
│  ┌─────────────────────────────┐   │
│  │ Shared Backbone             │   │
│  │ (Attention layers)          │   │
│  └──────┬──────────┬───────────┘   │
│         │          │               │
│      ┌──▼──┐    ┌──▼──┐           │
│      │Task1│    │Task2│           │
│      │Returns     Volatility       │
│      └──┬──┘    └──┬──┘           │
└─────────┼───────────┼──────────────┘
          │           │
          ▼           ▼
   ┌─────────────────────┐
   │ Decision Layer      │
   │ Score = R / Vol     │
   │ Buy/Sell/Hold       │
   └─────────────────────┘
          │
          ▼
   ┌─────────────────────┐
   │ Portfolio          │
   │ Optimization       │
   │ (with costs)       │
   └─────────────────────┘
          │
          ▼
   ┌─────────────────────┐
   │ Backtest Engine     │
   │ (realistic sim)     │
   └─────────────────────┘
```

### Model Architecture

**Input:** 
- Time series of OHLCV data (60 weeks of history)
- Graph features (sector embeddings, peer correlations)
- Static features (sector, market cap group)

**Shared Backbone:**
- 2 layers attention (8 heads, 64 dims)
- 1 FC layer (128 dims)
- Dropout 0.3, LayerNorm

**Task 1 - Return Prediction:**
- FC layer (64 dims)
- Output: Scalar (expected return next week)
- Loss: MSE

**Task 2 - Volatility Prediction:**
- FC layer (64 dims)
- Output: Scalar (expected volatility, clipped to [0.01, 0.1])
- Loss: MSE or MAE

**Decision Layer:**
- Score = predicted_return / predicted_volatility
- Percentile-based: Top 10 = BUY, Bottom 10 = SELL, Middle 30 = HOLD

**Training:**
- Multi-task loss: 0.5 * L_return + 0.5 * L_volatility
- Adam optimizer, learning rate 1e-3
- Early stopping on validation Sharpe ratio
- Batch size: 32, 200 epochs

---

## 5. Data Pipeline

### Data Sources

1. **Primary:** yfinance (OHLCV data)
   - Nifty 50 constituents
   - 5 years of weekly data (250+ weeks)
   - Free, reliable, covers entire history

2. **Alternative/Supplementary:** 
   - NSEpy (Indian exchange data)
   - Screener.in API (sector/fundamentals)
   - Manually curated Nifty 100 list

### Data Collection & Preprocessing

```python
1. Fetch weekly OHLCV for each stock (Friday close)
   - Resolution: Weekly (not daily) to match prediction horizon
   - Frequency: Every Friday 4 PM IST
   
2. Handle missing data:
   - Forward fill for holidays
   - Drop stocks with > 10% missing data
   
3. Compute returns & volatility:
   - Log returns: r_t = log(P_t / P_{t-1})
   - Rolling volatility: std(r_t[-4:])  # 4-week rolling std
   
4. Quality checks:
   - No NaN values
   - Liquidity check: Avg volume > 100K shares/day
   - Price check: P > 10 INR
   
5. Store in HDF5 for fast access
```

### Data Splits

- **Train:** 2019-2022 (180 weeks)
- **Validation:** 2023-Q1/Q2 (26 weeks)
- **Test:** 2023-Q3/Q4 + 2024 (52+ weeks)
- **Walk-forward:** Retrain every 13 weeks on expanding window

### Quality Assurance

- No look-ahead bias: Features at time t use only data up to t-1
- No data leakage: Test set never sees validation/train data
- Survivorship bias: Include delisted/removed stocks in backtesting
- Sector balance: Ensure all 10 sectors represented

---

## 6. Model Specification

### Architecture Details

**Input Layer:**
```
- Historical prices: 60 weeks × 4 features (OHLC) = 240 features
- Technical indicators: 15 features (RSI, MACD, Bollinger Bands, etc.)
- Graph features: 30 features (sector PCA, peer momentum, correlations)
- Sector encoding: 10 one-hot features
- Total input: 295 features
```

**Shared Backbone (Attention):**
```python
class MTLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention1 = MultiHeadAttention(d_model=64, n_heads=8)
        self.attention2 = MultiHeadAttention(d_model=64, n_heads=8)
        self.fc_shared = nn.Linear(295, 64)
        self.fc_backbone = nn.Linear(64, 128)
        self.dropout = nn.Dropout(0.3)
        self.ln = nn.LayerNorm(64)
        
        # Task heads
        self.task1_head = nn.Linear(128, 1)  # Return prediction
        self.task2_head = nn.Linear(128, 1)  # Volatility prediction
        
    def forward(self, x):
        # Shared
        x = self.fc_shared(x)
        x = self.attention1(x)
        x = self.ln(x)
        x = self.dropout(x)
        x = self.fc_backbone(x)
        x = self.dropout(x)
        
        # Task heads
        return_pred = self.task1_head(x)
        vol_pred = torch.clamp(self.task2_head(x), min=0.01, max=0.1)
        
        return return_pred, vol_pred
```

**Loss Function:**
```
L_total = 0.5 * MSE(return_pred, return_target) 
        + 0.5 * MSE(vol_pred, vol_target)
```

**Training Hyperparameters:**
- Optimizer: Adam (lr=1e-3, beta1=0.9, beta2=0.999)
- Batch size: 32
- Epochs: 200
- Early stopping: patience=20 on validation Sharpe ratio
- Gradient clipping: 1.0
- Weight decay: 1e-5

---

## 7. Feature Engineering

### Technical Indicators (15 features)

| Feature | Calculation | Why |
|---------|-------------|-----|
| RSI | 14-period relative strength | Overbought/oversold signals |
| MACD | 12-26-9 EMA | Momentum |
| Bollinger Bands | SMA ± 2*StdDev | Volatility extremes |
| ATR | Average true range | Volatility measure |
| OBV | On-balance volume | Volume momentum |
| CCI | Commodity channel index | Cyclic patterns |
| Stoch %K, %D | Stochastic oscillator | Price momentum |
| Williams %R | Range normalization | Momentum confirmation |
| ADX | Average directional index | Trend strength |

### Statistical Features (10 features)

| Feature | Calculation | Why |
|---------|-------------|-----|
| Returns | Log returns (1W, 4W, 12W) | Momentum at different scales |
| Volatility | Rolling std (4W window) | Risk measure |
| Skewness | Rolling skewness (12W) | Tail risk |
| Kurtosis | Rolling kurtosis (12W) | Extreme events |
| Autocorrelation | ACF at lag-1 | Mean reversion signal |
| Volume change | Volume MA ratio | Liquidity changes |

### Graph-Based Features (30 features)

**Sector Relationships:**
```
1. Sector PCA embedding (5 dims)
   - Compute correlation matrix for each sector
   - PCA on correlation to get sector "trend"
   
2. Peer momentum (5 dims)
   - Average return of other stocks in same sector
   - Sector momentum (10W rolling)
   
3. Correlation trends (10 dims)
   - Rolling correlation with 3 major sectors
   - Correlation volatility
   
4. Network centrality (5 dims)
   - Build correlation graph
   - Compute centrality (how correlated with rest of market)
   - Change in centrality (isolation vs integration)
   
5. Market regime (5 dims)
   - VIX-like indicator (market volatility)
   - Market trend (positive/negative)
   - Correlation regime (high/low)
```

### Feature Normalization

```python
# Per-feature normalization within train set
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

# Target normalization
y_return_scaler = StandardScaler()
y_vol_scaler = StandardScaler()
```

---

## 8. Evaluation & Backtesting

### Backtesting Methodology

**Walk-forward validation:**
```
Week 1-180 (Train) → Week 181-206 (Val) → Week 207-258 (Test)
    ↓ Retrain
Week 14-194 (Train) → Week 195-220 (Val) → Week 221-272 (Test)
    ↓ Retrain
...continues every 13 weeks
```

**Execution Model:**
```python
# Every Friday 4 PM IST
1. Generate predictions for next week
   - Top 10% (by Sharpe score) = BUY
   - Bottom 10% (by Sharpe score) = SELL
   - Middle 80% = HOLD
   
2. Transaction costs:
   - Entry cost: 0.1% (brokerage + slippage)
   - Exit cost: 0.1%
   - Total cost per round trip: 0.2%
   
3. Portfolio mechanics:
   - Equal weight buy signals
   - Rebalance weekly
   - No leverage (long-only)
   - Max position: 5% per stock
   
4. Profit/Loss calculation:
   - PnL = (exit_price - entry_price) * quantity - transaction_costs
```

### Evaluation Metrics

| Metric | Formula | Target | Interpretation |
|--------|---------|--------|-----------------|
| **Sharpe Ratio** | (mean_return - rf) / std | > 1.0 | Risk-adjusted return |
| **Win Rate** | % weeks profitable | > 52% | Consistency |
| **Information Ratio** | excess_return / tracking_error | > 0.5 | Alpha generation |
| **Profit Factor** | sum(wins) / sum(losses) | > 1.3 | Risk/reward |
| **Max Drawdown** | max cumulative loss | < 15% | Downside risk |
| **Calmar Ratio** | annual_return / max_drawdown | > 0.5 | Recovery speed |
| **Sortino Ratio** | return / downside_volatility | > 1.2 | Downside focus |

### Baseline Comparisons

1. **Buy & Hold (Nifty 50):** Sharpe ≈ 0.3-0.5
2. **Simple Momentum:** Top 10 past month performers
3. **Vol-weighted:** Inverse volatility weighting
4. **Equal Weight:** Random 10-stock portfolio
5. **Logistic Regression:** Traditional ML baseline

### Statistical Significance

- Bootstrap confidence intervals (95%) on Sharpe ratio
- Out-of-sample test on 2024 data
- Transaction cost sensitivity analysis (0.05%, 0.15%, 0.25%)

---

## 9. Interpretability & Explainability

### Attention Weights Analysis

```python
# Extract attention weights from model
attention_weights = model.attention1.get_weights()  # [batch_size, seq_len, seq_len]

# Interpretation:
# - Which past weeks are most important?
# - Do attention patterns change by market regime?
# - Can we find interpretable patterns (e.g., recent > distant)?
```

### Feature Importance

```python
# SHAP values for each prediction
import shap
explainer = shap.DeepExplainer(model, X_train_norm)
shap_values = explainer.shap_values(X_test_norm)

# Visualizations:
# 1. Mean |SHAP| for each feature (global importance)
# 2. SHAP force plot for individual predictions (why this stock?)
# 3. Feature interaction analysis
```

### Interpretable Decision Rules

```python
# Extract decision logic:
# "Stock X is recommended because:"
# 1. Return prediction: 2.5% (top 20th percentile)
# 2. Volatility prediction: 3.2% (moderate)
# 3. Sharpe score: 0.78 (above median)
# 4. Top factors: Sector momentum (SHAP: +0.4), 
#    Recent performance (SHAP: +0.3)
```

### Failure Analysis

- When does model make bad predictions?
  - High volatility regimes vs normal markets
  - Individual stock crises vs sector-wide moves
  
- Performance by stock characteristics:
  - Large-cap vs small-cap
  - High-beta vs low-beta
  - By sector

---

## 10. Implementation Timeline

### Week 1: Data Pipeline & Feature Engineering
**Goal:** Production-ready data, all features engineered

- Day 1-2: Data collection (yfinance, Nifty 50 historical)
- Day 3: Preprocessing, quality checks, HDF5 storage
- Day 4-5: Technical indicators implementation
- Day 6-7: Graph features (sector PCA, peer momentum, correlations)
- **Deliverable:** Feature matrix [N_stocks, N_weeks, N_features], no NaNs

### Week 2: Model Development & Training
**Goal:** Trained MTL model, baseline performance

- Day 1-2: Model architecture (attention backbone, task heads)
- Day 3-4: Training loop, loss functions, early stopping
- Day 5-6: Hyperparameter tuning (learning rate, dropout, batch size)
- Day 7: Model checkpointing, validation curve analysis
- **Deliverable:** Trained model, training curves, val Sharpe ratio

### Week 3: Backtesting & Evaluation
**Goal:** Rigorous backtesting, realistic results

- Day 1-2: Backtesting engine (transaction costs, slippage, walk-forward)
- Day 3-4: Generate recommendations, portfolio simulation
- Day 5-6: Compute metrics (Sharpe, win rate, drawdown)
- Day 7: Baseline comparisons, statistical significance
- **Deliverable:** Backtest results, metrics, equity curve

### Week 4: Interpretability & Paper
**Goal:** Explainability analysis + publishable research paper

- Day 1-2: Attention analysis, feature importance (SHAP)
- Day 3-4: Failure analysis, performance by stock characteristics
- Day 5-7: Write paper, create visualizations, finalize results
- **Deliverable:** Paper draft, figures, code repo

### Milestones

| Week | Milestone | Owner | Status |
|------|-----------|-------|--------|
| 1 | Data pipeline complete | Dev | ⬜ |
| 2 | Model trained, val Sharpe > 0.8 | Dev | ⬜ |
| 3 | Backtesting done, metrics computed | Dev | ⬜ |
| 4 | Interpretability + paper draft | Dev | ⬜ |

---

## 11. Risks & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| **Poor model performance (Sharpe < 0.8)** | Medium | High | Switch to simpler baseline, increase graph features |
| **Data quality issues (gaps, delisted stocks)** | Low | Medium | Build robust handling, use Yahoo Finance backup |
| **Overfitting on validation set** | Medium | High | Use walk-forward validation, test on held-out 2024 data |
| **Look-ahead bias in features** | Low | Critical | Code review, ensure all features use t-1 data only |
| **Transaction costs not realistic** | Low | Medium | Use actual NSE rates (0.08-0.15%), not assumptions |
| **Market regime change (crash)** | Low | High | Backtest on 2008, 2020 COVID periods; test robustness |
| **GPU memory constraints** | Low | Medium | Reduce batch size, use gradient checkpointing |
| **Publication rejection (not novel enough)** | Medium | Medium | Emphasize graph features + multi-task + Indian market |

### Contingency Plans

1. **If Sharpe < 0.8:** 
   - Add more graph features (technical indicators from peer stocks)
   - Try ensemble of MTL + simple momentum
   - Increase training data (use daily instead of weekly)

2. **If overfitting:** 
   - Increase dropout (0.3 → 0.5)
   - Add L2 regularization
   - Use simpler model (fewer attention heads)

3. **If rejected from conference:** 
   - Target IEEE Access (open-access journal)
   - Submit to Quantitative Finance or arXiv
   - Emphasize practical application for Indian markets

---

## 12. Constraints & Assumptions

### Computational Budget
- GPU: 1x GPU (V100 or better, 8GB VRAM sufficient)
- Storage: ~1GB for 5 years of historical data
- Runtime: Training < 2 hours, backtesting < 30 min

### Market Assumptions
1. **Liquidity:** Nifty 50 stocks have sufficient liquidity
2. **No leverage:** Long-only portfolio (can't short)
3. **Transaction costs:** 0.1% entry, 0.1% exit (realistic for retail)
4. **Market efficiency:** Weak form (technical patterns exist short-term)

### Data Assumptions
1. **No survivorship bias handling:** Focus on current Nifty 50 index
2. **Weekly data:** Friday close prices
3. **No corporate actions:** Assume splits/dividends already adjusted
4. **No delisted stocks:** Drop from analysis (worst case assumption)

### Model Assumptions
1. **Past predicts future:** Historical patterns repeat
2. **Stationarity:** Market regime doesn't change drastically
3. **Linear feature relationships:** After normalization
4. **Independence:** Stocks follow similar dynamics

---

## 13. Testing Strategy

### Unit Tests
```python
# tests/test_features.py
def test_feature_no_nan():
    X = generate_features(data)
    assert not X.isnull().any().any()

def test_feature_normalization():
    X_train, X_test = split_data()
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    assert abs(X_train_norm.mean()) < 1e-10
    assert abs(X_train_norm.std() - 1) < 1e-10

# tests/test_model.py
def test_model_output_shape():
    model = MTLModel()
    X = torch.randn(32, 295)
    returns, vols = model(X)
    assert returns.shape == (32, 1)
    assert vols.shape == (32, 1)

def test_no_look_ahead_bias():
    # Ensure features at week t use only data up to t-1
    X_week_5 = features_at_week(5)
    price_week_5 = prices_at_week(5)
    assert not uses_future_data(X_week_5, price_week_5)
```

### Integration Tests
```python
# tests/test_backtesting.py
def test_backtest_pnl_calculation():
    # Generate mock predictions
    predictions = generate_mock_predictions()
    # Run backtest
    pnl, equity_curve = backtest(predictions, prices)
    # Verify calculations
    assert len(pnl) == len(predictions)
    assert equity_curve[0] == initial_capital

def test_transaction_costs():
    # Buy at 100, sell at 102
    entry_cost = 100 * 0.001  # 0.1%
    exit_cost = 102 * 0.001   # 0.1%
    gross_pnl = (102 - 100) * 1
    net_pnl = gross_pnl - entry_cost - exit_cost
    assert net_pnl == approx(1.798)
```

### Validation Checklist
- [ ] No NaN in features or targets
- [ ] No look-ahead bias (all features use past data only)
- [ ] No data leakage (train/val/test splits honored)
- [ ] Transaction costs properly modeled
- [ ] Backtest equity curve is monotonic logical (no jumps)
- [ ] Metrics consistent across different periods
- [ ] Model generalizes to 2024 out-of-sample data

---

## 14. Publication & Research

### Target Venues

1. **Primary (ML conferences):**
   - NeurIPS 2024 (deadline: June 2024) ❌ Past
   - ICML 2024 (deadline: February 2024) ❌ Past
   - IEEE AI 2024/2025 ✅ Target

2. **Secondary (Finance/Applied):**
   - Quantitative Finance (journal)
   - IEEE Transactions on Neural Networks and Learning Systems
   - arXiv (preprint, always available)

### Paper Outline

**Title:** "Multi-Task Deep Learning with Graph Features for Risk-Adjusted Stock Selection in Emerging Markets"

**Sections:**
1. **Introduction** (2 pages)
   - Stock prediction is hard (EMH)
   - Our insight: Joint modeling of returns + risk
   - Graph features capture sector dynamics
   
2. **Related Work** (1.5 pages)
   - Stock prediction models (technical, fundamental, ML)
   - Multi-task learning applications
   - Graph neural networks in finance
   
3. **Methodology** (2 pages)
   - Problem formulation
   - MTL architecture
   - Graph feature construction
   - Training objective
   
4. **Experiments** (2 pages)
   - Dataset description (Nifty 50, 5 years)
   - Baselines
   - Results (Sharpe, win rate, etc.)
   - Ablation studies (MTL vs single-task, with/without graph features)
   
5. **Interpretability Analysis** (1 page)
   - Attention weights
   - SHAP-based feature importance
   - Failure modes
   
6. **Discussion & Limitations** (1 page)
   - Why does it work?
   - When does it fail?
   - Future work
   
7. **Conclusion** (0.5 pages)

**Total:** ~10 pages

### Reproducibility & Code

- GitHub repo: Clean, documented codebase
- Data: Scripts to fetch historical data (publicly available)
- Models: Checkpoints saved, loading instructions
- Notebooks: Jupyter notebooks for key experiments
- README: Clear instructions to reproduce all results

### Novel Contributions

1. **Joint MTL for stock selection:** Returns + volatility prediction
2. **Graph features:** Sector relationships, peer momentum, correlation dynamics
3. **Realistic constraints:** Transaction costs built into training
4. **Emerging market focus:** Application to Indian equities (Nifty 50)
5. **Interpretability:** Attention + SHAP analysis for trust

---

## Appendix: Reference & Resources

### Public Data Sources
- **yfinance:** `pip install yfinance` (free OHLCV)
- **NSEpy:** Indian exchange API
- **Screener.in:** Fundamental data

### Libraries
- PyTorch, NumPy, Pandas (core ML)
- Scikit-learn (preprocessing, baselines)
- SHAP (interpretability)
- Backtrader or custom backtester

### Hyperparameter Defaults
- Learning rate: 1e-3
- Dropout: 0.3
- Attention heads: 8
- Batch size: 32
- Epochs: 200
- Look-back: 60 weeks

### Key Formulas

**Sharpe Ratio (weekly):**
```
Sharpe = (mean_weekly_return - rf) / std_weekly_return
where rf ≈ 6.5% annual ≈ 0.125% weekly
```

**Win Rate:**
```
Win% = (# weeks with positive PnL) / total_weeks × 100
```

**Information Ratio:**
```
IR = (portfolio_return - benchmark_return) / tracking_error
```

