"""Project-wide configuration: paths, universe, split dates, hyperparameters."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
CHECKPOINTS = RESULTS / "checkpoints"

for _p in (DATA_RAW, DATA_PROCESSED, FIGURES, TABLES, CHECKPOINTS):
    _p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Universe:
    symbols: list[str]                 # canonical names, e.g. "TATAMOTORS"
    tickers: dict[str, str]            # canonical -> yfinance ticker, e.g. "TMPV.NS"
    sector_of: dict[str, str]
    sectors: list[str]
    names: dict[str, str]
    benchmark: str
    risk_free_annual: float
    spinoffs: dict[str, dict]

    @property
    def sector_index(self) -> dict[str, int]:
        return {s: i for i, s in enumerate(self.sectors)}

    @property
    def rf_weekly(self) -> float:
        return (1 + self.risk_free_annual) ** (1 / 52) - 1


def load_universe(path: Path | None = None) -> Universe:
    path = path or CONFIG_DIR / "universe.yaml"
    raw = yaml.safe_load(path.read_text())
    symbols = list(raw["stocks"].keys())
    tickers = {s: raw["stocks"][s].get("ticker", s) + ".NS" for s in symbols}
    return Universe(
        symbols=symbols,
        tickers=tickers,
        sector_of={s: raw["stocks"][s]["sector"] for s in symbols},
        sectors=list(raw["sectors"]),
        names={s: raw["stocks"][s]["name"] for s in symbols},
        benchmark=raw["benchmark"],
        risk_free_annual=float(raw["risk_free_annual"]),
        spinoffs=raw.get("spinoffs", {}) or {},
    )


@dataclass
class Splits:
    """PRD §5 splits. Bonus holdout (2025 -> today) is never touched during development."""
    data_start: str = "2017-01-01"      # 60-week lookback + ~35-week indicator warm-up before 2019-01
    train_start: str = "2019-01-01"
    train_end: str = "2022-12-31"
    val_start: str = "2023-01-01"
    val_end: str = "2023-06-30"
    test_start: str = "2023-07-01"
    test_end: str = "2024-12-31"
    holdout_start: str = "2025-01-01"


@dataclass
class DataConfig:
    lookback_weeks: int = 60
    min_price_inr: float = 10.0
    min_avg_daily_volume: float = 100_000
    max_missing_frac: float = 0.10
    vol_clip: tuple[float, float] = (0.01, 0.10)
    rolling_vol_window: int = 4


@dataclass
class ModelConfig:
    d_model: int = 64
    n_heads: int = 8
    n_layers: int = 2
    d_hidden: int = 128
    d_head: int = 64
    dropout: float = 0.3
    vol_clip: tuple[float, float] = (0.01, 0.10)


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32              # PRD default; one batch = one week's cross-section is used for the turnover term
    epochs: int = 200
    patience: int = 20
    grad_clip: float = 1.0
    w_return: float = 0.5
    w_vol: float = 0.5
    lambda_turnover: float = 0.0      # 0 = plain MTL; >0 adds differentiable transaction-cost term
    cost_per_side: float = 0.001
    tau: float = 0.1                  # softmax temperature for soft portfolio weights in the cost term
    target_mode: str = "raw"          # "raw" next-week return or "cs_demean" (minus weekly cross-sectional mean)
    early_stop_metric: str = "ic"     # "ic" (rank-IC, low noise) or "sharpe" (PRD; noisy on a 26-week val set)
    seed: int = 42
    device: str = "auto"


@dataclass
class BacktestConfig:
    top_frac: float = 0.10
    bottom_frac: float = 0.10
    cost_per_side: float = 0.001
    # PRD says 5%, but with a 50-stock universe the top decile is 5 names -> 5% cap would leave 75% in cash.
    # Default = fully invested across the 5 picks; the 5% cap is reported as a sensitivity.
    max_position: float = 0.20
    initial_capital: float = 1_000_000.0
    retrain_every_weeks: int = 13


@dataclass
class Config:
    splits: Splits = field(default_factory=Splits)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        cfg = cls()
        for section, values in raw.items():
            sub = getattr(cfg, section)
            for k, v in values.items():
                if isinstance(v, list):
                    v = tuple(v)
                setattr(sub, k, v)
        return cfg


def default_config() -> Config:
    p = CONFIG_DIR / "default.yaml"
    return Config.from_yaml(p) if p.exists() else Config()
