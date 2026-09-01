"""Central configuration for the pipeline.

Every knob that changes results lives here so runs are reproducible from a
single object that gets serialised into the run directory.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Deployments mount a persistent volume and point these at it. Without the
# override everything lands in the container's filesystem and is wiped on the
# next deploy — the fund, the watchlist and every backtest with it.
_STATE = Path(os.environ.get("QSM_DATA_DIR") or (PROJECT_ROOT / "data"))
DATA_DIR = _STATE
RAW_DIR = DATA_DIR / "raw"
RUNS_DIR = Path(os.environ.get("QSM_RUNS_DIR") or (PROJECT_ROOT / "runs"))

# Kaggle datasets known to work with the loaders in qsm.data.
KAGGLE_DATASETS = {
    # ~7k US stocks + ETFs, daily OHLCV, through Nov 2017. Files: Stocks/aapl.us.txt
    "huge": "borismarjanovic/price-volume-data-for-all-us-stocks-etfs",
    # ~5.8k US tickers, daily OHLCV + adj close, through Apr 2020. Files: stocks/AAPL.csv
    "jackson": "jacksoncrow/stock-market-dataset",
}


@dataclass
class DataConfig:
    source: str = "kaggle"
    """Where prices come from: "kaggle" (static archive) or "live" (price API)."""

    provider: str = "yahoo"
    """Live provider. "yahoo" needs no key; "tiingo" reads TIINGO_API_KEY."""

    universe: str = "sp100"
    """Preset ticker list for live mode. See qsm.universe.PRESETS."""

    tickers: tuple[str, ...] = ()
    """Extra explicit tickers for live mode, added to the preset."""

    max_age_hours: float = 12.0
    """Reuse cached live data younger than this instead of re-hitting the API."""

    dataset: str = "huge"
    """Key into KAGGLE_DATASETS, or a raw `owner/slug` Kaggle dataset id."""

    max_tickers: int = 500
    """Cap on universe size. Tickers are chosen by median dollar volume."""

    start: str = "2005-01-01"
    end: str | None = None

    min_price: float = 5.0
    """Point-in-time price floor: penny stocks have unrealistic fill assumptions."""

    min_dollar_volume: float = 2_000_000.0
    """Point-in-time 21d median dollar volume floor, in dollars."""

    min_history_days: int = 300
    """Bars of history a ticker needs before it becomes eligible."""


@dataclass
class FeatureConfig:
    momentum_windows: tuple[int, ...] = (1, 5, 10, 21, 63, 126, 252)
    vol_windows: tuple[int, ...] = (21, 63)
    rsi_window: int = 14
    bollinger_window: int = 20
    atr_window: int = 14
    beta_window: int = 63
    cross_sectional_rank: bool = True
    """Rank-normalise each feature within each date. Strongly recommended: it
    makes the model learn relative rather than absolute signal, and it is
    immune to level shifts across market regimes."""


@dataclass
class LabelConfig:
    horizon: int = 5
    """Forward return horizon in trading days."""

    excess: bool = True
    """Demean the forward return cross-sectionally, so the target is
    performance relative to the universe rather than market beta."""

    rank_target: bool = True
    """Map the target to cross-sectional ranks in [-0.5, 0.5]. Reduces the
    influence of a handful of huge moves that would otherwise dominate MSE."""


@dataclass
class SplitConfig:
    n_splits: int = 6
    test_size: int = 252
    """Trading days per out-of-sample block."""

    min_train_size: int = 756
    embargo: int = 5
    """Extra days purged between train and test on top of the label horizon."""

    expanding: bool = True
    """Expanding train window; set False for a rolling window of min_train_size."""


@dataclass
class ModelConfig:
    models: tuple[str, ...] = ("ridge", "lgbm")
    """Any of: ridge, lgbm, gru. 'gru' needs the [nn] extra (torch)."""

    ridge_alpha: float = 10.0

    lgbm_params: dict = field(
        default_factory=lambda: {
            "objective": "regression",
            "n_estimators": 600,
            "learning_rate": 0.02,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 200,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
            "reg_lambda": 5.0,
            "verbose": -1,
        }
    )

    gru_params: dict = field(
        default_factory=lambda: {
            "seq_len": 20,
            "hidden": 48,
            "layers": 1,
            "dropout": 0.2,
            "lr": 1e-3,
            "epochs": 12,
            "batch_size": 512,
            "patience": 3,
            # None = pin to 1 thread iff LightGBM is loaded (avoids a macOS
            # OpenMP deadlock); 0 = never touch torch's threading; N = use N.
            "torch_threads": None,
        }
    )

    val_fraction: float = 0.15
    """Tail of each training block held out for early stopping. Taken from the
    end of the block (and itself purged) so it is never randomly interleaved."""

    seed: int = 7


@dataclass
class BacktestConfig:
    quantile: float = 0.1
    """Long the top decile, short the bottom decile, by default."""

    long_short: bool = True
    gross_exposure: float = 1.0
    cost_bps: float = 10.0
    """One-way transaction cost in basis points applied to turnover."""

    max_weight: float = 0.02
    """Per-name weight cap, applied after normalisation."""

    signal_weighted: bool = False
    """False = equal weight within each leg; True = weight by signal rank."""

    execution_lag: int = 1
    """Days between forming the signal and earning the return. 1 means you get
    a full session to trade after the close that produced the signal. Set to 0
    only if you really intend to trade market-on-close on the signal date."""

    n_quantile_bins: int = 5
    """Bins for the monotonicity report."""

    min_names: int = 20
    """Minimum names in a cross-section before it is traded at all. Ranking 6
    stocks into deciles is noise, not a portfolio."""


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, default=str))

    @classmethod
    def from_json(cls, path: Path) -> "Config":
        raw = json.loads(Path(path).read_text())
        return cls(
            data=DataConfig(**raw["data"]),
            features=FeatureConfig(**raw["features"]),
            labels=LabelConfig(**raw["labels"]),
            splits=SplitConfig(**raw["splits"]),
            model=ModelConfig(**raw["model"]),
            backtest=BacktestConfig(**raw["backtest"]),
        )
