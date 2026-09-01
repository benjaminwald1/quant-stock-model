"""Data acquisition and universe construction.

Two sources are supported:

* Kaggle daily OHLCV archives (see ``KAGGLE_DATASETS``) — the real thing.
* A synthetic panel generator — lets the whole pipeline be exercised and
  unit-tested without credentials, and provides a *null* dataset (pure noise)
  that any honest backtest must score at roughly zero.

Everything downstream consumes one canonical long-format frame with columns
``[date, ticker, open, high, low, close, volume]``, sorted by (ticker, date).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .config import KAGGLE_DATASETS, RAW_DIR, DataConfig

log = logging.getLogger(__name__)

CANONICAL_COLS = ["date", "ticker", "open", "high", "low", "close", "volume"]


# --------------------------------------------------------------------------
# Kaggle download
# --------------------------------------------------------------------------
def download_kaggle(dataset: str, dest: Path | None = None, force: bool = False) -> Path:
    """Download and unzip a Kaggle dataset. Returns the extraction directory.

    Requires Kaggle API credentials at ``~/.kaggle/kaggle.json`` (or the
    KAGGLE_USERNAME / KAGGLE_KEY environment variables).
    """
    slug = KAGGLE_DATASETS.get(dataset, dataset)
    dest = Path(dest) if dest else RAW_DIR / slug.split("/")[-1]
    if dest.exists() and any(dest.iterdir()) and not force:
        log.info("Dataset already present at %s (use force=True to re-download)", dest)
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Could not import the Kaggle client. Install it with `uv pip install kaggle`."
        ) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except (Exception, SystemExit) as exc:
        # The kaggle client calls sys.exit(1) on missing credentials rather than
        # raising, so SystemExit has to be caught explicitly or the process dies
        # here with only the client's own banner and no context about what
        # was being attempted.
        raise RuntimeError(
            "Kaggle authentication failed — no usable credentials found.\n"
            "  Easiest:  kaggle auth login          (browser OAuth, nothing to store)\n"
            "  Or:       export KAGGLE_API_TOKEN=<token from "
            "https://www.kaggle.com/settings/api>\n"
            "  Or:       save that token to ~/.kaggle/access_token\n"
            "(Older kaggle clients instead want a kaggle.json at ~/.kaggle/kaggle.json.)\n"
            "No credentials? Run the pipeline on generated data instead: qsm run --synthetic"
        ) from exc

    log.info("Downloading %s -> %s (this is a few hundred MB)", slug, dest)
    api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)
    return dest


# --------------------------------------------------------------------------
# Per-file readers
# --------------------------------------------------------------------------
def _read_huge_file(path: Path) -> pd.DataFrame | None:
    """Read one `Stocks/aapl.us.txt` file from the borismarjanovic archive."""
    try:
        df = pd.read_csv(path, usecols=["Date", "Open", "High", "Low", "Close", "Volume"])
    except Exception:
        return None
    if df.empty:
        return None
    df.columns = [c.lower() for c in df.columns]
    df["ticker"] = path.name.split(".")[0].upper()
    return df


def _read_jackson_file(path: Path) -> pd.DataFrame | None:
    """Read one `stocks/AAPL.csv` file from the jacksoncrow archive.

    That archive ships raw OHLC plus an adjusted close; we rescale OHLC by the
    adjustment ratio so that every price series is split/dividend adjusted and
    returns are continuous across corporate actions.
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "Adj Close" not in df.columns:
        return None
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    ratio = df["adj_close"] / df["close"].replace(0, np.nan)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col] * ratio
    df["ticker"] = path.stem.upper()
    return df[["date", "open", "high", "low", "close", "volume", "ticker"]]


def _discover(root: Path, dataset: str) -> tuple[list[Path], callable]:
    """Locate the per-ticker price files and pick the matching reader."""
    root = Path(root)
    if dataset in ("huge", KAGGLE_DATASETS["huge"]):
        files = sorted(root.rglob("Stocks/*.txt")) or sorted(root.rglob("*.us.txt"))
        return files, _read_huge_file
    if dataset in ("jackson", KAGGLE_DATASETS["jackson"]):
        files = sorted(root.rglob("stocks/*.csv"))
        return files, _read_jackson_file
    # Unknown dataset: sniff the layout.
    txt = sorted(root.rglob("*.txt"))
    if txt:
        return txt, _read_huge_file
    return sorted(root.rglob("*.csv")), _read_jackson_file


# --------------------------------------------------------------------------
# Panel assembly
# --------------------------------------------------------------------------
def _clean(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame | None:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if cfg.start:
        df = df[df["date"] >= pd.Timestamp(cfg.start)]
    if cfg.end:
        df = df[df["date"] <= pd.Timestamp(cfg.end)]
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    if len(df) < cfg.min_history_days:
        return None
    return df.sort_values("date")


def load_panel(root: Path, cfg: DataConfig, workers: int = 8) -> pd.DataFrame:
    """Read a Kaggle archive into the canonical long panel.

    Universe selection keeps the ``max_tickers`` names with the highest median
    dollar volume over the sample. That is a survivorship-biased choice on a
    static archive — see the README; it is a property of the free data, not of
    the pipeline, and it inflates backtest returns.
    """
    files, reader = _discover(root, cfg.dataset)
    if not files:
        raise FileNotFoundError(f"No price files found under {root}")
    log.info("Reading %d ticker files", len(files))

    def work(path: Path):
        raw = reader(path)
        if raw is None:
            return None
        return _clean(raw, cfg)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = [f for f in pool.map(work, files) if f is not None]

    if not frames:
        raise ValueError("Every ticker was filtered out; loosen DataConfig thresholds.")

    # Rank by liquidity and keep the top N before concatenating everything.
    liquidity = sorted(
        ((float((f["close"] * f["volume"]).median()), i) for i, f in enumerate(frames)),
        reverse=True,
    )
    keep = [frames[i] for _, i in liquidity[: cfg.max_tickers]]
    panel = pd.concat(keep, ignore_index=True)[CANONICAL_COLS]
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    log.info(
        "Panel: %d rows, %d tickers, %s to %s",
        len(panel),
        panel["ticker"].nunique(),
        panel["date"].min().date(),
        panel["date"].max().date(),
    )
    return panel


def apply_universe_filters(panel: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Flag which (date, ticker) rows are tradable, using trailing data only.

    Adds a boolean ``tradable`` column. The filters look at the *previous*
    close and the *previous* 21-day median dollar volume, so a row's
    eligibility is knowable before the bar it refers to is traded.
    """
    out = panel.sort_values(["ticker", "date"]).copy()
    g = out.groupby("ticker", sort=False)
    prev_close = g["close"].shift(1)
    # usd_rate is present for live panels; archives are USD already.
    dollar_vol = out["close"] * out["volume"] * out.get("usd_rate", 1.0)
    prev_dv = dollar_vol.groupby(out["ticker"], sort=False).transform(
        lambda s: s.shift(1).rolling(21, min_periods=10).median()
    )
    bar_num = g.cumcount()
    out["tradable"] = (
        (prev_close >= cfg.min_price)
        & (prev_dv >= cfg.min_dollar_volume)
        & (bar_num >= cfg.min_history_days)
    ).fillna(False)
    return out


# --------------------------------------------------------------------------
# Synthetic panel
# --------------------------------------------------------------------------
def make_synthetic(
    n_tickers: int = 120,
    n_days: int = 2600,
    seed: int = 0,
    signal_strength: float = 0.05,
    start: str = "2010-01-04",
) -> pd.DataFrame:
    """Generate a synthetic OHLCV panel with a known, weak, learnable signal.

    Returns are ``beta * market + signal + idiosyncratic noise``. The signal is
    a lagged blend of 1-day reversal and 21-day momentum, so a correctly built
    model should recover a modest positive information coefficient. Setting
    ``signal_strength=0`` yields a pure-noise null: any strategy that looks
    profitable on *that* panel is leaking.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    tickers = [f"SYN{i:03d}" for i in range(n_tickers)]

    market = rng.normal(0.0003, 0.011, n_days)
    betas = rng.normal(1.0, 0.35, n_tickers)
    idio_vol = rng.uniform(0.012, 0.030, n_tickers)

    rets = np.zeros((n_days, n_tickers))
    noise = rng.normal(0, 1, (n_days, n_tickers)) * idio_vol
    for t in range(n_days):
        alpha = np.zeros(n_tickers)
        if t > 22 and signal_strength > 0:
            rev = -rets[t - 1]
            mom = rets[t - 22 : t - 1].sum(axis=0)
            z = 0.6 * (rev / (idio_vol + 1e-9)) + 0.4 * (mom / (idio_vol * np.sqrt(21) + 1e-9))
            z = np.clip(z, -3, 3)
            alpha = signal_strength * z * idio_vol
        rets[t] = betas * market[t] + alpha + noise[t]

    close = 50 * np.exp(np.cumsum(rets, axis=0)) * rng.uniform(0.5, 2.0, n_tickers)
    intraday = np.abs(rng.normal(0, 0.008, (n_days, n_tickers)))
    open_ = close * (1 + rng.normal(0, 0.004, (n_days, n_tickers)))
    high = np.maximum(open_, close) * (1 + intraday)
    low = np.minimum(open_, close) * (1 - intraday)
    volume = rng.lognormal(13.5, 0.7, (n_days, n_tickers)) * (1 + 3 * np.abs(rets))

    panel = pd.DataFrame(
        {
            "date": np.repeat(dates.values, n_tickers),
            "ticker": np.tile(tickers, n_days),
            "open": open_.ravel(),
            "high": high.ravel(),
            "low": low.ravel(),
            "close": close.ravel(),
            "volume": volume.ravel().round(),
        }
    )
    return panel.sort_values(["ticker", "date"]).reset_index(drop=True)
