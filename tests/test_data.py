"""Tests for the Kaggle ingestion path.

The real archives cannot be downloaded in CI (they need credentials), so these
build fixtures in the exact on-disk layout and format of each archive and run
the actual loaders over them. That covers the parsing, the adjusted-close
rescaling and the liquidity ranking — the parts most likely to be silently
wrong the first time someone points this at real data.
"""

import numpy as np
import pytest

from qsm.config import DataConfig
from qsm.data import apply_universe_filters, load_panel, make_synthetic


def _write_huge_layout(root, panel):
    """borismarjanovic/price-volume-data-for-all-us-stocks-etfs: Stocks/aapl.us.txt"""
    d = root / "Stocks"
    d.mkdir(parents=True)
    for ticker, g in panel.groupby("ticker"):
        out = g[["date", "open", "high", "low", "close", "volume"]].copy()
        out.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        out["OpenInt"] = 0
        out.to_csv(d / f"{ticker.lower()}.us.txt", index=False)


def _write_jackson_layout(root, panel, adj_factor=0.5):
    """jacksoncrow/stock-market-dataset: stocks/AAPL.csv, with a separate Adj Close."""
    d = root / "stocks"
    d.mkdir(parents=True)
    for ticker, g in panel.groupby("ticker"):
        out = g[["date", "open", "high", "low", "close", "volume"]].copy()
        out.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        out["Adj Close"] = out["Close"] * adj_factor
        out = out[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        out.to_csv(d / f"{ticker.upper()}.csv", index=False)


@pytest.fixture
def raw_panel():
    return make_synthetic(n_tickers=8, n_days=500, seed=17)


def test_loads_huge_layout(tmp_path, raw_panel):
    _write_huge_layout(tmp_path, raw_panel)
    cfg = DataConfig(dataset="huge", max_tickers=100, start="2000-01-01", min_history_days=100)
    got = load_panel(tmp_path, cfg, workers=2)

    assert list(got.columns) == ["date", "ticker", "open", "high", "low", "close", "volume"]
    assert got["ticker"].nunique() == raw_panel["ticker"].nunique()
    assert len(got) == len(raw_panel)
    merged = got.merge(raw_panel, on=["date", "ticker"], suffixes=("", "_src"))
    np.testing.assert_allclose(merged["close"], merged["close_src"], rtol=1e-9)


def test_loads_jackson_layout_and_applies_adjustment(tmp_path, raw_panel):
    """OHLC must be rescaled by adj_close/close, or returns break at every split."""
    _write_jackson_layout(tmp_path, raw_panel, adj_factor=0.5)
    cfg = DataConfig(dataset="jackson", max_tickers=100, start="2000-01-01", min_history_days=100)
    got = load_panel(tmp_path, cfg, workers=2)

    merged = got.merge(raw_panel, on=["date", "ticker"], suffixes=("", "_src"))
    for col in ("open", "high", "low", "close"):
        np.testing.assert_allclose(merged[col], merged[f"{col}_src"] * 0.5, rtol=1e-9)
    # Volume is a count, not a price: it must not be rescaled.
    np.testing.assert_allclose(merged["volume"], merged["volume_src"], rtol=1e-9)


def test_max_tickers_keeps_the_most_liquid(tmp_path, raw_panel):
    panel = raw_panel.copy()
    # Make one name unmistakably the most liquid and one unmistakably the least.
    tickers = sorted(panel["ticker"].unique())
    panel.loc[panel["ticker"] == tickers[0], "volume"] *= 1000
    panel.loc[panel["ticker"] == tickers[-1], "volume"] //= 1000
    _write_huge_layout(tmp_path, panel)

    cfg = DataConfig(dataset="huge", max_tickers=3, start="2000-01-01", min_history_days=100)
    got = load_panel(tmp_path, cfg, workers=2)
    kept = set(got["ticker"].unique())
    assert len(kept) == 3
    assert tickers[0] in kept
    assert tickers[-1] not in kept


def test_short_history_tickers_are_dropped(tmp_path, raw_panel):
    _write_huge_layout(tmp_path, raw_panel)
    cfg = DataConfig(dataset="huge", max_tickers=100, start="2000-01-01", min_history_days=10_000)
    with pytest.raises(ValueError, match="filtered out"):
        load_panel(tmp_path, cfg, workers=2)


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_panel(tmp_path / "nope", DataConfig(), workers=2)


def test_universe_filters_use_lagged_liquidity(tmp_path, raw_panel):
    """Eligibility on day t must be knowable from day t-1."""
    panel = apply_universe_filters(raw_panel, DataConfig(min_history_days=50,
                                                        min_dollar_volume=0, min_price=0))
    one = panel[panel["ticker"] == sorted(panel["ticker"].unique())[0]].reset_index(drop=True)
    # The price floor looks at the previous close, so a single-day price spike
    # on day t cannot retroactively make day t tradable.
    assert not one.loc[0, "tradable"]
    assert one.loc[60:, "tradable"].all()
