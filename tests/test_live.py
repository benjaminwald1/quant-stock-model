"""Tests for the live market-data path.

The provider is stubbed throughout: a test suite that depends on a third-party
price API is a test suite that fails on a plane, on a rate limit, or on a
holiday. What matters here is our own reshaping, cleaning, caching and error
handling — not that Yahoo is up.
"""

import numpy as np
import pandas as pd
import pytest

from qsm import live as live_mod
from qsm.config import Config
from qsm.data import CANONICAL_COLS
from qsm.universe import PRESETS, resolve


# ── universe ──────────────────────────────────────────────────────────────
def test_presets_are_non_empty_and_clean():
    for name, syms in PRESETS.items():
        assert len(syms) >= 20, f"{name} is too small for a cross-section"
        assert len(syms) == len(set(syms)), f"{name} has duplicates"
        assert all(s == s.upper().strip() for s in syms)


def test_resolve_merges_preset_and_custom_without_duplicates():
    got = resolve("dow30", "AAPL,aapl NVDA")
    assert got.count("AAPL") == 1
    assert "NVDA" in got
    assert len(got) == len(set(got))


def test_resolve_rejects_unknown_preset():
    with pytest.raises(ValueError, match="Unknown universe"):
        resolve("ftse100")


# ── provider reshaping ────────────────────────────────────────────────────
def _fake_multi(tickers, days=30):
    """A yfinance-shaped frame: column MultiIndex of (field, ticker)."""
    dates = pd.bdate_range("2024-01-02", periods=days)
    cols = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], tickers])
    rng = np.random.default_rng(0)
    data = rng.uniform(50, 150, (days, len(cols)))
    frame = pd.DataFrame(data, index=dates, columns=cols)
    frame.index.name = "Date"
    for t in tickers:
        frame[("Volume", t)] = rng.integers(1e6, 9e6, days)
    return frame


def test_multi_ticker_payload_becomes_canonical_long(monkeypatch, tmp_path):
    monkeypatch.setattr(live_mod, "LIVE_DIR", tmp_path)
    monkeypatch.setitem(live_mod.PROVIDERS, "stub",
                        lambda t, s, e: live_mod._yahoo_to_long(_fake_multi(t), t))

    panel = live_mod.fetch_live(["AAPL", "MSFT"], start="2024-01-01", provider="stub")
    # Canonical columns first, then the USD conversion factor used for liquidity.
    assert list(panel.columns) == CANONICAL_COLS + ["usd_rate"]
    assert set(panel["ticker"]) == {"AAPL", "MSFT"}
    assert panel["date"].is_monotonic_increasing is False  # sorted by ticker first
    assert panel.sort_values(["ticker", "date"]).equals(panel)


def test_single_ticker_payload_is_handled(monkeypatch, tmp_path):
    """yfinance drops the MultiIndex for one ticker; that shape must work too."""
    dates = pd.bdate_range("2024-01-02", periods=10)
    flat = pd.DataFrame(
        {"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 1e6}, index=dates)
    flat.index.name = "Date"
    monkeypatch.setattr(live_mod, "LIVE_DIR", tmp_path)
    monkeypatch.setitem(live_mod.PROVIDERS, "stub",
                        lambda t, s, e: live_mod._yahoo_to_long(flat, t))

    panel = live_mod.fetch_live(["AAPL"], provider="stub")
    assert set(panel["ticker"]) == {"AAPL"}
    assert len(panel) == 10


# ── cleaning ──────────────────────────────────────────────────────────────
def test_cleaning_drops_bad_rows_and_duplicates(monkeypatch, tmp_path):
    raw = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-04", "2024-01-05"],
        "ticker": ["AAPL"] * 5,
        "open": [10, 11, 0, 12, 13],       # a zero price is not a price
        "high": [11, 12, 5, 13, 14],
        "low": [9, 10, 1, 11, 12],
        "close": [10.5, 11.5, 2, 12.5, np.nan],   # NaN close is unusable
        "volume": [1e6] * 5,
    })
    monkeypatch.setattr(live_mod, "LIVE_DIR", tmp_path)
    monkeypatch.setitem(live_mod.PROVIDERS, "stub", lambda t, s, e: raw)

    panel = live_mod.fetch_live(["AAPL"], provider="stub")
    assert len(panel) == 3                      # zero-price, dup and NaN rows gone
    assert panel["date"].is_unique
    assert (panel[["open", "high", "low", "close"]] > 0).all().all()


def test_all_bad_data_raises_rather_than_returning_empty(monkeypatch, tmp_path):
    raw = pd.DataFrame({
        "date": ["2024-01-02"], "ticker": ["AAPL"], "open": [0.0],
        "high": [0.0], "low": [0.0], "close": [0.0], "volume": [0.0]})
    monkeypatch.setattr(live_mod, "LIVE_DIR", tmp_path)
    monkeypatch.setitem(live_mod.PROVIDERS, "stub", lambda t, s, e: raw)
    with pytest.raises(RuntimeError, match="no usable rows"):
        live_mod.fetch_live(["AAPL"], provider="stub")


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="Unknown provider"):
        live_mod.fetch_live(["AAPL"], provider="bloomberg")


# ── caching ───────────────────────────────────────────────────────────────
def test_cache_prevents_a_second_provider_call(monkeypatch, tmp_path):
    calls = []

    def stub(t, s, e):
        calls.append(1)
        return live_mod._yahoo_to_long(_fake_multi(t), t)

    monkeypatch.setattr(live_mod, "LIVE_DIR", tmp_path)
    monkeypatch.setitem(live_mod.PROVIDERS, "stub", stub)

    first = live_mod.fetch_live(["AAPL"], provider="stub")
    second = live_mod.fetch_live(["AAPL"], provider="stub")
    assert len(calls) == 1, "second call should have been served from cache"
    pd.testing.assert_frame_equal(first, second)

    live_mod.fetch_live(["AAPL"], provider="stub", force=True)
    assert len(calls) == 2, "force=True must bypass the cache"

    live_mod.fetch_live(["AAPL"], provider="stub", max_age_hours=0)
    assert len(calls) == 3, "a zero max age must always refetch"


def test_different_tickers_do_not_share_a_cache_entry(monkeypatch, tmp_path):
    monkeypatch.setattr(live_mod, "LIVE_DIR", tmp_path)
    monkeypatch.setitem(live_mod.PROVIDERS, "stub",
                        lambda t, s, e: live_mod._yahoo_to_long(_fake_multi(t), t))
    a = live_mod.fetch_live(["AAPL"], provider="stub")
    b = live_mod.fetch_live(["MSFT"], provider="stub")
    assert set(a["ticker"]) == {"AAPL"} and set(b["ticker"]) == {"MSFT"}


# ── freshness reporting ───────────────────────────────────────────────────
def test_latest_close_reports_real_freshness(monkeypatch, tmp_path):
    monkeypatch.setattr(live_mod, "LIVE_DIR", tmp_path)
    monkeypatch.setitem(live_mod.PROVIDERS, "stub",
                        lambda t, s, e: live_mod._yahoo_to_long(_fake_multi(t, days=40), t))
    panel = live_mod.fetch_live(["AAPL", "MSFT"], provider="stub")
    info = live_mod.latest_close(panel)
    assert info["tickers"] == 2
    assert info["last_date"] == str(panel["date"].max().date())
    assert info["age_days"] >= 0
    assert info["stale_tickers"] == 0


# ── config plumbing ───────────────────────────────────────────────────────
def test_live_source_flows_through_load_source(monkeypatch, tmp_path):
    from qsm.pipeline import load_source

    monkeypatch.setattr(live_mod, "LIVE_DIR", tmp_path)
    monkeypatch.setitem(live_mod.PROVIDERS, "stub",
                        lambda t, s, e: live_mod._yahoo_to_long(_fake_multi(t, 60), t))
    cfg = Config()
    cfg.data.source = "live"
    cfg.data.provider = "stub"
    cfg.data.universe = "dow30"
    panel = load_source(cfg)
    assert list(panel.columns) == CANONICAL_COLS + ["usd_rate"]
    assert panel["ticker"].nunique() == 30


def test_max_tickers_caps_the_live_universe_by_liquidity(monkeypatch, tmp_path):
    from qsm.pipeline import load_source

    monkeypatch.setattr(live_mod, "LIVE_DIR", tmp_path)
    monkeypatch.setitem(live_mod.PROVIDERS, "stub",
                        lambda t, s, e: live_mod._yahoo_to_long(_fake_multi(t, 60), t))
    cfg = Config()
    cfg.data.source = "live"
    cfg.data.provider = "stub"
    cfg.data.universe = "dow30"
    cfg.data.max_tickers = 7
    assert load_source(cfg)["ticker"].nunique() == 7


# ── real-time quotes ──────────────────────────────────────────────────────
def test_market_state_classifies_the_session():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    cases = {
        datetime(2026, 8, 26, 10, 30, tzinfo=et): "open",         # Wed mid-session
        datetime(2026, 8, 26, 8, 0, tzinfo=et): "pre-market",
        datetime(2026, 8, 26, 17, 0, tzinfo=et): "after-hours",
        datetime(2026, 8, 26, 22, 0, tzinfo=et): "closed",
        datetime(2026, 8, 29, 11, 0, tzinfo=et): "closed",        # Saturday
        datetime(2026, 8, 30, 11, 0, tzinfo=et): "closed",        # Sunday
    }
    for when, expected in cases.items():
        assert live_mod.market_state(when)["state"] == expected, when
    assert live_mod.market_state(datetime(2026, 8, 26, 10, 30, tzinfo=et))["is_open"]
    assert not live_mod.market_state(datetime(2026, 8, 29, 11, 0, tzinfo=et))["is_open"]


def test_session_boundaries_are_inclusive_at_the_open_and_exclusive_at_the_close():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    assert live_mod.market_state(datetime(2026, 8, 26, 9, 30, tzinfo=et))["state"] == "open"
    assert live_mod.market_state(datetime(2026, 8, 26, 9, 29, tzinfo=et))["state"] == "pre-market"
    assert live_mod.market_state(datetime(2026, 8, 26, 15, 59, tzinfo=et))["state"] == "open"
    assert live_mod.market_state(datetime(2026, 8, 26, 16, 0, tzinfo=et))["state"] == "after-hours"


def _stub_yf(monkeypatch, closes: dict):
    """Install a fake yfinance whose download() returns given close series."""
    import sys
    import types

    dates = pd.bdate_range("2026-08-24", periods=5)

    def download(tickers, **kw):
        syms = tickers if isinstance(tickers, list) else [tickers]
        frame = pd.DataFrame(
            {s: closes.get(s, [np.nan] * 5) for s in syms}, index=dates)
        return pd.concat({"Close": frame}, axis=1)

    mod = types.ModuleType("yfinance")
    mod.download = download
    monkeypatch.setitem(sys.modules, "yfinance", mod)


def test_quotes_compute_change_from_the_previous_close(monkeypatch):
    _stub_yf(monkeypatch, {"AAPL": [10, 11, 12, 100.0, 110.0],
                           "MSFT": [5, 5, 5, 200.0, 180.0]})
    live_mod._QUOTE_CACHE.update({"at": 0.0, "key": None, "data": {}})

    q = live_mod.fetch_quotes(["AAPL", "MSFT"], ttl=0)
    assert q["quotes"]["AAPL"]["price"] == 110.0
    assert q["quotes"]["AAPL"]["prev_close"] == 100.0
    assert q["quotes"]["AAPL"]["change_pct"] == pytest.approx(10.0)
    assert q["quotes"]["MSFT"]["change_pct"] == pytest.approx(-10.0)
    assert q["returned"] == 2


def test_quotes_skip_tickers_without_enough_history(monkeypatch):
    _stub_yf(monkeypatch, {"AAPL": [1, 2, 3, 4, 5.0],
                           "DEAD": [np.nan] * 5})
    live_mod._QUOTE_CACHE.update({"at": 0.0, "key": None, "data": {}})

    q = live_mod.fetch_quotes(["AAPL", "DEAD"], ttl=0)
    assert "AAPL" in q["quotes"]
    assert "DEAD" not in q["quotes"], "a dead symbol must not sink the batch"
    assert q["requested"] == 2 and q["returned"] == 1


def test_quote_cache_avoids_hammering_the_provider(monkeypatch):
    calls = []
    _stub_yf(monkeypatch, {"AAPL": [1, 2, 3, 4, 5.0]})
    import sys

    inner = sys.modules["yfinance"].download
    sys.modules["yfinance"].download = lambda *a, **k: (calls.append(1), inner(*a, **k))[1]
    live_mod._QUOTE_CACHE.update({"at": 0.0, "key": None, "data": {}})

    live_mod.fetch_quotes(["AAPL"], ttl=60)
    second = live_mod.fetch_quotes(["AAPL"], ttl=60)
    assert len(calls) == 1
    assert second["cached"] is True


def test_a_provider_outage_returns_empty_rather_than_raising(monkeypatch):
    import sys
    import types

    mod = types.ModuleType("yfinance")

    def boom(*a, **k):
        raise RuntimeError("upstream is down")

    mod.download = boom
    monkeypatch.setitem(sys.modules, "yfinance", mod)
    live_mod._QUOTE_CACHE.update({"at": 0.0, "key": None, "data": {}})

    q = live_mod.fetch_quotes(["AAPL"], ttl=0)
    assert q["quotes"] == {} and q["returned"] == 0
    assert "market" in q, "the UI still needs the session state to render honestly"


# ── currency handling ─────────────────────────────────────────────────────
def test_currency_is_detected_from_the_yahoo_suffix():
    cases = {
        "AAPL": "USD", "SHEL.L": "GBp", "MC.PA": "EUR", "SAP.DE": "EUR",
        "7203.T": "JPY", "0700.HK": "HKD", "005930.KS": "KRW", "2330.TW": "TWD",
        "RELIANCE.NS": "INR", "BHP.AX": "AUD", "RY.TO": "CAD", "PETR4.SA": "BRL",
        "NESN.SW": "CHF", "UNKNOWN.ZZ": "USD",
    }
    for ticker, expected in cases.items():
        assert live_mod.currency_of(ticker) == expected, ticker


def test_london_pence_is_a_hundredth_of_a_pound(monkeypatch):
    """.L quotes are in pence. Missing that overstates UK liquidity 100x."""
    import sys
    import types

    dates = pd.bdate_range("2026-08-24", periods=5)
    mod = types.ModuleType("yfinance")
    mod.download = lambda tickers, **kw: pd.concat(
        {"Close": pd.DataFrame({s: [1.30] * 5 for s in tickers}, index=dates)}, axis=1)
    monkeypatch.setitem(sys.modules, "yfinance", mod)

    rates = live_mod.usd_rates({"GBp", "EUR"})
    assert rates["GBp"] == pytest.approx(0.013)
    assert rates["EUR"] == pytest.approx(1.30)
    assert rates["USD"] == 1.0


def test_missing_fx_rate_falls_back_to_one(monkeypatch):
    import sys
    import types

    mod = types.ModuleType("yfinance")

    def boom(*a, **k):
        raise RuntimeError("fx down")

    mod.download = boom
    monkeypatch.setitem(sys.modules, "yfinance", mod)
    rates = live_mod.usd_rates({"JPY"})
    assert rates["JPY"] == 1.0, "a missing rate must degrade, not crash"


def test_usd_rate_column_is_attached_per_ticker(monkeypatch, tmp_path):
    monkeypatch.setattr(live_mod, "LIVE_DIR", tmp_path)
    monkeypatch.setattr(live_mod, "usd_rates", lambda cs: {"USD": 1.0, "JPY": 0.0065})
    monkeypatch.setitem(live_mod.PROVIDERS, "stub",
                        lambda t, s, e: live_mod._yahoo_to_long(_fake_multi(t), t))

    panel = live_mod.fetch_live(["AAPL", "7203.T"], provider="stub")
    rates = panel.groupby("ticker")["usd_rate"].first()
    assert rates["AAPL"] == 1.0
    assert rates["7203.T"] == pytest.approx(0.0065)


def test_liquidity_ranking_uses_usd_not_raw_units(monkeypatch, tmp_path):
    """A yen-quoted name must not outrank a US one purely on unit size."""
    from qsm.pipeline import load_source

    dates = pd.bdate_range("2024-01-02", periods=60)

    def stub(tickers, start, end):
        rows = []
        for t in tickers:
            # Identical real liquidity: ~$1m/day for both.
            px, vol = (10_000.0, 15.4) if t.endswith(".T") else (100.0, 10_000.0)
            for d in dates:
                rows.append({"date": d, "ticker": t, "open": px, "high": px,
                             "low": px, "close": px, "volume": vol})
        return pd.DataFrame(rows)

    monkeypatch.setattr(live_mod, "LIVE_DIR", tmp_path)
    monkeypatch.setattr(live_mod, "usd_rates", lambda cs: {"USD": 1.0, "JPY": 0.0065})
    monkeypatch.setitem(live_mod.PROVIDERS, "stub", stub)
    monkeypatch.setattr("qsm.universe.PRESETS", {"mix": ["AAA", "BBB", "7203.T", "6758.T"]})

    cfg = Config()
    cfg.data.source = "live"
    cfg.data.provider = "stub"
    cfg.data.universe = "mix"
    cfg.data.max_tickers = 2
    kept = set(load_source(cfg)["ticker"].unique())
    # In raw yen the .T names dominate; in USD all four are equal, so the cap
    # must not select purely by currency unit.
    assert kept != {"7203.T", "6758.T"}, "ranking is still picking currencies, not liquidity"
