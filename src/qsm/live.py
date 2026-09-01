"""Live market data from the exchange, via a public price API.

Yahoo (through ``yfinance``) is the default because it needs no API key and no
account. Keyed providers are supported through the same interface when the
relevant environment variable is present.

Two things matter more than the plumbing:

* **Adjusted prices.** Every provider here returns split/dividend-adjusted
  prices. Unadjusted series produce fake 50% "crashes" on split dates that a
  momentum model will happily learn.
* **This is a live *feed*, not a live *universe*.** A price API tells you what
  the companies trading today did in the past. It cannot tell you which
  companies were in the index in 2015 and have since been delisted. See
  ``qsm.universe`` — the survivorship problem is not solved by fresher data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .data import CANONICAL_COLS

log = logging.getLogger(__name__)

LIVE_DIR = DATA_DIR / "live"

# Yahoo suffix -> quote currency. Without this, ranking a global universe by
# `close * volume` just ranks currencies: a Tokyo name quoted in yen looks ~150x
# more liquid than an identical US one, and London is quoted in *pence*, not
# pounds, so it is out by a further factor of 100.
CURRENCY = {
    "L": "GBp", "PA": "EUR", "DE": "EUR", "AS": "EUR", "BR": "EUR", "MI": "EUR",
    "MC": "EUR", "LS": "EUR", "VI": "EUR", "HE": "EUR", "IR": "EUR", "F": "EUR",
    "SW": "CHF", "ST": "SEK", "CO": "DKK", "OL": "NOK",
    "T": "JPY", "HK": "HKD", "KS": "KRW", "KQ": "KRW", "TW": "TWD", "TWO": "TWD",
    "NS": "INR", "BO": "INR", "SI": "SGD", "SS": "CNY", "SZ": "CNY",
    "AX": "AUD", "NZ": "NZD", "TO": "CAD", "V": "CAD", "SA": "BRL", "MX": "MXN",
}


SYMBOL = {
    "USD": "$", "EUR": "\u20ac", "GBp": "p", "JPY": "\u00a5", "HKD": "HK$",
    "KRW": "\u20a9", "CHF": "CHF ", "AUD": "A$", "CAD": "C$", "BRL": "R$",
    "INR": "\u20b9", "TWD": "NT$", "SGD": "S$", "SEK": "kr", "DKK": "kr",
    "NOK": "kr", "CNY": "\u00a5", "MXN": "Mex$", "NZD": "NZ$",
}


def currency_of(ticker: str) -> str:
    """Quote currency for a Yahoo symbol. No suffix means USD."""
    if "." not in ticker:
        return "USD"
    return CURRENCY.get(ticker.rsplit(".", 1)[-1].upper(), "USD")


def usd_rates(currencies: set[str]) -> dict[str, float]:
    """Fetch {currency: units of USD per 1 unit} for the given currencies.

    GBp (pence) is handled as GBP/100. Anything that cannot be priced falls back
    to 1.0 and is logged, so a missing rate degrades the liquidity ranking for
    that market rather than silently dropping it.
    """
    rates = {"USD": 1.0}
    need = {c for c in currencies if c not in rates}
    if not need:
        return rates
    try:
        import yfinance as yf
    except ImportError:  # pragma: no cover - optional dependency
        return {c: 1.0 for c in currencies} | {"USD": 1.0}

    pairs = {c: f"{'GBP' if c == 'GBp' else c}USD=X" for c in need}
    try:
        data = yf.download(list(pairs.values()), period="5d", interval="1d",
                           progress=False, threads=True, auto_adjust=False)
        close = data["Close"] if "Close" in data else None
        if close is not None and isinstance(close, pd.Series):
            close = close.to_frame(list(pairs.values())[0])
    except Exception as exc:  # pragma: no cover - network variance
        log.warning("FX fetch failed (%s); liquidity ranking falls back to raw values", exc)
        close = None

    for cur, sym in pairs.items():
        rate = None
        if close is not None and sym in close.columns:
            series = close[sym].dropna()
            if len(series):
                rate = float(series.iloc[-1])
        if rate is None:
            log.warning("no FX rate for %s; using 1.0", cur)
            rate = 1.0
        rates[cur] = rate / 100.0 if cur == "GBp" else rate
    return rates
DEFAULT_PROVIDER = "yahoo"
CHUNK = 60


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------
def _yahoo(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    """Daily adjusted OHLCV from Yahoo Finance. No key required."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Live data needs yfinance. Install it with: uv pip install -e '.[live]'"
        ) from exc

    frames = []
    for i in range(0, len(tickers), CHUNK):
        batch = tickers[i : i + CHUNK]
        log.info("fetching %d/%d tickers from Yahoo", min(i + CHUNK, len(tickers)), len(tickers))
        raw = yf.download(
            batch, start=start, end=end, auto_adjust=True, progress=False,
            threads=False, group_by="column",
        )
        if raw is None or raw.empty:
            continue
        frames.append(_yahoo_to_long(raw, batch))
        if i + CHUNK < len(tickers):
            time.sleep(0.4)  # be a polite client
    if not frames:
        raise RuntimeError("Yahoo returned no data for any requested ticker.")
    return pd.concat(frames, ignore_index=True)


def _yahoo_to_long(raw: pd.DataFrame, batch: list[str]) -> pd.DataFrame:
    """Reshape yfinance output to the canonical long panel.

    yfinance returns a column MultiIndex for several tickers but flat columns
    for one, so both shapes have to be handled.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        stacked = raw.stack(level=-1, future_stack=True).reset_index()
        stacked.columns = [str(c) for c in stacked.columns]
        stacked = stacked.rename(columns={stacked.columns[0]: "date",
                                          stacked.columns[1]: "ticker"})
    else:
        stacked = raw.reset_index()
        stacked.columns = [str(c) for c in stacked.columns]
        stacked = stacked.rename(columns={stacked.columns[0]: "date"})
        stacked["ticker"] = batch[0]

    stacked.columns = [c.lower().replace(" ", "_") for c in stacked.columns]
    keep = ["date", "ticker", "open", "high", "low", "close", "volume"]
    missing = [c for c in keep if c not in stacked.columns]
    if missing:
        raise RuntimeError(f"Unexpected provider payload; missing {missing}")
    return stacked[keep]


def _tiingo(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    """Tiingo daily adjusted OHLCV. Needs TIINGO_API_KEY."""
    import urllib.parse
    import urllib.request

    token = os.environ.get("TIINGO_API_KEY")
    if not token:
        raise RuntimeError("TIINGO_API_KEY is not set in the environment.")

    rows = []
    for t in tickers:
        params = {"startDate": start, "format": "json", "token": token}
        if end:
            params["endDate"] = end
        url = (f"https://api.tiingo.com/tiingo/daily/{urllib.parse.quote(t)}/prices?"
               + urllib.parse.urlencode(params))
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                payload = json.loads(resp.read())
        except Exception as exc:
            log.warning("tiingo: %s failed (%s)", t, exc)
            continue
        for r in payload:
            rows.append({
                "date": r["date"][:10], "ticker": t,
                "open": r.get("adjOpen"), "high": r.get("adjHigh"),
                "low": r.get("adjLow"), "close": r.get("adjClose"),
                "volume": r.get("adjVolume"),
            })
        time.sleep(0.1)
    if not rows:
        raise RuntimeError("Tiingo returned no data for any requested ticker.")
    return pd.DataFrame(rows)


PROVIDERS = {"yahoo": _yahoo, "tiingo": _tiingo}


# --------------------------------------------------------------------------
# Fetch + cache
# --------------------------------------------------------------------------
def _cache_path(provider: str, tickers: list[str], start: str, end: str | None) -> Path:
    key = hashlib.sha1(
        json.dumps([provider, sorted(tickers), start, end]).encode()
    ).hexdigest()[:16]
    return LIVE_DIR / f"{provider}-{len(tickers)}n-{key}.parquet"


def fetch_live(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str | None = None,
    provider: str = DEFAULT_PROVIDER,
    max_age_hours: float = 12.0,
    force: bool = False,
) -> pd.DataFrame:
    """Download daily adjusted OHLCV and return the canonical long panel.

    Results are cached to parquet. A cache newer than ``max_age_hours`` is
    reused, so re-running a backtest does not re-hit the provider for data that
    only changes once a day.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {', '.join(PROVIDERS)}")
    tickers = [t.upper() for t in tickers]
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(provider, tickers, start, end)

    if cache.exists() and not force:
        age = datetime.now() - datetime.fromtimestamp(cache.stat().st_mtime)
        if age < timedelta(hours=max_age_hours):
            log.info("using cached live data (%.1fh old): %s", age.total_seconds() / 3600, cache.name)
            try:
                return pd.read_parquet(cache)
            except Exception as exc:
                # A cache truncated by a crash or a concurrent write must not
                # take the page down — drop it and fetch again.
                log.warning("cache %s unreadable (%s); refetching", cache.name, exc)
                cache.unlink(missing_ok=True)

    log.info("fetching %d tickers from %s since %s", len(tickers), provider, start)
    raw = PROVIDERS[provider](tickers, start, end)
    panel = _clean_live(raw, tickers)
    _atomic_parquet(panel, cache)
    log.info(
        "live panel: %d rows, %d tickers, %s to %s",
        len(panel), panel["ticker"].nunique(),
        panel["date"].min().date(), panel["date"].max().date(),
    )
    return panel


def _atomic_parquet(df: pd.DataFrame, dest: Path) -> None:
    """Write via a temp file and rename.

    Writing straight to the destination leaves a window where a concurrent
    reader sees a half-written file — which is exactly how the cache ended up
    corrupt. os.replace is atomic within a filesystem.
    """
    tmp = dest.with_suffix(dest.suffix + f".tmp{os.getpid()}")
    try:
        df.to_parquet(tmp)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _clean_live(raw: pd.DataFrame, requested: list[str]) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")

    got = set(df["ticker"].unique())
    missing = sorted(set(requested) - got)
    if missing:
        log.warning("no data for %d ticker(s): %s", len(missing),
                    ", ".join(missing[:12]) + ("…" if len(missing) > 12 else ""))
    if df.empty:
        raise RuntimeError("Provider returned no usable rows after cleaning.")

    out = df[CANONICAL_COLS].sort_values(["ticker", "date"]).reset_index(drop=True)

    # Per-ticker USD conversion factor, used only for liquidity comparisons.
    # Prices themselves stay in local currency: converting them would inject FX
    # returns into every stock's series, which is a different strategy.
    curs = {t: currency_of(t) for t in out["ticker"].unique()}
    rates = usd_rates(set(curs.values()))
    out["usd_rate"] = out["ticker"].map({t: rates.get(c, 1.0) for t, c in curs.items()})
    if len(set(curs.values())) > 1:
        log.info("currencies: %s", ", ".join(sorted(set(curs.values()))))
    return out


# --------------------------------------------------------------------------
# Real-time quotes
# --------------------------------------------------------------------------
_QUOTE_CACHE: dict = {"at": 0.0, "key": None, "data": {}}
QUOTE_TTL = 10.0
"""Seconds to reuse a quote response. Polling a public endpoint harder than
this gets you rate-limited, and daily-bar signals do not move in between."""


def market_state(now: datetime | None = None) -> dict:
    """Is the US equity market open right now?

    Quotes during a closed market are just the last close wearing a live badge,
    so the UI needs to be able to say which it is showing.
    """
    from zoneinfo import ZoneInfo

    et = (now or datetime.now(tz=ZoneInfo("America/New_York")))
    if et.tzinfo is None:
        et = et.replace(tzinfo=ZoneInfo("America/New_York"))
    et = et.astimezone(ZoneInfo("America/New_York"))

    weekday = et.weekday() < 5
    minutes = et.hour * 60 + et.minute
    regular = 9 * 60 + 30 <= minutes < 16 * 60
    pre = 4 * 60 <= minutes < 9 * 60 + 30
    post = 16 * 60 <= minutes < 20 * 60

    if weekday and regular:
        state = "open"
    elif weekday and pre:
        state = "pre-market"
    elif weekday and post:
        state = "after-hours"
    else:
        state = "closed"
    # NB: US market holidays are not modelled here, so a holiday reads as
    # "open" with quotes that never move. The staleness of the quote timestamp
    # is the honest signal, not this flag alone.
    return {"state": state, "is_open": state == "open",
            "exchange_time": et.strftime("%Y-%m-%d %H:%M:%S %Z")}


def fetch_quotes(tickers: list[str], ttl: float = QUOTE_TTL) -> dict:
    """Current price and day change for each ticker.

    Cached briefly so a page full of pollers does not multiply into a
    rate-limit. Failures degrade per-ticker rather than sinking the response.
    """
    tickers = [t.upper() for t in tickers][:200]
    key = tuple(sorted(tickers))
    now = time.time()
    if _QUOTE_CACHE["key"] == key and now - _QUOTE_CACHE["at"] < ttl:
        return {**_QUOTE_CACHE["data"], "cached": True}

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Live quotes need yfinance: uv pip install -e '.[live]'") from exc

    # One batched request for the whole list. The per-ticker fast_info route
    # issues an HTTP call per symbol — about 9s for 30 names and far worse for
    # a full universe, which is useless for anything calling itself live.
    # Daily bars give both numbers we need: the last bar is today's (and it
    # updates through the session), the one before it is the previous close.
    quotes: dict[str, dict] = {}
    try:
        raw = yf.download(tickers, period="5d", interval="1d", progress=False,
                          threads=True, auto_adjust=False, group_by="column")
        if raw is not None and not raw.empty:
            close = raw["Close"] if "Close" in raw else None
            if close is not None:
                if isinstance(close, pd.Series):        # single ticker
                    close = close.to_frame(tickers[0])
                for sym in close.columns:
                    series = close[sym].dropna()
                    if len(series) < 2:
                        continue
                    last, prev = float(series.iloc[-1]), float(series.iloc[-2])
                    if prev == 0:
                        continue
                    quotes[str(sym).upper()] = {
                        "price": round(last, 4),
                        "prev_close": round(prev, 4),
                        "change": round(last - prev, 4),
                        "change_pct": round((last / prev - 1) * 100, 3),
                    }
    except Exception as exc:
        log.warning("quote fetch failed: %s", exc)

    payload = {
        "quotes": quotes,
        "requested": len(tickers),
        "returned": len(quotes),
        "fetched_at": datetime.now().strftime("%H:%M:%S"),
        "market": market_state(),
        "cached": False,
    }
    _QUOTE_CACHE.update({"at": now, "key": key, "data": payload})
    return payload


def latest_close(panel: pd.DataFrame) -> dict:
    """Freshness summary for the UI: how current is this data, really."""
    last = panel["date"].max()
    per_ticker = panel.groupby("ticker")["date"].max()
    stale = per_ticker[per_ticker < last]
    return {
        "last_date": str(last.date()),
        "age_days": int((pd.Timestamp.now().normalize() - last.normalize()).days),
        "tickers": int(panel["ticker"].nunique()),
        "rows": int(len(panel)),
        "stale_tickers": int(len(stale)),
        "first_date": str(panel["date"].min().date()),
        "median_dollar_volume": float(
            np.nanmedian((panel["close"] * panel["volume"]).to_numpy())
        ),
    }


# --------------------------------------------------------------------------
# Chart history at multiple ranges
# --------------------------------------------------------------------------
# (yfinance period, interval) per range key. Short ranges use intraday bars:
# on daily data a "3D" chart would be three dots. Yahoo limits 1m to about a
# week of history and 5m to 60 days, which is what sets these choices.
RANGES = {
    "1d":  ("1d",  "1m"),
    "3d":  ("5d",  "5m"),
    "5d":  ("5d",  "5m"),
    "1mo": ("1mo", "1h"),
    "3mo": ("3mo", "1d"),
    "6mo": ("6mo", "1d"),
    "ytd": ("ytd", "1d"),
    "1y":  ("1y",  "1d"),
    "max": ("max", "1wk"),
}

# Bounded on purpose. Keyed by (ticker, range), an unbounded dict grows with
# every name anyone looks at — on a 4,000-name universe across nine ranges that
# is tens of thousands of price series held forever, which is how a long-lived
# server on a 1 GB box dies. Oldest entries go first.
_HISTORY_CACHE: dict = {}
_HISTORY_CACHE_MAX = 400


def _cache_put(cache: dict, key, value, limit: int) -> None:
    if len(cache) >= limit:
        for stale in list(cache)[: max(1, limit // 4)]:
            cache.pop(stale, None)
    cache[key] = value

HISTORY_TTL = 120.0
# Intraday charts are watched tick by tick, so they have to expire fast enough
# that the next poll shows the bar that just printed. A two-minute cache on a
# one-minute chart is a chart that is always two bars stale. Coarser intraday
# bars get a longer cache, because there is nothing new to fetch between them.
INTRADAY_TTL = {"1d": 15.0, "3d": 60.0, "5d": 60.0, "1mo": 120.0}


def price_history(ticker: str, range_key: str = "6mo") -> dict:
    """Close prices for one ticker over a named range, for charting.

    Intraday for short ranges, daily in the middle, weekly for max — otherwise
    "all time" on a 30-year listing is tens of thousands of points to draw a
    line nobody can read at that density.
    """
    key = (range_key or "6mo").lower()
    if key not in RANGES:
        raise ValueError(f"Unknown range '{range_key}'. Choose from: {', '.join(RANGES)}")

    cache_key = (ticker.upper(), key)
    now = time.time()
    ttl = INTRADAY_TTL.get(key, HISTORY_TTL)
    hit = _HISTORY_CACHE.get(cache_key)
    if hit and now - hit[0] < ttl:
        return hit[1]

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Chart history needs yfinance.") from exc

    period, interval = RANGES[key]
    try:
        raw = yf.download(ticker, period=period, interval=interval, progress=False,
                          threads=False, auto_adjust=True)
    except Exception as exc:
        log.warning("history fetch failed for %s (%s): %s", ticker, key, exc)
        raw = None

    if raw is None or raw.empty:
        payload = {"ticker": ticker.upper(), "range": key, "interval": interval,
                   "points": 0, "labels": [], "close": []}
        _cache_put(_HISTORY_CACHE, cache_key, (now, payload), _HISTORY_CACHE_MAX)
        return payload

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()

    # "3d" has no Yahoo period of its own; take the tail of the 5-day pull.
    if key == "3d" and len(close):
        cutoff = close.index.max() - pd.Timedelta(days=3)
        close = close[close.index >= cutoff]

    # "1d" means *this* session and nothing else. Around the open Yahoo will
    # return the tail of the previous session alongside the first bars of
    # today, which puts yesterday's prices on a chart labelled one day.
    if key == "1d" and len(close):
        day = close.index.normalize()
        close = close[day == day.max()]

    intraday = interval.endswith(("m", "h"))
    labels = [t.strftime("%Y-%m-%d %H:%M") if intraday else t.strftime("%Y-%m-%d")
              for t in close.index]

    payload = {
        "ticker": ticker.upper(),
        "range": key,
        "interval": interval,
        "intraday": intraday,
        "points": int(len(close)),
        "labels": labels,
        "close": [round(float(v), 4) for v in close],
        "first": labels[0] if labels else None,
        "last": labels[-1] if labels else None,
        "session": labels[-1][:10] if labels else None,
        "change_pct": round(float(close.iloc[-1] / close.iloc[0] - 1) * 100, 3)
        if len(close) > 1 else None,
    }
    _cache_put(_HISTORY_CACHE, cache_key, (now, payload), _HISTORY_CACHE_MAX)
    return payload


def intraday_returns(ticker: str, lookback_days: int = 5) -> np.ndarray:
    """Recent intraday bar-to-bar log returns, for projecting inside a session.

    A one-day-ahead projection built from *daily* bars is nine points of near
    straight line sitting beside a chart of 390 real intraday bars — it looks
    nothing like a trading day because it is not made of one. These are the
    returns to bootstrap from instead.
    """
    hist = price_history(ticker, "5d" if lookback_days <= 5 else "1mo")
    closes = np.asarray(hist.get("close", []), dtype=float)
    closes = closes[np.isfinite(closes) & (closes > 0)]
    if len(closes) < 30:
        return np.array([])
    r = np.diff(np.log(closes))
    # Drop the overnight gaps: they are not intraday moves and would inflate
    # the per-bar volatility used to fill in a session.
    keep = np.abs(r) < np.nanpercentile(np.abs(r), 99.5)
    return r[keep]


_DIP_CACHE: dict = {}
_DIP_CACHE_MAX = 64
DIP_TTL = 6 * 3600


def dip_profile(tickers: list[str], dip_pct: float = 0.01,
                lookback_days: int = 60) -> dict:
    """How often each name dips below its open, and when in the session.

    A base rate from the last ~60 sessions of 5-minute bars — not a forecast.
    It answers "when has this stock historically reached that price", which is
    the honest version of "when will the order fill".

    The reference is the session's true opening print and the opening bar's own
    low is excluded. Using the first bar's close as the reference and then
    testing that same bar's low reports a near-100% hit rate purely by
    construction, because the opening bar's range is wide.
    """
    import numpy as np

    key = (tuple(sorted(tickers)), round(dip_pct, 4))
    now = time.time()
    hit = _DIP_CACHE.get(key)
    if hit and now - hit[0] < DIP_TTL:
        return hit[1]

    try:
        import yfinance as yf
    except ImportError:  # pragma: no cover
        return {}

    try:
        raw = yf.download(tickers, period=f"{lookback_days}d", interval="5m",
                          progress=False, threads=True, auto_adjust=False, prepost=False)
    except Exception as exc:
        log.warning("dip profile fetch failed: %s", exc)
        return {}
    if raw is None or raw.empty or "Open" not in raw:
        return {}

    opens, lows = raw["Open"], raw["Low"]
    if isinstance(opens, pd.Series):
        opens, lows = opens.to_frame(tickers[0]), lows.to_frame(tickers[0])
    try:
        idx = opens.index.tz_convert("America/New_York")
    except (TypeError, AttributeError):
        idx = opens.index
    opens.index, lows.index = idx, idx

    out = {}
    for t in opens.columns:
        sessions = hits = 0
        minutes = []
        for _, grp in opens[t].groupby(opens.index.date):
            grp = grp.dropna()
            if len(grp) < 20:
                continue
            sessions += 1
            limit = float(grp.iloc[0]) * (1 - dip_pct)
            after = lows[t].loc[grp.index].dropna().iloc[1:]
            touched = after[after <= limit]
            if len(touched):
                hits += 1
                ts = touched.index[0]
                minutes.append(ts.hour * 60 + ts.minute)
        if not sessions:
            continue
        fmt = lambda m: f"{int(m) // 60:02d}:{int(m) % 60:02d}"          # noqa: E731
        out[str(t).upper()] = {
            "sessions": sessions,
            "hit_rate": round(hits / sessions, 4),
            "median_time": fmt(np.median(minutes)) if minutes else None,
            "typical_from": fmt(np.percentile(minutes, 25)) if minutes else None,
            "typical_to": fmt(np.percentile(minutes, 75)) if minutes else None,
        }

    _cache_put(_DIP_CACHE, key, (now, out), _DIP_CACHE_MAX)
    return out
