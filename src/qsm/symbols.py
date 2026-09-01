"""Fetch the full list of US-listed symbols from public exchange directories.

Nasdaq publishes two plain-text directories covering essentially every symbol
trading on a US exchange. They are the closest thing to a free, complete list.

The filtering matters more than the fetching. Raw, the files contain ETFs,
warrants, units, rights, preferred shares, test issues and companies in
delinquent filing status — none of which belong in a cross-sectional equity
model. What survives is common stock.
"""

from __future__ import annotations

import io
import logging
import urllib.request
from datetime import datetime, timedelta

import pandas as pd

from .config import DATA_DIR

log = logging.getLogger(__name__)

SYMBOL_DIR = DATA_DIR / "symbols"
NASDAQ_LISTED = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
SP500_WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Suffixes Nasdaq appends for non-common share classes.
_BAD_SUFFIX = ("W", "R", "U", "P")


def _get(url: str, cache_hours: float = 24.0) -> bytes:
    SYMBOL_DIR.mkdir(parents=True, exist_ok=True)
    cache = SYMBOL_DIR / (url.rsplit("/", 1)[-1].replace("%", "") or "page")
    if cache.exists():
        age = datetime.now() - datetime.fromtimestamp(cache.stat().st_mtime)
        if age < timedelta(hours=cache_hours):
            return cache.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (qsm research)"})
    raw = urllib.request.urlopen(req, timeout=30).read()
    cache.write_bytes(raw)
    return raw


def _clean(df: pd.DataFrame, symbol_col: str) -> list[str]:
    df = df[df[symbol_col].notna()].copy()
    df[symbol_col] = df[symbol_col].astype(str).str.strip()

    # Drop the trailing "File Creation Time" row these files end with.
    df = df[~df[symbol_col].str.contains("File Creation", case=False, na=False)]

    if "Test Issue" in df.columns:
        df = df[df["Test Issue"].astype(str).str.upper() != "Y"]
    if "ETF" in df.columns:
        df = df[df["ETF"].astype(str).str.upper() != "Y"]
    if "Financial Status" in df.columns:                 # D = delinquent, E = deficient
        df = df[~df["Financial Status"].astype(str).str.upper().isin(list("DEQ"))]

    syms = df[symbol_col].tolist()
    out = []
    for s in syms:
        if not s or not s.replace(".", "").replace("-", "").isalnum():
            continue
        if len(s) > 5:                     # units/warrants run long
            continue
        if "$" in s or "." in s:           # preferred and when-issued classes
            continue
        if len(s) == 5 and s[-1] in _BAD_SUFFIX:
            continue
        out.append(s.upper())
    return sorted(set(out))


def us_listed(cache_hours: float = 24.0) -> list[str]:
    """Every US-listed common stock, from both Nasdaq directories."""
    frames = []
    for url, col in ((NASDAQ_LISTED, "Symbol"), (OTHER_LISTED, "ACT Symbol")):
        try:
            df = pd.read_csv(io.BytesIO(_get(url, cache_hours)), sep="|")
            key = col if col in df.columns else df.columns[0]
            frames.append(_clean(df, key))
        except Exception as exc:
            log.warning("could not fetch %s: %s", url, exc)
    if not frames:
        raise RuntimeError("Could not fetch any US symbol directory.")
    syms = sorted(set().union(*(set(f) for f in frames)))
    log.info("US listed common stock: %d symbols", len(syms))
    return syms


def sp500(cache_hours: float = 168.0) -> list[str]:
    """Current S&P 500 constituents.

    Note this is *today's* membership applied to all of history — the same
    survivorship problem as every other preset here, and the reason a backtest
    over it flatters itself.
    """
    try:
        tables = pd.read_html(io.BytesIO(_get(SP500_WIKI, cache_hours)))
        for t in tables:
            if "Symbol" in t.columns:
                # Yahoo writes class shares with a dash, Wikipedia with a dot.
                return sorted({str(s).strip().upper().replace(".", "-")
                               for s in t["Symbol"].dropna()})
    except Exception as exc:
        log.warning("could not fetch S&P 500 list: %s", exc)
    raise RuntimeError("Could not fetch the S&P 500 constituent list.")
