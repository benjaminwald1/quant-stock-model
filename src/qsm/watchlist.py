"""A persistent list of tickers the user wants to keep an eye on.

Stored server-side as JSON rather than in browser storage, so it survives a
cleared cache, a different browser, or a machine restart — and so the server can
prime quotes for it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from .config import DATA_DIR

log = logging.getLogger(__name__)

WATCHLIST_PATH = DATA_DIR / "watchlist.json"
MAX_ITEMS = 100
DEFAULT = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]


def _read() -> dict:
    if not WATCHLIST_PATH.exists():
        return {"tickers": list(DEFAULT), "auto": [], "updated": None}
    try:
        raw = json.loads(WATCHLIST_PATH.read_text())
        tickers = [str(t).upper().strip() for t in raw.get("tickers", []) if str(t).strip()]
        auto = [str(t).upper().strip() for t in raw.get("auto", []) if str(t).strip()]
        return {"tickers": tickers, "auto": auto, "updated": raw.get("updated")}
    except Exception as exc:                       # a corrupt file must not brick the page
        log.warning("watchlist unreadable (%s); starting fresh", exc)
        return {"tickers": list(DEFAULT), "auto": [], "updated": None}


def _write(tickers: list[str], auto: list[str] | None = None) -> dict:
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if auto is None:
        auto = _read().get("auto", [])
    payload = {"tickers": tickers, "auto": auto,
               "updated": datetime.now().isoformat(timespec="seconds")}
    WATCHLIST_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def get() -> dict:
    return _read()


def add(ticker: str) -> dict:
    t = str(ticker).upper().strip()
    if not t:
        raise ValueError("Empty ticker.")
    if len(t) > 16:
        raise ValueError("That does not look like a ticker.")
    current = _read()["tickers"]
    if t in current:
        return {"tickers": current, "already": True}
    if len(current) >= MAX_ITEMS:
        raise ValueError(f"Watchlist is full ({MAX_ITEMS} maximum).")
    return _write(current + [t])


def remove(ticker: str) -> dict:
    t = str(ticker).upper().strip()
    current = _read()["tickers"]
    return _write([x for x in current if x != t])


def replace(tickers: list[str]) -> dict:
    seen, clean = set(), []
    for t in tickers[:MAX_ITEMS]:
        u = str(t).upper().strip()
        if u and u not in seen:
            seen.add(u)
            clean.append(u)
    return _write(clean)


def follow(tickers: list[str]) -> dict:
    """Add names on the model's behalf — each one only ever once.

    The model owns its picks, but it does not own the list. Every name it has
    already put here is remembered, so a name you delete stays deleted instead
    of reappearing on the next tick and turning removal into an argument you
    cannot win.
    """
    data = _read()
    current = list(data["tickers"])
    offered = set(data.get("auto") or [])
    added = []
    for t in tickers:
        u = str(t).upper().strip()
        if not u or u in current or u in offered:
            continue
        if len(current) >= MAX_ITEMS:
            break
        current.append(u)
        offered.add(u)
        added.append(u)
    if added:
        _write(current, auto=sorted(offered))
    return {"tickers": current, "added": added}
