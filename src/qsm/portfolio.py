"""A paper-trading ledger: what you hold, how much, and what it is worth.

Positions are recorded with the price and timestamp at the moment they were
added, so profit and loss is measured against a real entry rather than a
retrospective one. Nothing here places an order — it is a record of what you
say you own.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from .config import DATA_DIR

log = logging.getLogger(__name__)

PORTFOLIO_PATH = DATA_DIR / "portfolio.json"
MAX_POSITIONS = 200


def _read() -> list[dict]:
    if not PORTFOLIO_PATH.exists():
        return []
    try:
        raw = json.loads(PORTFOLIO_PATH.read_text())
        return list(raw.get("positions", []))
    except Exception as exc:
        log.warning("portfolio unreadable (%s); starting empty", exc)
        return []


def _write(positions: list[dict]) -> list[dict]:
    PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PORTFOLIO_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(
        {"positions": positions, "updated": datetime.now().isoformat(timespec="seconds")},
        indent=2))
    tmp.replace(PORTFOLIO_PATH)
    return positions


def get() -> list[dict]:
    return _read()


def buy(ticker: str, quantity: float, price: float, note: str = "") -> list[dict]:
    """Add or add to a position, recording entry price and time.

    Buying more of something already held blends the entry price by quantity,
    which is what an average cost basis means — keeping the first price would
    misstate the P&L on every subsequent purchase.
    """
    t = str(ticker).upper().strip()
    if not t:
        raise ValueError("Empty ticker.")
    quantity = float(quantity)
    price = float(price)
    if quantity <= 0:
        raise ValueError("Quantity must be greater than zero.")
    if price <= 0:
        raise ValueError("Price must be greater than zero.")

    positions = _read()
    for p in positions:
        if p["ticker"] == t:
            total_qty = p["quantity"] + quantity
            p["entry_price"] = round(
                (p["entry_price"] * p["quantity"] + price * quantity) / total_qty, 6)
            p["quantity"] = round(total_qty, 6)
            p["last_added"] = datetime.now().isoformat(timespec="seconds")
            return _write(positions)

    if len(positions) >= MAX_POSITIONS:
        raise ValueError(f"Portfolio is full ({MAX_POSITIONS} positions).")
    positions.append({
        "ticker": t,
        "quantity": round(quantity, 6),
        "entry_price": round(price, 6),
        "entry_at": datetime.now().isoformat(timespec="seconds"),
        "note": str(note)[:200],
    })
    return _write(positions)


def sell(ticker: str, quantity: float | None = None) -> list[dict]:
    """Reduce or close a position. No quantity means close it entirely."""
    t = str(ticker).upper().strip()
    positions = _read()
    if quantity is None:
        return _write([p for p in positions if p["ticker"] != t])

    quantity = float(quantity)
    out = []
    for p in positions:
        if p["ticker"] != t:
            out.append(p)
            continue
        remaining = round(p["quantity"] - quantity, 6)
        if remaining > 1e-9:
            p["quantity"] = remaining
            out.append(p)
    return _write(out)


def clear() -> list[dict]:
    return _write([])


BENCHMARKS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
    "VT": "World",
    "AGG": "US bonds",
}


def analytics(rows: list[dict], since: str | None = None,
              benchmarks: list[str] | None = None) -> dict:
    """Portfolio statistics and a like-for-like comparison against funds.

    The comparison is deliberately measured over the *same* window as the
    portfolio — from its earliest entry to the last close. Comparing a
    three-day-old portfolio against a fund's year-to-date return would be
    meaningless, and flattering about whichever happened to be up.
    """
    import numpy as np

    from .live import fetch_live

    held = [r for r in rows if r.get("value") is not None]
    if not held:
        return {"available": False, "reason": "no priced positions"}

    entries = [r.get("entry_at", "")[:10] for r in rows if r.get("entry_at")]
    start = since or (min(entries) if entries else None)
    if not start:
        return {"available": False, "reason": "no entry dates recorded"}

    value = sum(r["value"] for r in held)
    cost = sum(r["cost"] for r in held)
    weights = {r["ticker"]: r["value"] / value for r in held} if value else {}

    best = max(held, key=lambda r: r.get("pnl_pct") or -1e9)
    worst = min(held, key=lambda r: r.get("pnl_pct") or 1e9)
    top_w = max(weights.values()) if weights else 0.0
    # Herfindahl index: 1 / HHI is the "effective number of positions", which
    # says more about concentration than the raw count does.
    hhi = sum(w * w for w in weights.values()) or 1.0

    bench = {}
    syms = [b for b in (benchmarks or ["SPY", "QQQ"]) if b]
    try:
        # A portfolio opened today has no elapsed sessions to compare. Rather
        # than showing an empty panel, widen the window and label it as market
        # context — never as this portfolio's performance.
        import datetime as _dt

        fetch_start = min(start, str(_dt.date.fromisoformat(start) - _dt.timedelta(days=45)))
        panel = fetch_live(syms + [r["ticker"] for r in held], start=fetch_start)
        close = panel.pivot(index="date", columns="ticker", values="close").sort_index()
        held_window = close[close.index >= start]
        context_only = len(held_window) < 2
        if context_only:
            close = close.tail(22)          # ~1 month of context
        else:
            close = held_window
        if len(close) >= 2:
            first, last = close.iloc[0], close.iloc[-1]
            for b in syms:
                if b in close.columns and first.get(b, np.nan) > 0:
                    bench[b] = {
                        "name": BENCHMARKS.get(b, b),
                        "return_pct": round(float(last[b] / first[b] - 1) * 100, 3),
                    }
            # Portfolio return over the same window, at today's weights.
            port_ret = 0.0
            for t, w in weights.items():
                if t in close.columns and first.get(t, np.nan) > 0:
                    port_ret += w * float(last[t] / first[t] - 1)
            bench["_portfolio_same_window"] = None if context_only else round(port_ret * 100, 3)
            bench["_context_only"] = context_only
            bench["_window"] = {"from": str(close.index[0].date()),
                                "to": str(close.index[-1].date()),
                                "sessions": int(len(close) - 1)}
    except Exception as exc:  # pragma: no cover - network dependent
        log.warning("benchmark comparison failed: %s", exc)

    return {
        "available": True,
        "since": start,
        "value": round(value, 2),
        "cost": round(cost, 2),
        "pnl": round(value - cost, 2),
        "pnl_pct": round((value / cost - 1) * 100, 3) if cost else None,
        "positions": len(held),
        "effective_positions": round(1 / hhi, 2),
        "largest_weight_pct": round(top_w * 100, 2),
        "best": {"ticker": best["ticker"], "pnl_pct": best.get("pnl_pct")},
        "worst": {"ticker": worst["ticker"], "pnl_pct": worst.get("pnl_pct")},
        "weights": {k: round(v * 100, 2) for k, v in
                    sorted(weights.items(), key=lambda kv: -kv[1])},
        "benchmarks": bench,
    }


def allocate(balance: float, picks: list[dict], max_weight: float = 0.25,
             whole_shares: bool = True) -> dict:
    """Show how a given balance would be split across the model's picks.

    This describes what the backtested rule does with a notional sum. It is a
    simulation of a strategy whose own measured net Sharpe is below buy-and-hold
    — not a suggestion to place these trades.
    """
    balance = float(balance)
    if balance <= 0:
        raise ValueError("Balance must be greater than zero.")
    usable = [p for p in picks if p.get("price")]
    if not usable:
        return {"balance": balance, "rows": [], "invested": 0.0, "cash": balance}

    raw = {p["ticker"]: max(0.0, float(p.get("weight") or 0.0)) for p in usable}
    total = sum(raw.values())
    if total <= 0:                       # no weights supplied: equal split
        raw = {k: 1.0 for k in raw}
        total = float(len(raw))
    weights = {k: min(v / total, max_weight) for k, v in raw.items()}
    norm = sum(weights.values()) or 1.0
    weights = {k: v / norm for k, v in weights.items()}

    rows, invested = [], 0.0
    for p in usable:
        t = p["ticker"]
        target = balance * weights[t]
        price = float(p["price"])
        shares = int(target // price) if whole_shares else round(target / price, 4)
        spend = shares * price
        invested += spend
        rows.append({
            "ticker": t, "price": round(price, 4),
            "weight_pct": round(weights[t] * 100, 2),
            "target_value": round(target, 2),
            "shares": shares, "spend": round(spend, 2),
            "rank": p.get("rank"),
        })
    rows.sort(key=lambda r: -r["spend"])
    return {
        "balance": round(balance, 2),
        "invested": round(invested, 2),
        "cash": round(balance - invested, 2),
        "whole_shares": whole_shares,
        "rows": rows,
    }
