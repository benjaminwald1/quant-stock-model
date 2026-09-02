"""A tracked paper fund: fixed budget, the model's picks, real prices.

Distinct from the portfolio, which records whatever you tell it. This one is
run entirely by the model: it takes a budget, allocates it across the model's
own picks at the entry date, and is then marked to market against real prices
with no further input. That makes it a clean test of the strategy rather than
of your trading.

The fund is created *before* its entry date has traded, so it is stored as a
plan and filled from real prices once that session exists. Filling it at
today's price and back-dating the entry would be the most flattering possible
lie a tool like this could tell.
"""

from __future__ import annotations

import datetime as dt
import json
import logging

from .config import DATA_DIR

log = logging.getLogger(__name__)

FUND_PATH = DATA_DIR / "fund.json"


def _write(state: dict) -> dict:
    FUND_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = FUND_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(FUND_PATH)
    return state


def get() -> dict | None:
    if not FUND_PATH.exists():
        return None
    try:
        return json.loads(FUND_PATH.read_text())
    except Exception as exc:
        log.warning("fund unreadable (%s)", exc)
        return None


def create(budget: float, picks: list[dict], entry_date: str, exit_date: str,
           run: str, benchmark: str = "SPY") -> dict:
    """Plan an allocation. Nothing is priced until the entry session trades."""
    if budget <= 0:
        raise ValueError("Budget must be greater than zero.")
    usable = [p for p in picks if p.get("weight", 0) > 0]
    if not usable:
        raise ValueError("The model holds no long positions to allocate.")
    total = sum(p["weight"] for p in usable)

    return _write({
        "budget": round(float(budget), 2),
        "run": run,
        "benchmark": benchmark,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "status": "planned",
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "plan": [{"ticker": p["ticker"],
                  "weight": round(p["weight"] / total, 6),
                  "rank": p.get("rank")} for p in usable],
        "filled": None,
    })


def mark(closes, benchmark_closes) -> dict | None:
    """Fill the plan if its entry session exists, then mark it to market.

    ``closes`` is a date x ticker frame of real prices.
    """
    import pandas as pd

    state = get()
    if not state:
        return None

    entry = pd.Timestamp(state["entry_date"])
    sessions = closes.index
    if entry not in sessions:
        # Entry has not traded yet: report the plan, priced at nothing.
        state["status"] = "awaiting entry"
        state["message"] = (f"Entry is {state['entry_date']}; the market has not traded that "
                            f"session yet (last close {sessions[-1].date()}).")
        return state

    if not state.get("filled"):
        row = closes.loc[entry]
        filled, spent = [], 0.0
        for p in state["plan"]:
            price = row.get(p["ticker"])
            if price is None or price != price or price <= 0:
                continue
            shares = int((state["budget"] * p["weight"]) // price)
            if shares <= 0:
                continue
            filled.append({**p, "shares": shares, "entry_price": round(float(price), 4),
                           "cost": round(shares * float(price), 2)})
            spent += shares * float(price)
        state["filled"] = filled
        state["invested"] = round(spent, 2)
        state["cash"] = round(state["budget"] - spent, 2)
        state["status"] = "open"
        _write(state)

    last = sessions[-1]
    row = closes.loc[last]
    value = state["cash"]
    marks = []
    for p in state["filled"]:
        price = row.get(p["ticker"])
        price = float(price) if price is not None and price == price else None
        val = price * p["shares"] if price else None
        if val:
            value += val
        marks.append({**p, "price": price, "value": round(val, 2) if val else None,
                      "pnl": round(val - p["cost"], 2) if val else None,
                      "pnl_pct": round((val / p["cost"] - 1) * 100, 3) if val else None})

    exit_ts = pd.Timestamp(state["exit_date"])
    state["marks"] = marks
    state["as_of"] = str(last.date())
    state["value"] = round(value, 2)
    state["pnl"] = round(value - state["budget"], 2)
    state["pnl_pct"] = round((value / state["budget"] - 1) * 100, 3)
    state["closed"] = last >= exit_ts
    if state["closed"]:
        state["status"] = "closed"

    # Same money in the benchmark over the same days.
    try:
        b_in = float(benchmark_closes.loc[entry])
        b_out = float(benchmark_closes.loc[last])
        shares = int(state["budget"] // b_in)
        b_value = shares * b_out + (state["budget"] - shares * b_in)
        state["benchmark_result"] = {
            "ticker": state["benchmark"], "shares": shares,
            "entry_price": round(b_in, 2), "price": round(b_out, 2),
            "value": round(b_value, 2), "pnl": round(b_value - state["budget"], 2),
            "pnl_pct": round((b_value / state["budget"] - 1) * 100, 3),
        }
        state["vs_benchmark"] = round(state["pnl"] - state["benchmark_result"]["pnl"], 2)
    except Exception:
        state["benchmark_result"] = None
    return state


def clear() -> None:
    FUND_PATH.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Autopilot: signal-driven entry and exit
# --------------------------------------------------------------------------
def rebalance(state: dict, closes, ranks, enter_above: float = 90.0,
              exit_below: float = 30.0, max_positions: int = 5,
              live_prices: dict | None = None, market_open: bool = True,
              dip_pct: float = 0.005, reference_prices: dict | None = None,
              sizing: str = "equity", take_profit: str = "off",
              peak_lookback: int = 60) -> dict:
    """Act on the model's signal for one session.

    Hold a name while the model still ranks it highly; sell when it drops out.
    The two thresholds differ on purpose — buying above 90 but only selling
    below 30 puts a wide deadband between them. With a single threshold a name
    hovering at the line gets bought and sold on alternate days, and the
    turnover eats more than the signal is worth.

    The exit was 70 and is now 30, chosen on 2022-2024 and confirmed on
    2025-2026 (experiments/tune_exit.py). Holding through a rank dip beat
    reacting to it: about a quarter of the trades, and 12.30% a year on the
    holdout against 3.36%. The deadband is doing the work, not the entry line.

    Decisions are made once per session because that is the model's frequency:
    it consumes daily bars and emits one score per name per day. There is no
    intraday view to act on, so the fund does not pretend to have one.
    """
    if not state or closes.empty or ranks.empty:
        return state

    # Trade only while the market is actually open, at live prices. Filling at
    # the last close when the exchange is shut is not a trade anyone could have
    # made — it is a backfill wearing a timestamp.
    if not market_open:
        state["status"] = "waiting for the open"
        state["blocked_reason"] = "market closed"
        return state
    if not live_prices:
        state["status"] = "waiting for prices"
        state["blocked_reason"] = "no live quotes available"
        return state

    state.pop("blocked_reason", None)
    session = closes.index[-1]
    # Anchor the limit to the PREVIOUS CLOSE, not the live price at 9:30.
    # The opening print is the noisiest quote of the day: a name that gaps up
    # 3% still sits 1% under its own inflated open, so an open-anchored rule
    # buys strength and books it as a discount. Measured over 133 holds
    # (experiments/entry_anchor.py): prev-close anchoring returned 1.535% per
    # hold against 1.437% anchored to the open, and 1.265% buying at the open.
    prev_close = reference_prices or {}
    if not prev_close and len(closes) >= 1:
        prev_close = {t: float(v) for t, v in closes.iloc[-1].items() if v == v}
    rank_row = ranks.loc[session] if session in ranks.index else ranks.iloc[-1]
    stamp = dt.datetime.now().isoformat(timespec="seconds")

    holdings = {h["ticker"]: h for h in (state.get("holdings") or [])}
    cash = float(state.get("cash", state["budget"]))
    log = state.setdefault("trades", [])

    def price_of(t):
        v = live_prices.get(t)
        return float(v) if v is not None and v == v and v > 0 else None

    # ── take profit at a peak, when asked ────────────────────────────────
    # Off by default, and measured before being offered at all
    # (experiments/peak_exit.py): selling on a new 60-session high scored 5.49%
    # a year on the holdout against 12.30% for letting the rank rule decide, and
    # was worse in every window on both halves of the split. Selling at the top
    # of the model's own 80% band was worse again. It is the most intuitive rule
    # in trading and it caps the winners while the losers keep running.
    for ticker in list(holdings):
        if take_profit != "peak":
            break
        px = price_of(ticker)
        if px is None or ticker not in closes.columns:
            continue
        hist = closes[ticker].dropna()
        # Enough history to know what a peak is, scaled to the window asked
        # for. A flat floor silently disabled the rule for short lookbacks.
        if len(hist) < max(3, peak_lookback // 2):
            continue
        peak = float(hist.iloc[-peak_lookback:].max())
        if px < peak:
            continue
        h = holdings.pop(ticker)
        proceeds = h["shares"] * px
        cash += proceeds
        log.append({
            "date": stamp, "action": "sell", "ticker": ticker,
            "shares": h["shares"], "price": round(px, 4),
            "value": round(proceeds, 2),
            "pnl": round(proceeds - h["shares"] * h["entry_price"], 2),
            "reason": f"at a {peak_lookback}-session high of {peak:,.2f}",
        })

    # ── exits: the model no longer rates it ──────────────────────────────
    for ticker in list(holdings):
        r = rank_row.get(ticker)
        px = price_of(ticker)
        if px is None:
            continue
        if r is None or r != r or r < exit_below:
            h = holdings.pop(ticker)
            proceeds = h["shares"] * px
            cash += proceeds
            log.append({
                "date": stamp, "action": "sell", "ticker": ticker,
                "shares": h["shares"], "price": round(px, 4),
                "value": round(proceeds, 2),
                "pnl": round(proceeds - h["shares"] * h["entry_price"], 2),
                "reason": f"rank {r:.0f} fell below {exit_below:.0f}" if r == r
                          else "no longer scored",
            })

    # ── resting limit orders: fill any whose price has come down ─────────
    # Measured on this universe (experiments/limit_entry.py): with a fixed
    # number of slots and more candidates than slots, waiting for a dip beat
    # buying at the open — 1.83% vs 1.27% per hold. Missed names are simply
    # replaced by the next candidate that does dip, so capital is not idle.
    orders = state.get("orders") or []
    still_open = []
    for order in orders:
        t = order["ticker"]
        px = price_of(t)
        if t in holdings or px is None:
            continue
        if px <= order["limit"]:
            shares = int(order["budget"] // px)
            if shares > 0 and shares * px <= cash:
                spend = shares * px
                cash -= spend
                holdings[t] = {"ticker": t, "shares": shares,
                               "entry_price": round(px, 4), "entry_date": stamp}
                log.append({
                    "date": stamp, "action": "buy", "ticker": t, "shares": shares,
                    "price": round(px, 4), "value": round(spend, 2),
                    "reason": f"limit hit — dipped to {px:.2f}, "
                              f"{(1 - px / order['reference']) * 100:.1f}% below {order['reference']:.2f}",
                })
                continue
        # Re-anchor before carrying it forward. An order placed on Monday is
        # priced off Monday's close; by Thursday that is not "the previous
        # close" any more, and it also ignores any change to the configured
        # depth — which is how the book ended up running a mix of 1.0% and
        # 0.5% orders at the same time.
        ref_now = prev_close.get(t)
        if ref_now:
            order = {**order, "reference": round(float(ref_now), 4),
                     "reference_kind": "previous close",
                     "limit": round(float(ref_now) * (1 - dip_pct), 4),
                     "gap_pct": round((px / float(ref_now) - 1) * 100, 2)}
        still_open.append(order)

    # ── place limit orders for names the model rates but we do not hold ──
    room = max_positions - len(holdings) - len(still_open)
    if room > 0:
        held_or_pending = set(holdings) | {o["ticker"] for o in still_open}
        candidates = [(t, float(v)) for t, v in rank_row.items()
                      if v == v and v >= enter_above and t not in held_or_pending
                      and price_of(t) is not None]
        candidates.sort(key=lambda kv: -kv[1])
        picks = candidates[:room]
        if picks:
            # Size a slot against everything the fund is worth, not just the
            # cash lying idle, so profit is put back to work: a fund that has
            # grown to $6,000 opens $600 positions, not $500 ones forever.
            # Capped by cash, because a slot it cannot pay for is not a slot.
            per_slot = cash / max(1, len(picks) + len(still_open))
            if sizing == "equity":
                held_value = sum(
                    h["shares"] * (price_of(h["ticker"]) or h["entry_price"])
                    for h in holdings.values())
                per_slot = min((cash + held_value) / max_positions, cash)
            budget_each = per_slot
            for ticker, r in picks:
                px = price_of(ticker)
                ref = prev_close.get(ticker, px)
                # A $500 slot cannot buy an $830 share, and an order that can
                # never fill just blocks a slot. Skip it and let the next
                # affordable candidate take the place.
                if px > budget_each:
                    log.append({
                        "date": stamp, "action": "skip", "ticker": ticker,
                        "shares": 0, "price": round(px, 4),
                        "reason": f"share price ${px:,.0f} exceeds the ${budget_each:,.0f} slot",
                    })
                    continue
                limit = ref * (1 - dip_pct)
                still_open.append({
                    "ticker": ticker,
                    "reference": round(ref, 4),
                    "reference_kind": "previous close",
                    "live_at_placement": round(px, 4),
                    "limit": round(limit, 4),
                    "gap_pct": round((px / ref - 1) * 100, 2) if ref else None,
                    "budget": round(budget_each, 2), "rank": round(r, 1),
                    "placed": stamp,
                })
    state["orders"] = still_open
    state["dip_pct"] = dip_pct

    value = cash + sum(h["shares"] * (price_of(h["ticker"]) or h["entry_price"])
                       for h in holdings.values())
    state["holdings"] = list(holdings.values())
    state["cash"] = round(cash, 2)
    state["value"] = round(value, 2)
    state["pnl"] = round(value - state["budget"], 2)
    state["pnl_pct"] = round((value / state["budget"] - 1) * 100, 3)
    state["last_rebalance"] = stamp
    state["status"] = "trading"
    state.setdefault("history", []).append({"date": stamp, "value": round(value, 2)})
    state["trades"] = log[-200:]
    return _write(state)


def start_autopilot(budget: float, run: str, benchmark: str = "SPY") -> dict:
    """Begin an autopilot fund: all cash, no positions, no entry date."""
    if budget <= 0:
        raise ValueError("Budget must be greater than zero.")
    return _write({
        "budget": round(float(budget), 2),
        "run": run,
        "benchmark": benchmark,
        "mode": "autopilot",
        "status": "autopilot",
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "cash": round(float(budget), 2),
        "value": round(float(budget), 2),
        "pnl": 0.0,
        "pnl_pct": 0.0,
        "holdings": [],
        "trades": [],
        "history": [],
        "last_rebalance": None,
        "live_only": True,
    })
