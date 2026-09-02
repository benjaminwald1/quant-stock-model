"""Replay the paper fund's own rule over a past window, and test compounding.

The live fund places limit orders 1% under the previous close for names the
model ranks 90+, sells below 70, and holds at most ten. This runs exactly that
against real daily bars, so "what would it have made" is answered by the rule
rather than by an average.

Two sizing modes, because they differ once the fund is in profit:

  cash      a new slot is (idle cash / slots being opened) — what ships today
  equity    a new slot is (total equity / max positions), so gains are put
            back to work instead of only the cash left over

Fills use the session's actual low: an order at 1% below yesterday's close
fills only if the price genuinely traded there. Sells mark at the close of the
day the rank dropped. No costs or slippage are charged, which flatters both.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")

from qsm.web import stocks as stocks_mod                    # noqa: E402


def find_panel(run: str) -> str:
    """The cached OHLC panel that best covers this run's universe.

    Runs store closes only, but the limit rule needs each session's low to know
    whether an order would actually have filled. The live cache has the bars;
    pick whichever file covers the most of the run's names.
    """
    view = stocks_mod.load_view(Path("runs") / run)
    want = {str(c).upper() for c in view.rank.columns}
    best, best_hit = None, 0
    for p in Path("data/live").glob("*.parquet"):
        try:
            have = set(pd.read_parquet(p, columns=["ticker"])["ticker"].unique())
        except Exception:
            continue
        hit = len(want & {str(t).upper() for t in have})
        if hit > best_hit:
            best, best_hit = p, hit
    if best is None:
        raise SystemExit("no cached OHLC panel found under data/live")
    print(f"panel {best.name} covers {best_hit}/{len(want)} of the run's names")
    return str(best)


def replay(run: str, panel_path: str, months: int = 3, budget: float = 5000.0,
           sizing: str = "cash", enter_above: float = 90.0, exit_below: float = 70.0,
           max_positions: int = 10, dip_pct: float = 0.01, end: str | None = None,
           cost_bps: float = 10.0, take_profit: str | None = None,
           peak_lookback: int = 60) -> dict:
    view = stocks_mod.load_view(Path("runs") / run)
    px = pd.read_parquet(panel_path)
    px["date"] = pd.to_datetime(px["date"])

    close = px.pivot(index="date", columns="ticker", values="close").sort_index()
    low = px.pivot(index="date", columns="ticker", values="low").sort_index()
    high = px.pivot(index="date", columns="ticker", values="high").sort_index()

    # "At its peak" needs a definition. Two defensible ones:
    #   peak  the price has made a new `peak_lookback`-session high — the plain
    #         reading, a technical top with no model input
    #   band  the gain since entry has reached the top of the model's own 80%
    #         forecast for that name's decile, i.e. the move it predicted has
    #         already happened and it has no stated edge left
    rolling_max = close.rolling(peak_lookback, min_periods=peak_lookback // 2).max()
    bin_stats = None
    if take_profit == "band":
        from qsm.forecast import Z80, calibrate
        panel = pd.read_parquet(Path("runs") / run / "ticker_panel.parquet")
        cal = calibrate(view.signal, panel["fwd_ret"].unstack())
        nb = int(cal["n_bins"])
        bin_stats = (nb, Z80, [(float(b["mean"]), float(b["std"])) for b in cal["bins"]])

    ranks = view.rank.sort_index()
    dates = [d for d in close.index if d in ranks.index]
    if end:                                   # pin the finish so two runs align
        dates = [d for d in dates if d <= pd.Timestamp(end)]
    start = dates[-1] - pd.DateOffset(months=months)
    dates = [d for d in dates if d >= start]
    if len(dates) < 5:
        raise SystemExit("not enough overlapping sessions")

    cash = float(budget)
    equity_curve: list[float] = []
    cost_rate = cost_bps / 10_000.0
    costs_paid = 0.0
    holdings: dict[str, dict] = {}
    orders: dict[str, dict] = {}
    trades: list[dict] = []

    def equity(day):
        held = sum(h["shares"] * float(close.loc[day, h["ticker"]])
                   for h in holdings.values()
                   if h["ticker"] in close.columns and close.loc[day, h["ticker"]] == close.loc[day, h["ticker"]])
        return cash + held

    for i in range(1, len(dates)):
        day, prev = dates[i], dates[i - 1]
        # The signal is the previous close's, as the live fund uses it.
        rank_row = ranks.loc[prev]

        # ── exits ────────────────────────────────────────────────────────
        for t in list(holdings):
            r = rank_row.get(t)
            p = close.loc[day, t] if t in close.columns else float("nan")
            if p != p:
                continue

            peaked, why = False, ""
            if take_profit == "peak":
                rm = rolling_max.loc[day, t] if t in rolling_max.columns else float("nan")
                hi = high.loc[day, t] if t in high.columns else float("nan")
                if rm == rm and hi == hi and hi >= rm:
                    peaked, why = True, f"new {peak_lookback}-session high"
            elif take_profit == "band" and holdings[t].get("target"):
                hi = high.loc[day, t] if t in high.columns else float("nan")
                if hi == hi and hi >= holdings[t]["target"]:
                    peaked, why = True, "reached the model's 80% upper band"
                    p = holdings[t]["target"]      # filled at the target, not the high

            if peaked:
                h = holdings.pop(t)
                gross = h["shares"] * float(p)
                fee = gross * cost_rate
                costs_paid += fee
                cash += gross - fee
                trades.append({"date": str(day.date()), "action": "sell", "ticker": t,
                               "shares": h["shares"], "price": round(float(p), 4),
                               "pnl": round(h["shares"] * (float(p) - h["entry_price"]) - fee, 2),
                               "reason": why})
                continue

            if r is None or r != r or r < exit_below:
                h = holdings.pop(t)
                gross = h["shares"] * float(p)
                fee = gross * cost_rate
                costs_paid += fee
                cash += gross - fee
                trades.append({"date": str(day.date()), "action": "sell", "ticker": t,
                               "shares": h["shares"], "price": round(float(p), 4),
                               "pnl": round(h["shares"] * (float(p) - h["entry_price"]) - fee, 2),
                               "reason": f"rank {r:.0f} below {exit_below:.0f}" if r == r
                                         else "no longer scored"})

        # ── fills on resting orders ──────────────────────────────────────
        for t, o in list(orders.items()):
            if t in holdings:
                orders.pop(t, None)
                continue
            lo = low.loc[day, t] if t in low.columns else float("nan")
            if lo != lo:
                continue
            if float(lo) <= o["limit"]:
                fill = o["limit"]                      # filled at the limit, not the low
                shares = int(o["budget"] // (fill * (1 + cost_rate)))
                if shares > 0 and shares * fill * (1 + cost_rate) <= cash:
                    fee = shares * fill * cost_rate
                    costs_paid += fee
                    cash -= shares * fill + fee
                    tgt = None
                    if bin_stats:
                        nb, z, stats = bin_stats
                        rk = ranks.loc[prev].get(t)
                        if rk == rk:
                            bi = int(min(max(int(rk / 100 * nb), 0), nb - 1))
                            mu, sd = stats[bi]
                            tgt = fill * (1 + mu + z * sd)
                    holdings[t] = {"ticker": t, "shares": shares,
                                   "entry_price": round(fill, 4), "entry_date": str(day.date()),
                                   "target": tgt}
                    trades.append({"date": str(day.date()), "action": "buy", "ticker": t,
                                   "shares": shares, "price": round(fill, 4),
                                   "value": round(shares * fill, 2)})
                orders.pop(t, None)

        equity_curve.append(equity(day))

        # ── place new orders ─────────────────────────────────────────────
        orders.clear()                                  # orders are good for one session
        room = max_positions - len(holdings)
        if room > 0:
            cands = [(t, float(v)) for t, v in rank_row.items()
                     if v == v and v >= enter_above and t not in holdings
                     and t in close.columns and close.loc[prev, t] == close.loc[prev, t]]
            cands.sort(key=lambda kv: -kv[1])
            picks = cands[:room]
            if picks:
                slot = (cash / len(picks)) if sizing == "cash" else (equity(day) / max_positions)
                slot = min(slot, cash) if sizing == "equity" else slot
                for t, r in picks:
                    ref = float(close.loc[prev, t])
                    limit = ref * (1 - dip_pct)
                    if limit > slot:                    # a slot too small to buy one share
                        continue
                    orders[t] = {"ticker": t, "limit": round(limit, 4),
                                 "budget": round(slot, 2), "rank": r}

    # Equal-weight buy-and-hold of the same universe over the same window: the
    # honest "did picking help" comparison, since the model only claims to rank
    # names against each other and takes no view on the market itself.
    first, last = dates[0], dates[-1]
    # Only the names the model could actually have chosen from. Benchmarking a
    # 500-name picker against 5,600 tickers compares two different questions.
    pickable = {str(c) for c in ranks.columns}
    cols = [c for c in close.columns
            if c in pickable
            and close.loc[first, c] == close.loc[first, c]
            and close.loc[last, c] == close.loc[last, c]]
    bench_pct = float((close.loc[last, cols] / close.loc[first, cols] - 1).mean() * 100)

    end = dates[-1]
    value = equity(end)
    # Worst peak-to-trough on the equity curve: the number that decides whether
    # a strategy is survivable, not just whether it finishes up.
    peak, max_dd = -1e18, 0.0
    for v in equity_curve:
        peak = max(peak, v)
        if peak > 0:
            max_dd = min(max_dd, v / peak - 1)

    sells = [t for t in trades if t["action"] == "sell"]
    wins = [t for t in sells if t["pnl"] > 0]
    return {
        "sizing": sizing, "from": str(dates[0].date()), "to": str(end.date()),
        "sessions": len(dates), "budget": budget,
        "value": round(value, 2), "pnl": round(value - budget, 2),
        "pnl_pct": round((value / budget - 1) * 100, 2),
        "cash_left": round(cash, 2), "positions_held": len(holdings),
        "buys": sum(1 for t in trades if t["action"] == "buy"),
        "sells": len(sells),
        "realised": round(sum(t["pnl"] for t in sells), 2),
        "win_rate": round(len(wins) / len(sells), 3) if sells else None,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "costs_paid": round(costs_paid, 2), "cost_bps": cost_bps,
        "turnover_trades": sum(1 for t in trades if t["action"] in ("buy", "sell")),
        "benchmark_pct": round(bench_pct, 2),
        "universe": len(cols),
        "vs_benchmark_pct": round((value / budget - 1) * 100 - bench_pct, 2),
        "trades": trades,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="20260829-175555-sp500")
    ap.add_argument("--panel", default="auto")
    ap.add_argument("--months", type=int, default=3)
    ap.add_argument("--budget", type=float, default=5000.0)
    ap.add_argument("--end", default=None, help="last session to include, YYYY-MM-DD")
    ap.add_argument("--take-profit", default=None, choices=["peak", "band"])
    ap.add_argument("--show-trades", action="store_true")
    a = ap.parse_args()

    panel = find_panel(a.run) if a.panel == "auto" else a.panel
    rows = []
    for sizing in ("cash", "equity"):
        out = replay(a.run, panel, months=a.months, budget=a.budget, sizing=sizing,
                     end=a.end, take_profit=a.take_profit)
        rows.append(out)

    print(f"fund rule replayed on {a.run} · {rows[0]['from']} to {rows[0]['to']} "
          f"({rows[0]['sessions']} sessions) · ${a.budget:,.0f} budget\n")
    print(f"{'sizing':8s} {'end value':>12s} {'P&L':>10s} {'%':>8s} {'buys':>6s} "
          f"{'sells':>6s} {'win rate':>9s} {'vs universe':>12s}")
    for r in rows:
        wr = "—" if r["win_rate"] is None else f"{r['win_rate'] * 100:.0f}%"
        print(f"{r['sizing']:8s} {r['value']:12,.2f} {r['pnl']:10,.2f} "
              f"{r['pnl_pct']:7.2f}% {r['buys']:6d} {r['sells']:6d} {wr:>9s} "
              f"{r['vs_benchmark_pct']:11.2f}%")
    print(f"\nequal-weight buy and hold of the same {rows[0]['universe']:,} names: "
          f"{rows[0]['benchmark_pct']:+.2f}%")

    if a.show_trades:
        print("\nlast 15 trades (equity sizing):")
        for t in rows[1]["trades"][-15:]:
            print(" ", json.dumps(t))
