"""Does waiting for a dip beat buying at the open?

Simulates a limit order at each entry signal: place the order `x` percent below
the open. If that day's LOW reaches the limit, it fills at the limit price. If
it never dips, the order does not fill and the trade is missed.

That trade-off is the whole question. A better entry price is only worth having
if the trades you miss are not the ones that would have made the money — and
the names that never dip are, disproportionately, the ones that ran.
"""

from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "src")

from qsm.live import fetch_live                       # noqa: E402
from qsm.web.stocks import load_view                  # noqa: E402
from pathlib import Path                              # noqa: E402

HOLD = 5


def main(run: str = "20260829-175555-sp500", top: int = 10):
    d = Path("runs") / run
    view = load_view(d, "ensemble")
    ranks = view.rank.sort_index()

    tickers = list(view.signal.columns)
    panel = fetch_live(tickers, start="2024-01-01")
    o = panel.pivot(index="date", columns="ticker", values="open").sort_index()
    lo = panel.pivot(index="date", columns="ticker", values="low").sort_index()
    c = panel.pivot(index="date", columns="ticker", values="close").sort_index()

    dates = [d_ for d_ in ranks.index if d_ in o.index]
    signals = dates[:-(HOLD + 1)]
    print(f"{len(signals)} signal days, top {top} names each, {HOLD}-day hold\n")

    def run_rule(dip_pct: float | None):
        rets, fills, misses = [], 0, 0
        for i, sig in enumerate(signals):
            entry_day = dates[dates.index(sig) + 1]
            exit_day = dates[min(dates.index(sig) + 1 + HOLD, len(dates) - 1)]
            row = ranks.loc[sig].dropna().sort_values(ascending=False).head(top)
            for t in row.index:
                op = o.at[entry_day, t] if t in o.columns else np.nan
                low = lo.at[entry_day, t] if t in lo.columns else np.nan
                ex = c.at[exit_day, t] if t in c.columns else np.nan
                if not np.isfinite(op) or not np.isfinite(ex) or op <= 0:
                    continue
                if dip_pct is None:
                    rets.append(ex / op - 1); fills += 1
                    continue
                limit = op * (1 - dip_pct)
                if np.isfinite(low) and low <= limit:
                    rets.append(ex / limit - 1); fills += 1
                else:
                    misses += 1
        r = np.array(rets)
        return {
            "n": len(r), "fills": fills, "misses": misses,
            "fill_rate": fills / (fills + misses) if (fills + misses) else 0,
            "mean_pct": r.mean() * 100 if len(r) else np.nan,
            "total_pct": r.sum() * 100 if len(r) else np.nan,
            "win_rate": (r > 0).mean() * 100 if len(r) else np.nan,
        }

    print(f"{'rule':22s} {'fill rate':>10s} {'trades':>8s} {'mean/trade':>11s} "
          f"{'win rate':>9s} {'total':>10s}")
    base = run_rule(None)
    print(f"{'buy at the open':22s} {base['fill_rate']*100:9.1f}% {base['n']:8d} "
          f"{base['mean_pct']:10.3f}% {base['win_rate']:8.1f}% {base['total_pct']:9.1f}%")
    for dip in (0.005, 0.01, 0.02, 0.03):
        r = run_rule(dip)
        print(f"{'wait for -' + f'{dip*100:g}' + '% dip':22s} {r['fill_rate']*100:9.1f}% {r['n']:8d} "
              f"{r['mean_pct']:10.3f}% {r['win_rate']:8.1f}% {r['total_pct']:9.1f}%")

    print("\nfill rate = share of signals that ever reached the limit price;")
    print("total     = summed return, so missed trades simply contribute nothing.")




def capital_constrained(run: str = "20260829-175555-sp500", slots: int = 10,
                        pool: int = 40, budget: float = 5000.0):
    """The comparison that matches how the fund actually works.

    With a fixed budget you hold a fixed number of positions. So the question is
    not "does waiting miss trades" — it is "given more candidates than slots,
    does filling those slots with names that dipped beat taking the top-ranked
    ones at the open?" Missed names are replaced by the next candidate that did
    dip, so capital is never left idle.
    """
    d = Path("runs") / run
    view = load_view(d, "ensemble")
    ranks = view.rank.sort_index()
    panel = fetch_live(list(view.signal.columns), start="2024-01-01")
    o = panel.pivot(index="date", columns="ticker", values="open").sort_index()
    lo = panel.pivot(index="date", columns="ticker", values="low").sort_index()
    c = panel.pivot(index="date", columns="ticker", values="close").sort_index()

    dates = [x for x in ranks.index if x in o.index]
    # NON-OVERLAPPING holds only. Compounding a 5-day return computed on every
    # single signal day counts each day's move five times and inflates the
    # result into the millions. One entry per completed hold is the honest
    # sequence a real fund could actually have traded.
    signals = dates[: -(HOLD + 1)][::HOLD]

    def simulate(dip_pct):
        per_day = []
        for sig in signals:
            i = dates.index(sig)
            entry, exit_ = dates[i + 1], dates[min(i + 1 + HOLD, len(dates) - 1)]
            cands = ranks.loc[sig].dropna().sort_values(ascending=False).head(pool).index
            taken = []
            for t in cands:
                if len(taken) >= slots:
                    break
                op = o.at[entry, t] if t in o.columns else np.nan
                ex = c.at[exit_, t] if t in c.columns else np.nan
                if not np.isfinite(op) or not np.isfinite(ex) or op <= 0:
                    continue
                if dip_pct is None:
                    taken.append(ex / op - 1)
                else:
                    low = lo.at[entry, t] if t in lo.columns else np.nan
                    limit = op * (1 - dip_pct)
                    if np.isfinite(low) and low <= limit:
                        taken.append(ex / limit - 1)
            if taken:
                per_day.append(np.mean(taken))
        r = np.array(per_day)
        equity = float(np.prod(1 + r))
        years = len(r) * HOLD / 252
        ann = (equity ** (1 / years) - 1) * 100 if years > 0 and equity > 0 else float("nan")
        return {"days": len(r), "mean": r.mean() * 100, "final": budget * equity,
                "ann": ann, "win": (r > 0).mean() * 100}

    print(f"\n\ncapital-constrained, NON-OVERLAPPING holds: {slots} slots from the top "
          f"{pool} ranked, ${budget:,.0f}\n")
    print(f"{'rule':22s} {'holds':>6s} {'mean/hold':>10s} {'win rate':>9s} "
          f"{'ann. return':>12s} {'final':>12s}")
    base = simulate(None)
    print(f"{'buy at the open':22s} {base['days']:6d} {base['mean']:9.3f}% "
          f"{base['win']:8.1f}% {base['ann']:11.1f}% {base['final']:11,.0f}")
    for dip in (0.005, 0.01, 0.02):
        r = simulate(dip)
        better = "  <-- better" if r["final"] > base["final"] else ""
        print(f"{'wait for -' + f'{dip*100:g}' + '% dip':22s} {r['days']:6d} {r['mean']:9.3f}% "
              f"{r['win']:8.1f}% {r['ann']:11.1f}% {r['final']:11,.0f}{better}")


if __name__ == "__main__":
    main()
    capital_constrained()
