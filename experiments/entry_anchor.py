"""What should the limit price be measured against?

The fund currently anchors to the live price at 9:30, which is the opening
print — the single noisiest quote of the session. A name that gaps up 3% still
"dips" 1% below its own inflated open, so the rule buys strength and calls it a
discount.

Compares anchors on identical signals, non-overlapping holds, fixed slots:

  open x (1-d)         what it does today
  prev_close x (1-d)   a level set before the open, unaffected by the gap
  prev_close           fills only at or below yesterday's close
  close                ignore the open entirely and buy at the bell
"""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "src")

from qsm.live import fetch_live                       # noqa: E402
from qsm.web.stocks import load_view                  # noqa: E402

HOLD = 5
SLOTS = 10
POOL = 40
BUDGET = 5000.0


def main(run: str = "20260829-175555-sp500"):
    d = Path("runs") / run
    view = load_view(d, "ensemble")
    ranks = view.rank.sort_index()
    panel = fetch_live(list(view.signal.columns), start="2024-01-01")
    o = panel.pivot(index="date", columns="ticker", values="open").sort_index()
    lo = panel.pivot(index="date", columns="ticker", values="low").sort_index()
    c = panel.pivot(index="date", columns="ticker", values="close").sort_index()

    dates = [x for x in ranks.index if x in o.index]
    signals = dates[: -(HOLD + 1)][::HOLD]          # non-overlapping

    def simulate(mode: str, dip: float = 0.01):
        per_hold, fills, offered = [], 0, 0
        for sig in signals:
            i = dates.index(sig)
            entry, exit_ = dates[i + 1], dates[min(i + 1 + HOLD, len(dates) - 1)]
            prev = dates[i]                          # signal day = day before entry
            cands = ranks.loc[sig].dropna().sort_values(ascending=False).head(POOL).index
            taken = []
            for t in cands:
                if len(taken) >= SLOTS:
                    break
                offered += 1
                op = o.at[entry, t] if t in o.columns else np.nan
                low = lo.at[entry, t] if t in lo.columns else np.nan
                ex = c.at[exit_, t] if t in c.columns else np.nan
                pc = c.at[prev, t] if t in c.columns else np.nan
                if not all(np.isfinite(v) for v in (op, ex)) or op <= 0:
                    continue

                if mode == "open_limit":
                    limit = op * (1 - dip)
                elif mode == "prevclose_limit":
                    if not np.isfinite(pc):
                        continue
                    limit = pc * (1 - dip)
                elif mode == "at_prevclose":
                    if not np.isfinite(pc):
                        continue
                    limit = pc
                elif mode == "close":
                    fill = c.at[entry, t]
                    if np.isfinite(fill) and fill > 0:
                        taken.append(ex / fill - 1); fills += 1
                    continue
                else:                                 # buy at the open
                    taken.append(ex / op - 1); fills += 1
                    continue

                # A limit above the open fills immediately at the open.
                if limit >= op:
                    taken.append(ex / op - 1); fills += 1
                elif np.isfinite(low) and low <= limit:
                    taken.append(ex / limit - 1); fills += 1
            if taken:
                per_hold.append(np.mean(taken))
        r = np.array(per_hold)
        equity = float(np.prod(1 + r))
        years = len(r) * HOLD / 252
        return {"holds": len(r), "mean": r.mean() * 100, "win": (r > 0).mean() * 100,
                "final": BUDGET * equity,
                "ann": (equity ** (1 / years) - 1) * 100 if years > 0 else float("nan"),
                "fill_rate": fills / offered * 100 if offered else 0}

    print(f"{len(signals)} non-overlapping holds, {SLOTS} slots from top {POOL}, "
          f"${BUDGET:,.0f}\n")
    print(f"{'anchor':28s} {'fills':>7s} {'mean/hold':>10s} {'win':>7s} "
          f"{'ann':>9s} {'final':>11s}")
    rows = [
        ("buy at the open", "market", 0),
        ("open x 0.99  (current)", "open_limit", 0.01),
        ("prev close x 0.99", "prevclose_limit", 0.01),
        ("prev close x 0.98", "prevclose_limit", 0.02),
        ("at or below prev close", "at_prevclose", 0),
        ("buy at the close", "close", 0),
    ]
    base = None
    for label, mode, dip in rows:
        r = simulate(mode, dip)
        if base is None:
            base = r["final"]
        mark = "  <--" if r["final"] > base * 1.02 else ""
        print(f"{label:28s} {r['fill_rate']:6.1f}% {r['mean']:9.3f}% {r['win']:6.1f}% "
              f"{r['ann']:8.1f}% {r['final']:10,.0f}{mark}")


if __name__ == "__main__":
    main()
