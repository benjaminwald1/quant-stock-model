"""Does selling at the peak help?

Three exit rules on identical bars, chosen on the early years and scored on a
holdout that had no say:

  none   sell only when the rank falls through the exit (what ships)
  peak   also sell on a new 60-session high — the plain reading of "at its peak"
  band   also sell when the gain reaches the top of the model's own 80% forecast
         for that name's decile: the move it predicted has already happened

Selling winners is the most intuitive rule in trading and one of the easiest
ways to cap your upside, so it does not ship without this.
"""

from __future__ import annotations

import argparse
import sys
from statistics import mean

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")

from fund_replay import find_panel, replay                     # noqa: E402

SELECT = ["2022-08-31", "2023-08-31", "2024-08-30"]
HOLDOUT = ["2025-08-29", "2026-08-31"]
RULES = [("none", None), ("peak", "peak"), ("band", "band")]


def run(a, panel, tp, ends):
    yrs, trades, alph = [], 0, []
    for end in ends:
        r = replay(a.run, panel, months=12, budget=a.budget, sizing="equity",
                   exit_below=30.0, end=end, cost_bps=a.cost_bps, take_profit=tp)
        yrs.append(r["pnl_pct"])
        alph.append(r["pnl_pct"] - r["benchmark_pct"])
        trades += r["turnover_trades"]
    return yrs, trades, alph


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="20260831-171523-full15y")
    ap.add_argument("--budget", type=float, default=5000.0)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    a = ap.parse_args()
    panel = find_panel(a.run)

    for label, ends in (("SELECTION", SELECT), ("HOLDOUT — never used to choose", HOLDOUT)):
        print(f"\n{label}")
        print(f"{'rule':6s} " + " ".join(f"{e[:7]:>9s}" for e in ends)
              + f" {'worst':>8s} {'mean':>8s} {'trades':>7s} {'vs universe':>12s}")
        for name, tp in RULES:
            yrs, trades, alph = run(a, panel, tp, ends)
            print(f"{name:6s} " + " ".join(f"{v:8.2f}%" for v in yrs)
                  + f" {min(yrs):7.2f}% {mean(yrs):7.2f}% {trades:7d} {mean(alph):11.2f}%")
