"""The fund sits ~37% in cash. What is that costing?

Hedging showed the return here is market exposure, not stock selection. If that
is true then idle cash is the single largest drag in the design: the strategy is
only invested in the thing that pays it about two thirds of the time.

Parks unfilled cash in the index between fills, selling it down first whenever a
limit order hits so no trade is missed for lack of cash. Charged normal costs
both ways. Same selection/holdout split as everything else.
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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="20260831-171523-full15y")
    ap.add_argument("--budget", type=float, default=5000.0)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    a = ap.parse_args()
    panel = find_panel(a.run)

    for label, ends in (("SELECTION", SELECT), ("HOLDOUT", HOLDOUT)):
        print(f"\n{label}")
        print(f"{'cash':>10s} " + " ".join(f"{e[:7]:>9s}" for e in ends)
              + f" {'mean':>8s} {'worst yr':>9s} {'worst DD':>9s}")
        for park, name in ((False, "idle"), (True, "in SPY")):
            yrs, dds = [], []
            for end in ends:
                r = replay(a.run, panel, months=12, budget=a.budget, sizing="equity",
                           exit_below=30.0, max_positions=5, dip_pct=0.005,
                           end=end, cost_bps=a.cost_bps, park_cash=park)
                yrs.append(r["pnl_pct"])
                dds.append(r["max_drawdown_pct"])
            print(f"{name:>10s} " + " ".join(f"{v:8.2f}%" for v in yrs)
                  + f" {mean(yrs):7.2f}% {min(yrs):8.2f}% {min(dds):8.2f}%")
