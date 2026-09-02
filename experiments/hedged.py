"""Does hedging the market exposure help?

The model is calibrated on return *relative to the universe* — calibrate()
demeans every date — and the backtest that validated it was long/short. The
fund is long-only, so it carries a market bet the model never made. This shorts
`hedge` x the long book in SPY, pays 100bps a year to borrow it and normal
costs to trade it, and scores the result on the same split as everything else.

Return is not the point here; a hedge gives up upside in a rising market by
construction. The question is whether the model's stock picking survives on its
own once the market is taken out of it.
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
        print(f"{'hedge':>6s} " + " ".join(f"{e[:7]:>9s}" for e in ends)
              + f" {'mean':>8s} {'worst yr':>9s} {'worst DD':>9s}")
        for hedge in (0.0, 0.5, 1.0):
            yrs, dds = [], []
            for end in ends:
                r = replay(a.run, panel, months=12, budget=a.budget, sizing="equity",
                           exit_below=30.0, max_positions=5, dip_pct=0.005,
                           end=end, cost_bps=a.cost_bps, hedge=hedge)
                yrs.append(r["pnl_pct"])
                dds.append(r["max_drawdown_pct"])
            print(f"{hedge:6.0%} " + " ".join(f"{v:8.2f}%" for v in yrs)
                  + f" {mean(yrs):7.2f}% {min(yrs):8.2f}% {min(dds):8.2f}%")
