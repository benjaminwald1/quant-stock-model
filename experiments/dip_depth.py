"""How deep should the fund's buy limit sit under the previous close?

Shallower fills more often at a worse price; deeper gets a better price on the
trades it makes and misses the rest. The names that never dip are
disproportionately the ones that ran, so this cannot be reasoned out — the
missed trades have to be paid for in the result.

Chosen on the early years, scored on a holdout that had no say.
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
DEPTHS = [0.0025, 0.005, 0.0075, 0.010, 0.015, 0.020]


def run(a, panel, dip, ends):
    yrs, trades, alph, buys = [], 0, [], 0
    for end in ends:
        r = replay(a.run, panel, months=12, budget=a.budget, sizing="equity",
                   exit_below=30.0, end=end, cost_bps=a.cost_bps, dip_pct=dip)
        yrs.append(r["pnl_pct"])
        alph.append(r["pnl_pct"] - r["benchmark_pct"])
        trades += r["turnover_trades"]
        buys += r["buys"]
    return yrs, trades, alph, buys


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="20260831-171523-full15y")
    ap.add_argument("--budget", type=float, default=5000.0)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    a = ap.parse_args()
    panel = find_panel(a.run)

    for label, ends in (("SELECTION", SELECT), ("HOLDOUT — never used to choose", HOLDOUT)):
        print(f"\n{label}")
        print(f"{'dip':>7s} " + " ".join(f"{e[:7]:>9s}" for e in ends)
              + f" {'worst':>8s} {'mean':>8s} {'buys':>6s} {'vs universe':>12s}")
        for dip in DEPTHS:
            yrs, trades, alph, buys = run(a, panel, dip, ends)
            print(f"{dip * 100:6.2f}% " + " ".join(f"{v:8.2f}%" for v in yrs)
                  + f" {min(yrs):7.2f}% {mean(yrs):7.2f}% {buys:6d} {mean(alph):11.2f}%")
