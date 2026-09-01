"""Does trading more often make more money?

The fund buys above rank `enter` and sells below `exit`. The gap between them
is a deadband: widen it and the fund sits still, narrow it and it churns. This
sweeps that gap on real bars and charges the project's own 10bps a side, so the
extra trading has to pay for itself rather than being free.
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")

from fund_replay import find_panel, replay                     # noqa: E402

BANDS = [
    ("very slow", 95.0, 50.0),
    ("shipping",  90.0, 70.0),
    ("faster",    85.0, 75.0),
    ("fast",      80.0, 78.0),
    ("frantic",   75.0, 74.0),
    ("hair-trigger", 70.0, 69.0),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="20260831-171523-full15y")
    ap.add_argument("--months", type=int, default=3)
    ap.add_argument("--budget", type=float, default=5000.0)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    a = ap.parse_args()

    panel = find_panel(a.run)
    print(f"\n{a.run} · {a.months} months · ${a.budget:,.0f} · "
          f"{a.cost_bps:.0f}bps a side\n")
    print(f"{'band':14s} {'enter':>6s} {'exit':>5s} {'trades':>7s} {'costs':>9s} "
          f"{'P&L net':>10s} {'%':>8s} {'P&L if free':>12s}")

    for name, enter, exit_ in BANDS:
        net = replay(a.run, panel, months=a.months, budget=a.budget, sizing="equity",
                     enter_above=enter, exit_below=exit_, cost_bps=a.cost_bps)
        free = replay(a.run, panel, months=a.months, budget=a.budget, sizing="equity",
                      enter_above=enter, exit_below=exit_, cost_bps=0.0)
        print(f"{name:14s} {enter:6.0f} {exit_:5.0f} {net['turnover_trades']:7d} "
              f"{net['costs_paid']:9,.2f} {net['pnl']:10,.2f} {net['pnl_pct']:7.2f}% "
              f"{free['pnl']:12,.2f}")
