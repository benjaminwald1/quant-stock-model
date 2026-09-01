"""Pick the fund's entry/exit band honestly, then check the pick out of sample.

Sweeping every band and shipping whichever won is how you fit the past. The
ranking here flips between windows, so this splits the record: choose on the
early years, then score that choice on later years it never saw. A parameter
that only looks good on the half that selected it has not been shown to work.

Selection also rewards consistency rather than a single spike — a band is
scored by its *worst* year in the selection period, not its average, because a
setting that blows up in one regime is not one to leave running unattended.
"""

from __future__ import annotations

import argparse
import sys
from statistics import mean

import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "experiments")

from fund_replay import find_panel, replay                     # noqa: E402

GRID = [(e, x) for e in (95.0, 90.0, 85.0, 80.0)
        for x in (50.0, 60.0, 70.0, 78.0) if x < e]


def year_windows(end: str, n: int):
    """n one-year windows ending at `end`, most recent last."""
    stop = pd.Timestamp(end)
    out = []
    for i in range(n, 0, -1):
        out.append(str((stop - pd.DateOffset(years=i - 1)).date()))
    return out


def score(run, panel, budget, enter, exit_, end, cost_bps):
    r = replay(run, panel, months=12, budget=budget, sizing="equity",
               enter_above=enter, exit_below=exit_, end=end, cost_bps=cost_bps)
    return r["pnl_pct"], r["benchmark_pct"], r["turnover_trades"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="20260831-171523-full15y")
    ap.add_argument("--budget", type=float, default=5000.0)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    a = ap.parse_args()

    panel = find_panel(a.run)
    # Selection: the three years to Aug 2024. Holdout: the two years after.
    select_ends = ["2022-08-31", "2023-08-31", "2024-08-30"]
    holdout_ends = ["2025-08-29", "2026-08-31"]

    print(f"\nselection windows {select_ends}")
    print(f"holdout windows   {holdout_ends}\n")
    print(f"{'enter/exit':12s} " + " ".join(f"{e[:7]:>9s}" for e in select_ends)
          + f" {'worst':>8s} {'mean':>8s} {'trades':>7s}")

    results = []
    for enter, exit_ in GRID:
        yrs, trades = [], 0
        for end in select_ends:
            pnl, _b, n = score(a.run, panel, a.budget, enter, exit_, end, a.cost_bps)
            yrs.append(pnl)
            trades += n
        results.append({"enter": enter, "exit": exit_, "years": yrs,
                        "worst": min(yrs), "mean": mean(yrs), "trades": trades})
        print(f"{enter:.0f}/{exit_:<8.0f} " + " ".join(f"{v:8.2f}%" for v in yrs)
              + f" {min(yrs):7.2f}% {mean(yrs):7.2f}% {trades:7d}")

    # Rank by the worst year, tie-broken by the mean.
    results.sort(key=lambda r: (-r["worst"], -r["mean"]))
    pick = results[0]
    current = next(r for r in results if r["enter"] == 90.0 and r["exit"] == 70.0)
    print(f"\nselected on the early years: {pick['enter']:.0f}/{pick['exit']:.0f} "
          f"(worst year {pick['worst']:.2f}%, mean {pick['mean']:.2f}%)")
    print(f"currently shipping:          {current['enter']:.0f}/{current['exit']:.0f} "
          f"(worst year {current['worst']:.2f}%, mean {current['mean']:.2f}%)")

    print(f"\n{'HOLDOUT — never used to choose':40s}")
    print(f"{'band':12s} " + " ".join(f"{e[:7]:>9s}" for e in holdout_ends)
          + f" {'mean':>8s} {'vs universe':>12s}")
    for label, cand in (("selected", pick), ("shipping", current)):
        outs, alphas = [], []
        for end in holdout_ends:
            pnl, bench, _n = score(a.run, panel, a.budget, cand["enter"], cand["exit"],
                                   end, a.cost_bps)
            outs.append(pnl)
            alphas.append(pnl - bench)
        print(f"{cand['enter']:.0f}/{cand['exit']:<8.0f} "
              + " ".join(f"{v:8.2f}%" for v in outs)
              + f" {mean(outs):7.2f}% {mean(alphas):11.2f}%  ({label})")
