"""The two levers that actually move the fund: when it sells, and how many it holds.

The entry threshold turned out to be inert — with ten slots and thousands of
names ranked above any plausible cutoff, the top ten are the same whatever the
line is set to. So this sweeps the exit threshold and the position count, still
choosing on early years and scoring on a holdout that had no say.
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


def year(run, panel, budget, exit_, npos, end, cost_bps):
    r = replay(run, panel, months=12, budget=budget, sizing="equity",
               enter_above=90.0, exit_below=exit_, max_positions=npos,
               end=end, cost_bps=cost_bps)
    return r["pnl_pct"], r["benchmark_pct"], r["turnover_trades"]


def sweep(run, panel, budget, cost_bps, combos, ends, label):
    print(f"\n{label}")
    print(f"{'exit':>5s} {'slots':>6s} " + " ".join(f"{e[:7]:>9s}" for e in ends)
          + f" {'worst':>8s} {'mean':>8s} {'trades':>7s}")
    rows = []
    for exit_, npos in combos:
        yrs, trades = [], 0
        for end in ends:
            p, _b, n = year(run, panel, budget, exit_, npos, end, cost_bps)
            yrs.append(p)
            trades += n
        rows.append({"exit": exit_, "npos": npos, "years": yrs,
                     "worst": min(yrs), "mean": mean(yrs), "trades": trades})
        print(f"{exit_:5.0f} {npos:6d} " + " ".join(f"{v:8.2f}%" for v in yrs)
              + f" {min(yrs):7.2f}% {mean(yrs):7.2f}% {trades:7d}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="20260831-171523-full15y")
    ap.add_argument("--budget", type=float, default=5000.0)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    a = ap.parse_args()
    panel = find_panel(a.run)

    combos = [(x, 10) for x in (20.0, 30.0, 40.0, 50.0, 60.0, 70.0)]
    combos += [(40.0, n) for n in (5, 15, 20)]
    rows = sweep(a.run, panel, a.budget, a.cost_bps, combos, SELECT,
                 "SELECTION — early years")

    rows.sort(key=lambda r: (-r["worst"], -r["mean"]))
    pick = rows[0]
    cur = next(r for r in rows if r["exit"] == 70.0 and r["npos"] == 10)
    print(f"\npicked on selection: exit {pick['exit']:.0f}, {pick['npos']} slots "
          f"(worst {pick['worst']:.2f}%, mean {pick['mean']:.2f}%)")

    print("\nHOLDOUT — never used to choose")
    print(f"{'exit':>5s} {'slots':>6s} " + " ".join(f"{e[:7]:>9s}" for e in HOLDOUT)
          + f" {'mean':>8s} {'vs universe':>12s}")
    for cand, tag in ((pick, "picked"), (cur, "shipping")):
        outs, alph = [], []
        for end in HOLDOUT:
            p, b, _n = year(a.run, panel, a.budget, cand["exit"], cand["npos"],
                            end, a.cost_bps)
            outs.append(p)
            alph.append(p - b)
        print(f"{cand['exit']:5.0f} {cand['npos']:6d} "
              + " ".join(f"{v:8.2f}%" for v in outs)
              + f" {mean(outs):7.2f}% {mean(alph):11.2f}%  ({tag})")
