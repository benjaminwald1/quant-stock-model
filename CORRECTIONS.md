# Corrections

## 2026-09-03 — Sharpe ratios and returns were misreported

Earlier commit messages and analysis in this repo quoted the model's net Sharpe
as **0.50 against 1.62** for buy-and-hold, and an annual return around 20%.

Those were wrong. The summary table prints `sharpe` in column 1 and `t_stat` in
column 7; the quoted figures were the t-statistics.

The correct numbers for `20260831-171523-full15y` (us_all, 15 years, horizon 5),
taken from that run's own `metrics.json`:

| | quoted | actual |
| --- | --- | --- |
| ensemble Sharpe | 0.50 | **0.206** |
| buy & hold Sharpe | 1.62 | **0.660** |
| ensemble annual return | ~20% | **1.79%** |
| buy & hold annual return | ~65% | **12.12%** |

Every conclusion drawn from them still holds — the model loses to simply
holding the universe — but it loses by more than was stated, and its absolute
return is far lower.

Affected commit: "Three attempts to raise returns, all negative; guard the fund
from experiments". Git history is not rewritten; this file is the correction.

Unaffected: the fund replay figures in `experiments/fund_replay.py` (the
3.36% -> 18.48% holdout improvement). Those come from an independent simulation
of the trading rule and were computed correctly.
