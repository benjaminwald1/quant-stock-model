"""Prediction targets.

The label is the forward ``horizon``-day return. Two adjustments matter:

* **Excess**: subtract the cross-sectional mean for the date. Otherwise the
  model spends its capacity predicting the market, which a dollar-neutral
  long/short book cannot monetise anyway.
* **Rank**: map to cross-sectional ranks. Raw forward returns are fat-tailed,
  and under squared error a handful of biotech takeouts would set the entire
  fit. Ranks make the objective care about ordering, which is exactly what a
  quantile-sorted portfolio consumes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LabelConfig


def build_labels(panel: pd.DataFrame, cfg: LabelConfig | None = None) -> pd.DataFrame:
    """Return a long frame with columns ``fwd_ret``, ``target``, ``ret_next_1d``.

    ``fwd_ret``     raw forward ``horizon``-day return (for diagnostics)
    ``target``      what the model is trained on (excess and/or ranked)
    ``ret_next_1d`` next-day return, the atom the backtester compounds
    """
    cfg = cfg or LabelConfig()
    close = panel.pivot(index="date", columns="ticker", values="close").sort_index()

    fwd = close.shift(-cfg.horizon) / close - 1
    next_1d = close.shift(-1) / close - 1

    target = fwd.copy()
    if cfg.excess:
        target = target.sub(target.mean(axis=1), axis=0)
    if cfg.rank_target:
        valid = target.notna().sum(axis=1)
        target = (target.rank(axis=1, pct=True, na_option="keep") - 0.5).where(
            valid.ge(20), np.nan
        )

    out = pd.concat(
        [
            fwd.stack(future_stack=True).rename("fwd_ret"),
            target.stack(future_stack=True).rename("target"),
            next_1d.stack(future_stack=True).rename("ret_next_1d"),
        ],
        axis=1,
    )
    out.index.names = ["date", "ticker"]
    return out.sort_index()
