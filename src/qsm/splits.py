"""Purged, embargoed walk-forward cross-validation.

Random k-fold on a financial panel is the single most common way to produce a
backtest that cannot be traded. Two things go wrong:

1. **Time leakage** — training on 2019 to predict 2015.
2. **Label overlap** — the label at date ``t`` spans ``t .. t+h``. A training
   row at ``t = test_start - 1`` therefore contains realised returns from
   *inside* the test window. The fix is to purge the last ``h`` training dates,
   plus an embargo for good measure (serial correlation lingers past ``h``).

Both are handled here. Splits are returned as arrays of *dates*, not row
positions, because the panel is unbalanced — different tickers exist on
different days.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SplitConfig


@dataclass
class Split:
    index: int
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex

    @property
    def label(self) -> str:
        return (
            f"fold{self.index}: train {self.train_dates[0].date()}..{self.train_dates[-1].date()} "
            f"| test {self.test_dates[0].date()}..{self.test_dates[-1].date()}"
        )


def purged_walk_forward(
    dates: pd.DatetimeIndex, cfg: SplitConfig, horizon: int
) -> list[Split]:
    """Generate expanding (or rolling) train/test blocks with purge + embargo."""
    dates = pd.DatetimeIndex(sorted(pd.unique(dates)))
    n = len(dates)
    purge = horizon + cfg.embargo

    test_size = cfg.test_size
    needed = cfg.min_train_size + purge + test_size
    if n < needed:
        # Shrink the test blocks rather than failing outright: short samples
        # (and unit tests) should still produce a usable, correctly purged split.
        test_size = max(20, (n - cfg.min_train_size - purge) // max(1, cfg.n_splits))
        if n < cfg.min_train_size + purge + test_size:
            raise ValueError(
                f"Need at least {cfg.min_train_size + purge + 20} dates to build a split; got {n}."
            )

    # Lay the test blocks out backwards from the end of the sample so the most
    # recent data always gets evaluated.
    starts = []
    end = n
    for _ in range(cfg.n_splits):
        start = end - test_size
        if start - purge < cfg.min_train_size:
            break
        starts.append(start)
        end = start
    starts.reverse()

    splits = []
    for i, start in enumerate(starts):
        train_end = start - purge
        train_start = 0 if cfg.expanding else max(0, train_end - cfg.min_train_size)
        splits.append(
            Split(
                index=i,
                train_dates=dates[train_start:train_end],
                test_dates=dates[start : start + test_size],
            )
        )
    return splits


def train_val_split(
    train_dates: pd.DatetimeIndex, val_fraction: float, horizon: int, embargo: int = 0
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Carve a validation tail off a training block, purged the same way.

    The validation set is the *end* of the training window, never a random
    sample: early stopping on randomly interleaved dates leaks neighbouring
    days into the stopping decision and picks far too many trees.
    """
    n = len(train_dates)
    n_val = int(n * val_fraction)
    purge = horizon + embargo
    if n_val < 20 or n - n_val - purge < 50:
        return train_dates, pd.DatetimeIndex([])
    val = train_dates[n - n_val :]
    fit = train_dates[: n - n_val - purge]
    return fit, val


def check_no_overlap(splits: list[Split], horizon: int, embargo: int) -> None:
    """Assert the purge actually holds. Cheap insurance, called by the pipeline."""
    for s in splits:
        gap = np.busday_count(
            s.train_dates[-1].date(), s.test_dates[0].date()
        )
        if gap < horizon + embargo:
            raise AssertionError(
                f"{s.label}: only {gap} business days between train end and test start, "
                f"need >= {horizon + embargo}"
            )
