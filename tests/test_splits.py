import pandas as pd
import pytest

from qsm.config import SplitConfig
from qsm.splits import Split, check_no_overlap, purged_walk_forward, train_val_split


@pytest.fixture
def dates():
    return pd.bdate_range("2010-01-01", periods=2600)


def test_train_always_precedes_test(dates):
    for s in purged_walk_forward(dates, SplitConfig(), horizon=5):
        assert s.train_dates[-1] < s.test_dates[0]
        assert len(s.train_dates) >= SplitConfig().min_train_size


def test_purge_gap_covers_label_horizon(dates):
    horizon, embargo = 10, 5
    cfg = SplitConfig(embargo=embargo)
    splits = purged_walk_forward(dates, cfg, horizon=horizon)
    check_no_overlap(splits, horizon, embargo)
    for s in splits:
        gap = dates.get_loc(s.test_dates[0]) - dates.get_loc(s.train_dates[-1])
        assert gap >= horizon + embargo


def test_test_blocks_do_not_overlap_each_other(dates):
    splits = purged_walk_forward(dates, SplitConfig(), horizon=5)
    seen = set()
    for s in splits:
        assert not seen & set(s.test_dates), "test blocks must be disjoint"
        seen |= set(s.test_dates)


def test_rolling_window_is_bounded(dates):
    cfg = SplitConfig(expanding=False, min_train_size=756)
    for s in purged_walk_forward(dates, cfg, horizon=5):
        assert len(s.train_dates) <= cfg.min_train_size


def test_validation_tail_is_purged_and_at_the_end(dates):
    fit, val = train_val_split(dates[:1000], val_fraction=0.2, horizon=5, embargo=5)
    assert fit[-1] < val[0]
    gap = dates.get_loc(val[0]) - dates.get_loc(fit[-1])
    assert gap >= 10
    assert val[-1] == dates[999]


def test_short_sample_raises_rather_than_silently_leaking():
    with pytest.raises(ValueError):
        purged_walk_forward(pd.bdate_range("2020-01-01", periods=50), SplitConfig(), horizon=5)


def test_check_no_overlap_catches_a_bad_split(dates):
    bad = [Split(0, dates[:500], dates[500:600])]  # zero gap
    with pytest.raises(AssertionError):
        check_no_overlap(bad, horizon=5, embargo=5)
