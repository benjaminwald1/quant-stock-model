"""The load-bearing test: features must not change when the future changes.

If a feature at date t is truly a function of data through t, then appending
more data after t cannot move it. Recomputing on a truncated panel and diffing
against the full-history version catches almost every accidental lookahead:
a forgotten shift, a centred rolling window, a global normalisation, an
`ffill(limit=None)` that reaches backwards from a later date.
"""

import numpy as np

from qsm.features import compute_features
from qsm.labels import build_labels
from qsm.config import LabelConfig


def test_features_are_causal(panel):
    cut = panel["date"].sort_values().unique()[600]
    truncated = panel[panel["date"] <= cut]

    full_f = compute_features(panel)
    trunc_f = compute_features(truncated)

    # Compare on every (date, ticker) the truncated run produced.
    common = trunc_f.index.intersection(full_f.index)
    assert len(common) > 10_000, "truncated panel should still cover most rows"

    a = full_f.loc[common].sort_index()
    b = trunc_f.loc[common].sort_index()

    for col in a.columns:
        x, y = a[col], b[col]
        both = x.notna() & y.notna()
        assert (x.isna() == y.isna()).all(), f"{col}: NaN pattern depends on future data"
        np.testing.assert_allclose(
            x[both].to_numpy(), y[both].to_numpy(), rtol=1e-9, atol=1e-9,
            err_msg=f"feature '{col}' changed when future data was appended -> lookahead",
        )


def test_label_looks_forward_by_exactly_the_horizon(panel):
    horizon = 5
    labels = build_labels(panel, LabelConfig(horizon=horizon, excess=False, rank_target=False))
    close = panel.pivot(index="date", columns="ticker", values="close").sort_index()

    ticker = close.columns[0]
    series = close[ticker]
    for i in (100, 300, 500):
        expected = series.iloc[i + horizon] / series.iloc[i] - 1
        got = labels.loc[(series.index[i], ticker), "fwd_ret"]
        assert abs(got - expected) < 1e-12

    # The last `horizon` dates cannot have a label; they must be NaN, not filled.
    tail = labels.loc[(close.index[-horizon:], slice(None)), "fwd_ret"]
    assert tail.isna().all()


def test_excess_target_is_cross_sectionally_neutral(panel):
    labels = build_labels(panel, LabelConfig(horizon=5, excess=True, rank_target=False))
    per_date = labels["target"].groupby(level="date").mean().dropna()
    assert per_date.abs().max() < 1e-10, "excess target should sum to zero within each date"


def test_ranked_features_are_bounded(panel):
    f = compute_features(panel)
    assert f.min().min() >= -0.5 - 1e-9
    assert f.max().max() <= 0.5 + 1e-9
