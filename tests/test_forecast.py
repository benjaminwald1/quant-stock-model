"""Tests for the forward value estimate.

The point of this feature is not the target price — it is the interval around
it. These tests exist mostly to stop the interval from silently collapsing,
because a point estimate without its error bars is the most misleading thing
this project could display.
"""

import numpy as np
import pandas as pd
import pytest

from qsm.forecast import Z80, calibrate, project


@pytest.fixture
def panel():
    """A signal with a genuine but small edge, buried in realistic noise."""
    rng = np.random.default_rng(4)
    dates = pd.bdate_range("2020-01-01", periods=400)
    cols = [f"T{i:03d}" for i in range(60)]
    signal = pd.DataFrame(rng.normal(size=(400, 60)), index=dates, columns=cols)
    edge = 0.004 * (signal.rank(axis=1, pct=True) - 0.5)
    noise = pd.DataFrame(rng.normal(0, 0.04, (400, 60)), index=dates, columns=cols)
    return signal, edge + noise


def test_calibration_recovers_a_monotone_staircase(panel):
    signal, fwd = panel
    cal = calibrate(signal, fwd)
    assert cal["observations"] > 20_000
    assert cal["monotonicity"] > 0.8, "planted edge should show as an ascending staircase"
    assert cal["spread_top_minus_bottom"] > 0
    means = [b["mean"] for b in cal["bins"]]
    assert means[-1] > means[0]


def test_calibration_finds_nothing_in_noise():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2020-01-01", periods=400)
    cols = [f"T{i:03d}" for i in range(60)]
    signal = pd.DataFrame(rng.normal(size=(400, 60)), index=dates, columns=cols)
    fwd = pd.DataFrame(rng.normal(0, 0.04, (400, 60)), index=dates, columns=cols)
    cal = calibrate(signal, fwd)
    assert abs(cal["spread_top_minus_bottom"]) < 0.004, "no edge should be found in noise"


def test_every_bin_reports_dispersion(panel):
    """A bin with a mean but no std would license a point estimate with no band."""
    signal, fwd = panel
    for b in calibrate(signal, fwd)["bins"]:
        assert b["std"] > 0
        assert b["n"] > 0


def test_projection_brackets_the_target_and_scales_with_price(panel):
    signal, fwd = panel
    cal = calibrate(signal, fwd)
    last = (signal.rank(axis=1, pct=True) * 100).iloc[-1]
    prices = {c: 100.0 for c in signal.columns}
    rows = project(last, prices, cal, horizon=5)

    assert len(rows) == len(signal.columns)
    for r in rows:
        assert r["target_low"] < r["target"] < r["target_high"]
        # project() rounds to 4dp, so compare at that resolution.
        assert r["target"] == pytest.approx(r["price"] * (1 + r["expected_return"]), abs=1e-3)
        assert r["target_low"] == pytest.approx(
            r["price"] * (1 + r["expected_return"] - Z80 * r["uncertainty"]), abs=1e-3)
        assert r["horizon_days"] == 5


def test_the_interval_dwarfs_the_estimate(panel):
    """The honesty check: on equity data the band must be far wider than the edge.

    If this ever fails, either the calibration has broken or someone has quietly
    shrunk the uncertainty — and the UI would start implying a precision the
    model does not have.
    """
    signal, fwd = panel
    cal = calibrate(signal, fwd)
    last = (signal.rank(axis=1, pct=True) * 100).iloc[-1]
    rows = project(last, {c: 100.0 for c in signal.columns}, cal, horizon=5)
    for r in rows:
        width = r["target_high"] - r["target_low"]
        assert width > 10 * abs(r["target"] - r["price"]), (
            f"{r['ticker']}: interval is not wide enough relative to the point estimate")
        assert r["signal_to_noise"] < 0.5


def test_projection_is_sorted_best_first(panel):
    signal, fwd = panel
    cal = calibrate(signal, fwd)
    last = (signal.rank(axis=1, pct=True) * 100).iloc[-1]
    rows = project(last, {c: 100.0 for c in signal.columns}, cal, horizon=5)
    exp = [r["expected_return"] for r in rows]
    assert exp == sorted(exp, reverse=True)


def test_names_without_a_live_price_are_skipped(panel):
    signal, fwd = panel
    cal = calibrate(signal, fwd)
    last = (signal.rank(axis=1, pct=True) * 100).iloc[-1]
    prices = {c: 100.0 for c in list(signal.columns)[:10]}
    prices["T011"] = 0.0          # a zero price is not a price
    prices["T012"] = float("nan")
    rows = project(last, prices, cal, horizon=5)
    assert len(rows) == 10
    assert {"T011", "T012"} & {r["ticker"] for r in rows} == set()


def test_too_little_history_refuses_to_calibrate():
    dates = pd.bdate_range("2020-01-01", periods=5)
    cols = ["A", "B"]
    small = pd.DataFrame(1.0, index=dates, columns=cols)
    with pytest.raises(ValueError, match="Not enough"):
        calibrate(small, small)
