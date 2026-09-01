"""Tests for the live-update machinery.

The load-bearing property is timing: a forecast made on day t over horizon h
cannot be scored until t+h, so any weight applied on day t must be built only
from forecasts made on or before t-h. Get that wrong and the backtest scores a
prediction using the outcome it was predicting.
"""

import numpy as np
import pandas as pd
import pytest

from qsm.online import (adaptive_weights, blend, ledger_scorecard, record,
                        resolve, score_predictions, weight_summary)


@pytest.fixture
def setup():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2024-01-01", periods=300)
    cols = [f"T{i:02d}" for i in range(40)]
    idx = pd.MultiIndex.from_product([dates, cols], names=["date", "ticker"])

    fwd = pd.DataFrame(rng.normal(0, 0.03, (300, 40)), index=dates, columns=cols)
    # "good" tracks the outcome; "bad" is noise.
    good = fwd + rng.normal(0, 0.03, (300, 40))
    bad = pd.DataFrame(rng.normal(0, 0.03, (300, 40)), index=dates, columns=cols)
    preds = pd.DataFrame(
        {"good": good.stack(future_stack=True).reindex(idx),
         "bad": bad.stack(future_stack=True).reindex(idx)}, index=idx)
    return preds, fwd


def test_scoring_separates_a_good_model_from_a_bad_one(setup):
    preds, fwd = setup
    ic = score_predictions(preds, fwd)
    assert ic["good"].mean() > 0.3
    assert abs(ic["bad"].mean()) < 0.1


def test_weights_never_use_an_unresolved_outcome(setup):
    """The core anti-lookahead property, checked directly.

    Blank out every IC from a cut-off onwards. Weights at dates before
    `cut + horizon` must be unchanged, because they were never entitled to see
    that data anyway; only later dates may differ.
    """
    preds, fwd = setup
    ic = score_predictions(preds, fwd)
    horizon = 5

    full = adaptive_weights(ic, horizon=horizon, halflife=30)
    cut = ic.index[200]
    truncated_ic = ic.copy()
    truncated_ic.loc[cut:] = np.nan
    truncated = adaptive_weights(truncated_ic, horizon=horizon, halflife=30)

    unaffected = full.index[full.index < cut]
    pd.testing.assert_frame_equal(full.loc[unaffected], truncated.loc[unaffected])


def test_weights_lag_by_at_least_the_horizon(setup):
    """A model that only starts working at date D must not be favoured before D+h."""
    preds, fwd = setup
    ic = pd.DataFrame(0.0, index=fwd.index, columns=["good", "bad"])
    switch = ic.index[100]
    ic.loc[switch:, "good"] = 0.5          # 'good' suddenly becomes brilliant

    horizon = 10
    w = adaptive_weights(ic, horizon=horizon, halflife=5, min_history=1)
    before = w.loc[:ic.index[100 + horizon - 1]]
    assert before["good"].max() <= 0.5 + 1e-9, "weights reacted before the outcome was knowable"


def test_weights_are_a_valid_distribution(setup):
    preds, fwd = setup
    w = adaptive_weights(score_predictions(preds, fwd), horizon=5, halflife=30)
    assert (w >= -1e-12).all().all()
    np.testing.assert_allclose(w.sum(axis=1).to_numpy(), 1.0, atol=1e-9)


def test_no_history_falls_back_to_equal_weights(setup):
    preds, fwd = setup
    ic = score_predictions(preds, fwd)
    w = adaptive_weights(ic, horizon=5, halflife=30, min_history=40)
    assert w.iloc[0].tolist() == pytest.approx([0.5, 0.5])


def test_blend_standardises_before_combining():
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2024-01-01", periods=30), [f"T{i}" for i in range(25)]],
        names=["date", "ticker"])
    rng = np.random.default_rng(9)
    preds = pd.DataFrame({"a": rng.normal(0, 1, len(idx)),
                          "b": rng.normal(0, 1000, len(idx))}, index=idx)
    w = pd.DataFrame(0.5, index=idx.get_level_values("date").unique(), columns=["a", "b"])
    out = blend(preds, w)
    # Without standardisation the 1000x column would dominate entirely.
    assert abs(out.corr(preds["a"]) - out.corr(preds["b"])) < 0.25


def test_weight_summary_reports_movement(setup):
    preds, fwd = setup
    w = adaptive_weights(score_predictions(preds, fwd), horizon=5, halflife=30)
    s = weight_summary(w)
    assert set(s["mean"]) == {"good", "bad"}
    assert s["mean_abs_change"] >= 0


# ── ledger ────────────────────────────────────────────────────────────────
def test_ledger_records_only_the_newest_forecast(setup, tmp_path, monkeypatch):
    import qsm.online as online

    monkeypatch.setattr(online, "LEDGER_PATH", tmp_path / "ledger.parquet")
    preds, _ = setup
    n = record(preds, run="r1")
    assert n == 40, "only the last date's forecasts are live"

    led = pd.read_parquet(online.LEDGER_PATH)
    assert led["date"].nunique() == 1
    assert led["date"].max() == preds.index.get_level_values("date").max()


def test_ledger_appends_without_duplicating(setup, tmp_path, monkeypatch):
    import qsm.online as online

    monkeypatch.setattr(online, "LEDGER_PATH", tmp_path / "ledger.parquet")
    preds, _ = setup
    record(preds, run="r1")
    record(preds, run="r2")          # same date logged twice
    led = pd.read_parquet(online.LEDGER_PATH)
    assert len(led) == 40, "re-logging the same date must not duplicate rows"
    assert set(led["run"]) == {"r2"}, "the newer entry should win"


def test_unresolved_forecasts_are_reported_as_pending(setup, tmp_path, monkeypatch):
    import qsm.online as online

    monkeypatch.setattr(online, "LEDGER_PATH", tmp_path / "ledger.parquet")
    preds, fwd = setup
    record(preds, run="r1")

    closes = (1 + fwd.fillna(0)).cumprod() * 100
    card = ledger_scorecard(closes, horizon=5)
    # The last date's forward return does not exist yet.
    assert card["logged"] == 40
    assert card["resolved"] == 0
    assert card["pending"] == 40


def test_resolved_forecasts_get_scored(setup, tmp_path, monkeypatch):
    import qsm.online as online

    monkeypatch.setattr(online, "LEDGER_PATH", tmp_path / "ledger.parquet")
    preds, fwd = setup
    # Log an older date so its outcome is already known.
    old = preds.index.get_level_values("date")[100]
    record(preds, run="r1", as_of=old)

    closes = (1 + fwd.fillna(0)).cumprod() * 100
    res = resolve(closes, horizon=5)
    assert res["realised"].notna().all(), "an elapsed horizon must resolve"


def test_missing_ledger_is_not_an_error(tmp_path, monkeypatch):
    import qsm.online as online

    monkeypatch.setattr(online, "LEDGER_PATH", tmp_path / "nothing.parquet")
    assert resolve(pd.DataFrame(), horizon=5).empty
    assert ledger_scorecard(pd.DataFrame(), horizon=5)["logged"] == 0
