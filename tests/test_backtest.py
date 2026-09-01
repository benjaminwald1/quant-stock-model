import numpy as np
import pandas as pd
import pytest

from qsm.backtest import (
    information_coefficient,
    performance_metrics,
    run_backtest,
    signal_to_weights,
)
from qsm.config import BacktestConfig


@pytest.fixture
def signal():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", periods=250)
    cols = [f"T{i:02d}" for i in range(40)]
    return pd.DataFrame(rng.normal(size=(250, 40)), index=dates, columns=cols)


def test_book_is_dollar_neutral_and_fully_invested(signal):
    cfg = BacktestConfig(quantile=0.2, long_short=True, gross_exposure=1.0, max_weight=0.1)
    w = signal_to_weights(signal, cfg)
    assert w.sum(axis=1).abs().max() < 1e-9, "long and short legs must offset"
    np.testing.assert_allclose(w.abs().sum(axis=1).to_numpy(), 1.0, atol=1e-9)


def test_weight_cap_is_respected_without_breaking_neutrality(signal):
    cfg = BacktestConfig(quantile=0.5, long_short=True, max_weight=0.03)
    w = signal_to_weights(signal, cfg)
    assert w.abs().max().max() <= 0.03 + 1e-9
    assert w.sum(axis=1).abs().max() < 1e-9


def test_weight_cap_holds_when_it_actually_binds(signal):
    """The binding case: too few names for the cap to reach full gross.

    Top/bottom decile of 40 names is 4 per leg. At a 2% cap a leg can carry at
    most 8%, well under the 50% it would otherwise take, so the cap must bind
    and the book must end up deliberately under-invested rather than quietly
    exceeding the limit.
    """
    cfg = BacktestConfig(quantile=0.1, long_short=True, max_weight=0.02, gross_exposure=1.0)
    w = signal_to_weights(signal, cfg)
    assert w.abs().max().max() <= 0.02 + 1e-9, "cap must not be silently discarded"
    assert w.sum(axis=1).abs().max() < 1e-9, "capping must not break neutrality"
    gross = w.abs().sum(axis=1)
    assert gross.max() < 0.5, "an infeasible cap must leave the book under-invested"
    assert gross.max() > 0.0


def test_uncapped_leg_reaches_full_gross(signal):
    cfg = BacktestConfig(quantile=0.25, long_short=True, max_weight=0.0, gross_exposure=1.0)
    w = signal_to_weights(signal, cfg)
    np.testing.assert_allclose(w.abs().sum(axis=1).to_numpy(), 1.0, atol=1e-9)


def test_cap_water_fills_rather_than_truncating(signal):
    """With a feasible-but-binding cap, freed exposure moves to uncapped names."""
    n = signal.shape[1]
    cfg = BacktestConfig(quantile=0.5, long_short=True, max_weight=1.0 / n,
                         gross_exposure=1.0, signal_weighted=True)
    w = signal_to_weights(signal, cfg)
    assert w.abs().max().max() <= 1.0 / n + 1e-9
    # Signal-weighting concentrates; the cap flattens it, but the leg should
    # still deploy close to its full 0.5 because enough names remain.
    assert w.clip(lower=0).sum(axis=1).mean() > 0.45


def test_long_only_mode_has_no_shorts(signal):
    w = signal_to_weights(signal, BacktestConfig(long_short=False, quantile=0.2, max_weight=1.0))
    assert (w >= -1e-12).all().all()
    np.testing.assert_allclose(w.sum(axis=1).to_numpy(), 1.0, atol=1e-9)


def test_holding_period_reduces_turnover(signal):
    cfg = BacktestConfig(quantile=0.2)
    fast = signal_to_weights(signal, cfg, holding_days=1).diff().abs().sum(axis=1).mean()
    slow = signal_to_weights(signal, cfg, holding_days=10).diff().abs().sum(axis=1).mean()
    assert slow < fast / 2


def test_execution_lag_shifts_the_book_forward():
    """A signal that perfectly predicts day t's return earns nothing at lag 1."""
    dates = pd.bdate_range("2020-01-01", periods=120)
    cols = ["A", "B", "C", "D"]
    rng = np.random.default_rng(1)
    ret = pd.DataFrame(rng.normal(0, 0.02, (120, 4)), index=dates, columns=cols)
    cheating = ret.copy()  # signal == the very return it is about to earn

    cfg = BacktestConfig(quantile=0.25, cost_bps=0, max_weight=1.0, min_names=4)
    lag0 = run_backtest(cheating, ret, cfg, holding_days=1, execution_lag=0)
    lag1 = run_backtest(cheating, ret, cfg, holding_days=1, execution_lag=1)
    assert lag0["metrics"]["sharpe"] > 10, "lag 0 should monetise a perfect signal"
    assert abs(lag1["metrics"]["sharpe"]) < 3, "lag 1 must destroy a same-day-only signal"


def test_costs_reduce_returns_monotonically(signal):
    rng = np.random.default_rng(2)
    ret = pd.DataFrame(rng.normal(0, 0.015, signal.shape), index=signal.index, columns=signal.columns)
    cheap = run_backtest(signal, ret, BacktestConfig(cost_bps=0.0), holding_days=5)
    dear = run_backtest(signal, ret, BacktestConfig(cost_bps=50.0), holding_days=5)
    assert dear["pnl"].sum() < cheap["pnl"].sum()
    assert cheap["costs"].sum() == 0.0


def test_performance_metrics_on_a_known_series():
    pnl = pd.Series(0.001, index=pd.bdate_range("2020-01-01", periods=252))
    m = performance_metrics(pnl)
    assert m["hit_rate"] == 1.0
    assert m["max_drawdown"] == 0.0
    assert abs(m["total_return"] - (1.001**252 - 1)) < 1e-9
    assert np.isinf(m["sharpe"]) or np.isnan(m["sharpe"]) or m["sharpe"] > 100


def test_max_drawdown_is_measured_peak_to_trough():
    pnl = pd.Series([0.5, -0.5, 0.0], index=pd.bdate_range("2020-01-01", periods=3))
    # 1.0 -> 1.5 -> 0.75: a 50% fall from the peak.
    assert abs(performance_metrics(pnl)["max_drawdown"] - (-0.5)) < 1e-12


def test_information_coefficient_recovers_a_perfect_ranking():
    dates = pd.bdate_range("2020-01-01", periods=60)
    cols = [f"T{i}" for i in range(30)]
    rng = np.random.default_rng(3)
    fwd = pd.DataFrame(rng.normal(size=(60, 30)), index=dates, columns=cols)
    assert information_coefficient(fwd, fwd)["ic_mean"] > 0.999
    assert information_coefficient(-fwd, fwd)["ic_mean"] < -0.999
    noise = pd.DataFrame(rng.normal(size=(60, 30)), index=dates, columns=cols)
    assert abs(information_coefficient(noise, fwd)["ic_mean"]) < 0.15
