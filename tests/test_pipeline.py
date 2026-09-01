"""End-to-end checks, including the null experiment."""

import numpy as np
import pandas as pd
import pytest

from qsm.config import Config
from qsm.data import apply_universe_filters, make_synthetic
from qsm.pipeline import NON_FEATURE_COLS, add_ensemble, feature_list, prepare_dataset, run_synthetic


def _small_cfg() -> Config:
    cfg = Config()
    cfg.splits.n_splits = 2
    cfg.splits.min_train_size = 500
    cfg.splits.test_size = 200
    cfg.data.min_history_days = 260
    cfg.model.models = ("ridge", "lgbm")
    cfg.model.lgbm_params["n_estimators"] = 120
    return cfg


def test_universe_filter_uses_only_prior_information():
    panel = make_synthetic(n_tickers=10, n_days=400, seed=5)
    flagged = apply_universe_filters(panel, Config().data)
    # A ticker cannot be tradable before it has the required history.
    first = flagged.groupby("ticker", sort=False).head(300)
    assert not first["tradable"].any()


def test_prepare_dataset_shape_and_columns():
    cfg = _small_cfg()
    ds = prepare_dataset(cfg, panel=make_synthetic(n_tickers=25, n_days=800, seed=6))
    assert {"target", "fwd_ret", "ret_next_1d"} <= set(ds.columns)
    assert ds.index.names == ["date", "ticker"]
    assert len(ds) > 0 and ds["target"].notna().mean() > 0.9


def test_end_to_end_produces_all_artifacts(tmp_path):
    cfg = _small_cfg()
    run_dir, summary, results = run_synthetic(
        cfg, signal_strength=0.06, n_tickers=40, n_days=1100, seed=7, tag="pytest",
        run_dir=tmp_path / "run",
    )
    for name in ("summary.csv", "metrics.json", "equity_curves.csv", "config.json",
                 "oos_predictions.parquet", "folds.txt", "quantile_returns.csv"):
        assert (run_dir / name).exists(), f"missing artifact: {name}"
    assert {"ridge", "lgbm", "ensemble"} <= set(summary.index)
    assert summary.loc["ridge", "sharpe"] == pytest.approx(
        results["ridge"]["metrics"]["sharpe"], rel=1e-9
    )


def test_signal_is_recovered_when_it_exists(tmp_path):
    """Sanity in the other direction: a planted signal must be found."""
    cfg = _small_cfg()
    _, summary, _ = run_synthetic(
        cfg, signal_strength=0.10, n_tickers=50, n_days=1100, seed=8, tag="pytest-signal",
        run_dir=tmp_path / "run",
    )
    assert summary.loc["ridge", "ic_mean"] > 0.01


@pytest.mark.parametrize("seed", [21, 22])
def test_null_panel_yields_no_alpha(seed, tmp_path):
    """The leakage guard. No signal exists, so none may be found.

    The bound is on the IC t-statistic rather than on the Sharpe ratio. That
    matters: over a ~200-day out-of-sample window the Sharpe estimate has a
    standard error near 1.3, so a null run landing at Sharpe 1.6 is unremarkable
    noise and bounding it directly would just produce a flaky test. The IC
    t-stat is computed across ~200 daily cross-sections and is correspondingly
    better behaved. Measured across 12 null runs it stayed inside +/-2.6; the
    leaky positive control below clears +/-20.
    """
    cfg = _small_cfg()
    _, summary, _ = run_synthetic(
        cfg, signal_strength=0.0, n_tickers=50, n_days=1100, seed=seed,
        tag=f"pytest-null{seed}", run_dir=tmp_path / "run",
    )
    for model in ("ridge", "lgbm"):
        assert abs(summary.loc[model, "ic_mean"]) < 0.05, f"{model} found signal in noise"
        assert abs(summary.loc[model, "ic_t_stat"]) < 4.0, f"{model} found signal in noise"


def test_leakage_guard_has_teeth():
    """Positive control: deliberately leak, and confirm the null test would catch it.

    A guard that has never been shown to fail is not evidence of anything. Here
    a noisy view of the label is handed to the model as a feature — lookahead,
    on the same pure-noise panel the test above certifies as clean. If this does
    not blow through the threshold, the threshold is useless.

    The leak is deliberately *partial*. Feeding the label through cleanly pins
    the daily IC at exactly 1.0, whose standard deviation is zero, so the t-stat
    degenerates to NaN and proves nothing. A noisy leak is also the realistic
    failure mode: real lookahead arrives as a slightly-too-informative feature,
    not as a perfect copy of the answer.
    """
    from qsm.pipeline import evaluate, prepare_dataset, walk_forward
    from qsm.data import make_synthetic as _mk

    cfg = _small_cfg()
    cfg.model.models = ("ridge",)
    ds = prepare_dataset(cfg, panel=_mk(n_tickers=50, n_days=1100, seed=21, signal_strength=0.0))
    ds = ds.copy()
    rng = np.random.default_rng(0)
    ds["leaked_target"] = ds["target"] + rng.normal(0, 0.6, len(ds))

    preds, _, _ = walk_forward(ds, cfg)
    results = evaluate(preds, ds, cfg)
    ic_t = results["ridge"]["metrics"]["ic_t_stat"]
    ic_mean = results["ridge"]["metrics"]["ic_mean"]
    assert ic_mean > 0.2, f"the injected leak did not even register (ic_mean={ic_mean:.3f})"
    assert abs(ic_t) > 20, f"leak produced only ic_t_stat={ic_t:.1f}; the guard is too loose"


def test_ensemble_standardises_before_averaging():
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2020-01-01", periods=30), [f"T{i}" for i in range(25)]],
        names=["date", "ticker"],
    )
    rng = np.random.default_rng(9)
    preds = pd.DataFrame(
        {"a": rng.normal(0, 1, len(idx)), "b": rng.normal(0, 1000, len(idx))}, index=idx
    )
    out = add_ensemble(preds)
    # Without standardisation the 1000x-scale column would dominate entirely.
    corr_a = out["ensemble"].corr(out["a"])
    corr_b = out["ensemble"].corr(out["b"])
    assert abs(corr_a - corr_b) < 0.25


def test_close_is_carried_but_never_used_as_a_feature():
    """`close` rides along for charting and must not reach a model.

    A raw price level is not comparable across names — a $500 stock is not
    "higher ranked" than a $5 one — so if it ever leaked into the inputs the
    model would fit ticker identity instead of signal.
    """
    cfg = _small_cfg()
    ds = prepare_dataset(cfg, panel=make_synthetic(n_tickers=25, n_days=800, seed=11))
    assert "close" in ds.columns, "close should be carried for the per-ticker views"
    assert "close" in NON_FEATURE_COLS
    feats = feature_list(ds)
    assert "close" not in feats
    for col in ("target", "fwd_ret", "ret_next_1d"):
        assert col not in feats, f"{col} is a label, not an input"
    # Everything left really is a feature.
    assert len(feats) > 20


def test_ticker_panel_is_written_and_aligned(tmp_path):
    """The per-ticker panel must cover exactly the out-of-sample predictions."""
    import pandas as pd

    cfg = _small_cfg()
    run_dir, _, _ = run_synthetic(
        cfg, signal_strength=0.06, n_tickers=40, n_days=1100, seed=12,
        tag="pytest-panel", run_dir=tmp_path / "run",
    )
    path = run_dir / "ticker_panel.parquet"
    assert path.exists(), "ticker_panel.parquet should be written for the stock search"

    panel = pd.read_parquet(path)
    preds = pd.read_parquet(run_dir / "oos_predictions.parquet")
    assert len(panel) == len(preds)
    assert panel.index.names == ["date", "ticker"]
    for col in ("close", "ret_next_1d", "ridge", "lgbm"):
        assert col in panel.columns
    assert panel["close"].notna().mean() > 0.99
