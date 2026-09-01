"""Tests for the per-ticker search and drill-down views."""

import numpy as np
import pytest

from qsm.config import Config
from qsm.pipeline import run_synthetic
from qsm.web.stocks import detail, load_view, search, summarise


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    cfg = Config()
    cfg.splits.n_splits = 2
    cfg.splits.min_train_size = 500
    cfg.splits.test_size = 200
    cfg.data.min_history_days = 260
    cfg.model.lgbm_params["n_estimators"] = 120
    d, _, _ = run_synthetic(
        cfg, signal_strength=0.06, n_tickers=60, n_days=1100, seed=31,
        tag="pytest-stocks", run_dir=tmp_path_factory.mktemp("run") / "r",
    )
    return d


def test_contributions_reconcile_with_the_book(run_dir):
    """Per-stock P&L must sum to the book's gross P&L.

    If these two ever drift apart, the stock page is quietly telling a different
    story from the headline metrics, which is worse than showing nothing.
    """
    view = load_view(run_dir)
    per_stock = summarise(view)["contribution"].sum()
    book = view.contrib.sum(axis=1).sum()
    assert abs(per_stock - book) < 1e-9


def test_weights_match_the_backtest_convention(run_dir):
    """The view must use the run's own execution lag and dollar-neutrality."""
    view = load_view(run_dir)
    assert view.weights.sum(axis=1).abs().max() < 1e-9, "book must stay dollar-neutral"
    # execution_lag=1 by default, so the first row is never invested.
    assert view.weights.iloc[0].abs().sum() < 1e-12


def test_search_filters_by_substring(run_dir):
    view = load_view(run_dir)
    everything = search(view, limit=500)
    assert everything["total_matches"] == everything["universe"]

    hit = search(view, q="SYN00", limit=500)
    assert 0 < hit["total_matches"] < everything["total_matches"]
    assert all("SYN00" in r["ticker"] for r in hit["rows"])

    assert search(view, q="ZZZZ")["total_matches"] == 0


def test_search_is_case_insensitive(run_dir):
    view = load_view(run_dir)
    assert search(view, q="syn01")["total_matches"] == search(view, q="SYN01")["total_matches"]


def test_position_filter_matches_reported_position(run_dir):
    view = load_view(run_dir)
    longs = search(view, position="long", limit=500)
    shorts = search(view, position="short", limit=500)
    assert longs["total_matches"] > 0 and shorts["total_matches"] > 0
    assert all(r["position"] == "long" and r["latest_weight"] > 0 for r in longs["rows"])
    assert all(r["position"] == "short" and r["latest_weight"] < 0 for r in shorts["rows"])
    # A dollar-neutral decile book holds both sides.
    assert abs(longs["total_matches"] - shorts["total_matches"]) <= max(
        3, 0.5 * longs["total_matches"])


def test_sorting_orders_correctly(run_dir):
    view = load_view(run_dir)
    by_rank = [r["latest_rank"] for r in search(view, sort="rank", limit=20)["rows"]]
    assert by_rank == sorted(by_rank, reverse=True)

    best = [r["contribution"] for r in search(view, sort="contribution", limit=20)["rows"]]
    assert best == sorted(best, reverse=True)

    worst = [r["contribution"] for r in search(view, sort="worst", limit=20)["rows"]]
    assert worst == sorted(worst)

    names = [r["ticker"] for r in search(view, sort="ticker", limit=20)["rows"]]
    assert names == sorted(names)


def test_limit_is_respected_but_total_is_honest(run_dir):
    view = load_view(run_dir)
    res = search(view, limit=5)
    assert res["returned"] == 5
    assert res["total_matches"] == res["universe"] > 5


def test_detail_is_consistent_with_the_summary_row(run_dir):
    view = load_view(run_dir)
    ticker = search(view, limit=1)["rows"][0]["ticker"]
    d = detail(view, ticker)
    row = summarise(view).loc[ticker]

    assert d["ticker"] == ticker
    assert len(d["dates"]) == len(d["rank"]) == len(d["cum_contribution"])
    assert d["stats"]["contribution"] == pytest.approx(row["contribution"], abs=1e-6)
    assert d["stats"]["days_long"] == int(row["days_long"])
    assert d["stats"]["days_short"] == int(row["days_short"])


def test_detail_is_case_insensitive_and_404s_cleanly(run_dir):
    view = load_view(run_dir)
    ticker = view.signal.columns[0]
    assert detail(view, ticker.lower())["ticker"] == ticker
    with pytest.raises(KeyError):
        detail(view, "NOT_A_TICKER")


def test_ranks_are_percentiles(run_dir):
    view = load_view(run_dir)
    r = view.rank.to_numpy()
    r = r[~np.isnan(r)]
    assert r.min() >= 0 and r.max() <= 100


def test_missing_panel_raises_a_useful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Re-run"):
        load_view(tmp_path)
