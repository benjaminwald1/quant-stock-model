"""End-to-end orchestration: data -> features -> walk-forward training -> backtest."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import information_coefficient, quantile_returns, run_backtest
from .config import RAW_DIR, RUNS_DIR, Config
from .data import apply_universe_filters, load_panel, make_synthetic
from .features import compute_features, feature_columns
from .labels import build_labels
from .models import build_model
from .splits import check_no_overlap, purged_walk_forward, train_val_split

log = logging.getLogger(__name__)

# Models that consume NaN natively; everything else gets neutral-filled.
NAN_TOLERANT = {"lgbm"}

# Columns carried in the modelling frame that are NOT inputs. `close` rides
# along so the per-ticker views can plot a price, and a raw price level would be
# actively harmful as a feature — it is not comparable across names and carries
# no cross-sectional information. tests/test_pipeline.py pins this down.
NON_FEATURE_COLS = ("target", "fwd_ret", "ret_next_1d", "close")


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
def load_source(cfg: Config, data_root: Path | None = None) -> pd.DataFrame:
    """Fetch the price panel from whichever source the config names."""
    if cfg.data.source == "live":
        from .live import fetch_live
        from .universe import resolve

        tickers = resolve(cfg.data.universe, list(cfg.data.tickers))
        panel = fetch_live(
            tickers,
            start=cfg.data.start,
            end=cfg.data.end,
            provider=cfg.data.provider,
            max_age_hours=cfg.data.max_age_hours,
        )
        # A live feed hands back every name asked for; the liquidity cap still
        # applies so the universe is sized the same way as the archive path.
        if panel["ticker"].nunique() > cfg.data.max_tickers:
            liq = (panel["close"] * panel["volume"]
                   * panel.get("usd_rate", 1.0)).groupby(panel["ticker"]).median()
            keep = set(liq.nlargest(cfg.data.max_tickers).index)
            panel = panel[panel["ticker"].isin(keep)].reset_index(drop=True)
        return panel

    root = Path(data_root) if data_root else RAW_DIR / cfg.data.dataset
    return load_panel(root, cfg.data)


def prepare_dataset(cfg: Config, panel: pd.DataFrame | None = None,
                    data_root: Path | None = None) -> pd.DataFrame:
    """Build the modelling frame: features + labels, restricted to tradable rows."""
    if panel is None:
        panel = load_source(cfg, data_root)

    panel = apply_universe_filters(panel, cfg.data)
    log.info("Computing features on %d rows", len(panel))
    feats = compute_features(panel, cfg.features)
    labels = build_labels(panel, cfg.labels)

    indexed = panel.set_index(["date", "ticker"])
    ds = feats.join(labels, how="inner").join(indexed[["tradable", "close"]], how="left")
    ds = ds[ds["tradable"].fillna(False)]

    # Drop rows that are mostly missing: a name with 60% of its features
    # undefined is a name without enough history to have an opinion about.
    fcols = [c for c in feature_columns(feats)]
    keep = ds[fcols].notna().mean(axis=1) >= 0.5
    ds = ds[keep].drop(columns=["tradable"])
    log.info(
        "Dataset: %d rows, %d features, %d dates",
        len(ds), len(fcols), ds.index.get_level_values("date").nunique(),
    )
    return ds


def feature_list(ds: pd.DataFrame) -> list[str]:
    """The model inputs: every column of the modelling frame that is not a label,
    a realised return, or the price carried along for charting."""
    return [c for c in ds.columns if c not in NON_FEATURE_COLS]


def _xy(ds: pd.DataFrame, fcols: list[str], dates, nan_ok: bool, need_target: bool = True):
    idx = ds.index.get_level_values("date")
    block = ds[idx.isin(dates)]
    if need_target:
        block = block[block["target"].notna()]
    X = block[fcols]
    if not nan_ok:
        X = X.fillna(0.0)
    return X, block["target"]


# --------------------------------------------------------------------------
# Walk-forward training
# --------------------------------------------------------------------------
def walk_forward(ds: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Train each model on every fold and stitch the out-of-sample predictions.

    Nothing in the returned prediction frame was produced by a model that had
    seen the date it refers to, or any date within `horizon + embargo` of it.
    """
    fcols = feature_list(ds)
    dates = pd.DatetimeIndex(sorted(ds.index.get_level_values("date").unique()))
    splits = purged_walk_forward(dates, cfg.splits, cfg.labels.horizon)
    check_no_overlap(splits, cfg.labels.horizon, cfg.splits.embargo)

    preds: dict[str, list[pd.Series]] = {m: [] for m in cfg.model.models}
    importances: dict[str, list[pd.Series]] = {m: [] for m in cfg.model.models}

    for split in splits:
        log.info(split.label)
        fit_dates, val_dates = train_val_split(
            split.train_dates, cfg.model.val_fraction, cfg.labels.horizon, cfg.splits.embargo
        )
        for name in cfg.model.models:
            nan_ok = name in NAN_TOLERANT
            X_fit, y_fit = _xy(ds, fcols, fit_dates, nan_ok)
            X_val, y_val = _xy(ds, fcols, val_dates, nan_ok) if len(val_dates) else (None, None)
            X_test, _ = _xy(ds, fcols, split.test_dates, nan_ok, need_target=False)
            if len(X_fit) == 0 or len(X_test) == 0:
                continue

            t0 = time.time()
            model = build_model(name, cfg.model).fit(X_fit, y_fit, X_val, y_val)
            p = pd.Series(model.predict(X_test), index=X_test.index, name=name)
            preds[name].append(p)
            imp = model.importance()
            if imp is not None:
                importances[name].append(imp)
            log.info(
                "  %-6s fit=%d val=%d test=%d  %.1fs",
                name, len(X_fit), len(X_val) if X_val is not None else 0, len(X_test),
                time.time() - t0,
            )

    pred_frame = pd.concat(
        [pd.concat(v).rename(k) for k, v in preds.items() if v], axis=1
    ).sort_index()

    imp_frame = pd.DataFrame(
        {k: pd.concat(v, axis=1).mean(axis=1) for k, v in importances.items() if v}
    )
    if not imp_frame.empty:
        imp_frame = imp_frame.div(imp_frame.sum(axis=0), axis=1).sort_values(
            imp_frame.columns[-1], ascending=False
        )
    return pred_frame, imp_frame, splits


def add_ensemble(preds: pd.DataFrame) -> pd.DataFrame:
    """Average the models after standardising each one within each date.

    Raw prediction scales differ wildly between a ridge and a boosted tree, so
    they have to be put on a common footing before averaging — otherwise the
    ensemble is just whichever model happens to have the largest variance.
    """
    if preds.shape[1] < 2:
        return preds
    z = {}
    for col in preds.columns:
        wide = preds[col].unstack()
        zz = wide.sub(wide.mean(axis=1), axis=0).div(wide.std(axis=1).replace(0, np.nan), axis=0)
        z[col] = zz.stack(future_stack=True)
    zf = pd.DataFrame(z)
    out = preds.copy()
    out["ensemble"] = zf.mean(axis=1)
    return out


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(preds: pd.DataFrame, ds: pd.DataFrame, cfg: Config) -> dict:
    """Backtest and score every prediction column."""
    # Restrict everything to the out-of-sample window. Leaving the in-sample
    # dates in would pad the P&L with flat days, which quietly deflates the
    # volatility, the Sharpe and the hit rate all at once.
    oos = pd.DatetimeIndex(sorted(preds.index.get_level_values("date").unique()))
    ret_next = ds["ret_next_1d"].unstack().reindex(index=oos)
    fwd = ds["fwd_ret"].unstack().reindex(index=oos)
    results = {}
    for col in preds.columns:
        signal = preds[col].unstack().reindex(index=ret_next.index, columns=ret_next.columns)
        bt = run_backtest(
            signal, ret_next, cfg.backtest,
            holding_days=cfg.labels.horizon, execution_lag=cfg.backtest.execution_lag,
        )
        ic = information_coefficient(signal, fwd, min_names=cfg.backtest.min_names)
        ic_series = ic.pop("ic_series", None)
        qr = quantile_returns(
            signal, ret_next, cfg.backtest.n_quantile_bins, cfg.backtest.execution_lag
        )
        results[col] = {
            "metrics": {**bt["metrics"], **ic},
            "equity": bt["equity"],
            "pnl": bt["pnl"],
            "benchmark": bt["benchmark"],
            "benchmark_metrics": bt["benchmark_metrics"],
            "quantiles": qr,
            "ic_series": ic_series,
        }
    return results


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def summary_table(results: dict) -> pd.DataFrame:
    keys = [
        "sharpe", "sharpe_before_costs", "ann_return", "ann_vol", "max_drawdown",
        "hit_rate", "t_stat", "ic_mean", "icir", "ic_t_stat", "ic_positive_rate",
        "avg_daily_turnover", "n_days",
    ]
    rows = {name: {k: r["metrics"].get(k, np.nan) for k in keys} for name, r in results.items()}
    first = next(iter(results.values()))
    bench = first["benchmark_metrics"]
    rows["buy&hold universe"] = {k: bench.get(k, np.nan) for k in keys}
    return pd.DataFrame(rows).T


def save_report(run_dir: Path, cfg: Config, results: dict, importances: pd.DataFrame,
                preds: pd.DataFrame, splits: list, ds: pd.DataFrame | None = None) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(run_dir / "config.json")

    summary = summary_table(results)
    summary.to_csv(run_dir / "summary.csv")
    if not importances.empty:
        importances.to_csv(run_dir / "feature_importance.csv")
    preds.to_parquet(run_dir / "oos_predictions.parquet")

    # Per-ticker panel for the stock search / drill-down views: the model's
    # out-of-sample scores next to the price and the return actually realised.
    if ds is not None:
        cols = [c for c in ("close", "ret_next_1d", "fwd_ret") if c in ds.columns]
        preds.join(ds[cols], how="left").to_parquet(run_dir / "ticker_panel.parquet")

    best = max(results, key=lambda k: results[k]["metrics"].get("sharpe", -np.inf))
    curves = pd.DataFrame({k: v["equity"] for k, v in results.items()})
    curves["buy&hold universe"] = (1 + results[best]["benchmark"]).cumprod()
    curves.to_csv(run_dir / "equity_curves.csv")
    results[best]["quantiles"].to_csv(run_dir / "quantile_returns.csv")

    (run_dir / "folds.txt").write_text("\n".join(s.label for s in splits))
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {k: {m: (None if isinstance(v, float) and np.isnan(v) else v)
                 for m, v in r["metrics"].items()} for k, r in results.items()},
            indent=2, default=float,
        )
    )
    _plot(run_dir, curves, results[best], best)
    return run_dir


def _plot(run_dir: Path, curves: pd.DataFrame, best: dict, best_name: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), gridspec_kw={"height_ratios": [2, 1, 1]})
    for col in curves.columns:
        style = "--" if col == "buy&hold universe" else "-"
        axes[0].plot(curves.index, curves[col], style, linewidth=1.4, label=col)
    axes[0].set_title("Out-of-sample equity curves (net of costs)")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    eq = best["equity"]
    dd = eq / eq.cummax() - 1
    axes[1].fill_between(dd.index, dd.to_numpy(), 0, color="crimson", alpha=0.4)
    axes[1].set_title(f"Drawdown — {best_name}")
    axes[1].grid(alpha=0.3)

    ic = best.get("ic_series")
    if ic is not None and len(ic):
        axes[2].bar(ic.index, ic.rolling(21).mean().to_numpy(), width=2.0, color="steelblue")
        axes[2].axhline(0, color="black", linewidth=0.8)
        axes[2].set_title(f"21-day rolling information coefficient — {best_name}")
        axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(run_dir / "report.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------
def run(cfg: Config, panel: pd.DataFrame | None = None, data_root: Path | None = None,
        run_dir: Path | None = None, tag: str = "run") -> tuple[Path, pd.DataFrame, dict]:
    ds = prepare_dataset(cfg, panel=panel, data_root=data_root)
    preds, importances, splits = walk_forward(ds, cfg)
    preds = add_ensemble(preds)
    results = evaluate(preds, ds, cfg)
    run_dir = run_dir or RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{tag}"
    save_report(run_dir, cfg, results, importances, preds, splits, ds=ds)
    return run_dir, summary_table(results), results


def run_synthetic(cfg: Config | None = None, signal_strength: float = 0.05,
                  n_tickers: int = 120, n_days: int = 2600, seed: int = 0,
                  tag: str = "synthetic", run_dir: Path | None = None
                  ) -> tuple[Path, pd.DataFrame, dict]:
    """Run the full pipeline on generated data. Used for smoke tests and for the
    null experiment (``signal_strength=0`` must score ~0 Sharpe)."""
    cfg = cfg or Config()
    panel = make_synthetic(
        n_tickers=n_tickers, n_days=n_days, seed=seed, signal_strength=signal_strength
    )
    return run(cfg, panel=panel, tag=tag, run_dir=run_dir)


# --------------------------------------------------------------------------
# Live update cycle
# --------------------------------------------------------------------------
def update(cfg: Config, tag: str = "update") -> dict:
    """Refetch the market, retrain, log today's forecasts, score the old ones.

    This is what "learning from its mistakes" actually consists of:

    * the model is retrained on data that now includes everything the market has
      done since the last run, so yesterday's errors are part of the fit;
    * every live forecast is written to a ledger, and once its horizon elapses
      the realised return is joined on, giving a running record of how the model
      is doing *out of sample in real time* rather than in replay.

    Note what it deliberately does not do: reweight the models toward whichever
    has been hot lately. That was measured (experiments/adaptive.py) and made
    results worse at every setting tried.
    """
    from . import online

    cfg.data.max_age_hours = 0.0        # always pull fresh bars
    panel = load_source(cfg)
    ds = prepare_dataset(cfg, panel=panel)
    preds, importances, splits = walk_forward(ds, cfg)
    preds = add_ensemble(preds)
    results = evaluate(preds, ds, cfg)

    run_dir = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{tag}"
    save_report(run_dir, cfg, results, importances, preds, splits, ds=ds)

    logged = online.record(preds, run=run_dir.name)
    closes = panel.pivot(index="date", columns="ticker", values="close").sort_index()
    scorecard = online.ledger_scorecard(closes, cfg.labels.horizon)

    entry = {
        "run": run_dir.name,
        "universe": cfg.data.universe,
        "tickers": int(panel["ticker"].nunique()),
        "last_bar": str(panel["date"].max().date()),
        "forecasts_logged": logged,
        "ledger_resolved": scorecard.get("resolved", 0),
        "ledger_pending": scorecard.get("pending", 0),
        "live_ic": {m: v.get("ic") for m, v in scorecard.get("models", {}).items()},
        "backtest_ic": {m: round(float(r["metrics"].get("ic_mean") or 0), 5)
                        for m, r in results.items()},
    }
    online.log_update(entry)
    return {"run_dir": run_dir, "summary": summary_table(results), "scorecard": scorecard,
            "entry": entry}
