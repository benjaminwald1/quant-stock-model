"""Per-ticker views over a finished run: search, ranking, and drill-down.

Everything here is derived from `ticker_panel.parquet` — the out-of-sample
model scores sitting next to the price and the return actually realised.

Position sizes come from the same `signal_to_weights` the backtest uses, driven
by the run's own saved config. That matters: a per-stock P&L computed by some
parallel bit of arithmetic would drift away from the headline numbers and
quietly contradict them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest import signal_to_weights
from ..config import BacktestConfig
from ..live import SYMBOL, currency_of

_CACHE: dict[tuple, "StockView"] = {}
_CACHE_LIMIT = 6


@dataclass
class StockView:
    model: str
    models: list[str]
    signal: pd.DataFrame       # date x ticker, raw model score
    rank: pd.DataFrame         # date x ticker, cross-sectional percentile 0-100
    weights: pd.DataFrame      # date x ticker, as actually held
    ret_next: pd.DataFrame
    close: pd.DataFrame
    contrib: pd.DataFrame      # date x ticker, P&L contribution


def _pick_model(cols: list[str], model: str | None) -> str:
    candidates = [c for c in cols if c not in ("close", "ret_next_1d", "fwd_ret")]
    if not candidates:
        raise ValueError("The ticker panel has no model columns.")
    if model and model in candidates:
        return model
    return "ensemble" if "ensemble" in candidates else candidates[0]


def load_view(run_dir: Path, model: str | None = None) -> StockView:
    """Build (and memoise) the per-ticker view for one run and model."""
    path = run_dir / "ticker_panel.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "This run predates the stock search and has no ticker_panel.parquet. "
            "Re-run to enable per-stock views."
        )

    key = (str(run_dir), model, path.stat().st_mtime)
    if key in _CACHE:
        return _CACHE[key]

    panel = pd.read_parquet(path)
    chosen = _pick_model(list(panel.columns), model)
    model_cols = [c for c in panel.columns if c not in ("close", "ret_next_1d", "fwd_ret")]

    signal = panel[chosen].unstack()
    ret_next = panel["ret_next_1d"].unstack().reindex_like(signal)
    close = (panel["close"].unstack().reindex_like(signal)
             if "close" in panel.columns else pd.DataFrame(index=signal.index))

    cfg = BacktestConfig()
    horizon = 5
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        raw = json.loads(cfg_path.read_text())
        cfg = BacktestConfig(**raw["backtest"])
        horizon = int(raw["labels"]["horizon"])

    weights = signal_to_weights(signal, cfg, holding_days=horizon)
    held = weights.shift(cfg.execution_lag).fillna(0.0)
    contrib = held * ret_next.fillna(0.0)

    n_valid = signal.notna().sum(axis=1)
    rank = (signal.rank(axis=1, pct=True, na_option="keep") * 100).where(n_valid.ge(2))

    view = StockView(chosen, model_cols, signal, rank, held, ret_next, close, contrib)
    if len(_CACHE) >= _CACHE_LIMIT:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = view
    return view


def _f(v) -> float | None:
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    except (TypeError, ValueError):
        return None


def summarise(view: StockView) -> pd.DataFrame:
    """One row per ticker over the whole out-of-sample window."""
    last = view.rank.index[-1]
    last_rank = view.rank.loc[last]
    last_w = view.weights.loc[last]

    total = view.contrib.sum(axis=0)
    days = view.signal.notna().sum(axis=0)
    long_days = (view.weights > 1e-12).sum(axis=0)
    short_days = (view.weights < -1e-12).sum(axis=0)
    traded = long_days + short_days
    win = (view.contrib > 0).sum(axis=0) / traded.replace(0, np.nan)

    price = view.close.ffill().iloc[-1] if not view.close.empty else pd.Series(index=days.index, dtype=float)
    px_chg = (view.close.iloc[-1] / view.close.iloc[0] - 1) if not view.close.empty else pd.Series(index=days.index, dtype=float)

    out = pd.DataFrame({
        "latest_rank": last_rank,
        "latest_weight": last_w,
        "mean_rank": view.rank.mean(axis=0),
        "contribution": total,
        "days": days,
        "days_long": long_days,
        "days_short": short_days,
        "hit_rate": win,
        "last_price": price,
        "price_change": px_chg,
    })
    out.index.name = "ticker"
    return out


SORTS = {
    "rank": ("latest_rank", False),
    "contribution": ("contribution", False),
    "worst": ("contribution", True),
    "ticker": (None, True),
    "days_long": ("days_long", False),
    "days_short": ("days_short", False),
}


def search(view: StockView, q: str = "", position: str = "all",
           sort: str = "rank", limit: int = 60) -> dict:
    """Filter and rank the universe for the stock browser."""
    table = summarise(view)

    if q:
        needle = q.strip().upper()
        table = table[table.index.str.upper().str.contains(needle, regex=False)]

    if position == "long":
        table = table[table["latest_weight"] > 1e-12]
    elif position == "short":
        table = table[table["latest_weight"] < -1e-12]
    elif position == "held":
        table = table[table["latest_weight"].abs() > 1e-12]

    total = len(table)
    col, ascending = SORTS.get(sort, SORTS["rank"])
    table = table.sort_index(ascending=True) if col is None else \
        table.sort_values(col, ascending=ascending, na_position="last")

    rows = []
    for ticker, r in table.head(limit).iterrows():
        w = r["latest_weight"]
        cur = currency_of(str(ticker))
        rows.append({
            "ticker": ticker,
            "currency": cur,
            "symbol": SYMBOL.get(cur, ""),
            "latest_rank": _f(r["latest_rank"]),
            "latest_weight": _f(w),
            "position": "long" if w > 1e-12 else ("short" if w < -1e-12 else "flat"),
            "mean_rank": _f(r["mean_rank"]),
            "contribution": _f(r["contribution"]),
            "days": int(r["days"]) if r["days"] == r["days"] else 0,
            "days_long": int(r["days_long"]),
            "days_short": int(r["days_short"]),
            "hit_rate": _f(r["hit_rate"]),
            "last_price": _f(r["last_price"]),
            "price_change": _f(r["price_change"]),
        })

    return {
        "model": view.model,
        "models": view.models,
        "as_of": str(view.rank.index[-1].date()),
        "total_matches": total,
        "universe": int(view.signal.shape[1]),
        "returned": len(rows),
        "rows": rows,
    }


def detail(view: StockView, ticker: str, max_points: int = 700) -> dict:
    """Full history for one name: price, signal rank, position, cumulative P&L."""
    match = [c for c in view.signal.columns if c.upper() == ticker.upper()]
    if not match:
        raise KeyError(ticker)
    col = match[0]

    frame = pd.DataFrame({
        "rank": view.rank[col],
        "weight": view.weights[col],
        "contrib": view.contrib[col],
        "ret": view.ret_next[col],
    })
    if col in view.close.columns:
        frame["close"] = view.close[col]
    frame = frame[frame["rank"].notna() | frame["weight"].abs().gt(0)]
    frame["cum"] = frame["contrib"].fillna(0.0).cumsum()

    step = max(1, len(frame) // max_points)
    thin = frame.iloc[::step]

    traded = int((frame["weight"].abs() > 1e-12).sum())
    return {
        "ticker": col,
        "model": view.model,
        "dates": [str(d.date()) for d in thin.index],
        "close": [_f(v) for v in thin["close"]] if "close" in thin else [],
        "rank": [_f(v) for v in thin["rank"]],
        "weight": [_f(v) for v in thin["weight"]],
        "cum_contribution": [_f(v) for v in thin["cum"]],
        "stats": {
            "days": int(frame["rank"].notna().sum()),
            "days_traded": traded,
            "days_long": int((frame["weight"] > 1e-12).sum()),
            "days_short": int((frame["weight"] < -1e-12).sum()),
            "contribution": _f(frame["contrib"].sum()),
            "mean_rank": _f(frame["rank"].mean()),
            "hit_rate": _f((frame["contrib"] > 0).sum() / traded) if traded else None,
            "best_day": _f(frame["contrib"].max()),
            "worst_day": _f(frame["contrib"].min()),
        },
    }
