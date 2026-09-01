"""Learning from realised mistakes as new market data arrives.

Two pieces:

* **A ledger.** Every forecast is recorded, and once its horizon has elapsed the
  realised return is joined back on. Without this there is no error signal to
  learn from — you cannot improve on mistakes you never wrote down.
* **Adaptive weights.** The ensemble stops being a flat average and leans toward
  whichever model has actually been right lately, measured on resolved
  forecasts only.

The timing rule is the whole game. A forecast made on day ``t`` over horizon
``h`` is not scoreable until ``t + h``. So the weights used on day ``t`` may
only draw on forecasts made up to ``t - h``. Using anything fresher would score
a prediction with the very outcome it was trying to predict, and the backtest
would look wonderful and be worthless.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from .config import DATA_DIR

log = logging.getLogger(__name__)

LEDGER_PATH = DATA_DIR / "ledger.parquet"
HISTORY_PATH = DATA_DIR / "update_history.json"


def score_predictions(preds: pd.DataFrame, fwd_ret: pd.DataFrame,
                      min_names: int = 20) -> pd.DataFrame:
    """Daily information coefficient per model — the error record.

    Returns a date x model frame of realised cross-sectional rank correlations.
    A row exists only once that day's forecasts have actually resolved.
    """
    out = {}
    for model in preds.columns:
        wide = preds[model].unstack()
        actual = fwd_ret.reindex(index=wide.index, columns=wide.columns)
        both = wide.notna() & actual.notna()
        sr = wide.where(both).rank(axis=1)
        ar = actual.where(both).rank(axis=1)
        sm = sr.sub(sr.mean(axis=1), axis=0)
        am = ar.sub(ar.mean(axis=1), axis=0)
        cov = (sm * am).sum(axis=1)
        denom = np.sqrt((sm ** 2).sum(axis=1) * (am ** 2).sum(axis=1))
        ic = (cov / denom.replace(0, np.nan)).where(both.sum(axis=1) >= min_names)
        out[model] = ic
    return pd.DataFrame(out)


def adaptive_weights(ic: pd.DataFrame, horizon: int, halflife: int = 60,
                     floor: float = 0.0, min_history: int = 40) -> pd.DataFrame:
    """Per-date ensemble weights from each model's recently realised skill.

    ``ic`` is shifted by ``horizon`` before any averaging, so the weight applied
    on a given day is built only from forecasts that had already resolved by
    then. Weights are non-negative and sum to one; before enough history has
    accumulated they fall back to an equal split.
    """
    resolved = ic.shift(horizon)
    smooth = resolved.ewm(halflife=halflife, min_periods=min_history).mean()

    raw = smooth.clip(lower=floor)
    total = raw.sum(axis=1)

    n = ic.shape[1]
    weights = raw.div(total.replace(0, np.nan), axis=0)
    # No history, or every model scoring at or below the floor: split evenly.
    weights = weights.fillna(1.0 / n)
    weights[total.isna() | (total <= 0)] = 1.0 / n
    return weights


def blend(preds: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Combine model predictions using per-date weights.

    Each model is standardised within a date first: a ridge score and a boosted
    tree score are on different scales, and averaging them raw would silently
    hand the ensemble to whichever has the larger variance.
    """
    pieces = {}
    for model in preds.columns:
        wide = preds[model].unstack()
        z = wide.sub(wide.mean(axis=1), axis=0).div(wide.std(axis=1).replace(0, np.nan), axis=0)
        w = weights[model].reindex(z.index).fillna(1.0 / preds.shape[1])
        pieces[model] = z.mul(w, axis=0)
    total = None
    for z in pieces.values():
        total = z if total is None else total.add(z, fill_value=0.0)
    return total.stack(future_stack=True).rename("adaptive")


def weight_summary(weights: pd.DataFrame) -> dict:
    """How much the weights actually moved — a flat result means no adaptation."""
    return {
        "mean": {c: round(float(weights[c].mean()), 4) for c in weights.columns},
        "min": {c: round(float(weights[c].min()), 4) for c in weights.columns},
        "max": {c: round(float(weights[c].max()), 4) for c in weights.columns},
        "mean_abs_change": round(float(weights.diff().abs().sum(axis=1).mean()), 5),
    }


# --------------------------------------------------------------------------
# Persistent ledger
# --------------------------------------------------------------------------
def record(preds: pd.DataFrame, run: str, as_of: pd.Timestamp | None = None) -> int:
    """Append the newest forecasts to the on-disk ledger.

    Only the most recent date is kept: that is the live forecast, the one whose
    outcome is genuinely unknown when it is written. Storing the whole
    backtested history would mix real out-of-sample calls with replayed ones.
    """
    if preds.empty:
        return 0
    dates = preds.index.get_level_values("date")
    as_of = as_of or dates.max()
    latest = preds[dates == as_of].copy()
    if latest.empty:
        return 0

    latest = latest.reset_index()
    latest["run"] = run
    latest["logged_at"] = datetime.now().isoformat(timespec="seconds")

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LEDGER_PATH.exists():
        prior = pd.read_parquet(LEDGER_PATH)
        combined = pd.concat([prior, latest], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
    else:
        combined = latest
    combined.to_parquet(LEDGER_PATH, index=False)
    return len(latest)


def resolve(closes: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Join realised returns onto logged forecasts whose horizon has elapsed."""
    if not LEDGER_PATH.exists():
        return pd.DataFrame()
    led = pd.read_parquet(LEDGER_PATH)
    led["date"] = pd.to_datetime(led["date"])

    fwd = (closes.shift(-horizon) / closes - 1).stack(future_stack=True).rename("realised")
    fwd.index.names = ["date", "ticker"]
    out = led.merge(fwd.reset_index(), on=["date", "ticker"], how="left")
    return out


def ledger_scorecard(closes: pd.DataFrame, horizon: int) -> dict:
    """How the live forecasts have actually done — the running error record."""
    res = resolve(closes, horizon)
    if res.empty:
        return {"logged": 0, "resolved": 0, "models": {}}

    done = res[res["realised"].notna()]
    models = [c for c in res.columns
              if c not in ("date", "ticker", "run", "logged_at", "realised")]
    scores = {}
    for m in models:
        sub = done[[m, "realised", "date"]].dropna()
        if len(sub) < 20:
            scores[m] = {"n": int(len(sub)), "ic": None}
            continue
        per_day = sub.groupby("date").apply(
            lambda g: g[m].corr(g["realised"], method="spearman"), include_groups=False)
        per_day = per_day.dropna()
        scores[m] = {
            "n": int(len(sub)),
            "days": int(len(per_day)),
            "ic": round(float(per_day.mean()), 5) if len(per_day) else None,
            "ic_t": round(float(per_day.mean() / per_day.std(ddof=1) * np.sqrt(len(per_day))), 2)
            if len(per_day) > 2 and per_day.std(ddof=1) > 0 else None,
        }
    return {
        "logged": int(len(res)),
        "resolved": int(len(done)),
        "pending": int(len(res) - len(done)),
        "first": str(res["date"].min().date()),
        "last": str(res["date"].max().date()),
        "models": scores,
    }


def log_update(entry: dict) -> None:
    """Append one line to the update history shown in the UI."""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    hist = []
    if HISTORY_PATH.exists():
        try:
            hist = json.loads(HISTORY_PATH.read_text())
        except Exception:
            hist = []
    hist.append({**entry, "at": datetime.now().isoformat(timespec="seconds")})
    HISTORY_PATH.write_text(json.dumps(hist[-200:], indent=2))


def update_history(limit: int = 20) -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text())[-limit:][::-1]
    except Exception:
        return []
