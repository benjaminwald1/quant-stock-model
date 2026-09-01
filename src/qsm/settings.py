"""User preferences, stored server-side.

Kept next to the watchlist rather than in browser storage for the same reason:
settings should follow the user, not the browser profile they happened to open.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from .config import DATA_DIR

log = logging.getLogger(__name__)

SETTINGS_PATH = DATA_DIR / "settings.json"

DEFAULTS: dict = {
    # display
    "theme": "system",             # system | light | dark
    "detail_level": "simple",      # simple | detailed
    "default_range": "6mo",
    "currency_symbols": True,
    # run defaults
    "default_universe": "sp500",
    "default_horizon": 5,
    "default_folds": 6,
    "default_cost_bps": 10.0,
    "default_quantile": 0.1,
    "default_models": ["ridge", "lgbm"],
    # live behaviour
    "live_autoupdate": False,
    "live_interval_s": 20,
    "watchlist_autoupdate": True,
    "dip_pct": 0.01,               # limit-order depth below the reference price
    "fund_sizing": "equity",       # size slots off total equity, so gains compound
    # Trading frequency, chosen out of sample in experiments/tune_exit.py: the
    # band was picked on 2022-2024 and scored on 2025-2026, which had no say in
    # the choice. Selling at 30 rather than 70 cut trading roughly fourfold and
    # raised the holdout years from 3.36% to 12.30% a year. Trading *less* is
    # what helped; every faster setting was worse on both halves.
    #
    # `fund_enter_above` is close to inert on a large universe — with ten slots
    # and thousands of names above any plausible line, the top ten are the same
    # either way. It still bites on a small universe like dow30.
    "fund_enter_above": 90.0,
    "fund_exit_below": 30.0,
    "fund_max_positions": 10,
    # Sell winners at a peak. Off because it was measured and it loses: on the
    # holdout, 5.49% a year selling at a 60-session high and 8.87% selling at
    # the model's 80% upper band, against 12.30% for leaving it to the rank
    # rule — worse in every window tested, at 2-3x the trades. See
    # experiments/peak_exit.py.
    "fund_take_profit": "off",
    # learning
    "auto_learn": True,            # retrain and score the ledger once a day
    "auto_learn_universe": "us_all",   # blank = repeat whatever the last run used
    "auto_learn_years": 15,            # how far back to train each night
    "auto_learn_max_tickers": 6000,    # cap after the liquidity filters
}

_ENUMS = {
    "fund_sizing": {"equity", "cash"},
    "fund_take_profit": {"off", "peak"},
    "theme": {"system", "light", "dark"},
    "detail_level": {"simple", "detailed"},
    "default_range": {"1d", "3d", "5d", "1mo", "3mo", "6mo", "ytd", "1y", "max"},
}
_BOUNDS = {
    "fund_enter_above": (50.0, 100.0),
    "fund_exit_below": (0.0, 99.0),
    "fund_max_positions": (1, 50),
    "auto_learn_years": (1, 40),
    "auto_learn_max_tickers": (30, 10_000),
    "default_horizon": (1, 63),
    "default_folds": (1, 12),
    "default_cost_bps": (0.0, 200.0),
    "default_quantile": (0.01, 0.5),
    "live_interval_s": (5, 600),
    "dip_pct": (0.001, 0.10),
}


def get() -> dict:
    out = dict(DEFAULTS)
    if SETTINGS_PATH.exists():
        try:
            out.update({k: v for k, v in json.loads(SETTINGS_PATH.read_text()).items()
                        if k in DEFAULTS})
        except Exception as exc:
            log.warning("settings unreadable (%s); using defaults", exc)
    return out


def update(patch: dict) -> dict:
    """Validate and persist a partial update; unknown keys are ignored."""
    current = get()
    errors = []
    for key, value in (patch or {}).items():
        if key not in DEFAULTS:
            continue
        if key in _ENUMS and value not in _ENUMS[key]:
            errors.append(f"{key} must be one of {sorted(_ENUMS[key])}")
            continue
        if key in _BOUNDS:
            lo, hi = _BOUNDS[key]
            try:
                value = type(DEFAULTS[key])(value)
            except (TypeError, ValueError):
                errors.append(f"{key} must be a number")
                continue
            if not lo <= value <= hi:
                errors.append(f"{key} must be between {lo} and {hi}")
                continue
        if key == "default_models":
            value = [m for m in (value or []) if m in ("ridge", "lgbm", "gru")]
            if not value:
                errors.append("default_models must include at least one model")
                continue
        if isinstance(DEFAULTS[key], bool):
            value = bool(value)
        current[key] = value

    if errors:
        raise ValueError("; ".join(errors))

    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(
        {**current, "updated": datetime.now().isoformat(timespec="seconds")}, indent=2))
    return current


def reset() -> dict:
    if SETTINGS_PATH.exists():
        SETTINGS_PATH.unlink()
    return dict(DEFAULTS)
