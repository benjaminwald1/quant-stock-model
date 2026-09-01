"""Predictive features.

Everything here is computed on wide ``date x ticker`` matrices. That layout is
fast, and — more importantly — it makes lookahead hard to write by accident:
a rolling window over a column only ever reaches backwards, and cross-sectional
operations only ever reach across a single row (one date).

The contract for every feature: the value at date ``t`` uses only information
available at the close of ``t``. ``tests/test_no_lookahead.py`` enforces it by
recomputing on a truncated history and diffing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig


def to_wide(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pivot the long panel into one `date x ticker` matrix per price field."""
    wide = {}
    for col in ("open", "high", "low", "close", "volume"):
        wide[col] = panel.pivot(index="date", columns="ticker", values=col).sort_index()
    if "tradable" in panel.columns:
        wide["tradable"] = (
            panel.pivot(index="date", columns="ticker", values="tradable")
            .sort_index()
            .fillna(False)
            .astype(bool)
        )
    return wide


def _rsi(close: pd.DataFrame, window: int) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int) -> pd.DataFrame:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).stack(), (high - prev_close).abs().stack(), (low - prev_close).abs().stack()],
        axis=1,
    ).max(axis=1).unstack()
    return tr.reindex_like(close).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def _rolling_beta(ret: pd.DataFrame, mkt: pd.Series, window: int) -> pd.DataFrame:
    """Rolling beta of each column against the equal-weight market return."""
    mkt_mean = mkt.rolling(window, min_periods=window // 2).mean()
    mkt_var = mkt.rolling(window, min_periods=window // 2).var()
    xm = ret.mul(mkt, axis=0).rolling(window, min_periods=window // 2).mean()
    x_mean = ret.rolling(window, min_periods=window // 2).mean()
    cov = xm.sub(x_mean.mul(mkt_mean, axis=0))
    return cov.div(mkt_var.replace(0, np.nan), axis=0)


def compute_features(panel: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Build the feature matrix. Returns a long frame indexed by (date, ticker)."""
    cfg = cfg or FeatureConfig()
    w = to_wide(panel)
    close, high, low, open_, volume = w["close"], w["high"], w["low"], w["open"], w["volume"]

    ret1 = close.pct_change()
    dollar_vol = close * volume
    mkt = ret1.mean(axis=1)  # equal-weight market return, cross-sectional

    feats: dict[str, pd.DataFrame] = {}

    # --- momentum / trend -------------------------------------------------
    for n in cfg.momentum_windows:
        feats[f"ret_{n}d"] = close.pct_change(n)
    # Classic 12-1 momentum: one year of return, skipping the most recent month
    # (that last month is short-term reversal, not momentum).
    feats["mom_12_1"] = close.shift(21) / close.shift(252) - 1

    sma20 = close.rolling(20, min_periods=15).mean()
    sma50 = close.rolling(50, min_periods=35).mean()
    sma200 = close.rolling(200, min_periods=150).mean()
    feats["close_over_sma50"] = close / sma50 - 1
    feats["close_over_sma200"] = close / sma200 - 1
    feats["sma20_over_sma50"] = sma20 / sma50 - 1

    high_252 = close.rolling(252, min_periods=180).max()
    low_252 = close.rolling(252, min_periods=180).min()
    feats["dist_52w_high"] = close / high_252 - 1
    feats["dist_52w_low"] = close / low_252 - 1

    # --- volatility -------------------------------------------------------
    vols = {}
    for n in cfg.vol_windows:
        vols[n] = ret1.rolling(n, min_periods=max(5, n // 2)).std()
        feats[f"vol_{n}d"] = vols[n]
    if len(cfg.vol_windows) >= 2:
        short, long = cfg.vol_windows[0], cfg.vol_windows[-1]
        feats["vol_ratio"] = vols[short] / vols[long].replace(0, np.nan) - 1
    base_vol = vols[cfg.vol_windows[0]].replace(0, np.nan)

    feats["atr_pct"] = _atr(high, low, close, cfg.atr_window) / close
    feats["range_5d"] = ((high - low) / close).rolling(5, min_periods=3).mean()

    # --- short-term reversal (vol-scaled, so it is comparable across names) --
    feats["rev_1d"] = -ret1 / base_vol
    feats["rev_5d"] = -close.pct_change(5) / (base_vol * np.sqrt(5))

    # --- oscillators ------------------------------------------------------
    feats["rsi"] = _rsi(close, cfg.rsi_window)
    ema12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
    ema26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
    macd = ema12 - ema26
    feats["macd_hist"] = (macd - macd.ewm(span=9, min_periods=9, adjust=False).mean()) / close
    bb_std = close.rolling(cfg.bollinger_window, min_periods=cfg.bollinger_window // 2).std()
    bb_mid = close.rolling(cfg.bollinger_window, min_periods=cfg.bollinger_window // 2).mean()
    feats["bollinger_pctb"] = (close - bb_mid) / (2 * bb_std.replace(0, np.nan))

    # --- volume / liquidity ----------------------------------------------
    log_dv = np.log(dollar_vol.replace(0, np.nan))
    dv_mean = log_dv.rolling(63, min_periods=30).mean()
    dv_std = log_dv.rolling(63, min_periods=30).std()
    feats["dollar_vol_z"] = (log_dv - dv_mean) / dv_std.replace(0, np.nan)
    feats["dollar_vol_log"] = log_dv
    feats["vol_trend"] = (
        dollar_vol.rolling(5, min_periods=3).mean()
        / dollar_vol.rolling(63, min_periods=30).mean().replace(0, np.nan)
        - 1
    )
    # Amihud illiquidity: price impact per dollar traded.
    feats["amihud"] = np.log(
        (ret1.abs() / dollar_vol.replace(0, np.nan)).rolling(21, min_periods=10).mean() + 1e-15
    )

    # --- intraday shape ---------------------------------------------------
    feats["gap"] = (open_ / close.shift(1) - 1) / base_vol
    feats["intraday_ret"] = (close / open_ - 1) / base_vol
    feats["close_loc"] = (close - low) / (high - low).replace(0, np.nan)

    # --- market relative --------------------------------------------------
    beta = _rolling_beta(ret1, mkt, cfg.beta_window)
    feats["beta"] = beta
    mkt_21 = mkt.rolling(21, min_periods=15).sum()
    feats["idio_mom_21d"] = close.pct_change(21) - beta.mul(mkt_21, axis=0)
    resid = ret1.sub(beta.mul(mkt, axis=0))
    feats["idio_vol_63d"] = resid.rolling(63, min_periods=30).std()

    # --- assemble ---------------------------------------------------------
    frames = []
    for name, mat in feats.items():
        mat = mat.replace([np.inf, -np.inf], np.nan)
        if cfg.cross_sectional_rank:
            mat = _cross_sectional_rank(mat)
        frames.append(mat.stack(future_stack=True).rename(name))

    out = pd.concat(frames, axis=1)
    out.index.names = ["date", "ticker"]
    return out.sort_index()


def _cross_sectional_rank(mat: pd.DataFrame, min_names: int = 20) -> pd.DataFrame:
    """Rank each date's cross-section into [-0.5, 0.5].

    Ranking within a date is what turns a raw quantity into a *relative* signal.
    It also neutralises regime shifts: a 3% daily move means something very
    different in 2008 than in 2017, but "top decile today" always means the same.
    """
    valid = mat.notna().sum(axis=1)
    ranked = mat.rank(axis=1, pct=True, na_option="keep") - 0.5
    return ranked.where(valid.ge(min_names), np.nan)


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if not c.startswith("_")]
