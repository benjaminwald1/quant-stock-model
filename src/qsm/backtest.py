"""Portfolio construction, costs, and performance measurement.

The strategy is deliberately plain: rank the out-of-sample predictions within
each day, go long the top quantile and short the bottom, hold for the signal
horizon, pay to trade. Plain is the point — an exotic overlay makes it hard to
tell whether the *forecast* has any content, which is the thing under test.

Timing convention
-----------------
Predictions at date ``t`` are formed from data through the close of ``t``.
With ``execution_lag=1`` (the default) the resulting weights are applied to the
return from the close of ``t+1`` to the close of ``t+2``, i.e. you have a full
day to trade after the signal exists. Set it to 0 only if you genuinely intend
to send market-on-close orders on the signal date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BacktestConfig

TRADING_DAYS = 252


def signal_to_weights(
    signal: pd.DataFrame, cfg: BacktestConfig, holding_days: int = 1
) -> pd.DataFrame:
    """Turn a wide `date x ticker` signal into portfolio weights.

    Weights sum to zero (dollar-neutral) and their absolute values sum to
    ``gross_exposure``. Averaging the last ``holding_days`` target books is the
    standard way to run an h-day signal daily: it is equivalent to holding h
    overlapping sleeves, and it cuts turnover by roughly a factor of h without
    diluting the signal.
    """
    ranks = signal.rank(axis=1, pct=True, na_option="keep")
    n_valid = signal.notna().sum(axis=1)
    q = cfg.quantile

    enough = n_valid.ge(cfg.min_names).to_numpy()[:, None]
    long_mask = ranks.ge(1 - q) & enough
    short_mask = ranks.le(q) & enough

    if cfg.signal_weighted:
        centered = (ranks - 0.5).where(long_mask | short_mask, 0.0)
        raw = centered.fillna(0.0)
    else:
        raw = long_mask.astype(float) - (short_mask.astype(float) if cfg.long_short else 0.0)

    if not cfg.long_short:
        raw = raw.clip(lower=0)

    # Size each leg separately so the book is dollar-neutral even when the two
    # legs hold different numbers of names (they do, whenever data is missing).
    target = cfg.gross_exposure / 2 if cfg.long_short else cfg.gross_exposure
    cap = cfg.max_weight * cfg.gross_exposure if cfg.max_weight else None

    longs = _size_leg(raw.clip(lower=0), target, cap)
    if cfg.long_short:
        shorts = _size_leg(raw.clip(upper=0).abs(), target, cap)
        # A quantile cut rarely splits evenly (the top decile of 40 names is 5
        # names, the bottom is 4), so under a binding cap the two legs reach
        # different gross levels and the book would drift net long or short.
        # Dollar-neutrality is the hard constraint and gross exposure the soft
        # one, so both legs are trimmed to the smaller of the two.
        long_gross, short_gross = longs.sum(axis=1), shorts.sum(axis=1)
        achievable = pd.concat([long_gross, short_gross], axis=1).min(axis=1)
        longs = longs.mul(achievable / long_gross.replace(0, np.nan), axis=0).fillna(0.0)
        shorts = shorts.mul(achievable / short_gross.replace(0, np.nan), axis=0).fillna(0.0)
        w = longs - shorts
    else:
        w = longs

    if holding_days > 1:
        w = w.rolling(holding_days, min_periods=1).mean()
    return w.fillna(0.0)


def _size_leg(leg: pd.DataFrame, target: float, cap: float | None,
              max_iter: int = 16) -> pd.DataFrame:
    """Scale one non-negative leg to ``target`` gross, respecting a per-name cap.

    Naively capping and then re-normalising does not work: the re-normalisation
    pushes the capped names straight back over the limit, so the cap silently
    does nothing. This instead water-fills — names that hit the cap are locked
    there and the remaining exposure is redistributed among the rest, repeating
    until nothing else breaches.

    When the cap is infeasible (fewer than ``target / cap`` names available) the
    leg ends up under-invested, and that is the honest answer: a 2% per-name
    limit across 5 names supports a 10% leg, not a 50% one. Reporting the target
    gross anyway would claim a book the constraint forbids.
    """
    w = leg.astype(float).fillna(0.0)
    if cap is None:
        total = w.sum(axis=1).replace(0, np.nan)
        return w.div(total, axis=0).fillna(0.0) * target

    locked = pd.DataFrame(False, index=w.index, columns=w.columns)
    for _ in range(max_iter):
        locked_mass = w.where(locked, 0.0).sum(axis=1)
        free_mass = w.where(~locked, 0.0).sum(axis=1)
        remaining = (target - locked_mass).clip(lower=0)
        scale = remaining / free_mass.replace(0, np.nan)
        w = w.where(locked, w.mul(scale, axis=0)).fillna(0.0)

        over = (~locked) & (w > cap + 1e-12)
        if not bool(over.to_numpy().any()):
            break
        w = w.where(~over, cap)
        locked = locked | over
    return w


def run_backtest(
    signal: pd.DataFrame,
    ret_next: pd.DataFrame,
    cfg: BacktestConfig,
    holding_days: int = 1,
    execution_lag: int = 1,
) -> dict:
    """Backtest a wide signal against next-day returns.

    Returns a dict with the daily P&L series, the weight book, and metrics.
    """
    ret_next = ret_next.reindex(index=signal.index, columns=signal.columns)
    weights = signal_to_weights(signal, cfg, holding_days=holding_days)

    # Shift the book forward so today's signal is only paid tomorrow's return.
    held = weights.shift(execution_lag).fillna(0.0)

    gross_pnl = (held * ret_next.fillna(0.0)).sum(axis=1)
    traded = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    costs = traded * cfg.cost_bps / 1e4
    net_pnl = gross_pnl - costs

    equity = (1 + net_pnl).cumprod()
    bench = ret_next.mean(axis=1).fillna(0.0)  # equal-weight universe

    return {
        "weights": held,
        "gross_pnl": gross_pnl,
        "costs": costs,
        "pnl": net_pnl,
        "equity": equity,
        "benchmark": bench,
        "turnover": traded,
        "metrics": performance_metrics(net_pnl, turnover=traded, gross_pnl=gross_pnl),
        "benchmark_metrics": performance_metrics(bench),
    }


def performance_metrics(
    pnl: pd.Series, turnover: pd.Series | None = None, gross_pnl: pd.Series | None = None
) -> dict:
    """Standard performance statistics for a daily return series."""
    pnl = pnl.dropna()
    if pnl.empty:
        return {}
    n = len(pnl)
    years = n / TRADING_DAYS
    equity = (1 + pnl).cumprod()
    total = float(equity.iloc[-1] - 1)
    ann_ret = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else np.nan
    ann_vol = float(pnl.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(TRADING_DAYS)) if pnl.std(ddof=1) > 0 else np.nan

    downside = pnl[pnl < 0].std(ddof=1)
    sortino = float(pnl.mean() / downside * np.sqrt(TRADING_DAYS)) if downside and downside > 0 else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = float(drawdown.min())

    out = {
        "n_days": n,
        "years": round(years, 2),
        "total_return": total,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": float(ann_ret / abs(max_dd)) if max_dd < 0 and not np.isnan(ann_ret) else np.nan,
        "hit_rate": float((pnl > 0).mean()),
        # Under an iid null the Sharpe estimate has standard error 1/sqrt(years),
        # so this is the t-statistic on "the mean return is zero".
        "t_stat": float(sharpe * np.sqrt(years)) if not np.isnan(sharpe) else np.nan,
    }
    if turnover is not None:
        out["avg_daily_turnover"] = float(turnover.mean())
        out["ann_turnover"] = float(turnover.mean() * TRADING_DAYS)
    if gross_pnl is not None:
        g = performance_metrics(gross_pnl)
        out["sharpe_before_costs"] = g["sharpe"]
        out["ann_return_before_costs"] = g["ann_return"]
    return out


def information_coefficient(
    signal: pd.DataFrame, fwd_ret: pd.DataFrame, min_names: int = 20
) -> dict:
    """Daily rank correlation between the forecast and what actually happened.

    The IC is the cleanest read on forecast quality because it is independent of
    portfolio construction. Daily cross-sectional ICs of 0.02-0.05 are normal for
    real equity signals; anything above ~0.15 on daily data should be treated as
    a bug until proven otherwise.
    """
    fwd_ret = fwd_ret.reindex(index=signal.index, columns=signal.columns)
    s_rank = signal.rank(axis=1, na_option="keep")
    r_rank = fwd_ret.rank(axis=1, na_option="keep")
    both = signal.notna() & fwd_ret.notna()
    s_rank = s_rank.where(both)
    r_rank = r_rank.where(both)

    sm = s_rank.sub(s_rank.mean(axis=1), axis=0)
    rm = r_rank.sub(r_rank.mean(axis=1), axis=0)
    cov = (sm * rm).sum(axis=1)
    denom = np.sqrt((sm**2).sum(axis=1) * (rm**2).sum(axis=1))
    ic = (cov / denom.replace(0, np.nan)).where(both.sum(axis=1) >= min_names)
    ic = ic.dropna()
    if ic.empty:
        return {"ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan, "ic_positive_rate": np.nan}
    return {
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic.std(ddof=1)),
        "icir": float(ic.mean() / ic.std(ddof=1)) if ic.std(ddof=1) > 0 else np.nan,
        "ic_positive_rate": float((ic > 0).mean()),
        "ic_t_stat": float(ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic))) if ic.std(ddof=1) > 0 else np.nan,
        "ic_series": ic,
    }


def quantile_returns(signal: pd.DataFrame, ret_next: pd.DataFrame, n_bins: int = 5,
                     execution_lag: int = 1) -> pd.DataFrame:
    """Mean forward return by signal quantile — the monotonicity check.

    A signal you can trust produces a roughly monotone staircase from bin 1 to
    bin N. A signal that only works in the extreme bins is usually picking up
    something small, illiquid, or already dead.
    """
    ranks = signal.rank(axis=1, pct=True, na_option="keep")
    ret_next = ret_next.reindex(index=signal.index, columns=signal.columns)
    rows = {}
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = ranks.gt(lo) & ranks.le(hi) if b > 0 else ranks.le(hi)
        w = mask.astype(float)
        w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0)
        r = (w.shift(execution_lag) * ret_next.fillna(0.0)).sum(axis=1)
        rows[f"Q{b + 1}"] = {
            "mean_daily_ret_bps": float(r.mean() * 1e4),
            "ann_return": float((1 + r).prod() ** (TRADING_DAYS / max(len(r), 1)) - 1),
            "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.std(ddof=1) > 0 else np.nan,
        }
    return pd.DataFrame(rows).T
