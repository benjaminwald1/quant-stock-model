"""Turning model scores into a forward value estimate — with its error bars.

The model does not predict prices. It predicts a *cross-sectional rank*: which
names should outperform their peers over the next `horizon` days. Getting from
that to "AAPL will be worth $X" needs one honest extra step, done here:

1. Take the out-of-sample predictions and the returns that actually followed.
2. Bin the predictions and measure, empirically, what each bin went on to earn
   and how widely those outcomes were spread.
3. Apply that mapping to today's score, and attach the spread as an interval.

Step 3 is the part most tools skip, and it is the only part that keeps the
number honest. For daily equity signals the spread is many times the central
estimate — the forecast is a faint tilt inside an enormous cloud of noise, and
a bare point estimate hides exactly that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 80% interval under a normal approximation. The realised distribution of
# equity returns is fatter-tailed than this, so treat it as a floor on the
# uncertainty, never a ceiling.
Z80 = 1.2816


def calibrate(signal: pd.DataFrame, fwd_ret: pd.DataFrame, n_bins: int = 10) -> dict:
    """Learn what each signal bin actually earned, out of sample.

    `fwd_ret` is the realised forward return over the label horizon. Returns
    are demeaned per date, so the result is performance *relative to the
    universe* — the only thing a dollar-neutral ranking can claim to forecast.
    """
    fwd = fwd_ret.reindex(index=signal.index, columns=signal.columns)
    excess = fwd.sub(fwd.mean(axis=1), axis=0)

    ranks = signal.rank(axis=1, pct=True, na_option="keep")
    both = ranks.notna() & excess.notna()
    r = ranks.where(both).to_numpy().ravel()
    e = excess.where(both).to_numpy().ravel()
    ok = ~np.isnan(r) & ~np.isnan(e)
    r, e = r[ok], e[ok]
    if len(r) < 500:
        raise ValueError("Not enough out-of-sample observations to calibrate.")

    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(r, edges[1:-1], right=True), 0, n_bins - 1)

    bins = []
    for b in range(n_bins):
        vals = e[idx == b]
        if len(vals) < 20:
            bins.append({"bin": b + 1, "n": int(len(vals)), "mean": 0.0,
                         "std": float(np.nanstd(e)), "hit": None})
            continue
        bins.append({
            "bin": b + 1,
            "n": int(len(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "hit": float(np.mean(vals > 0)),
        })

    means = np.array([b["mean"] for b in bins])
    return {
        "bins": bins,
        "n_bins": n_bins,
        "observations": int(len(r)),
        # Does the staircase actually ascend? Spearman of bin index vs outcome.
        "monotonicity": float(pd.Series(means).corr(pd.Series(range(n_bins)), method="spearman")),
        "spread_top_minus_bottom": float(means[-1] - means[0]),
        "pooled_std": float(np.std(e, ddof=1)),
    }


def project(
    latest_rank: pd.Series,
    prices: dict[str, float],
    calibration: dict,
    horizon: int,
) -> list[dict]:
    """Project each name forward `horizon` trading days from its live price.

    Every figure returned is an expectation *relative to the universe*. The
    market's own drift is deliberately not added: this model has no view on it,
    and inventing one would dress up a market forecast as a stock forecast.
    """
    bins = calibration["bins"]
    n_bins = calibration["n_bins"]
    out = []

    for ticker, rank in latest_rank.dropna().items():
        price = prices.get(str(ticker))
        if price is None or not np.isfinite(price) or price <= 0:
            continue
        b = int(np.clip(int(rank / 100 * n_bins), 0, n_bins - 1))
        stats = bins[b]
        mu, sd = stats["mean"], stats["std"]

        lo_r, hi_r = mu - Z80 * sd, mu + Z80 * sd
        out.append({
            "ticker": str(ticker),
            "price": round(float(price), 4),
            "bin": b + 1,
            "rank": round(float(rank), 1),
            "expected_return": round(mu, 6),
            "uncertainty": round(sd, 6),
            "low_return": round(lo_r, 6),
            "high_return": round(hi_r, 6),
            "target": round(float(price) * (1 + mu), 4),
            "target_low": round(float(price) * (1 + lo_r), 4),
            "target_high": round(float(price) * (1 + hi_r), 4),
            "hit_rate": stats["hit"],
            "sample": stats["n"],
            # How small the edge is next to the noise. Below ~0.1 the interval
            # swamps the estimate and the direction is barely meaningful.
            "signal_to_noise": round(abs(mu) / sd, 4) if sd > 0 else None,
            "horizon_days": horizon,
        })
    out.sort(key=lambda x: x["expected_return"], reverse=True)
    return out


def path(price: float, mean: float, std: float, horizon: int, steps: int = 12) -> dict:
    """Project a price forward as a widening cone, not a line.

    The central path interpolates the expected excess return linearly. The band
    widens with the square root of elapsed time, which is how uncertainty
    actually accumulates in a random walk — the honest shape is a cone that
    flares, never a trend line ruled into the future.

    A straight extrapolation of recent movement would be the single most
    misleading thing this project could draw: it implies the past direction
    persists, which is exactly what the measured hit rates say it does not.
    """
    out = {"days": [], "mid": [], "low": [], "high": []}
    for i in range(steps + 1):
        frac = i / steps
        t_days = frac * horizon
        mu = mean * frac
        sd = std * np.sqrt(frac) if frac > 0 else 0.0
        out["days"].append(round(t_days, 3))
        out["mid"].append(round(price * (1 + mu), 4))
        out["low"].append(round(price * (1 + mu - Z80 * sd), 4))
        out["high"].append(round(price * (1 + mu + Z80 * sd), 4))
    return out


def trend_path(closes: np.ndarray, price_now: float, horizon: int,
               lookback: int = 60, steps: int = 12) -> dict | None:
    """Extrapolate the recent price trend forward.

    A log-linear least-squares fit over the last ``lookback`` bars, continued
    past today at the same slope. ``r2`` reports how well that straight line
    described the window it was fitted to — note that a tight fit to the past
    implies nothing about the future, which is what ``trend_skill`` measures.
    """
    y = np.asarray(closes, dtype=float)
    y = y[np.isfinite(y) & (y > 0)]
    if len(y) < max(10, lookback // 3):
        return None
    y = np.log(y[-lookback:])
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    days, path_ = [], []
    for i in range(steps + 1):
        t = i / steps * horizon
        days.append(round(t, 3))
        path_.append(round(float(price_now * np.exp(slope * t)), 4))
    return {
        "days": days,
        "path": path_,
        "slope_pct_per_day": round(float(np.expm1(slope) * 100), 4),
        "r2": round(float(r2), 4),
        "lookback": int(min(lookback, len(y))),
        "end_price": path_[-1],
        "total_pct": round((path_[-1] / price_now - 1) * 100, 3),
    }


def trend_skill(close: pd.DataFrame, horizon: int, lookback: int = 60) -> dict:
    """Measure whether extrapolating the trend actually predicts anything.

    Computes, across the whole panel, the rank correlation between the trailing
    ``lookback``-day slope and the forward ``horizon``-day return, plus how
    often the trend's *direction* was right. This is the number that belongs
    next to a projected trend line, because "the line fits the past well" and
    "the line predicts the future" are unrelated claims.
    """
    logp = np.log(close.where(close > 0))
    slope = (logp - logp.shift(lookback)) / lookback          # avg daily log drift
    fwd = close.shift(-horizon) / close - 1

    both = slope.notna() & fwd.notna()
    s = slope.where(both).to_numpy().ravel()
    f = fwd.where(both).to_numpy().ravel()
    ok = np.isfinite(s) & np.isfinite(f)
    s, f = s[ok], f[ok]
    if len(s) < 500:
        return {"observations": int(len(s)), "ic": None, "direction_accuracy": None}

    ic = float(pd.Series(s).corr(pd.Series(f), method="spearman"))
    predicted_up = s > 0
    actually_up = f > 0
    return {
        "observations": int(len(s)),
        "lookback": lookback,
        "horizon": horizon,
        "ic": round(ic, 5),
        "direction_accuracy": round(float(np.mean(predicted_up == actually_up)), 4),
        "base_rate_up": round(float(np.mean(actually_up)), 4),
    }


def swing_points(closes: np.ndarray, window: int = 10) -> dict:
    """Find the real turning points in the price history.

    A bar is a swing low if it is the lowest close within ``window`` bars either
    side, and a swing high by the mirror rule. These are facts about what
    happened — unlike a projected turning point, which would be a guess dressed
    as one.
    """
    y = np.asarray(closes, dtype=float)
    lows, highs = [], []
    for i in range(window, len(y) - window):
        seg = y[i - window : i + window + 1]
        if not np.isfinite(y[i]):
            continue
        if y[i] == np.nanmin(seg):
            lows.append(i)
        elif y[i] == np.nanmax(seg):
            highs.append(i)
    return {"lows": lows, "highs": highs, "window": window}


def simulate_paths(price: float, mean: float, sigma_daily: float, horizon: int,
                   n_paths: int = 8, steps_per_day: int = 2, seed: int = 0) -> dict:
    """Draw plausible future price paths consistent with the calibrated forecast.

    Each path is a single random walk with the model's expected drift and the
    stock's own volatility. Every one of them has dips and turning points —
    which is precisely the lesson: they land in different places on every draw.
    The dips are real features of how prices move, and none of them is
    predictable. Re-seed and they move.

    These are samples from a distribution, never a forecast of a path.
    """
    rng = np.random.default_rng(seed)
    n_steps = max(2, int(horizon * steps_per_day))
    dt = horizon / n_steps
    mu_daily = mean / horizon
    sd_step = sigma_daily * np.sqrt(dt)

    days = [round(i * dt, 3) for i in range(n_steps + 1)]
    paths, turning = [], []
    for _ in range(n_paths):
        shocks = rng.normal(mu_daily * dt, sd_step, n_steps)
        walk = price * np.exp(np.cumsum(np.concatenate([[0.0], shocks])))
        paths.append([round(float(v), 4) for v in walk])
        # where this particular draw happened to bottom out
        turning.append(int(np.argmin(walk)))

    spread = (max(turning) - min(turning)) * dt if turning else 0.0
    return {
        "days": days,
        "paths": paths,
        "low_step_of_each_path": turning,
        "low_day_spread": round(float(spread), 2),
        "distinct_low_days": len(set(turning)),
        "n_paths": n_paths,
        "sigma_daily": round(float(sigma_daily), 6),
    }


def realised_vol(closes: np.ndarray, lookback: int = 60) -> float:
    """Daily volatility of log returns over the recent window."""
    y = np.asarray(closes, dtype=float)
    y = y[np.isfinite(y) & (y > 0)][-(lookback + 1):]
    if len(y) < 10:
        return 0.02
    r = np.diff(np.log(y))
    sd = float(np.std(r, ddof=1))
    return sd if np.isfinite(sd) and sd > 0 else 0.02


def dip_statistics(price: float, mean: float, sigma_daily: float, horizon: int,
                   n_sims: int = 2000, steps_per_day: int = 2, seed: int = 0) -> dict:
    """Summarise where the low lands across many simulated futures.

    Returns two markers rather than a cloud of paths:

    * **average** — the mean depth and mean timing of the low across all draws.
      A stable summary, but it describes the centre of a scatter, not a date to
      expect.
    * **most reliable** — the single step that held the low most often, with the
      share of draws that agreed. That share is the honest confidence, and for a
      random walk it stays low: the minimum clusters at the endpoints (the
      arcsine law) and never concentrates on one interior day.
    """
    rng = np.random.default_rng(seed)
    n_steps = max(2, int(horizon * steps_per_day))
    dt = horizon / n_steps
    mu_step = (mean / horizon) * dt
    sd_step = sigma_daily * np.sqrt(dt)

    shocks = rng.normal(mu_step, sd_step, (n_sims, n_steps))
    walks = price * np.exp(np.cumsum(np.concatenate(
        [np.zeros((n_sims, 1)), shocks], axis=1), axis=1))

    low_step = walks.argmin(axis=1)
    low_price = walks.min(axis=1)

    counts = np.bincount(low_step, minlength=n_steps + 1)
    modal_step = int(counts.argmax())
    modal_share = float(counts[modal_step] / n_sims)

    return {
        "average": {
            "day": round(float(low_step.mean() * dt), 2),
            "price": round(float(low_price.mean()), 4),
            "drawdown_pct": round(float((low_price.mean() / price - 1) * 100), 3),
        },
        "most_reliable": {
            "day": round(float(modal_step * dt), 2),
            "price": round(float(np.median(walks[low_step == modal_step].min(axis=1))), 4)
            if counts[modal_step] else round(float(price), 4),
            "probability": round(modal_share, 4),
        },
        "n_sims": n_sims,
        "horizon": horizon,
        "day_distribution": [
            {"day": round(i * dt, 2), "share": round(float(counts[i] / n_sims), 4)}
            for i in range(n_steps + 1)
        ],
    }


def central_path(closes: np.ndarray, price: float, mean: float, horizon: int,
                 n_draws: int = 201, block: int = 3, substeps: int = 8,
                 seed: int = 0) -> dict | None:
    """One representative future: the median outcome among many simulations.

    Simulates ``n_draws`` bootstrapped futures and returns the single one whose
    terminal price is the median. That keeps the realistic fluctuation of an
    actual path — a random pick would too — while being the central scenario
    rather than an arbitrary draw. The seed is fixed per stock so the line is
    stable between page loads instead of flickering to a new shape each time.
    """
    sims = bootstrap_paths(closes, price, mean, horizon,
                           n_paths=n_draws, block=block, substeps=substeps, seed=seed)
    if sims is None:
        return None
    ends = np.array([pa[-1] for pa in sims["paths"]])
    order = np.argsort(ends)
    pick = int(order[len(order) // 2])
    chosen = np.asarray(sims["paths"][pick], dtype=float)
    days = sims["days"]

    lo_i, hi_i = int(chosen.argmin()), int(chosen.argmax())

    # Where the extremes landed across every draw, so the two dots on the shown
    # path can be read against the range rather than as fixed levels.
    all_lows = np.array([min(pa) for pa in sims["paths"]])
    all_highs = np.array([max(pa) for pa in sims["paths"]])
    lo_days = np.array([float(np.argmin(pa)) for pa in sims["paths"]]) / max(substeps, 1)
    hi_days = np.array([float(np.argmax(pa)) for pa in sims["paths"]]) / max(substeps, 1)

    return {
        "days": days,
        "path": [round(float(v), 4) for v in chosen],
        "n_draws": n_draws,
        "block": sims["block"],
        "source_days": sims["source_days"],
        "end": round(float(chosen[-1]), 4),
        "low": {"price": round(float(chosen[lo_i]), 4), "day": days[lo_i], "index": lo_i},
        "high": {"price": round(float(chosen[hi_i]), 4), "day": days[hi_i], "index": hi_i},
        "across_draws": {
            "low_p10": round(float(np.percentile(all_lows, 10)), 2),
            "low_p90": round(float(np.percentile(all_lows, 90)), 2),
            "high_p10": round(float(np.percentile(all_highs, 10)), 2),
            "high_p90": round(float(np.percentile(all_highs, 90)), 2),
            "low_day_p10": round(float(np.percentile(lo_days, 10)), 2),
            "low_day_p90": round(float(np.percentile(lo_days, 90)), 2),
            "high_day_p10": round(float(np.percentile(hi_days, 10)), 2),
            "high_day_p90": round(float(np.percentile(hi_days, 90)), 2),
        },
        "percentile": 50,
    }


def bootstrap_paths(closes: np.ndarray, price: float, mean: float, horizon: int,
                    n_paths: int = 5, block: int = 3, substeps: int = 8,
                    seed: int = 0) -> dict | None:
    """Simulate futures that look like this stock's own past.

    Two steps, chosen so the result is not merely wiggly but statistically
    faithful:

    1. **Block bootstrap.** Daily returns are resampled in contiguous blocks
       drawn from the stock's real history, so volatility clustering and fat
       tails survive — a calm week stays calm, a violent one stays violent.
       Gaussian noise would produce a path that wiggles but is too well-behaved
       to pass for a real chart.
    2. **Brownian bridge.** Each day is filled in with sub-steps pinned to the
       bootstrapped daily closes, giving intraday texture at the right scale
       instead of straight lines between points.

    The paths are then recentred so their average drift matches the model's
    calibrated expectation. Each is one equally likely scenario; the ensemble is
    the forecast, never any single line.
    """
    y = np.asarray(closes, dtype=float)
    y = y[np.isfinite(y) & (y > 0)]
    if len(y) < 120:
        return None
    hist = np.diff(np.log(y))
    if len(hist) < 60:
        return None

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(horizon / block))
    paths = []

    for _ in range(n_paths):
        starts = rng.integers(0, len(hist) - block, size=n_blocks)
        daily = np.concatenate([hist[s0 : s0 + block] for s0 in starts])[:horizon]
        # Shift by the *historical* mean, not this draw's own mean. Subtracting
        # the draw's mean would force every path to the same total return and
        # collapse the terminal spread — every simulated future ending on the
        # same price, which is the opposite of the point.
        daily = daily - hist.mean() + (mean / horizon)

        pts = [price]
        for r in daily:
            step_end = pts[-1] * np.exp(r)
            # Brownian bridge between the two daily closes.
            sd = abs(r) / np.sqrt(max(substeps, 1))
            prev = pts[-1]
            for k in range(1, substeps + 1):
                frac = k / substeps
                target = prev * np.exp(r * frac)
                jitter = 0.0 if k == substeps else rng.normal(0, sd) * np.sqrt(frac * (1 - frac))
                pts.append(float(target * np.exp(jitter)))
            pts[-1] = float(step_end)
        paths.append([round(v, 4) for v in pts])

    n_pts = len(paths[0])
    days = [round(i / substeps, 3) for i in range(n_pts)]
    return {
        "days": days,
        "paths": paths,
        "n_paths": n_paths,
        "block": block,
        "substeps": substeps,
        "source_days": int(len(hist)),
    }


def intraday_path(price: float, mean_total: float, bar_returns: np.ndarray,
                  n_steps: int = 78, n_draws: int = 201, block: int = 6,
                  seed: int = 0) -> dict | None:
    """Project a session using the stock's own intraday bar returns.

    Same block-bootstrap idea as ``central_path``, but the sampling unit is an
    intraday bar rather than a day, so the result has the texture of a real
    session at the same resolution as the chart beside it.
    """
    r = np.asarray(bar_returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 30:
        return None

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_steps / block))
    drift = mean_total / n_steps

    paths = np.empty((n_draws, n_steps + 1))
    for k in range(n_draws):
        starts = rng.integers(0, len(r) - block, size=n_blocks)
        steps = np.concatenate([r[s0 : s0 + block] for s0 in starts])[:n_steps]
        steps = steps - r.mean() + drift
        paths[k] = price * np.exp(np.concatenate([[0.0], np.cumsum(steps)]))

    ends = paths[:, -1]
    chosen = paths[int(np.argsort(ends)[n_draws // 2])]
    lo_i, hi_i = int(chosen.argmin()), int(chosen.argmax())
    days = [round(i / n_steps, 5) for i in range(n_steps + 1)]

    return {
        "days": days,
        "path": [round(float(v), 4) for v in chosen],
        "n_draws": n_draws,
        "block": block,
        "source_days": int(len(r)),
        "end": round(float(chosen[-1]), 4),
        "low": {"price": round(float(chosen[lo_i]), 4), "day": days[lo_i], "index": lo_i},
        "high": {"price": round(float(chosen[hi_i]), 4), "day": days[hi_i], "index": hi_i},
        "across_draws": {
            "low_p10": round(float(np.percentile(paths.min(axis=1), 10)), 2),
            "low_p90": round(float(np.percentile(paths.min(axis=1), 90)), 2),
            "high_p10": round(float(np.percentile(paths.max(axis=1), 10)), 2),
            "high_p90": round(float(np.percentile(paths.max(axis=1), 90)), 2),
            "low_day_p10": 0.0, "low_day_p90": 1.0,
            "high_day_p10": 0.0, "high_day_p90": 1.0,
        },
        "percentile": 50,
        "intraday": True,
        "bar_sigma": round(float(r.std(ddof=1)), 6),
    }
