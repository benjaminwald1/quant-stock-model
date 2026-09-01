"""Which forward curve is actually the most accurate?

Three candidate pictures of a stock's future, scored on the same out-of-sample
rows against what the price really did H days later:

  flat   — no change; the price today is the forecast (the martingale baseline)
  trend  — the last 60 days' slope carried forward
  model  — today's price adjusted by the model's calibrated expected return

Scored by mean absolute error, root mean squared error, and direction accuracy.
Accuracy here means "closest to the realised price", which is the only sense in
which one line can be called the most accurate picture of the future.
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from qsm.forecast import calibrate                 # noqa: E402
from qsm.web.stocks import load_view               # noqa: E402
from pathlib import Path                           # noqa: E402

H = 5
LOOKBACK = 60


def main(run: str):
    d = Path("runs") / run
    view = load_view(d, "ensemble")
    panel = pd.read_parquet(d / "ticker_panel.parquet")

    close = view.close.sort_index()
    fwd_ret = panel["fwd_ret"].unstack().reindex_like(close)
    actual = close * (1 + fwd_ret)

    # 1. flat
    flat = close.copy()

    # 2. trend: average daily log drift over LOOKBACK, carried H days
    logp = np.log(close.where(close > 0))
    slope = (logp - logp.shift(LOOKBACK)) / LOOKBACK
    trend = close * np.exp(slope * H)

    # 3. model: calibrated expected excess return for the name's signal decile
    cal = calibrate(view.signal, panel["fwd_ret"].unstack())
    means = np.array([b["mean"] for b in cal["bins"]])
    ranks = view.signal.rank(axis=1, pct=True, na_option="keep")
    bin_idx = np.clip((ranks * cal["n_bins"]).astype(float) - 1e-9, 0, cal["n_bins"] - 1)
    exp_ret = pd.DataFrame(
        np.where(np.isnan(bin_idx), np.nan, means[np.nan_to_num(bin_idx).astype(int)]),
        index=ranks.index, columns=ranks.columns)
    # NOTE: an earlier version added `fwd_ret.mean(axis=1)` here to supply the
    # market's drift. That is the *realised* cross-sectional mean from t to t+H
    # — future information, unknowable at t. It leaked the market's actual move
    # into the model's forecast and inflated direction accuracy to 66%. The
    # model forecasts excess return only, so the honest counterpart to "flat"
    # is to assume no market drift, exactly as the other two lines do.
    model = close * (1 + exp_ret)

    both = actual.notna() & close.notna() & trend.notna() & model.notna()
    n = int(both.to_numpy().sum())
    print(f"scored on {n:,} out-of-sample (date, ticker) pairs, horizon {H}d\n")

    def score(name, pred):
        err = ((pred - actual) / close).where(both).to_numpy().ravel()
        err = err[np.isfinite(err)]
        realised = ((actual - close) / close).where(both).to_numpy().ravel()
        predicted = ((pred - close) / close).where(both).to_numpy().ravel()
        ok = np.isfinite(realised) & np.isfinite(predicted)
        dir_acc = float(np.mean((predicted[ok] > 0) == (realised[ok] > 0)))
        return {
            "MAE %": np.mean(np.abs(err)) * 100,
            "RMSE %": np.sqrt(np.mean(err ** 2)) * 100,
            "direction %": dir_acc * 100,
        }

    rows = {"flat (no change)": score("flat", flat),
            "trend (60d slope)": score("trend", trend),
            "model (calibrated)": score("model", model)}

    print(f"{'forecast':22s} {'MAE %':>8s} {'RMSE %':>8s} {'direction %':>12s}")
    for k, v in rows.items():
        print(f"{k:22s} {v['MAE %']:8.3f} {v['RMSE %']:8.3f} {v['direction %']:12.2f}")

    best_mae = min(rows, key=lambda k: rows[k]["MAE %"])
    print(f"\nlowest error: {best_mae}")
    base = rows["flat (no change)"]["MAE %"]
    for k, v in rows.items():
        if k != "flat (no change)":
            print(f"  {k}: {(v['MAE %'] / base - 1) * 100:+.2f}% MAE vs flat")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "20260829-092258-sp100-live")
