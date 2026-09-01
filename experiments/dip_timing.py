"""Can the timing of a stock's short-term low be predicted?

An honest test of the question "when will it dip". For every (date, ticker) we
label which of the next H trading days actually held the lowest close, then try
to predict that label from the full feature set using the same purged
walk-forward protocol the return model uses.

Three yardsticks, because accuracy alone is easy to misread:

* the base rate — always guessing the single most common day;
* multi-class accuracy out of sample;
* the only one that matters economically — does buying on the predicted day
  beat buying on a fixed day?
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from qsm.config import Config                      # noqa: E402
from qsm.features import compute_features          # noqa: E402
from qsm.data import apply_universe_filters        # noqa: E402
from qsm.live import fetch_live                    # noqa: E402
from qsm.splits import purged_walk_forward         # noqa: E402
from qsm.universe import resolve                   # noqa: E402

H = 5


def build():
    cfg = Config()
    cfg.data.min_dollar_volume = 5_000_000
    panel = fetch_live(resolve("sp100"), start="2005-01-01")
    panel = apply_universe_filters(panel, cfg.data)
    feats = compute_features(panel, cfg.features)

    close = panel.pivot(index="date", columns="ticker", values="close").sort_index()
    # Forward window t+1 .. t+H, then which offset held the low.
    fwd = np.stack([close.shift(-k).to_numpy() for k in range(1, H + 1)], axis=-1)
    with np.errstate(invalid="ignore"):
        argmin = np.argmin(fwd, axis=-1).astype(float)
    argmin[np.isnan(fwd).any(axis=-1)] = np.nan
    label = pd.DataFrame(argmin, index=close.index, columns=close.columns)

    # Return from buying at each offset and holding to the end of the window.
    end = close.shift(-H)
    buy_at = {k: (end / close.shift(-k) - 1) for k in range(1, H + 1)}

    ds = feats.join(label.stack(future_stack=True).rename("low_day"), how="inner")
    ds = ds[ds["low_day"].notna()]
    ds = ds.join(panel.set_index(["date", "ticker"])["tradable"], how="left")
    ds = ds[ds["tradable"].fillna(False)].drop(columns=["tradable"])
    return cfg, ds, buy_at


def main():
    import lightgbm as lgb

    cfg, ds, buy_at = build()
    fcols = [c for c in ds.columns if c != "low_day"]
    dates = pd.DatetimeIndex(sorted(ds.index.get_level_values("date").unique()))
    print(f"rows {len(ds):,} | features {len(fcols)} | dates {len(dates):,}")

    dist = ds["low_day"].value_counts(normalize=True).sort_index()
    print("\nWhich day actually held the low (unconditional):")
    for k, v in dist.items():
        print(f"  day {int(k) + 1}: {v * 100:5.2f}%")
    base = float(dist.max())
    print(f"base rate (always guess day {int(dist.idxmax()) + 1}): {base * 100:.2f}%")

    cfg.splits.n_splits = 4
    cfg.splits.min_train_size = 750
    cfg.splits.test_size = 252
    splits = purged_walk_forward(dates, cfg.splits, H)

    accs, preds_all, truth_all, idx_all = [], [], [], []
    for sp in splits:
        idx = ds.index.get_level_values("date")
        tr = ds[idx.isin(sp.train_dates)]
        te = ds[idx.isin(sp.test_dates)]
        if te.empty:
            continue
        m = lgb.LGBMClassifier(objective="multiclass", num_class=H, n_estimators=250,
                               learning_rate=0.05, num_leaves=31, min_child_samples=200,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                               verbose=-1, random_state=7)
        m.fit(tr[fcols], tr["low_day"].astype(int))
        p = m.predict(te[fcols])
        acc = float((p == te["low_day"].astype(int).to_numpy()).mean())
        accs.append(acc)
        preds_all.append(p)
        truth_all.append(te["low_day"].astype(int).to_numpy())
        idx_all.append(te.index)
        print(f"  {sp.label[:44]}  acc {acc * 100:5.2f}%  (base {base * 100:.2f}%)")

    pred = np.concatenate(preds_all)
    truth = np.concatenate(truth_all)
    index = idx_all[0].append(idx_all[1:]) if len(idx_all) > 1 else idx_all[0]
    acc = float((pred == truth).mean())
    print(f"\nOUT-OF-SAMPLE ACCURACY  {acc * 100:.2f}%   vs base rate {base * 100:.2f}%")
    print(f"lift over always-guessing: {(acc - base) * 100:+.2f} pp")

    # Does the model at least concentrate its guesses usefully?
    print("\nWhat the model guessed vs what happened:")
    print(pd.crosstab(pd.Series(pred + 1, name="predicted"),
                      pd.Series(truth + 1, name="actual"), normalize="index").round(3).to_string())

    # The economic test: buy on the predicted day, versus fixed rules.
    print("\nMean return from buying at a chosen day and holding to day 5:")
    stacked = {k: v.stack(future_stack=True) for k, v in buy_at.items()}
    rows = {}
    for k in range(1, H + 1):
        rows[f"always day {k}"] = float(stacked[k].reindex(index).mean())
    model_ret = np.array([stacked[int(p) + 1].get(ix, np.nan) for p, ix in zip(pred, index)])
    rows["model's predicted low day"] = float(np.nanmean(model_ret))
    best_fixed = max(rows[f"always day {k}"] for k in range(1, H + 1))
    for k, v in rows.items():
        print(f"  {k:28s} {v * 100:+.4f}%")
    print(f"\nmodel minus best fixed rule: "
          f"{(rows['model\'s predicted low day'] - best_fixed) * 100:+.4f}%")


if __name__ == "__main__":
    main()
