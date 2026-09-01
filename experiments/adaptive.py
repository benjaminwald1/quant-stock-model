"""Does learning from realised mistakes actually improve the forecast?

Compares three ways of combining the models, on identical out-of-sample rows:

  static    equal-weight average (what the app does today)
  adaptive  weights from each model's recently realised IC
  best      always the single model that scored best over the whole sample
            (not implementable live — an upper bound, shown for scale)

Adapting to recent performance is exactly the kind of change that feels
obviously right and often chases noise, so it should not ship without this.
"""

from __future__ import annotations

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, "src")

from qsm.backtest import information_coefficient, run_backtest   # noqa: E402
from qsm.config import BacktestConfig                            # noqa: E402
from qsm.online import adaptive_weights, blend, score_predictions, weight_summary  # noqa: E402

import json                                                       # noqa: E402


def main(run: str, halflife: int = 60):
    d = Path("runs") / run
    panel = pd.read_parquet(d / "ticker_panel.parquet")
    cfg_raw = json.loads((d / "config.json").read_text())
    h = cfg_raw["labels"]["horizon"]
    bt_cfg = BacktestConfig(**cfg_raw["backtest"])

    models = [c for c in panel.columns
              if c not in ("close", "ret_next_1d", "fwd_ret", "ensemble")]
    preds = panel[models]
    fwd = panel["fwd_ret"].unstack()
    ret_next = panel["ret_next_1d"].unstack()

    ic = score_predictions(preds, fwd)
    print(f"run {run} | models {models} | horizon {h}d")
    print("whole-sample IC:", {m: round(float(ic[m].mean()), 5) for m in models})

    w = adaptive_weights(ic, horizon=h, halflife=halflife)
    print("\nweight behaviour:", json.dumps(weight_summary(w), indent=1))

    candidates = {
        "static (equal weight)": panel["ensemble"] if "ensemble" in panel.columns
        else blend(preds, pd.DataFrame(1 / len(models), index=ic.index, columns=models)),
        "adaptive (learns)": blend(preds, w),
    }
    best_model = max(models, key=lambda m: ic[m].mean())
    candidates[f"best single ({best_model}, hindsight)"] = preds[best_model]

    print(f"\n{'combiner':34s} {'IC':>8s} {'IC t':>7s} {'net Sharpe':>11s} {'gross':>7s}")
    for name, series in candidates.items():
        sig = series.unstack().reindex(index=ret_next.index, columns=ret_next.columns)
        icr = information_coefficient(sig, fwd, min_names=bt_cfg.min_names)
        bt = run_backtest(sig, ret_next, bt_cfg, holding_days=h,
                          execution_lag=bt_cfg.execution_lag)
        m = bt["metrics"]
        print(f"{name:34s} {icr['ic_mean']:8.4f} {icr['ic_t_stat']:7.2f} "
              f"{m['sharpe']:11.3f} {m['sharpe_before_costs']:7.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "20260829-175555-sp500",
         int(sys.argv[2]) if len(sys.argv) > 2 else 60)
