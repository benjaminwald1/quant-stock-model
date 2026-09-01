"""Models.

All three share one interface so the pipeline can treat them interchangeably:

    model.fit(X, y, X_val, y_val) -> self
    model.predict(X) -> np.ndarray
    model.importance() -> pd.Series | None

``X`` is a long frame indexed by ``(date, ticker)``. The tabular models ignore
the index; the sequence model uses it to reconstruct per-ticker histories.

A note on model choice: on cross-sectionally ranked features with a ranked
target, gradient boosting is the workhorse — it handles the nonlinear,
interaction-heavy structure of these features without needing much data
per parameter. Ridge is kept as a baseline because a linear model on ranked
features is a genuinely strong quant benchmark, and if the fancy model cannot
beat it, the fancy model is not earning its complexity.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ModelConfig

log = logging.getLogger(__name__)


class BaseModel:
    name = "base"

    def fit(self, X, y, X_val=None, y_val=None):  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def importance(self) -> pd.Series | None:
        return None


class RidgeModel(BaseModel):
    """L2 linear regression on standardised features."""

    name = "ridge"

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.pipe = Pipeline(
            [("scale", StandardScaler()), ("model", Ridge(alpha=alpha, random_state=None))]
        )
        self.columns: list[str] = []

    def fit(self, X, y, X_val=None, y_val=None):
        self.columns = list(X.columns)
        self.pipe.fit(X.to_numpy(dtype=np.float64), y.to_numpy(dtype=np.float64))
        return self

    def predict(self, X) -> np.ndarray:
        return self.pipe.predict(X[self.columns].to_numpy(dtype=np.float64))

    def importance(self) -> pd.Series | None:
        coef = self.pipe.named_steps["model"].coef_
        return pd.Series(np.abs(coef), index=self.columns).sort_values(ascending=False)


class LGBMModel(BaseModel):
    """Gradient-boosted trees, early-stopped on a purged validation tail."""

    name = "lgbm"

    def __init__(self, params: dict | None = None, seed: int = 7):
        import lightgbm as lgb

        self.lgb = lgb
        self.params = dict(params or {})
        self.params.setdefault("random_state", seed)
        self.model = None
        self.columns: list[str] = []

    def fit(self, X, y, X_val=None, y_val=None):
        self.columns = list(X.columns)
        self.model = self.lgb.LGBMRegressor(**self.params)
        kwargs = {}
        if X_val is not None and len(X_val) > 0:
            # LightGBM >= 4.7 renamed eval_set to eval_X / eval_y.
            import inspect

            if "eval_X" in inspect.signature(self.model.fit).parameters:
                kwargs["eval_X"], kwargs["eval_y"] = X_val[self.columns], y_val
            else:
                kwargs["eval_set"] = [(X_val[self.columns], y_val)]
            kwargs["callbacks"] = [
                self.lgb.early_stopping(50, verbose=False),
                self.lgb.log_evaluation(0),
            ]
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X[self.columns])

    def importance(self) -> pd.Series | None:
        if self.model is None:
            return None
        imp = self.model.booster_.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self.columns).sort_values(ascending=False)


class GRUModel(BaseModel):
    """Recurrent net over a rolling window of each ticker's feature history.

    The tabular models see one row — today's features. This one sees the last
    ``seq_len`` days, so it can pick up path dependence (how a signal built up,
    not just where it ended). Requires the ``[nn]`` extra.
    """

    name = "gru"

    def __init__(self, params: dict | None = None, seed: int = 7):
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The GRU model needs PyTorch. Install it with: uv pip install -e '.[nn]'"
            ) from exc
        self.torch, self.nn = torch, nn

        p = {
            "seq_len": 20, "hidden": 48, "layers": 1, "dropout": 0.2,
            "lr": 1e-3, "epochs": 12, "batch_size": 512, "patience": 3,
        }
        p.update(params or {})

        # LightGBM links Homebrew's libomp; PyTorch ships its own copy. With both
        # loaded in one process on macOS the two OpenMP thread pools deadlock and
        # training hangs forever rather than failing. Pinning torch to one thread
        # sidesteps it, at the cost of a large slowdown on big folds.
        #
        # `torch_threads` overrides this: 0 leaves torch's threading alone (safe
        # when LightGBM is not in the run), N sets an explicit count.
        threads = p.pop("torch_threads", None)
        if threads:
            torch.set_num_threads(int(threads))
        elif threads is None and "lightgbm" in sys.modules:
            torch.set_num_threads(1)
            log.warning(
                "GRU pinned to 1 thread because LightGBM is loaded in this process "
                "(duplicate OpenMP runtimes deadlock on macOS). Expect slow fits on "
                "large folds. For full speed, run the GRU on its own: "
                "`qsm run --models gru`."
            )

        self.p = p
        self.seed = seed
        self.net = None
        self.columns: list[str] = []
        self.mean_ = None
        self.std_ = None

    # -- sequence assembly ------------------------------------------------
    def _sequences(self, X: pd.DataFrame, y: pd.Series | None):
        """Build (n, seq_len, n_features) windows, one per (ticker, date).

        Windows are cut per ticker over its own date-ordered history, so they
        never straddle two names, and never reach past the target date.
        """
        seq_len = self.p["seq_len"]
        Xs = X[self.columns].astype(np.float32)
        Xs = ((Xs - self.mean_) / self.std_).fillna(0.0).clip(-5, 5)
        arr = Xs.to_numpy(dtype=np.float32)
        tickers = X.index.get_level_values("ticker").to_numpy()
        order = np.argsort(tickers, kind="stable")

        windows, targets, positions = [], [], []
        for start, stop in _group_bounds(tickers[order]):
            idx = order[start:stop]
            block = arr[idx]
            if len(block) < seq_len:
                continue
            strided = np.lib.stride_tricks.sliding_window_view(block, seq_len, axis=0)
            strided = np.ascontiguousarray(strided.transpose(0, 2, 1))
            windows.append(strided)
            pos = idx[seq_len - 1 :]
            positions.append(pos)
            if y is not None:
                targets.append(y.to_numpy(dtype=np.float32)[pos])
        if not windows:
            return None, None, None
        W = np.concatenate(windows)
        P = np.concatenate(positions)
        T = np.concatenate(targets) if y is not None else None
        return W, T, P

    def _build(self, n_features: int):
        torch, nn = self.torch, self.nn
        torch.manual_seed(self.seed)

        class Net(nn.Module):
            def __init__(self, n_in, hidden, layers, dropout):
                super().__init__()
                self.gru = nn.GRU(
                    n_in, hidden, num_layers=layers, batch_first=True,
                    dropout=dropout if layers > 1 else 0.0,
                )
                self.head = nn.Sequential(
                    nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, 1)
                )

            def forward(self, x):
                out, _ = self.gru(x)
                return self.head(out[:, -1]).squeeze(-1)

        return Net(n_features, self.p["hidden"], self.p["layers"], self.p["dropout"])

    def fit(self, X, y, X_val=None, y_val=None):
        torch = self.torch
        self.columns = list(X.columns)
        self.mean_ = X[self.columns].mean()
        self.std_ = X[self.columns].std().replace(0, 1.0)

        Wtr, Ttr, _ = self._sequences(X, y)
        if Wtr is None:
            raise ValueError("Not enough history per ticker to build GRU sequences.")
        self.net = self._build(len(self.columns))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.p["lr"], weight_decay=1e-4)
        loss_fn = self.nn.MSELoss()

        ds = torch.utils.data.TensorDataset(torch.from_numpy(Wtr), torch.from_numpy(Ttr))
        dl = torch.utils.data.DataLoader(ds, batch_size=self.p["batch_size"], shuffle=True)

        val = None
        if X_val is not None and len(X_val) > 0:
            Wv, Tv, _ = self._sequences(X_val, y_val)
            if Wv is not None:
                val = (torch.from_numpy(Wv), torch.from_numpy(Tv))

        best, best_state, bad = np.inf, None, 0
        for epoch in range(self.p["epochs"]):
            self.net.train()
            for xb, yb in dl:
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            if val is None:
                continue
            self.net.eval()
            with torch.no_grad():
                vloss = float(loss_fn(self.net(val[0]), val[1]))
            if vloss < best - 1e-6:
                best, bad = vloss, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.p["patience"]:
                    log.info("GRU early stop at epoch %d (val %.6f)", epoch, best)
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, X) -> np.ndarray:
        torch = self.torch
        W, _, P = self._sequences(X, None)
        # Rows without a full lookback get 0 — a neutral score, which the
        # cross-sectional standardisation downstream treats as "no opinion".
        out = np.zeros(len(X), dtype=np.float64)
        if W is None:
            return out
        self.net.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(W), 4096):
                preds.append(self.net(torch.from_numpy(W[i : i + 4096])).numpy())
        out[P] = np.concatenate(preds)
        return out


def _group_bounds(sorted_keys: np.ndarray):
    """Yield (start, stop) slices for each run of equal values."""
    if len(sorted_keys) == 0:
        return
    change = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
    bounds = np.concatenate([[0], change, [len(sorted_keys)]])
    for a, b in zip(bounds[:-1], bounds[1:]):
        yield int(a), int(b)


def build_model(name: str, cfg: ModelConfig) -> BaseModel:
    if name == "ridge":
        return RidgeModel(alpha=cfg.ridge_alpha)
    if name == "lgbm":
        return LGBMModel(params=cfg.lgbm_params, seed=cfg.seed)
    if name == "gru":
        return GRUModel(params=cfg.gru_params, seed=cfg.seed)
    raise ValueError(f"Unknown model '{name}'. Choose from: ridge, lgbm, gru.")
