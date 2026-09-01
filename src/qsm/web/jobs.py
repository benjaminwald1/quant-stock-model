"""Background job execution for the web UI.

Runs are long (seconds to many minutes), so the HTTP layer never executes one
inline. Jobs go to a single-worker pool — one at a time, deliberately: two
concurrent runs would fight over cores and memory and make both slower than
running them in sequence.

Log lines are captured off the `qsm` logger and buffered per job, which is what
the UI polls to show live progress.
"""

from __future__ import annotations

import logging
import threading
import traceback
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import KAGGLE_DATASETS, RAW_DIR, RUNS_DIR, Config

MAX_LOG_LINES = 2000


@dataclass
class Job:
    id: str
    kind: str
    label: str
    status: str = "queued"          # queued | running | done | error
    created: datetime = field(default_factory=datetime.now)
    finished: datetime | None = None
    params: dict = field(default_factory=dict)
    log: deque = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))
    run_dir: str | None = None
    error: str | None = None
    result: dict | None = None

    def to_dict(self, include_log: bool = False, log_from: int = 0) -> dict:
        out = {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "created": self.created.isoformat(timespec="seconds"),
            "finished": self.finished.isoformat(timespec="seconds") if self.finished else None,
            "elapsed": round(
                ((self.finished or datetime.now()) - self.created).total_seconds(), 1
            ),
            "params": self.params,
            "run_dir": self.run_dir,
            "error": self.error,
            "result": self.result,
            "log_len": len(self.log),
        }
        if include_log:
            out["log"] = list(self.log)[log_from:]
        return out


class _JobLogHandler(logging.Handler):
    """Route qsm log records into whichever job owns the emitting thread."""

    def __init__(self, registry: "JobRegistry"):
        super().__init__()
        self.registry = registry

    def emit(self, record: logging.LogRecord) -> None:
        job = self.registry.job_for_thread(threading.get_ident())
        if job is None:
            return
        try:
            stamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            job.log.append(f"{stamp}  {record.getMessage()}")
        except Exception:  # pragma: no cover - logging must never raise
            pass


class JobRegistry:
    """Tracks jobs and maps worker threads back to the job they are running."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._threads: dict[int, str] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qsm-job")

        handler = _JobLogHandler(self)
        handler.setLevel(logging.INFO)
        qsm_log = logging.getLogger("qsm")
        qsm_log.setLevel(logging.INFO)
        qsm_log.addHandler(handler)
        qsm_log.propagate = False

    # -- bookkeeping -------------------------------------------------------
    def job_for_thread(self, ident: int) -> Job | None:
        with self._lock:
            job_id = self._threads.get(ident)
            return self._jobs.get(job_id) if job_id else None

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def _bind(self, job: Job) -> None:
        with self._lock:
            self._threads[threading.get_ident()] = job.id

    def _unbind(self) -> None:
        with self._lock:
            self._threads.pop(threading.get_ident(), None)

    # -- submission --------------------------------------------------------
    def submit(self, kind: str, label: str, params: dict, fn) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, label=label, params=params)
        self._jobs[job.id] = job

        def wrapper():
            self._bind(job)
            job.status = "running"
            job.log.append(f"{datetime.now():%H:%M:%S}  starting {label}")
            try:
                fn(job)
                job.status = "done"
                job.log.append(f"{datetime.now():%H:%M:%S}  finished")
            except Exception as exc:
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.log.append(f"ERROR  {job.error}")
                for line in traceback.format_exc().splitlines()[-12:]:
                    job.log.append(f"       {line}")
            finally:
                job.finished = datetime.now()
                self._unbind()

        self._pool.submit(wrapper)
        return job


# --------------------------------------------------------------------------
# Job bodies
# --------------------------------------------------------------------------
def config_from_params(p: dict) -> Config:
    """Translate the web form into a Config, validating as we go."""
    cfg = Config()
    cfg.data.source = "live" if p.get("mode") == "live" else "kaggle"
    cfg.data.provider = p.get("provider", "yahoo")
    cfg.data.universe = p.get("universe", "sp100")
    cfg.data.tickers = tuple(
        t for t in str(p.get("tickers") or "").replace(",", " ").split() if t
    )
    if p.get("refresh"):
        cfg.data.max_age_hours = 0.0
    cfg.data.dataset = p.get("dataset", "huge")
    cfg.data.max_tickers = int(p.get("max_tickers", 500))
    if p.get("start"):
        cfg.data.start = p["start"]
    cfg.data.end = p.get("end") or None

    cfg.labels.horizon = int(p.get("horizon", 5))
    cfg.splits.n_splits = int(p.get("folds", 6))
    cfg.splits.embargo = int(p.get("embargo", 5))

    models = tuple(p.get("models") or ("ridge", "lgbm"))
    if not models:
        raise ValueError("Pick at least one model.")
    for m in models:
        if m not in ("ridge", "lgbm", "gru"):
            raise ValueError(f"Unknown model '{m}'.")
    cfg.model.models = models

    cfg.backtest.cost_bps = float(p.get("cost_bps", 10.0))
    cfg.backtest.quantile = float(p.get("quantile", 0.1))
    cfg.backtest.execution_lag = int(p.get("execution_lag", 1))
    if not 0.01 <= cfg.backtest.quantile <= 0.5:
        raise ValueError("Quantile must be between 0.01 and 0.5.")
    return cfg


def make_run_job(params: dict):
    from ..pipeline import run, run_synthetic

    def body(job: Job) -> None:
        cfg = config_from_params(params)
        tag = params.get("tag") or "web"
        if params.get("mode") == "live":
            run_dir, summary, _ = run(cfg, tag=tag)
        elif params.get("mode") == "synthetic":
            run_dir, summary, _ = run_synthetic(
                cfg,
                signal_strength=float(params.get("signal_strength", 0.05)),
                n_tickers=int(params.get("max_tickers", 120)),
                n_days=int(params.get("n_days", 2600)),
                seed=int(params.get("seed", 0)),
                tag=tag,
            )
        else:
            root = Path(params["data_root"]) if params.get("data_root") else (
                RAW_DIR / KAGGLE_DATASETS.get(cfg.data.dataset, cfg.data.dataset).split("/")[-1]
            )
            if not Path(root).exists():
                raise FileNotFoundError(
                    f"No price data at {root}. Download it first (Data tab), "
                    f"or switch the run to synthetic data."
                )
            run_dir, summary, _ = run(cfg, data_root=root, tag=tag)
        job.run_dir = run_dir.name
        job.result = {"summary_rows": summary.shape[0]}

    return body


def make_null_test_job(params: dict):
    from ..pipeline import run_synthetic

    def body(job: Job) -> None:
        seeds = params.get("seeds") or [11, 12, 13]
        worst = 0.0
        rows = []
        for seed in seeds:
            cfg = Config()
            cfg.splits.n_splits = 3
            cfg.model.models = tuple(params.get("models") or ("ridge", "lgbm"))
            _, summary, _ = run_synthetic(
                cfg, signal_strength=0.0, n_tickers=80, n_days=1800,
                seed=int(seed), tag=f"null-{seed}",
            )
            for model in summary.index:
                if model == "buy&hold universe":
                    continue
                t = summary.loc[model, "ic_t_stat"]
                rows.append({
                    "seed": int(seed), "model": model,
                    "ic_mean": _num(summary.loc[model, "ic_mean"]),
                    "ic_t_stat": _num(t),
                    "sharpe_before_costs": _num(summary.loc[model, "sharpe_before_costs"]),
                })
                if t == t:  # not NaN
                    worst = max(worst, abs(float(t)))
        job.result = {"rows": rows, "worst_abs_t": round(worst, 3), "passed": worst < 4}

    return body


def make_fetch_job(params: dict):
    from ..data import download_kaggle

    def body(job: Job) -> None:
        dest = download_kaggle(params.get("dataset", "huge"), force=bool(params.get("force")))
        job.result = {"path": str(dest)}

    return body


def _num(v: Any) -> float | None:
    try:
        f = float(v)
        return None if f != f else round(f, 6)
    except (TypeError, ValueError):
        return None


def list_run_dirs() -> list[dict]:
    """Every finished run on disk, newest first."""
    if not RUNS_DIR.exists():
        return []
    out = []
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not d.is_dir() or not (d / "summary.csv").exists():
            continue
        out.append({
            "name": d.name,
            "modified": datetime.fromtimestamp(d.stat().st_mtime).isoformat(timespec="seconds"),
            "has_report": (d / "report.png").exists(),
        })
    return out
