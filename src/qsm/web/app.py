"""FastAPI app serving the local research UI.

Bound to localhost by default. There is no authentication because there is
nothing multi-user here: it drives a local pipeline over local files, and it is
not meant to be exposed to a network.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import DATA_DIR, KAGGLE_DATASETS, RAW_DIR, RUNS_DIR  # noqa: F401
from . import stocks as stocks_mod
from .jobs import JobRegistry, list_run_dirs, make_fetch_job, make_null_test_job, make_run_job

STATIC = Path(__file__).parent / "static"
log = logging.getLogger(__name__)
app = FastAPI(title="qsm", docs_url=None, redoc_url=None)
registry = JobRegistry()


# --------------------------------------------------------------------------
# Request models
# --------------------------------------------------------------------------
class RunRequest(BaseModel):
    mode: str = Field(default="synthetic", pattern="^(synthetic|kaggle|live)$")
    dataset: str = "huge"
    provider: str = Field(default="yahoo", pattern="^(yahoo|tiingo)$")
    universe: str = "sp100"
    tickers: str = ""
    refresh: bool = False
    max_tickers: int = Field(default=500, ge=20, le=7000)
    start: str | None = None
    end: str | None = None
    horizon: int = Field(default=5, ge=1, le=63)
    folds: int = Field(default=6, ge=1, le=12)
    embargo: int = Field(default=5, ge=0, le=63)
    models: list[str] = ["ridge", "lgbm"]
    cost_bps: float = Field(default=10.0, ge=0, le=200)
    quantile: float = Field(default=0.1, gt=0.0, le=0.5)
    execution_lag: int = Field(default=1, ge=0, le=5)
    signal_strength: float = Field(default=0.05, ge=0.0, le=0.5)
    n_days: int = Field(default=2600, ge=600, le=8000)
    seed: int = 0
    tag: str = "web"


class NullTestRequest(BaseModel):
    seeds: list[int] = [11, 12, 13]
    models: list[str] = ["ridge", "lgbm"]


class FetchRequest(BaseModel):
    dataset: str = "huge"
    force: bool = False


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def _asset_stamp() -> str:
    """Fingerprint the UI assets so a changed file always defeats the cache.

    no-cache headers alone are not enough in practice: a browser that already
    holds app.js can keep running the old copy after an update, which presents
    as a feature that silently does nothing.
    """
    newest = 0.0
    for name in ("app.js", "styles.css"):
        f = STATIC / name
        if f.exists():
            newest = max(newest, f.stat().st_mtime)
    return str(int(newest))


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (STATIC / "index.html").read_text()
    stamp = _asset_stamp()
    html = html.replace("/static/app.js", f"/static/app.js?v={stamp}")
    html = html.replace("/static/styles.css", f"/static/styles.css?v={stamp}")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache, must-revalidate"})


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
@app.get("/api/status")
def status() -> dict:
    datasets = {}
    for key, slug in KAGGLE_DATASETS.items():
        path = RAW_DIR / slug.split("/")[-1]
        n = 0
        if path.exists():
            n = sum(1 for _ in path.rglob("*.txt")) + sum(1 for _ in path.rglob("*.csv"))
        datasets[key] = {"slug": slug, "path": str(path), "present": n > 0, "files": n}

    torch_ok = importlib.util.find_spec("torch") is not None

    return {
        "datasets": datasets,
        "torch_available": torch_ok,
        "runs_dir": str(RUNS_DIR),
        "active_jobs": sum(1 for j in registry.list() if j.status in ("queued", "running")),
    }


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------
@app.get("/api/live/status")
def live_status() -> dict:
    """Is live data usable, and how fresh is anything already cached?"""
    from .. import universe as uni

    available = importlib.util.find_spec("yfinance") is not None
    cached = []
    live_dir = DATA_DIR / "live"
    if live_dir.exists():
        for f in sorted(live_dir.glob("*.parquet"), key=lambda x: -x.stat().st_mtime)[:5]:
            age_h = (time.time() - f.stat().st_mtime) / 3600
            cached.append({"file": f.name, "age_hours": round(age_h, 1),
                           "mb": round(f.stat().st_size / 1e6, 1)})
    return {
        "available": available,
        "providers": ["yahoo"] + (["tiingo"] if os.environ.get("TIINGO_API_KEY") else []),
        "universes": {k: len(v) for k, v in uni.PRESETS.items()},
        "dynamic_universes": {k: d for k, (d, _) in uni.DYNAMIC.items()},
        "cached": cached,
    }


class WatchRequest(BaseModel):
    ticker: str = ""
    tickers: list[str] | None = None


class SettingsPatch(BaseModel):
    model_config = {"extra": "allow"}


@app.get("/api/settings")
def settings_get() -> JSONResponse:
    from .. import settings as st
    from ..live import market_state

    cfg = st.get()
    return JSONResponse(_clean({
        "settings": cfg,
        "defaults": st.DEFAULTS,
        "environment": {
            "torch": importlib.util.find_spec("torch") is not None,
            "yfinance": importlib.util.find_spec("yfinance") is not None,
            "kaggle_configured": bool(
                os.environ.get("KAGGLE_API_TOKEN")
                or (Path.home() / ".kaggle" / "access_token").exists()
                or (Path.home() / ".kaggle" / "kaggle.json").exists()),
            "tiingo_key": bool(os.environ.get("TIINGO_API_KEY")),
            "market": market_state(),
            "runs_dir": str(RUNS_DIR),
            "data_dir": str(DATA_DIR),
        },
    }))


@app.post("/api/settings")
def settings_post(patch: SettingsPatch) -> JSONResponse:
    from .. import settings as st

    try:
        return JSONResponse(_clean({"settings": st.update(patch.model_dump())}))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/settings/reset")
def settings_reset() -> JSONResponse:
    from .. import settings as st

    return JSONResponse(_clean({"settings": st.reset()}))


@app.get("/api/ledger")
def ledger() -> JSONResponse:
    """Live forecast record: what was predicted, and how it actually turned out."""
    from .. import online

    runs = list_run_dirs()
    horizon, closes = 5, None
    if runs:
        d = RUNS_DIR / runs[0]["name"]
        cfg_path = d / "config.json"
        if cfg_path.exists():
            horizon = int(json.loads(cfg_path.read_text())["labels"]["horizon"])
        panel_path = d / "ticker_panel.parquet"
        if panel_path.exists() and "close" in pd.read_parquet(panel_path).columns:
            closes = pd.read_parquet(panel_path)["close"].unstack()
    try:
        card = online.ledger_scorecard(closes if closes is not None else pd.DataFrame(), horizon)
    except Exception as exc:
        log.warning("ledger scorecard failed: %s", exc)
        card = {"logged": 0, "resolved": 0, "models": {}}
    return JSONResponse(_clean({"scorecard": card, "history": online.update_history(15),
                                "horizon": horizon}))


@app.get("/api/watchlist")
def watchlist_get(quotes: bool = True) -> JSONResponse:
    """The watchlist, with live prices and the latest model view of each name."""
    from .. import watchlist as wl
    from ..live import SYMBOL, currency_of, fetch_quotes

    items = wl.get()
    tickers = items["tickers"]
    payload: dict = {"tickers": tickers, "updated": items["updated"], "rows": []}
    if not tickers:
        return JSONResponse(payload)

    quote_map, market = {}, None
    if quotes:
        try:
            q = fetch_quotes(tickers)
            quote_map, market = q.get("quotes", {}), q.get("market")
        except Exception as exc:
            log.warning("watchlist quotes failed: %s", exc)

    # Latest model scores from the most recent run that covers each ticker.
    scores: dict[str, dict] = {}
    runs = list_run_dirs()
    if runs:
        for entry in runs[:5]:
            try:
                view = stocks_mod.load_view(RUNS_DIR / entry["name"])
            except Exception:
                continue
            last_rank, last_w = view.rank.iloc[-1], view.weights.iloc[-1]
            for t in tickers:
                if t in view.signal.columns and t not in scores:
                    r = last_rank.get(t)
                    w = float(last_w.get(t, 0.0))
                    if r == r:                                   # not NaN
                        scores[t] = {
                            "rank": round(float(r), 1),
                            "position": "long" if w > 1e-12 else ("short" if w < -1e-12 else "flat"),
                            "run": entry["name"],
                            "as_of": str(view.rank.index[-1].date()),
                        }
            if len(scores) == len(tickers):
                break

    for t in tickers:
        cur = currency_of(t)
        q = quote_map.get(t)
        payload["rows"].append({
            "ticker": t,
            "currency": cur,
            "symbol": SYMBOL.get(cur, ""),
            "price": q["price"] if q else None,
            "change_pct": q["change_pct"] if q else None,
            "prev_close": q["prev_close"] if q else None,
            "model": scores.get(t),
        })
    payload["market"] = market

    # The model keeps its own list. Whatever it holds, or has an order resting
    # on, belongs on the watchlist without anyone typing it in — and the rows
    # say which is which so a name you added yourself is still yours.
    try:
        state = _fund_state()
    except Exception as exc:
        log.warning("watchlist could not read the fund: %s", exc)
        state = None
    if state:
        payload["fund"] = {
            "budget": state.get("budget"), "value": state.get("value"),
            "pnl": state.get("pnl"), "pnl_pct": state.get("pnl_pct"),
            "cash": state.get("cash"), "checks_every": state.get("checks_every"),
            "benchmark_result": state.get("benchmark_result"),
            "vs_benchmark": state.get("vs_benchmark"),
            "holdings": [h["ticker"] for h in (state.get("holdings") or [])],
            "orders": [o["ticker"] for o in (state.get("orders") or [])],
            "resting": [
                {"ticker": o.get("ticker"), "limit": o.get("limit"),
                 "reference": o.get("reference"),
                 "pct_below": round((1 - o["limit"] / o["reference"]) * 100, 2)
                 if o.get("reference") else None,
                 "timing": o.get("timing")}
                for o in (state.get("orders") or [])
            ],
        }

    # When each name has historically reached a dip, so the card can say when a
    # limit order would plausibly fill rather than only what price to wait for.
    try:
        from .. import settings as _st
        from ..live import dip_profile

        dip = float((_st.get() or {}).get("dip_pct", 0.01) or 0.01)
        timing = dip_profile(tickers, dip)
        for row in payload["rows"]:
            row["dip"] = timing.get(row["ticker"])
        payload["dip_pct"] = dip
    except Exception as exc:
        log.warning("watchlist dip timing failed: %s", exc)

    return JSONResponse(_clean(payload))


@app.post("/api/watchlist")
def watchlist_add(req: WatchRequest) -> JSONResponse:
    from .. import watchlist as wl

    try:
        if req.tickers is not None:
            return JSONResponse(_clean(wl.replace(req.tickers)))
        return JSONResponse(_clean(wl.add(req.ticker)))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/watchlist/{ticker}")
def watchlist_remove(ticker: str) -> JSONResponse:
    from .. import watchlist as wl

    return JSONResponse(_clean(wl.remove(ticker)))


@app.get("/api/history/{ticker}")
def price_history(ticker: str, range: str = "6mo") -> JSONResponse:
    """Chart history for one ticker at a named range (1d … max)."""
    from ..live import price_history as ph

    try:
        return JSONResponse(_clean(ph(ticker, range)))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


class TradeRequest(BaseModel):
    ticker: str
    quantity: float = 1.0
    price: float | None = None
    note: str = ""


@app.get("/api/portfolio")
def portfolio_get() -> JSONResponse:
    """Holdings with live valuation and profit/loss against the entry price."""
    from .. import portfolio as pf
    from ..live import SYMBOL, currency_of, fetch_quotes

    positions = pf.get()
    if not positions:
        return JSONResponse({"positions": [], "total": None})

    tickers = [p["ticker"] for p in positions]
    quotes, market = {}, None
    try:
        q = fetch_quotes(tickers)
        quotes, market = q.get("quotes", {}), q.get("market")
    except Exception as exc:
        log.warning("portfolio quotes failed: %s", exc)

    rows, cost_total, value_total, unpriced = [], 0.0, 0.0, 0
    for p in positions:
        q = quotes.get(p["ticker"])
        price = q["price"] if q else None
        cost = p["entry_price"] * p["quantity"]
        value = price * p["quantity"] if price is not None else None
        pnl = (value - cost) if value is not None else None
        # Only count a position in the totals when it is priced. Adding its cost
        # while dropping its value invents a loss the size of the position.
        if value is not None:
            cost_total += cost
            value_total += value
        else:
            unpriced += 1
        cur = currency_of(p["ticker"])
        rows.append({
            **p,
            "currency": cur,
            "symbol": SYMBOL.get(cur, ""),
            "price": price,
            "cost": round(cost, 2),
            "value": round(value, 2) if value is not None else None,
            "pnl": round(pnl, 2) if pnl is not None else None,
            "pnl_pct": round((value / cost - 1) * 100, 3) if value and cost else None,
            "day_change_pct": q["change_pct"] if q else None,
        })

    return JSONResponse(_clean({
        "positions": rows,
        "market": market,
        "total": {
            "cost": round(cost_total, 2),
            "value": round(value_total, 2),
            "pnl": round(value_total - cost_total, 2),
            "pnl_pct": round((value_total / cost_total - 1) * 100, 3) if cost_total else None,
            "positions": len(rows),
            "priced": len(rows) - unpriced,
            "unpriced": unpriced,
        },
    }))


class FundRequest(BaseModel):
    budget: float = 5000.0
    top_n: int = Field(default=10, ge=1, le=50)
    benchmark: str = "SPY"
    run: str | None = None


def _fund_run(state: dict) -> str:
    """The newest run that scores everything this fund is holding.

    Switching to a run that has never heard of a holding would read as "no
    longer scored" and dump the position for a reason that is really about
    coverage, not the model changing its mind.
    """
    held = {h["ticker"] for h in (state.get("holdings") or [])}
    pinned = state.get("run")
    for entry in list_run_dirs()[:10]:
        name = entry["name"]
        try:
            tickers, _ = _run_universe(name)
        except Exception:
            continue
        if not tickers:
            continue
        if held <= tickers:
            return name
    return pinned


def _fund_state():
    """Load the fund, act on any new session, and mark it to market."""
    from .. import fund as fd
    from ..live import fetch_live

    state = fd.get()
    if not state:
        return None

    if state.get("mode") == "autopilot":
        from ..live import fetch_quotes

        # Follow the freshest run that still covers what the fund holds, not the
        # run it was created against. A finished run's ranks never move, so a
        # fund pinned to one can only ever buy: every holding keeps the score it
        # had the day the run was saved, and the sell rule — rank below 70 —
        # becomes unreachable. The nightly retrain is what makes exits possible.
        active = _fund_run(state)
        if active != state.get("run"):
            state["run"] = active
            log.info("fund now following %s", active)
        view = stocks_mod.load_view(RUNS_DIR / active)
        ranks = view.rank.sort_index()

        # Live quotes for the names it could hold or buy — execution prices must
        # be what the market is showing right now, not the last stored close.
        latest = ranks.iloc[-1]
        watch = sorted({*(h["ticker"] for h in state.get("holdings", [])),
                        *[t for t, v in latest.items() if v == v and v >= 90.0][:25]})
        market, quotes = {"is_open": False, "state": "unknown"}, {}
        try:
            q = fetch_quotes(watch) if watch else {}
            quotes = {k: v["price"] for k, v in (q.get("quotes") or {}).items()}
            market = q.get("market") or market
        except Exception as exc:
            log.warning("fund quotes failed: %s", exc)

        state["market"] = market
        state["next_open"] = None if market.get("is_open") else next_market_open()

        # Historical fill timing for each resting order, so the page can say
        # when these names have actually reached that price before.
        pending = [o["ticker"] for o in (state.get("orders") or [])]
        if pending:
            try:
                from ..live import dip_profile

                prof = dip_profile(pending, state.get("dip_pct", 0.01))
                for o in state["orders"]:
                    o["timing"] = prof.get(o["ticker"])
            except Exception as exc:
                log.warning("dip profile failed: %s", exc)
        _watchlist_follow_fund(state)
        state["checks_every"] = f"{_TICK_SECONDS}s while open"
        state["autotrader_running"] = bool(_TICKER_THREAD and _TICKER_THREAD.is_alive())
        # Previous closes are the anchor for limit prices; live quotes are only
        # used to decide whether an order fills.
        prev_close = {t: float(v) for t, v in view.close.sort_index().iloc[-1].items()
                      if v == v}
        from .. import settings as _fs

        prefs = _fs.get() or {}
        enter = float(prefs.get("fund_enter_above", 90.0))
        exit_ = float(prefs.get("fund_exit_below", 70.0))
        if exit_ >= enter:      # an inverted band would buy and sell the same name
            exit_ = max(0.0, enter - 1.0)
        state = fd.rebalance(state, view.close.sort_index(), ranks,
                             live_prices=quotes, market_open=bool(market.get("is_open")),
                             reference_prices=prev_close,
                             enter_above=enter, exit_below=exit_,
                             max_positions=int(prefs.get("fund_max_positions", 10)),
                             sizing=prefs.get("fund_sizing", "equity"))
        bench_sym = state.get("benchmark", "SPY")
        try:
            spy = fetch_live([bench_sym], start="2026-01-01")
            bc = spy.pivot(index="date", columns="ticker", values="close")[bench_sym]
            hist = state.get("history") or []
            if hist:
                first, last = hist[0]["date"], hist[-1]["date"]
                a, b = float(bc.loc[first]), float(bc.loc[last])
                sh = int(state["budget"] // a)
                val = sh * b + (state["budget"] - sh * a)
                state["benchmark_result"] = {
                    "ticker": bench_sym, "shares": sh,
                    "entry_price": round(a, 2), "price": round(b, 2),
                    "value": round(val, 2), "pnl": round(val - state["budget"], 2),
                    "pnl_pct": round((val / state["budget"] - 1) * 100, 3),
                }
                state["vs_benchmark"] = round(state["pnl"] - state["benchmark_result"]["pnl"], 2)
        except Exception as exc:
            log.warning("fund benchmark failed: %s", exc)
        # Mark holdings at live prices where available.
        def _px(t):
            v = quotes.get(t)
            if v is None:
                v = view.close.iloc[-1].get(t)
            return float(v) if v is not None and v == v else None

        state["marks"] = []
        for h in state.get("holdings", []):
            px = _px(h["ticker"])
            cost = h["shares"] * h["entry_price"]
            state["marks"].append({
                **h, "price": round(px, 4) if px else None,
                "cost": round(cost, 2),
                "value": round(h["shares"] * px, 2) if px else None,
            })
        if state["marks"]:
            live_val = state["cash"] + sum(m["value"] for m in state["marks"]
                                           if m["value"] is not None)
            state["value"] = round(live_val, 2)
            state["pnl"] = round(live_val - state["budget"], 2)
            state["pnl_pct"] = round((live_val / state["budget"] - 1) * 100, 3)
        for m in state["marks"]:
            if m["value"] is not None:
                m["pnl"] = round(m["value"] - m["cost"], 2)
                m["pnl_pct"] = round((m["value"] / m["cost"] - 1) * 100, 3)
        state["as_of"] = state.get("last_rebalance")
        return state

    tickers = [p["ticker"] for p in state["plan"]] + [state.get("benchmark", "SPY")]
    panel = fetch_live(tickers, start="2026-01-01")
    closes = panel.pivot(index="date", columns="ticker", values="close").sort_index()
    bench = closes[state.get("benchmark", "SPY")] if state.get("benchmark", "SPY") in closes else None
    return fd.mark(closes, bench)


def _watchlist_follow_fund(state: dict) -> None:
    """Put whatever the model owns or is bidding for onto the watchlist.

    Called from the state builder rather than the page, so the list keeps up
    with the background trading loop while the browser is shut. Names are only
    ever added: removing one is the user's call, and a name they took off is
    not re-added while the model still holds it — that would be an argument
    the page always wins.
    """
    from .. import watchlist as wl

    names = [h["ticker"] for h in (state.get("holdings") or [])]
    names += [o["ticker"] for o in (state.get("orders") or [])]
    if not names:
        return

    try:
        wl.follow(names)
    except Exception as exc:                 # a full list must not stop trading
        log.warning("watchlist follow failed: %s", exc)


@app.get("/api/fund")
def fund_get() -> JSONResponse:
    try:
        return JSONResponse(_clean(_fund_state() or {"exists": False}))
    except Exception as exc:
        log.warning("fund mark failed: %s", exc)
        raise HTTPException(503, "Could not value the fund") from exc


@app.post("/api/fund")
def fund_create(req: FundRequest) -> JSONResponse:
    from .. import fund as fd

    plan = json.loads(trade_plan(req.run).body)
    name = plan["run"]
    view = stocks_mod.load_view(RUNS_DIR / name)
    res = stocks_mod.search(view, position="long", sort="rank", limit=req.top_n)
    picks = [{"ticker": r["ticker"], "weight": r["latest_weight"], "rank": r["latest_rank"]}
             for r in res["rows"]]
    try:
        fd.create(req.budget, picks, plan["entry"]["date"], plan["exit"]["date"],
                  run=name, benchmark=req.benchmark.upper())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(_clean(_fund_state()))


@app.post("/api/fund/autopilot")
def fund_autopilot(req: FundRequest) -> JSONResponse:
    """Start a fund that trades itself off the model's daily signal."""
    from .. import fund as fd

    runs = list_run_dirs()
    if not runs:
        raise HTTPException(409, "No runs yet — run a backtest first.")
    name = req.run or next((r["name"] for r in runs if "sp500" in r["name"]), runs[0]["name"])
    try:
        fd.start_autopilot(req.budget, run=name, benchmark=req.benchmark.upper())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(_clean(_fund_state()))


@app.delete("/api/fund")
def fund_delete() -> JSONResponse:
    from .. import fund as fd

    fd.clear()
    return JSONResponse({"exists": False})


@app.get("/api/find-run")
def find_run(ticker: str, limit: int = 5) -> JSONResponse:
    """Which runs cover this ticker, largest universe first.

    The newest run is not necessarily the useful one: a 30-name Dow run will
    sit at the top of the list while the ticker you asked about only exists in
    a 500-name run.
    """
    t = ticker.strip().upper()
    found = []
    for entry in list_run_dirs()[:25]:
        # Reads the panel's ticker column, not the whole StockView. Building a
        # view per run re-derives weights over the full history — a second or
        # more each — to answer "is this name in here", on the path a watchlist
        # click waits behind.
        try:
            tickers, as_of = _run_universe(entry["name"])
        except Exception:
            continue
        if t in tickers:
            found.append({"run": entry["name"], "universe": len(tickers), "as_of": as_of})
        if len(found) >= limit:
            break
    found.sort(key=lambda r: -r["universe"])
    return JSONResponse({"ticker": t, "runs": found})


# Ticker -> which runs cover it. Building a full StockView per run to answer
# "does this run have GOOGL" reads the whole panel and re-derives weights, which
# is far too heavy for a search box. Only the ticker column is needed, and it is
# keyed on the parquet's mtime so a re-run invalidates it on its own.
_UNIVERSE_INDEX: dict[str, tuple[float, frozenset, str | None]] = {}
INDEX_RUN_LIMIT = 25


def _run_universe(name: str) -> tuple[frozenset, str | None]:
    """The set of tickers in one run, and the last date it covers."""
    path = RUNS_DIR / name / "ticker_panel.parquet"
    if not path.exists():
        return frozenset(), None
    mtime = path.stat().st_mtime
    hit = _UNIVERSE_INDEX.get(name)
    if hit and hit[0] == mtime:
        return hit[1], hit[2]

    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=["ticker", "date"])
        tickers = frozenset(str(t).upper() for t in table.column("ticker").to_pylist())
        dates = table.column("date").to_pylist()
        as_of = str(max(dates))[:10] if dates else None
    except Exception as exc:
        log.warning("could not index %s: %s", name, exc)
        tickers, as_of = frozenset(), None

    _UNIVERSE_INDEX[name] = (mtime, tickers, as_of)
    return tickers, as_of


@app.get("/api/search-runs")
def search_runs(q: str, exclude: str | None = None, limit: int = 12) -> JSONResponse:
    """Tickers matching `q` in any run other than `exclude`.

    The stock search is scoped to one run's universe, so a name the model has
    scored — in a different run — reads as "not available in this software".
    This is how those names stay findable without changing the selected run.
    """
    needle = q.strip().upper()
    if not needle:
        return JSONResponse({"q": needle, "rows": []})

    best: dict[str, dict] = {}
    for entry in list_run_dirs()[:INDEX_RUN_LIMIT]:
        name = entry["name"]
        if name == exclude:
            continue
        tickers, as_of = _run_universe(name)
        for t in tickers:
            if needle not in t:
                continue
            # Prefer the broadest run covering the name: a 30-name Dow run and a
            # 500-name S&P run both "have" AAPL, but only one is worth switching
            # to for a cross-section.
            prev = best.get(t)
            if prev is None or len(tickers) > prev["universe"]:
                best[t] = {"ticker": t, "run": name, "universe": len(tickers), "as_of": as_of}

    rows = sorted(best.values(), key=lambda r: (len(r["ticker"]), r["ticker"]))
    return JSONResponse({"q": needle, "rows": rows[:max(1, min(int(limit), 50))]})


@app.get("/api/plan")
def trade_plan(run: str | None = None) -> JSONResponse:
    """The strategy's own entry and exit rule, with dates.

    Not a chosen moment — this is the rule the backtest actually measured:
    enter `execution_lag` sessions after the signal, hold for the label horizon.
    Any figure here comes from the tested rule, not from picking a time.
    """
    import datetime as dt

    runs = list_run_dirs()
    if not runs:
        raise HTTPException(409, "No runs yet.")
    name = run or next((r["name"] for r in runs if "sp500" in r["name"]), runs[0]["name"])
    d = RUNS_DIR / name
    cfg = json.loads((d / "config.json").read_text())
    horizon = int(cfg["labels"]["horizon"])
    lag = int(cfg["backtest"]["execution_lag"])

    view = stocks_mod.load_view(d)
    signal_date = view.rank.index[-1].date()

    def sessions_after(day: dt.date, n: int) -> dt.date:
        count = 0
        while count < n:
            day += dt.timedelta(days=1)
            if day.weekday() < 5:      # holidays are not modelled — see note below
                count += 1
        return day

    entry = sessions_after(signal_date, lag)
    exit_ = sessions_after(entry, horizon)

    summary = pd.read_csv(d / "summary.csv", index_col=0)
    model = view.model if view.model in summary.index else summary.index[0]

    return JSONResponse(_clean({
        "run": name,
        "model": view.model,
        "signal_date": str(signal_date),
        "entry": {"date": str(entry), "weekday": entry.strftime("%A"), "when": "at the open"},
        "exit": {"date": str(exit_), "weekday": exit_.strftime("%A"), "when": "at the close"},
        "horizon": horizon,
        "execution_lag": lag,
        "holidays_modelled": False,
        "net_sharpe": _num_or_none(summary.loc[model, "sharpe"]),
        "benchmark_sharpe": _num_or_none(summary.loc["buy&hold universe", "sharpe"])
        if "buy&hold universe" in summary.index else None,
        "hit_rate": _num_or_none(summary.loc[model, "hit_rate"]),
    }))


@app.get("/api/portfolio/analytics")
def portfolio_analytics(benchmarks: str = "SPY,QQQ") -> JSONResponse:
    from .. import portfolio as pf

    data = json.loads(portfolio_get().body)
    syms = [b.strip().upper() for b in benchmarks.split(",") if b.strip()]
    try:
        return JSONResponse(_clean(pf.analytics(data.get("positions", []), benchmarks=syms)))
    except Exception as exc:
        log.warning("analytics failed: %s", exc)
        raise HTTPException(503, "Could not compute analytics") from exc


class AllocateRequest(BaseModel):
    balance: float = 10000.0
    run: str | None = None
    top_n: int = Field(default=10, ge=1, le=50)
    max_weight: float = Field(default=0.25, gt=0, le=1.0)
    whole_shares: bool = True


@app.post("/api/portfolio/allocate")
def portfolio_allocate(req: AllocateRequest) -> JSONResponse:
    """Split a notional balance across the model's current long picks."""
    from .. import portfolio as pf
    from ..live import fetch_quotes

    runs = list_run_dirs()
    if not runs:
        raise HTTPException(409, "No runs yet — run a backtest first.")
    name = req.run or runs[0]["name"]
    try:
        view = stocks_mod.load_view(RUNS_DIR / name)
    except Exception as exc:
        raise HTTPException(409, str(exc)) from exc

    res = stocks_mod.search(view, position="long", sort="rank", limit=req.top_n)
    picks = res["rows"]
    if not picks:
        raise HTTPException(409, "The model holds no long positions in that run.")

    quotes = {}
    try:
        quotes = fetch_quotes([p["ticker"] for p in picks]).get("quotes", {})
    except Exception:
        pass
    for p in picks:
        q = quotes.get(p["ticker"])
        p["price"] = q["price"] if q else p.get("last_price")
        p["weight"] = p.get("latest_weight")
        p["rank"] = p.get("latest_rank")

    try:
        out = pf.allocate(req.balance, picks, max_weight=req.max_weight,
                          whole_shares=req.whole_shares)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    summary = pd.read_csv(RUNS_DIR / name / "summary.csv", index_col=0)
    best = view.model if view.model in summary.index else summary.index[0]
    bench = "buy&hold universe"
    out["strategy"] = {
        "run": name, "model": view.model, "as_of": str(view.rank.index[-1].date()),
        "net_sharpe": _num_or_none(summary.loc[best, "sharpe"]),
        "benchmark_sharpe": _num_or_none(summary.loc[bench, "sharpe"])
        if bench in summary.index else None,
        "ann_return": _num_or_none(summary.loc[best, "ann_return"]),
    }
    return JSONResponse(_clean(out))


def _num_or_none(v):
    try:
        f = float(v)
        return None if f != f else round(f, 4)
    except (TypeError, ValueError):
        return None


@app.post("/api/portfolio/buy")
def portfolio_buy(req: TradeRequest) -> JSONResponse:
    from .. import portfolio as pf
    from ..live import fetch_quotes

    price = req.price
    if price is None:
        try:
            q = fetch_quotes([req.ticker]).get("quotes", {}).get(req.ticker.upper())
            price = q["price"] if q else None
        except Exception:
            price = None
    if price is None:
        raise HTTPException(422, f"No price available for {req.ticker}")
    try:
        return JSONResponse(_clean({"positions": pf.buy(req.ticker, req.quantity, price, req.note)}))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/portfolio/sell")
def portfolio_sell(req: TradeRequest) -> JSONResponse:
    from .. import portfolio as pf

    qty = req.quantity if req.quantity and req.quantity > 0 else None
    return JSONResponse(_clean({"positions": pf.sell(req.ticker, qty)}))


@app.post("/api/portfolio/clear")
def portfolio_clear() -> JSONResponse:
    from .. import portfolio as pf

    return JSONResponse({"positions": pf.clear()})


@app.get("/api/live/quotes")
def live_quotes(tickers: str = "", run: str | None = None) -> JSONResponse:
    """Current price and day change. `run` pulls the ticker list from a run."""
    from ..live import fetch_quotes

    syms = [t for t in tickers.replace(",", " ").split() if t]
    if run and not syms:
        try:
            view = stocks_mod.load_view(_run_dir(run))
            syms = list(view.signal.columns)
        except Exception:
            syms = []
    if not syms:
        raise HTTPException(400, "No tickers requested")
    try:
        return JSONResponse(_clean(fetch_quotes(syms)))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/api/runs")
def start_run(req: RunRequest) -> dict:
    params = req.model_dump()
    if req.mode == "kaggle":
        label = f"{req.dataset} · {req.max_tickers} names · h={req.horizon} · {'+'.join(req.models)}"
    elif req.mode == "live":
        label = f"live/{req.provider} · {req.universe} · h={req.horizon} · {'+'.join(req.models)}"
    else:
        label = f"synthetic · {req.max_tickers} names · h={req.horizon} · {'+'.join(req.models)}"
    job = registry.submit("run", label, params, make_run_job(params))
    return job.to_dict()


@app.post("/api/null-test")
def start_null_test(req: NullTestRequest) -> dict:
    params = req.model_dump()
    label = f"null test · seeds {', '.join(str(s) for s in req.seeds)}"
    job = registry.submit("null-test", label, params, make_null_test_job(params))
    return job.to_dict()


@app.post("/api/fetch")
def start_fetch(req: FetchRequest) -> dict:
    params = req.model_dump()
    job = registry.submit("fetch", f"download {req.dataset}", params, make_fetch_job(params))
    return job.to_dict()


@app.get("/api/jobs")
def jobs() -> list[dict]:
    return [j.to_dict() for j in registry.list()]


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str, log_from: int = 0) -> dict:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job")
    return job.to_dict(include_log=True, log_from=log_from)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
@app.get("/api/runs")
def runs() -> list[dict]:
    return list_run_dirs()


def _run_dir(name: str) -> Path:
    # Reject anything that is not a plain directory name directly under runs/.
    if "/" in name or "\\" in name or name in ("", ".", ".."):
        raise HTTPException(400, "Bad run name")
    path = (RUNS_DIR / name).resolve()
    if not str(path).startswith(str(RUNS_DIR.resolve())) or not path.is_dir():
        raise HTTPException(404, "No such run")
    return path


def _clean(obj):
    """JSON cannot carry NaN/Infinity; convert them to null."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


@app.get("/api/runs/{name}/results")
def run_results(name: str) -> JSONResponse:
    d = _run_dir(name)

    summary = pd.read_csv(d / "summary.csv", index_col=0)
    payload: dict = {
        "name": name,
        "summary": {
            "columns": list(summary.columns),
            "rows": [{"model": i, **r} for i, r in summary.to_dict(orient="index").items()],
        },
    }

    curves = pd.read_csv(d / "equity_curves.csv", index_col=0, parse_dates=True)
    # Downsample long curves; a 3000-point line does not render 3000 points of
    # information on a 900px chart.
    step = max(1, len(curves) // 900)
    thin = curves.iloc[::step]
    payload["equity"] = {
        "dates": [str(x.date()) for x in thin.index],
        "series": {c: [None if pd.isna(v) else float(v) for v in thin[c]] for c in thin.columns},
    }

    qpath = d / "quantile_returns.csv"
    if qpath.exists():
        q = pd.read_csv(qpath, index_col=0)
        payload["quantiles"] = {
            "bins": list(q.index),
            "ann_return": [float(v) for v in q["ann_return"]],
            "sharpe": [float(v) for v in q["sharpe"]],
        }

    ipath = d / "feature_importance.csv"
    if ipath.exists():
        imp = pd.read_csv(ipath, index_col=0)
        col = imp.columns[-1]
        top = imp.sort_values(col, ascending=False).head(18)
        payload["importance"] = {
            "features": list(top.index),
            "models": list(top.columns),
            "values": {c: [float(v) for v in top[c]] for c in top.columns},
        }

    for fname, key in (("config.json", "config"), ("metrics.json", "metrics")):
        fp = d / fname
        if fp.exists():
            payload[key] = json.loads(fp.read_text())

    fp = d / "folds.txt"
    if fp.exists():
        payload["folds"] = fp.read_text().strip().splitlines()

    payload["has_report"] = (d / "report.png").exists()
    return JSONResponse(_clean(payload))


@app.get("/api/runs/{name}/stocks")
def stock_search(name: str, q: str = "", position: str = "all", sort: str = "rank",
                 model: str | None = None, limit: int = 60) -> JSONResponse:
    """Search the run's universe. `q` is a case-insensitive ticker substring."""
    d = _run_dir(name)
    try:
        view = stocks_mod.load_view(d, model)
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    limit = max(1, min(int(limit), 500))
    return JSONResponse(_clean(stocks_mod.search(view, q, position, sort, limit)))


@app.get("/api/runs/{name}/stocks/{ticker}")
def stock_detail(name: str, ticker: str, model: str | None = None,
                 project: bool = False, horizon: int | None = None) -> JSONResponse:
    d = _run_dir(name)
    try:
        view = stocks_mod.load_view(d, model)
        payload = stocks_mod.detail(view, ticker)
        if project:
            payload["projection"] = _projection_for(d, view, payload["ticker"], horizon)
        return JSONResponse(_clean(payload))
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, f"{ticker} is not in this run's universe") from exc


# Calibration is the same for every ticker in a run, and rebuilding it per
# request is the slowest part of a projection. Keyed on the panel's mtime.
_CAL_CACHE: dict[str, tuple[float, dict, int]] = {}
_SKILL_CACHE: dict[tuple, dict] = {}


def _trend_skill(d: Path, view, horizon: int) -> dict:
    """Cached whole-panel trend skill.

    This is a property of the run, not of the ticker: the same number comes back
    for every name in it. Recomputing a spearman correlation over five million
    points on each chart draw cost about four seconds a request on a 4,000-name
    run, paid again on every poll.
    """
    from ..forecast import trend_skill

    key = (d.name, horizon, (d / "ticker_panel.parquet").stat().st_mtime)
    hit = _SKILL_CACHE.get(key)
    if hit is None:
        hit = trend_skill(view.close, horizon, lookback=60)
        if len(_SKILL_CACHE) > 32:
            _SKILL_CACHE.clear()
        _SKILL_CACHE[key] = hit
    return hit


def _calibration(d: Path, view) -> tuple[dict, int]:
    from ..forecast import calibrate

    panel_path = d / "ticker_panel.parquet"
    mtime = panel_path.stat().st_mtime
    hit = _CAL_CACHE.get(d.name)
    if hit and hit[0] == mtime:
        return hit[1], hit[2]

    panel = pd.read_parquet(panel_path)
    if "fwd_ret" not in panel.columns:
        raise HTTPException(409, "This run has no realised forward returns to calibrate against.")
    cal = calibrate(view.signal, panel["fwd_ret"].unstack())
    horizon = 5
    cfg_path = d / "config.json"
    if cfg_path.exists():
        horizon = int(json.loads(cfg_path.read_text())["labels"]["horizon"])
    _CAL_CACHE[d.name] = (mtime, cal, horizon)
    return cal, horizon


_EXPAND_CACHE: dict[tuple, object] = {}


def _expanding_bins(d: Path, view, horizon: int, step: int = 63):
    """Bin means as they would have looked before each date, not after.

    Refit periodically on everything up to the boundary minus the horizon, so a
    forecast is never calibrated using the outcome it is trying to predict.

    Done with one pass of running sums rather than by calling ``calibrate`` per
    boundary: that re-ranked and re-indexed a multi-million-row frame ~70 times
    and took 75 seconds on a 4,000-name run, which the chart then waited on.
    """
    import numpy as np

    key = (d.name, horizon, step, (d / "ticker_panel.parquet").stat().st_mtime)
    hit = _EXPAND_CACHE.get(key)
    if hit is not None:
        return hit

    panel = pd.read_parquet(d / "ticker_panel.parquet")
    sig = view.signal
    fwd = panel["fwd_ret"].unstack().reindex(index=sig.index, columns=sig.columns)

    # Rank within each date, and return relative to that date's universe — the
    # same two transforms calibrate() does, but computed once.
    ranks = sig.rank(axis=1, pct=True, na_option="keep").to_numpy()
    excess = fwd.sub(fwd.mean(axis=1), axis=0).to_numpy()

    n_dates, n_names = ranks.shape
    row_of = np.repeat(np.arange(n_dates), n_names)
    r, e = ranks.ravel(), excess.ravel()
    ok = np.isfinite(r) & np.isfinite(e)
    r, e, row_of = r[ok], e[ok], row_of[ok]          # already date-ordered

    N_BINS = 10
    edges = np.linspace(0, 1, N_BINS + 1)
    b = np.clip(np.digitize(r, edges[1:-1], right=True), 0, N_BINS - 1)

    sums = np.zeros(N_BINS)
    counts = np.zeros(N_BINS)
    dates = list(sig.index)
    fitted: list[tuple] = []
    ptr = 0
    for i in range(step, n_dates, step):
        cutoff_row = max(0, i - horizon)          # embargo the horizon
        while ptr < len(row_of) and row_of[ptr] < cutoff_row:
            np.add.at(sums, b[ptr], e[ptr])
            counts[b[ptr]] += 1
            ptr += 1
        if counts.sum() < 500:
            continue
        means = np.where(counts >= 20, sums / np.maximum(counts, 1), 0.0)
        fitted.append((dates[i], [float(v) for v in means]))

    def lookup(day):
        chosen = None
        for start_date, means in fitted:
            if start_date <= day:
                chosen = means
            else:
                break
        return chosen

    _EXPAND_CACHE[key] = lookup
    return lookup


@app.get("/api/runs/{name}/stocks/{ticker}/track")
def stock_track(name: str, ticker: str, model: str | None = None,
                limit: int = 400, honest: int = 1) -> JSONResponse:
    """What the model predicted for this name on every past date, and what happened.

    The live cone shows one forecast, from today, which can never be checked by
    looking at it. This is the same calculation run over the whole out-of-sample
    sample: each date's rank goes through the same calibration bins to a
    predicted price `horizon` sessions later, placed on the date it was
    predicting *for*, beside the price that actually printed. It is the model's
    report card rather than its promise.
    """
    d = _run_dir(name)
    try:
        view = stocks_mod.load_view(d, model)
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    t = ticker.upper()
    if t not in view.rank.columns:
        raise HTTPException(404, f"{t} is not in this run's universe")

    cal, horizon = _calibration(d, view)
    n_bins = int(cal["n_bins"])

    # Bins fit on the whole sample know how each past date turned out, so
    # scoring a 2026-05 call against them is marking your own homework with the
    # answers in front of you. Refit on an expanding window instead: each call
    # is scored only against what the model could have known before it, with a
    # `horizon` embargo so the outcome it is predicting is not in its own
    # calibration. `honest=0` restores the old behaviour for comparison.
    bins_at = _expanding_bins(d, view, horizon) if honest else None
    full_means = [float(b["mean"]) for b in cal["bins"]]

    ranks = view.rank[t]
    closes = view.close[t] if t in view.close.columns else None
    if closes is None:
        raise HTTPException(409, "This run has no close prices for that ticker.")

    dates = list(view.rank.index)
    rows = []
    for i, day in enumerate(dates):
        r, px = ranks.iloc[i], closes.iloc[i]
        j = i + horizon                      # the session this call was about
        if r != r or px != px or j >= len(dates):
            continue
        actual = closes.iloc[j]
        if actual != actual:
            continue
        b = int(min(max(int(r / 100 * n_bins), 0), n_bins - 1))
        means = bins_at(day) if bins_at else full_means
        if means is None:
            continue                          # not enough history to calibrate yet
        predicted = float(px) * (1 + means[b])
        rows.append({
            "from": str(day.date()), "for": str(dates[j].date()),
            "rank": round(float(r), 1), "decile": b + 1,
            "price_then": round(float(px), 4),
            "predicted": round(predicted, 4),
            "actual": round(float(actual), 4),
            "predicted_pct": round(means[b] * 100, 4),
            "actual_pct": round((float(actual) / float(px) - 1) * 100, 4),
        })

    rows = rows[-max(1, min(int(limit), 2000)):]
    hits = [r for r in rows if r["predicted_pct"] != 0]
    right = sum(1 for r in hits
                if (r["actual_pct"] > 0) == (r["predicted_pct"] > 0))
    return JSONResponse(_clean({
        "ticker": t, "run": name, "horizon": horizon, "points": len(rows),
        "calibration": "expanding, embargoed" if honest else "whole sample (hindsight)",
        "directional_hit_rate": round(right / len(hits), 4) if hits else None,
        "mean_abs_error_pct": round(
            sum(abs(r["actual_pct"] - r["predicted_pct"]) for r in rows) / len(rows), 4)
        if rows else None,
        "rows": rows,
    }))


@app.get("/api/runs/{name}/forecast")
def run_forecast(name: str, model: str | None = None, limit: int = 200) -> JSONResponse:
    """Forward value estimates for the run's universe, priced off live quotes."""
    from ..forecast import calibrate, project
    from ..live import fetch_quotes

    d = _run_dir(name)
    try:
        view = stocks_mod.load_view(d, model)
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc

    panel = pd.read_parquet(d / "ticker_panel.parquet")
    if "fwd_ret" not in panel.columns:
        raise HTTPException(409, "This run has no realised forward returns to calibrate against.")
    fwd = panel["fwd_ret"].unstack()

    horizon = 5
    cfg_path = d / "config.json"
    if cfg_path.exists():
        horizon = int(json.loads(cfg_path.read_text())["labels"]["horizon"])

    try:
        cal = calibrate(view.signal, fwd)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    tickers = list(view.signal.columns)[:limit]
    try:
        quotes = fetch_quotes(tickers)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    prices = {k: v["price"] for k, v in quotes.get("quotes", {}).items()}

    rows = project(view.rank.iloc[-1], prices, cal, horizon)
    return JSONResponse(_clean({
        "model": view.model,
        "horizon": horizon,
        "as_of": str(view.rank.index[-1].date()),
        "market": quotes.get("market"),
        "priced_at": quotes.get("fetched_at"),
        "calibration": cal,
        "rows": rows,
    }))


def _projection_for(d: Path, view, ticker: str, horizon: int | None = None) -> dict | None:
    """Live price plus a calibrated forward cone for one ticker."""
    from ..forecast import (calibrate, central_path, dip_statistics,
                            intraday_path, path, realised_vol, swing_points,
                            trend_path, trend_skill)
    from ..live import fetch_quotes

    try:
        # Shared, mtime-keyed cache. Calibrating per request re-read and
        # re-ranked the whole panel — a second on 500 names, seven and a half on
        # 4,000, paid again on every chart draw and every poll.
        cal, calibrated = _calibration(d, view)
        # The chart may ask for a different horizon than the model was fitted
        # at. Drift scales linearly in time and dispersion with its square
        # root, which is the right scaling for a random walk — but the drift
        # itself is only *validated* at `calibrated`, so the payload says so
        # and the UI flags any horizon that departs from it.
        horizon = int(horizon or calibrated)
        horizon = max(1, min(horizon, 252))
        scale = horizon / calibrated

        quotes = fetch_quotes([ticker]).get("quotes", {})
        q = quotes.get(ticker)
        rank = float(view.rank[ticker].dropna().iloc[-1])
        b = int(min(max(int(rank / 100 * cal["n_bins"]), 0), cal["n_bins"] - 1))
        stats = dict(cal["bins"][b])
        stats["mean"] = stats["mean"] * scale
        stats["std"] = stats["std"] * (scale ** 0.5)

        # Fall back to the last close in the sample if no live quote is available.
        price = q["price"] if q else float(view.close[ticker].dropna().iloc[-1])
        closes = view.close[ticker].dropna()
        arr = closes.to_numpy()
        trend = trend_path(arr, float(price), horizon, lookback=60)
        skill = _trend_skill(d, view, horizon)
        sigma = realised_vol(arr)
        dips = dip_statistics(float(price), stats["mean"], sigma, horizon, n_sims=2000, seed=11)

        # Stable per-ticker seed: the same stock shows the same line each visit.
        seed = abs(hash(ticker)) % 100_000
        scenarios, cone_std = None, stats["std"]
        if horizon <= 1:
            # Inside a single session, project from intraday bars. Daily-bar
            # sampling gives a nine-point near-straight line beside a chart of
            # 390 real intraday bars, which is why it looked fake.
            from ..live import intraday_returns

            bars = intraday_returns(ticker)
            if len(bars) >= 30:
                n_steps = 78                       # a session at 5-minute resolution
                scenarios = intraday_path(float(price), stats["mean"], bars,
                                          n_steps=n_steps, seed=seed)
                # Size the cone from the same intraday bars, so the band matches
                # the movement actually on screen rather than a daily estimate.
                cone_std = float(bars.std(ddof=1) * (n_steps ** 0.5))
        if scenarios is None:
            scenarios = central_path(arr, float(price), stats["mean"], horizon, seed=seed)
        swings = swing_points(arr[-180:], window=10)

        from ..live import SYMBOL, currency_of

        cur = currency_of(ticker)
        return {
            "currency": cur,
            "symbol": SYMBOL.get(cur, ""),
            "price": round(float(price), 4),
            "live": q is not None,
            "trend": trend,
            "trend_skill": skill,
            "dips": dips,
            "scenarios": scenarios,
            "swings": swings,
            "change_pct": q["change_pct"] if q else None,
            "decile": b + 1,
            "expected_return": round(stats["mean"], 6),
            "uncertainty": round(stats["std"], 6),
            "hit_rate": stats["hit"],
            "horizon": horizon,
            "calibrated_horizon": calibrated,
            "extrapolated": horizon != calibrated,
            "cone": path(float(price), stats["mean"], cone_std, horizon),
            "cone_std": round(float(cone_std), 6),
        }
    except Exception as exc:  # a missing projection must not break the page
        log.warning("projection failed for %s: %s", ticker, exc)
        return None


@app.get("/api/runs/{name}/report.png")
def run_report(name: str) -> FileResponse:
    path = _run_dir(name) / "report.png"
    if not path.exists():
        raise HTTPException(404, "No report image for this run")
    return FileResponse(path, media_type="image/png")


@app.get("/api/runs/{name}/download/{artifact}")
def run_artifact(name: str, artifact: str) -> FileResponse:
    allowed = {
        "summary.csv", "metrics.json", "equity_curves.csv", "quantile_returns.csv",
        "feature_importance.csv", "oos_predictions.parquet", "config.json", "folds.txt",
        "ticker_panel.parquet",
    }
    if artifact not in allowed:
        raise HTTPException(400, "Not a downloadable artifact")
    path = _run_dir(name) / artifact
    if not path.exists():
        raise HTTPException(404, "Missing artifact")
    return FileResponse(path, filename=f"{name}-{artifact}")


class _NoCacheStatic(StaticFiles):
    """Serve the UI assets with revalidation.

    Without this the browser happily keeps a stale app.js after the files
    change, and the page silently runs old code against a new API — which looks
    like a broken feature rather than a caching problem.
    """

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        return resp


# ── access control ────────────────────────────────────────────────────────
# Three modes, in order of exposure:
#   loopback only   no gate — nothing but this machine can connect
#   LAN (--lan)     shared word-pair, enough for a home network
#   public          password login with signed sessions
ACCESS_TOKEN: str | None = None
REQUIRE_LOGIN = os.environ.get("QSM_REQUIRE_LOGIN", "").lower() in ("1", "true", "yes")

_WORDS = (
    "amber anchor apple arrow autumn basil beacon birch bison bloom breeze "
    "cedar cherry cinder clover cobalt comet copper coral cosmic crimson delta "
    "dune ember falcon fern flint forest garnet ginger glacier granite harbor "
    "hazel heron indigo ivory jasper jungle juniper kernel lagoon lantern "
    "lemon lilac lunar maple marble meadow mesa mint moss nectar nimbus north "
    "oak ocean olive onyx opal orbit otter pebble pepper pine pixel plum "
    "quartz quill raven reef ripple river rowan saffron sage sandy shale "
    "silver slate solar spruce stone summit thistle thunder tide timber "
    "topaz tulip umber valley velvet vertex violet walnut willow zephyr"
).split()


def make_key(words: int = 2) -> str:
    """A short, sayable access key: 'copper-otter' beats 'L8J1WaVcnBRQ'."""
    import secrets

    return "-".join(secrets.choice(_WORDS) for _ in range(words))


def _page(title: str, body: str, status: int = 200) -> HTMLResponse:
    return HTMLResponse(f"""<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>qsm — {title}</title>
<style>
 body{{margin:0;min-height:100vh;display:grid;place-items:center;background:#0f1216;color:#e7ebf1;
      font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}}
 form,div.box{{width:min(92vw,340px);text-align:center}}
 h1{{font-family:ui-monospace,Menlo,monospace;font-size:1.4rem;letter-spacing:-.03em;margin:0 0 .2rem}}
 p{{color:#9aa5b5;font-size:.85rem;margin:0 0 1.4rem}}
 input{{width:100%;padding:.8rem;font-size:1rem;border-radius:10px;box-sizing:border-box;
       border:1px solid #262c36;background:#161a21;color:#e7ebf1;margin-bottom:.6rem}}
 input:focus{{outline:2px solid #1b2540;border-color:#5b8cff}}
 button{{width:100%;padding:.8rem;font-size:1rem;font-weight:600;border:0;
        border-radius:10px;background:#5b8cff;color:#fff;cursor:pointer}}
 .err{{color:#f0736f;font-size:.82rem;margin-top:.8rem;min-height:1.2em}}
</style>{body}""", status_code=status)


@app.get("/login", response_class=HTMLResponse)
def login_page(error: str = "") -> HTMLResponse:
    from . import auth

    if not auth.is_configured():
        return _page("setup", """<div class=box><h1>qsm</h1>
          <p>No password is set. Run this once on the server:</p>
          <input readonly value="qsm set-password" onclick="this.select()">
          <p style="margin-top:1rem">Then reload this page.</p></div>""", 503)
    return _page("sign in", f"""<form method=POST action="/login">
      <h1>qsm</h1><p>Sign in to the console</p>
      <input type=password name=password autofocus placeholder="Password"
             autocapitalize=off autocorrect=off>
      <button>Sign in</button>
      <div class=err>{'Wrong password.' if error else ''}</div></form>""")


@app.post("/login")
async def login_submit(request: Request):
    from . import auth

    form = await request.form()
    if not auth.verify(str(form.get("password", ""))):
        # Cheap throttle: makes a brute-force attempt expensive without state.
        import asyncio

        await asyncio.sleep(1.0)
        return RedirectResponse("/login?error=1", status_code=303)
    resp = RedirectResponse("/", status_code=303)
    resp.set_cookie(auth.COOKIE, auth.issue(), max_age=auth.SESSION_DAYS * 86400,
                    httponly=True, samesite="lax",
                    secure=os.environ.get("QSM_HTTPS", "1") == "1")
    return resp


@app.get("/logout")
def logout():
    from . import auth

    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(auth.COOKIE)
    return resp


@app.middleware("http")
async def _gate(request, call_next):
    from . import auth

    path = request.url.path
    if path in ("/login", "/logout") or path.startswith("/static"):
        return await call_next(request)

    if REQUIRE_LOGIN:
        if not auth.check(request.cookies.get(auth.COOKIE)):
            if path.startswith("/api/"):
                return JSONResponse({"detail": "Not signed in"}, status_code=401)
            return RedirectResponse("/login", status_code=303)
        return await call_next(request)

    if ACCESS_TOKEN:
        client = request.client.host if request.client else ""
        if client not in ("127.0.0.1", "::1", "localhost"):
            supplied = (request.query_params.get("k")
                        or request.headers.get("x-qsm-token")
                        or request.cookies.get("qsm_token"))
            if supplied != ACCESS_TOKEN:
                tried = "k" in request.query_params
                return _page("sign in", f"""<form method=GET>
                  <h1>qsm</h1><p>Enter the access key shown in the terminal</p>
                  <input name=k autofocus placeholder="two-words" autocapitalize=off>
                  <button>Open</button>
                  <div class=err>{'That key is not right.' if tried else ''}</div></form>""", 401)
            response = await call_next(request)
            response.set_cookie("qsm_token", ACCESS_TOKEN, max_age=30 * 86400,
                                httponly=True, samesite="lax")
            return response
    return await call_next(request)


app.mount("/static", _NoCacheStatic(directory=str(STATIC)), name="static")


def _loopback_sockets(port: int) -> list:
    """Listening sockets on both IPv6 and IPv4 loopback.

    `localhost` resolves to ::1 before 127.0.0.1 on macOS, so a server bound
    only to IPv4 is refused by browsers that try IPv6 first — the page simply
    fails to load. curl falls back to IPv4 and hides the problem.

    Both sockets are loopback-only, so this stays off the network.
    """
    import socket

    made = []
    for family, addr in ((socket.AF_INET6, "::1"), (socket.AF_INET, "127.0.0.1")):
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind((addr, port))
            sock.listen(128)
            sock.set_inheritable(True)
            made.append(sock)
        except OSError as exc:
            log.warning("could not bind %s:%s (%s)", addr, port, exc)
    return made


# ── background trading loop ───────────────────────────────────────────────
# Without this the fund only acts when someone loads the page: leave the
# browser shut on Monday morning and it sits in cash through the open. A tool
# that claims to trade the market has to check the market by itself.
_TICKER_THREAD = None
_TICK_SECONDS = 60


def _fund_loop() -> None:
    import time

    from ..live import market_state

    while True:
        try:
            from .. import fund as fd

            state = fd.get()
            if state and state.get("mode") == "autopilot":
                mk = market_state()
                # Poll often while open, idle slowly when shut.
                if mk["is_open"]:
                    _fund_state()          # rebalances as a side effect
                    time.sleep(_TICK_SECONDS)
                    continue
        except Exception as exc:           # never let the loop die
            log.warning("fund loop: %s", exc)
        time.sleep(300)


# ── daily learning ────────────────────────────────────────────────────────
# The model only improves if it is refit on data that now includes the days it
# got wrong, and only knows whether it got them wrong if yesterday's forecasts
# are scored against what happened. Both are one `update`: refetch, retrain the
# walk-forward, resolve the ledger, log fresh forecasts.
#
# Deliberately NOT here: re-weighting the ensemble toward whichever model has
# been right lately. That machinery exists (online.adaptive_weights) and was
# measured on 20260829-175555-sp500 before being wired in — it scored IC 0.0233
# / net Sharpe 0.473 against 0.0249 / 0.537 for the flat average, and got closer
# to the flat average the slower it adapted. It learns noise. Equal weight
# stays until an experiment says otherwise.
_LEARN_THREAD = None
_LEARN_CHECK_SECONDS = 900
_LEARN_AFTER_ET = (16, 20)          # a margin past the close, for the last bar
_LEARN_STATE = DATA_DIR / "last_learn.json"


def _learn_marker() -> dict:
    try:
        return json.loads(_LEARN_STATE.read_text())
    except Exception:
        return {}


def _learn_due() -> str | None:
    """The session date to learn from, or None if there is nothing to do yet."""
    import datetime as dt
    from zoneinfo import ZoneInfo

    from .. import settings as st

    if not (st.get() or {}).get("auto_learn", True):
        return None
    now = dt.datetime.now(tz=ZoneInfo("America/New_York"))
    if now.weekday() >= 5:                     # no session to learn from
        return None
    if (now.hour, now.minute) < _LEARN_AFTER_ET:
        return None                            # today's close has not settled
    today = str(now.date())
    mark = _learn_marker()
    if mark.get("session") != today:
        return today
    if mark.get("done"):
        return None
    # Retried, not abandoned — but a job that keeps failing must not relaunch a
    # full retrain every fifteen minutes for the rest of the evening.
    return today if mark.get("failed", 0) < 3 else None


def _run_daily_update(session: str) -> None:
    import datetime as _dt

    from ..config import Config
    from ..pipeline import update

    runs = list_run_dirs()
    if not runs:
        log.info("daily learning: no run to repeat yet")
        return

    # Repeat whatever the newest run was set up to do, with today's data.
    # Config(**raw) leaves the nested sections as plain dicts; from_json is the
    # constructor that actually rebuilds them.
    cfg = Config.from_json(RUNS_DIR / runs[0]["name"] / "config.json")
    cfg.data.source = "live"
    from .. import settings as st

    cfg = _learning_config(cfg, st.get() or {})

    log.info("daily learning: retraining on %s for session %s", cfg.data.universe, session)
    try:
        out = update(cfg, tag="daily")
    except Exception:
        prior = _learn_marker()
        n = (prior.get("failed", 0) if prior.get("session") == session else 0) + 1
        _LEARN_STATE.parent.mkdir(parents=True, exist_ok=True)
        _LEARN_STATE.write_text(json.dumps({
            "session": session, "done": False, "failed": n,
            "ran_at": _dt.datetime.now().isoformat(timespec="seconds"),
        }, indent=2))
        raise
    e = out["entry"]
    _LEARN_STATE.parent.mkdir(parents=True, exist_ok=True)
    _LEARN_STATE.write_text(json.dumps({
        "session": session, "done": True,
        "ran_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "run": e["run"], "last_bar": e["last_bar"],
        "forecasts_logged": e["forecasts_logged"],
        "ledger_resolved": e["ledger_resolved"],
        "live_ic": e["live_ic"],
    }, indent=2))
    log.info("daily learning: %s · %s forecasts logged · %s resolved",
             e["last_bar"], e["forecasts_logged"], e["ledger_resolved"])


def _warm_universe_index() -> None:
    for entry in list_run_dirs()[:INDEX_RUN_LIMIT]:
        try:
            _run_universe(entry["name"])
        except Exception as exc:
            log.warning("could not pre-index %s: %s", entry["name"], exc)
    log.info("ticker index warm for %d runs", len(_UNIVERSE_INDEX))


def _learning_config(cfg, prefs: dict):
    """Point a config at the universe and depth the nightly retrain should use.

    Breadth is the point of a cross-sectional model: it ranks names against each
    other, so a 30-name universe gives it 30 things to compare and a 15-year
    window gives it several regimes rather than one. The liquidity filters
    already in the config do the pruning, so the cap here is a ceiling and not a
    target.
    """
    import datetime as _dt

    chosen = prefs.get("auto_learn_universe") or ""
    if chosen:
        cfg.data.universe = chosen
        cfg.data.tickers = ()

    # Take the modelling shape from settings, not from whichever run happened to
    # be newest. Otherwise a two-minute dow30 test at 4pm silently becomes
    # tonight's fold count, and the model quietly retrains itself worse.
    cfg.labels.horizon = int(prefs.get("default_horizon") or cfg.labels.horizon)
    cfg.splits.n_splits = int(prefs.get("default_folds") or cfg.splits.n_splits)
    models = tuple(prefs.get("default_models") or ())
    if models:
        cfg.model.models = models
    years = int(prefs.get("auto_learn_years") or 0)
    if years > 0:
        start = _dt.date.today() - _dt.timedelta(days=int(round(years * 365.25)))
        cfg.data.start = start.isoformat()
    cap = int(prefs.get("auto_learn_max_tickers") or 0)
    if cap > 0:
        cfg.data.max_tickers = cap
    return cfg


def _learn_loop() -> None:
    import time

    while True:
        try:
            session = _learn_due()
            if session:
                _run_daily_update(session)
        except Exception as exc:               # never let the loop die
            log.warning("learning loop: %s", exc)
        time.sleep(_LEARN_CHECK_SECONDS)


@app.get("/api/activity")
def activity(limit: int = 400, kind: str = "all") -> JSONResponse:
    """Everything the model did, newest first, with the time it did it.

    Three streams that were only ever legible separately — the fund's fills and
    the reasons it skipped, the nightly retrains, and the names it put on the
    watchlist — merged onto one clock. "It bought CAT" is not an answer; "at
    11:24:36 it bought 1 CAT at 789.98 because the price dipped 1.3% below the
    800.25 previous close" is.
    """
    from .. import fund as fd, online
    from .. import watchlist as wl

    events: list[dict] = []

    state = fd.get() or {}
    for t in state.get("trades") or []:
        act = t.get("action")
        if act == "buy":
            detail = (f"bought {t['shares']} {t['ticker']} at "
                      f"{t['price']:,.2f} — {t.get('reason', '')}")
        elif act == "sell":
            pnl = t.get("pnl")
            detail = (f"sold {t['shares']} {t['ticker']} at {t['price']:,.2f}"
                      + (f" for {'+' if pnl >= 0 else '−'}${abs(pnl):,.2f}" if pnl is not None else "")
                      + f" — {t.get('reason', '')}")
        else:
            detail = f"passed on {t['ticker']} — {t.get('reason', '')}"
        events.append({
            "at": t.get("date"), "stream": "trade", "action": act,
            "ticker": t.get("ticker"), "detail": detail,
        })

    for h in online.update_history(60):
        ics = ", ".join(f"{m} {v:+.4f}" for m, v in (h.get("live_ic") or {}).items()
                        if v is not None) or "nothing resolved yet"
        events.append({
            "at": h.get("at"), "stream": "learning", "action": "retrain",
            "ticker": None,
            "detail": (f"retrained on {h.get('tickers')} names through {h.get('last_bar')} "
                       f"({h.get('universe')}) — logged {h.get('forecasts_logged')} forecasts, "
                       f"resolved {h.get('ledger_resolved')}; live IC {ics}"),
            "run": h.get("run"),
        })

    data = wl.get()
    if data.get("auto"):
        events.append({
            "at": data.get("updated"), "stream": "watchlist", "action": "follow",
            "ticker": None,
            "detail": ("put " + ", ".join(data["auto"]) + " on the watchlist "
                       "because it holds them or has an order resting on them"),
        })

    if kind != "all":
        events = [e for e in events if e["stream"] == kind]
    events.sort(key=lambda e: e.get("at") or "", reverse=True)
    total = len(events)
    events = events[:max(1, min(int(limit), 2000))]

    counts: dict[str, int] = {}
    for e in events:
        counts[e["action"]] = counts.get(e["action"], 0) + 1

    return JSONResponse(_clean({
        "events": events, "total": total, "counts": counts,
        "streams": ["trade", "learning", "watchlist"],
        "fund_run": state.get("run"),
        "autotrader": bool(_TICKER_THREAD and _TICKER_THREAD.is_alive()),
        "learner": bool(_LEARN_THREAD and _LEARN_THREAD.is_alive()),
    }))


@app.get("/api/learning")
def learning_state() -> JSONResponse:
    """What the nightly retrain has done, and when it next runs."""
    from .. import online, settings as st

    cfg = st.get() or {}
    return JSONResponse(_clean({
        "enabled": bool(cfg.get("auto_learn", True)),
        "running": bool(_LEARN_THREAD and _LEARN_THREAD.is_alive()),
        "checks_every_s": _LEARN_CHECK_SECONDS,
        "runs_after_et": f"{_LEARN_AFTER_ET[0]:02d}:{_LEARN_AFTER_ET[1]:02d}",
        "last": _learn_marker() or None,
        "due": _learn_due(),
        "history": online.update_history(10),
        "ensemble": {
            "mode": "equal weight",
            "why": "Adaptive weighting was measured on 20260829-175555-sp500 and lost to "
                   "the flat average — IC 0.0233 vs 0.0249, net Sharpe 0.473 vs 0.537 — at "
                   "every adaptation rate tried. It learns noise, so it is not used.",
        },
    }))


@app.on_event("startup")
def _start_fund_loop() -> None:
    import threading

    global _TICKER_THREAD, _LEARN_THREAD
    if _TICKER_THREAD is None:
        _TICKER_THREAD = threading.Thread(target=_fund_loop, daemon=True,
                                          name="qsm-fund-loop")
        _TICKER_THREAD.start()
        log.info("fund loop started (checks every %ss while the market is open)", _TICK_SECONDS)
    # Warm the ticker index off the request path. Building it lazily meant the
    # first watchlist click or cross-run search after a restart waited ~12s
    # while several hundred MB of parquet were read.
    threading.Thread(target=_warm_universe_index, daemon=True,
                     name="qsm-index-warm").start()
    if _LEARN_THREAD is None:
        _LEARN_THREAD = threading.Thread(target=_learn_loop, daemon=True,
                                         name="qsm-learn-loop")
        _LEARN_THREAD.start()
        log.info("learning loop started (retrains after %02d:%02d ET each session)",
                 *_LEARN_AFTER_ET)


def next_market_open() -> dict:
    """When the exchange next opens, in both exchange and local time."""
    import datetime as dt
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    now = dt.datetime.now(tz=et)
    candidate = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if now >= candidate or now.weekday() >= 5:
        candidate += dt.timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += dt.timedelta(days=1)
    local = candidate.astimezone()
    return {
        "exchange": candidate.strftime("%A %Y-%m-%d %-I:%M %p %Z"),
        "local": local.strftime("%A %Y-%m-%d %-I:%M %p %Z"),
        "iso": candidate.isoformat(),
        "minutes_away": int((candidate - now).total_seconds() // 60),
        "holidays_modelled": False,
    }


def lan_address() -> str | None:
    """This machine's LAN IP, for reaching the console from a phone."""
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))          # no packets sent; just picks the route
        addr = s.getsockname()[0]
        s.close()
        return addr
    except Exception:
        return None


def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False,
          token: str | None = None) -> None:
    import uvicorn

    global ACCESS_TOKEN
    ACCESS_TOKEN = token

    if reload:
        uvicorn.run("qsm.web.app:app", host=host, port=port, reload=True, log_level="warning")
        return

    # Explicit host (e.g. a container binding) keeps uvicorn's normal path.
    if host not in ("127.0.0.1", "localhost", "::1"):
        uvicorn.run(app, host=host, port=port, log_level="warning")
        return

    sockets = _loopback_sockets(port)
    if not sockets:
        raise RuntimeError(f"Could not bind port {port} on loopback.")
    config = uvicorn.Config(app, log_level="warning")
    uvicorn.Server(config).run(sockets=sockets)
