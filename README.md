# qsm — quantitative equity forecasting on Kaggle price data

An end-to-end research pipeline: daily OHLCV in, cross-sectional return
forecasts out, evaluated by a walk-forward backtest that pays transaction costs.

Three models are trained side by side — a ridge baseline, gradient-boosted
trees, and a GRU sequence network — and combined into an ensemble.

**This is research infrastructure, not an investment product.** It does not
place orders and it is not investment advice. Read the *Honest limitations*
section before you take any number here seriously.

---

## Quick start

```bash
uv venv --python 3.12 && uv pip install -e ".[dev,web]"
```

Open the console in a browser — this is the easiest way in:

```bash
.venv/bin/qsm serve
```

### Stocks tab

Search the universe by ticker, filter to what the book is currently long or
short, sort by best/worst contributors, and click any name for its price, signal
rank over time, position weight and cumulative P&L contribution.

Position sizes come from the same `signal_to_weights` the backtest uses, driven
by the run's own saved config, so the per-stock figures reconcile exactly with
the book's gross P&L (`tests/test_stocks.py` asserts this). They sum to *gross*,
not to the headline net return: trading costs are charged against the whole book
and are not attributable to any single name.

Runs created before this feature have no `ticker_panel.parquet`; the tab says so
and asks you to re-run.

It opens automatically at **http://qsm.localhost:8000**, and you can type that
straight into any browser. Every `*.localhost` name resolves to loopback
system-wide (RFC 6761), so this needs no `/etc/hosts` entry, no root, and no DNS
setup — and it works in Safari and Firefox, not just Chrome. Pick a different
name with `--name research.localhost` if you prefer.

Dropping the `:8000` would mean binding port 80, which macOS reserves for root.
Not worth running a research server as root, so the port stays.

### Always-on

To have the console running whenever you type that URL, install it as a
per-user LaunchAgent — `~/Library/LaunchAgents`, no sudo, no root, nothing
system-wide:

```bash
./scripts/install-service.sh
```

It starts at login and `KeepAlive` restarts it if it ever dies. Logs go to
`logs/console.log`. To undo completely:

```bash
./scripts/uninstall-service.sh
```

Use a different port with `QSM_PORT=8010 ./scripts/install-service.sh`.

Configure a run in the sidebar, press **Run backtest**, and watch the log stream
live. Results land in the Results tab with an interpretation of what the numbers
actually mean, an equity curve, the decile staircase, and feature importances.
The **Run null test** button is the leakage check — use it after any change.

Or run the same thing headless, on generated data, with no credentials:

```bash
.venv/bin/qsm run --synthetic
```

For real data, authenticate with Kaggle first, then fetch and run:

```bash
.venv/bin/kaggle auth login
```

```bash
.venv/bin/qsm fetch --dataset huge && .venv/bin/qsm run --dataset huge --max-tickers 500
```

macOS note: LightGBM needs OpenMP — `brew install libomp`.

### Live market data

```bash
.venv/bin/qsm live --universe sp100
```

Daily adjusted OHLCV straight from the exchange through the last close, via
Yahoo — **no API key and no account needed**. `--provider tiingo` uses
`TIINGO_API_KEY` if you have one. Downloads are cached for 12 hours
(`--refresh` to force). Universes: `sp100`, `megacap`, `dow30`, plus
`--tickers PLTR,SNOW` for anything else.

Available in the web console as the **Live** source, and the Stocks tab then
shows what the model scores on the most recent close.

### Real-time prices

The Stocks tab has an **auto-update** switch that polls current prices every 20
seconds and shows each name's live price and day change. The indicator reports
the actual session — `Market open` (green, pulsing) versus `Market closed ·
showing last close` (amber) — because a quote pulled at the weekend is a stale
close wearing a live badge, and the UI should say so.

Quotes are fetched in **one batched request** for the whole table; the
per-symbol route takes ~9s for 30 names and ~40s for a full universe, which is
not "live" by any useful definition. Responses are cached for 10s, and polling
pauses while the browser tab is hidden.

### Forward value estimate

The Stocks tab has a **Show forward value estimate** switch: a target price per
name over the label horizon, with an 80% interval.

The model does not predict prices — it predicts a cross-sectional *rank*. The
bridge is empirical: bin the out-of-sample predictions, measure what each bin
actually went on to earn and how widely those outcomes scattered, then apply
that mapping to today's score and attach the scatter as an interval.

On a live S&P 100 run, calibrated over 208,000 out-of-sample observations:

| decile | mean 5-day excess return | std | hit rate |
|---|---|---|---|
| 1 (worst) | -0.08% | 3.9% | 48% |
| 10 (best) | **+0.28%** | **4.8%** | 51% |

Monotone across all ten bins (rank correlation 0.89), so the edge is real — and
the dispersion is about **17x the estimate**. That ratio is the honest headline,
and the UI leads with it. A target is where the centre of a very wide cloud
sits, not where the price is going, and `test_forecast.py` fails the build if
the interval ever stops dwarfing the point estimate.

Estimates are excess return *versus the universe*: the model has no view on
market direction, and adding one would dress a market forecast up as a stock
forecast.

**What does not update in real time: the model.** It is trained on daily bars
with a multi-day horizon, so its scores change at most once a day, when you
re-run. The prices tick; the signal does not.

**What live data does and does not fix.** It fixes staleness: the sample now
runs to yesterday instead of 2017. It does *not* fix survivorship bias, and
arguably makes it worse — a preset of companies prominent **today** is selected
on survival, so every historical backtest over it is optimistic and cannot see
the names that were delisted along the way. Point-in-time constituent data
(CRSP, Compustat, Sharadar) is the only real fix, and it is not free.

### Datasets

| key | Kaggle dataset | coverage |
|---|---|---|
| `huge` | `borismarjanovic/price-volume-data-for-all-us-stocks-etfs` | ~7k US stocks + ETFs, daily, through Nov 2017 |
| `jackson` | `jacksoncrow/stock-market-dataset` | ~5.8k US tickers, daily + adjusted close, through Apr 2020 |

Both layouts are parsed by `qsm.data`; `jackson`'s OHLC is rescaled by
`adj_close / close` so returns stay continuous across splits and dividends.

### Commands

```bash
qsm serve                                     # web console at 127.0.0.1:8000
qsm run --synthetic --models ridge lgbm gru   # full pipeline on generated data
qsm run --horizon 10 --folds 6 --cost-bps 15  # real data, 10-day horizon
qsm null-test                                 # leakage check (see below)
qsm report runs/<dir>                         # reprint a finished run
```

The web console binds to localhost and has no authentication — it drives a local
pipeline over local files and is not built to face a network. `--host` exists for
container use; think before you widen it.

### Runtime

Measured on this machine, 500 names x 13 years (1.45M rows, 6 walk-forward folds):

| models | wall clock |
|---|---|
| ridge + lgbm | 37s |
| ridge + lgbm + gru | 19 min |

The GRU dominates that, and mostly because it is pinned to one thread whenever
LightGBM shares the process (see Notes). Run it alone for full speed.

---

## How it works

```
Kaggle OHLCV
   -> point-in-time universe filter   (price, dollar volume, history floors)
   -> 33 features, ranked within each day
   -> label: forward h-day return, market-demeaned, ranked
   -> purged walk-forward CV          (ridge | lgbm | gru)
   -> out-of-sample predictions -> long/short book -> costs -> metrics
```

**Features** (`features.py`) — momentum across seven horizons plus 12-1,
moving-average distances, 52-week-range position, realised and idiosyncratic
volatility, ATR, vol-scaled short-term reversal, RSI, MACD, Bollinger %b,
dollar-volume trend and z-score, Amihud illiquidity, gap and intraday shape,
rolling market beta. Each is ranked cross-sectionally into `[-0.5, 0.5]`, which
turns raw quantities into *relative* signals and makes them comparable across
regimes — top decile today means the same thing in 2008 and in 2017.

**Label** — forward `horizon`-day return, demeaned within each date (so the
model predicts relative performance, not market direction, which a
dollar-neutral book cannot monetise anyway) and rank-transformed (so a handful
of takeover pops cannot dominate a squared-error fit).

**Validation** (`splits.py`) — expanding walk-forward. Because the label at date
`t` spans `t..t+h`, the last `h + embargo` training dates are **purged**, so no
training row contains a return realised inside its test window. Early stopping
uses a purged tail of the training block, never a random sample.

**Backtest** (`backtest.py`) — rank the day's forecasts, long the top decile,
short the bottom, equal weight within each leg, each leg normalised to half the
gross exposure so the book is dollar-neutral. Target weights are averaged over
the holding period (equivalent to `h` overlapping sleeves), which cuts turnover
roughly `h`-fold. Costs are charged on traded notional. **Signals formed at the
close of `t` earn the return from `t+1` to `t+2`** (`execution_lag=1`), so you
get a full session to trade after the signal exists.

Reported: Sharpe before and after costs, annualised return and vol, max
drawdown, Calmar, hit rate, turnover, and the information coefficient — the
daily rank correlation between forecast and outcome, which measures forecast
quality independently of portfolio construction.

---

## Does it leak?

Backtest bugs mostly produce *good-looking* results, so the pipeline is built to
be falsifiable. The synthetic generator can emit a panel with **no signal at
all**, and a correct pipeline must find nothing in it:

```bash
qsm null-test
```

Measured across 12 null runs, the IC t-statistic stayed within ±2.6 and
scattered symmetrically about zero; mean pre-cost Sharpe was 0.19. After costs
the strategy correctly just bleeds the spread — which is exactly what trading a
signal with no edge should do.

That guard is itself tested. `test_leakage_guard_has_teeth` injects a noisy copy
of the label as a feature on that same null panel and asserts the check fires
(it clears |t| > 20). A guard never shown to fail proves nothing.

The other load-bearing test is `test_features_are_causal`: every feature is
recomputed on a truncated panel and diffed against the full-history version. If
appending future data moves any past feature value, that is lookahead, and the
test fails. This catches forgotten shifts, centred windows, and global
normalisations — the usual culprits.

```bash
.venv/bin/python -m pytest tests -q     # 37 tests, ~10s
```

---

## Honest limitations

Read these before believing any backtested number.

1. **Survivorship bias — the big one.** Both Kaggle archives are static
   snapshots containing tickers that *existed on the scrape date*. Companies
   that went bankrupt or were delisted are simply absent. Backtesting on a
   universe pre-filtered for survival inflates returns substantially, and no
   amount of careful cross-validation fixes it. Point-in-time index membership
   (CRSP, Compustat, Sharadar) is the real remedy.
2. **The data stops in 2017/2020.** Neither archive covers the COVID crash, the
   2022 drawdown, or anything current. Results say nothing about today's market.
3. **No fundamentals, no sector neutrality.** The book can unintentionally
   concentrate in one sector, so some of what looks like alpha may be an
   uncompensated factor tilt. GICS sectors and a factor-risk model would let you
   neutralise and attribute properly.
4. **Costs are a flat per-notional charge.** Real costs include the spread,
   market impact that scales with participation rate, borrow fees on the short
   leg, and hard-to-borrow names you cannot short at all. Illiquid names look
   far better on paper than they trade.
5. **Corporate actions.** `huge` ships pre-adjusted prices and is trusted as-is;
   errors there propagate silently into returns.
6. **Multiple-testing.** Every knob you turn while watching the out-of-sample
   Sharpe spends some of that sample's validity. The walk-forward is honest
   *per run*; it cannot protect you from selecting the best of fifty runs.
7. **Capacity.** A decile book on 500 liquid names at daily turnover has a real
   capacity limit; this pipeline does not model it.

A realistic expectation for a signal like this on clean data is a daily IC
around 0.01–0.03 and a pre-cost Sharpe near 1, with costs taking a large bite.
Materially higher numbers usually mean a bug — start with the null test.

---

## Layout

```
scripts/        install-service.sh / uninstall-service.sh (LaunchAgent)
src/qsm/
  web/          FastAPI console: jobs.py (background runs), app.py, static/
  config.py     every tunable, serialised into each run directory
  data.py       Kaggle download, both archive parsers, synthetic generator
  features.py   33 features on wide date x ticker matrices
  labels.py     forward returns, market-demeaned and ranked
  splits.py     purged + embargoed walk-forward CV
  models.py     ridge / lightgbm / GRU behind one interface
  backtest.py   weights, costs, performance and IC metrics
  pipeline.py   orchestration and reporting
  cli.py        qsm entry point
tests/          37 tests, incl. causality, purging and the null experiment
runs/<stamp>/   summary.csv metrics.json equity_curves.csv quantile_returns.csv
                feature_importance.csv oos_predictions.parquet report.png folds.txt
                ticker_panel.parquet  (per-stock search + drill-down)
```

## Extending it

The highest-value additions, roughly in order: point-in-time universe data (kills
limitation 1), sector neutralisation, a proper cost model with participation-rate
impact, and volatility targeting on the portfolio. Adding a model means
subclassing `BaseModel` with `fit`/`predict` and registering it in
`models.build_model`.

## Notes

`lgbm` and `gru` in one process both load an OpenMP runtime (Homebrew's and
PyTorch's). On macOS the two thread pools deadlock, so `GRUModel` pins torch to a
single thread when LightGBM is present — without it, training hangs indefinitely
rather than erroring.
