"""Command-line interface.

    qsm live                     run on live market data (no API key needed)
    qsm update                   refetch, retrain, log forecasts, score old ones
    qsm fetch                    download the Kaggle dataset
    qsm run                      full pipeline on the downloaded data
    qsm run --synthetic          full pipeline on generated data (no credentials)
    qsm null-test                leakage check: pipeline must score ~0 on noise
    qsm report <run_dir>         reprint the summary for a finished run
    qsm serve                    open the local web console in a browser
"""

from __future__ import annotations

import argparse
import os
import logging
import sys
from pathlib import Path

import pandas as pd

from .config import KAGGLE_DATASETS, RAW_DIR, Config
from .data import download_kaggle


def _log(verbosity: int) -> None:
    logging.basicConfig(
        level=logging.INFO if verbosity else logging.WARNING,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def _config_from_args(args) -> Config:
    cfg = Config()
    cfg.data.dataset = args.dataset
    cfg.data.max_tickers = args.max_tickers
    if args.start:
        cfg.data.start = args.start
    if args.end:
        cfg.data.end = args.end
    cfg.labels.horizon = args.horizon
    cfg.splits.n_splits = args.folds
    cfg.model.models = tuple(args.models)
    cfg.backtest.cost_bps = args.cost_bps
    cfg.backtest.quantile = args.quantile
    return cfg


def _add_live(p: argparse.ArgumentParser) -> None:
    p.add_argument("--universe", default="sp100",
                   help="Preset ticker list: sp100, megacap, dow30 (default: sp100)")
    p.add_argument("--tickers", default=None,
                   help="Extra symbols, comma or space separated (e.g. 'PLTR,SNOW')")
    p.add_argument("--provider", default="yahoo", choices=["yahoo", "tiingo"],
                   help="yahoo needs no key; tiingo reads TIINGO_API_KEY")
    p.add_argument("--refresh", action="store_true", help="Ignore the local cache")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", default="huge", choices=[*KAGGLE_DATASETS, "custom"],
                   help="Kaggle dataset key (default: huge)")
    p.add_argument("--data-root", type=Path, default=None,
                   help="Directory holding the extracted price files")
    p.add_argument("--max-tickers", type=int, default=500,
                   help="Universe size, most liquid first (default: 500)")
    p.add_argument("--start", default=None, help="Sample start date, YYYY-MM-DD")
    p.add_argument("--end", default=None, help="Sample end date, YYYY-MM-DD")
    p.add_argument("--horizon", type=int, default=5, help="Forecast horizon in trading days")
    p.add_argument("--folds", type=int, default=6, help="Walk-forward folds")
    p.add_argument("--models", nargs="+", default=["ridge", "lgbm"],
                   choices=["ridge", "lgbm", "gru"], help="Models to train")
    p.add_argument("--cost-bps", type=float, default=10.0, help="One-way cost in bps")
    p.add_argument("--quantile", type=float, default=0.1, help="Long/short quantile")
    p.add_argument("-v", "--verbose", action="count", default=1)


def _port_in_use(host: str, port: int) -> bool:
    import socket

    with socket.socket() as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qsm", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Download a Kaggle price dataset")
    p_fetch.add_argument("--dataset", default="huge")
    p_fetch.add_argument("--dest", type=Path, default=None)
    p_fetch.add_argument("--force", action="store_true")
    p_fetch.add_argument("-v", "--verbose", action="count", default=1)

    p_live = sub.add_parser("live", help="Run the pipeline on live market data")
    _add_common(p_live)
    _add_live(p_live)
    p_live.add_argument("--tag", default="live", help="Suffix for the run directory")

    p_upd = sub.add_parser("update", help="Refetch, retrain and score logged forecasts")
    _add_common(p_upd)
    _add_live(p_upd)
    p_upd.add_argument("--tag", default="update")

    p_run = sub.add_parser("run", help="Run the full pipeline")
    _add_common(p_run)
    _add_live(p_run)
    p_run.add_argument("--live", action="store_true", help="Use live data instead of Kaggle")
    p_run.add_argument("--synthetic", action="store_true",
                       help="Use generated data instead of Kaggle (no credentials needed)")
    p_run.add_argument("--tag", default="run", help="Suffix for the run directory")

    p_null = sub.add_parser("null-test", help="Leakage check on a pure-noise panel")
    p_null.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    p_null.add_argument("-v", "--verbose", action="count", default=0)

    p_pw = sub.add_parser("set-password", help="Set the login password for public deployments")
    p_pw.add_argument("--password", default=None,
                      help="Read from stdin or prompted if omitted")

    p_serve = sub.add_parser("serve", help="Run the local web console")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--name", default="localhost",
                         help="Hostname to print and open (any *.localhost name works "
                              "with no setup; default: qsm.localhost)")
    p_serve.add_argument("--lan", action="store_true",
                         help="Also serve on this machine's network address, for phones and "
                              "other devices. Requires an access key.")
    p_serve.add_argument("--no-open", action="store_true", help="Do not open a browser")
    p_serve.add_argument("-v", "--verbose", action="count", default=1)

    p_rep = sub.add_parser("report", help="Reprint a finished run's summary")
    p_rep.add_argument("run_dir", type=Path)
    p_rep.add_argument("-v", "--verbose", action="count", default=0)

    args = parser.parse_args(argv)
    _log(getattr(args, 'verbose', 0))
    pd.set_option("display.width", 220)

    if args.command == "fetch":
        try:
            dest = download_kaggle(args.dataset, args.dest, force=args.force)
        except RuntimeError as exc:
            print(f"\n{exc}", file=sys.stderr)
            return 2
        print(f"Data ready at: {dest}")
        return 0

    if args.command == "set-password":
        import getpass

        from .web import auth

        pw = args.password or os.environ.get("QSM_NEW_PASSWORD")
        if not pw:
            pw = getpass.getpass("New password (min 10 chars): ")
            if pw != getpass.getpass("Repeat: "):
                print("Passwords did not match.", file=sys.stderr)
                return 1
        try:
            auth.set_password(pw)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print("Password set. Start with QSM_REQUIRE_LOGIN=1 to require it.")
        return 0

    if args.command == "serve":
        try:
            from .web.app import serve
        except ImportError:
            print("The web console needs extra packages. Install them with:\n"
                  "  uv pip install -e \".[web]\"", file=sys.stderr)
            return 2

        # Every *.localhost name resolves to loopback system-wide (RFC 6761), so
        # this works in any browser with no /etc/hosts entry and no root.
        pretty = args.name if args.host == "127.0.0.1" else args.host
        url = f"http://{pretty}:{args.port}"

        token = None
        if args.lan:
            import subprocess

            from .web.app import lan_address, make_key

            args.host = "0.0.0.0"
            token = make_key()

            # Bonjour name: far easier to type on a phone than an IP, and it
            # keeps working when DHCP hands out a different address.
            host_name = None
            try:
                out = subprocess.run(["scutil", "--get", "LocalHostName"],
                                     capture_output=True, text=True, timeout=3)
                if out.returncode == 0 and out.stdout.strip():
                    host_name = out.stdout.strip() + ".local"
            except Exception:
                pass
            ip = lan_address()

            print("\n  LAN mode — open this on any device on your network:\n")
            if host_name:
                print(f"      http://{host_name}:{args.port}")
            if ip:
                print(f"      http://{ip}:{args.port}    (if the name above does not resolve)")
            print(f"\n  Access key:  {token}      (typed once per device)")
            print("\n  No accounts here: anyone on this network with that key can read your data\n"
                  "  and place paper trades. Not for public Wi-Fi.\n", flush=True)

        if _port_in_use(args.host, args.port):
            print(f"Something is already listening on {args.host}:{args.port}.\n"
                  f"If that is a qsm console, just open {url}\n"
                  f"Otherwise pick another port: qsm serve --port 8010", file=sys.stderr)
            return 2

        # flush=True: without it the banner sits in the buffer whenever stdout
        # is a pipe or a log file, and the user sees nothing until the server stops.
        print(f"\n  qsm console   {url}\n"
              f"  also at       http://127.0.0.1:{args.port}\n"
              f"  ctrl-c to stop\n", flush=True)
        if not args.no_open:
            import threading
            import webbrowser
            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        try:
            serve(host=args.host, port=args.port, token=token)
        except KeyboardInterrupt:
            print("\nstopped")
        return 0

    if args.command == "report":
        summary = pd.read_csv(args.run_dir / "summary.csv", index_col=0)
        print(summary.round(4).to_string())
        return 0

    from .pipeline import run, run_synthetic  # imported late: pulls in the heavy stack

    if args.command == "null-test":
        print("Null panel: no signal exists, so pre-cost Sharpe and IC must sit at ~0.")
        print("Anything materially positive here means the pipeline is leaking.\n")
        rows = {}
        for seed in args.seeds:
            cfg = Config()
            cfg.splits.n_splits = 3
            _, summary, _ = run_synthetic(
                cfg, signal_strength=0.0, n_tickers=80, n_days=1800, seed=seed,
                tag=f"null-{seed}",
            )
            for model in summary.index:
                if model == "buy&hold universe":
                    continue
                rows[(seed, model)] = summary.loc[model, ["sharpe_before_costs", "ic_mean", "ic_t_stat"]]
        out = pd.DataFrame(rows).T
        out.index.names = ["seed", "model"]
        print(out.round(4).to_string())
        worst = out["ic_t_stat"].abs().max()
        ok = pd.notna(worst) and worst < 4
        print(f"\nLargest |IC t-stat| across all runs: {worst:.2f}  "
              f"({'PASS' if ok else 'FAIL — investigate leakage'})")
        return 0 if ok else 1

    cfg = _config_from_args(args)

    if args.command == "update":
        from .pipeline import update

        cfg.data.source = "live"
        cfg.data.provider = args.provider
        cfg.data.universe = args.universe
        cfg.data.tickers = tuple(
            t for t in (args.tickers or "").replace(",", " ").split() if t)
        out = update(cfg, tag=args.tag)
        e = out["entry"]
        print(f"\nUpdated to {e['last_bar']} · {e['tickers']} tickers · "
              f"{e['forecasts_logged']} new forecasts logged")
        sc = out["scorecard"]
        print(f"ledger: {sc.get('resolved', 0)} resolved, {sc.get('pending', 0)} awaiting outcome")
        for m, v in sc.get("models", {}).items():
            ic = v.get("ic")
            print(f"  {m:10s} live IC {ic if ic is not None else 'n/a':>9} "
                  f"over {v.get('days', 0)} scored days")
        print("\n" + out["summary"].round(4).to_string())
        print(f"\nArtifacts: {out['run_dir']}")
        return 0

    if args.command == "live" or getattr(args, "live", False):
        cfg.data.source = "live"
        cfg.data.provider = args.provider
        cfg.data.universe = args.universe
        cfg.data.tickers = tuple(
            t for t in (args.tickers or "").replace(",", " ").split() if t
        )
        if args.refresh:
            cfg.data.max_age_hours = 0.0
        from .universe import resolve

        names = resolve(cfg.data.universe, list(cfg.data.tickers))
        print(f"Live data: {len(names)} tickers from {cfg.data.provider}, since {cfg.data.start}")
        print("Note: these are companies listed TODAY. Backtesting them is survivorship-biased —\n"
              "      the names that failed along the way are not in the list.\n")
        run_dir, summary, _ = run(cfg, tag=args.tag)
        print("\n" + "=" * 100)
        print(summary.round(4).to_string())
        print("=" * 100)
        print(f"\nArtifacts written to: {run_dir}")
        return 0

    if args.synthetic:
        run_dir, summary, _ = run_synthetic(cfg, tag=args.tag)
    else:
        root = args.data_root or RAW_DIR / KAGGLE_DATASETS.get(args.dataset, args.dataset).split("/")[-1]
        if not Path(root).exists():
            print(f"No data at {root}. Run `qsm fetch --dataset {args.dataset}` first, "
                  f"or pass --data-root, or use --synthetic.", file=sys.stderr)
            return 2
        run_dir, summary, _ = run(cfg, data_root=root, tag=args.tag)

    print("\n" + "=" * 100)
    print(summary.round(4).to_string())
    print("=" * 100)
    print(f"\nArtifacts written to: {run_dir}")
    print("  summary.csv  metrics.json  equity_curves.csv  quantile_returns.csv")
    print("  feature_importance.csv  oos_predictions.parquet  report.png  folds.txt")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
