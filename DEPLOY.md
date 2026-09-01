# Deploying the console

The app is a Python service, not a static site. It needs a host that runs
containers and keeps a disk. Cloudflare Pages cannot do that — see the note at
the bottom.

## What you do vs what is already done

Already in the repo: `Dockerfile`, `fly.toml`, `.dockerignore`, password login,
and configurable storage paths.

You do: buy the domain, create the host account, run the commands below. Those
need payment details and an account, which is why they are yours to run.

## Deploy to Fly.io

```bash
brew install flyctl          # or: curl -L https://fly.io/install.sh | sh
fly auth signup              # or `fly auth login`
cd /Users/Johnny/quant-stock-model
```

**1. Create the app** (decline when it offers to overwrite `fly.toml` — the
volume mount and health check in there are the point):

```bash
fly launch --no-deploy --name qsm-console --region iad
```

**2. Create the disk.** Without it every deploy wipes the fund, the watchlist
and all backtests:

```bash
fly volumes create qsm_state --size 3 --region iad
```

**3. Set a password** — it is hashed before it is stored, and the value never
reaches the image:

```bash
fly secrets set QSM_PASSWORD_HASH="$(
  python3 - <<'PY'
import base64, getpass, hashlib, json, secrets
pw = getpass.getpass("Password (min 10 chars): ")
assert len(pw) >= 10, "too short"
salt = secrets.token_bytes(16)
print(base64.b64encode(json.dumps({
    "user": "owner",
    "salt": base64.b64encode(salt).decode(),
    "hash": base64.b64encode(hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 240000)).decode(),
    "iterations": 240000,
}).encode()).decode())
PY
)"
fly secrets set QSM_SECRET_KEY="$(openssl rand -hex 32)"
```

**4. Deploy:**

```bash
fly deploy
```

Fly builds the image on its own machines, so Docker is not needed locally.

**5. Point your domain at it:**

```bash
fly certs add quantuminvestments.com
fly certs show quantuminvestments.com     # prints the DNS records to add
```

Add those records at your registrar. HTTPS is issued automatically once they
resolve.

## After it is live

Backtests and the fund run on the server, so it keeps trading with the browser
shut. To seed the first run:

```bash
fly ssh console -C "qsm live --universe sp500"
```

## Costs

About **$5–7/month**: one shared-cpu-1x machine with 1 GB RAM, plus a 3 GB
volume. `auto_stop_machines` is off on purpose — a sleeping machine does not
trade at the open.

## Vercel (read-only snapshot)

`vercel.json` is in the repo and points at `scripts/build_static.py`. What
deploys is the frozen console: every read-only response baked to JSON, ~99 MB,
no server.

```bash
npm i -g vercel
cd /Users/Johnny/quant-stock-model
python3 scripts/build_static.py     # writes dist/
vercel deploy --prebuilt=false --prod
```

**What works there:** results, metrics, charts, the per-stock pages with their
projections and the past-accuracy overlay, the activity log as it stood, and the
watchlist and fund frozen at build time. The page says *frozen snapshot* and
disables every control that would need a backend.

**What cannot work there, at all.** Vercel is serverless, and this app is not:

| the app needs | Vercel gives |
| --- | --- |
| a thread trading every 60s while the market is open | functions that run only on request |
| a 35-minute nightly retrain over 14.5M rows | a 10–60s function timeout |
| writable `fund.json`, `watchlist.json`, `ledger.parquet` | read-only disk outside `/tmp` |
| ~24 GB of runs and cached bars | a 250 MB bundle |
| live Yahoo quotes on every load | fine, but nothing to persist them to |

So a Vercel deploy is a **shop window, not the shop**. The fund does not trade,
prices do not move, and nothing learns.

If you want a real URL with the fund still running, put the frontend on Vercel
and keep the backend on Fly (above), then point `api()` in `app.js` at the Fly
host and enable CORS. Or skip Vercel and use Fly for both — it already works and
costs about $5/month.

## Why not Cloudflare Pages

Pages serves files. This app runs pandas and LightGBM to score stocks, calls
Yahoo for live prices, and writes JSON state. None of that survives on a static
host. `scripts/build_static.py` can produce a frozen read-only snapshot for
Pages, but the fund will not trade and prices will not update.

If you want the domain on Cloudflare, keep DNS there and point it at Fly with a
CNAME — Cloudflare handles the domain, Fly runs the app.

## Before you make it public

- The disclaimers matter more on a public URL than on localhost. A stranger
  landing on a page showing "STX · long · rank 100" may act on it, from a model
  whose measured result was **net Sharpe 0.54 against 1.16 for simply holding
  the index**.
- Login is one shared password. It is enough to keep the internet out; it is
  not multi-user, and there is no audit trail.
