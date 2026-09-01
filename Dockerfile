# syntax=docker/dockerfile:1
FROM python:3.12-slim

# LightGBM needs libgomp at runtime; without it the import fails with a dlopen
# error that looks nothing like a missing system package.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so code edits do not invalidate the layer.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[web,live]"

# State lives on a mounted volume, not in the image — otherwise the fund,
# watchlist and every backtest are wiped on each deploy.
ENV QSM_DATA_DIR=/state/data \
    QSM_RUNS_DIR=/state/runs \
    QSM_REQUIRE_LOGIN=1 \
    QSM_HTTPS=1 \
    PYTHONUNBUFFERED=1
RUN mkdir -p /state/data /state/runs

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl -fs http://127.0.0.1:8080/login || exit 1

CMD ["qsm", "serve", "--host", "0.0.0.0", "--port", "8080", "--no-open"]
