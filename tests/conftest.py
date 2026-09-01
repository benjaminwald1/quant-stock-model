import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from qsm.data import make_synthetic


@pytest.fixture(scope="session")
def panel():
    return make_synthetic(n_tickers=30, n_days=900, seed=42, signal_strength=0.05)


@pytest.fixture(autouse=True)
def _isolate_user_data(tmp_path, monkeypatch):
    """Never let a test write to the real data directory.

    ``fund.rebalance`` persists as a side effect, so calling it in a test
    overwrites the live fund file — which is exactly how a test run once
    destroyed a running paper fund. Anything that writes under DATA_DIR gets
    redirected at the module attribute each writer actually uses.
    """
    from qsm import fund, online, portfolio, settings, watchlist

    redirect = {
        fund: [("FUND_PATH", "fund.json")],
        watchlist: [("WATCHLIST_PATH", "watchlist.json")],
        settings: [("SETTINGS_PATH", "settings.json")],
        portfolio: [("PORTFOLIO_PATH", "portfolio.json")],
        online: [("LEDGER_PATH", "ledger.parquet"), ("HISTORY_PATH", "update_history.json")],
    }
    for mod, names in redirect.items():
        for attr, filename in names:
            if hasattr(mod, attr):
                monkeypatch.setattr(mod, attr, tmp_path / filename)
    yield
