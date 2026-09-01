"""The paper fund's entry and exit rules.

The exit test exists because the rule was correct and still never fired: the
fund followed the run it was created against, and a finished run's ranks never
move, so every holding kept the score it had the day that run was saved and
"rank below 70" was unreachable. The fund could only ever buy. These pin down
both the rule and the freshness it depends on.
"""

from __future__ import annotations

import pandas as pd

from qsm import fund as fd


def _state(**over):
    base = {
        "budget": 5000.0, "cash": 1000.0, "mode": "autopilot",
        "holdings": [
            {"ticker": "AAA", "shares": 2, "entry_price": 100.0, "entry_date": "x"},
            {"ticker": "BBB", "shares": 1, "entry_price": 200.0, "entry_date": "x"},
        ],
        "orders": [], "trades": [],
    }
    base.update(over)
    return base


def _frames(rank_aaa: float, rank_bbb: float):
    idx = pd.to_datetime(["2026-08-31"])
    closes = pd.DataFrame({"AAA": [104.0], "BBB": [210.0]}, index=idx)
    ranks = pd.DataFrame({"AAA": [rank_aaa], "BBB": [rank_bbb]}, index=idx)
    return closes, ranks


PRICES = {"AAA": 104.0, "BBB": 210.0}


def test_sells_when_rank_falls_through_the_exit():
    closes, ranks = _frames(95.0, 22.0)
    out = fd.rebalance(_state(), closes, ranks, live_prices=PRICES,
                       market_open=True, reference_prices=PRICES)

    sells = [t for t in out["trades"] if t["action"] == "sell"]
    assert [s["ticker"] for s in sells] == ["BBB"]
    assert sells[0]["pnl"] == 10.0                      # 210 out against 200 in
    assert [h["ticker"] for h in out["holdings"]] == ["AAA"]
    assert out["cash"] == 1210.0                        # proceeds returned to cash


def test_holds_through_the_deadband():
    # 42 is well under the 90 entry but above the 30 exit. The wide deadband is
    # deliberate and was chosen out of sample: holding through a rank dip beat
    # reacting to it by roughly four times fewer trades and 12.30% a year
    # against 3.36% on the holdout. A narrower band sells here and does worse.
    closes, ranks = _frames(95.0, 42.0)
    out = fd.rebalance(_state(), closes, ranks, live_prices=PRICES,
                       market_open=True, reference_prices=PRICES)

    assert not [t for t in out["trades"] if t["action"] == "sell"]
    assert {h["ticker"] for h in out["holdings"]} == {"AAA", "BBB"}


def test_sells_a_name_the_model_no_longer_scores():
    closes, ranks = _frames(95.0, float("nan"))
    out = fd.rebalance(_state(), closes, ranks, live_prices=PRICES,
                       market_open=True, reference_prices=PRICES)

    sells = [t for t in out["trades"] if t["action"] == "sell"]
    assert [s["ticker"] for s in sells] == ["BBB"]
    assert "no longer scored" in sells[0]["reason"]


def test_does_not_trade_while_the_exchange_is_shut():
    closes, ranks = _frames(95.0, 42.0)
    out = fd.rebalance(_state(), closes, ranks, live_prices=PRICES,
                       market_open=False, reference_prices=PRICES)

    assert not out["trades"]
    assert out["blocked_reason"] == "market closed"


def test_equity_sizing_grows_the_slot_as_the_fund_grows():
    """Profit is put back to work rather than sitting as a bigger cash pile.

    Two funds with identical cash but different unrealised gains should not
    size a new position the same way: the one that is worth more buys more.
    """
    idx = pd.to_datetime(["2026-08-31"])
    closes = pd.DataFrame({"AAA": [100.0], "NEW": [50.0]}, index=idx)
    ranks = pd.DataFrame({"AAA": [95.0], "NEW": [99.0]}, index=idx)
    prices = {"AAA": 100.0, "NEW": 50.0}

    def slot_for(shares_held):
        state = {
            "budget": 5000.0, "cash": 1000.0, "mode": "autopilot",
            "holdings": [{"ticker": "AAA", "shares": shares_held,
                          "entry_price": 100.0, "entry_date": "x"}],
            "orders": [], "trades": [],
        }
        out = fd.rebalance(state, closes, ranks, live_prices=prices, market_open=True,
                           reference_prices=prices, sizing="equity")
        return next(o["budget"] for o in out["orders"] if o["ticker"] == "NEW")

    small, large = slot_for(10), slot_for(40)      # $1,000 vs $4,000 of stock
    assert large > small
    assert small == round((1000.0 + 10 * 100.0) / 10, 2)
    assert large == round((1000.0 + 40 * 100.0) / 10, 2)


def test_a_slot_never_exceeds_available_cash():
    idx = pd.to_datetime(["2026-08-31"])
    closes = pd.DataFrame({"AAA": [100.0], "NEW": [10.0]}, index=idx)
    ranks = pd.DataFrame({"AAA": [95.0], "NEW": [99.0]}, index=idx)
    prices = {"AAA": 100.0, "NEW": 10.0}
    state = {
        "budget": 5000.0, "cash": 120.0, "mode": "autopilot",
        "holdings": [{"ticker": "AAA", "shares": 40, "entry_price": 100.0, "entry_date": "x"}],
        "orders": [], "trades": [],
    }
    out = fd.rebalance(state, closes, ranks, live_prices=prices, market_open=True,
                       reference_prices=prices, sizing="equity")
    # equity/10 would be $410, but only $120 is actually spendable.
    assert next(o["budget"] for o in out["orders"] if o["ticker"] == "NEW") == 120.0
