"""Ticker universes for live data.

These lists are *static snapshots* of large, liquid US names. That is a real
limitation and worth stating plainly: backtesting a list of companies chosen
because they are prominent **today** is survivorship bias in its purest form.
Every name here survived to the present; the ones that blew up on the way are
absent, and the backtest cannot see them.

Use these to exercise the machinery and to generate current signals. Do not
read a historical backtest over them as an unbiased estimate of anything.
"""

from __future__ import annotations

# Mega/large-cap US names across sectors — enough breadth for a cross-section.
SP100 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "CRM",
    "AMD", "ADBE", "CSCO", "ACN", "INTC", "IBM", "QCOM", "TXN", "NOW", "INTU",
    "AMAT", "MU", "ADI", "LRCX", "KLAC", "PANW", "SNPS", "CDNS", "ANET", "MSI",
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "SCHW",
    "BLK", "C", "SPGI", "CB", "PGR", "MMC", "ICE", "CME", "PNC", "USB",
    "UNH", "LLY", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "AMGN",
    "ISRG", "BSX", "SYK", "GILD", "VRTX", "REGN", "CI", "ELV", "MDT", "HCA",
    "WMT", "COST", "PG", "KO", "PEP", "HD", "MCD", "NKE", "LOW", "SBUX",
    "TGT", "CL", "MDLZ", "MO", "PM", "KMB", "GIS", "SYY", "DG", "ORLY",
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "WMB",
    "CAT", "DE", "BA", "HON", "GE", "LMT", "RTX", "UPS", "UNP", "ADP",
    "LIN", "APD", "SHW", "ECL", "NEM", "FCX", "DOW", "NUE",
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE",
    "AMT", "PLD", "EQIX", "SPG", "O", "PSA", "CCI",
    "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS", "CHTR",
]

MEGACAP = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "BRK-B", "LLY",
    "JPM", "V", "UNH", "XOM", "MA", "COST", "WMT", "PG", "JNJ", "HD",
    "ORCL", "NFLX", "MRK", "ABBV", "CVX", "AMD", "KO", "PEP", "ADBE", "CRM",
]

DOW30 = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "GS",
    "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT",
    "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT", "DOW",
]

# ── Europe ───────────────────────────────────────────────────────────────
EUROPE = [
    # UK (.L — quoted in pence, see qsm.live.CURRENCY)
    "SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "BP.L", "GSK.L", "DGE.L", "RIO.L",
    "BATS.L", "REL.L", "LSEG.L", "NG.L", "VOD.L", "BARC.L", "LLOY.L", "PRU.L",
    "AAL.L", "TSCO.L", "IMB.L", "CPG.L", "RR.L", "STAN.L", "SSE.L", "BA.L",
    # France
    "MC.PA", "OR.PA", "TTE.PA", "AIR.PA", "SAN.PA", "SU.PA", "AI.PA", "BNP.PA",
    "EL.PA", "CS.PA", "DG.PA", "RMS.PA", "KER.PA", "SAF.PA", "VIE.PA", "ORA.PA",
    # Germany
    "SAP.DE", "SIE.DE", "ALV.DE", "MBG.DE", "SHL.DE", "SDF.DE", "BAS.DE", "BMW.DE",
    "SY1.DE", "DTE.DE", "BAYN.DE", "IFX.DE", "RWE.DE", "VOW3.DE", "MUV2.DE", "DBK.DE",
    "ADS.DE", "HEN3.DE", "EOAN.DE", "DB1.DE",
    # Netherlands / Belgium
    "ASML.AS", "INGA.AS", "AD.AS", "PHIA.AS", "HEIA.AS", "WKL.AS", "ABN.AS",
    "ABI.BR", "UCB.BR", "KBC.BR",
    # Switzerland
    "NESN.SW", "NOVN.SW", "UHR.SW", "ZURN.SW", "ABBN.SW", "SREN.SW", "GIVN.SW", "LONN.SW",
    # Italy / Spain / Nordics
    "ISP.MI", "ENI.MI", "UCG.MI", "ENEL.MI", "STLAM.MI", "G.MI",
    "ITX.MC", "SAN.MC", "IBE.MC", "BBVA.MC", "TEF.MC", "REP.MC",
    "VOLV-B.ST", "ATCO-A.ST", "SEB-A.ST", "INVE-B.ST", "HM-B.ST",
    "NOVO-B.CO", "MAERSK-B.CO", "DSV.CO", "EQNR.OL", "DNB.OL", "NOKIA.HE", "SAMPO.HE",
]

# ── Asia-Pacific ────────────────────────────────────────────────────────
ASIA = [
    # Japan
    "7203.T", "6758.T", "9984.T", "8306.T", "6861.T", "9432.T", "8035.T", "4063.T",
    "6501.T", "7267.T", "8316.T", "9433.T", "4502.T", "6902.T", "7741.T", "8058.T",
    "8031.T", "6098.T", "4661.T", "6367.T", "7974.T", "6273.T", "4568.T", "8766.T",
    # Hong Kong / China
    "0700.HK", "0941.HK", "1299.HK", "0005.HK", "0388.HK", "1810.HK", "3690.HK",
    "9988.HK", "2318.HK", "0883.HK", "1398.HK", "0939.HK", "2628.HK", "0175.HK",
    # Korea / Taiwan
    "005930.KS", "000660.KS", "005380.KS", "051910.KS", "035420.KS", "006400.KS",
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2412.TW",
    # India
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS",
    "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "LT.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    # Australia / Singapore
    "BHP.AX", "CBA.AX", "CSL.AX", "NAB.AX", "WBC.AX", "ANZ.AX", "WES.AX", "MQG.AX",
    "RIO.AX", "TLS.AX", "WOW.AX", "FMG.AX",
    "D05.SI", "O39.SI", "U11.SI", "Z74.SI",
]

# ── Americas outside the US ─────────────────────────────────────────────
AMERICAS = [
    "RY.TO", "TD.TO", "SHOP.TO", "ENB.TO", "CNR.TO", "BMO.TO", "BNS.TO", "CP.TO",
    "SU.TO", "TRP.TO", "ABX.TO", "MFC.TO",
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "B3SA3.SA", "ABEV3.SA",
    "WEGE3.SA", "BBAS3.SA", "RENT3.SA",
    "CNQ.TO", "ATD.TO", "T.TO", "NTR.TO",
]

WORLD = SP100 + EUROPE + ASIA + AMERICAS

PRESETS: dict[str, list[str]] = {
    "world": WORLD,
    "sp100": SP100,
    "europe": EUROPE,
    "asia": ASIA,
    "americas": AMERICAS,
    "megacap": MEGACAP,
    "dow30": DOW30,
}


# Presets fetched live from public directories rather than hardcoded. They are
# current membership applied to all of history — see the module docstring.
DYNAMIC = {
    "sp500": ("S&P 500 constituents", "sp500"),
    "us_all": ("every US-listed common stock", "us_listed"),
}


def dynamic_preset(name: str) -> list[str]:
    from . import symbols

    _, fn = DYNAMIC[name]
    return getattr(symbols, fn)()


def resolve(universe: str | None = None, tickers: list[str] | str | None = None) -> list[str]:
    """Turn a preset name and/or an explicit ticker list into a clean symbol list."""
    out: list[str] = []
    if universe:
        key = universe.strip().lower()
        if key in DYNAMIC:
            out += dynamic_preset(key)
        elif key in PRESETS:
            out += PRESETS[key]
        else:
            known = sorted(set(PRESETS) | set(DYNAMIC))
            raise ValueError(f"Unknown universe '{universe}'. Choose from: {', '.join(known)}")
    if tickers:
        if isinstance(tickers, str):
            tickers = [t for t in tickers.replace(",", " ").split() if t]
        out += [t.strip().upper() for t in tickers if t.strip()]
    if not out:
        out = list(SP100)

    seen, unique = set(), []
    for t in out:
        u = t.upper()
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique
