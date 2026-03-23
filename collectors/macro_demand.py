"""
Signal 5: Macro Demand Proxies
-------------------------------
Business jet demand correlates with:
1. Corporate profits (S&P 500 earnings → C-suite budgets for jets)
2. UHNW population growth (billionaires/centi-millionaires)
3. Jet fuel prices (lower fuel → lower operating cost → more flying)
4. USD/EUR exchange rate (strong USD → US buyers find Falcons cheaper)
5. Business confidence indices (PMI, CEO confidence)

In production: use FRED API, Bloomberg, World Bank data.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def collect_macro_data(use_live: bool = False) -> pd.DataFrame:
    """
    Collect weekly macro indicators relevant to business jet demand.
    
    In production:
    - FRED API (free, key required): SP500 earnings, jet fuel, USD/EUR
    - Quandl: economic indicators
    - Bloomberg Terminal: UHNW data, business confidence
    """
    dates = pd.date_range(end=datetime(2026, 3, 23), periods=52, freq="W")
    n = len(dates)
    np.random.seed(99)

    # S&P 500 level (proxy for corporate wealth) — trending up
    sp500 = 5800 + 800 * np.linspace(0, 1, n) + np.random.normal(0, 80, n)
    # Correction in Q1 2026 due to tariff concerns
    sp500[40:48] -= 300

    # Jet-A fuel price (USD/gallon) — moderate, slight decline
    fuel_price = 5.80 - 0.60 * np.linspace(0, 1, n) + np.random.normal(0, 0.15, n)

    # USD/EUR rate — USD strengthening benefits US buyers
    usd_eur = 0.92 + 0.03 * np.linspace(0, 1, n) + np.random.normal(0, 0.008, n)

    # Global PMI (>50 = expansion) — mildly expansionary
    pmi = 51.5 + 1.5 * np.sin(np.linspace(0, 2 * np.pi, n)) + np.random.normal(0, 0.5, n)

    # UHNW count index (quarterly data, interpolated weekly)
    # ~400K UHNW individuals globally, growing ~5% per year
    uhnw_idx = 100 + 5 * np.linspace(0, 1, n) + np.random.normal(0, 0.3, n)

    # Bizjet utilization rate (% of fleet flying per month — from WingX/EUROCONTROL)
    utilization = 68 + 4 * np.sin(np.linspace(0, 2 * np.pi, n)) + np.random.normal(0, 1.5, n)
    # Summer peak
    utilization[16:28] += 5

    df = pd.DataFrame(
        {
            "sp500": sp500.round(0),
            "jet_fuel_usd_gal": fuel_price.round(2),
            "usd_eur": usd_eur.round(4),
            "global_pmi": pmi.round(1),
            "uhnw_index": uhnw_idx.round(1),
            "bizjet_utilization_pct": utilization.round(1),
        },
        index=dates,
    )
    df.index.name = "date"
    print("[Macro Data] Generated 52 weeks of macro indicators.")
    return df


def compute_macro_signal(df: pd.DataFrame) -> pd.Series:
    """
    Compute macro demand signal (0-100).
    
    Bullish when: stocks up, fuel down, USD strong, PMI >50, UHNW growing,
    utilization high.
    """
    signals = {}

    # S&P 500: 12-week momentum
    signals["equity"] = df["sp500"].pct_change(12).fillna(0)

    # Fuel: inverted — lower is better for demand
    signals["fuel"] = -df["jet_fuel_usd_gal"].pct_change(12).fillna(0)

    # PMI: above 50 is good, normalize
    signals["pmi"] = (df["global_pmi"] - 50) / 5

    # Utilization: higher = more demand
    signals["util"] = (df["bizjet_utilization_pct"] - 65) / 10

    # UHNW growth
    signals["uhnw"] = df["uhnw_index"].pct_change(12).fillna(0) * 10

    # USD/EUR: stronger USD (higher) benefits US buyers of French jets
    signals["fx"] = df["usd_eur"].pct_change(12).fillna(0) * 5

    weights = {"equity": 0.25, "fuel": 0.15, "pmi": 0.20, "util": 0.20, "uhnw": 0.10, "fx": 0.10}

    composite = pd.Series(0.0, index=df.index)
    for key, w in weights.items():
        composite += w * signals[key]

    signal = 50 + composite * 25
    signal = signal.clip(0, 100)
    signal.name = "macro_signal"
    return signal


if __name__ == "__main__":
    df = collect_macro_data()
    signal = compute_macro_signal(df)
    print(df.tail(10))
    print("\nSignal:")
    print(signal.tail(10))
