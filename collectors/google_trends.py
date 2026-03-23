"""
Signal 1: Google Trends
-----------------------
Tracks search interest for key terms related to Falcon jet purchasing.
Rising search interest for "Dassault Falcon" or "business jet" is a leading
indicator of prospective buyer activity.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def collect_google_trends(use_live: bool = False) -> pd.DataFrame:
    """
    Collect Google Trends data for Falcon-related keywords.
    
    If use_live=True, attempts to use pytrends (may be rate-limited).
    Otherwise, generates realistic synthetic data based on known
    delivery patterns and seasonality.
    """
    keywords = [
        "Dassault Falcon",
        "Falcon 10X",
        "business jet for sale",
        "Falcon 6X",
        "private jet purchase",
    ]

    if use_live:
        try:
            from pytrends.request import TrendReq

            pytrends = TrendReq(hl="en-US", tz=360)
            pytrends.build_payload(keywords[:5], timeframe="today 12-m", geo="")
            df = pytrends.interest_over_time()
            if not df.empty:
                df = df.drop(columns=["isPartial"], errors="ignore")
                df.index.name = "date"
                print("[Google Trends] Live data collected successfully.")
                return df
        except Exception as e:
            print(f"[Google Trends] Live fetch failed ({e}), using synthetic data.")

    # --- Synthetic data modeled on realistic patterns ---
    dates = pd.date_range(end=datetime(2026, 3, 23), periods=52, freq="W")
    n = len(dates)
    np.random.seed(42)

    # Dassault Falcon: baseline ~30, spike around air shows (Jun, Nov) and earnings (Mar)
    base = 30 + 10 * np.sin(np.linspace(0, 4 * np.pi, n))
    # Add spikes for: Paris Air Show (Jun), NBAA (Oct), Falcon 10X unveil (Mar 2026)
    spikes = np.zeros(n)
    spikes[12:14] += 25  # ~Jun air show
    spikes[min(30,n):min(32,n)] += 15  # ~Oct NBAA
    spikes[max(0,n-4):n] += 35  # Mar 2026 Falcon 10X unveiling
    dassault_falcon = np.clip(base + spikes + np.random.normal(0, 5, n), 5, 100)

    # Falcon 10X: low baseline, big spike in Mar 2026
    f10x_base = 8 + np.random.normal(0, 3, n)
    f10x_base[max(0,n-4):n] += 70  # Falcon 10X unveil spike
    falcon_10x = np.clip(f10x_base, 0, 100)

    # Business jet for sale: steady, slight uptrend
    bj_sale = 40 + 5 * np.linspace(0, 1, n) + np.random.normal(0, 4, n)

    # Falcon 6X: moderate interest
    f6x = 15 + np.random.normal(0, 4, n)
    f6x[12:14] += 10

    # Private jet purchase: correlated with macro sentiment
    pj_purchase = 35 + 8 * np.sin(np.linspace(0, 2 * np.pi, n)) + np.random.normal(0, 5, n)

    df = pd.DataFrame(
        {
            "Dassault Falcon": dassault_falcon.astype(int),
            "Falcon 10X": falcon_10x.astype(int).clip(0, 100),
            "business jet for sale": bj_sale.astype(int),
            "Falcon 6X": f6x.astype(int).clip(0, 100),
            "private jet purchase": pj_purchase.astype(int),
        },
        index=dates,
    )
    df.index.name = "date"
    print("[Google Trends] Synthetic data generated (52 weeks).")
    return df


def compute_trend_signal(df: pd.DataFrame) -> pd.Series:
    """
    Compute a composite trend signal from 0-100.
    Uses rolling z-score: a reading above 0 means interest is above
    its 3-month average → bullish for sales.
    """
    # Weight: Dassault Falcon most important, then purchase-intent terms
    weights = {
        "Dassault Falcon": 0.30,
        "Falcon 10X": 0.20,
        "business jet for sale": 0.20,
        "Falcon 6X": 0.10,
        "private jet purchase": 0.20,
    }

    composite = pd.Series(0.0, index=df.index)
    for col, w in weights.items():
        if col in df.columns:
            rolling_mean = df[col].rolling(12, min_periods=4).mean()
            rolling_std = df[col].rolling(12, min_periods=4).std().replace(0, 1)
            z = (df[col] - rolling_mean) / rolling_std
            composite += w * z

    # Normalize to 0-100 scale
    signal = 50 + composite * 15  # center at 50
    signal = signal.clip(0, 100)
    signal.name = "google_trends_signal"
    return signal


if __name__ == "__main__":
    df = collect_google_trends(use_live=False)
    signal = compute_trend_signal(df)
    print(df.tail(10))
    print("\nSignal (last 10 weeks):")
    print(signal.tail(10))
