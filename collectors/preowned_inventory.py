"""
Signal 2: Pre-Owned Falcon Inventory
-------------------------------------
Tracks the number of Falcon jets listed for sale on resale platforms.
FEWER listings = tighter market = stronger demand for NEW aircraft.

In production: scrape Controller.com, AvBuyer, JetNet.
Here: realistic synthetic data calibrated to actual market (≈100-130 Falcons
typically listed at any time, out of ~2,100 in service).
"""

import pandas as pd
import numpy as np
from datetime import datetime


# Real-world reference points (from Controller.com and industry data)
FLEET_IN_SERVICE = 2100
TYPICAL_LISTINGS_RANGE = (95, 135)  # normal range
MODELS = ["Falcon 900LX", "Falcon 7X", "Falcon 8X", "Falcon 2000LXS", "Falcon 6X", "Falcon 50"]


def collect_preowned_inventory(use_live: bool = False) -> pd.DataFrame:
    """
    Collect weekly pre-owned Falcon inventory counts.

    In a real implementation, this would scrape:
    - Controller.com/listings/for-sale/dassault/aircraft
    - AvBuyer.com
    - amstat.com / JetNet (paid)
    
    Key metrics:
    - total_listings: total Falcons for sale
    - avg_days_on_market: how long listings sit (longer = weaker demand)
    - new_listings_this_week: fresh supply entering market
    - asking_price_index: avg asking price indexed to 100
    """
    dates = pd.date_range(end=datetime(2026, 3, 23), periods=52, freq="W")
    n = len(dates)
    np.random.seed(123)

    # Inventory trend: was high mid-2025, declining into 2026 (market tightening)
    base_inventory = 125 - 20 * np.linspace(0, 1, n)  # gradual decline
    seasonal = 5 * np.sin(np.linspace(0, 2 * np.pi, n))  # slight seasonality
    noise = np.random.normal(0, 3, n)
    total_listings = np.clip(base_inventory + seasonal + noise, 70, 145).astype(int)

    # Days on market: declining = market absorbing faster
    dom_base = 180 - 40 * np.linspace(0, 1, n)
    dom = np.clip(dom_base + np.random.normal(0, 15, n), 80, 250).astype(int)

    # New listings per week
    new_per_week = np.clip(5 + np.random.normal(0, 1.5, n), 1, 12).astype(int)

    # Asking price index (100 = baseline Q2 2025, rising = sellers confident)
    price_idx = 100 + 12 * np.linspace(0, 1, n) + np.random.normal(0, 2, n)

    # Inventory by model (proportional breakdown)
    model_shares = {
        "Falcon 900LX": 0.28,
        "Falcon 7X": 0.22,
        "Falcon 8X": 0.15,
        "Falcon 2000LXS": 0.18,
        "Falcon 6X": 0.05,
        "Falcon 50": 0.12,
    }

    df = pd.DataFrame(
        {
            "total_listings": total_listings,
            "avg_days_on_market": dom,
            "new_listings_week": new_per_week,
            "asking_price_index": price_idx.round(1),
        },
        index=dates,
    )

    # Add model-level columns
    for model, share in model_shares.items():
        col_name = model.lower().replace(" ", "_").replace("-", "") + "_count"
        df[col_name] = (total_listings * share + np.random.normal(0, 1, n)).clip(0).astype(int)

    df.index.name = "date"
    print(f"[Pre-Owned Inventory] Generated 52 weeks. Latest: {total_listings[-1]} listings.")
    return df


def compute_inventory_signal(df: pd.DataFrame) -> pd.Series:
    """
    Compute inventory signal (0-100).
    
    Logic (INVERTED — fewer listings = stronger signal):
    - Declining inventory → bullish for new sales (score > 50)
    - Rising inventory → bearish (score < 50)
    - Fast-selling (low days on market) → additional boost
    """
    # Inventory change rate (4-week rolling)
    inv_change = df["total_listings"].pct_change(4)
    # Days on market change
    dom_change = df["avg_days_on_market"].pct_change(4)
    # Price momentum
    price_mom = df["asking_price_index"].pct_change(4)

    # Combine: inventory declining + DOM declining + prices rising = bullish
    raw = (-inv_change * 0.50) + (-dom_change * 0.30) + (price_mom * 0.20)

    # Normalize to 0-100
    signal = 50 + raw * 200  # scale factor
    signal = signal.clip(0, 100)
    signal.name = "inventory_signal"
    return signal


if __name__ == "__main__":
    df = collect_preowned_inventory()
    signal = compute_inventory_signal(df)
    print(df[["total_listings", "avg_days_on_market", "asking_price_index"]].tail(10))
    print("\nSignal (last 10):")
    print(signal.tail(10))
