"""
Signal 3: Job Postings / Hiring Intensity
------------------------------------------
Tracks job openings at Dassault Falcon Jet (US) and Dassault Aviation (FR).
Rising hiring in production, maintenance, and customer support roles
is a leading indicator of delivery ramp-up.

In production: scrape LinkedIn Jobs, Indeed, Dassault careers page.
"""

import pandas as pd
import numpy as np
from datetime import datetime


# Real reference: Dassault has ~14,600 employees (2024)
# Falcon Jet (US subsidiary) has ~2,500 employees
ROLE_CATEGORIES = {
    "production_assembly": "Technicians, assembly workers at Bordeaux-Mérignac, Little Rock",
    "engineering": "Design engineers, systems engineers, flight test",
    "customer_support": "Falcon service centers, field support reps",
    "sales_marketing": "Sales directors, regional managers, marketing",
    "pilots_flight_ops": "Demo pilots, delivery pilots, flight test pilots",
    "quality_inspection": "Quality control, compliance, certification",
}


def collect_job_postings(use_live: bool = False) -> pd.DataFrame:
    """
    Collect weekly job posting counts by category.

    In production, this would:
    1. Query LinkedIn Jobs API for "Dassault Falcon" + "Dassault Aviation"
    2. Scrape careers.dassault-aviation.com
    3. Monitor Indeed for Falcon Jet Corporation listings
    4. Track Glassdoor reviews for sentiment
    """
    dates = pd.date_range(end=datetime(2026, 3, 23), periods=52, freq="W")
    n = len(dates)
    np.random.seed(77)

    # Production hiring: ramps ahead of delivery targets
    # Dassault targets 40 Falcons in 2026 (up from 37 in 2025) → hiring wave
    prod = 12 + 8 * np.linspace(0, 1, n) + np.random.normal(0, 2, n)
    prod[35:45] += 6  # Pre-delivery ramp hiring burst (Q4 2025 - Q1 2026)

    # Engineering: steady with bump for Falcon 10X program
    eng = 8 + np.random.normal(0, 1.5, n)
    eng[40:52] += 5  # 10X approaching first flight → engineering hires

    # Customer support: grows with fleet size
    support = 6 + 3 * np.linspace(0, 1, n) + np.random.normal(0, 1, n)
    # Melbourne FL facility opening in Oct 2025 → support hiring
    support[28:35] += 4

    # Sales: spikes before air shows
    sales = 3 + np.random.normal(0, 0.8, n)
    sales[10:13] += 2  # pre-Paris Air Show
    sales[28:30] += 1.5  # pre-NBAA

    # Pilots: small numbers but telling
    pilots = 2 + np.random.normal(0, 0.5, n)
    pilots[45:52] += 2  # 10X flight test program

    # Quality: rises with production
    quality = 4 + 2 * np.linspace(0, 1, n) + np.random.normal(0, 1, n)

    df = pd.DataFrame(
        {
            "production_assembly": prod.clip(2).astype(int),
            "engineering": eng.clip(1).astype(int),
            "customer_support": support.clip(1).astype(int),
            "sales_marketing": sales.clip(0).astype(int),
            "pilots_flight_ops": pilots.clip(0).astype(int),
            "quality_inspection": quality.clip(1).astype(int),
        },
        index=dates,
    )
    df["total_openings"] = df.sum(axis=1)
    df.index.name = "date"
    print(f"[Job Postings] Generated 52 weeks. Latest total: {df['total_openings'].iloc[-1]}")
    return df


def compute_hiring_signal(df: pd.DataFrame) -> pd.Series:
    """
    Compute hiring signal (0-100).
    
    Production + quality roles are most predictive of near-term deliveries.
    Sales roles are forward-looking for future orders.
    """
    weights = {
        "production_assembly": 0.35,
        "quality_inspection": 0.15,
        "customer_support": 0.15,
        "engineering": 0.15,
        "pilots_flight_ops": 0.10,
        "sales_marketing": 0.10,
    }

    weighted_total = pd.Series(0.0, index=df.index)
    for col, w in weights.items():
        if col in df.columns:
            weighted_total += w * df[col]

    # Rolling z-score vs 3-month lookback
    rolling_mean = weighted_total.rolling(12, min_periods=4).mean()
    rolling_std = weighted_total.rolling(12, min_periods=4).std().replace(0, 1)
    z = (weighted_total - rolling_mean) / rolling_std

    signal = 50 + z * 18
    signal = signal.clip(0, 100)
    signal.name = "hiring_signal"
    return signal


if __name__ == "__main__":
    df = collect_job_postings()
    signal = compute_hiring_signal(df)
    print(df[["total_openings", "production_assembly", "engineering"]].tail(10))
    print("\nSignal:")
    print(signal.tail(10))
