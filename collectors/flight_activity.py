"""
Signal 4: Flight Activity Around Dassault Facilities
-----------------------------------------------------
Tracks ADS-B transponder data for Falcon-type aircraft operating
near key Dassault facilities. More demo flights and delivery flights
= stronger sales pipeline.

Key airports monitored:
- LFBD (Bordeaux-Mérignac): Main production & flight test center
- KTEB (Teterboro, NJ): Dassault Falcon Jet US HQ, demo flights
- KMLB (Melbourne, FL): New Falcon service center (opened Oct 2025)
- LFMI (Istres): Military flight test center (Falcon 10X trials)

In production: use ADS-B Exchange API, FlightRadar24 API, or
OpenSky Network to track flights with Dassault-registered callsigns.
"""

import pandas as pd
import numpy as np
from datetime import datetime


MONITORED_AIRPORTS = {
    "LFBD": {"name": "Bordeaux-Mérignac", "role": "Production & test", "lat": 44.8283, "lon": -0.7156},
    "KTEB": {"name": "Teterboro NJ", "role": "US HQ & demos", "lat": 40.8501, "lon": -74.0608},
    "KMLB": {"name": "Melbourne FL", "role": "Service center", "lat": 28.1028, "lon": -80.6453},
    "LFMI": {"name": "Istres", "role": "Flight test", "lat": 43.5247, "lon": 4.9284},
}


def collect_flight_activity(use_live: bool = False) -> pd.DataFrame:
    """
    Collect weekly Falcon flight activity near Dassault facilities.
    
    In production, you would:
    1. Use ADS-B Exchange API (adsbexchange.com) — filter ICAO hex codes
       registered to Dassault or known Falcon demo aircraft
    2. Count unique flights within 25nm of each airport
    3. Classify: test flight (pattern work), demo (A→B→A same day),
       delivery (one-way to new operator base), ferry
    """
    dates = pd.date_range(end=datetime(2026, 3, 23), periods=52, freq="W")
    n = len(dates)
    np.random.seed(55)

    # Bordeaux: production flights increase with delivery ramp
    lfbd_test = 15 + 5 * np.linspace(0, 1, n) + np.random.normal(0, 3, n)
    lfbd_test[48:52] += 12  # Falcon 10X flight test campaign begins
    
    # Teterboro: demo flights = direct sales indicator
    kteb_demo = 8 + np.random.normal(0, 2, n)
    kteb_demo[10:14] += 4  # Pre-air show demo tour
    kteb_demo[28:32] += 5  # Pre-NBAA demo push
    kteb_demo[42:48] += 3  # Year-end sales push

    # Melbourne: new facility, ramping up
    kmlb = np.zeros(n)
    tail = n - 26
    kmlb[26:] = 3 + 4 * np.linspace(0, 1, tail) + np.random.normal(0, 1.5, tail)

    # Istres: military test (Falcon 10X approaching first flight)
    lfmi = 2 + np.random.normal(0, 1, n)
    lfmi[44:52] += 8  # Intensive 10X pre-first-flight ground/taxi tests

    # Delivery flights (one-way from LFBD to customer bases worldwide)
    deliveries = 0.7 + np.random.poisson(0.7, n)  # ~0-3 per week
    deliveries = (deliveries * (1 + 0.3 * np.linspace(0, 1, n))).astype(int)

    df = pd.DataFrame(
        {
            "lfbd_flights": lfbd_test.clip(3).astype(int),
            "kteb_demo_flights": kteb_demo.clip(1).astype(int),
            "kmlb_flights": kmlb.clip(0).astype(int),
            "lfmi_test_flights": lfmi.clip(0).astype(int),
            "delivery_flights": deliveries,
        },
        index=dates,
    )
    df["total_falcon_flights"] = df.sum(axis=1)
    df.index.name = "date"
    print(f"[Flight Activity] Generated 52 weeks. Latest weekly flights: {df['total_falcon_flights'].iloc[-1]}")
    return df


def compute_flight_signal(df: pd.DataFrame) -> pd.Series:
    """
    Compute flight activity signal (0-100).
    
    Demo flights (Teterboro) and delivery flights are most predictive
    of near-term revenue. Production test flights lead deliveries by 4-8 weeks.
    """
    weights = {
        "kteb_demo_flights": 0.30,   # Direct sales pipeline indicator
        "delivery_flights": 0.30,     # Actual handovers happening
        "lfbd_flights": 0.25,         # Production throughput
        "kmlb_flights": 0.10,         # Service activity (retention)
        "lfmi_test_flights": 0.05,    # R&D (long-term, less weight)
    }

    weighted = pd.Series(0.0, index=df.index)
    for col, w in weights.items():
        if col in df.columns:
            weighted += w * df[col]

    rolling_mean = weighted.rolling(12, min_periods=4).mean()
    rolling_std = weighted.rolling(12, min_periods=4).std().replace(0, 1)
    z = (weighted - rolling_mean) / rolling_std

    signal = 50 + z * 16
    signal = signal.clip(0, 100)
    signal.name = "flight_activity_signal"
    return signal


if __name__ == "__main__":
    df = collect_flight_activity()
    signal = compute_flight_signal(df)
    print(df.tail(10))
    print("\nSignal:")
    print(signal.tail(10))
