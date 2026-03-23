"""
Signal 6: Satellite / Production Facility Monitoring
------------------------------------------------------
Count aircraft visible on the ramp at Dassault's Bordeaux-Mérignac facility.

Fewer aircraft parked = deliveries are flowing out.
More aircraft parked = production bottleneck or weak demand.

In production: use Planet Labs or Maxar satellite imagery APIs
with computer vision (YOLO/SAM) to detect and count aircraft.
Typical revisit: 1-3 days with Planet's SkySat constellation.

Coordinates: Dassault production ramp at LFBD
  Lat: 44.8312, Lon: -0.7189 (approximate)
"""

import pandas as pd
import numpy as np
from datetime import datetime


def collect_satellite_data(use_live: bool = False) -> pd.DataFrame:
    """
    Collect weekly aircraft counts from satellite imagery analysis.
    
    In production:
    1. Planet Labs API → order SkySat imagery over LFBD
    2. Run aircraft detection model (pre-trained YOLO on DOTA dataset)
    3. Count objects classified as 'aircraft' within Dassault ramp polygon
    4. Distinguish Falcon (smaller, twin/tri-engine) from Rafale (delta wing)
    
    Typical ramp holds 8-20 Falcons in various stages of completion.
    """
    dates = pd.date_range(end=datetime(2026, 3, 23), periods=52, freq="W")
    n = len(dates)
    np.random.seed(44)

    # Falcon count on ramp: declining = good (deliveries outpacing production)
    # Factory holiday shutdown in Aug → brief accumulation
    falcon_ramp = 14 - 3 * np.linspace(0, 1, n) + np.random.normal(0, 1.5, n)
    falcon_ramp[18:22] += 4  # Summer shutdown accumulation (Aug)
    falcon_ramp[50:52] -= 2  # Year-end delivery push

    # Rafale count for reference (not used in Falcon signal)
    rafale_ramp = 6 + np.random.normal(0, 1, n)

    # Paint shop activity (painted = nearly ready for delivery)
    painted_ready = np.clip(2 + np.random.poisson(1.5, n), 0, 6)
    # More painted aircraft in Q4 = year-end delivery push
    painted_ready[40:52] += 1

    # Cloud cover flag (impacts data quality)
    cloud_pct = 30 + 20 * np.sin(np.linspace(0, 2 * np.pi, n)) + np.random.normal(0, 10, n)
    usable_image = (cloud_pct < 60).astype(int)

    df = pd.DataFrame(
        {
            "falcon_ramp_count": falcon_ramp.clip(3, 22).astype(int),
            "rafale_ramp_count": rafale_ramp.clip(2, 12).astype(int),
            "painted_ready_count": painted_ready.astype(int),
            "cloud_cover_pct": cloud_pct.clip(0, 100).round(0).astype(int),
            "image_usable": usable_image,
        },
        index=dates,
    )
    df.index.name = "date"
    print(f"[Satellite] Generated 52 weeks. Latest ramp count: {df['falcon_ramp_count'].iloc[-1]} Falcons")
    return df


def compute_satellite_signal(df: pd.DataFrame) -> pd.Series:
    """
    Compute satellite signal (0-100).
    
    INVERTED: fewer aircraft on ramp = deliveries flowing = bullish.
    Painted aircraft count adds to confidence.
    Only use weeks with usable imagery.
    """
    # Forward-fill for cloudy weeks
    usable = df[df["image_usable"] == 1]["falcon_ramp_count"]
    ramp_filled = df["falcon_ramp_count"].copy()
    ramp_filled[df["image_usable"] == 0] = np.nan
    ramp_filled = ramp_filled.ffill()

    # Ramp count: lower is better (inverted)
    ramp_z = -(ramp_filled - ramp_filled.rolling(12, min_periods=4).mean()) / \
              ramp_filled.rolling(12, min_periods=4).std().replace(0, 1)

    # Painted count: higher is better
    painted_z = (df["painted_ready_count"] - df["painted_ready_count"].rolling(12, min_periods=4).mean()) / \
                 df["painted_ready_count"].rolling(12, min_periods=4).std().replace(0, 1)

    composite = 0.70 * ramp_z + 0.30 * painted_z

    signal = 50 + composite * 15
    signal = signal.clip(0, 100)
    signal.name = "satellite_signal"
    return signal


if __name__ == "__main__":
    df = collect_satellite_data()
    signal = compute_satellite_signal(df)
    print(df.tail(10))
    print("\nSignal:")
    print(signal.tail(10))
