#!/usr/bin/env python3
"""
Falcon Alternative Data Pipeline
=================================
Main orchestrator that:
1. Collects data from all 6 alternative data sources
2. Computes individual signals (each normalized 0-100)
3. Builds the Falcon Sales Composite Indicator (FSCI)
4. Generates a multi-panel dashboard chart
5. Outputs a weekly report and quarterly predictions

Usage:
    python main.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from datetime import datetime

# Collectors
from collectors.google_trends import collect_google_trends, compute_trend_signal
from collectors.preowned_inventory import collect_preowned_inventory, compute_inventory_signal
from collectors.job_postings import collect_job_postings, compute_hiring_signal
from collectors.flight_activity import collect_flight_activity, compute_flight_signal
from collectors.macro_demand import collect_macro_data, compute_macro_signal
from collectors.satellite_ramp import collect_satellite_data, compute_satellite_signal

# Analysis
from analysis.composite import (
    build_composite_signal,
    compute_quarterly_prediction,
    generate_report,
    GUIDANCE_2026,
    KNOWN_DELIVERIES,
)


def create_dashboard(composite_df, raw_data, output_path):
    """Create a comprehensive multi-panel dashboard."""
    
    # ── Color palette (dark finance theme) ──
    BG = "#0a0e17"
    PANEL_BG = "#111827"
    GRID = "#1e293b"
    TEXT = "#e2e8f0"
    TEXT_DIM = "#94a3b8"
    ACCENT = "#3b82f6"
    GREEN = "#22c55e"
    RED = "#ef4444"
    ORANGE = "#f59e0b"
    PURPLE = "#a78bfa"
    CYAN = "#06b6d4"
    PINK = "#ec4899"

    SIGNAL_COLORS = {
        "google_trends_signal": CYAN,
        "inventory_signal": GREEN,
        "hiring_signal": ORANGE,
        "flight_activity_signal": PURPLE,
        "macro_signal": PINK,
        "satellite_signal": "#facc15",
    }

    fig = plt.figure(figsize=(24, 18), facecolor=BG)
    fig.suptitle(
        "DASSAULT FALCON — Alternative Data Composite Indicator",
        fontsize=22, fontweight="bold", color=TEXT, y=0.98,
        fontfamily="monospace",
    )
    fig.text(0.5, 0.955, f"Report Date: {datetime.now().strftime('%B %d, %Y')}  ·  Ticker: AM.PA (Euronext Paris)  ·  2026 Delivery Guidance: {GUIDANCE_2026} Falcons",
             ha="center", fontsize=11, color=TEXT_DIM, fontfamily="monospace")

    gs = gridspec.GridSpec(4, 3, hspace=0.35, wspace=0.25,
                           left=0.06, right=0.97, top=0.93, bottom=0.04)

    # ════════════════════════════════════════════════
    # Panel 1: FSCI Main Chart (top, full width)
    # ════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(PANEL_BG)
    
    dates = composite_df.index
    fsci = composite_df["FSCI"]
    
    # Fill zones
    ax1.axhspan(0, 30, alpha=0.08, color=RED)
    ax1.axhspan(30, 45, alpha=0.04, color=RED)
    ax1.axhspan(55, 70, alpha=0.04, color=GREEN)
    ax1.axhspan(70, 100, alpha=0.08, color=GREEN)
    ax1.axhline(50, color=TEXT_DIM, linewidth=0.5, linestyle="--", alpha=0.5)
    
    # Zone labels
    ax1.text(dates[1], 85, "STRONG BUY", fontsize=8, color=GREEN, alpha=0.6, fontfamily="monospace")
    ax1.text(dates[1], 20, "STRONG SELL", fontsize=8, color=RED, alpha=0.6, fontfamily="monospace")
    
    # FSCI line with gradient fill
    ax1.plot(dates, fsci, color=ACCENT, linewidth=2.5, zorder=5)
    ax1.fill_between(dates, 50, fsci, where=(fsci >= 50), color=GREEN, alpha=0.15)
    ax1.fill_between(dates, 50, fsci, where=(fsci < 50), color=RED, alpha=0.15)
    
    # 4-week moving average
    ma4 = fsci.rolling(4).mean()
    ax1.plot(dates, ma4, color=ORANGE, linewidth=1.2, linestyle="--", alpha=0.7, label="4-wk MA")
    
    # Current value annotation
    latest_fsci = fsci.iloc[-1]
    color = GREEN if latest_fsci > 55 else RED if latest_fsci < 45 else ORANGE
    ax1.annotate(
        f"  {latest_fsci:.1f}",
        xy=(dates[-1], latest_fsci), fontsize=14, fontweight="bold",
        color=color, fontfamily="monospace",
    )
    
    # Mark known earnings dates
    earnings_dates = [
        datetime(2025, 7, 22), datetime(2026, 3, 5),
    ]
    for ed in earnings_dates:
        if dates[0] <= ed <= dates[-1]:
            ax1.axvline(ed, color=ORANGE, linewidth=1, linestyle=":", alpha=0.5)
            ax1.text(ed, 95, " EARNINGS", fontsize=7, color=ORANGE, alpha=0.7,
                    rotation=0, fontfamily="monospace")

    ax1.set_ylim(0, 100)
    ax1.set_ylabel("FSCI Score (0-100)", color=TEXT_DIM, fontsize=10, fontfamily="monospace")
    ax1.set_title("Falcon Sales Composite Indicator (FSCI)", color=TEXT, fontsize=14,
                  fontweight="bold", loc="left", fontfamily="monospace")
    ax1.legend(loc="upper left", fontsize=8, facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT_DIM)
    ax1.tick_params(colors=TEXT_DIM, labelsize=8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax1.grid(True, color=GRID, alpha=0.3)
    for spine in ax1.spines.values():
        spine.set_color(GRID)

    # ════════════════════════════════════════════════
    # Panel 2: Individual Signals (stacked)
    # ════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.set_facecolor(PANEL_BG)
    ax2.axhline(50, color=TEXT_DIM, linewidth=0.5, linestyle="--", alpha=0.3)
    
    signal_names = {
        "google_trends_signal": "Google Trends",
        "inventory_signal": "Pre-Owned Inventory",
        "hiring_signal": "Hiring Intensity",
        "flight_activity_signal": "Flight Activity",
        "macro_signal": "Macro Demand",
        "satellite_signal": "Satellite/Ramp",
    }
    
    for col, label in signal_names.items():
        if col in composite_df.columns:
            ax2.plot(dates, composite_df[col], label=label,
                    color=SIGNAL_COLORS.get(col, TEXT_DIM), linewidth=1.3, alpha=0.85)
    
    ax2.set_ylim(15, 85)
    ax2.set_title("Individual Signal Breakdown", color=TEXT, fontsize=12,
                  fontweight="bold", loc="left", fontfamily="monospace")
    ax2.legend(loc="upper left", fontsize=7, ncol=3, facecolor=PANEL_BG,
              edgecolor=GRID, labelcolor=TEXT_DIM)
    ax2.tick_params(colors=TEXT_DIM, labelsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.grid(True, color=GRID, alpha=0.3)
    for spine in ax2.spines.values():
        spine.set_color(GRID)

    # ════════════════════════════════════════════════
    # Panel 3: Signal Heatmap / Current Readings
    # ════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.set_facecolor(PANEL_BG)
    ax3.set_xlim(0, 100)
    
    y_positions = list(range(len(signal_names)))
    latest_values = []
    labels = []
    colors = []
    
    for col, label in signal_names.items():
        val = composite_df[col].iloc[-1] if col in composite_df.columns else 50
        latest_values.append(val)
        labels.append(label)
        c = GREEN if val > 55 else RED if val < 45 else ORANGE
        colors.append(c)
    
    bars = ax3.barh(y_positions, latest_values, color=colors, alpha=0.7, height=0.6,
                    edgecolor=[c for c in colors], linewidth=0.5)
    ax3.axvline(50, color=TEXT_DIM, linewidth=1, linestyle="--", alpha=0.4)
    
    ax3.set_yticks(y_positions)
    ax3.set_yticklabels(labels, fontsize=8, color=TEXT, fontfamily="monospace")
    
    for i, (val, c) in enumerate(zip(latest_values, colors)):
        ax3.text(val + 1, i, f"{val:.0f}", va="center", fontsize=9,
                fontweight="bold", color=c, fontfamily="monospace")
    
    ax3.set_title("Current Signal Readings", color=TEXT, fontsize=12,
                  fontweight="bold", loc="left", fontfamily="monospace")
    ax3.tick_params(colors=TEXT_DIM, labelsize=8)
    ax3.grid(True, axis="x", color=GRID, alpha=0.3)
    for spine in ax3.spines.values():
        spine.set_color(GRID)

    # ════════════════════════════════════════════════
    # Panel 4: Pre-Owned Inventory Chart
    # ════════════════════════════════════════════════
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor(PANEL_BG)
    inv_data = raw_data["inventory"]
    ax4.fill_between(inv_data.index, inv_data["total_listings"], alpha=0.3, color=CYAN)
    ax4.plot(inv_data.index, inv_data["total_listings"], color=CYAN, linewidth=1.5)
    ax4.set_title("Pre-Owned Falcon Listings", color=TEXT, fontsize=11,
                  fontweight="bold", loc="left", fontfamily="monospace")
    ax4.set_ylabel("# Listed", color=TEXT_DIM, fontsize=9, fontfamily="monospace")
    ax4.tick_params(colors=TEXT_DIM, labelsize=7)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax4.grid(True, color=GRID, alpha=0.3)
    for spine in ax4.spines.values():
        spine.set_color(GRID)

    # ════════════════════════════════════════════════
    # Panel 5: Google Trends
    # ════════════════════════════════════════════════
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor(PANEL_BG)
    trends = raw_data["trends"]
    ax5.plot(trends.index, trends["Dassault Falcon"], color=ACCENT, linewidth=1.3, label="Dassault Falcon")
    ax5.plot(trends.index, trends["Falcon 10X"], color=PINK, linewidth=1.3, label="Falcon 10X")
    ax5.plot(trends.index, trends["business jet for sale"], color=GREEN, linewidth=1, alpha=0.7, label="Biz jet for sale")
    ax5.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT_DIM)
    ax5.set_title("Google Search Trends", color=TEXT, fontsize=11,
                  fontweight="bold", loc="left", fontfamily="monospace")
    ax5.tick_params(colors=TEXT_DIM, labelsize=7)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax5.grid(True, color=GRID, alpha=0.3)
    for spine in ax5.spines.values():
        spine.set_color(GRID)

    # ════════════════════════════════════════════════
    # Panel 6: Flight Activity
    # ════════════════════════════════════════════════
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor(PANEL_BG)
    flights = raw_data["flights"]
    ax6.stackplot(
        flights.index,
        flights["lfbd_flights"], flights["kteb_demo_flights"],
        flights["delivery_flights"], flights["kmlb_flights"],
        labels=["Bordeaux Prod.", "Teterboro Demos", "Deliveries", "Melbourne Svc"],
        colors=[ACCENT, PURPLE, GREEN, ORANGE], alpha=0.7,
    )
    ax6.legend(fontsize=7, loc="upper left", facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT_DIM)
    ax6.set_title("Weekly Falcon Flights", color=TEXT, fontsize=11,
                  fontweight="bold", loc="left", fontfamily="monospace")
    ax6.tick_params(colors=TEXT_DIM, labelsize=7)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax6.grid(True, color=GRID, alpha=0.3)
    for spine in ax6.spines.values():
        spine.set_color(GRID)

    # ════════════════════════════════════════════════
    # Panel 7: Historical Deliveries + Prediction
    # ════════════════════════════════════════════════
    ax7 = fig.add_subplot(gs[3, 0])
    ax7.set_facecolor(PANEL_BG)
    
    years = ["2023", "2024", "2025", "2026E"]
    totals = [34, 31, 37, None]
    
    # 2026 prediction from FSCI
    latest_fsci_val = composite_df["FSCI"].tail(8).mean()
    predicted_2026 = int(GUIDANCE_2026 * (1 + (latest_fsci_val - 50) / 50 * 0.25))
    totals[3] = predicted_2026
    
    bar_colors = [TEXT_DIM, TEXT_DIM, ACCENT, GREEN if predicted_2026 >= GUIDANCE_2026 else RED]
    bars = ax7.bar(years, totals, color=bar_colors, alpha=0.8, width=0.5, edgecolor=bar_colors)
    ax7.axhline(GUIDANCE_2026, color=ORANGE, linestyle="--", linewidth=1, alpha=0.7)
    ax7.text(3.3, GUIDANCE_2026, f" Guide: {GUIDANCE_2026}", fontsize=8, color=ORANGE,
            fontfamily="monospace", va="center")
    
    for bar, val in zip(bars, totals):
        ax7.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", fontsize=10, fontweight="bold", color=TEXT,
                fontfamily="monospace")
    
    ax7.set_title("Annual Falcon Deliveries", color=TEXT, fontsize=11,
                  fontweight="bold", loc="left", fontfamily="monospace")
    ax7.set_ylabel("Aircraft", color=TEXT_DIM, fontsize=9, fontfamily="monospace")
    ax7.tick_params(colors=TEXT_DIM, labelsize=8)
    ax7.grid(True, axis="y", color=GRID, alpha=0.3)
    for spine in ax7.spines.values():
        spine.set_color(GRID)

    # ════════════════════════════════════════════════
    # Panel 8: Satellite Ramp Count
    # ════════════════════════════════════════════════
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.set_facecolor(PANEL_BG)
    sat = raw_data["satellite"]
    ax8.bar(sat.index, sat["falcon_ramp_count"], width=5, color=PURPLE, alpha=0.6)
    ax8.plot(sat.index, sat["falcon_ramp_count"].rolling(4).mean(),
            color=PURPLE, linewidth=2, label="4-wk avg")
    # Mark cloudy weeks
    cloudy = sat[sat["image_usable"] == 0]
    if not cloudy.empty:
        ax8.scatter(cloudy.index, cloudy["falcon_ramp_count"], marker="x",
                   color=RED, s=30, alpha=0.5, label="Cloudy (interpolated)")
    ax8.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=GRID, labelcolor=TEXT_DIM)
    ax8.set_title("Falcon Ramp Count (Satellite)", color=TEXT, fontsize=11,
                  fontweight="bold", loc="left", fontfamily="monospace")
    ax8.set_ylabel("Aircraft on ramp", color=TEXT_DIM, fontsize=9, fontfamily="monospace")
    ax8.tick_params(colors=TEXT_DIM, labelsize=7)
    ax8.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax8.grid(True, color=GRID, alpha=0.3)
    for spine in ax8.spines.values():
        spine.set_color(GRID)

    # ════════════════════════════════════════════════
    # Panel 9: Signal Weights + Methodology
    # ════════════════════════════════════════════════
    ax9 = fig.add_subplot(gs[3, 2])
    ax9.set_facecolor(PANEL_BG)
    ax9.axis("off")
    
    methodology = (
        "METHODOLOGY\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Signal Weights:\n"
        "  Pre-Owned Inventory    25%\n"
        "  Flight Activity        20%\n"
        "  Google Trends          15%\n"
        "  Hiring Intensity       15%\n"
        "  Satellite/Ramp         15%\n"
        "  Macro Demand           10%\n\n"
        "Scoring: Each signal normalized\n"
        "to 0-100 via rolling z-score.\n\n"
        "  > 55 = BEAT guidance\n"
        "  45-55 = IN-LINE\n"
        "  < 45 = MISS guidance\n\n"
        "⚠ NOT FINANCIAL ADVICE\n"
        "Synthetic data for demonstration"
    )
    ax9.text(0.05, 0.95, methodology, transform=ax9.transAxes,
            fontsize=9, color=TEXT_DIM, fontfamily="monospace",
            verticalalignment="top", linespacing=1.4)
    for spine in ax9.spines.values():
        spine.set_color(GRID)

    plt.savefig(output_path, dpi=180, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"\n[Dashboard] Saved to {output_path}")


def main():
    print("=" * 60)
    print("  FALCON ALTERNATIVE DATA PIPELINE")
    print("  Running all collectors...")
    print("=" * 60)
    
    # ── Step 1: Collect all data ──
    trends_data = collect_google_trends(use_live=False)
    inventory_data = collect_preowned_inventory()
    jobs_data = collect_job_postings()
    flights_data = collect_flight_activity()
    macro_data = collect_macro_data()
    satellite_data = collect_satellite_data()
    
    # ── Step 2: Compute individual signals ──
    print("\n[Analysis] Computing signals...")
    signals = {
        "google_trends_signal": compute_trend_signal(trends_data),
        "inventory_signal": compute_inventory_signal(inventory_data),
        "hiring_signal": compute_hiring_signal(jobs_data),
        "flight_activity_signal": compute_flight_signal(flights_data),
        "macro_signal": compute_macro_signal(macro_data),
        "satellite_signal": compute_satellite_signal(satellite_data),
    }
    
    # ── Step 3: Build composite ──
    composite_df = build_composite_signal(signals)
    
    # ── Step 4: Generate report ──
    report = generate_report(composite_df)
    print(report)
    
    # ── Step 5: Quarterly predictions ──
    predictions = compute_quarterly_prediction(composite_df["FSCI"])
    print("\n[Predictions] Quarterly Falcon Delivery Estimates:")
    print(predictions.to_string(index=False))
    
    # ── Step 6: Create dashboard ──
    raw_data = {
        "trends": trends_data,
        "inventory": inventory_data,
        "jobs": jobs_data,
        "flights": flights_data,
        "macro": macro_data,
        "satellite": satellite_data,
    }
    
    output_path = os.path.join(os.path.dirname(__file__), "output", "falcon_dashboard.png")
    create_dashboard(composite_df, raw_data, output_path)
    
    # ── Step 7: Export data ──
    csv_path = os.path.join(os.path.dirname(__file__), "output", "fsci_weekly.csv")
    composite_df.to_csv(csv_path)
    print(f"[Export] Weekly FSCI data saved to {csv_path}")
    
    print("\n✅ Pipeline complete.")
    return composite_df, raw_data


if __name__ == "__main__":
    main()
