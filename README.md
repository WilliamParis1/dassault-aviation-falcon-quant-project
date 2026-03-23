# Falcon Sales Predictor — Alternative Data Project

## Thesis
Predict whether Dassault Aviation's Falcon business jet deliveries will **beat or miss**
guidance ahead of official earnings releases, using alternative (non-financial) data signals.

## Signals tracked

| # | Signal | Source | Rationale |
|---|--------|--------|-----------|
| 1 | **Google Trends** — "Dassault Falcon", "Falcon 10X", "business jet for sale" | pytrends | Rising search interest → more prospective buyers |
| 2 | **Pre-owned inventory** — # of Falcons listed on resale sites | Controller.com, AvBuyer | Fewer listings → tighter market → new sales up |
| 3 | **Job postings** — Dassault Falcon Jet hiring intensity | LinkedIn, Indeed | More hires at service/production → ramp-up signal |
| 4 | **Flight activity** — Demo/test flights from LFBD (Bordeaux) & TEB (Teterboro) | ADS-B Exchange, FlightRadar24 | More demo flights → more active sales pipeline |
| 5 | **Macro demand proxy** — Corporate profits, UHNW wealth index, fuel prices | FRED, World Bank | Wealthy buyers + cheap fuel → jet demand |
| 6 | **Satellite / parking ramp** — Aircraft count at Bordeaux-Mérignac facility | Planet, Maxar | Fewer parked aircraft → deliveries accelerating |

## Architecture
```
collectors/          — One module per data source
analysis/            — Signal normalization, composite scoring, backtesting
output/              — Charts, reports, CSV exports
main.py              — Orchestrator: collect → score → visualize
```

## Usage
```bash
python main.py
```
