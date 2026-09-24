# Zomato Delivery Operations — Data Engineering Pipeline & Time Series Analysis

A Databricks-based data engineering pipeline (bronze → silver → gold / medallion
architecture) built on the [Zomato Delivery Operations Analytics Dataset](
https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset)
(45,584 real delivery records), with an automated data quality gate and a
time-series analysis of daily order volume.

## Architecture

```
data/raw/Zomato Dataset.csv          <- source CSV, uploaded to a Unity Catalog volume
        │
        ▼
┌─────────────────┐
│  01_ingest       │  Loads CSV as-is, sanitizes Delta-invalid column names,
│  (bronze/raw)    │  adds ingestion metadata. workspace.rawdata.zomato_orders
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  02_staging      │  Types & cleans the data: parses dates/times (with fixes
│  (silver)        │  for several real source-data bugs — see below), derives
└─────────────────┘  order_datetime, hour/day/week/weekend fields. Fails the
        │             pipeline if drop rate exceeds 5%. workspace.staging.orders
        ▼
┌─────────────────┐
│  03_marts        │  6 analysis-ready aggregate tables: daily/hourly/weekly
│  (gold)          │  order volume, delivery time by city, delivery time by
└─────────────────┘  conditions, daily exogenous features. workspace.marts.*
        │
        ▼
┌─────────────────┐
│  04_quality_     │  8 automated assertions (schema, row counts, uniqueness,
│  checks           │  value ranges, categorical validity, null rates,
└─────────────────┘  referential integrity). Fails loudly with a named reason
                      rather than silently shipping bad data.
```

All four notebooks are wired into a single Databricks Workflow
(`zomato_de_pipeline`) as dependent tasks (`ingest → staging → marts →
quality_checks`) running on serverless compute.

`05_time_series_analysis` builds on top of the marts tables and is run
separately (analysis, not part of the scheduled pipeline).

## Real data quality issues found and fixed

This dataset looks clean at a glance but has several genuine defects that
the pipeline explicitly handles rather than ignores:

1. **Literal `"NaN"` text instead of true nulls.** `Time_Orderd`,
   `Weather_conditions`, `Road_traffic_density`, `City`, `Type_of_order`,
   and `Type_of_vehicle` all encode missing values as the string `"NaN"`,
   not an actual null. A naive `NULLIF(col, '')` misses this entirely —
   fixed with `NULLIF(NULLIF(TRIM(col), ''), 'NaN')`.

2. **Excel-style fractional-day time values.** Some `Time_Orderd` /
   `Time_Order_picked` values are fractions like `0.833333333` (meaning
   20:00) instead of `"HH:mm"` strings — recovered by converting the
   fraction to seconds-of-day rather than discarding the row.

3. **A genuinely invalid time value** (`24:05:00`, hour 24 doesn't exist)
   that broke a naive `CAST(... AS DOUBLE)` under Databricks' ANSI SQL
   mode — fixed with `TRY_CAST` and a stricter regex that only accepts
   valid hours (00–23).

4. **`TRY_CAST('NaN' AS DOUBLE)` returns floating-point NaN, not SQL
   NULL** — and Spark treats NaN as *greater than any other number* in
   comparisons, so a naive `rating > 5` check silently passed every NaN
   rating as "valid" (1961 rows). Fixed with an explicit `isnan()` check.

5. **A genuine sentinel value**: 53 `delivery_person_rating` values of
   exactly `6.0` on a 1–5 scale — not a NaN artifact, a real out-of-range
   value, nulled out explicitly.

All of these were caught by the automated data quality checks in
`04_quality_checks.py`, not discovered by manual inspection — the checks
did their job.

## Time series analysis findings

- **No strong long-term trend** — daily order volume holds fairly steady
  (~1000–1150/day) over the available window.
- **An 11-day gap** in the data (10 consecutive missing days + 1 isolated
  day) — the contiguous Mar 1 – Apr 6 block was used for analysis rather
  than force-bridging a large gap.
- **A genuine period-2 oscillation, not weekly seasonality.** The daily
  plot looks like a weekly pattern at first glance, but day-of-week
  averages are flat (~1000–1060 across all seven days) — confirmed via
  ACF, which shows a clean period-2 signature (lag-1 strongly negative,
  lag-2 strongly positive, decaying). This is more consistent with a
  data-generation artifact than a real "weekends are quieter" business
  pattern, and is reported as such rather than presented as a real trend.
- **Exogenous regressors (traffic/weather/festival) made the backtested
  forecast worse**, not better (MAE 23.4 vs. 19.1 for the series-only
  baseline). With only 30 training days, estimating extra regressor
  coefficients adds noise rather than signal — an honest result, not a
  hidden one.

## Tech stack

- Databricks Free Edition (Serverless compute, Unity Catalog)
- PySpark / Spark SQL, Delta Lake
- Python: pandas, statsmodels (seasonal decomposition, Holt-Winters,
  SARIMAX), matplotlib

## Reproducing this

1. Databricks Free Edition workspace (community.cloud.databricks.com or
   the newer Free Edition signup)
2. Upload `Zomato Dataset.csv` to a Unity Catalog volume at
   `/Volumes/workspace/rawdata/landing/`
3. Import the notebooks in `notebooks/` into your workspace
4. Run `01_ingest` → `02_staging` → `03_marts` → `04_quality_checks` in
   order (or wire them into a Databricks Job with those dependencies)
5. Run `05_time_series_analysis` for the forecasting/decomposition work
