# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Marts (Gold layer)
# MAGIC Analysis-ready, pre-aggregated tables: daily/hourly/weekly order volume,
# MAGIC delivery time cut by city and by operating conditions, and a daily
# MAGIC exogenous-features table (traffic/weather/festival) used later for
# MAGIC forecasting.

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.marts")

# 1. Daily order volume -- the core time series for demand forecasting
spark.sql("""
CREATE OR REPLACE TABLE workspace.marts.daily_order_volume AS
SELECT
    order_date,
    order_day_name,
    is_weekend,
    MAX(CASE WHEN is_festival THEN 1 ELSE 0 END) = 1  AS is_festival_day,
    COUNT(*)                                           AS order_count,
    ROUND(AVG(delivery_time_min), 2)                    AS avg_delivery_time_min,
    ROUND(AVG(pickup_wait_min), 2)                       AS avg_pickup_wait_min
FROM workspace.staging.orders
WHERE order_date IS NOT NULL
GROUP BY order_date, order_day_name, is_weekend
ORDER BY order_date
""")

# COMMAND ----------

# 2. Hourly order volume -- intraday pattern (peak lunch/dinner etc)
spark.sql("""
CREATE OR REPLACE TABLE workspace.marts.hourly_order_volume AS
SELECT
    order_date,
    order_hour,
    COUNT(*)                            AS order_count,
    ROUND(AVG(delivery_time_min), 2)     AS avg_delivery_time_min
FROM workspace.staging.orders
WHERE order_hour IS NOT NULL
GROUP BY order_date, order_hour
ORDER BY order_date, order_hour
""")

# COMMAND ----------

# 3. Weekly rollup -- smoother trend line for seasonality checks
spark.sql("""
CREATE OR REPLACE TABLE workspace.marts.weekly_order_volume AS
SELECT
    order_year,
    order_week,
    MIN(order_date)                     AS week_start,
    COUNT(*)                            AS order_count,
    ROUND(AVG(delivery_time_min), 2)     AS avg_delivery_time_min
FROM workspace.staging.orders
GROUP BY order_year, order_week
ORDER BY order_year, order_week
""")

# COMMAND ----------

# 4. Delivery time by city type over time
spark.sql("""
CREATE OR REPLACE TABLE workspace.marts.delivery_time_by_city_daily AS
SELECT
    order_date,
    city_type,
    COUNT(*)                            AS order_count,
    ROUND(AVG(delivery_time_min), 2)     AS avg_delivery_time_min
FROM workspace.staging.orders
WHERE city_type IS NOT NULL
GROUP BY order_date, city_type
ORDER BY order_date, city_type
""")

# COMMAND ----------

# 5. Operating-condition breakdown (weather / traffic / festival)
spark.sql("""
CREATE OR REPLACE TABLE workspace.marts.delivery_time_by_conditions AS
SELECT
    weather,
    traffic_density,
    is_festival,
    COUNT(*)                            AS order_count,
    ROUND(AVG(delivery_time_min), 2)     AS avg_delivery_time_min,
    ROUND(STDDEV(delivery_time_min), 2)   AS stddev_delivery_time_min
FROM workspace.staging.orders
GROUP BY weather, traffic_density, is_festival
ORDER BY order_count DESC
""")

# COMMAND ----------

# 6. Daily exogenous features -- used as regressors in the forecasting
#    notebook (05_time_series_analysis)
spark.sql("""
CREATE OR REPLACE TABLE workspace.marts.daily_conditions AS
SELECT
    order_date,
    ROUND(AVG(CASE WHEN traffic_density = 'Jam' THEN 1.0 ELSE 0.0 END), 3)   AS pct_traffic_jam,
    ROUND(AVG(CASE WHEN traffic_density = 'High' THEN 1.0 ELSE 0.0 END), 3)  AS pct_traffic_high,
    ROUND(AVG(CASE WHEN weather IN ('Stormy', 'Fog', 'Sandstorms') THEN 1.0 ELSE 0.0 END), 3) AS pct_bad_weather,
    MAX(CASE WHEN is_festival THEN 1 ELSE 0 END)                             AS is_festival_day,
    ROUND(AVG(multiple_deliveries), 3)                                       AS avg_multiple_deliveries
FROM workspace.staging.orders
WHERE order_date IS NOT NULL
GROUP BY order_date
ORDER BY order_date
""")

# COMMAND ----------

for tbl in ["daily_order_volume", "hourly_order_volume", "weekly_order_volume",
            "delivery_time_by_city_daily", "delivery_time_by_conditions", "daily_conditions"]:
    n = spark.table(f"workspace.marts.{tbl}").count()
    print(f"{tbl}: {n} rows")
