# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Staging (Silver layer)
# MAGIC Cleans and types the bronze data, builds a real `order_datetime`, and
# MAGIC derives the time-series fields (hour/day/week/weekend) downstream
# MAGIC analysis needs.
# MAGIC
# MAGIC This layer also fixes several real data-quality issues found in the
# MAGIC source CSV (see README for the full story):
# MAGIC - `Time_Orderd` / `Time_Order_picked` mix literal `"NaN"` text with
# MAGIC   Excel-style fractional-day numbers (e.g. `0.833333333` meaning 20:00)
# MAGIC   instead of `"HH:mm"` strings, and a handful of genuinely invalid times
# MAGIC   (e.g. `24:05:00`).
# MAGIC - `weather`, `traffic_density`, `city_type`, `order_type`, `vehicle_type`
# MAGIC   encode missing values as the literal string `"NaN"`, not a true null.
# MAGIC - `Delivery_person_Ratings` casts `"NaN"` text to floating-point NaN
# MAGIC   (not SQL NULL) -- and Spark treats NaN as greater than any other
# MAGIC   number in comparisons, so a naive `rating > 5` check silently passes
# MAGIC   every NaN rating as "valid". Also contains a real sentinel value of
# MAGIC   `6.0` (ratings are 1-5) that isn't a NaN artifact at all.

# COMMAND ----------

from pyspark.sql.functions import expr

def normalize_time_expr(col_name):
    return expr(f"""
        CASE
            WHEN {col_name} IS NULL OR {col_name} = 'NaN' THEN NULL
            WHEN {col_name} RLIKE '^([01][0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?$'
                THEN substring({col_name}, 1, 5)
            WHEN TRY_CAST({col_name} AS DOUBLE) IS NOT NULL
                 AND TRY_CAST({col_name} AS DOUBLE) BETWEEN 0 AND 1
            THEN date_format(
                     timestamp_seconds(CAST(TRY_CAST({col_name} AS DOUBLE) * 86400 AS BIGINT)),
                     'HH:mm'
                 )
            ELSE NULL
        END
    """)

bronze = spark.table("workspace.rawdata.zomato_orders")

normalized = (
    bronze
    .withColumn("time_ordered_clean", normalize_time_expr("Time_Orderd"))
    .withColumn("time_picked_clean", normalize_time_expr("Time_Order_picked"))
)
normalized.createOrReplaceTempView("normalized_orders")

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.staging")

staging_sql = """
CREATE OR REPLACE TABLE workspace.staging.orders AS
WITH typed AS (
    SELECT
        ID                                          AS order_id,
        Delivery_person_ID                          AS delivery_person_id,
        TRY_CAST(Delivery_person_Age AS INT)        AS delivery_person_age,
        CASE
            WHEN TRY_CAST(Delivery_person_Ratings AS DOUBLE) IS NULL THEN NULL
            WHEN isnan(TRY_CAST(Delivery_person_Ratings AS DOUBLE)) THEN NULL
            WHEN TRY_CAST(Delivery_person_Ratings AS DOUBLE) NOT BETWEEN 1 AND 5 THEN NULL
            ELSE TRY_CAST(Delivery_person_Ratings AS DOUBLE)
        END                                          AS delivery_person_rating,
        TO_DATE(Order_Date, 'dd-MM-yyyy')           AS order_date,
        TO_TIMESTAMP(time_ordered_clean, 'HH:mm')   AS time_ordered_raw,
        TO_TIMESTAMP(time_picked_clean, 'HH:mm')    AS time_picked_raw,
        NULLIF(NULLIF(TRIM(Weather_conditions), ''), 'NaN')   AS weather,
        NULLIF(NULLIF(TRIM(Road_traffic_density), ''), 'NaN') AS traffic_density,
        TRY_CAST(Vehicle_condition AS INT)          AS vehicle_condition,
        NULLIF(NULLIF(TRIM(Type_of_order), ''), 'NaN')        AS order_type,
        NULLIF(NULLIF(TRIM(Type_of_vehicle), ''), 'NaN')      AS vehicle_type,
        TRY_CAST(multiple_deliveries AS INT)        AS multiple_deliveries,
        CASE WHEN Festival = 'Yes' THEN TRUE
             WHEN Festival = 'No' THEN FALSE END    AS is_festival,
        NULLIF(NULLIF(TRIM(City), ''), 'NaN')                 AS city_type,
        TRY_CAST(Time_taken_min AS INT)              AS delivery_time_min
    FROM normalized_orders
),
with_datetime AS (
    SELECT *,
        CASE WHEN order_date IS NOT NULL AND time_ordered_raw IS NOT NULL
             THEN TO_TIMESTAMP(CONCAT(order_date, ' ', DATE_FORMAT(time_ordered_raw, 'HH:mm:ss')))
        END AS order_datetime,
        CASE WHEN order_date IS NOT NULL AND time_picked_raw IS NOT NULL
             THEN TO_TIMESTAMP(CONCAT(order_date, ' ', DATE_FORMAT(time_picked_raw, 'HH:mm:ss')))
        END AS pickup_datetime
    FROM typed
    WHERE order_date IS NOT NULL
)
SELECT *,
    YEAR(order_date)                         AS order_year,
    MONTH(order_date)                        AS order_month,
    WEEKOFYEAR(order_date)                   AS order_week,
    DAYOFWEEK(order_date)                    AS order_dow,
    DATE_FORMAT(order_date, 'EEEE')          AS order_day_name,
    HOUR(order_datetime)                     AS order_hour,
    (DAYOFWEEK(order_date) IN (1, 7))        AS is_weekend,
    CASE WHEN pickup_datetime IS NOT NULL AND order_datetime IS NOT NULL
         THEN (UNIX_TIMESTAMP(pickup_datetime) - UNIX_TIMESTAMP(order_datetime)) / 60
    END                                       AS pickup_wait_min
FROM with_datetime
"""
spark.sql(staging_sql)

raw_count = spark.table("workspace.rawdata.zomato_orders").count()
staged_count = spark.table("workspace.staging.orders").count()
drop_rate = (raw_count - staged_count) / raw_count if raw_count else 0

print(f"raw: {raw_count}, staged: {staged_count}, dropped: {raw_count - staged_count} ({drop_rate:.2%})")

# Data-quality gate: fail loudly rather than silently ship a badly
# degraded dataset to the marts layer.
MAX_ACCEPTABLE_DROP_RATE = 0.05
if drop_rate > MAX_ACCEPTABLE_DROP_RATE:
    raise ValueError(f"Staging drop rate {drop_rate:.2%} exceeds threshold {MAX_ACCEPTABLE_DROP_RATE:.2%}")
