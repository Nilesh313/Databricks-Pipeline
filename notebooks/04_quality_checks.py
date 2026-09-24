# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Data Quality Checks
# MAGIC A battery of assertions against the staging and marts tables. Fails
# MAGIC loudly, with a specific named reason, rather than letting bad data
# MAGIC silently reach the marts layer or downstream analysis.
# MAGIC
# MAGIC When first run, this caught two real bugs in the staging logic:
# MAGIC literal `"NaN"` text not being nulled out for `weather` /
# MAGIC `traffic_density` / `city_type`, and a Spark-specific NaN-comparison
# MAGIC surprise in `delivery_person_rating` (see 02_staging.py for the fix).

# COMMAND ----------

failures = []

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(f"{name}: {detail}")

# ---------------------------------------------------------------
# 1. Schema check -- staging has the columns downstream code relies on
# ---------------------------------------------------------------
expected_cols = {
    "order_id", "order_date", "order_datetime", "pickup_datetime",
    "order_hour", "order_dow", "order_day_name", "is_weekend",
    "weather", "traffic_density", "city_type", "is_festival",
    "delivery_time_min", "pickup_wait_min",
}
actual_cols = set(spark.table("workspace.staging.orders").columns)
missing = expected_cols - actual_cols
check("staging.orders has all expected columns", len(missing) == 0, f"missing: {missing}" if missing else "")

# COMMAND ----------

# ---------------------------------------------------------------
# 2. Row-count sanity -- staging shouldn't have silently lost most of the data
# ---------------------------------------------------------------
raw_count = spark.table("workspace.rawdata.zomato_orders").count()
staged_count = spark.table("workspace.staging.orders").count()
drop_rate = (raw_count - staged_count) / raw_count if raw_count else 1
check("staging drop rate <= 5%", drop_rate <= 0.05, f"{drop_rate:.2%} dropped ({raw_count} -> {staged_count})")

# ---------------------------------------------------------------
# 3. Uniqueness -- order_id should be a primary key
# ---------------------------------------------------------------
dup_count = spark.sql("""
    SELECT COUNT(*) AS c FROM (
        SELECT order_id FROM workspace.staging.orders GROUP BY order_id HAVING COUNT(*) > 1
    )
""").collect()[0]["c"]
check("order_id is unique", dup_count == 0, f"{dup_count} duplicate order_id values")

# COMMAND ----------

# ---------------------------------------------------------------
# 4. Range checks -- values should be physically plausible
# ---------------------------------------------------------------
bad_delivery_time = spark.sql("""
    SELECT COUNT(*) AS c FROM workspace.staging.orders
    WHERE delivery_time_min IS NOT NULL AND (delivery_time_min <= 0 OR delivery_time_min > 120)
""").collect()[0]["c"]
check("delivery_time_min in plausible range (0, 120]", bad_delivery_time == 0, f"{bad_delivery_time} rows out of range")

bad_rating = spark.sql("""
    SELECT COUNT(*) AS c FROM workspace.staging.orders
    WHERE delivery_person_rating IS NOT NULL AND (delivery_person_rating < 1 OR delivery_person_rating > 5)
""").collect()[0]["c"]
check("delivery_person_rating in [1, 5]", bad_rating == 0, f"{bad_rating} rows out of range")

# COMMAND ----------

# ---------------------------------------------------------------
# 5. Categorical validity -- catches upstream schema drift / typos
# ---------------------------------------------------------------
known_weather = {"Fog", "Stormy", "Sandstorms", "Windy", "Cloudy", "Sunny"}
known_traffic = {"Jam", "High", "Medium", "Low"}
known_city = {"Metropolitian", "Urban", "Semi-Urban"}

def check_categorical(col, known_values):
    bad = spark.sql(f"""
        SELECT COUNT(*) AS c FROM workspace.staging.orders
        WHERE {col} IS NOT NULL AND {col} NOT IN ({','.join(repr(v) for v in known_values)})
    """).collect()[0]["c"]
    check(f"{col} only has known values", bad == 0, f"{bad} unexpected values")

check_categorical("weather", known_weather)
check_categorical("traffic_density", known_traffic)
check_categorical("city_type", known_city)

# COMMAND ----------

# ---------------------------------------------------------------
# 6. Null-rate gates on fields the marts layer depends on
# ---------------------------------------------------------------
null_rates = spark.sql("""
    SELECT
        SUM(CASE WHEN order_datetime IS NULL THEN 1 ELSE 0 END) / COUNT(*) AS order_datetime,
        SUM(CASE WHEN city_type IS NULL THEN 1 ELSE 0 END) / COUNT(*)      AS city_type
    FROM workspace.staging.orders
""").collect()[0]
check("order_datetime null rate <= 15%", null_rates["order_datetime"] <= 0.15, f"{null_rates['order_datetime']:.2%}")
check("city_type null rate <= 5%", null_rates["city_type"] <= 0.05, f"{null_rates['city_type']:.2%}")

# COMMAND ----------

# ---------------------------------------------------------------
# 7. Referential integrity -- marts totals should reconcile with staging
# ---------------------------------------------------------------
marts_total = spark.table("workspace.marts.daily_order_volume").agg({"order_count": "sum"}).collect()[0][0]
staging_with_date = spark.sql("SELECT COUNT(*) AS c FROM workspace.staging.orders WHERE order_date IS NOT NULL").collect()[0]["c"]
check("marts.daily_order_volume total reconciles with staging", marts_total == staging_with_date,
      f"marts sum={marts_total}, staging={staging_with_date}")

# COMMAND ----------

# ---------------------------------------------------------------
# Final gate
# ---------------------------------------------------------------
print(f"\n{len(failures)} check(s) failed")
if failures:
    raise ValueError("Data quality checks failed:\n" + "\n".join(failures))
print("All data quality checks passed.")
