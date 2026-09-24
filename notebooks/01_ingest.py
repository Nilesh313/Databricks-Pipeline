# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Ingest (Bronze layer)
# MAGIC Reads the raw Zomato Delivery Operations CSV from a Unity Catalog volume
# MAGIC and lands it, untouched, as a Delta table. This is the pipeline's
# MAGIC "landing zone" -- no cleaning or type casting happens here, so the
# MAGIC bronze table is always a faithful, re-derivable copy of the source.

# COMMAND ----------

import re
from pyspark.sql.functions import current_timestamp, lit

raw_path = "/Volumes/workspace/rawdata/landing/Zomato Dataset.csv"

df_raw = spark.read.csv(raw_path, header=True, inferSchema=False)

# COMMAND ----------

# Delta table column names can't contain spaces, commas, semicolons,
# braces, parentheses, newlines, tabs or '=' -- the source CSV has a
# "Time_taken (min)" column that violates this, so sanitize all column
# names before writing.
def clean_col(c):
    return re.sub(r"[ ,;{}()\n\t=]+", "_", c).strip("_")

renamed_df = df_raw.toDF(*[clean_col(c) for c in df_raw.columns])

bronze_df = (
    renamed_df
    .withColumn("_source_file", lit("Zomato Dataset.csv"))
    .withColumn("_ingested_at", current_timestamp())
)

spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.rawdata")
bronze_df.write.format("delta").mode("overwrite").saveAsTable("workspace.rawdata.zomato_orders")

row_count = spark.table("workspace.rawdata.zomato_orders").count()
print(f"Ingested {row_count} rows into workspace.rawdata.zomato_orders")

if row_count == 0:
    raise ValueError("Ingestion produced zero rows -- aborting.")
