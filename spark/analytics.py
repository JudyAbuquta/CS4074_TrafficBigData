"""
SQL Analytics — run after feature engineering.
Generates summary statistics for the report.
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import os

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-arm64"

spark = (SparkSession.builder
         .appName("TrafficAnalytics")
         .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

df = spark.read.parquet("hdfs://localhost:9000/traffic/features/")

print("\n── Severity distribution by zone ────────────────────")
df.groupBy("zone", "severity_label") \
  .count() \
  .orderBy("zone", "severity_label") \
  .show(40)

print("\n── Average risk score by hour bin ───────────────────")
df.groupBy("hour_bin") \
  .agg(
      F.round(F.avg("risk_score"), 1).alias("avg_risk"),
      F.count("*").alias("events")
  ).orderBy("avg_risk", ascending=False) \
  .show()

print("\n── Emergency vehicle incidents ───────────────────────")
df.filter(F.col("is_emergency") == 1) \
  .groupBy("vehicle_type", "severity_label") \
  .count() \
  .orderBy("vehicle_type", "count") \
  .show()

print("\n── Top 3 highest-risk intersections ─────────────────")
df.groupBy("intersection_id") \
  .agg(F.round(F.avg("risk_score"), 1).alias("avg_risk")) \
  .orderBy("avg_risk", ascending=False) \
  .show(3)

spark.stop()
