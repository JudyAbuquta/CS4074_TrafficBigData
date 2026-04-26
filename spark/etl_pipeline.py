import os
import sys
import time

os.environ["JAVA_HOME"]  = "/usr/lib/jvm/java-17-openjdk-arm64"

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType, IntegerType, BooleanType
)

# ── schema ────────────────────────────────────────────────────────────────────
RAW_SCHEMA = StructType([
    StructField("event_id",        StringType(),  True),
    StructField("timestamp",       StringType(),  True),
    StructField("intersection_id", StringType(),  True),
    StructField("zone",            StringType(),  True),
    StructField("speed_kmh",       FloatType(),   True),
    StructField("congestion",      StringType(),  True),
    StructField("vehicle_type",    StringType(),  True),
    StructField("vehicle_count",   IntegerType(), True),
    StructField("weather",         StringType(),  True),
    StructField("signal_phase",    StringType(),  True),
    StructField("incident_flag",   BooleanType(), True),
    StructField("record_type",     StringType(),  True),
])

VALID_CONGESTION   = ["low", "medium", "high", "critical"]
VALID_VEHICLE_TYPE = ["car", "truck", "bus", "motorcycle", "ambulance", "police"]
VALID_WEATHER      = ["clear", "foggy", "rainy", "dusty", "hot"]

def create_spark(app_name="TrafficETL"):
    return (SparkSession.builder
            .appName(app_name)
            .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse")
            .config("spark.sql.parquet.compression.codec", "snappy")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.driver.extraJavaOptions",
                    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
                    "--add-opens=java.base/java.nio=ALL-UNNAMED "
                    "--add-opens=java.base/java.lang=ALL-UNNAMED "
                    "--add-opens=java.base/java.util=ALL-UNNAMED")
            .config("spark.executor.extraJavaOptions",
                    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
                    "--add-opens=java.base/java.nio=ALL-UNNAMED "
                    "--add-opens=java.base/java.lang=ALL-UNNAMED "
                    "--add-opens=java.base/java.util=ALL-UNNAMED")
            .getOrCreate())

def run_etl(input_path, output_path, spark=None):
    own_spark = spark is None
    if own_spark:
        spark = create_spark()

    spark.sparkContext.setLogLevel("WARN")
    print(f"\n[ETL] Reading from: {input_path}")
    t0 = time.time()

    # ── 1. read ───────────────────────────────────────────────────────────────
    df_raw    = spark.read.schema(RAW_SCHEMA).json(input_path)
    total_raw = df_raw.count()
    print(f"[ETL] Raw records loaded: {total_raw:,}")

    # ── 2. remove fully null rows ─────────────────────────────────────────────
    df = df_raw.dropna(
        how="all",
        subset=["event_id", "timestamp", "intersection_id"]
    )

    # ── 3. fix nulls ──────────────────────────────────────────────────────────
    df = df.fillna({
        "speed_kmh":     30.0,
        "vehicle_count": 1,
        "congestion":    "low",
        "vehicle_type":  "car",
        "weather":       "clear",
        "signal_phase":  "unknown",
        "incident_flag": False,
        "record_type":   "normal",
    })

    # ── 4. fix wrong types ────────────────────────────────────────────────────
    df = df.withColumn(
        "speed_kmh",
        F.when(F.col("speed_kmh").cast("float").isNotNull(),
               F.col("speed_kmh").cast("float")).otherwise(30.0)
    ).withColumn(
        "vehicle_count",
        F.when(F.col("vehicle_count").cast("int").isNotNull(),
               F.col("vehicle_count").cast("int")).otherwise(1)
    )

    # ── 5. filter out-of-range values ─────────────────────────────────────────
    df = df.filter(
        (F.col("speed_kmh").between(0, 200)) &
        (F.col("vehicle_count").between(0, 500))
    )

    # ── 6. remove duplicates ──────────────────────────────────────────────────
    df = df.dropDuplicates(["event_id"])

    # ── 7. validate categoricals ──────────────────────────────────────────────
    df = df.filter(F.col("congestion").isin(VALID_CONGESTION))
    df = df.filter(F.col("vehicle_type").isin(VALID_VEHICLE_TYPE))
    df = df.filter(F.col("weather").isin(VALID_WEATHER))

    # ── 8. parse timestamp ────────────────────────────────────────────────────
    df = df.withColumn(
        "event_ts", F.to_timestamp("timestamp")
    ).withColumn(
        "date", F.to_date("event_ts")
    ).withColumn(
        "hour", F.hour("event_ts")
    ).drop("timestamp")

    total_clean = df.count()
    dropped     = total_raw - total_clean
    elapsed     = time.time() - t0

    print(f"[ETL] Clean records:   {total_clean:,}")
    print(f"[ETL] Dropped records: {dropped:,}  ({dropped/max(total_raw,1)*100:.1f}%)")
    print(f"[ETL] Time so far:     {elapsed:.1f}s")

    # ── 9. write partitioned Parquet ──────────────────────────────────────────
    print(f"[ETL] Writing Parquet to: {output_path}")
    (df.write
       .mode("append")
       .partitionBy("zone", "date")
       .parquet(output_path))

    total_elapsed = time.time() - t0
    print(f"[ETL] Done. Total time: {total_elapsed:.1f}s")

    if own_spark:
        spark.stop()

    return {
        "total_raw":   total_raw,
        "total_clean": total_clean,
        "dropped":     dropped,
        "elapsed_sec": round(total_elapsed, 2)
    }

if __name__ == "__main__":
    input_path  = sys.argv[1] if len(sys.argv) > 1 else "hdfs://localhost:9000/traffic/raw/processed/"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "hdfs://localhost:9000/traffic/processed/"
    run_etl(input_path, output_path)
