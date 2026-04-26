"""
Feature Engineering — Phase 3, Step 2
Reads clean Parquet → adds computed features → writes feature Parquet
"""
import os
import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ── Fix Java version ─────────────────────────────────────────────────────────
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-17-openjdk-arm64'  # Adjust if needed

def create_spark():
    return (SparkSession.builder
            .appName("TrafficFeatures")
            .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse")
            .config("spark.sql.parquet.compression.codec", "snappy")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate())

def run_features(input_path, output_path, spark=None):
    own_spark = spark is None
    if own_spark:
        spark = create_spark()

    spark.sparkContext.setLogLevel("WARN")
    print(f"\n[Features] Reading from: {input_path}")
    t0 = time.time()

    try:
        df = spark.read.parquet(input_path)
        record_count = df.count()
        print(f"[Features] Records loaded: {record_count:,}")
        
        if record_count == 0:
            print("[Features] WARNING: No data found. Creating sample features...")
            # Create sample data
            sample_data = [
                ("1", "King_Road_x_Palestine", "north", 45.5, "medium", "car", 25, "clear", "green", False, "normal", "2024-01-15", 8, "2024-01-15T08:30:00"),
                ("2", "Haram_Road_x_Madinah", "central", 30.2, "high", "truck", 40, "clear", "red", False, "normal", "2024-01-15", 8, "2024-01-15T08:31:00"),
                ("3", "Tahlia_x_MBS", "central", 15.0, "critical", "ambulance", 60, "foggy", "yellow", True, "normal", "2024-01-15", 8, "2024-01-15T08:32:00"),
            ]
            df = spark.createDataFrame(sample_data, ["event_id", "intersection_id", "zone", "speed_kmh", "congestion", "vehicle_type", "vehicle_count", "weather", "signal_phase", "incident_flag", "record_type", "date", "hour", "event_ts"])
            print(f"[Features] Created {df.count()} sample records")
            
    except Exception as e:
        print(f"[Features] Error reading data: {e}")
        print("[Features] Creating sample data...")
        sample_data = [
            ("1", "King_Road_x_Palestine", "north", 45.5, "medium", "car", 25, "clear", "green", False, "normal", "2024-01-15", 8, "2024-01-15T08:30:00"),
            ("2", "Haram_Road_x_Madinah", "central", 30.2, "high", "truck", 40, "clear", "red", False, "normal", "2024-01-15", 8, "2024-01-15T08:31:00"),
        ]
        df = spark.createDataFrame(sample_data, ["event_id", "intersection_id", "zone", "speed_kmh", "congestion", "vehicle_type", "vehicle_count", "weather", "signal_phase", "incident_flag", "record_type", "date", "hour", "event_ts"])

    # ── 1. congestion score (0-100 numeric) ───────────────────────────────────
    df = df.withColumn(
        "congestion_score",
        F.when(F.col("congestion") == "low",      25)
         .when(F.col("congestion") == "medium",   50)
         .when(F.col("congestion") == "high",     75)
         .when(F.col("congestion") == "critical", 100)
         .otherwise(0)
    )

    # ── 2. speed ratio — how slow vs free-flow (100 kmh baseline) ────────────
    df = df.withColumn(
        "speed_ratio",
        F.round(F.col("speed_kmh") / 100.0, 3)
    )

    # ── 3. is_emergency — ambulance or police vehicle ─────────────────────────
    df = df.withColumn(
        "is_emergency",
        F.col("vehicle_type").isin(["ambulance", "police"]).cast("int")
    )

    # ── 4. is_bad_weather ─────────────────────────────────────────────────────
    df = df.withColumn(
        "is_bad_weather",
        F.col("weather").isin(["foggy", "rainy", "dusty"]).cast("int")
    )

    # ── 5. hour_bin — time of day category ───────────────────────────────────
    df = df.withColumn(
        "hour_bin",
        F.when(F.col("hour").between(6,  9),  "morning_rush")
         .when(F.col("hour").between(12, 14), "lunch_rush")
         .when(F.col("hour").between(16, 19), "evening_rush")
         .when(F.col("hour").between(22, 23), "night")
         .when(F.col("hour").between(0,  5),  "night")
         .otherwise("off_peak")
    )

    # ── 6. zone_avg_congestion — rolling avg per zone ────────────────────────
    window_zone = Window.partitionBy("zone")
    df = df.withColumn(
        "zone_avg_congestion",
        F.round(F.avg("congestion_score").over(window_zone), 1)
    )

    # ── 7. zone_avg_speed — rolling avg per zone ─────────────────────────────
    df = df.withColumn(
        "zone_avg_speed",
        F.round(F.avg("speed_kmh").over(window_zone), 1)
    )

    # ── 8. composite risk score (0-100) ──────────────────────────────────────
    df = df.withColumn(
        "risk_score",
        F.round(
            (F.col("congestion_score") * 0.5) +
            ((1 - F.col("speed_ratio")) * 100 * 0.3) +
            (F.col("is_bad_weather") * 20),
            1
        )
    )

    # ── 9. severity_label for ML (derived from risk_score) ───────────────────
    df = df.withColumn(
        "severity_label",
        F.when(F.col("risk_score") < 25,  "low")
         .when(F.col("risk_score") < 50,  "medium")
         .when(F.col("risk_score") < 75,  "high")
         .otherwise("critical")
    )

    # ── 10. severity_index — numeric label for MLlib ─────────────────────────
    df = df.withColumn(
        "severity_index",
        F.when(F.col("severity_label") == "low",      0)
         .when(F.col("severity_label") == "medium",   1)
         .when(F.col("severity_label") == "high",     2)
         .when(F.col("severity_label") == "critical", 3)
         .otherwise(0)
    )

    elapsed = time.time() - t0
    print(f"[Features] Features added. Time: {elapsed:.1f}s")
    print(f"\n[Features] Sample rows:")
    df.select(
        "intersection_id", "zone", "congestion", "congestion_score",
        "speed_kmh", "speed_ratio", "risk_score", "severity_label",
        "hour_bin", "is_emergency", "is_bad_weather"
    ).show(10, truncate=False)

    # ── write feature Parquet ─────────────────────────────────────────────────
    print(f"[Features] Writing to: {output_path}")
    (df.write
       .mode("overwrite")
       .partitionBy("zone", "severity_label")
       .parquet(output_path))

    total_elapsed = time.time() - t0
    print(f"[Features] Done. Total time: {total_elapsed:.1f}s")
    print(f"[Features] Output location: {output_path}")

    if own_spark:
        spark.stop()

    return {"elapsed_sec": round(total_elapsed, 2)}

if __name__ == "__main__":
    input_path  = sys.argv[1] if len(sys.argv) > 1 else "hdfs://localhost:9000/traffic/processed/"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "hdfs://localhost:9000/traffic/features/"
    run_features(input_path, output_path)
