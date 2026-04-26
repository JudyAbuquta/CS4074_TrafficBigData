"""
Spark Bridge — Phase 4, Step 3
Loads the trained MLlib model and batch-scores records from HDFS.
Outputs enriched records that the 3 agents consume.
"""
import json
import time
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-arm64"
MODEL_PATH    = "hdfs://localhost:9000/traffic/models/severity_classifier"
FEATURES_PATH = "hdfs://localhost:9000/traffic/features/"

def create_spark():
    return (SparkSession.builder
            .appName("SparkBridge")
            .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse")
            .getOrCreate())

def load_and_score(limit=50):
    """
    Load the saved MLlib model.
    Score a batch of traffic feature records.
    Return enriched dicts for the agents.
    """
    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    print("[Bridge] Loading saved MLlib model...")
    model = PipelineModel.load(MODEL_PATH)

    print("[Bridge] Loading feature records from HDFS...")
    df = spark.read.parquet(FEATURES_PATH).limit(limit)

    # The model expects a 'text' column — synthesize it from feature columns
    df = df.withColumn(
        "text",
        F.concat_ws(" ",
            F.col("intersection_id"),
            F.col("congestion"),
            F.col("vehicle_type"),
            F.col("weather"),
            F.col("hour_bin"),
            F.when(F.col("is_emergency") == 1, F.lit("emergency vehicle"))
             .otherwise(F.lit("normal vehicle"))
        )
    )

    print("[Bridge] Running distributed inference...")
    t0   = time.time()
    preds = model.transform(df)
    elapsed = time.time() - t0
    print(f"[Bridge] Inference done in {elapsed:.2f}s for {limit} records")

    # select columns agents need
    enriched = preds.select(
        "event_id", "intersection_id", "zone",
        "congestion", "congestion_score", "risk_score",
        "vehicle_type", "is_emergency",
        "weather", "is_bad_weather",
        "speed_kmh", "vehicle_count",
        "hour_bin", "severity_label",
        "predicted_severity" if "predicted_severity" in preds.columns
        else "severity_label",
    ).collect()

    spark.stop()

    return [row.asDict() for row in enriched]

if __name__ == "__main__":
    records = load_and_score(limit=10)
    print(f"\n[Bridge] Sample enriched record:")
    print(json.dumps(records[0], indent=2, default=str))

# Save output for the pipeline to consume
out_path = Path("data/spark_predictions.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(records, f, indent=2, default=str)
print(f"[Bridge] Predictions saved to {out_path}")
