"""
Scalability Benchmark — Phase 3, Step 4
Runs ETL + Feature Engineering at 1x, 10x, 100x scale.
Records time, throughput, and data quality metrics.
Results saved to data/benchmark_results.json
"""
import json
import time
import random
import uuid
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType,
    FloatType, IntegerType, BooleanType
)

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-arm64"
sys.path.insert(0, str(Path(__file__).parent))
from etl_pipeline       import run_etl
from feature_engineering import run_features

INTERSECTIONS  = [
    "King_Road_x_Palestine","Haram_Road_x_Madinah","Tahlia_x_MBS",
    "Corniche_x_Andalus","Prince_Sultan_x_Sari","Al_Hamra_x_Falastin",
    "Airport_Road_x_Asfan","Al_Waziriyah_x_Sitteen"
]
ZONES = {i: z for i, z in zip(INTERSECTIONS,
    ["north","central","central","west","south","north","east","south"])}
VEHICLE_TYPES = ["car","truck","bus","motorcycle","ambulance","police"]
WEATHER_CONDS = ["clear","foggy","rainy","dusty","hot"]
CONGESTION    = ["low","medium","high","critical"]

def random_ts(days_back=30):
    base = datetime.utcnow() - timedelta(days=random.randint(0, days_back))
    return base.replace(
        hour=random.randint(0,23),
        minute=random.randint(0,59)
    ).isoformat()

def generate_batch(n, adversarial_pct=0.1):
    """Generate n synthetic sensor records, adversarial_pct fraction are bad."""
    records = []
    for _ in range(n):
        inter = random.choice(INTERSECTIONS)
        r = {
            "event_id":        str(uuid.uuid4()),
            "timestamp":       random_ts(),
            "intersection_id": inter,
            "zone":            ZONES[inter],
            "speed_kmh":       round(random.uniform(5, 120), 1),
            "congestion":      random.choice(CONGESTION),
            "vehicle_type":    random.choice(VEHICLE_TYPES),
            "vehicle_count":   random.randint(1, 80),
            "weather":         random.choice(WEATHER_CONDS),
            "signal_phase":    random.choice(["green","red","yellow"]),
            "incident_flag":   random.random() < 0.05,
            "record_type":     "normal"
        }
        # inject adversarial faults
        if random.random() < adversarial_pct:
            fault = random.choice(["null_speed","null_congestion","wrong_type","out_of_range"])
            if fault == "null_speed":
                r["speed_kmh"] = None
            elif fault == "null_congestion":
                r["congestion"] = None
            elif fault == "wrong_type":
                r["vehicle_count"] = "many"
            elif fault == "out_of_range":
                r["speed_kmh"] = 999.9
            r["record_type"] = "adversarial"
        records.append(r)
    return records

def run_benchmark():
    spark = (SparkSession.builder
             .appName("TrafficBenchmark")
             .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse")
             .config("spark.sql.shuffle.partitions", "8")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    scales = [
        {"name": "1x",   "records": 10_000},
        {"name": "10x",  "records": 100_000},
        {"name": "100x", "records": 1_000_000},
    ]

    results = []

    for scale in scales:
        n    = scale["records"]
        name = scale["name"]
        print(f"\n{'='*55}")
        print(f"[Benchmark] Running scale: {name}  ({n:,} records)")
        print(f"{'='*55}")

        # ── generate data ─────────────────────────────────────────────────────
        print(f"[Benchmark] Generating {n:,} records...")
        t_gen = time.time()
        records = generate_batch(n, adversarial_pct=0.1)

        # write to local JSON then push to HDFS
        local_path = f"/tmp/benchmark_{name}.json"
        with open(local_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        hdfs_raw  = f"hdfs://localhost:9000/traffic/benchmark/{name}/raw/"
        hdfs_proc = f"hdfs://localhost:9000/traffic/benchmark/{name}/processed/"
        hdfs_feat = f"hdfs://localhost:9000/traffic/benchmark/{name}/features/"

        os.system(f"hdfs dfs -mkdir -p {hdfs_raw}")
        os.system(f"hdfs dfs -put -f {local_path} {hdfs_raw}")
        os.remove(local_path)
        gen_time = time.time() - t_gen
        print(f"[Benchmark] Data ready in {gen_time:.1f}s")

        # ── run ETL ───────────────────────────────────────────────────────────
        t_etl  = time.time()
        etl_stats = run_etl(hdfs_raw, hdfs_proc, spark=spark)
        etl_time  = time.time() - t_etl

        # ── run features ──────────────────────────────────────────────────────
        t_feat  = time.time()
        run_features(hdfs_proc, hdfs_feat, spark=spark)
        feat_time = time.time() - t_feat

        total_time = etl_time + feat_time
        throughput = n / total_time

        result = {
            "scale":           name,
            "records":         n,
            "etl_time_sec":    round(etl_time, 2),
            "feat_time_sec":   round(feat_time, 2),
            "total_time_sec":  round(total_time, 2),
            "throughput_rps":  round(throughput, 0),
            "clean_records":   etl_stats["total_clean"],
            "dropped_records": etl_stats["dropped"],
            "drop_rate_pct":   round(etl_stats["dropped"] / n * 100, 1),
            "timestamp":       datetime.utcnow().isoformat()
        }
        results.append(result)

        print(f"\n[Benchmark] ── {name} Results ──────────────────")
        print(f"  ETL time:        {etl_time:.1f}s")
        print(f"  Feature time:    {feat_time:.1f}s")
        print(f"  Total time:      {total_time:.1f}s")
        print(f"  Throughput:      {throughput:,.0f} records/sec")
        print(f"  Clean records:   {etl_stats['total_clean']:,}")
        print(f"  Drop rate:       {result['drop_rate_pct']}%")

    spark.stop()

    # ── save results ──────────────────────────────────────────────────────────
    out_path = Path("data/benchmark_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*55}")
    print(f"[Benchmark] All scales complete. Results saved to {out_path}")
    print(f"{'='*55}\n")
    print(f"{'Scale':<8} {'Records':>10} {'ETL(s)':>8} {'Feat(s)':>8} {'Total(s)':>10} {'rec/s':>10} {'Drop%':>7}")
    print("-" * 65)
    for r in results:
        print(f"{r['scale']:<8} {r['records']:>10,} {r['etl_time_sec']:>8.1f} "
              f"{r['feat_time_sec']:>8.1f} {r['total_time_sec']:>10.1f} "
              f"{r['throughput_rps']:>10,.0f} {r['drop_rate_pct']:>6.1f}%")

if __name__ == "__main__":
    run_benchmark()
