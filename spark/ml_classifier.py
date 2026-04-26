"""
Spark MLlib Classifier — Phase 4, Step 1
Ports TF-IDF + LogReg + SVM to distributed Spark MLlib.
Trains on incident_cases.json from HDFS.
Saves model to HDFS for agents to load.
"""
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-arm64"

import json
import time
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, HashingTF, IDF,
    StringIndexer, IndexToString
)
from pyspark.ml.classification import (
    LogisticRegression, LinearSVC, OneVsRest
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# ── schema ─────────────────────────────────────────────────────────────────────
INCIDENT_SCHEMA = StructType([
    StructField("id",            StringType(), True),
    StructField("timestamp",     StringType(), True),
    StructField("road",          StringType(), True),
    StructField("incident_type", StringType(), True),
    StructField("vehicle_type",  StringType(), True),
    StructField("weather",       StringType(), True),
    StructField("severity",      StringType(), True),
    StructField("text",          StringType(), True),
])

def create_spark():
    return (SparkSession.builder
            .appName("TrafficMLClassifier")
            .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate())

def build_pipeline(classifier="logreg"):
    """Build full MLlib Pipeline: tokenize → remove stopwords → TF → IDF → classify."""

    tokenizer = Tokenizer(inputCol="text", outputCol="words")

    remover = StopWordsRemover(inputCol="words", outputCol="filtered")

    hashingTF = HashingTF(
        inputCol="filtered", outputCol="raw_features", numFeatures=10000
    )

    idf = IDF(inputCol="raw_features", outputCol="features", minDocFreq=2)

    label_indexer = StringIndexer(
        inputCol="severity", outputCol="label", handleInvalid="keep"
    )

    if classifier == "logreg":
        clf = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=100,
            regParam=0.01,
            elasticNetParam=0.0,
            family="multinomial"
        )
    else:  # svm — wrap LinearSVC in OneVsRest for multiclass
        svc = LinearSVC(
            featuresCol="features",
            labelCol="label",
            maxIter=100,
            regParam=0.01
        )
        clf = OneVsRest(
            classifier=svc,
            featuresCol="features",
            labelCol="label"
        )

    return Pipeline(stages=[
        tokenizer, remover, hashingTF, idf, label_indexer, clf
    ]), label_indexer

def evaluate(predictions, label="test"):
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    acc = evaluator_acc.evaluate(predictions)
    f1  = evaluator_f1.evaluate(predictions)
    print(f"[ML] {label} — Accuracy: {acc:.4f}  F1: {f1:.4f}")
    return {"accuracy": round(acc, 4), "f1": round(f1, 4)}

def train_and_evaluate(spark, df_train, df_test, classifier="logreg", label="clean"):
    print(f"\n[ML] Training {classifier.upper()} on {label} data...")
    t0 = time.time()

    pipeline, _ = build_pipeline(classifier)

    model    = pipeline.fit(df_train)
    preds    = model.transform(df_test)
    metrics  = evaluate(preds, label=f"{classifier}/{label}")
    elapsed  = time.time() - t0

    print(f"[ML] Training time: {elapsed:.1f}s")
    metrics["elapsed_sec"] = round(elapsed, 2)
    metrics["classifier"]  = classifier
    metrics["condition"]   = label

    return model, metrics

def run_classifier():
    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    # ── 1. load incident data ─────────────────────────────────────────────────
    print("\n[ML] Loading incident_cases from HDFS...")
    df = spark.read.schema(INCIDENT_SCHEMA).json(
        "hdfs://localhost:9000/traffic/raw/incident_cases.json"
    )
    df = df.filter(F.col("text").isNotNull() & F.col("severity").isNotNull())
    total = df.count()
    print(f"[ML] Loaded {total:,} records")

    if total == 0:
        raise RuntimeError(
            "No records loaded. Check that incident_cases.json exists at "
            "hdfs://localhost:9000/traffic/raw/ and is non-empty."
        )

    print(f"[ML] Class distribution:")
    df.groupBy("severity").count().orderBy("severity").show()

    # ── 2. train/test split ───────────────────────────────────────────────────
    df_train, df_test = df.randomSplit([0.8, 0.2], seed=42)
    print(f"[ML] Train: {df_train.count():,}  Test: {df_test.count():,}")

    all_results = []

    # ── 3. train both classifiers on clean data ───────────────────────────────
    for clf_name in ["logreg", "svm"]:
        model, metrics = train_and_evaluate(
            spark, df_train, df_test,
            classifier=clf_name, label="clean"
        )
        all_results.append(metrics)

        # save the logreg model to HDFS (agents will load this)
        if clf_name == "logreg":
            model_path = "hdfs://localhost:9000/traffic/models/severity_classifier"
            model.write().overwrite().save(model_path)
            print(f"[ML] Model saved to: {model_path}")

    # ── 4. adversarial test A: 20% null fields (simulate sensor dropout) ──────
    print("\n[ML] Adversarial test A — 20% null text fields")
    df_null = df_test.withColumn(
        "text",
        F.when(F.rand(seed=1) < 0.2, F.lit("unknown incident reported"))
         .otherwise(F.col("text"))
    )
    pipeline_a, _ = build_pipeline("logreg")
    model_a  = pipeline_a.fit(df_train)
    preds_a  = model_a.transform(df_null)
    metrics_a = evaluate(preds_a, label="logreg/null_20pct")
    metrics_a["classifier"] = "logreg"
    metrics_a["condition"]  = "null_20pct"
    all_results.append(metrics_a)

    # ── 5. adversarial test B: class imbalance (90% low severity) ────────────
#    print("\n[ML] Adversarial test B — class imbalance (90% low)")
#   df_low   = df_train.filter(F.col("severity") == "low")
#    df_other = df_train.filter(F.col("severity") != "low")
#    ratio    = int(df_other.count() * 9)
#    df_imbal = df_low.limit(ratio).union(df_other)
#    pipeline_b, _ = build_pipeline("logreg")
#    model_b  = pipeline_b.fit(df_imbal)
#    preds_b  = model_b.transform(df_test)
#    metrics_b = evaluate(preds_b, label="logreg/imbalanced")
#    metrics_b["classifier"] = "logreg"
#    metrics_b["condition"]  = "imbalanced_90pct_low"
#    all_results.append(metrics_b)

    # ── 6. adversarial test C: noisy labels (10% random relabelling) ──────────
    print("\n[ML] Adversarial test C — 10% noisy labels")
    labels   = ["low", "medium", "high", "critical"]
    df_noisy = df_train.withColumn(
        "severity",
        F.when(F.rand(seed=2) < 0.1,
               F.array([F.lit(l) for l in labels]).getItem(
                   (F.rand(seed=3) * 4).cast("int")
               ))
         .otherwise(F.col("severity"))
    )
    pipeline_c, _ = build_pipeline("logreg")
    model_c  = pipeline_c.fit(df_noisy)
    preds_c  = model_c.transform(df_test)
    metrics_c = evaluate(preds_c, label="logreg/noisy_labels")
    metrics_c["classifier"] = "logreg"
    metrics_c["condition"]  = "noisy_labels_10pct"
    all_results.append(metrics_c)

    # ── 7. print summary table ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[ML] ── Results Summary ──────────────────────────────────")
    print(f"{'Classifier':<10} {'Condition':<25} {'Accuracy':>10} {'F1':>8} {'Time(s)':>8}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['classifier']:<10} {r['condition']:<25} "
              f"{r['accuracy']:>10.4f} {r['f1']:>8.4f} "
              f"{r.get('elapsed_sec', '-'):>8}")

    # ── 8. save results ────────────────────────────────────────────────────────
    out = Path("data/ml_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[ML] Results saved to {out}")

    spark.stop()

if __name__ == "__main__":
    run_classifier()
