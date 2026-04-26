# Traffic Intelligence System (Kafka + Spark + MLlib + Multi-Agent)

A distributed, real-time traffic coordination pipeline where autonomous agents process live sensor data and make semantic traffic control decisions, enriched with:

* Distributed stream ingestion via **Apache Kafka**
* Large-scale ETL and feature engineering via **Apache Spark**
* Severity classification via **Spark MLlib** (TF-IDF + Logistic Regression + SVM)
* **Retrieval-Augmented Generation (RAG)** over traffic policies and emergency protocols
* **Knowledge Graph reasoning** over  intersections and hospitals
* Adversarial stress testing for pipeline robustness
* Scalability benchmarking at 1x, 10x, and 100x data scale

This project demonstrates how a fully distributed traffic intelligence system can combine classical ML, semantic retrieval, and graph reasoning to support real-time urban traffic coordination.

---

## Project Overview

The system is composed of four layers:

**1. Ingestion Layer**
Live sensor events are streamed into Kafka by the producer, validated by the consumer, and written to HDFS in partitioned batches. Invalid records are quarantined separately.

**2. Processing Layer**
Raw HDFS records pass through a 9-step ETL cleaning pipeline, then through feature engineering which computes risk scores, congestion metrics, time-of-day bins, and severity labels.

**3. ML Layer**
A TF-IDF text classification pipeline is trained on incident descriptions using Logistic Regression and SVM. The trained model is saved to HDFS and loaded by the agent bridge at inference time.

**4. Agent Layer**
Three cooperative agents process each enriched record in sequence:

* **VehicleAgent** — Assesses intersection conditions, retrieves relevant traffic policies via RAG, and raises spillover alerts to neighbouring intersections when risk is high.

* **IncidentDetectionAgent** — Determines incident severity using the MLlib prediction, escalates to critical for emergency vehicles in high congestion, retrieves protocol context via RAG, and identifies the nearest hospital via the Knowledge Graph.

* **TrafficLightAgent** — Sets signal phase durations and issues a control decision based on severity and recommended action (monitor / rebalance / reroute / emergency override).

---

## Core Components

**RAG Retrieval**
Keyword overlap scoring over four synthetic datasets — incident cases, traffic policies, road rules, and emergency protocols. Top-k matching documents are passed to agents as decision context.

**Knowledge Graph**
Built with NetworkX as a directed graph of intersections and hospitals. Supports nearest-hospital lookup via shortest path and spillover alert propagation to neighbouring nodes.

**Adversarial Testing**
The producer and classifier both support adversarial modes: null fields, malformed JSON, duplicate IDs, wrong types, and noisy training labels — used to evaluate pipeline robustness.

---

## Big Data Design

This project was built as part of a Big Data course and addresses the core challenges of large-scale data engineering:

**Volume** — The pipeline is designed to handle millions of sensor records. The scalability benchmark tests ETL and feature engineering at 1x (10K), 10x (100K), and 100x (1M) record counts to measure throughput and processing time at scale.

**Velocity** — Apache Kafka handles real-time stream ingestion from simulated edge sensors. The consumer processes and flushes records continuously, partitioning output to HDFS by date and hour for time-aware downstream queries.

**Variety** — The system handles structured sensor records, free-text incident descriptions, and graph-structured road network data — all within the same pipeline.

**Veracity** — A 9-step ETL cleaning pipeline addresses null fields, wrong types, out-of-range values, and duplicates. Adversarial injection tests validate that the pipeline degrades gracefully under real-world data quality issues.

**Distributed Processing** — All heavy computation runs on Apache Spark: ETL, feature engineering, SQL analytics, MLlib training, and batch inference. HDFS provides the shared storage layer across all stages.

---

## File Structure

```
.
├── data_generator.py          # Generates synthetic RAG training datasets
├── kafka_producer.py          # Streams sensor events to Kafka
├── kafka_consumer.py          # Validates events and writes to HDFS
├── etl_pipeline.py            # 9-step cleaning pipeline → partitioned Parquet
├── feature_engineering.py     # Computes 10 features including risk score
├── spark_classifier.py        # Trains TF-IDF + LogReg/SVM severity classifier
├── register_tables.py         # Registers HDFS Parquet as Spark SQL tables
├── sql_analytics.py           # Summary queries for reporting
├── scalability_benchmark.py   # ETL + features at 1x / 10x / 100x scale
├── run_full_pipeline.py       # End-to-end runner: Spark → agents → output
├── agents/
│   ├── spark_bridge.py        # Loads MLlib model, scores records for agents
│   └── integrated_pipeline.py # Three-agent decision pipeline
├── notebooks/
│   └── ml_results.py          # Generates benchmark + accuracy charts as PNGs
└── data/
    ├── raw/                   # Synthetic JSON datasets
    ├── spark_predictions.json # Bridge output consumed by agents
    └── agent_outputs.json     # Final agent decisions
```

---

## ML Evaluation Results

Classifiers trained on synthetic incident text descriptions:

| Classifier | Condition | Accuracy | F1 |
|---|---|---|---|
| Logistic Regression | Clean | 1.0000 | 1.0000 |
| SVM | Clean | 1.0000 | 1.0000 |
| Logistic Regression | 20% null fields | 0.8488 | 0.8546 |
| Logistic Regression | 10% noisy labels | 1.0000 | 1.0000 |

The pipeline maintains strong performance under clean conditions and degrades gracefully under adversarial data quality scenarios. Full results are saved to `data/ml_results.json`.

---

## Setup & How to Run

1. **Clone the repo**
   ```
   git clone https://github.com/JudyAbuquta/CS4074_TrafficBigData.git
   cd CS4074_TrafficBigData
   ```

2. **Create a virtual environment**
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```
   pip install pyspark kafka-python networkx matplotlib numpy
   ```

4. **Set Java home (Java 17 required)**
   ```
   export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64
   ```

5. **Run the pipeline in order**
   ```
   python3 data_generator.py
   python3 kafka_producer.py --total 1000
   python3 kafka_consumer.py
   python3 etl_pipeline.py
   python3 feature_engineering.py
   python3 spark_classifier.py
   python3 run_full_pipeline.py
   ```

   After the model is trained, use `run_full_pipeline.py` directly for subsequent runs.

---

## Contributors

* **[Judy Abuquta](https://github.com/JudyAbuquta)**
* **[Celine Al Harake](https://github.com/CelineHarakee)**
* **[Layal Canoe](https://github.com/layalcanoe)**
