"""
Kafka Consumer
Reads raw traffic sensor events from the 'traffic-sensors' topic,
validates each record against a required schema, and writes valid records
to HDFS in partitioned batches. Invalid records are quarantined separately.
"""
import json
import os
import time
from datetime import datetime
from kafka import KafkaConsumer
import subprocess

# ── HDFS Helpers ──────────────────────────────────────────────────────────────
def hdfs_write(local_path, hdfs_path):
    subprocess.run(
        ["hdfs", "dfs", "-put", "-f", local_path, hdfs_path],
        check=True, capture_output=True
    )

def hdfs_mkdir(path):
    subprocess.run(["hdfs", "dfs", "-mkdir", "-p", path],
                   capture_output=True)

# ── Schema Validation ─────────────────────────────────────────────────────────

# Fields every incoming record must contain, mapped to their expected Python type.
# Tuple values allow multiple acceptable types (e.g. int or float for numeric fields).
REQUIRED_FIELDS = {
    "event_id":       str,
    "timestamp":      str,
    "intersection_id":str,
    "zone":           str,
    "speed_kmh":      (int, float),
    "congestion":     str,
    "vehicle_type":   str,
    "vehicle_count":  int,
    "weather":        str,
}

def validate(record):
    """
    Check a parsed record against the required schema and value constraints.
    Returns a (is_valid, reason) tuple — reason is 'ok' on success,
    or a short error code describing the first failure found.
    """
    if not isinstance(record, dict):
        return False, "not_a_dict"
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in record:
            return False, f"missing_field:{field}"
        val = record[field]
        if val is None:
            return False, f"null_field:{field}"
        if not isinstance(val, expected_type):
            return False, f"wrong_type:{field}"
    if not (0 <= record["speed_kmh"] <= 200):
        return False, "speed_out_of_range"
    if not (0 <= record["vehicle_count"] <= 500):
        return False, "count_out_of_range"
    if record["congestion"] not in ["low", "medium", "high", "critical"]:
        return False, "invalid_congestion_value"
    return True, "ok"

# ── Batch Writer ──────────────────────────────────────────────────────────────
class BatchWriter:
    def __init__(self, hdfs_base, batch_size=100, flush_interval=30):
        """
        Buffer valid and invalid records into separate batches.
        A flush is triggered when either batch reaches batch_size records
        or flush_interval seconds have elapsed since the last flush.
        """
        self.hdfs_base      = hdfs_base
        self.batch_size     = batch_size
        self.flush_interval = flush_interval
        self.good_batch     = []
        self.bad_batch      = []
        self.last_flush     = time.time()
        self.total_good     = 0
        self.total_bad      = 0

    def add(self, record, is_valid, reason):
        if is_valid:
            self.good_batch.append(record)
        else:
            self.bad_batch.append({
                "reason":    reason,
                "raw":       record if isinstance(record, dict) else str(record),
                "timestamp": datetime.utcnow().isoformat()
            })
        if (len(self.good_batch) >= self.batch_size or
            len(self.bad_batch)  >= self.batch_size or
            time.time() - self.last_flush > self.flush_interval):
            self.flush()

    def flush(self):
        ts   = datetime.utcnow()
        date = ts.strftime("%Y-%m-%d")
        hour = ts.strftime("%H")

        # Valid records are written to a date/hour partition for downstream Spark jobs.
        if self.good_batch:
            self._write_batch(
                self.good_batch,
                f"{self.hdfs_base}/processed/date={date}/hour={hour}"
            )
            self.total_good += len(self.good_batch)
            self.good_batch  = []

        # Invalid records are written to a date-level quarantine partition for review.
        if self.bad_batch:
            self._write_batch(
                self.bad_batch,
                f"{self.hdfs_base}/quarantine/date={date}"
            )
            self.total_bad += len(self.bad_batch)
            self.bad_batch  = []

        self.last_flush = time.time()
        print(f"[Consumer] Flushed — total good: {self.total_good}  bad: {self.total_bad}")

    def _write_batch(self, records, hdfs_path):
        hdfs_mkdir(hdfs_path)
        ts        = datetime.utcnow().strftime("%H%M%S_%f")
        local_tmp = f"/tmp/traffic_batch_{ts}.json"
        with open(local_tmp, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        try:
            hdfs_write(local_tmp, f"{hdfs_path}/batch_{ts}.json")
        finally:
            os.remove(local_tmp)

# ── Main Consumer Loop ────────────────────────────────────────────────────────
def run():
    """
    Start the Kafka consumer, validate each incoming message, and route it
    through the BatchWriter. Logs a progress summary every 50 messages
    and performs a final flush on shutdown.
    """
    consumer = KafkaConsumer(
        "traffic-sensors",
        bootstrap_servers="localhost:9092",
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="traffic-consumer-group",
        value_deserializer=lambda m: m.decode("utf-8")
    )
    writer = BatchWriter(
        hdfs_base="/traffic/raw",
        batch_size=100,
        flush_interval=30
    )
    print("[Consumer] Listening on topic: traffic-sensors")
    stats = {"valid": 0, "invalid": 0, "parse_error": 0}
    try:
        for message in consumer:
            raw = message.value
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                stats["parse_error"] += 1
                writer.add(raw, False, "json_parse_error")
                continue
            is_valid, reason = validate(record)
            if is_valid:
                stats["valid"] += 1
            else:
                stats["invalid"] += 1
            writer.add(record, is_valid, reason)
            total = sum(stats.values())
            if total % 50 == 0:
                print(f"[Consumer] Processed {total} | "
                      f"valid={stats['valid']} "
                      f"invalid={stats['invalid']} "
                      f"parse_err={stats['parse_error']}")
    except KeyboardInterrupt:
        print("\n[Consumer] Shutting down — flushing remaining records...")
        writer.flush()
        print(f"[Consumer] Final stats: {stats}")
    finally:
        consumer.close()

if __name__ == "__main__":
    run()
