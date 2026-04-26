"""
Full End-to-End Pipeline
Runs Spark MLlib inference, passes enriched records through the three-agent
pipeline, and saves both intermediate and final outputs to disk.
"""
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-arm64"
import json
from pathlib import Path
from agents.spark_bridge import load_and_score
from agents.integrated_pipeline import run_pipeline

print("="*60)
print("FULL PIPELINE: Spark MLlib → Agents")
print("="*60)

# Step 1: Run Spark inference and save enriched records for inspection.
print("\n[Step 1] Running Spark inference...")
records = load_and_score(limit=10)

bridge_out = Path("data/spark_predictions.json")
with open(bridge_out, "w") as f:
    json.dump(records, f, indent=2, default=str)
print(f"[Step 1] Saved {len(records)} enriched records to {bridge_out}")

# Step 2: Pass enriched records through the agent pipeline.
print("\n[Step 2] Running agent pipeline...")
results = run_pipeline(records)

# Step 3: Persist agent outputs and print a per-scenario summary.
out = Path("data/agent_outputs.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for i, r in enumerate(results):
    print(f"\nScenario {i+1}: {r['tla']['intersection_id']}")
    print(f"  Severity : {r['ida']['severity']}")
    print(f"  Action   : {r['tla']['action']}")
    print(f"  Green/Red: {r['tla']['timing_sec']['green']}s / {r['tla']['timing_sec']['red']}s")
    print(f"  Decision : {r['tla']['decision'][:70]}...")
