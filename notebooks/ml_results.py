"""
Results Visualization — generates charts from benchmark and ML results.
Run with: python3 notebooks/ml_results.py
Outputs PNG files to notebooks/figures/
"""
import json
import os
from pathlib import Path

# Use non-interactive backend (no display needed on server)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

figures_dir = Path("notebooks/figures")
figures_dir.mkdir(parents=True, exist_ok=True)

# ── 1. benchmark chart: processing time vs scale ─────────────────────────────
bench_path = Path("data/benchmark_results.json")
if bench_path.exists():
    with open(bench_path) as f:
        bench = json.load(f)

    scales     = [r["scale"]          for r in bench]
    etl_times  = [r["etl_time_sec"]   for r in bench]
    feat_times = [r["feat_time_sec"]  for r in bench]
    throughput = [r["throughput_rps"] for r in bench]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(scales))
    w = 0.35
    ax1.bar(x - w/2, etl_times,  w, label="ETL",     color="#378ADD")
    ax1.bar(x + w/2, feat_times, w, label="Features", color="#1D9E75")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scales)
    ax1.set_xlabel("Scale")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("Processing Time vs Scale")
    ax1.legend()

    ax2.plot(scales, throughput, "o-", color="#BA7517", linewidth=2, markersize=8)
    ax2.set_xlabel("Scale")
    ax2.set_ylabel("Records / second")
    ax2.set_title("Throughput vs Scale")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "benchmark_chart.png", dpi=150)
    print(f"[Viz] Saved benchmark_chart.png")

# ── 2. ML accuracy: clean vs adversarial ─────────────────────────────────────
ml_path = Path("data/ml_results.json")
if ml_path.exists():
    with open(ml_path) as f:
        ml = json.load(f)

    conditions = [r["condition"]  for r in ml]
    accuracies = [r["accuracy"]   for r in ml]
    f1_scores  = [r["f1"]         for r in ml]

    labels_short = {
        "clean":               "Clean",
        "null_20pct":          "20% Nulls",
        "imbalanced_90pct_low":"Imbalanced",
        "noisy_labels_10pct":  "10% Noise",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(conditions))
    w = 0.35
    bars1 = ax.bar(x - w/2, accuracies, w, label="Accuracy", color="#378ADD")
    bars2 = ax.bar(x + w/2, f1_scores,  w, label="F1 Score",  color="#7F77DD")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([labels_short.get(c, c) for c in conditions], rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("Classifier Performance: Clean vs Adversarial Conditions")
    ax.legend()
    ax.axhline(y=0.7, color="red", linestyle="--", alpha=0.5, label="0.7 threshold")

    # add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(figures_dir / "ml_accuracy_chart.png", dpi=150)
    print(f"[Viz] Saved ml_accuracy_chart.png")

print(f"\n[Viz] All figures saved to {figures_dir}/")
print("[Viz] Use these in your report and poster.")
