"""
Integrated Agent Pipeline — Phase 4, Step 4
Vehicle Agent, Incident Detection Agent, Traffic Light Agent
now consume Spark MLlib predictions from the bridge.
RAG and KG reasoning remain from the original NLP project.
"""
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-arm64"

import json
import random
import time
import networkx as nx
from pathlib import Path

# ── Knowledge Graph (from original project) ───────────────────────────────────
def build_knowledge_graph():
    G = nx.DiGraph()
    intersections = [
        "King_Road_x_Palestine", "Haram_Road_x_Madinah",
        "Tahlia_x_MBS",          "Corniche_x_Andalus",
        "Prince_Sultan_x_Sari",  "Al_Hamra_x_Falastin",
        "Airport_Road_x_Asfan",  "Al_Waziriyah_x_Sitteen"
    ]
    hospitals = ["King_Fahd_Hospital", "Al_Noor_Hospital"]
    zones     = ["north", "central", "west", "south", "east"]

    for node in intersections + hospitals + zones:
        G.add_node(node)

    edges = [
        ("King_Road_x_Palestine",  "Haram_Road_x_Madinah",   {"type": "CONNECTED_TO", "dist_km": 2.1}),
        ("Haram_Road_x_Madinah",   "Tahlia_x_MBS",           {"type": "CONNECTED_TO", "dist_km": 1.8}),
        ("Tahlia_x_MBS",           "Corniche_x_Andalus",     {"type": "CONNECTED_TO", "dist_km": 3.2}),
        ("Prince_Sultan_x_Sari",   "Al_Hamra_x_Falastin",    {"type": "CONNECTED_TO", "dist_km": 1.5}),
        ("Al_Hamra_x_Falastin",    "King_Road_x_Palestine",  {"type": "CONNECTED_TO", "dist_km": 0.9}),
        ("Airport_Road_x_Asfan",   "Al_Waziriyah_x_Sitteen", {"type": "CONNECTED_TO", "dist_km": 4.1}),
        ("King_Road_x_Palestine",  "King_Fahd_Hospital",     {"type": "NEAR",         "dist_km": 0.8}),
        ("Haram_Road_x_Madinah",   "Al_Noor_Hospital",       {"type": "NEAR",         "dist_km": 1.2}),
    ]
    G.add_edges_from(edges)
    return G

KG = build_knowledge_graph()

def nearest_hospital(intersection):
    hospitals = ["King_Fahd_Hospital", "Al_Noor_Hospital"]
    best, best_len = None, float("inf")
    for h in hospitals:
        try:
            path = nx.shortest_path(KG, source=intersection, target=h)
            if len(path) < best_len:
                best, best_len = h, len(path)
        except nx.NetworkXNoPath:
            continue
    return best or "Al_Noor_Hospital"

def neighbors(intersection):
    return list(KG.successors(intersection))

# ── simple in-memory RAG ───────────────────────────────────────────────────────
def load_policies():
    path = Path("data/raw/traffic_policies.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []

def load_emergency_protocols():
    path = Path("data/raw/emergency_protocols.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []

POLICIES  = load_policies()
PROTOCOLS = load_emergency_protocols()

def rag_retrieve(query, documents, top_k=2):
    query_words = set(query.lower().split())
    scored = []
    for doc in documents:
        text  = doc.get("text", "").lower()
        score = sum(1 for w in query_words if w in text)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]

# ── Agent 1: Vehicle Agent ────────────────────────────────────────────────────
def vehicle_agent(record):
    inter      = record["intersection_id"]
    zone       = record["zone"]
    congestion = record["congestion"]
    speed      = record["speed_kmh"]
    vehicle    = record["vehicle_type"]
    weather    = record["weather"]
    risk       = record["risk_score"]
    hour_bin   = record["hour_bin"]
    nbrs       = neighbors(inter)

    query       = f"{congestion} congestion {vehicle} {weather}"
    policies    = rag_retrieve(query, POLICIES, top_k=2)
    policy_text = "; ".join(p["text"][:80] for p in policies) if policies else "Standard protocol applies"

    message = (
        f"[VA] Intersection {inter} (zone: {zone}) — "
        f"{vehicle} at {speed:.1f} km/h. "
        f"Congestion: {congestion}. Weather: {weather}. "
        f"Period: {hour_bin}. Risk score: {risk:.1f}. "
        f"Policy context: {policy_text}."
    )

    alerts = []
    if risk >= 75:
        for nbr in nbrs:
            alerts.append(f"[VA→{nbr}] High-risk spillover warning from {inter} (risk={risk:.1f})")

    return {
        "agent":     "VehicleAgent",
        "message":   message,
        "alerts":    alerts,
        "risk":      risk,
        "neighbors": nbrs
    }

# ── Agent 2: Incident Detection Agent ────────────────────────────────────────
SEVERITY_THRESHOLDS = {
    "low":      (0,   25),
    "medium":   (25,  50),
    "high":     (50,  75),
    "critical": (75, 101),
}

def incident_agent(record, va_output):
    risk       = record["risk_score"]
    is_emerg   = record["is_emergency"]
    congestion = record["congestion"]
    vehicle    = record["vehicle_type"]
    inter      = record["intersection_id"]
    spark_sev  = record.get("severity_label", "low")

    severity = spark_sev

    if is_emerg and congestion in ["high", "critical"]:
        severity = "critical"

    query     = f"{vehicle} {severity} {congestion}"
    protocols = rag_retrieve(query, PROTOCOLS, top_k=1)
    protocol  = protocols[0]["text"][:100] if protocols else "Follow standard emergency protocol"

    actions = {
        "low":      "monitor",
        "medium":   "rebalance_signal_timing",
        "high":     "activate_reroute",
        "critical": "emergency_priority_clear"
    }

    hospital = nearest_hospital(inter) if severity in ["high", "critical"] else None

    report = {
        "agent":               "IncidentDetectionAgent",
        "intersection_id":     inter,
        "severity":            severity,
        "spark_predicted_sev": spark_sev,
        "risk_score":          risk,
        "is_emergency":        bool(is_emerg),
        "recommended_action":  actions[severity],
        "protocol":            protocol,
        "nearest_hospital":    hospital,
        "va_risk_confirmed":   va_output["risk"] >= 50 and severity in ["high", "critical"]
    }

    print(f"[IDA] {inter}: severity={severity}  action={actions[severity]}"
          + (f"  → route to {hospital}" if hospital else ""))
    return report

# ── Agent 3: Traffic Light Agent ─────────────────────────────────────────────
PHASE_DURATIONS = {
    "low":      {"green": 30, "yellow": 5, "red": 25},
    "medium":   {"green": 25, "yellow": 5, "red": 30},
    "high":     {"green": 20, "yellow": 5, "red": 35},
    "critical": {"green":  0, "yellow": 3, "red": 60},
}

def traffic_light_agent(record, ida_output):
    severity = ida_output["severity"]
    action   = ida_output["recommended_action"]
    inter    = record["intersection_id"]
    is_emerg = ida_output["is_emergency"]

    timing = PHASE_DURATIONS[severity].copy()

    if is_emerg and action == "emergency_priority_clear":
        timing = {"green": 60, "yellow": 3, "red": 0}
        decision_text = f"EMERGENCY OVERRIDE: Full green phase for {inter}. Clear all lanes."
    elif action == "activate_reroute":
        timing["red"] += 10
        decision_text = f"Rerouting activated at {inter}. Extended red to divert traffic."
    elif action == "rebalance_signal_timing":
        decision_text = f"Rebalancing signal timing at {inter} to ease medium congestion."
    else:
        decision_text = f"Normal monitoring at {inter}. Standard timing maintained."

    query   = f"{severity} {action} signal"
    context = rag_retrieve(query, POLICIES, top_k=1)
    policy  = context[0]["text"][:80] if context else "Default timing policy applied"

    result = {
        "agent":           "TrafficLightAgent",
        "intersection_id": inter,
        "severity":        severity,
        "action":          action,
        "timing_sec":      timing,
        "decision":        decision_text,
        "policy_context":  policy,
        "is_emergency":    is_emerg,
    }

    print(f"[TLA] {inter}: {action} | green={timing['green']}s  red={timing['red']}s")
    return result

# ── Full pipeline runner ──────────────────────────────────────────────────────
def run_pipeline(records):
    print(f"\n[Pipeline] Processing {len(records)} records through agent pipeline\n")
    outputs = []

    for i, record in enumerate(records):
        print(f"── Scenario {i+1}: {record['intersection_id']} "
              f"({'EMERGENCY' if record['is_emergency'] else 'normal'}) ──")

        va  = vehicle_agent(record)
        ida = incident_agent(record, va)
        tla = traffic_light_agent(record, ida)

        outputs.append({"va": va, "ida": ida, "tla": tla})
        print()

    return outputs

if __name__ == "__main__":
    bridge_output_path = Path("data/spark_predictions.json")

    if bridge_output_path.exists():
        with open(bridge_output_path) as f:
            records = json.load(f)[:5]
        print(f"[Pipeline] Loaded {len(records)} records from Spark bridge")
    else:
        print("[Pipeline] No bridge output found — using synthetic test records")
        records = [
            {
                "event_id":        f"test-{i}",
                "intersection_id": random.choice([
                    "King_Road_x_Palestine", "Haram_Road_x_Madinah",
                    "Tahlia_x_MBS", "Airport_Road_x_Asfan"
                ]),
                "zone":            random.choice(["north", "central", "south", "east"]),
                "congestion":      random.choice(["low", "medium", "high", "critical"]),
                "congestion_score":random.choice([25, 50, 75, 100]),
                "risk_score":      round(random.uniform(10, 95), 1),
                "vehicle_type":    random.choice(["car", "ambulance", "truck", "bus"]),
                "is_emergency":    random.random() < 0.25,
                "weather":         random.choice(["clear", "foggy", "rainy"]),
                "is_bad_weather":  random.random() < 0.3,
                "speed_kmh":       round(random.uniform(10, 100), 1),
                "vehicle_count":   random.randint(1, 60),
                "hour_bin":        random.choice(["morning_rush", "off_peak", "evening_rush"]),
                "severity_label":  random.choice(["low", "medium", "high", "critical"]),
            }
            for i in range(5)
        ]

    results = run_pipeline(records)

    out_path = Path("data/agent_outputs.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Pipeline] Outputs saved to {out_path}")
