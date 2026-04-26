"""
Synthetic Data Generator
Produces four JSON datasets used by the RAG retrieval and agent pipeline:
incident cases, traffic policies, road rules, and emergency protocols.
All records are randomly generated from fixed templates and vocabulary lists.
"""
import json
import uuid
import random
from datetime import datetime, timedelta
from pathlib import Path

# Fixed seed ensures reproducible datasets across runs.
random.seed(42)

ROADS        = ["King Road", "Haram Road", "Tahlia Street", "Corniche Road",
                "Prince Sultan Road", "Airport Road", "Al Hamra", "Al Waziriyah"]
INTERSECTIONS = [
    "King_Road_x_Palestine", "Haram_Road_x_Madinah", "Tahlia_x_MBS",
    "Corniche_x_Andalus",    "Prince_Sultan_x_Sari",  "Al_Hamra_x_Falastin",
    "Airport_Road_x_Asfan",  "Al_Waziriyah_x_Sitteen"
]
INCIDENT_TYPES    = ["collision", "breakdown", "emergency_delay", "road_block",
                     "pedestrian_crossing", "flood", "fire", "none"]
SEVERITY_LABELS   = ["low", "medium", "high", "critical"]
VEHICLE_TYPES     = ["car", "truck", "bus", "motorcycle", "ambulance", "police"]
WEATHER_CONDS     = ["clear", "foggy", "rainy", "dusty", "hot"]

def random_timestamp(days_back=30):
    """
    Return a random UTC ISO timestamp within the last days_back days,
    with a randomized hour, minute, and second.
    """
    base = datetime.utcnow() - timedelta(days=random.randint(0, days_back))
    return base.replace(
        hour=random.randint(0,23),
        minute=random.randint(0,59),
        second=random.randint(0,59)
    ).isoformat()

# ── 1. Incident Cases ─────────────────────────────────────────────────────────
def gen_incident_cases(n=1500):
    """Generate n synthetic traffic incident records with free-text descriptions."""
    records = []
    for _ in range(n):
        severity  = random.choice(SEVERITY_LABELS)
        inc_type  = random.choice(INCIDENT_TYPES)
        road      = random.choice(ROADS)
        weather   = random.choice(WEATHER_CONDS)
        veh       = random.choice(VEHICLE_TYPES)

        # Varied sentence templates to simulate realistic dispatcher language.
        templates = [
            f"A {inc_type} involving a {veh} on {road}. Weather: {weather}. Severity assessed as {severity}.",
            f"Reported {inc_type} near {road} intersection. Conditions {weather}. Response needed: {severity} priority.",
            f"{veh.capitalize()} {inc_type} detected on {road} under {weather} conditions. Level: {severity}.",
            f"Traffic incident: {inc_type} on {road}. {weather.capitalize()} weather. Category: {severity} severity.",
        ]
        records.append({
            "id":           str(uuid.uuid4()),
            "timestamp":    random_timestamp(),
            "road":         road,
            "incident_type":inc_type,
            "vehicle_type": veh,
            "weather":      weather,
            "severity":     severity,
            "text":         random.choice(templates)
        })
    return records

# ── 2. Traffic Policies ───────────────────────────────────────────────────────
def gen_traffic_policies(n=1000):
    """Generate n synthetic traffic policy records mapping conditions to agent actions."""
    actions   = ["extend_green", "reduce_green", "trigger_emergency_phase",
                 "activate_reroute", "hold_phase", "send_alert"]
    conditions= ["congestion > 70%", "emergency vehicle detected",
                 "queue length > 30", "weather visibility < 50m",
                 "incident severity >= high", "vehicle count > 60"]
    records = []
    for _ in range(n):
        cond   = random.choice(conditions)
        action = random.choice(actions)
        records.append({
            "policy_id":  str(uuid.uuid4()),
            "condition":  cond,
            "action":     action,
            "priority":   random.randint(1, 5),
            "text": f"When {cond}, agent should {action}. Priority level {random.randint(1,5)}."
        })
    return records

# ── 3. Road Rules ─────────────────────────────────────────────────────────────
def gen_road_rules(n=800):
    """Generate n road rule records by cycling through a fixed set of rule templates."""
    rules = [
        "Ambulances have unconditional right of way at all intersections.",
        "During school hours (7-9am, 1-3pm), reduce speed limits by 20kmh near zones.",
        "Fog visibility below 50m: activate hazard warnings and cut speed to 40kmh.",
        "Emergency vehicles trigger automatic green phase extension of up to 60 seconds.",
        "Maximum queue length before automatic rerouting is activated: 25 vehicles.",
        "Night mode (10pm-5am): extend green phases by 15 seconds on arterial roads.",
        "Police override: all phases yield immediately to police signal.",
        "Incident blocking >1 lane: activate alternate route protocol within 2 minutes.",
    ]
    records = []
    for i in range(n):
        rule = rules[i % len(rules)]
        records.append({
            "rule_id":   str(uuid.uuid4()),
            "category":  random.choice(["speed", "priority", "emergency", "routing", "timing"]),
            "text":      rule,
            "active":    True
        })
    return records

# ── 4. Emergency Protocols ────────────────────────────────────────────────────
def gen_emergency_protocols(n=700):
    """Generate n emergency protocol records by cycling through a fixed set of protocol templates."""
    protocols = [
        "Ambulance on active call: clear all lanes within 200m radius.",
        "Fire truck response: pre-clear route from station to incident location.",
        "Police pursuit: lock all crossing phases to red until pursuit passes.",
        "Mass casualty event: activate city-wide emergency corridor protocol.",
        "VIP convoy: pre-signal all intersections along route 5 minutes ahead.",
    ]
    records = []
    for i in range(n):
        records.append({
            "protocol_id": str(uuid.uuid4()),
            "type":        random.choice(["ambulance", "fire", "police", "mass_casualty", "vip"]),
            "text":        protocols[i % len(protocols)],
            "response_time_sec": random.randint(10, 120)
        })
    return records

# ── Write All Datasets ────────────────────────────────────────────────────────
def main():
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)

    datasets = {
        "incident_cases.json":       gen_incident_cases(1500),
        "traffic_policies.json":     gen_traffic_policies(1000),
        "road_rules.json":           gen_road_rules(800),
        "emergency_protocols.json":  gen_emergency_protocols(700),
    }

    for filename, records in datasets.items():
        path = out / filename
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"[DataGen] Wrote {len(records):,} records → {path}")

    print(f"\n[DataGen] Total records: {sum(len(v) for v in datasets.values()):,}")

if __name__ == "__main__":
    main()
