"""
Kafka Producer
Generates and streams synthetic traffic sensor events to the 'traffic-sensors' topic.
Supports two modes: normal event generation and adversarial injection,
where a configurable fraction of records are deliberately malformed for pipeline stress-testing.
"""
import json
import time
import random
import uuid
import argparse
from datetime import datetime
from kafka import KafkaProducer

# ── Data Pools ────────────────────────────────────────────────────────────────
INTERSECTIONS = [
    "King_Road_x_Palestine", "Haram_Road_x_Madinah", "Tahlia_x_MBS",
    "Corniche_x_Andalus", "Prince_Sultan_x_Sari", "Al_Hamra_x_Falastin",
    "Airport_Road_x_Asfan", "Al_Waziriyah_x_Sitteen"
]
ZONES = {
    "King_Road_x_Palestine": "north", "Haram_Road_x_Madinah": "central",
    "Tahlia_x_MBS": "central",        "Corniche_x_Andalus": "west",
    "Prince_Sultan_x_Sari": "south",  "Al_Hamra_x_Falastin": "north",
    "Airport_Road_x_Asfan": "east",   "Al_Waziriyah_x_Sitteen": "south"
}
VEHICLE_TYPES  = ["car", "truck", "bus", "motorcycle", "ambulance", "police"]
WEATHER_CONDS  = ["clear", "foggy", "rainy", "dusty", "hot"]
CONGESTION_LVL = ["low", "medium", "high", "critical"]

def make_event(adversarial=False, burst=False):
    """
    Build a single traffic sensor event dict.
    If adversarial is True, randomly corrupt the event to simulate bad data
    (null fields, malformed JSON, duplicate IDs, or wrong field types).
    The burst parameter is reserved for future rate-spike behaviour.
    """
    intersection = random.choice(INTERSECTIONS)
    event = {
        "event_id":        str(uuid.uuid4()),
        "timestamp":       datetime.now().isoformat(),
        "intersection_id": intersection,
        "zone":            ZONES[intersection],
        "speed_kmh":       round(random.uniform(5, 120), 1),
        "congestion":      random.choice(CONGESTION_LVL),
        "vehicle_type":    random.choice(VEHICLE_TYPES),
        "vehicle_count":   random.randint(1, 80),
        "weather":         random.choice(WEATHER_CONDS),
        "signal_phase":    random.choice(["green", "red", "yellow"]),
        "incident_flag":   random.random() < 0.05,
        "record_type":     "normal"
    }

    # ── Adversarial Injection ─────────────────────────────────────────────────
    if adversarial:
        fault = random.choice(["null_fields", "corrupt_json", "duplicate", "wrong_type"])

        if fault == "null_fields":
            nullable = ["speed_kmh", "congestion", "vehicle_type", "weather", "vehicle_count"]
            for field in random.sample(nullable, random.randint(1, 3)):
                event[field] = None
            event["record_type"] = "null_fields"

        elif fault == "corrupt_json":
            return "{ broken json ::::" + str(random.randint(0, 9999))

        elif fault == "duplicate":
            event["event_id"] = "DUPLICATE-FIXED-ID-001"
            event["record_type"] = "duplicate"

        elif fault == "wrong_type":
            event["speed_kmh"]     = "FAST"
            event["vehicle_count"] = "many"
            event["record_type"]   = "wrong_type"

    return event

def run(rate=10, adversarial_pct=0.0, burst=False, total=None):
    """
    Stream events to Kafka at the given rate (events per second).
    adversarial_pct controls the fraction of deliberately malformed records (0.0–1.0).
    Runs indefinitely until interrupted, or stops after total events if total is set.
    """
    producer = KafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: (
            v.encode() if isinstance(v, str)
            else json.dumps(v).encode()
        )
    )

    print(f"[Producer] Starting — rate={rate}/sec  adversarial={adversarial_pct*100:.0f}%  burst={burst}")
    sent = 0
    interval = 1.0 / rate

    try:
        while True:
            is_adversarial = random.random() < adversarial_pct
            event = make_event(adversarial=is_adversarial, burst=burst)

            producer.send("traffic-sensors", value=event)
            sent += 1

            if sent % 100 == 0:
                print(f"[Producer] Sent {sent} events")

            if total and sent >= total:
                print(f"[Producer] Sent {sent} events")
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n[Producer] Stopped. Total sent: {sent}")
    finally:
        producer.flush()
        producer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate",       type=int,   default=10,  help="Events per second")
    parser.add_argument("--adversarial",type=float, default=0.0, help="Fraction of bad records (0.0-1.0)")
    parser.add_argument("--burst",      action="store_true",     help="10x rate spike mode")
    parser.add_argument("--total",      type=int,   default=None,help="Stop after N events")
    args = parser.parse_args()

    # Burst mode multiplies the base rate by 10x to simulate traffic spikes.
    actual_rate = args.rate * 10 if args.burst else args.rate
    run(rate=actual_rate, adversarial_pct=args.adversarial, burst=args.burst, total=args.total)
