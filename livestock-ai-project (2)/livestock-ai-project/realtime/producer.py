"""
producer.py
-----------
Simulates a real-time telemetry feed from smart collars / ear-tags, the way
it would arrive from an MQTT broker or ingestion API in production (see
level_5_lld/LLD.md for the target architecture: Sensors -> Ingestion API ->
Feature Store). It writes each reading straight into telemetry.db so the
dashboard's "Live Telemetry" tab can pick it up within seconds.

This is a *simulator*, not a real sensor integration — there is no public
live-streaming API for cattle biometrics, so this reproduces realistic
statistical behaviour instead:
  - Per-animal random-walk vitals (heart rate, body temperature, activity)
    so each animal drifts smoothly rather than jumping randomly.
  - Behaviour state machine (grazing/lying/standing/walking/ruminating) with
    realistic dwell times and transition probabilities.
  - GPS position that wanders inside a fixed farm geofence.
  - Periodic injected anomalies (fever spike, tachycardia, prolonged
    inactivity) that cross clinical thresholds and automatically raise a row
    in live_alerts — so the dashboard has something real to show without
    waiting for a rare random event.

Run:
    python producer.py                  # runs until Ctrl+C, ~1 reading/animal/3s
    python producer.py --interval 1     # faster tick for demos
    python producer.py --duration 120   # stop automatically after 120s
    python producer.py --reset          # wipe telemetry.db before starting
"""
import argparse
import math
import os
import random
import time
from datetime import datetime, timezone

import telemetry_db as db

ANIMALS = [f"cow_{i:03d}" for i in range(1, 21)]
BEHAVIORS = ["grazing", "lying", "standing", "walking", "ruminating"]
TRANSITIONS = {
    "grazing":     {"grazing": 0.75, "walking": 0.15, "standing": 0.07, "lying": 0.02, "ruminating": 0.01},
    "walking":     {"walking": 0.5, "grazing": 0.25, "standing": 0.2, "lying": 0.03, "ruminating": 0.02},
    "standing":    {"standing": 0.55, "grazing": 0.2, "walking": 0.15, "lying": 0.05, "ruminating": 0.05},
    "lying":       {"lying": 0.7, "ruminating": 0.15, "standing": 0.1, "grazing": 0.04, "walking": 0.01},
    "ruminating":  {"ruminating": 0.65, "lying": 0.2, "standing": 0.1, "grazing": 0.04, "walking": 0.01},
}
BEHAVIOR_ACTIVITY_BASE = {"grazing": 0.45, "walking": 0.75, "standing": 0.3, "lying": 0.08, "ruminating": 0.12}

# Farm geofence: a small bounding box (placeholder coordinates near Chennai region farmland)
FARM_CENTER_LAT, FARM_CENTER_LON = 12.90, 79.95
FARM_RADIUS_DEG = 0.004  # roughly a few hundred metres

NORMAL_HR_RANGE = (48, 84)        # bpm, healthy adult cattle resting-to-active range
NORMAL_TEMP_RANGE = (37.8, 39.2)  # deg C, healthy cattle body temp


class AnimalState:
    def __init__(self, animal_id):
        self.animal_id = animal_id
        self.behavior = random.choice(BEHAVIORS)
        self.heart_rate = random.uniform(*NORMAL_HR_RANGE)
        self.body_temp = random.uniform(*NORMAL_TEMP_RANGE)
        angle = random.uniform(0, 2 * math.pi)
        r = FARM_RADIUS_DEG * math.sqrt(random.uniform(0, 1))
        self.lat = FARM_CENTER_LAT + r * math.cos(angle)
        self.lon = FARM_CENTER_LON + r * math.sin(angle)
        self.anomaly_ticks_remaining = 0
        self.anomaly_kind = None

    def maybe_start_anomaly(self, p=0.01):
        if self.anomaly_ticks_remaining == 0 and random.random() < p:
            self.anomaly_kind = random.choice(["fever", "tachycardia", "inactivity"])
            self.anomaly_ticks_remaining = random.randint(6, 14)

    def step(self):
        # behaviour transition
        self.behavior = random.choices(
            list(TRANSITIONS[self.behavior].keys()),
            weights=list(TRANSITIONS[self.behavior].values()),
        )[0]

        self.maybe_start_anomaly()

        target_hr_lo, target_hr_hi = NORMAL_HR_RANGE
        target_temp_lo, target_temp_hi = NORMAL_TEMP_RANGE
        activity_base = BEHAVIOR_ACTIVITY_BASE[self.behavior]

        if self.anomaly_ticks_remaining > 0:
            if self.anomaly_kind == "fever":
                target_temp_lo, target_temp_hi = 39.8, 40.9
            elif self.anomaly_kind == "tachycardia":
                target_hr_lo, target_hr_hi = 95, 130
            elif self.anomaly_kind == "inactivity":
                activity_base = 0.03
                self.behavior = "lying"
            self.anomaly_ticks_remaining -= 1
            if self.anomaly_ticks_remaining == 0:
                self.anomaly_kind = None

        # smooth random-walk toward the (possibly anomalous) target band
        target_hr = random.uniform(target_hr_lo, target_hr_hi)
        target_temp = random.uniform(target_temp_lo, target_temp_hi)
        self.heart_rate += (target_hr - self.heart_rate) * 0.3 + random.uniform(-1.5, 1.5)
        self.body_temp += (target_temp - self.body_temp) * 0.25 + random.uniform(-0.05, 0.05)
        activity = max(0.0, min(1.0, activity_base + random.uniform(-0.08, 0.08)))

        # GPS wander, clamped to the geofence
        self.lat += random.uniform(-0.0003, 0.0003)
        self.lon += random.uniform(-0.0003, 0.0003)
        dlat, dlon = self.lat - FARM_CENTER_LAT, self.lon - FARM_CENTER_LON
        dist = math.hypot(dlat, dlon)
        if dist > FARM_RADIUS_DEG:
            scale = FARM_RADIUS_DEG / dist
            self.lat = FARM_CENTER_LAT + dlat * scale
            self.lon = FARM_CENTER_LON + dlon * scale

        is_anomaly = self.anomaly_kind is not None or self.heart_rate > NORMAL_HR_RANGE[1] * 1.15 \
            or self.body_temp > NORMAL_TEMP_RANGE[1] + 0.5

        return {
            "animal_id": self.animal_id,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "heart_rate": round(self.heart_rate, 1),
            "body_temp_c": round(self.body_temp, 2),
            "activity_index": round(activity, 2),
            "lat": round(self.lat, 6),
            "lon": round(self.lon, 6),
            "behavior": self.behavior,
            "is_anomaly": int(is_anomaly),
        }, self.anomaly_kind


def maybe_raise_alert(reading, anomaly_kind):
    if not reading["is_anomaly"]:
        return
    condition_map = {
        "fever": ("high", "elevated body temperature (possible infection)"),
        "tachycardia": ("high", "elevated heart rate (possible pain/stress/heat stress)"),
        "inactivity": ("medium", "prolonged inactivity (possible lameness/illness)"),
        None: ("medium", "vitals outside normal range"),
    }
    severity, condition = condition_map.get(anomaly_kind, condition_map[None])
    if reading["body_temp_c"] > 40.5 or reading["heart_rate"] > 120:
        severity = "critical"
    db.insert_alert({
        "animal_id": reading["animal_id"],
        "ts": reading["ts"],
        "severity": severity,
        "condition": condition,
        "message": f"{reading['animal_id']}: {condition} — HR {reading['heart_rate']} bpm, "
                   f"temp {reading['body_temp_c']}\u00b0C, activity {reading['activity_index']}",
    })


def run(interval: float, duration: float, reset: bool):
    if reset and os.path.exists(db.DB_PATH):
        os.remove(db.DB_PATH)
    db.init_db()

    states = {a: AnimalState(a) for a in ANIMALS}
    print(f"[producer] streaming {len(ANIMALS)} animals every {interval}s -> {db.DB_PATH}")
    print("[producer] Ctrl+C to stop")

    start = time.time()
    tick = 0
    try:
        while True:
            for state in states.values():
                reading, anomaly_kind = state.step()
                db.insert_reading(reading)
                maybe_raise_alert(reading, anomaly_kind)
            tick += 1
            if tick % 10 == 0:
                print(f"[producer] tick {tick} | rows so far: {db.row_count()}")
            if duration and (time.time() - start) >= duration:
                print("[producer] duration reached, stopping.")
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[producer] stopped by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate real-time livestock telemetry.")
    parser.add_argument("--interval", type=float, default=3.0, help="seconds between ticks (default 3)")
    parser.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = run forever)")
    parser.add_argument("--reset", action="store_true", help="wipe existing telemetry.db first")
    args = parser.parse_args()
    run(args.interval, args.duration, args.reset)
