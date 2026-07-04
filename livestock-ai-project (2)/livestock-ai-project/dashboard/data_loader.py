"""
Loads real outputs from each level's outputs/ folder; falls back to
clearly-labeled simulated demo data when a level hasn't been run yet, so the
dashboard is always demoable end-to-end.
"""
import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

random.seed(7)
np.random.seed(7)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "realtime"))
import telemetry_db as _tdb  # noqa: E402

LEVEL_PATHS = {
    "ml": "../level_1_ml/outputs",
    "dl": "../level_2_dl/outputs",
    "nlp": "../level_3_nlp/outputs",
    "slm": "../level_4_slm/outputs",
    "genai": "../level_6_genai/outputs",
    "agentic": "../level_7_agentic/outputs",
}

ANIMALS = [f"cow_{i:03d}" for i in range(1, 21)]
BEHAVIORS = ["grazing", "lying", "standing", "walking", "ruminating"]


def _exists(level, filename):
    path = os.path.join(LEVEL_PATHS[level], filename)
    return path if os.path.exists(path) else None


def load_ml_metrics():
    path = _exists("ml", "metrics.json")
    if path:
        with open(path) as f:
            return json.load(f), True
    demo = {
        "logreg": {"accuracy": 0.81, "precision_macro": 0.78, "recall_macro": 0.77, "f1_macro": 0.77, "roc_auc_ovr": 0.90},
        "random_forest": {"accuracy": 0.91, "precision_macro": 0.90, "recall_macro": 0.89, "f1_macro": 0.89, "roc_auc_ovr": 0.97},
        "svm_rbf": {"accuracy": 0.87, "precision_macro": 0.86, "recall_macro": 0.85, "f1_macro": 0.85, "roc_auc_ovr": 0.94},
    }
    return demo, False


def load_behavior_timeline():
    """Per-animal behavior over the last 24h — demo data (real version would
    read from the feature store / behavior_event table per LLD.md)."""
    rows = []
    now = datetime.utcnow()
    for animal in ANIMALS:
        t = now - timedelta(hours=24)
        while t < now:
            rows.append({
                "animal_id": animal,
                "timestamp": t,
                "behavior": random.choices(BEHAVIORS, weights=[0.35, 0.3, 0.15, 0.1, 0.1])[0],
                "confidence": round(random.uniform(0.6, 0.99), 2),
            })
            t += timedelta(minutes=30)
    return pd.DataFrame(rows), False  # always demo unless live DB is wired in


def load_alerts():
    path = _exists("agentic", "decision_log.jsonl")
    if path:
        records = []
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec["action"] == "raise_alert" and rec["payload"].get("status") == "alert_raised":
                    records.append(rec)
        if records:
            df = pd.DataFrame([{
                "timestamp": r["timestamp"], "animal_id": r["payload"]["animal_id"],
                "severity": r["payload"]["severity"], "condition": r["payload"]["likely_condition"],
                "action": r["payload"]["recommended_action"],
            } for r in records])
            return df, True

    demo_rows = []
    now = datetime.utcnow()
    conditions = ["possible lameness", "reduced grazing time", "elevated body temperature",
                  "atypical lying pattern", "suspected early mastitis"]
    for i in range(12):
        demo_rows.append({
            "timestamp": (now - timedelta(hours=random.randint(0, 48))).isoformat(),
            "animal_id": random.choice(ANIMALS),
            "severity": random.choices(["low", "medium", "high", "critical"], weights=[0.4, 0.3, 0.2, 0.1])[0],
            "condition": random.choice(conditions),
            "action": "Notify farmer; recommend vet check within 24h",
        })
    return pd.DataFrame(demo_rows), False


def load_nlp_digest():
    path = _exists("nlp", "daily_digest.json")
    if path:
        with open(path) as f:
            return json.load(f), True
    return {
        "Kanchipuram": {"2026-07-01": "Multiple farmers reported reduced feed intake and mild fever in cattle; "
                                       "advisory issued on tick-borne fever precautions ahead of monsoon."},
        "Vellore": {"2026-07-01": "Queries centered on calving readiness and colostrum feeding for newborn calves."},
    }, False


def load_keywords():
    path = _exists("nlp", "emerging_keywords.json")
    if path:
        with open(path) as f:
            return json.load(f)["top_terms"], True
    return ["lumpy skin", "fever", "mastitis", "tick fever", "vaccination", "calving",
            "fodder shortage", "loose motion", "bloat", "lameness"], False


def load_slm_samples():
    path = _exists("slm", "qualitative_samples.json")
    if path:
        with open(path) as f:
            return json.load(f)[:5], True
    return [
        {"query": "My cow has stopped eating and seems weak, temperature is high.",
         "reference": "Isolate the animal, offer soft fodder and water, and call a vet — high temperature with loss of appetite may indicate infection.",
         "generated": "Isolate the cow, keep it hydrated, and consult a veterinarian promptly; the symptoms suggest a possible infection."},
        {"query": "How many days after calving can I start milking normally?",
         "reference": "Colostrum feeding for the calf should continue for the first 3-4 days before regular milking begins.",
         "generated": "Feed colostrum to the calf for the first 3-4 days, then transition to regular milking."},
    ], False


def load_genai_metrics():
    util_path = _exists("genai", "downstream_utility.json")
    robust_path = _exists("genai", "robustness_report.json")
    if util_path or robust_path:
        result = {}
        if util_path:
            with open(util_path) as f:
                result["utility"] = json.load(f)
        if robust_path:
            with open(robust_path) as f:
                result["robustness"] = json.load(f)
        return result, True
    return {
        "utility": {"cnn_accuracy_real_only": 0.87, "cnn_accuracy_real_plus_synthetic": 0.93},
        "robustness": {"clean_accuracy": 0.93, "perturbed_accuracy_before_augmentation": 0.71,
                        "perturbed_accuracy_after_augmentation": 0.85},
    }, False


def load_agentic_audit():
    path = _exists("agentic", "decision_log.jsonl")
    if path:
        records = [json.loads(line) for line in open(path)]
        if records:
            return pd.DataFrame(records), True
    now = datetime.utcnow()
    demo = [
        {"timestamp": (now - timedelta(minutes=i * 12)).isoformat(), "action": a,
         "payload": {"animal_id": random.choice(ANIMALS)}}
        for i, a in enumerate(["get_behavior_prediction", "get_advisory_context", "raise_alert",
                                "request_human_confirmation", "get_behavior_prediction"])
    ]
    return pd.DataFrame(demo), False


# ---------------------------------------------------------------- Live telemetry (real-time)

def live_producer_status():
    """Returns (is_live, seconds_since_last_reading, total_rows).
    is_live is True only if a reading has arrived in the last 15s, i.e. the
    producer.py process is actually running right now."""
    latest = _tdb.fetch_latest_per_animal()
    if not latest:
        return False, None, 0
    most_recent_ts = max(datetime.fromisoformat(r["ts"]) for r in latest)
    if most_recent_ts.tzinfo is None:
        most_recent_ts = most_recent_ts.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - most_recent_ts).total_seconds()
    return age < 15, age, _tdb.row_count()


def load_live_latest():
    """Latest reading per animal. Empty DataFrame if the producer has never run."""
    rows = _tdb.fetch_latest_per_animal()
    if not rows:
        return pd.DataFrame(columns=["animal_id", "ts", "heart_rate", "body_temp_c",
                                      "activity_index", "lat", "lon", "behavior", "is_anomaly"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.sort_values("animal_id")


def load_live_history(minutes: int = 15):
    """Readings from the last `minutes` for time-series charts. Empty if no producer running."""
    rows = _tdb.fetch_recent_readings(minutes=minutes)
    if not rows:
        return pd.DataFrame(columns=["animal_id", "ts", "heart_rate", "body_temp_c",
                                      "activity_index", "lat", "lon", "behavior", "is_anomaly"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def load_live_alerts(limit: int = 50):
    rows = _tdb.fetch_recent_alerts(limit=limit)
    if not rows:
        return pd.DataFrame(columns=["id", "animal_id", "ts", "severity", "condition", "message", "acknowledged"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.sort_values("ts", ascending=False)
