"""
Tool wrappers the agent can call: perception (ML/DL/NLP model outputs) and
action (alerting / intervention / logging).

These wrap the REAL trained artifacts from Levels 1-4 (joblib/keras models),
matching the API contracts defined in level_5_lld/LLD.md.
"""
import json
import os
import time
from datetime import datetime

import numpy as np

DECISION_LOG = "outputs/decision_log.jsonl"
os.makedirs("outputs", exist_ok=True)

# --- Rate limiting state (per animal, in-memory demo; use Redis in production) ---
_alert_history = {}
MAX_ALERTS_PER_HOUR = 2


def get_behavior_prediction(animal_id: str, recent_window: dict) -> dict:
    """
    Calls the Level ML/DL behavior model on the most recent accelerometer window.
    In production this hits the ML/DL Inference Service (level_5_lld API);
    here we show the direct model-loading path for the demo.
    """
    import joblib
    model_path = "../level_1_ml/outputs/model_random_forest.pkl"
    scaler_path = "../level_1_ml/outputs/scaler.pkl"
    le_path = "../level_1_ml/outputs/label_encoder.pkl"

    if not all(os.path.exists(p) for p in [model_path, scaler_path, le_path]):
        # Fallback for demo/simulation mode when Level-ML hasn't been run yet
        return {"behavior": "lying", "confidence": 0.55, "anomaly_score": 0.3, "note": "SIMULATED (no trained model found)"}

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)

    features = np.array([[recent_window[k] for k in sorted(recent_window.keys())]])
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]
    behavior = le.inverse_transform([pred])[0]

    # Anomaly score: how far this window's lying/inactivity ratio deviates
    # from the animal's own rolling baseline (simplified proxy here).
    anomaly_score = float(1 - proba.max())

    return {"behavior": behavior, "confidence": float(proba.max()), "anomaly_score": anomaly_score}


def get_advisory_context(query_text: str) -> dict:
    """Calls the Level SLM fine-tuned advisor for relevant management advice."""
    model_dir = "../level_4_slm/outputs/slm-livestock-advisor"
    if not os.path.exists(model_dir):
        return {"advice": "SIMULATED: monitor closely and consult a vet if symptoms persist.", "note": "no fine-tuned SLM found"}

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    inputs = tokenizer(f"Provide farming advice for this livestock query: {query_text}",
                        return_tensors="pt", truncation=True, max_length=128)
    out = model.generate(**inputs, max_new_tokens=128)
    return {"advice": tokenizer.decode(out[0], skip_special_tokens=True)}


def raise_alert(animal_id: str, severity: str, likely_condition: str, recommended_action: str) -> dict:
    """POST /v1/alerts equivalent — rate-limited, audit-logged."""
    now = time.time()
    history = _alert_history.setdefault(animal_id, [])
    history[:] = [t for t in history if now - t < 3600]

    if len(history) >= MAX_ALERTS_PER_HOUR:
        result = {"status": "suppressed_rate_limit", "animal_id": animal_id}
    else:
        history.append(now)
        result = {
            "status": "alert_raised", "animal_id": animal_id, "severity": severity,
            "likely_condition": likely_condition, "recommended_action": recommended_action,
        }
    _log_decision("raise_alert", result)
    return result


def request_human_confirmation(animal_id: str, proposed_action: str) -> dict:
    """
    Human-in-the-loop gate. In production this pushes to the farmer/vet app
    and awaits a webhook callback; the demo simulates a pending state.
    """
    result = {"status": "pending_human_confirmation", "animal_id": animal_id, "proposed_action": proposed_action}
    _log_decision("request_human_confirmation", result)
    return result


def _log_decision(action: str, payload: dict):
    record = {"timestamp": datetime.utcnow().isoformat(), "action": action, "payload": payload}
    with open(DECISION_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")
