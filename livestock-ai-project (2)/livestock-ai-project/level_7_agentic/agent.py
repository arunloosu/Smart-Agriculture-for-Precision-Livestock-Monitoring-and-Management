"""
Agentic AI orchestrator — perceive -> reason -> plan -> act loop over the
real Levels 1-4 model artifacts, with human-in-the-loop safety gating.

Usage:
    export ANTHROPIC_API_KEY=...
    python agent.py --simulate
"""
import argparse
import json
import os
import random
import time

import anthropic

from tools import get_behavior_prediction, get_advisory_context, raise_alert, request_human_confirmation

CONFIDENCE_FLOOR = 0.6  # below this, queue for human review instead of auto-alerting

TOOLS = [
    {
        "name": "get_behavior_prediction",
        "description": "Get the latest ML/DL behavior classification and anomaly score for an animal from its recent sensor window.",
        "input_schema": {
            "type": "object",
            "properties": {
                "animal_id": {"type": "string"},
                "recent_window": {"type": "object", "description": "feature_name -> value map of the latest accelerometer window"},
            },
            "required": ["animal_id", "recent_window"],
        },
    },
    {
        "name": "get_advisory_context",
        "description": "Get farmer-advisory text from the fine-tuned SLM for a natural-language description of a symptom or situation.",
        "input_schema": {"type": "object", "properties": {"query_text": {"type": "string"}}, "required": ["query_text"]},
    },
    {
        "name": "raise_alert",
        "description": "Notify the farmer/vet of a likely health issue. Rate-limited to avoid alert fatigue.",
        "input_schema": {
            "type": "object",
            "properties": {
                "animal_id": {"type": "string"},
                "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                "likely_condition": {"type": "string"},
                "recommended_action": {"type": "string"},
            },
            "required": ["animal_id", "severity", "likely_condition", "recommended_action"],
        },
    },
    {
        "name": "request_human_confirmation",
        "description": "REQUIRED before any automated physical intervention (e.g., isolating an animal, changing feed dispenser settings). Never skip this for 'intervention' actions.",
        "input_schema": {
            "type": "object",
            "properties": {"animal_id": {"type": "string"}, "proposed_action": {"type": "string"}},
            "required": ["animal_id", "proposed_action"],
        },
    },
]

TOOL_IMPL = {
    "get_behavior_prediction": lambda **kw: get_behavior_prediction(**kw),
    "get_advisory_context": lambda **kw: get_advisory_context(**kw),
    "raise_alert": lambda **kw: raise_alert(**kw),
    "request_human_confirmation": lambda **kw: request_human_confirmation(**kw),
}

SYSTEM_PROMPT = """You are the autonomous livestock health agent for a precision agriculture platform.
Given a sensor event, use the available tools to:
1. Check the behavior model for anomaly signals.
2. If the anomaly score is meaningful, pull relevant advisory context.
3. Decide on an action:
   - If confidence is high and there's a clear likely condition: raise_alert.
   - If the action would involve physically intervening on the animal (not just notifying a human): you MUST call request_human_confirmation first and never bypass it.
   - If confidence is low, do not raise an alert — state that you are deferring to routine monitoring instead of guessing.
Be conservative: false alerts cause alert fatigue and erode farmer trust. Only escalate when the evidence supports it.
"""


def simulate_sensor_event():
    animals = ["cow_001", "cow_002", "cow_017"]
    return {
        "animal_id": random.choice(animals),
        "recent_window": {
            "mean_x": round(random.uniform(-1, 1), 3), "mean_y": round(random.uniform(-1, 1), 3),
            "mean_z": round(random.uniform(-1, 1), 3), "std_x": round(random.uniform(0, 2), 3),
            "std_y": round(random.uniform(0, 2), 3), "std_z": round(random.uniform(0, 2), 3),
            "min_x": -1.0, "min_y": -1.0, "min_z": -1.0, "max_x": 1.0, "max_y": 1.0, "max_z": 1.0,
            "rms_x": round(random.uniform(0, 1), 3), "rms_y": round(random.uniform(0, 1), 3),
            "rms_z": round(random.uniform(0, 1), 3), "zcr_x": round(random.uniform(0, 1), 3),
            "corr_xy": 0.1, "corr_xz": 0.1,
        },
    }


def run_agent_turn(client, event):
    messages = [{
        "role": "user",
        "content": f"New sensor event for review: {json.dumps(event)}. Decide what to do.",
    }]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            final_text = "".join(b.text for b in response.content if b.type == "text")
            return final_text

        tool_results = []
        for tu in tool_uses:
            result = TOOL_IMPL[tu.name](**tu.input)
            tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": json.dumps(result)})
        messages.append({"role": "user", "content": tool_results})


def main(args):
    client = anthropic.Anthropic()
    os.makedirs("outputs", exist_ok=True)

    for i in range(args.n_events if args.simulate else 1):
        event = simulate_sensor_event()
        print(f"\n=== Event {i+1}: {event} ===")
        outcome = run_agent_turn(client, event)
        print(f"Agent decision summary: {outcome}")
        time.sleep(0.5)

    print("\nFull audit trail: outputs/decision_log.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true", help="Run against simulated sensor events")
    parser.add_argument("--n_events", type=int, default=5)
    args = parser.parse_args()
    main(args)
