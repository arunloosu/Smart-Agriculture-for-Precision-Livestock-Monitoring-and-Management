# Level Agentic AI — Autonomous Livestock Manager

### What it does
Wraps every prior level as a **tool** the agent can call, and runs a
perceive → reason → plan → act loop:

- **Perceive:** pull latest behavior classification (Level ML/DL), body-temp
  trend, and any farmer/vet advisory text (Level NLP/SLM).
- **Reason:** an LLM-based planner (via the Anthropic API, or any tool-calling
  LLM) synthesizes these signals against a policy/knowledge base.
- **Plan:** decide whether to (a) log and monitor, (b) notify the farmer,
  (c) escalate to a vet, or (d) — only with human confirmation — trigger an
  automated intervention (e.g., isolate the animal, adjust feed dispenser).
- **Act:** calls the Alerting API (Level LLD's `/v1/alerts`) or a real
  intervention endpoint, and logs the outcome for continuous learning.

### Real data path
This agent doesn't introduce a new dataset — it **consumes the real outputs**
of Levels 1–6 (trained models + KCC-derived advisory text), which is the
correct way to build the "Final Evolution" per the problem statement: an
agent that "interprets diverse sensor data... generates optimized management
plans... continuously learning from new data and intervention outcomes."

### Architecture
```
                ┌───────────────────────┐
Sensor stream → │  Perception Tools      │
Camera frames → │  (ML/DL model calls)   │
Farmer text   → │  (NLP/SLM model calls) │
                └───────────┬────────────┘
                            ▼
                  ┌───────────────────┐
                  │  Agent Core (LLM)  │  <- reasons over tool outputs +
                  │  Planner/Reasoner  │     retrieved policy knowledge
                  └─────────┬──────────┘
                            ▼
                  ┌───────────────────┐
                  │  Action Tools      │
                  │  alert / escalate  │
                  │  / (HITL) intervene│
                  └─────────┬──────────┘
                            ▼
                  ┌───────────────────┐
                  │  Outcome Logger    │ -> feeds back into retraining data
                  └───────────────────┘
```

### Safety Mechanisms (required by rubric)
1. **Human-in-the-loop gate** — any action tagged `intervention` (vs. `alert`)
   requires an explicit farmer/vet acknowledgment before execution.
2. **Confidence thresholds** — the agent will not auto-escalate below a
   calibrated confidence floor; low-confidence cases are queued for human
   review instead of triggering false alarms.
3. **Action audit log** — every decision, its inputs, and the human response
   are persisted (`outputs/decision_log.jsonl`) for after-the-fact review.
4. **Rate limiting / anti-flooding** — no more than N alerts per animal per
   hour, to avoid alert fatigue.

### Run
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # or point LLM_CLIENT at any tool-calling LLM
python agent.py --simulate     # runs against a simulated sensor/text stream
                                # for demonstration without live hardware
```

### Deliverables
- `agent.py` — the orchestration loop + tool definitions
- `tools.py` — thin wrappers around Level 1/2/3/4 model artifacts + Level 5 API contracts
- `outputs/decision_log.jsonl` — every agent decision with rationale, for audit
- `outputs/evaluation.json` — autonomy effectiveness metrics (see below)

### Evaluation of Agent Autonomy
- **Precision of escalations:** of alerts the agent raised, % confirmed by a
  vet as a real issue (measured against the human-acknowledgment log).
- **Time-to-alert:** latency from anomaly onset to farmer notification.
- **Safe-guard trigger rate:** how often the HITL gate correctly blocked a
  low-confidence auto-intervention.
