# Dashboard — Precision Livestock Monitoring
### Farmer/Vet Operations View (Streamlit)

A single operational dashboard that pulls together the outputs of every level
in the project — this is the "front door" a farmer or vet would actually use,
and it maps directly onto the `GET /v1/animals/.../behavior`, `GET /v1/alerts`,
and `POST /v1/advisory/query` endpoints defined in `level_5_lld/LLD.md`.

### What it shows
| Tab | Data Source |
|---|---|
| **Farm Overview** | KPI roll-up across all levels (animals monitored, open alerts, model confidence) |
| **Behavior Monitoring** | Level ML/DL outputs — behavior distribution, per-animal timeline, confusion matrix |
| **Health Alerts** | Level Agentic AI `decision_log.jsonl` — open/acknowledged alerts, severity breakdown |
| **Advisory Insights (NLP/SLM)** | Level NLP `daily_digest.json` / `emerging_keywords.json`, Level SLM generated advice samples |
| **Gen AI Augmentation** | Level Gen AI `downstream_utility.json` / `robustness_report.json`, synthetic image gallery |
| **System Architecture** | Renders the Level LLD architecture/sequence diagrams |
| **Agent Audit Log** | Full decision-by-decision trace from the agentic orchestrator |

### Real data vs. demo mode
The dashboard looks for each level's real `outputs/` folder
(`../level_1_ml/outputs/`, `../level_3_nlp/outputs/`, etc.). **If a level
hasn't been run yet, it automatically falls back to clearly-labeled demo data**
(a "DEMO DATA" badge appears on that tab) so you can see the full dashboard
immediately, then watch tabs switch to real data as you run each level's
pipeline.

### Run
```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```
Open the URL Streamlit prints (default `http://localhost:8501`).

### Wiring in real data
Run any upstream level's script as documented in its own README, e.g.:
```bash
cd ../level_1_ml && python train.py --data ./data/cattle_behavior.csv
cd ../level_7_agentic && python agent.py --simulate
```
Then refresh the dashboard — it re-reads the `outputs/` folders on every page load.
