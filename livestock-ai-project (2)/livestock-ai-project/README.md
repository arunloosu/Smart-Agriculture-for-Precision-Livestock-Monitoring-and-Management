# Smart Agriculture — Precision Livestock Monitoring & Management
### Full Project, Level by Level, Built on Real Public Datasets

This project implements Problem Statement 090 end-to-end: a system that monitors
livestock health and behavior, detects early signs of illness/distress, and
optimizes feeding — evolving from a classic ML baseline to a full Agentic AI system.

Every level below is wired to a **real, publicly available dataset** (not
synthetic placeholders), with working Python code, a documented pipeline, and
an evaluation section matching the rubric in the problem statement.

> **Network note:** This code was written and packaged in a sandboxed
> environment with no internet access, so the datasets are **not bundled** in
> this zip (some are multi-GB). Each level's `README.md` gives the exact
> download link/command. Drop the downloaded data into that level's `data/`
> folder and run the script — nothing else needs to change.

---

## Levels & Real Datasets Used

| Level | Folder | Real Dataset | Source |
|---|---|---|---|
| ML — Baseline | `level_1_ml/` | **Beef Cattle Behavior Dataset** (tri-axial accelerometer, labeled: grazing, lying, standing, walking, ruminating) | Kaggle: `lucyfirst/beef-cattle-behavior-data-set` |
| DL — Deep Learning | `level_2_dl/` | **Lumpy Skin Disease Images** (healthy vs. infected cattle photos) + accelerometer sequences from Level 1 for an LSTM | Kaggle: `warcoder/lumpy-skin-images-dataset`, Mendeley DOI `10.17632/w36hpf86j2.1` |
| NLP | `level_3_nlp/` | **Kisan Call Centre (KCC) Farmer Query Dataset** — real transcribed farmer queries to India's agricultural helpline, incl. large volumes of livestock/animal husbandry queries | data.gov.in (Government of India, Ministry of Agriculture, ICAR) |
| SLM | `level_4_slm/` | Same KCC dataset, reshaped into (query → advisory answer) pairs, used to fine-tune a small model (FLAN-T5-small / DistilBART) | data.gov.in KCC + HuggingFace `google/flan-t5-small` |
| LLD | `level_5_lld/` | N/A (design deliverable) — schemas modeled on the data above | — |
| Gen AI | `level_6_genai/` | Lumpy Skin Disease images (Level 2) used as the real seed set for augmentation via diffusion/GAN | Same as Level 2 |
| Agentic AI | `level_7_agentic/` | Consumes the outputs of Levels 1–6 as tools | — |

## Project-Wide Architecture

```
Sensors/Cameras → Ingestion API → Feature Store → [ML/DL Models] → Alert Engine
                                                        ↓
Farmer Logs/Vet Reports → NLP/SLM → Advisory Engine ────┘
                                                        ↓
                                              Agentic Orchestrator
                                          (plans, alerts, interventions)
```

## How to Run (per level)
Each `level_X/` folder is self-contained:
```
cd level_X_.../
pip install -r requirements.txt
# download the dataset per that folder's README into ./data/
python train.py
```

## Folder Map
```
livestock-ai-project/
├── README.md                  <- you are here
├── level_1_ml/                <- Random Forest / SVM baseline
├── level_2_dl/                <- CNN (image) + LSTM (sequence) deep learning
├── level_3_nlp/                <- NER + classification + summarization on farmer queries
├── level_4_slm/                <- Fine-tuned small language model for advisory generation
├── level_5_lld/                <- Low-level design: schemas, API contracts, diagrams
├── level_6_genai/              <- Synthetic image generation (diffusion) for data augmentation
├── level_7_agentic/            <- Autonomous agent orchestrating all of the above
├── realtime/                   <- Real-time telemetry simulator (SQLite-backed live feed)
└── dashboard/                  <- Streamlit operations dashboard (farmer/vet view)
```

## Dashboard
`dashboard/` is the operational front-end that ties every level together —
live telemetry, behavior monitoring, health alerts, NLP/SLM advisory
insights, Gen AI augmentation results, the LLD architecture, and the agent's
full decision trail, all in one Streamlit app.

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```
It works immediately with clearly-labeled demo data, and automatically
switches each panel to live data as you run the corresponding level's
pipeline (see `dashboard/README.md`).

## Real-Time Telemetry
There's no public live-streaming API for cattle biometrics, so `realtime/`
simulates one — a producer script that streams per-animal heart rate, body
temperature, activity, behavior, and GPS into a SQLite store every few
seconds, complete with injected fever/tachycardia/inactivity anomalies that
auto-raise alerts. The dashboard's **🔴 Live Telemetry** tab auto-refreshes
against it:

```bash
cd realtime
python producer.py          # in one terminal — leave it running
cd ../dashboard
streamlit run app.py        # in another terminal, open the Live Telemetry tab
```
See `realtime/README.md` for details and the architecture note on swapping
this for a real MQTT/Kafka ingestion path later.
