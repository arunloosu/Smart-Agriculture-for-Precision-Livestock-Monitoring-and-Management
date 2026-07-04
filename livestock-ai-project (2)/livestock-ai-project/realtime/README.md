# Real-Time Telemetry (Live Simulator)

There is no public live-streaming API for cattle biometrics, so this module
simulates one realistically, matching the ingestion architecture described in
`level_5_lld/LLD.md` (`Sensors/Cameras → Ingestion API → Feature Store`).
Swap `producer.py`'s write calls for a real MQTT/Kafka consumer later and the
dashboard needs no changes — it only talks to `telemetry_db.py`.

## What it simulates
For 20 animals, every tick (default every 3s):
- **Heart rate** (bpm) and **body temperature** (°C) — smooth per-animal
  random walk around clinically normal ranges (48–84 bpm, 37.8–39.2°C).
- **Behavior** — a state machine (grazing/lying/standing/walking/ruminating)
  with realistic transition probabilities and dwell times.
- **GPS position** — wanders inside a fixed farm geofence.
- **Injected anomalies** — occasional fever, tachycardia, or prolonged
  inactivity episodes that cross clinical thresholds and automatically write
  a row to `live_alerts` (severity: medium/high/critical).

Everything is written to `telemetry.db` (SQLite, WAL mode) so the Streamlit
dashboard can read it concurrently while the producer keeps writing.

## Run it

```bash
cd realtime
pip install -r ../dashboard/requirements.txt   # no extra deps needed (sqlite3 is stdlib)
python producer.py                 # streams forever, ~1 reading/animal/3s
python producer.py --interval 1    # faster, good for a live demo
python producer.py --duration 120  # auto-stop after 2 minutes
python producer.py --reset         # wipe telemetry.db and start clean
```

Leave it running in a terminal, then in another terminal:

```bash
cd ../dashboard
streamlit run app.py
```

Open the **🔴 Live Telemetry** tab — it auto-refreshes every few seconds and
reflects whatever the producer is currently writing. Stop the producer with
Ctrl+C at any point; the dashboard tab will simply stop updating and label
the data as stale rather than erroring out.

## Files
| File | Purpose |
|---|---|
| `telemetry_db.py` | SQLite schema + read/write helpers (`readings`, `live_alerts` tables) |
| `producer.py` | Standalone script simulating the live sensor feed |
| `telemetry.db` | Created on first run; safe to delete any time |
