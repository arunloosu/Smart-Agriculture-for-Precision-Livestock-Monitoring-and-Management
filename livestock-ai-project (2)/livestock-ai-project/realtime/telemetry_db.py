"""
telemetry_db.py
----------------
Lightweight SQLite-backed store for real-time livestock telemetry.

Why SQLite (and not just an in-memory queue)?
  - The producer (simulating sensor/edge-gateway ingestion) and the Streamlit
    dashboard (the consumer) are separate processes. SQLite with WAL mode
    lets one process write while another reads concurrently, with zero
    external infra (no Kafka/Redis needed) — appropriate for a course/demo
    project, and it's a drop-in replacement point: swap this module for a
    real MQTT/Kafka consumer later without touching the dashboard code.

Schema
  readings(animal_id, ts, heart_rate, body_temp_c, activity_index,
           lat, lon, behavior, is_anomaly)
  live_alerts(id, animal_id, ts, severity, condition, message, acknowledged)
"""
import os
import sqlite3
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "telemetry.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    animal_id       TEXT NOT NULL,
    ts              TEXT NOT NULL,
    heart_rate      REAL NOT NULL,
    body_temp_c     REAL NOT NULL,
    activity_index  REAL NOT NULL,
    lat             REAL NOT NULL,
    lon             REAL NOT NULL,
    behavior        TEXT NOT NULL,
    is_anomaly      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_readings_ts ON readings (ts);
CREATE INDEX IF NOT EXISTS idx_readings_animal ON readings (animal_id);

CREATE TABLE IF NOT EXISTS live_alerts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    animal_id     TEXT NOT NULL,
    ts            TEXT NOT NULL,
    severity      TEXT NOT NULL,
    condition     TEXT NOT NULL,
    message       TEXT NOT NULL,
    acknowledged  INTEGER NOT NULL DEFAULT 0
);
"""


@contextmanager
def get_conn(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str = DB_PATH):
    with get_conn(db_path) as conn:
        conn.executescript(_SCHEMA)


def insert_reading(reading: dict, db_path: str = DB_PATH):
    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO readings
               (animal_id, ts, heart_rate, body_temp_c, activity_index, lat, lon, behavior, is_anomaly)
               VALUES (:animal_id, :ts, :heart_rate, :body_temp_c, :activity_index, :lat, :lon, :behavior, :is_anomaly)""",
            reading,
        )


def insert_alert(alert: dict, db_path: str = DB_PATH):
    with get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO live_alerts (animal_id, ts, severity, condition, message, acknowledged)
               VALUES (:animal_id, :ts, :severity, :condition, :message, 0)""",
            alert,
        )


def fetch_recent_readings(minutes: int = 30, db_path: str = DB_PATH):
    if not os.path.exists(db_path):
        return []
    with get_conn(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM readings WHERE ts >= datetime('now', ?) ORDER BY ts ASC",
            (f"-{minutes} minutes",),
        )
        return [dict(r) for r in cur.fetchall()]


def fetch_latest_per_animal(db_path: str = DB_PATH):
    if not os.path.exists(db_path):
        return []
    with get_conn(db_path) as conn:
        cur = conn.execute(
            """SELECT r.* FROM readings r
               INNER JOIN (
                   SELECT animal_id, MAX(ts) AS max_ts FROM readings GROUP BY animal_id
               ) latest ON r.animal_id = latest.animal_id AND r.ts = latest.max_ts"""
        )
        return [dict(r) for r in cur.fetchall()]


def fetch_recent_alerts(limit: int = 50, db_path: str = DB_PATH):
    if not os.path.exists(db_path):
        return []
    with get_conn(db_path) as conn:
        cur = conn.execute(
            "SELECT * FROM live_alerts ORDER BY ts DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in cur.fetchall()]


def row_count(db_path: str = DB_PATH) -> int:
    if not os.path.exists(db_path):
        return 0
    with get_conn(db_path) as conn:
        cur = conn.execute("SELECT COUNT(*) AS n FROM readings")
        return cur.fetchone()["n"]
