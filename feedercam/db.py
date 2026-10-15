"""The SQLite visit log on the laptop.

`species` is the model's guess. `corrected_species` is set when you fix a
prediction in the dashboard. The `labeled_visits` view gives the final answer
(the correction if there is one) and is what the counts should use.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS visits (
    uid               TEXT PRIMARY KEY,
    device            TEXT,
    started_at        TEXT NOT NULL,
    ended_at          TEXT NOT NULL,
    n_frames          INTEGER,
    det_conf          REAL,
    species           TEXT,
    species_conf      REAL,
    top3              TEXT,           -- JSON list of [species, prob]
    model_version     TEXT,
    classified_at     TEXT,
    corrected_species TEXT,
    reviewed_at       TEXT
);

CREATE TABLE IF NOT EXISTS crops (
    id        INTEGER PRIMARY KEY,
    visit_uid TEXT NOT NULL REFERENCES visits(uid),
    path      TEXT NOT NULL UNIQUE,   -- relative to the incoming folder
    det_conf  REAL,
    seen_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_visits_started ON visits(started_at);
CREATE INDEX IF NOT EXISTS idx_crops_visit ON crops(visit_uid);

CREATE VIEW IF NOT EXISTS labeled_visits AS
SELECT *,
       COALESCE(corrected_species, species) AS final_species,
       date(started_at) AS day
FROM visits;
"""


def connect(db_path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def ingest(conn, incoming_dir) -> int:
    """Load every visit in incoming/visits/*.jsonl. Safe to run repeatedly."""
    added = 0
    for log in sorted(Path(incoming_dir, "visits").glob("*.jsonl")):
        for line in log.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            cur = conn.execute(
                """INSERT OR IGNORE INTO visits
                   (uid, device, started_at, ended_at, n_frames, det_conf)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (rec["uid"], rec["device"], rec["started_at"], rec["ended_at"],
                 rec["n_frames"], rec["det_conf"]),
            )
            added += cur.rowcount
            conn.executemany(
                """INSERT OR IGNORE INTO crops (visit_uid, path, det_conf, seen_at)
                   VALUES (?, ?, ?, ?)""",
                [(rec["uid"], c["path"], c["det_conf"], c["seen_at"]) for c in rec["crops"]],
            )
    conn.commit()
    return added


def visits_to_classify(conn, model_version=None):
    """Visits with no prediction yet, or (if model_version is given) an older model's."""
    if model_version:
        return conn.execute(
            "SELECT uid FROM visits WHERE model_version IS NOT ? ORDER BY started_at",
            (model_version,),
        ).fetchall()
    return conn.execute(
        "SELECT uid FROM visits WHERE species IS NULL ORDER BY started_at"
    ).fetchall()


def crops_for(conn, uid):
    return conn.execute(
        "SELECT path, det_conf FROM crops WHERE visit_uid = ? ORDER BY id", (uid,)
    ).fetchall()


def set_prediction(conn, uid, species, conf, top3, model_version):
    conn.execute(
        """UPDATE visits SET species = ?, species_conf = ?, top3 = ?,
                  model_version = ?, classified_at = ?
           WHERE uid = ?""",
        (species, conf, json.dumps(top3), model_version,
         datetime.now().isoformat(timespec="seconds"), uid),
    )


def set_correction(conn, uid, species):
    """Record a human label. Pass species=None to clear a correction."""
    reviewed = datetime.now().isoformat(timespec="seconds") if species else None
    conn.execute(
        "UPDATE visits SET corrected_species = ?, reviewed_at = ? WHERE uid = ?",
        (species, reviewed, uid),
    )
    conn.commit()
