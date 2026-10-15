"""Save finished visits to the incoming folder as crops plus a JSON-lines log.

The capture side (the Pi) only ever appends to these files. Anything
downstream, such as copying visits to a laptop for species ID, reads from this
folder, so the Pi never has to share a database file with another machine.

Layout:
    incoming/crops/2026-10-15/<visit uid>_0.jpg
    incoming/visits/2026-10-15.jsonl      (one JSON object per visit)
"""

import json
from pathlib import Path
from uuid import uuid4

import cv2


def visit_uid(device, started_at) -> str:
    return f"{device}-{started_at:%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"


def save_visit(visit, incoming_dir, device) -> dict:
    incoming_dir = Path(incoming_dir)
    day = visit.started_at.strftime("%Y-%m-%d")
    uid = visit_uid(device, visit.started_at)

    crop_dir = incoming_dir / "crops" / day
    crop_dir.mkdir(parents=True, exist_ok=True)
    crops = []
    for i, c in enumerate(visit.crops):
        rel = Path("crops") / day / f"{uid}_{i}.jpg"
        cv2.imwrite(str(incoming_dir / rel), c.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        crops.append({
            "path": rel.as_posix(),
            "det_conf": round(c.conf, 4),
            "seen_at": c.seen_at.isoformat(timespec="milliseconds"),
        })

    record = {
        "uid": uid,
        "device": device,
        "track_id": visit.track_id,
        "started_at": visit.started_at.isoformat(timespec="milliseconds"),
        "ended_at": visit.ended_at.isoformat(timespec="milliseconds"),
        "n_frames": visit.n_frames,
        "det_conf": round(visit.max_conf, 4),
        "crops": crops,
    }
    # Crops are written first, so a log line never points at a missing image.
    log_dir = incoming_dir / "visits"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"{day}.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    return record
