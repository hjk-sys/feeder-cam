"""Tests for the parts of the pipeline that don't need a model or a camera."""

import json
from datetime import datetime, timedelta

import numpy as np

from feedercam import db
from feedercam.crops import square_box
from feedercam.motion import MotionDetector
from feedercam.store import save_visit
from feedercam.visits import Detection, VisitTracker

T0 = datetime(2026, 10, 15, 8, 0, 0)


def frame(value=100, h=360, w=640):
    return np.full((h, w, 3), value, dtype=np.uint8)


# --- motion -----------------------------------------------------------------

def test_still_scene_has_no_motion():
    m = MotionDetector()
    for _ in range(5):
        assert not m.is_motion(frame())


def test_new_object_is_motion():
    m = MotionDetector()
    m.is_motion(frame())
    f = frame()
    f[100:200, 200:300] = 255  # something bright appears
    assert m.is_motion(f)


# --- crops ------------------------------------------------------------------

def test_square_box_is_square_and_padded():
    x1, y1, x2, y2 = square_box((100, 100, 140, 120), 640, 360, padding=0.25)
    assert x2 - x1 == y2 - y1 == 60  # longest side 40 * 1.5


def test_square_box_stays_inside_frame():
    x1, y1, x2, y2 = square_box((0, 0, 50, 50), 640, 360, padding=0.5)
    assert x1 == 0 and y1 == 0 and x2 <= 640 and y2 <= 360


# --- visits -----------------------------------------------------------------

def feed(tracker, seconds, track_ids, fps=10):
    """Feed `seconds` of frames where the given track IDs are visible."""
    finished = []
    n = int(seconds * fps)
    start = feed.t
    for i in range(n):
        ts = start + timedelta(seconds=i / fps)
        dets = [Detection(t, (100, 100, 160, 150), 0.8) for t in track_ids]
        finished += tracker.update(ts, frame(), dets)
    feed.t = start + timedelta(seconds=n / fps)
    return finished


def test_one_bird_sitting_is_one_visit():
    feed.t = T0
    tr = VisitTracker(end_after_s=4, min_frames=3)
    assert feed(tr, 10, [1]) == []          # still there: nothing finished yet
    done = feed(tr, 5, [])                   # gone for 5s > end_after_s
    assert len(done) == 1
    assert done[0].n_frames == 100
    assert abs(done[0].duration_s - 9.9) < 0.01


def test_two_birds_are_two_visits():
    feed.t = T0
    tr = VisitTracker(end_after_s=2)
    feed(tr, 3, [1, 2])
    assert len(feed(tr, 3, [])) == 2


def test_flicker_is_ignored():
    feed.t = T0
    tr = VisitTracker(end_after_s=1, min_frames=3)
    feed(tr, 0.2, [7])                       # only 2 frames
    assert feed(tr, 2, []) == []


def test_crops_are_limited_and_spread_out():
    feed.t = T0
    tr = VisitTracker(end_after_s=1, crops_per_visit=3, min_crop_gap_s=1.0)
    feed(tr, 10, [1])
    visit = tr.flush()[0]
    assert len(visit.crops) == 3
    times = sorted(c.seen_at for c in visit.crops)
    gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
    assert all(g >= 1.0 for g in gaps)


# --- store + database ---------------------------------------------------------

def make_visit():
    feed.t = T0
    tr = VisitTracker(end_after_s=1)
    feed(tr, 3, [1])
    return tr.flush()[0]


def test_save_ingest_classify_correct(tmp_path):
    incoming = tmp_path / "incoming"
    rec = save_visit(make_visit(), incoming, "test-pi")
    assert all((incoming / c["path"]).exists() for c in rec["crops"])

    conn = db.connect(tmp_path / "feeder.db")
    assert db.ingest(conn, incoming) == 1
    assert db.ingest(conn, incoming) == 0    # re-running doesn't duplicate
    assert len(db.crops_for(conn, rec["uid"])) == len(rec["crops"])

    assert [r["uid"] for r in db.visits_to_classify(conn)] == [rec["uid"]]
    db.set_prediction(conn, rec["uid"], "Carolina Wren", 0.55, [["Carolina Wren", 0.55]], "v1")
    assert db.visits_to_classify(conn) == []
    assert len(db.visits_to_classify(conn, model_version="v2")) == 1

    db.set_correction(conn, rec["uid"], "House Wren")
    row = conn.execute("SELECT final_species FROM labeled_visits").fetchone()
    assert row["final_species"] == "House Wren"


def test_low_confidence_track_is_not_a_visit():
    feed.t = T0
    tr = VisitTracker(end_after_s=1, min_conf=0.9)  # feed() uses conf 0.8
    feed(tr, 3, [1])
    assert tr.flush() == []


def test_save_visit_writes_crops_and_log(tmp_path):
    rec = save_visit(make_visit(), tmp_path, "test-pi")
    assert rec["crops"] and all((tmp_path / c["path"]).exists() for c in rec["crops"])
    logged = [json.loads(l) for l in (tmp_path / "visits" / "2026-10-15.jsonl").read_text().splitlines()]
    assert logged == [rec]
    assert rec["uid"].startswith("test-pi-20261015-080000-")
