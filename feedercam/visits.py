"""Turn per-frame tracked detections into visits.

The tracker gives each bird an ID that persists across frames. A visit is one
track ID from the first frame it appears until it has been gone for
`end_after_s` seconds. This is what keeps one bird sitting for ten seconds from
being counted as hundreds of birds.
"""

from dataclasses import dataclass, field
from datetime import datetime

from .crops import crop


@dataclass
class Detection:
    track_id: int
    xyxy: tuple  # (x1, y1, x2, y2) in pixels
    conf: float


@dataclass
class Crop:
    score: float
    conf: float
    seen_at: datetime
    image: object  # BGR numpy array


@dataclass
class Visit:
    track_id: int
    started_at: datetime
    ended_at: datetime
    n_frames: int = 0
    max_conf: float = 0.0
    crops: list = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return (self.ended_at - self.started_at).total_seconds()


class VisitTracker:
    def __init__(self, end_after_s=4.0, min_frames=3, crops_per_visit=3,
                 crop_padding=0.15, min_crop_gap_s=1.0, min_conf=0.0):
        self.end_after_s = end_after_s
        self.min_frames = min_frames
        # The tracker is fed low-confidence boxes too (that's how ByteTrack holds
        # on to a bird through a blurry frame), so a visit only counts if at
        # least one of its frames was a confident detection.
        self.min_conf = min_conf
        self.crops_per_visit = crops_per_visit
        self.crop_padding = crop_padding
        # Consecutive frames are nearly identical, so kept crops must be spaced
        # out in time. That gives the classifier a few different poses to vote on.
        self.min_crop_gap_s = min_crop_gap_s
        self.active: dict[int, Visit] = {}

    def update(self, ts, frame, detections) -> list:
        """Record one frame's detections. Returns visits that just finished."""
        for det in detections:
            visit = self.active.get(det.track_id)
            if visit is None:
                visit = self.active[det.track_id] = Visit(det.track_id, ts, ts)
            visit.ended_at = ts
            visit.n_frames += 1
            visit.max_conf = max(visit.max_conf, det.conf)
            self._offer_crop(visit, ts, frame, det)
        return self._expire(ts)

    def flush(self) -> list:
        """End every active visit (call when the video or camera stops)."""
        done = [v for v in self.active.values() if self._counts(v)]
        self.active.clear()
        return done

    def _counts(self, visit) -> bool:
        return visit.n_frames >= self.min_frames and visit.max_conf >= self.min_conf

    def _expire(self, now) -> list:
        gone = [tid for tid, v in self.active.items()
                if (now - v.ended_at).total_seconds() > self.end_after_s]
        finished = [self.active.pop(tid) for tid in gone]
        return [v for v in finished if self._counts(v)]

    def _offer_crop(self, visit, ts, frame, det):
        # Prefer confident detections of large (close-up) birds.
        x1, y1, x2, y2 = det.xyxy
        score = det.conf * ((x2 - x1) * (y2 - y1)) ** 0.5

        near = [c for c in visit.crops
                if abs((c.seen_at - ts).total_seconds()) < self.min_crop_gap_s]
        if near:
            # Too close in time to a crop we already kept: only replace it if better.
            if score <= max(c.score for c in near):
                return
            for c in near:
                visit.crops.remove(c)
        elif len(visit.crops) >= self.crops_per_visit:
            if score <= min(c.score for c in visit.crops):
                return

        image = crop(frame, det.xyxy, self.crop_padding)
        visit.crops.append(Crop(score, det.conf, ts, image))
        visit.crops.sort(key=lambda c: c.score, reverse=True)
        del visit.crops[self.crops_per_visit:]
