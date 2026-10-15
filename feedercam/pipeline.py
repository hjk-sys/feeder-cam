"""The capture loop: frames -> motion gate -> detect + track -> visits -> disk.

This is what runs on the Pi. It also runs on a laptop against a video file,
which is how the pipeline is tested without the feeder.
"""

import logging
from collections import Counter

import cv2

from .config import resolve
from .detect import BirdDetector
from .motion import MotionDetector
from .sources import open_source
from .store import save_visit
from .visits import VisitTracker

log = logging.getLogger(__name__)


def run(cfg, source=None, preview=False, annotate_path=None, max_seconds=None) -> Counter:
    cam_cfg, mot_cfg, det_cfg, vis_cfg = cfg["camera"], cfg["motion"], cfg["detector"], cfg["visits"]
    incoming = resolve(cfg["paths"]["incoming"])
    device_name = cfg["device_name"]

    frames = open_source(cam_cfg, source)
    detector = BirdDetector(
        det_cfg["weights"], det_cfg["imgsz"], det_cfg["conf"], det_cfg["tracker"],
        det_cfg.get("device"),
    )
    motion = None
    if mot_cfg["enabled"]:
        motion = MotionDetector(mot_cfg["downscale_width"], mot_cfg["threshold"], mot_cfg["min_area_frac"])
    visits = VisitTracker(
        vis_cfg["end_after_s"], vis_cfg["min_frames"], vis_cfg["crops_per_visit"], vis_cfg["crop_padding"],
        min_conf=vis_cfg.get("min_conf", 0.0),
    )

    stats = Counter()
    writer = None
    first_ts = last_motion = None

    def finish(visit):
        rec = save_visit(visit, incoming, device_name)
        stats["visits"] += 1
        log.info("visit %s: %.1fs, %d frames, det conf %.2f, %d crops",
                 rec["uid"], visit.duration_s, visit.n_frames, visit.max_conf, len(visit.crops))

    log.info("capture started; writing visits to %s", incoming)
    try:
        for i, (ts, frame) in enumerate(frames):
            first_ts = first_ts or ts
            if max_seconds and (ts - first_ts).total_seconds() > max_seconds:
                break
            stats["frames"] += 1

            # Run the detector while something moves, for a cooldown afterwards,
            # and while any bird is still being tracked (a bird sitting still
            # makes no motion but is still visiting).
            moving = motion is None or motion.is_motion(frame)
            if moving:
                last_motion = ts
            recently_moved = last_motion and (ts - last_motion).total_seconds() < mot_cfg["cooldown_s"]
            awake = moving or recently_moved or bool(visits.active)

            detections = []
            if awake and i % det_cfg["every_n_frames"] == 0:
                detections = detector.track(frame)
                stats["frames_detected"] += 1
                stats["detections"] += len(detections)

            for visit in visits.update(ts, frame, detections):
                finish(visit)

            if preview or annotate_path:
                shown = _draw(frame, detections, awake, ts)
                if annotate_path:
                    if writer is None:
                        h, w = shown.shape[:2]
                        fps = getattr(frames, "fps", cam_cfg["fps"])
                        writer = cv2.VideoWriter(str(annotate_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    writer.write(shown)
                if preview:
                    cv2.imshow("feeder-cam", shown)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        for visit in visits.flush():
            finish(visit)
        frames.close()
        if writer is not None:
            writer.release()
        if preview:
            cv2.destroyAllWindows()

    log.info("done: %d frames, detector ran on %d, %d visits saved",
             stats["frames"], stats["frames_detected"], stats["visits"])
    return stats


def _draw(frame, detections, awake, ts):
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(out, f"bird #{d.track_id} {d.conf:.2f}", (x1, max(y1 - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    status = "DETECTING" if awake else "idle"
    cv2.putText(out, f"{ts:%H:%M:%S}  {status}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out
