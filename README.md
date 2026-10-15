# feeder-cam

A Raspberry Pi camera watches a backyard bird feeder in North Carolina. A YOLOv8
detector finds each bird, and ByteTrack follows it across frames so that one
bird sitting at the feeder is logged as one visit, not hundreds of frames.

```
 Raspberry Pi
 ─────────────────────────────────────────────────────────────────────
 camera ─▶ motion gate ─▶ YOLOv8n + ByteTrack ─▶ visits ─▶ crops + JSONL log
                                                            (data/incoming)
```

## Status

| Stage | State |
|---|---|
| Capture (Pi camera / USB webcam / video file) | written; tested on video files, not yet on the Pi |
| Motion gate | written, unit tested |
| Detection + tracking (YOLOv8n + ByteTrack) | working on simulated test footage; thresholds not yet tuned on real feeder video |
| Visit logging (best crops + one JSON line per visit) | working, unit tested |
| Species identification | planned |

## Try it on a laptop

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest

# Run detection on any video; writes visits to data/incoming and a boxed copy of the video
.venv/bin/python -m feedercam run --source some_clip.mp4 --annotate boxed.mp4

# Or watch a webcam live (press q to quit)
.venv/bin/python -m feedercam --set camera.source=usb run --preview
```

The first run downloads `yolov8n.pt` (6 MB), YOLOv8's smallest pretrained
model. It already knows "bird" from the COCO dataset, so no training is needed
for detection.

## Running on the Raspberry Pi

```bash
git clone <this repo> feeder-cam && cd feeder-cam
bash pi/install.sh                     # picamera2, venv, NCNN export, systemd unit
.venv/bin/python -m feedercam --config config.pi.yaml run    # try it in the foreground
sudo systemctl enable --now feedercam  # then run it on boot
journalctl -u feedercam -f             # watch the log
```

`install.sh` exports YOLOv8n to NCNN, which runs several times faster than
PyTorch on the Pi's CPU, and writes a `config.pi.yaml` that uses it. If the Pi
still can't keep up, raise `detector.every_n_frames` to 2 or 3.

## Output

Each finished visit becomes up to three crops and one line of JSON:

```
data/incoming/crops/2026-10-15/feeder-pi-20261015-080020-7005bd_0.jpg
data/incoming/visits/2026-10-15.jsonl
```

```json
{"uid": "feeder-pi-20261015-080020-7005bd", "device": "feeder-pi", "track_id": 4,
 "started_at": "2026-10-15T08:00:20.100", "ended_at": "2026-10-15T08:00:26.100",
 "n_frames": 61, "det_conf": 0.81, "crops": [{"path": "crops/2026-10-15/...", ...}]}
```

## Design notes

- **Visits, not frames.** A bird sitting for ten seconds is about 100 frames.
  ByteTrack gives it one ID across those frames, and a visit ends when that ID
  has been gone for 4 seconds.
- **Weak boxes go to the tracker.** The detector threshold is low (0.1) because
  ByteTrack uses low-confidence boxes to hold on to a bird through a blurry or
  half-hidden frame. A visit only counts if at least one of its frames was
  confident (0.4), so weak boxes on their own never create a visit.
- **Motion gate.** The detector only runs while something moves, for a few
  seconds afterwards, and while a tracked bird is still present (a bird sitting
  still makes no motion but is still visiting).
- **Best crops per visit.** Up to three crops, at least a second apart, ranked
  by detector confidence and size. Square and padded, ready for a classifier.
- **The Pi only appends files.** Nothing on the Pi is a shared database, so
  copying `data/incoming/` elsewhere (e.g. with rsync) is always safe.

## Project layout

```
feedercam/        the pipeline package (python -m feedercam run ...)
  sources.py        Pi camera, USB webcam, or video file
  motion.py         background-subtraction motion gate
  detect.py         YOLOv8 + ByteTrack
  visits.py         tracks -> visits, picks the best crops
  store.py          visits -> crops + JSONL
  pipeline.py       the capture loop
  config.py         loads config.yaml (+ --set overrides)
pi/               Pi install script + systemd unit
tests/            unit tests (no camera or model needed)
config.yaml       all settings
```

Built with Ultralytics YOLOv8, OpenCV, and picamera2. Written with help from
Claude (Anthropic).
