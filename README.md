# feeder-cam

A Raspberry Pi camera watches a backyard bird feeder in North Carolina. A YOLOv8
detector finds each bird, ByteTrack follows it so one visit counts once, and a
fine-tuned Vision Transformer identifies the species. Everything lands in a
SQLite log with a Streamlit dashboard for counts and for correcting the model.

```
 Raspberry Pi                                     Laptop
 ───────────────────────────────────              ─────────────────────────────────────
 camera ─▶ motion gate ─▶ YOLOv8n + ByteTrack     rsync ─▶ ingest ─▶ ViT classifier
                              │                                 │            │
                     visits + best crops ───────────────▶  SQLite log ◀──────┘
                     (data/incoming)                            │
                                                     Streamlit dashboard
                                                                │
                                         review low-confidence ─┴─▶ retrain
```

## Status

| Stage | State |
|---|---|
| Capture (Pi camera / USB / video file) + motion gate | written, tested on video; not yet run on the Pi |
| Detection + tracking (YOLOv8n + ByteTrack) | working on simulated footage; thresholds not yet tuned on real feeder video |
| Visit logging (crops + JSONL on the Pi, SQLite on the laptop) | working, unit tested |
| Species classifier (ViT-Small, 30 NC classes) | code + data ready; first full training run not finished |
| Dashboard + review queue | working on simulated data |
| Retraining from corrections | written; waiting on real feeder data |
| Real feeder footage | **not yet**: next step |

## Quick start (laptop)

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
```

### Try detection on a video or webcam

```bash
# Run detection on any video; writes visits to data/incoming and a boxed copy of the video
.venv/bin/python -m feedercam run --source some_clip.mp4 --annotate boxed.mp4

# Or watch a webcam live (press q to quit)
.venv/bin/python -m feedercam --set camera.source=usb run --preview
```

The first run downloads `yolov8n.pt` (6 MB), YOLOv8's smallest pretrained
model. It already knows "bird" from the COCO dataset, so no training is needed
for detection.

### Train the species model

```bash
.venv/bin/python training/fetch_inat.py --per-species 200   # CC-licensed iNaturalist photos
.venv/bin/python training/make_crops.py                     # crop each photo to the bird with YOLO
.venv/bin/python training/train_vit.py --epochs 10          # fine-tune ViT, writes models/species_vit.pt
```

The species list is [training/species_nc.txt](training/species_nc.txt). Edit
it to match your feeder; eBird's bar chart for your county shows what's around.

### Test the whole pipeline without a feeder

```bash
scripts/run_sim.sh 3
```

This builds a 3-minute video of held-out birds visiting a fake feeder, runs
detection, tracking, logging, and classification on it, and scores the result
against the answer key. It's easier than a real camera (sharp photos, no glass,
no rain), so treat the score as an upper bound.

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

The Pi only runs capture and detection, writing each finished visit to
`data/incoming/` (a few crops plus one line of JSON). Species classification
runs on the laptop, because a ViT is too slow for the Pi's CPU and nobody needs
the species name the instant a bird lands.

### Output

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

## The daily loop (laptop)

```bash
scripts/sync.sh pi@feeder.local        # rsync new visits, ingest, classify, print counts
.venv/bin/streamlit run dashboard/app.py
```

In the dashboard's **Review** tab, fix predictions the model wasn't sure about.
Then:

```bash
.venv/bin/python -m feedercam export   # reviewed crops -> data/training/corrections/
.venv/bin/python training/train_vit.py # retrain with them included
.venv/bin/python -m feedercam classify --redo
```

## Design notes

- **Visits, not frames.** A bird sitting for ten seconds is about 100 frames.
  ByteTrack gives it one ID across those frames, and a visit ends when that ID
  has been gone for 4 seconds. Counting is only as good as the tracking, so the
  simulator reports "counted more than once" separately.
- **Weak boxes go to the tracker.** The detector threshold is low (0.1) because
  ByteTrack uses low-confidence boxes to hold on to a bird through a blurry or
  half-hidden frame. A visit only counts if at least one of its frames was
  confident (0.4), so weak boxes on their own never create a visit.
- **Motion gate.** The detector only runs while something moves, for a few
  seconds afterwards, and while a tracked bird is still present (a bird sitting
  still makes no motion but is still visiting).
- **Several crops per visit.** Up to three crops, at least a second apart,
  ranked by detector confidence and size. The classifier averages over them,
  which helps when the bird is facing away in one of them.
- **Train on what the model will see.** Training photos are cropped by the same
  YOLO model that crops feeder frames. Photos where YOLO finds no usable bird
  (too small, hidden in leaves, a feather, a screenshot) are skipped instead of
  being used whole.
- **Stable splits.** Train/val/test membership comes from a hash of each
  filename, so adding photos later never moves a test image into training.
- **Squirrels.** COCO has no squirrel class and YOLO sometimes calls one a
  bird, so "Eastern Gray Squirrel" is a classifier class.

## Results

_To be filled in after the first training run._

## Project layout

```
feedercam/        the pipeline package (python -m feedercam ...)
  sources.py        Pi camera, USB webcam, or video file
  motion.py         background-subtraction motion gate
  detect.py         YOLOv8 + ByteTrack
  visits.py         tracks -> visits, picks the best crops
  store.py          visits -> crops + JSONL (Pi side)
  db.py             SQLite log (laptop side)
  classify.py       ViT inference, correction export
  pipeline.py       the capture loop
training/         data download, cropping, ViT fine-tuning
dashboard/        Streamlit app
pi/               Pi install script + systemd unit
scripts/          sync from the Pi, feeder simulator + scorer
tests/            unit tests (no camera or model needed)
config.yaml       all settings
```

## Data and credits

Training photos come from [iNaturalist](https://www.inaturalist.org) research-grade
observations under CC0, CC-BY, and CC-BY-NC licenses. They are not committed
to this repo. `training/fetch_inat.py` re-downloads them and records each
photographer's credit in `data/training/attribution.csv`.

Built with Ultralytics YOLOv8, PyTorch, timm, and Streamlit. 
