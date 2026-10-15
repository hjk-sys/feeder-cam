"""Crop each training photo down to the animal, using the same YOLO detector.

At the feeder, the classifier only ever sees YOLO crops. Training on the same
kind of crop (rather than on whole photos with lots of background) keeps
training and real use as similar as possible.

    python training/make_crops.py

Input:  data/training/raw/<species>/*.jpg
Output: data/training/crops/<species>/*.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from feedercam.crops import crop  # noqa: E402

# COCO classes a feeder animal might be detected as. Birds should come out as
# "bird"; a squirrel usually shows up as "cat", "dog", or "bear", if at all.
ANIMALS = {"bird", "cat", "dog", "bear", "sheep"}
MAX_SIDE = 448  # the classifier uses 224px, so bigger crops just waste disk
MIN_SIDE = 40   # a bird smaller than this in a 500px photo is too blurry to learn from


def best_box(result, want_bird):
    boxes = result.boxes
    best, best_score = None, 0.0
    for xyxy, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
        name = result.names[int(cls)]
        if want_bird and name != "bird":
            continue
        x1, y1, x2, y2 = xyxy
        if max(x2 - x1, y2 - y1) < MIN_SIDE:
            continue
        score = conf * ((x2 - x1) * (y2 - y1)) ** 0.5
        if score > best_score:
            best, best_score = xyxy, score
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=ROOT / "data" / "training" / "raw", type=Path)
    ap.add_argument("--out", default=ROOT / "data" / "training" / "crops", type=Path)
    ap.add_argument("--weights", default="yolov8n.pt")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    animal_ids = [i for i, n in model.names.items() if n in ANIMALS]

    for folder in sorted(p for p in args.raw.iterdir() if p.is_dir()):
        want_bird = "squirrel" not in folder.name.lower()
        out = args.out / folder.name
        out.mkdir(parents=True, exist_ok=True)
        skip_log = out / "_skipped.txt"  # photos with no usable animal, so reruns skip them too
        skipped_before = set(skip_log.read_text().split()) if skip_log.exists() else set()
        paths = [p for p in sorted(folder.glob("*.jpg"))
                 if not (out / p.name).exists() and p.name not in skipped_before]
        detected = fallback = skipped = 0
        for start in range(0, len(paths), 32):
            batch = paths[start:start + 32]
            results = model.predict([str(p) for p in batch], conf=0.15, classes=animal_ids, verbose=False)
            for path, result in zip(batch, results):
                img = result.orig_img
                box = best_box(result, want_bird)
                if box is None and want_bird:
                    # Usually a bird too small or hidden to learn from, or a photo
                    # of a feather or nest. Leaving it out beats teaching the
                    # model that "Northern Cardinal" looks like a bush.
                    with open(skip_log, "a") as f:
                        f.write(path.name + "\n")
                    skipped += 1
                    continue
                if box is None:
                    # Squirrel photos are nearly always framed on the squirrel,
                    # and COCO has no squirrel class, so keep the whole photo.
                    h, w = img.shape[:2]
                    box = (0, 0, w, h)
                    fallback += 1
                else:
                    detected += 1
                c = crop(img, box, padding=0.15)
                if max(c.shape[:2]) > MAX_SIDE:
                    c = cv2.resize(c, (MAX_SIDE, MAX_SIDE), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out / path.name), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
        total = len(list(out.glob("*.jpg")))
        print(f"{folder.name}: {total} crops (this run: {detected} detected, "
              f"{fallback} whole-photo, {skipped} skipped)")


if __name__ == "__main__":
    main()
