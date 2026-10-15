"""Make a simulated feeder video with a known answer key.

Birds from the held-out test split are pasted onto a backyard-ish background.
They arrive, fidget, hop, and leave, sometimes two at once. The answer key
(which species was where, and when) lets score_sim.py measure the whole
pipeline: did it find each visit, count it once, and name the species?

It is a stand-in until there's real feeder footage. Pasted photos are easier
than a real camera (sharp, well-lit, no glass or rain), so treat the score as
an upper bound.

    python scripts/simulate_feeder.py --minutes 3
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from train_vit import split_of  # noqa: E402

W, H, FPS = 1280, 720, 10
SLOTS = [(330, 300), (900, 300)]  # left and right of the feeder: where birds perch


def background():
    """Sky-to-foliage gradient, a feeder pole and tray, some blurry leaves."""
    rng = np.random.default_rng(1)
    y = np.linspace(0, 1, H)[:, None, None]
    sky, leaves = np.array([215, 200, 170]), np.array([70, 115, 80])  # BGR
    img = (sky * (1 - y) + leaves * y).repeat(W, axis=1).astype(np.float32)
    for _ in range(40):
        c = (int(rng.integers(0, W)), int(rng.integers(H // 3, H)))
        cv2.circle(img, c, int(rng.integers(20, 70)), [float(v) for v in leaves * rng.uniform(0.6, 1.2)], -1)
    img = cv2.GaussianBlur(img, (0, 0), 12)
    cv2.rectangle(img, (620, 0), (660, H), (60, 60, 70), -1)        # pole
    cv2.rectangle(img, (200, 470), (1080, 500), (40, 70, 110), -1)  # tray
    return img


def soft_mask(size, feather=0.12):
    """Alpha mask that fades at the edges so pasted photos don't look like cards."""
    ramp = np.clip(np.minimum(np.arange(size), np.arange(size)[::-1]) / (size * feather), 0, 1)
    return np.outer(ramp, ramp)[..., None]


def plan_visits(crops_by_species, seconds, rng):
    visits, t = [], 2.0
    while t < seconds - 8:
        species = rng.choice(sorted(crops_by_species))
        dur = rng.uniform(3, 9)
        slot = rng.randrange(len(SLOTS))
        visits.append({"species": species, "file": str(rng.choice(crops_by_species[species])),
                       "start_s": round(t, 2), "end_s": round(t + dur, 2), "slot": slot,
                       "size": rng.randint(170, 280)})
        if rng.random() < 0.25:  # a second bird shows up on the other side
            sp2 = rng.choice(sorted(crops_by_species))
            s2 = t + rng.uniform(0.5, 2)
            visits.append({"species": sp2, "file": str(rng.choice(crops_by_species[sp2])),
                           "start_s": round(s2, 2), "end_s": round(s2 + rng.uniform(3, 7), 2),
                           "slot": 1 - slot, "size": rng.randint(170, 280)})
        t = max(v["end_s"] for v in visits) + rng.uniform(6, 12)  # empty feeder between visits
    return visits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, default=ROOT / "data" / "training" / "crops")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "sim" / "feeder_sim.mp4")
    ap.add_argument("--minutes", type=float, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    crops_by_species = {}
    for d in sorted(p for p in args.crops.iterdir() if p.is_dir()):
        test = [p for p in sorted(d.glob("*.jpg")) if split_of(p, 15, 15) == "test"]
        if test:
            crops_by_species[d.name] = test
    visits = plan_visits(crops_by_species, args.minutes * 60, rng)

    bg = background()
    images = {v["file"]: cv2.imread(v["file"]) for v in visits}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    noise = np.random.default_rng(args.seed)
    offsets = {i: [0.0, 0.0] for i in range(len(visits))}

    for f in range(int(args.minutes * 60 * FPS)):
        t = f / FPS
        frame = bg * (1 + 0.03 * np.sin(t / 20))  # slow light change, like a passing cloud
        for i, v in enumerate(visits):
            if not v["start_s"] <= t < v["end_s"]:
                continue
            off = offsets[i]
            off[0] = float(np.clip(off[0] + rng.gauss(0, 2), -40, 40))  # fidget
            off[1] = float(np.clip(off[1] + rng.gauss(0, 1), -15, 15))
            if rng.random() < 0.01:
                off[0] += rng.choice([-30, 30])                            # hop
            s = v["size"]
            bird = cv2.resize(images[v["file"]], (s, s)).astype(np.float32)
            cx, cy = SLOTS[v["slot"]]
            x = int(np.clip(cx + off[0] - s / 2, 0, W - s))
            y = int(np.clip(cy + off[1] - s / 2, 0, H - s))
            a = soft_mask(s)
            frame[y:y + s, x:x + s] = a * bird + (1 - a) * frame[y:y + s, x:x + s]
        frame = frame + noise.normal(0, 2, frame.shape)  # sensor noise
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    writer.release()

    # Stamp the file's modified time as the "recording" start, which is what the
    # pipeline uses as t=0 for a video file. Then visit times line up with the key.
    start = datetime(2026, 10, 15, 8, 0, 0)
    os.utime(args.out, (start.timestamp(), start.timestamp()))
    key = {"video": str(args.out), "start_time": start.isoformat(), "visits": visits}
    key_path = args.out.with_suffix(".json")
    key_path.write_text(json.dumps(key, indent=2))
    print(f"wrote {args.out} ({args.minutes} min, {len(visits)} visits) and {key_path.name}")


if __name__ == "__main__":
    main()
