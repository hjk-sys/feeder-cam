"""Species classification with a fine-tuned ViT, run in batches on the laptop.

A visit has up to a few crops. Each crop gets a probability for every species,
and the visit's prediction is the average. Averaging several poses is more
reliable than trusting one frame where the bird might be facing away.
"""

import logging
import shutil

import torch
from PIL import Image
from torchvision import transforms

from . import db
from .config import resolve

log = logging.getLogger(__name__)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    return "cpu"


def eval_transform(img_size, mean, std):
    # Crops are already square, so a plain resize doesn't distort them.
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class SpeciesClassifier:
    def __init__(self, checkpoint_path, device=None):
        import timm

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.classes = ckpt["classes"]
        self.version = ckpt["version"]
        self.device = device or pick_device()
        self.model = timm.create_model(ckpt["arch"], pretrained=False, num_classes=len(self.classes))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(self.device)
        self.transform = eval_transform(ckpt["img_size"], ckpt["mean"], ckpt["std"])

    @torch.no_grad()
    def predict(self, images) -> torch.Tensor:
        """PIL images in, (N, num_classes) probabilities out."""
        batch = torch.stack([self.transform(im.convert("RGB")) for im in images]).to(self.device)
        return self.model(batch).softmax(dim=1).cpu()


def classify_pending(conn, cfg, redo=False) -> int:
    """Classify visits that have no prediction (or, with redo, an older model's)."""
    clf_cfg = cfg["classifier"]
    clf = SpeciesClassifier(resolve(clf_cfg["checkpoint"]))
    incoming = resolve(cfg["paths"]["incoming"])
    pending = db.visits_to_classify(conn, clf.version if redo else None)
    log.info("model %s on %s: %d visits to classify", clf.version, clf.device, len(pending))

    done = 0
    chunk = max(1, clf_cfg["batch_size"] // cfg["visits"]["crops_per_visit"])
    for start in range(0, len(pending), chunk):
        images, owners = [], []
        for row in pending[start:start + chunk]:
            for crop in db.crops_for(conn, row["uid"]):
                path = incoming / crop["path"]
                if path.exists():  # crops may not have synced from the Pi yet
                    images.append(Image.open(path))
                    owners.append(row["uid"])
        if not images:
            continue
        probs = clf.predict(images)

        for uid in dict.fromkeys(owners):  # unique, in order
            idx = [i for i, o in enumerate(owners) if o == uid]
            mean = probs[idx].mean(dim=0)
            top = mean.topk(min(3, len(clf.classes)))
            top3 = [[clf.classes[i], round(p, 4)] for p, i in zip(top.values.tolist(), top.indices.tolist())]
            db.set_prediction(conn, uid, top3[0][0], top3[0][1], top3, clf.version)
            done += 1
        conn.commit()
    return done


SKIP_LABELS = {"Unknown"}


def export_corrections(conn, cfg) -> int:
    """Copy crops from reviewed visits into training/corrections/<label>/.

    Retraining with these is what improves the model on your own feeder's
    lighting, angles, and regulars.
    """
    incoming = resolve(cfg["paths"]["incoming"])
    out_root = resolve(cfg["paths"]["training"]) / "corrections"
    rows = conn.execute(
        """SELECT c.path, v.corrected_species FROM crops c
           JOIN visits v ON v.uid = c.visit_uid
           WHERE v.corrected_species IS NOT NULL"""
    ).fetchall()
    copied = 0
    for row in rows:
        label = row["corrected_species"]
        src = incoming / row["path"]
        if not src.exists():
            continue
        # If a label was changed after an earlier export, drop the stale copy.
        for old in out_root.glob(f"*/{src.name}"):
            if old.parent.name != label:
                old.unlink()
        if label in SKIP_LABELS:
            continue
        dest = out_root / label / src.name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied += 1
    return copied
