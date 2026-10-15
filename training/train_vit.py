"""Fine-tune a pretrained Vision Transformer to recognize feeder species.

The ViT starts from ImageNet-21k weights, so it already knows general visual
features like feathers, beaks, and color patches. Fine-tuning only has to teach
it to tell these ~30 classes apart, which works with a few hundred images each.

    python training/train_vit.py --epochs 10

Reads class folders from data/training/crops and data/training/corrections
(your reviewed feeder crops), and writes models/species_vit.pt plus a metrics
JSON with the training curve and per-species test accuracy.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from feedercam.classify import eval_transform, pick_device  # noqa: E402
from feedercam.config import load_config, resolve  # noqa: E402


def split_of(path, val_pct, test_pct):
    """Assign an image to train/val/test by hashing its filename.

    Unlike a random shuffle, this never moves an image between splits when
    new images are added later, so a test image never leaks into training.
    """
    h = int(hashlib.md5(path.name.encode()).hexdigest(), 16) % 100
    if h < test_pct:
        return "test"
    if h < test_pct + val_pct:
        return "val"
    return "train"


def gather(roots, val_pct, test_pct):
    classes = sorted({d.name for r in roots if r.exists() for d in r.iterdir()
                      if d.is_dir() and not d.name.startswith((".", "_"))})
    index = {c: i for i, c in enumerate(classes)}
    splits = defaultdict(list)
    for r in roots:
        for c in classes:
            for p in sorted((r / c).glob("*.jpg")) if (r / c).exists() else []:
                splits[split_of(p, val_pct, test_pct)].append((p, index[c]))
    return classes, splits


class Images(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), label


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds += model(x.to(device)).argmax(1).cpu().tolist()
        labels += y.tolist()
    return preds, labels


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", type=Path, default=[
        resolve(cfg["paths"]["training"]) / "crops",
        resolve(cfg["paths"]["training"]) / "corrections",
    ])
    ap.add_argument("--arch", default=cfg["classifier"]["arch"])
    ap.add_argument("--out", type=Path, default=resolve(cfg["classifier"]["checkpoint"]))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5, help="backbone learning rate (the new head gets 10x)")
    ap.add_argument("--val-pct", type=int, default=15)
    ap.add_argument("--test-pct", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    import timm

    torch.manual_seed(0)
    device = pick_device()
    classes, splits = gather(args.data, args.val_pct, args.test_pct)
    print(f"{len(classes)} classes | train {len(splits['train'])} | val {len(splits['val'])} "
          f"| test {len(splits['test'])} | device {device}")

    model = timm.create_model(args.arch, pretrained=True, num_classes=len(classes))
    data_cfg = timm.data.resolve_data_config(model=model)
    img_size, mean, std = data_cfg["input_size"][-1], list(data_cfg["mean"]), list(data_cfg["std"])

    # Augmentation: small zooms, flips, and lighting changes, roughly what
    # varies between feeder visits. No vertical flips; birds aren't upside down
    # (except nuthatches, which manage fine without help).
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = eval_transform(img_size, mean, std)

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers,
                     persistent_workers=args.workers > 0)
    train_dl = DataLoader(Images(splits["train"], train_tf), shuffle=True, drop_last=True, **loader_kw)
    val_dl = DataLoader(Images(splits["val"], eval_tf), **loader_kw)
    test_dl = DataLoader(Images(splits["test"], eval_tf), **loader_kw)

    model.to(device)
    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([
        {"params": backbone, "lr": args.lr},
        {"params": head, "lr": args.lr * 10},  # the head starts random, so it learns faster
    ], weight_decay=0.05)

    # One epoch of linear warmup, then cosine decay to zero.
    total_steps = args.epochs * len(train_dl)
    warmup = len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    history, best_acc, best_state = [], -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, loss_sum, correct, seen = time.time(), 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        preds, labels = evaluate(model, val_dl, device)
        val_acc = sum(p == l for p, l in zip(preds, labels)) / max(1, len(labels))
        history.append({"epoch": epoch, "train_loss": round(loss_sum / seen, 4),
                        "train_acc": round(correct / seen, 4), "val_acc": round(val_acc, 4)})
        print(f"epoch {epoch:2d}  loss {loss_sum / seen:.3f}  train acc {correct / seen:.3f}  "
              f"val acc {val_acc:.3f}  ({time.time() - t0:.0f}s)")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Final score on the test split, which played no part in choosing the model.
    model.load_state_dict(best_state)
    preds, labels = evaluate(model, test_dl, device)
    test_acc = sum(p == l for p, l in zip(preds, labels)) / max(1, len(labels))
    per_class = {}
    for i, c in enumerate(classes):
        idx = [k for k, l in enumerate(labels) if l == i]
        per_class[c] = {"n": len(idx), "acc": round(sum(preds[k] == i for k in idx) / max(1, len(idx)), 3)}
    confusions = Counter((classes[l], classes[p]) for p, l in zip(preds, labels) if p != l)
    print(f"\ntest accuracy {test_acc:.3f} on {len(labels)} images")
    print("most confused (true -> predicted):")
    for (t, p), n in confusions.most_common(8):
        print(f"  {n:3d}  {t} -> {p}")

    version = f"vit-{datetime.now():%Y%m%d-%H%M}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "classes": classes, "arch": args.arch,
                "img_size": img_size, "mean": mean, "std": std, "version": version}, args.out)
    metrics = {
        "version": version, "arch": args.arch, "classes": classes,
        "sizes": {k: len(v) for k, v in splits.items()},
        "history": history, "best_val_acc": round(best_acc, 4), "test_acc": round(test_acc, 4),
        "per_class": per_class,
        "confusions": [{"true": t, "pred": p, "n": n} for (t, p), n in confusions.most_common()],
    }
    args.out.with_name(args.out.stem + "_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nsaved {args.out} ({version})")


if __name__ == "__main__":
    main()
