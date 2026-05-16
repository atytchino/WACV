# -*- coding: utf-8 -*-
r"""
C1 Strict Trainer (NO conversion) + Class Listing + Confusion Matrix each Epoch
=============================================================================

Key properties (unchanged):
- mode = gray or rgb (choose ONE)
- No PIL convert() calls. Images are used "as-is".
- Full dataset scan before training. Raises detailed error if any image violates mode.
- Classes inferred from subfolder names in train_root; val_root must match.
- Multi-GPU via nn.DataParallel (uses all visible GPUs).
- Saves checkpoints each epoch + best + last, with meta.
- image_size can be AUTO: set --image_size 0 (default) to auto-pick a single fixed size.

New in this version:
- Prints class count + class names immediately after dataset verification.
- Prints confusion matrix after EACH epoch on VAL.
  (Also prints per-class accuracy.)

Run (PowerShell):
  python .\C1_agnostic_trainer_2026-02-04_CM.py `
    --train_root "F:/DATA/TRAIN" `
    --val_root   "F:/DATA/VAL" `
    --out_root   "F:/RUNS/c1_strict" `
    --mode gray `
    --epochs 20 --batch_size 64 --image_size 0 `
    --lr 1e-4 --workers 4 --amp `
    --log_every 50 `
    --cm_max_classes 12

Notes:
- Even with AUTO image_size, training uses ONE fixed size for batching stability.
- Confusion matrix prints full when classes <= cm_max_classes; otherwise prints top-left block.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Strict modes (no conversion):
GRAY_MODES = {"L"}     # strict grayscale only (PIL mode L)
RGB_MODES  = {"RGB"}   # strict rgb only (PIL mode RGB)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def infer_classes(root: Path) -> List[str]:
    classes = [p.name for p in root.iterdir() if p.is_dir()]
    classes.sort()
    if not classes:
        raise RuntimeError(f"No class subfolders found in: {root}")
    return classes


def list_images_under(root: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in IMG_EXTS:
                files.append(p)
    return files


def auto_choose_image_size(
    files: List[Path],
    sample_n: int = 200,
    buckets: Tuple[int, ...] = (224, 256, 320, 384, 512),
) -> int:
    """
    Chooses a single fixed image_size based on median(min(width,height)) over a sample.
    Returns nearest value in buckets.
    """
    if not files:
        return 224
    take = files if len(files) <= sample_n else random.sample(files, sample_n)
    mins: List[int] = []
    for p in take:
        try:
            with Image.open(p) as im:
                w, h = im.size
            mins.append(int(min(w, h)))
        except Exception:
            continue
    if not mins:
        return 224
    med = int(np.median(mins))
    chosen = min(buckets, key=lambda b: abs(b - med))
    print(f"[AUTO image_size] median(min_side)≈{med} -> chosen={chosen} from {list(buckets)}")
    return int(chosen)


def scan_dataset_modes(root: Path, expected: str, max_show: int = 25) -> None:
    """
    Scan every image under root recursively. Error if any image violates expected mode.
    expected: "gray" or "rgb"
    """
    if expected not in ("gray", "rgb"):
        raise ValueError("expected must be 'gray' or 'rgb'")

    bad: List[Tuple[str, str]] = []
    files = list_images_under(root)
    total = len(files)

    if total == 0:
        raise RuntimeError(f"No images found under: {root}")

    for p in files:
        try:
            with Image.open(p) as im:
                mode = im.mode
        except Exception as e:
            bad.append((str(p), f"open_error:{e}"))
            continue

        if expected == "gray":
            if mode not in GRAY_MODES:
                bad.append((str(p), f"mode={mode} (expected L)"))
        else:
            if mode not in RGB_MODES:
                bad.append((str(p), f"mode={mode} (expected RGB)"))

    if bad:
        head = "\n".join([f"  {i+1:02d}) {pp} [{info}]" for i, (pp, info) in enumerate(bad[:max_show])])
        raise RuntimeError(
            f"[DATASET MODE ERROR]\n"
            f"  root     : {root}\n"
            f"  expected : {expected}\n"
            f"  scanned  : {total} images\n"
            f"  bad      : {len(bad)} images (showing first {min(max_show, len(bad))})\n"
            f"{head}\n\n"
            f"Fix: convert/move these files OFFLINE so that the dataset is truly {expected}.\n"
            f"Trainer does NOT convert images by design."
        )

    print(f"[SCAN OK] root={root} expected={expected} images={total}")


class StrictClassFolderDataset(Dataset):
    """
    Dataset: root/<class_name>/*, returns (x, y, path).
    Opens images without conversion; checks PIL mode against expected.
    """
    def __init__(self, root: Path, classes: List[str], tfm, expected_mode: str):
        self.root = root
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.tfm = tfm
        self.expected_mode = expected_mode

        self.samples: List[Tuple[Path, int]] = []
        for c in self.classes:
            d = root / c
            if not d.exists():
                raise RuntimeError(f"Missing class folder in {root}: {c}")
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    self.samples.append((p, self.class_to_idx[c]))

        if not self.samples:
            raise RuntimeError(f"No images found in {root} for classes {self.classes}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, y = self.samples[i]
        with Image.open(path) as im:
            mode = im.mode
            if self.expected_mode == "gray":
                if mode not in GRAY_MODES:
                    raise RuntimeError(f"GRAY mode violation at runtime: {path} mode={mode}")
            else:
                if mode not in RGB_MODES:
                    raise RuntimeError(f"RGB mode violation at runtime: {path} mode={mode}")
            x = self.tfm(im)  # no conversion inside
        return x, int(y), str(path)


def build_model(num_classes: int, mode: str) -> nn.Module:
    m = models.resnet34(weights=None)
    if mode == "gray":
        m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # mode rgb uses default conv1 (3-ch)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


@torch.no_grad()
def confusion_matrix_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Vectorized confusion matrix (rows=true, cols=pred).
    logits: [B,C], target: [B]
    """
    pred = logits.argmax(dim=1)
    t = target.view(-1).to(torch.int64)
    p = pred.view(-1).to(torch.int64)
    idx = num_classes * t + p
    cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm.to(torch.int64)


@torch.no_grad()
def evaluate_with_cm(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> Tuple[float, float, torch.Tensor]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for xb, yb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = F.cross_entropy(logits, yb, reduction="mean")

        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
        loss_sum += float(loss.item()) * int(yb.numel())

        cm += confusion_matrix_from_logits(logits.detach().cpu(), yb.detach().cpu(), num_classes)

    acc = correct / max(1, total)
    avg_loss = loss_sum / max(1, total)
    return avg_loss, acc, cm


def print_class_list(classes: List[str]) -> None:
    print("\n[CLASSES] Loaded and verified")
    print(f"  num_classes = {len(classes)}")
    for i, c in enumerate(classes):
        print(f"  {i:02d}: {c}")
    print()


def print_confusion_matrix(cm: torch.Tensor, classes: List[str], title: str, max_classes: int = 12) -> None:
    C = int(cm.size(0))
    print(f"\n[CONFUSION] {title}  (rows=true, cols=pred)")
    if C <= max_classes:
        header = "        " + " ".join([f"{i:4d}" for i in range(C)])
        print(header)
        for i in range(C):
            row = " ".join([f"{int(cm[i, j]):4d}" for j in range(C)])
            print(f"{i:4d}:  {row}   | {classes[i]}")
    else:
        k = max_classes
        print(f"(showing top-left {k}x{k} of {C}x{C})")
        header = "        " + " ".join([f"{i:4d}" for i in range(k)])
        print(header)
        for i in range(k):
            row = " ".join([f"{int(cm[i, j]):4d}" for j in range(k)])
            print(f"{i:4d}:  {row}   | {classes[i]}")

    # Per-class accuracy
    diag = torch.diag(cm).to(torch.float32)
    row_sum = cm.sum(dim=1).to(torch.float32).clamp_min(1.0)
    per_acc = (diag / row_sum).cpu().numpy()
    print("\n[PER-CLASS ACC]")
    for i in range(min(C, max_classes)):
        print(f"  {i:02d} {classes[i]:>20s}: {per_acc[i]*100:6.2f}% (n={int(row_sum[i].item())})")
    if C > max_classes:
        print(f"  ... ({C-max_classes} more classes not shown)")
    print()


def save_checkpoint(
    out_dir: Path,
    name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    meta: dict
) -> None:
    ensure_dir(out_dir)
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    payload = {
        "epoch": epoch,
        "state_dict": state_dict,
        "optimizer": optimizer.state_dict(),
        "best_val_acc": float(best_val_acc),
        "meta": meta,
    }
    torch.save(payload, out_dir / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", required=True)
    ap.add_argument("--val_root", required=True)
    ap.add_argument("--out_root", required=True)

    ap.add_argument("--mode", choices=["gray", "rgb"], required=True,
                    help="Strict dataset mode. No conversion is performed.")

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=0,  # 0 -> AUTO
                    help="Fixed size for batching. Use 0 to auto-pick from dataset.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--cm_max_classes", type=int, default=12)

    args = ap.parse_args()
    seed_all(args.seed)

    train_root = Path(args.train_root)
    val_root = Path(args.val_root)
    out_root = Path(args.out_root)
    ckpt_dir = out_root / "ckpts"
    ensure_dir(out_root)
    ensure_dir(ckpt_dir)

    # infer classes from train_root subfolders
    classes = infer_classes(train_root)
    val_classes = infer_classes(val_root)
    if classes != val_classes:
        raise RuntimeError(f"Train/Val class folders mismatch:\ntrain={classes}\nval  ={val_classes}")

    print_class_list(classes)

    # strict scan (full dataset)
    scan_dataset_modes(train_root, args.mode)
    scan_dataset_modes(val_root, args.mode)

    # auto image_size if requested
    if args.image_size <= 0:
        train_files = list_images_under(train_root)
        args.image_size = auto_choose_image_size(train_files)
    print(f"[image_size] using fixed size {args.image_size}x{args.image_size} for batching")

    # transforms WITHOUT conversion
    if args.mode == "gray":
        mean, std = [0.5], [0.5]
    else:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    tfm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = StrictClassFolderDataset(train_root, classes, tfm, expected_mode=args.mode)
    val_ds   = StrictClassFolderDataset(val_root,   classes, tfm, expected_mode=args.mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} | GPUs visible: {torch.cuda.device_count()} | mode={args.mode}")

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin, drop_last=False)

    model = build_model(num_classes=len(classes), mode=args.mode).to(device)

    # Use all GPUs via DataParallel
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"[dp] Using nn.DataParallel on {torch.cuda.device_count()} GPUs")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    # logs
    (out_root / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
    csv_path = out_root / "metrics.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_sec"])

    best_val_acc = -1.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for xb, yb, _ in train_loader:
            global_step += 1
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=(args.amp and device.type == "cuda")):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                running_correct += int((pred == yb).sum().item())
                running_total += int(yb.numel())
                running_loss += float(loss.item()) * int(yb.numel())

            if args.log_every > 0 and (global_step % args.log_every == 0):
                train_acc = running_correct / max(1, running_total)
                train_loss = running_loss / max(1, running_total)
                print(f"[E{epoch:02d} step {global_step:06d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f}")

        train_loss = running_loss / max(1, running_total)
        train_acc = running_correct / max(1, running_total)

        val_loss, val_acc, cm = evaluate_with_cm(model, val_loader, device, num_classes=len(classes))

        dt = time.time() - t0
        print(f"[E{epoch:02d}] TRAIN loss={train_loss:.4f} acc={train_acc:.4f} | VAL loss={val_loss:.4f} acc={val_acc:.4f} | time={dt:.1f}s")

        print_confusion_matrix(cm, classes, title=f"VAL epoch {epoch:02d}", max_classes=args.cm_max_classes)

        meta = {
            "mode": args.mode,
            "classes": classes,
            "image_size": int(args.image_size),
            "mean": mean,
            "std": std,
        }

        # save every epoch + last
        save_checkpoint(ckpt_dir, f"c1_epoch{epoch:03d}.pth", model, optimizer, epoch, best_val_acc, meta)
        save_checkpoint(ckpt_dir, "c1_last.pth", model, optimizer, epoch, best_val_acc, meta)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(ckpt_dir, "c1_best.pth", model, optimizer, epoch, best_val_acc, meta)
            print(f"[BEST] c1_best.pth val_acc={best_val_acc:.4f}")

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.6f}",
                                    f"{val_loss:.6f}", f"{val_acc:.6f}", f"{dt:.2f}"])

    print("[DONE] C1 strict training complete.")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
