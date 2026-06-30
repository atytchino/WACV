#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
C1 Trainer (TRAINER-COMPATIBLE v2)
====================================

CRITICAL: This version uses the *exact* ResNet34LF_BN class from the watermark
trainer (Trainer_MULTIBIT.py lines 765-859). Previous version only matched
state_dict keys but had wrong forward signature, causing NaN C1 outputs during
watermark training.

Differences from previous version:
  - Includes wm_head Sequential module (AdaptiveAvgPool2d, Flatten, Linear, ReLU, Linear)
  - Has wm_affine as nn.Parameter (shape [num_classes]) instead of scalar buffer
  - Forward accepts (x, gate=True, gate_target=None, detach_gate=False,
    detach_affine=False, return_raw=False) and returns 3-tuple (logits, wm_logit, x4)
  - Uses base.conv1.stride = (1, 1)  (different from torchvision default)
  - _wrap_blur modifies existing downsample Sequential (changes Conv stride 2→1,
    inserts BlurPool after) — different from previous version's replacement approach

During C1 training:
  - Forward is called with gate=False (only training the base classifier)
  - wm_head and wm_affine receive no training signal — they remain at init
  - This is fine: trainer uses them during watermark training, not during
    C1 pre-training

Usage:
  python C1_trainer_compatible.py `
    --train_root "E:\TLD\train" `
    --val_root   "E:\TLD\val" `
    --out_root   "E:\C1_TRAINED\TLD" `
    --epochs 15 --batch_size 16 --image_size 512 `
    --workers 2 --lr 1e-4 --weight_decay 1e-4 --amp `
    --log_every 50 --seed 1337
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as tv_models


# ============================================================================
# Module-level transform (must be picklable for multiprocessing dataloaders).
# Watermark trainer feeds C1 inputs as xN = x01 * 2 - 1 (range [-1, 1]).
# We train C1 on the SAME range, otherwise BN running stats are calibrated
# to [0, 1] and accuracy drops to ~20% when called from trainer with [-1, 1].
# See Trainer_MULTIBIT.py line 212 and line 4156.
# ============================================================================

class ToTensorMinusOneOne:
    """Convert PIL image to [-1, 1] tensor, matching watermark trainer's xN."""
    def __call__(self, pil):
        t = transforms.functional.to_tensor(pil)  # [0, 1]
        return t * 2.0 - 1.0  # [-1, 1]


# ============================================================================
# BlurPool — identical to trainer's lines 644-654
# ============================================================================

class BlurPool(nn.Module):
    def __init__(self, ch: int, filt=(1, 2, 1)):
        super().__init__()
        f = torch.tensor(filt, dtype=torch.float32)
        k = (f[:, None] * f[None, :])
        k = (k / k.sum()).view(1, 1, 3, 3).repeat(ch, 1, 1, 1)
        self.register_buffer("k", k)
        self.groups = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.k, stride=2, padding=1, groups=self.groups)


# ============================================================================
# ResNet34LF_BN — EXACT copy of trainer's class (lines 765-859)
# ============================================================================

class ResNet34LF_BN(nn.Module):
    """Leak-proof classifier + watermark detector with BatchNorm.

    This is used ONLY for loading BN-trained checkpoints (e.g., external frozen C1).
    We keep C2 on GroupNorm to avoid BN+DataParallel eval collapse.
    """

    def __init__(self, num_classes: int, gate_strength: float = 2.10):
        super().__init__()
        # SECURITY: register as buffers so they persist in state_dict.
        self.register_buffer('gate_strength', torch.tensor(float(gate_strength)))
        self.register_buffer('destructive_strength', torch.tensor(1.0))

        base = tv_models.resnet34(weights=None)
        base.conv1.stride = (1, 1)
        self.base = base
        self._wrap_blur(base.layer2)
        self._wrap_blur(base.layer3)
        self._wrap_blur(base.layer4)

        self.wm_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)
        self.wm_affine = nn.Parameter(torch.zeros(num_classes))

    @staticmethod
    def _wrap_blur(layer):
        for b in layer:
            if b.downsample is not None:
                seq = []
                for sm in b.downsample:
                    if isinstance(sm, nn.Conv2d) and sm.stride == (2, 2):
                        sm.stride = (1, 1)
                        seq += [sm, BlurPool(sm.out_channels)]
                    else:
                        seq.append(sm)
                b.downsample = nn.Sequential(*seq)

    def forward(
            self,
            x: torch.Tensor,
            gate: bool = True,
            gate_target: Optional[torch.Tensor] = None,
            detach_gate: bool = False,
            detach_affine: bool = False,
            return_raw: bool = False,
    ):
        if gate_target is not None:
            raise RuntimeError("gate_target запрещён (label leakage).")

        x = self.base.relu(self.base.bn1(self.base.conv1(x)))
        x = self.base.maxpool(x)
        x1 = self.base.layer1(x)
        x2 = self.base.layer2(x1)
        x3 = self.base.layer3(x2)
        x4 = self.base.layer4(x3)

        pooled = self.base.avgpool(x4).flatten(1)
        logits_raw = self.base.fc(pooled)
        wm_logit = self.wm_head(x4).squeeze(1)

        logits = logits_raw
        if gate:
            g = torch.tanh(wm_logit).unsqueeze(1)
            if detach_gate:
                g = g.detach()
            w = self.wm_affine.view(1, -1)
            if detach_affine:
                w = w.detach()

            ds = getattr(self, "destructive_strength", 1.0)
            open_factor = (g + 1.0) * 0.5
            close_factor = 1.0 - open_factor
            inv_logits = -logits_raw * ds
            logits = (open_factor * logits_raw +
                      close_factor * inv_logits +
                      self.gate_strength * (g * w))

        if return_raw:
            return logits, wm_logit, logits_raw
        return logits, wm_logit, x4


# ============================================================================
# Training loop
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", required=True, type=Path)
    ap.add_argument("--val_root", required=True, type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--amp", action="store_true", help="Enable fp16 AMP")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Datasets — uses module-level ToTensorMinusOneOne for picklability
    tfm_train = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        ToTensorMinusOneOne(),
    ])
    tfm_val = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        ToTensorMinusOneOne(),
    ])

    train_ds = datasets.ImageFolder(str(args.train_root), transform=tfm_train)
    val_ds = datasets.ImageFolder(str(args.val_root), transform=tfm_val)

    assert train_ds.classes == val_ds.classes, "Train/val class mismatch"
    n_classes = len(train_ds.classes)
    print(f"[CLASSES] N={n_classes}: {train_ds.classes}")
    print(f"[DATA] train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # Model — EXACT architecture matching watermark trainer
    model = ResNet34LF_BN(num_classes=n_classes, gate_strength=2.10).to(device)
    print(f"[MODEL] ResNet34LF_BN (trainer-identical) num_classes={n_classes}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] params={n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    args.out_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out_root / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    classes_txt = args.out_root / "classes.txt"
    classes_txt.write_text("\n".join(train_ds.classes), encoding="utf-8")

    metrics_csv = args.out_root / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_acc",
            "val_loss", "val_acc", "epoch_time_sec",
        ])

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss, running_correct, running_n = 0.0, 0, 0

        for step, (xb, yb) in enumerate(train_loader, 1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=args.amp):
                # During C1 pre-training: gate=False, train only the base classifier.
                # wm_head and wm_affine get no gradient — they stay at init.
                logits, _, _ = model(xb, gate=False)
                loss = F.cross_entropy(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                pred = logits.argmax(1)
                running_correct += int((pred == yb).sum().item())
                running_n += yb.size(0)
                running_loss += float(loss.item()) * yb.size(0)

            if step % args.log_every == 0:
                cur_acc = running_correct / max(1, running_n)
                cur_loss = running_loss / max(1, running_n)
                print(f"[E{epoch:02d} step {step:05d}] "
                      f"loss={cur_loss:.4f} acc={cur_acc:.4f}")

        train_loss = running_loss / max(1, running_n)
        train_acc = running_correct / max(1, running_n)

        # Validation
        model.eval()
        v_loss, v_correct, v_n = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=args.amp):
                    logits, _, _ = model(xb, gate=False)
                    loss = F.cross_entropy(logits, yb)
                pred = logits.argmax(1)
                v_correct += int((pred == yb).sum().item())
                v_n += yb.size(0)
                v_loss += float(loss.item()) * yb.size(0)
        val_loss = v_loss / max(1, v_n)
        val_acc = v_correct / max(1, v_n)
        dt = time.time() - t0

        print(f"[E{epoch:02d}] TRAIN loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"VAL loss={val_loss:.4f} acc={val_acc:.4f} | time={dt:.1f}s")

        with metrics_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, f"{train_loss:.6f}", f"{train_acc:.4f}",
                f"{val_loss:.6f}", f"{val_acc:.4f}", f"{dt:.2f}",
            ])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "num_classes": n_classes,
                "classes": train_ds.classes,
                "arch": "ResNet34LF_BN_trainer_identical",
            }
            torch.save(ckpt, ckpt_dir / "c1_best.pth")
            print(f"[BEST] c1_best.pth val_acc={val_acc:.4f}")

        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "best_val_acc": best_val_acc,
            "num_classes": n_classes,
            "classes": train_ds.classes,
            "arch": "ResNet34LF_BN_trainer_identical",
        }, ckpt_dir / "c1_last.pth")

    print(f"[DONE] best val_acc={best_val_acc:.4f}")
    print(f"[DONE] best ckpt: {ckpt_dir / 'c1_best.pth'}")


if __name__ == "__main__":
    main()
