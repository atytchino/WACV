# -*- coding: utf-8 -*-
r"""
AutoEncoder trainer for color datasets (AFHQ, Tomato Leaf Disease).
=================================================================

Trains the UniversalAutoEncoder (from AE_ContentBound.py) on RGB color
images. Produces a checkpoint compatible with the watermark trainer's
`--ae_module AE_ContentBound --ae_class UniversalAutoEncoder` interface.

Loss: L1 + (1 - SSIM) on luma, with optional perceptual chroma term.
The dual loss matches what the watermark trainer expects: high luma
fidelity (PSNR_y target ~36-40 dB) plus reasonable chroma reconstruction.

Mirrors C1_agnostic_trainer conventions:
  - argparse CLI with --train_root / --val_root / --out_root
  - strict mode checking (no auto-conversion)
  - AMP-enabled, multi-GPU via DataParallel
  - per-epoch checkpoint + best + last + metrics CSV

Run (PowerShell):
  python .\train_ae_color.py `
    --train_root "E:\AFHQ\train" `
    --val_root   "E:\AFHQ\val" `
    --out_root   "E:\AE_TRAINED\AFHQ" `
    --epochs 20 --batch_size 16 --image_size 128 `
    --lr 2e-4 --workers 4 --amp `
    --ssim_lam 0.5 --chroma_lam 0.5

  python .\train_ae_color.py `
    --train_root "E:\TLD\train" `
    --val_root   "E:\TLD\val" `
    --out_root   "E:\AE_TRAINED\TLD" `
    --epochs 20 --batch_size 16 --image_size 128 `
    --lr 2e-4 --workers 4 --amp `
    --ssim_lam 0.5 --chroma_lam 0.5
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Import the AE architecture from AE_ContentBound.py (must be in same folder)
from AE_ContentBound import UniversalAutoEncoder, AEConfig

# Class verification helpers — case-insensitive train/val class alignment
from wm_dataset_configs import (
    verify_class_alignment,
    infer_classes_from_subfolders,
)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# -------------------------
# Reproducibility
# -------------------------

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -------------------------
# Dataset: flat scan of all images under a root (ignores class folders for AE)
# -------------------------

def list_images_under(root: Path) -> List[Path]:
    """Recursively collect all image files. Class structure is irrelevant for AE."""
    files: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in IMG_EXTS:
                files.append(p)
    files.sort()
    return files


class ImageOnlyDataset(Dataset):
    """Reads RGB images for autoencoder training.

    Strict RGB mode — raises if any image is not in mode 'RGB'. This matches
    the convention from the C1 trainer (no implicit conversions).
    """

    def __init__(self, root: Path, tfm, strict_rgb: bool = True):
        self.root = root
        self.tfm = tfm
        self.strict_rgb = strict_rgb
        self.files: List[Path] = list_images_under(root)
        if not self.files:
            raise RuntimeError(f"No images found under: {root}")
        print(f"[AE-DATA] {root}: {len(self.files)} images")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        p = self.files[i]
        with Image.open(p) as im:
            if self.strict_rgb and im.mode != "RGB":
                # Convert lazily for AE training (different from C1's strict policy);
                # AE needs RGB input regardless of source mode. We still warn once.
                im = im.convert("RGB")
            x = self.tfm(im)
        return x, str(p)


# -------------------------
# Loss helpers (luma-aware)
# -------------------------

def _rgb_to_luma01(x01: torch.Tensor) -> torch.Tensor:
    """Convert RGB [0,1] tensor to luma Y (single channel) using BT.601 weights."""
    if x01.size(1) == 1:
        return x01
    r = x01[:, 0:1]
    g = x01[:, 1:2]
    b = x01[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _gaussian_kernel1d(window: int, sigma: float, device, dtype) -> torch.Tensor:
    x = torch.arange(window, device=device, dtype=dtype) - (window - 1) / 2.0
    g = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def _gaussian_kernel2d(window: int, sigma: float, device, dtype) -> torch.Tensor:
    k1 = _gaussian_kernel1d(window, sigma, device=device, dtype=dtype)
    k2 = k1[:, None] @ k1[None, :]
    return k2.unsqueeze(0).unsqueeze(0)


def ssim_y(a01: torch.Tensor, b01: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """SSIM on luma channel. Returns scalar tensor in [-1, 1] (typically [0, 1])."""
    x = _rgb_to_luma01(a01)
    y = _rgb_to_luma01(b01)
    _, _, H, W = x.shape
    w = int(min(window, H, W))
    if w < 3:
        # Tiny image fallback: 1 - L1
        l1 = (x - y).abs().mean()
        return (1.0 - l1).clamp(-1.0, 1.0)
    if (w % 2) == 0:
        w -= 1
    pad = w // 2
    k = _gaussian_kernel2d(w, sigma, device=x.device, dtype=x.dtype)

    def conv(z: torch.Tensor) -> torch.Tensor:
        return F.conv2d(z, k, padding=pad)

    mux = conv(x)
    muy = conv(y)
    ex2 = conv(x * x)
    ey2 = conv(y * y)
    exy = conv(x * y)

    sigx2 = (ex2 - mux * mux).clamp_min(0.0)
    sigy2 = (ey2 - muy * muy).clamp_min(0.0)
    sigxy = exy - mux * muy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    num = (2 * mux * muy + c1) * (2 * sigxy + c2)
    den = (mux * mux + muy * muy + c1) * (sigx2 + sigy2 + c2)
    ssim_map = num / den.clamp_min(1e-6)
    return ssim_map.mean().clamp(-1.0, 1.0)


def psnr_y(a01: torch.Tensor, b01: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """PSNR on luma channel, in dB."""
    x = _rgb_to_luma01(a01)
    y = _rgb_to_luma01(b01)
    mse = ((x - y) ** 2).mean().clamp_min(eps)
    return -10.0 * torch.log10(mse)


def chroma_loss(a01: torch.Tensor, b01: torch.Tensor) -> torch.Tensor:
    """L1 on chroma channels (Cb, Cr). Zero for grayscale inputs."""
    if a01.size(1) == 1 or b01.size(1) == 1:
        return a01.new_tensor(0.0)
    # Compute CbCr for both
    yA = _rgb_to_luma01(a01)
    yB = _rgb_to_luma01(b01)
    cbA = 0.564 * (a01[:, 2:3] - yA)
    crA = 0.713 * (a01[:, 0:1] - yA)
    cbB = 0.564 * (b01[:, 2:3] - yB)
    crB = 0.713 * (b01[:, 0:1] - yB)
    return (cbA - cbB).abs().mean() + (crA - crB).abs().mean()


# -------------------------
# Evaluation
# -------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    n_batches = 0
    sum_l1 = 0.0
    sum_psnr = 0.0
    sum_ssim = 0.0
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        recon = model(xb).clamp(0, 1)
        l1 = (recon - xb).abs().mean().item()
        ps = psnr_y(recon, xb).item()
        ss = ssim_y(recon, xb).item()
        sum_l1 += l1
        sum_psnr += ps
        sum_ssim += ss
        n_batches += 1
    n = max(1, n_batches)
    return {
        "l1": sum_l1 / n,
        "psnr_y": sum_psnr / n,
        "ssim_y": sum_ssim / n,
    }


# -------------------------
# Checkpoint save (compatible with watermark trainer's load_ae)
# -------------------------

def save_checkpoint(
        out_dir: Path,
        name: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_metric: float,
        meta: dict,
) -> None:
    ensure_dir(out_dir)
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    payload = {
        "epoch": epoch,
        # Watermark trainer's load_ae looks for 'state_dict' or top-level keys
        "state_dict": state_dict,
        "ae_state_dict": state_dict,  # alternate alias
        "optimizer": optimizer.state_dict(),
        "best_metric": float(best_metric),
        "meta": meta,
    }
    torch.save(payload, out_dir / name)


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", required=True, help="Folder with training images (class subfolders OK)")
    ap.add_argument("--val_root", required=True, help="Folder with validation images")
    ap.add_argument("--out_root", required=True, help="Output folder for checkpoints + logs")

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4,
                    help="Batch size. Default 4 for 512x512 RGB; "
                         "use 16+ for smaller resolutions.")
    ap.add_argument("--image_size", type=int, default=512,
                    help="Square image size for AE training. "
                         "Set to 0 to use native resolution without resize.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--log_every", type=int, default=50)

    # Loss weights
    ap.add_argument("--l1_lam", type=float, default=1.0, help="Weight of L1 reconstruction loss")
    ap.add_argument("--ssim_lam", type=float, default=0.5, help="Weight of (1 - SSIM_y)")
    ap.add_argument("--chroma_lam", type=float, default=0.5, help="Weight of chroma L1 (CbCr)")

    # AE config tweaks (rarely needed)
    ap.add_argument("--gn_groups", type=int, default=32)
    ap.add_argument("--mult", type=int, default=16, help="Spatial padding multiple (e.g., 16 for /16 latent)")

    args = ap.parse_args()
    seed_all(args.seed)

    train_root = Path(args.train_root)
    val_root = Path(args.val_root)
    out_root = Path(args.out_root)
    ckpt_dir = out_root / "ckpts"
    ensure_dir(out_root)
    ensure_dir(ckpt_dir)

    # ── Verify train/val class alignment (case-insensitive) ──
    # AE training doesn't use class labels, but we still validate structure
    # so dataset preparation issues are caught BEFORE long training runs.
    try:
        canonical_classes = verify_class_alignment(
            train_root, val_root, case_insensitive=True
        )
        print(f"[CLASS CHECK] OK — {len(canonical_classes)} classes match "
              f"between train and val (case-insensitive)")
        print(f"[CLASS CHECK] classes: {canonical_classes}")
    except RuntimeError as e:
        print(f"[CLASS CHECK FAILED]\n{e}")
        raise SystemExit(1)

    # ── Transforms ──
    # If args.image_size > 0: resize to that square size (default 128 for speed).
    # If args.image_size == 0: use images at their native resolution.
    # Center-crop to square is implicit via Resize((s, s)) which scales both
    # dimensions; for non-square native images this distorts aspect ratio.
    # For AFHQ (512x512 native) and TLD (512x512 native) the images are already
    # square, so this is a no-op when image_size == 512.
    if args.image_size > 0:
        tfm = transforms.Compose([
            transforms.Resize(
                (args.image_size, args.image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ])
        print(f"[TFM] resize to {args.image_size}x{args.image_size}")
    else:
        tfm = transforms.Compose([
            transforms.ToTensor(),
        ])
        print(f"[TFM] using native image resolution (no resize)")

    train_ds = ImageOnlyDataset(train_root, tfm)
    val_ds = ImageOnlyDataset(val_root, tfm)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} | visible GPUs: {torch.cuda.device_count()}")

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin, drop_last=False)

    # Build AE
    ae_cfg = AEConfig()
    ae_cfg.gn_groups = int(args.gn_groups)
    ae_cfg.mult = int(args.mult)
    model = UniversalAutoEncoder(cfg=ae_cfg).to(device)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"[dp] Using nn.DataParallel on {torch.cuda.device_count()} GPUs")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    # Logs
    csv_path = out_root / "metrics.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "val_l1", "val_psnr_y", "val_ssim_y", "epoch_time_sec",
            ])

    best_metric = -float("inf")  # we'll track val_ssim_y (higher = better)

    # ── LR warmup schedule ──
    # Linearly ramp LR from 0 to args.lr over the first WARMUP_STEPS optimizer
    # steps. Prevents early-training explosions on high-contrast datasets
    # (e.g. TLD leaves with sharp disease spots) where initial random weights
    # produce large activations that AMP fp16 can overflow on.
    WARMUP_STEPS = 200
    target_lr = float(args.lr)

    def _set_lr(opt, lr_val: float) -> None:
        for pg in opt.param_groups:
            pg["lr"] = lr_val

    n_nan_skips_total = 0  # diagnostic counter for steps skipped due to NaN/Inf

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        running_loss = 0.0
        running_n = 0
        global_step = 0
        n_nan_skips_epoch = 0

        for xb, _ in train_loader:
            global_step += 1
            # Determine the overall optimizer step index across all epochs so
            # warmup ramps continuously even if epoch 1 ends mid-warmup.
            overall_step = (epoch - 1) * len(train_loader) + global_step

            # LR warmup: linear from 0 to target_lr over WARMUP_STEPS
            if overall_step <= WARMUP_STEPS:
                warmup_lr = target_lr * (overall_step / max(1, WARMUP_STEPS))
                _set_lr(optimizer, warmup_lr)
            elif overall_step == WARMUP_STEPS + 1:
                _set_lr(optimizer, target_lr)
                print(f"[LR] warmup complete at step {overall_step}, "
                      f"lr set to {target_lr:.2e}")

            xb = xb.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=(args.amp and device.type == "cuda")):
                # IMPORTANT: do NOT clamp during training — clamp blocks
                # gradients in saturated regions (output exactly 0 or 1)
                # which is catastrophic when many pixels saturate on a
                # poorly-initialized model. Use raw output for loss compute.
                # The UniversalAutoEncoder's forward_plain already does a
                # final clamp internally for visualization, but loss uses raw.
                recon = model(xb)
                # Soft saturation guard — only clamp for loss compute if recon
                # has wandered outside a reasonable range. Use the wider
                # [-0.1, 1.1] window so gradients still flow at boundaries.
                recon_for_loss = recon.clamp(-0.1, 1.1)
                L_l1 = (recon_for_loss - xb).abs().mean()
                L_ssim = 1.0 - ssim_y(recon_for_loss.clamp(0, 1), xb)
                L_chroma = chroma_loss(recon_for_loss.clamp(0, 1), xb)
                loss = (
                        args.l1_lam * L_l1
                        + args.ssim_lam * L_ssim
                        + args.chroma_lam * L_chroma
                )

            # NaN / Inf guard — skip the step rather than poison optimizer state
            if not torch.isfinite(loss):
                n_nan_skips_epoch += 1
                n_nan_skips_total += 1
                if n_nan_skips_total <= 5 or (n_nan_skips_total % 20 == 0):
                    print(f"[E{epoch:02d} step {global_step:05d}] "
                          f"WARN: non-finite loss ({loss.item()}), skipping step. "
                          f"L1={L_l1.item()} 1-SSIM={L_ssim.item()} "
                          f"chroma={L_chroma.item()}")
                # Clear any stale grads from previous backward
                optimizer.zero_grad(set_to_none=True)
                # If too many NaN skips in a row, the model is broken — abort.
                if n_nan_skips_total > 50:
                    print(f"[FATAL] {n_nan_skips_total} consecutive NaN losses — "
                          f"aborting. Likely causes: LR too high, AMP overflow "
                          f"on high-contrast batches, or dataset has corrupt images.")
                    return
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Pre-step weight check: are any gradients non-finite?
            # If yes, skip the step entirely — better than poisoning weights.
            grads_ok = True
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grads_ok = False
                    break
            if not grads_ok:
                n_nan_skips_epoch += 1
                n_nan_skips_total += 1
                if n_nan_skips_total <= 5 or (n_nan_skips_total % 20 == 0):
                    print(f"[E{epoch:02d} step {global_step:05d}] "
                          f"WARN: non-finite gradient — skipping optimizer step.")
                optimizer.zero_grad(set_to_none=True)
                # scaler.update() with skipped step shrinks scale, allowing recovery
                scaler.update()
                if n_nan_skips_total > 50:
                    print(f"[FATAL] {n_nan_skips_total} non-finite gradient skips — aborting.")
                    return
                continue
            # Tighter gradient clipping — was 1.0, reduce to 0.5 for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()

            # Post-step weight check: did optimizer.step poison weights with NaN?
            # If yes, abort cleanly so user knows to restart with --no-amp.
            weights_ok = True
            for p in model.parameters():
                if not torch.isfinite(p).all():
                    weights_ok = False
                    break
            if not weights_ok:
                print(f"[FATAL] Weights contain NaN/Inf after optimizer step at "
                      f"epoch {epoch}, step {global_step}.")
                print(f"[FATAL] This is unrecoverable — the model is poisoned.")
                print(f"[FATAL] Restart with --no-amp and/or --lr 1e-4 for stability.")
                return

            running_loss += float(loss.item()) * xb.size(0)
            running_n += xb.size(0)

            if args.log_every > 0 and (global_step % args.log_every == 0):
                avg_l = running_loss / max(1, running_n)
                print(f"[E{epoch:02d} step {global_step:05d}] "
                      f"loss={avg_l:.4f} L1={L_l1.item():.4f} "
                      f"1-SSIM={L_ssim.item():.4f} chroma={L_chroma.item():.4f}")

        train_loss = running_loss / max(1, running_n)
        val_stats = evaluate(model, val_loader, device)
        dt = time.time() - t0

        nan_note = ""
        if n_nan_skips_epoch > 0:
            nan_note = f" | nan_skips={n_nan_skips_epoch}"
        print(f"[E{epoch:02d}] TRAIN loss={train_loss:.4f} | "
              f"VAL L1={val_stats['l1']:.4f} PSNR_y={val_stats['psnr_y']:.2f}dB "
              f"SSIM_y={val_stats['ssim_y']:.4f} | time={dt:.1f}s{nan_note}")

        meta = {
            "dataset_train": str(train_root),
            "dataset_val": str(val_root),
            "image_size": int(args.image_size),
            "gn_groups": int(args.gn_groups),
            "mult": int(args.mult),
            "ae_class": "UniversalAutoEncoder",
            "ae_module": "AE_ContentBound",
        }

        # Always save epoch + last
        save_checkpoint(ckpt_dir, f"ae_epoch{epoch:03d}.pth", model, optimizer, epoch, best_metric, meta)
        save_checkpoint(ckpt_dir, "ae_last.pth", model, optimizer, epoch, best_metric, meta)

        # Track best by val SSIM
        if val_stats["ssim_y"] > best_metric:
            best_metric = val_stats["ssim_y"]
            save_checkpoint(ckpt_dir, "ae_best.pth", model, optimizer, epoch, best_metric, meta)
            print(f"[BEST] ae_best.pth val_SSIM_y={best_metric:.4f}")

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{val_stats['l1']:.6f}",
                f"{val_stats['psnr_y']:.4f}",
                f"{val_stats['ssim_y']:.6f}",
                f"{dt:.2f}",
            ])

    print("[DONE] AE color training complete.")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
