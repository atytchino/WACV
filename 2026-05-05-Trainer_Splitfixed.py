#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Agnostic watermark trainer (AE-agnostic) with:
  - No-upscale resize + pad to 160x160 (default)
  - Valid-mask propagation to avoid watermarking padded borders
  - Dynamic latent/skip spatial sizes (no hard-coded 32/64)
  - C2 backbone switched to GroupNorm (fixes BN+DataParallel eval collapse)
  - Safer C2 loss weighting schedule (fixes epoch-5 collapse)

This file is derived from your FIXED18 trainer and intentionally keeps the same
high-level interfaces:
  - AE must expose: enc(y01)->{'latent','s64'}, forward_plain(x01), embed_external_wm_gray(), embed_external_wm()

Tested assumptions:
  - Input images are loaded from train_root/<class>/* and val_root/<class>/*
  - Autoencoder is frozen; gradients flow only to ROI masks + pattern generators and C2

Author: ChatGPT (patched for Feb 2026 request)
"""

from __future__ import annotations

import argparse
import io
import json
import hashlib
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageDraw, ImageFont

import torchvision
from torchvision import models
from torchvision.transforms import functional as TF

# -------------------------
# Globals / utils
# -------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, nn.DataParallel) else m


def torch_load_trusted(path: Path, map_location=None):
    """Thin wrapper so you can later centralize any safety logic."""
    return torch.load(path, map_location=map_location)


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def sha256_file(path: Optional[Path], chunk_size: int = 1 << 20) -> Optional[str]:
    try:
        if path is None:
            return None
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        h = hashlib.sha256()
        with p.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def json_safe(obj):
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return json_safe(obj.item())
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return json_safe(obj.detach().cpu().item())
        return {
            "type": "tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return json_safe(vars(obj))
    return str(obj)


def write_json(path: Path, data: object) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(json_safe(data), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# -------------------------
# Resize+Pad transform (NO UPSCALE)
# -------------------------

class PadToSquareNoUpscale:
    """Convert to RGB, optionally downscale so max side <= size, then pad to (size,size).

    Returns:
      xN: [3,size,size] in [-1,1]
      valid_mask: [1,size,size] in {0,1}
    """

    def __init__(
            self,
            size: int = 160,
            pad_value: float = 0.0,
            interpolation: int = Image.BILINEAR,
            center: bool = True,
    ):
        self.size = int(size)
        self.pad_value = float(pad_value)
        self.interpolation = interpolation
        self.center = bool(center)

    def __call__(self, img: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(img, Image.Image):
            raise TypeError("PadToSquareNoUpscale expects a PIL.Image")

        img = img.convert("RGB")
        w, h = img.size
        if w <= 0 or h <= 0:
            raise ValueError(f"Bad image size: {(w, h)}")

        # scale (no upsample)
        scale = min(1.0, self.size / float(max(w, h)))
        if scale < 1.0:
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            if (nw, nh) != (w, h):
                img = img.resize((nw, nh), resample=self.interpolation)
        else:
            nw, nh = w, h

        # canvas + valid mask
        pv = int(round(self.pad_value * 255.0))
        canvas = Image.new("RGB", (self.size, self.size), (pv, pv, pv))
        mask = Image.new("L", (self.size, self.size), 0)

        if self.center:
            left = (self.size - nw) // 2
            top = (self.size - nh) // 2
        else:
            left, top = 0, 0

        canvas.paste(img, (left, top))
        mask.paste(Image.new("L", (nw, nh), 255), (left, top))

        x01 = TF.to_tensor(canvas)  # [3,H,W] in [0,1]
        valid = TF.to_tensor(mask).clamp(0, 1)  # [1,H,W] in {0,1}
        xN = x01 * 2.0 - 1.0
        return xN, valid


# -------------------------
# Dataset (path-aware + valid mask)
# -------------------------

def infer_classes(root: Path) -> List[str]:
    classes = [p.name for p in root.iterdir() if p.is_dir()]
    classes.sort()
    if not classes:
        raise RuntimeError(f"No class subfolders found in {root}")
    return classes


class DiskClassFolderWithPathsAndMask(Dataset):
    """Loads images from root/<class>/* and returns (xN, valid_mask, y, path)."""

    def __init__(self, root: Path, classes: List[str], tfm):
        self.root = Path(root)
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.tfm = tfm
        self.samples: List[Tuple[Path, int]] = []

        for cls_name in self.classes:
            d = self.root / cls_name
            if not d.exists():
                raise RuntimeError(f"Missing class folder in {self.root}: {cls_name}")
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    self.samples.append((p, self.class_to_idx[cls_name]))

        if not self.samples:
            raise RuntimeError(f"No images found under {self.root} for classes {self.classes}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, y = self.samples[i]
        with Image.open(path) as img:
            xN, vm = self.tfm(img)
        return xN, vm, int(y), str(path)


class DiskImageFolderWithPathsAndMask(Dataset):
    """Loads images from a directory (recursively) and returns (xN, valid_mask, path).

    This is used for inference/eval outside of training where we don't need class folders.
    """

    def __init__(self, root: Path, tfm, max_images: int = 0):
        self.root = Path(root)
        self.tfm = tfm
        self.samples: List[Path] = []

        if not self.root.exists():
            raise RuntimeError(f"Infer root not found: {self.root}")

        for p in self.root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                self.samples.append(p)

        self.samples.sort()
        if max_images and max_images > 0:
            self.samples = self.samples[: int(max_images)]

        if not self.samples:
            raise RuntimeError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path = self.samples[i]
        with Image.open(path) as img:
            xN, vm = self.tfm(img)
        return xN, vm, str(path)


class DiskPathListWithMask(Dataset):
    """Loads images from an explicit list of paths."""

    def __init__(self, paths: List[Path], tfm):
        self.paths = [Path(p) for p in paths]
        self.tfm = tfm
        if not self.paths:
            raise RuntimeError("Empty infer path list")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        path = self.paths[i]
        with Image.open(path) as img:
            xN, vm = self.tfm(img)
        return xN, vm, str(path)


# -------------------------
# Metrics (mask-aware)
# -------------------------

def _masked_mean(x: torch.Tensor, m: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """x: [B,C,H,W], m: [B,1,H,W] in {0,1}."""
    if m is None:
        return x.mean()
    m3 = m.repeat(1, x.size(1), 1, 1)
    denom = m3.sum().clamp_min(1.0)
    return (x * m3).sum() / denom


def psnr_torch(a01: torch.Tensor, b01: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, eps: float = 1e-12) -> torch.Tensor:
    """a01,b01: [B,3,H,W] in [0,1]. valid_mask: [B,1,H,W]"""
    if valid_mask is None:
        mse = F.mse_loss(a01, b01, reduction="mean").clamp_min(eps)
    else:
        m3 = valid_mask.repeat(1, a01.size(1), 1, 1)
        denom = m3.sum().clamp_min(1.0)
        mse = (((a01 - b01) ** 2) * m3).sum() / denom
        mse = mse.clamp_min(eps)
    return 10.0 * torch.log10(1.0 / mse)


def _masked_mean_per_sample(x: torch.Tensor, m: Optional[torch.Tensor]) -> torch.Tensor:
    """Per-sample masked mean.
    x: [B,C,H,W] (or [B,1,H,W]); m: [B,1,H,W] in {0,1}.
    Returns: [B] float tensor.
    """
    if m is None:
        return x.mean(dim=(1, 2, 3))
    m3 = m.repeat(1, x.size(1), 1, 1)
    denom = m3.sum(dim=(1, 2, 3)).clamp_min(1.0)
    num = (x * m3).sum(dim=(1, 2, 3))
    return num / denom


def mae_torch(a01: torch.Tensor, b01: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean absolute error in [0,1], mask-aware."""
    return _masked_mean((a01 - b01).abs(), valid_mask)


def psnr_y_torch(a01: torch.Tensor, b01: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, eps: float = 1e-12) -> torch.Tensor:
    """PSNR on luma only (Y), mask-aware."""
    ya = 0.299 * a01[:, 0:1] + 0.587 * a01[:, 1:2] + 0.114 * a01[:, 2:3]
    yb = 0.299 * b01[:, 0:1] + 0.587 * b01[:, 1:2] + 0.114 * b01[:, 2:3]
    if valid_mask is None:
        mse = F.mse_loss(ya, yb, reduction="mean").clamp_min(eps)
    else:
        denom = valid_mask.sum().clamp_min(1.0)
        mse = (((ya - yb) ** 2) * valid_mask).sum() / denom
        mse = mse.clamp_min(eps)
    return 10.0 * torch.log10(1.0 / mse)


def sat_hi_frac(x01: torch.Tensor, valid_mask: Optional[torch.Tensor], hi: float = 0.99) -> float:
    """Fraction of pixels above hi, mask-aware (averaged over batch+channels)."""
    if valid_mask is None:
        return float((x01 > hi).float().mean().item())
    m3 = valid_mask.repeat(1, x01.size(1), 1, 1)
    denom = m3.sum().clamp_min(1.0)
    num = ((x01 > hi).float() * m3).sum()
    return float((num / denom).item())


# -----------------------------
# SSIM (mask-aware, windowed)
# -----------------------------
# We compute SSIM on luma for RGB (to match the watermark pipeline preference).
# Implementation is mask-aware: padded pixels do not contribute, and local
# statistics are computed with a masked (weighted) Gaussian window.
#
# NOTE: valid_mask is expected to be [B,1,H,W] in {0,1}.

_SSIM_KERNEL_CACHE: Dict[Tuple[int, float, str, str], torch.Tensor] = {}


def _gaussian_kernel2d(window: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (int(window), float(sigma), str(device), str(dtype))
    k = _SSIM_KERNEL_CACHE.get(key, None)
    if k is not None:
        return k
    coords = torch.arange(window, device=device, dtype=dtype) - (window - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * (sigma ** 2)))
    g = g / g.sum()
    k2 = torch.outer(g, g)
    k2 = k2 / k2.sum()
    k2 = k2.view(1, 1, window, window)
    _SSIM_KERNEL_CACHE[key] = k2
    return k2


def _to_luma01(x01: torch.Tensor) -> torch.Tensor:
    # x01: [B,C,H,W] in [0,1]
    if x01.size(1) == 1:
        return x01
    return (0.299 * x01[:, 0:1] + 0.587 * x01[:, 1:2] + 0.114 * x01[:, 2:3]).clamp(0.0, 1.0)


def ssim_y_torch(
        a01: torch.Tensor,
        b01: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        window: int = 11,
        sigma: float = 1.5,
        c1: float = 0.01 ** 2,
        c2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Mask-aware SSIM computed on luma (Y).

    If valid_mask is None, the function falls back to an all-ones mask
    (i.e., normal SSIM).
    """
    if a01.dim() != 4 or b01.dim() != 4:
        raise ValueError("ssim_y_torch expects [B,C,H,W] tensors")
    if a01.shape != b01.shape:
        raise ValueError(f"ssim_y_torch: shape mismatch {tuple(a01.shape)} vs {tuple(b01.shape)}")

    x = _to_luma01(a01)
    y = _to_luma01(b01)

    if valid_mask is None:
        m = torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device, dtype=x.dtype)
    else:
        m = valid_mask
        if m.dim() == 3:
            m = m[:, None, :, :]
        if m.dim() != 4 or m.size(1) != 1:
            raise ValueError("valid_mask must be [B,1,H,W] or [B,H,W]")
        if m.size(2) != x.size(2) or m.size(3) != x.size(3):
            m = F.interpolate(m.to(dtype=x.dtype, device=x.device), size=(x.size(2), x.size(3)), mode="nearest")
        m = m.to(dtype=x.dtype, device=x.device)

    _, _, H, W = x.shape
    w = int(min(int(window), H, W))
    if w < 3:
        # For tiny images fall back to a masked L1 proxy (1 - L1).
        l1 = mae_torch(x, y, valid_mask=m)
        return (1.0 - l1).clamp(0.0, 1.0)
    if (w % 2) == 0:
        w -= 1
    pad = w // 2
    k = _gaussian_kernel2d(w, sigma, device=x.device, dtype=x.dtype)

    def conv(z: torch.Tensor) -> torch.Tensor:
        return F.conv2d(z, k, padding=pad)

    wm = conv(m).clamp_min(1e-6)  # effective mass of mask in local window
    mux = conv(x * m) / wm
    muy = conv(y * m) / wm

    ex2 = conv(x * x * m) / wm
    ey2 = conv(y * y * m) / wm
    exy = conv(x * y * m) / wm

    sigx2 = (ex2 - mux * mux).clamp_min(0.0)
    sigy2 = (ey2 - muy * muy).clamp_min(0.0)
    sigxy = exy - mux * muy

    num = (2 * mux * muy + c1) * (2 * sigxy + c2)
    den = (mux * mux + muy * muy + c1) * (sigx2 + sigy2 + c2)
    ssim_map = num / den.clamp_min(1e-6)

    # Only score pixels whose *center* is valid (prevents padding cheating).
    v = ((m > 0.5) & (wm > 1e-6)).to(ssim_map.dtype)
    ssim_mean = (ssim_map * v).sum() / v.sum().clamp_min(1.0)
    return ssim_mean.clamp(-1.0, 1.0)


# -------------------------
# ROI masks + low-texture prior
# -------------------------

class MiniUNetMask(nn.Module):
    """Lightweight mask predictor producing [B,1,H,W] in (0,1)."""

    def __init__(self, in_ch: int, mid: int = 64):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, mid, 3, 1, 1), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(mid, mid, 3, 1, 1), nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(nn.Conv2d(mid, mid, 3, 1, 1), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(mid, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc1(x)
        h = self.enc2(h)
        h = self.dec1(h)
        return torch.sigmoid(self.out(h))


class LocalVariance(nn.Module):
    """Local variance (low texture = low variance)."""

    def __init__(self, k: int = 9):
        super().__init__()
        self.k = int(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = F.avg_pool2d(x, self.k, 1, self.k // 2)
        mean2 = F.avg_pool2d(x * x, self.k, 1, self.k // 2)
        var = (mean2 - mean * mean).clamp_min(0.0)
        return var


# -------------------------
# Watermark pattern generators
# -------------------------

def _gn_groups(ch: int, max_groups: int = 8) -> int:
    g = min(max_groups, ch)
    while g > 1 and (ch % g) != 0:
        g -= 1
    return max(1, g)


class _MiniUNetWM(nn.Module):
    """Small image-conditioned watermark refiner that keeps interfaces stable.

    It avoids transposed convolutions, so it is much less likely to invent checkerboard.
    """

    def __init__(self, in_ch: int, mid: int = 64):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, 1, 0, bias=False),
            nn.GroupNorm(_gn_groups(mid), mid),
            nn.SiLU(inplace=True),
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(mid, mid, 3, 1, 1, bias=False),
            nn.GroupNorm(_gn_groups(mid), mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, mid, 3, 1, 1, bias=False),
            nn.GroupNorm(_gn_groups(mid), mid),
            nn.SiLU(inplace=True),
        )
        self.down = nn.Sequential(
            nn.Conv2d(mid, mid, 3, 2, 1, bias=False),
            nn.GroupNorm(_gn_groups(mid), mid),
            nn.SiLU(inplace=True),
        )
        self.bott = nn.Sequential(
            nn.Conv2d(mid, mid, 3, 1, 1, bias=False),
            nn.GroupNorm(_gn_groups(mid), mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, mid, 3, 1, 1, bias=False),
            nn.GroupNorm(_gn_groups(mid), mid),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(mid + mid, mid, 3, 1, 1, bias=False),
            nn.GroupNorm(_gn_groups(mid), mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, mid, 3, 1, 1, bias=False),
            nn.GroupNorm(_gn_groups(mid), mid),
            nn.SiLU(inplace=True),
        )
        self.out = nn.Conv2d(mid, in_ch, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.reduce(x)
        h1 = self.enc1(x0)
        h2 = self.down(h1)
        hb = self.bott(h2)
        hu = F.interpolate(hb, size=h1.shape[-2:], mode="bilinear", align_corners=False)
        h = self.fuse(torch.cat([h1, hu], dim=1))
        return self.out(h)


class GLat(_MiniUNetWM):
    def __init__(self, ch: int = 1024):
        super().__init__(in_ch=ch, mid=64)


class G64(_MiniUNetWM):
    def __init__(self, ch: int = 512):
        super().__init__(in_ch=ch, mid=64)


class FreqController(nn.Module):
    """Predict per-image low-vs-mid band mixing weights from simple image statistics."""

    def __init__(self, in_dim: int = 6, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 2),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


# -------------------------
# C2 model (classifier + wm logit) — GroupNorm (fixes BN collapse)
# -------------------------

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


class ResNet34LF_GN(nn.Module):
    """Leak-proof classifier + watermark detector with GroupNorm.

    The *main* reason for GN here: with DataParallel your per-GPU batch can be tiny,
    BatchNorm running stats become garbage, and eval collapses (exactly your E05/E06 symptom).
    """

    def __init__(self, num_classes: int, gate_strength: float = 2.10, gn_groups: int = 32):
        super().__init__()
        self.gate_strength = float(gate_strength)

        def gn_layer(c: int):
            # 32 divides {64,128,256,512} (ResNet34 channels)
            g = min(gn_groups, c)
            if c % g != 0:
                # fall back to largest divisor
                for gg in reversed(range(1, g + 1)):
                    if c % gg == 0:
                        g = gg
                        break
            return nn.GroupNorm(g, c)

        base = models.resnet34(weights=None, norm_layer=gn_layer)
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
            logits = logits_raw + self.gate_strength * (g * w)

        if return_raw:
            return logits, wm_logit, logits_raw
        return logits, wm_logit, x4



class ResNet34LF_BN(nn.Module):
    """Leak-proof classifier + watermark detector with BatchNorm.

    This is used ONLY for loading BN-trained checkpoints (e.g., external frozen C1).
    We keep C2 on GroupNorm to avoid BN+DataParallel eval collapse.
    """

    def __init__(self, num_classes: int, gate_strength: float = 2.10):
        super().__init__()
        self.gate_strength = float(gate_strength)

        base = models.resnet34(weights=None)  # default BatchNorm2d
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
            logits = logits_raw + self.gate_strength * (g * w)

        if return_raw:
            return logits, wm_logit, logits_raw
        return logits, wm_logit, x4


def _extract_state_dict_generic(raw_obj) -> Dict[str, torch.Tensor]:
    """Try common checkpoint layouts and return a flat state_dict."""
    if isinstance(raw_obj, dict):
        for k in ("state_dict", "model_state_dict", "model", "net", "c1", "classifier", "weights"):
            if k in raw_obj and isinstance(raw_obj[k], dict):
                cand = raw_obj[k]
                if cand and all(isinstance(v, torch.Tensor) for v in cand.values()):
                    return cand
        # sometimes the dict itself is the state dict
        if raw_obj and all(isinstance(v, torch.Tensor) for v in raw_obj.values()):
            return raw_obj
    # fallback
    if isinstance(raw_obj, dict) and raw_obj:
        # last resort: keep tensor entries only
        cand = {k: v for k, v in raw_obj.items() if isinstance(v, torch.Tensor)}
        if cand:
            return cand
    raise RuntimeError("Could not extract a valid state_dict from checkpoint")


def load_c1_classifier(
    ckpt_path: Path,
    num_classes: int,
    device: torch.device,
    gate_strength: float = 2.10,
    gn_groups: int = 32,
) -> nn.Module:
    """Load a frozen C1 classifier checkpoint with auto-detect.

    Supports:
      - state_dict directly
      - dict with keys like state_dict/model/model_state_dict/...
      - DataParallel prefix 'module.'
      - torchvision-style keys (conv1., layer1., fc.) -> remapped to our wrapper as base.*
      - BatchNorm checkpoints (running_mean/var) -> ResNet34LF_BN
      - GroupNorm checkpoints -> ResNet34LF_GN
    """
    ckpt_path = Path(ckpt_path)
    raw = torch_load_trusted(ckpt_path, map_location=device)
    sd = _extract_state_dict_generic(raw)

    # strip DataParallel prefix
    sd = {(k.replace("module.", "", 1) if k.startswith("module.") else k): v for k, v in sd.items()}

    # remap torchvision-style keys (conv1., layer*, bn1., fc.) into our wrapper (base.*)
    has_base_prefix = any(k.startswith("base.") for k in sd.keys())
    if not has_base_prefix and any(k.startswith(("conv1.", "bn1.", "layer", "fc.")) for k in sd.keys()):
        sd2 = {}
        for k, v in sd.items():
            if k.startswith("wm_head.") or k.startswith("wm_affine"):
                sd2[k] = v
            elif k.startswith("fc."):
                sd2["base.fc." + k[3:]] = v
            else:
                sd2["base." + k] = v
        sd = sd2

    # detect BN vs GN by presence of running stats
    is_bn = any(("running_mean" in k) or ("running_var" in k) or ("num_batches_tracked" in k) for k in sd.keys())

    if is_bn:
        c1 = ResNet34LF_BN(num_classes=num_classes, gate_strength=gate_strength).to(device)
        kind = "BN"
    else:
        c1 = ResNet34LF_GN(num_classes=num_classes, gate_strength=gate_strength, gn_groups=gn_groups).to(device)
        kind = "GN"

    # conv1 channel auto-adapt: allow grayscale (1ch) C1 checkpoints to load into a 3ch model.
    # We assume the input images are grayscale replicated across RGB, so repeating weights keeps behavior stable.
    k_conv = "base.conv1.weight"
    if k_conv in sd and isinstance(sd[k_conv], torch.Tensor):
        w = sd[k_conv]
        try:
            if w.ndim == 4 and w.shape[1] == 1 and getattr(c1.base.conv1, "in_channels", 3) == 3:
                sd[k_conv] = w.repeat(1, 3, 1, 1) / 3.0
        except Exception:
            pass

    res = c1.load_state_dict(sd, strict=False)

    # Light diagnostics (don't spam)
    mk = list(res.missing_keys)
    uk = list(res.unexpected_keys)
    if mk or uk:
        print(f"[C1 LOAD] {kind} strict=False | missing={len(mk)} unexpected={len(uk)}")
        if mk:
            print("  missing (first 10):")
            for k in mk[:10]:
                print("   ", k)
        if uk:
            print("  unexpected (first 10):")
            for k in uk[:10]:
                print("   ", k)
    else:
        print(f"[C1 LOAD] {kind} strict=True-ish (no missing/unexpected)")

    c1.eval()
    for p in c1.parameters():
        p.requires_grad_(False)
    return c1


# -------------------------
# Controller
# -------------------------

@dataclass
class Controller:
    eps: float = 0.10
    r_skip: float = 0.66
    ema_delta: float = 0.0
    ema_wm_gap: float = 0.0
    ema_acc_gap_g: float = 0.0
    ema_gate_spec_gap: float = 0.0
    ema_margin_gate_spec: float = 0.0
    # FIX-4: hysteresis counters (anti-oscillation for eps)
    _eps_consec_up: int = 0
    _eps_consec_down: int = 0
    _eps_hysteresis: int = 3


# -------------------------
# AE loader (dynamic import) — robust for nested/standalone AE files
# -------------------------

def load_ae(ae_ckpt: Path, ae_module: str, ae_class: str, ae_py_path: Path, device: torch.device):
    """Load UniversalAutoEncoder weights with key remapping + robust module import."""
    import importlib

    ae_py_path = Path(ae_py_path)

    # Allow passing ae_module as a .py file
    if ae_module.endswith(".py"):
        p = Path(ae_module)
        ae_module = p.stem
        ae_py_path = p.parent

    # Allow passing ae_py_path as a .py file
    if ae_py_path.suffix.lower() == ".py":
        ae_py_path = ae_py_path.parent

    sys.path.insert(0, str(ae_py_path))
    try:
        mod = importlib.import_module(ae_module)
        cls = getattr(mod, ae_class)
    finally:
        if sys.path and sys.path[0] == str(ae_py_path):
            sys.path.pop(0)

    ae = cls().to(device)

    raw = torch_load_trusted(ae_ckpt, map_location=device)

    # extract state dict
    sd = None
    if isinstance(raw, dict):
        for k in ("state_dict", "model_state_dict", "model", "ae", "net", "weights"):
            if k in raw and isinstance(raw[k], dict):
                sd = raw[k]
                break
        if sd is None and all(isinstance(v, torch.Tensor) for v in raw.values()):
            sd = raw
    else:
        sd = raw

    if sd is None:
        raise RuntimeError(f"Could not extract state_dict from AE checkpoint: {ae_ckpt}")

    # strip DataParallel prefix
    sd = {(kk.replace("module.", "", 1) if kk.startswith("module.") else kk): vv for kk, vv in sd.items()}

    # remap Sequential decoder keys -> ModuleDict decoder keys (older checkpoints)
    up_re = re.compile(r"^(dec_[yc])\.(up(?:32to64|64to128|128to256|256to512))\.(\d+)\.(weight|bias)$")
    idx_map = {
        (0, "weight"): "conv1.weight",
        (0, "bias"): "conv1.bias",
        (1, "weight"): "nr.0.weight",
        (1, "bias"): "nr.0.bias",
        (3, "weight"): "conv2.weight",
        (3, "bias"): "conv2.bias",
        (4, "weight"): "nr2.0.weight",
        (4, "bias"): "nr2.0.bias",
    }

    remapped = {}
    for k, v in sd.items():
        m = up_re.match(k)
        if m:
            branch, block, idx_s, wb = m.group(1), m.group(2), m.group(3), m.group(4)
            idx = int(idx_s)
            suffix = idx_map.get((idx, wb))
            if suffix is not None:
                remapped[f"{branch}.{block}.{suffix}"] = v
                continue
        remapped[k] = v
    sd = remapped

    load_res = ae.load_state_dict(sd, strict=False)

    miss = [k for k in load_res.missing_keys if not (k.startswith("enc.") or k.startswith("dec."))]
    unexp = load_res.unexpected_keys
    if miss:
        print(f"[AE LOAD] missing_keys (non-alias): {len(miss)} (showing up to 20)")
        for k in miss[:20]:
            print("  ", k)
    if unexp:
        print(f"[AE LOAD] unexpected_keys: {len(unexp)} (showing up to 20)")
        for k in unexp[:20]:
            print("  ", k)

    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    return ae


# -------------------------
# Config
# -------------------------

@dataclass
class TrainConfig:
    train_root: Path
    val_root: Path
    out_root: Path

    ae_ckpt: Path
    ae_module: str
    ae_class: str
    ae_py_path: Path

    # GPU
    gpus: str = ""
    use_dataparallel: bool = True

    # training
    epochs: int = 30
    batch_size: int = 12
    num_workers: int = 4

    # image pipeline
    image_size: int = 160
    pad_value: float = 0.0

    # training synthesis mix (prod-like vs augmented)
    # With probability train_prod_mix_prob we synthesize using k_factor=1.0 and varpercent=False
    # to match validate/production distribution.
    train_prod_mix_prob: float = 0.70

    # residual clamp (luma delta) to prevent rare spikes / border leakage
    wm_res_clip: float = 0.10
    wm_res_clip_mode: str = "tanh"  # {"tanh","hard","none"}

    # logging
    print_every: int = 1
    val_probe_every: int = 5  # set 0 to disable
    val_probe_batches: int = 8  # how many val batches to probe
    val_every: int = 1
    collage_every: int = 50
    pub_collage_enable: bool = True
    pub_collage_per_class: int = 10
    pub_collage_keep_train_debug: bool = False

    export_wm_dataset: bool = True
    wm_jpeg_quality: int = 92

    wm_export_format: str = "png"  # {"png","jpg"}
    avoid_black_thr: float = 0.00  # if >0, exclude near-black pixels from ROI + embedding (in luma [0,1])
    avoid_white_thr: float = 1.01   # if <=1.0, exclude near-white pixels from ROI+embedding (luma in [0,1])

    # auto band-mix profile: let the system choose low vs mid band per image while disallowing trivial HF codes
    profile: str = "auto_bandmix"   # {"auto_bandmix","legacy"}
    band_low_k: int = 11
    band_mid_k: int = 5
    band_energy_norm: bool = True  # normalize low/mid branches before mixing so weights reflect actual injected energy
    band_norm_max_gain: float = 4.0  # clamp RMS restoration gain after branch normalization
    band_mid_floor: float = 0.20  # optional minimum mid-band share in auto_bandmix (0 disables)
    soft_roi_blur_k: int = 3
    freq_ctrl_temp: float = 1.0

    # eval/infer embedding profile (to match production / out-of-training behavior)
    eval_eps: float = 0.0  # if >0, override ctrl.eps during eval/infer
    eval_r_skip: float = -1.0  # if >=0, override ctrl.r_skip during eval/infer

    # delta shaping (keeps clean→wm delta "budget" but avoids gray veils / background lift)
    delta_remove_dc: bool = True  # subtract mean(delta) on valid region (prevents global brightness shift)
    delta_hp_beta: float = 0.70  # 0 disables; higher suppresses low-frequency residual ("veil")
    delta_hp_window: int = 9  # box-blur window for lowpass estimate (odd preferred)
    delta_post_blur_k: int = 3      # post-blur on deltaY to suppress grid artifacts (0 disables)
    delta_post_blur_mix: float = 0.35  # 0..1: blend between original delta and post-blurred delta
    grid_boundary_lambda: float = 0.20  # penalize 8x8-ish block boundaries to kill checkerboard (0 disables)
    bg_protect_thr: float = 0.06  # luma threshold (in [0,1]) to treat as "dark background" for protection
    bg_delta_scale: float = 0.20  # scale delta in dark bg pixels (0..1); energy is re-normalized elsewhere
    delta_renorm: bool = True  # preserve RMS(delta) after shaping (keeps clean→wm delta stable)
    delta_renorm_max: float = 3.0  # clamp renorm gain to avoid pathological blow-ups
    headroom_margin: float = 0.98    # clamp delta so base+delta stays within [0,1] with margin (reduces saturation artifacts)
    min_delta_rms_k: float = 0.015  # minimum delta RMS as fraction of eps (prevents zero-watermark collapse)
    min_delta_lambda: float = 0.20  # weight of min-delta hinge loss
    max_alpha_boost: float = 12.0  # maximum adaptive boost for alpha when AE response is too weak

    # optional image-adaptive shaping (OFF by default)
    freq_adapt: bool = False
    freq_beta_min: float = 0.50
    freq_beta_max: float = 0.80
    freq_bg_scale_min: float = 0.10
    freq_bg_scale_max: float = 0.30
    freq_tex_window: int = 9

    # runtime mode
    mode: str = "train"  # {"train","infer"}

    # inference / watermark-application (outside training)
    system_ckpt: Optional[Path] = None  # wm_system_eXXX.pth
    c2_eval_ckpt: Optional[Path] = None  # c2_eval_eXXX.pth (optional train resume)
    opt_g_ckpt: Optional[Path] = None    # opt_g_eXXX.pth (optional train resume)
    infer_root: Optional[Path] = None
    infer_list: Optional[Path] = None
    infer_out: Optional[Path] = None
    infer_suffix: str = "_watermarked"
    infer_save_base: bool = False
    infer_save_diff: bool = False
    infer_max_images: int = 0
    # production-like ("real life") validation
    real_val_every: int = 1  # run every N epochs (0 disables)
    real_val_max_batches: int = 0  # 0 = full val loader
    real_val_jpeg_quality: int = 92  # 0 disables jpeg roundtrip
    real_val_jpeg_quality_lo: int = 85  # additional JPEG stress test quality
    real_val_resize_small: int = 144  # resize roundtrip small side (160->small->160); 0 disables
    real_val_print_confusion: bool = True
    real_val_diag_unmasked: bool = True
    # automatic safety stop: stop training when real-life classification drops below
    # (random_accuracy - real_val_stop_margin_pp) for the selected metric/scope.
    # Example: 4 classes => random=25%, margin=5pp => stop at 20%.
    real_val_stop_enable: bool = True
    real_val_stop_margin_pp: float = 5.0
    real_val_stop_patience: int = 1
    real_val_stop_metric: str = "both"   # {"raw","base","both"}
    real_val_stop_scope: str = "worst"   # {"primary","worst"}

    # controller
    psnr_dyn_margin: float = 0.50
    ctrl_init_eps: float = 0.10
    ctrl_init_r_skip: float = 0.66
    ctrl_dmargin_target: float = 0.06  # realistic target for this setup; 0.30 overdrives eps
    ctrl_wm_gap_target: float = 0.08  # target separation between wm_head(clean) and wm_head(wm)
    ctrl_detwarm_eps_floor: float = 0.08  # keep eps non-trivial while detector is warming up
    ctrl_delta_abs_floor: float = 0.0010  # if watermark amplitude falls below this, force eps up
    ctrl_eps_max: float = 0.12         # prevent runaway watermark strength
    ctrl_eps_up: float = 0.0010        # normal controller eps increase
    ctrl_eps_zero_boost: float = 0.0015  # extra increase only when watermark nearly vanished
    ctrl_class_gap_target: float = 0.20  # desired gated class split before eps is allowed to relax
    ctrl_gate_specific_target: float = 0.15  # desired excess of gated split over non-gated split
    ctrl_margin_gate_specific_target: float = 0.20  # desired margin-based gate-specific gap before eps is allowed to relax

    # watermark energy allocation
    r_lat_min: float = 0.33
    r_lat_max: float = 1.00
    val_r_skip_min: float = 0.60
    val_r_skip_max: float = 0.75
    skip_hp_mix: float = 0.50


    # anti-grid controls for skip (S64) watermark
    w64_hp_k: int = 5            # kernel for lowpass/highpass split in S64 domain
    w64_post_blur_k: int = 3     # additional smoothing kernel in S64 domain (0 disables)
    # IMPORTANT: for this UNet16 AE there is no explicit channel-wise L2 normalize.
    # Defaults set to 1.0 to avoid accidental over-amplification.
    alpha_lat_gain: float = 1.0
    alpha_skip_gain: float = 1.0

    # latent/skip watermark distribution guard (soft quota)
    lat_quota_min: float = 0.20
    lat_quota_lambda: float = 0.10
    lat_quota_warmup_epochs: int = 2

    # spectral guard on luma residual (optional)
    spec_window: int = 9
    spec_lowfreq_max: float = 0.55
    spec_lowfreq_min: float = 0.25
    spec_lambda: float = 0.05

    # ROI hyperparams
    keep_lat: float = 0.32
    keep_64: float = 0.58
    roi_teach_epochs: int = 3

    # ROI loss weights
    lam_area: float = 0.25
    lam_overlap: float = 0.30
    lam_tv: float = 0.05
    lam_sparse: float = 0.02
    lam_bin: float = 0.001
    lam_tex: float = 0.25
    lam_teacher: float = 0.20

    # watermark threshold calibration
    thr_quantile: float = 0.995

    # C2 objectives
    clean_uniform: bool = False  # legacy fallback; fail-clean key-gap is recommended for split-oriented training
    w_clean: float = 1.20        # global multiplier for clean CE + clean-gated sabotage/consistency
    clean_consistency: bool = False
    clean_consistency_temp: float = 1.0
    c2_keygap_margin: float = 0.22
    c2_keygap_w_suppress: float = 4.0
    c2_keygap_w_cap: float = 1.5
    c2_ng_invar_w: float = 1.00
    c2_ng_kl_temp: float = 1.0
    c2_gate_specific_w: float = 1.50
    c2_gate_specific_margin: float = 0.35
    c2_clean_ng_keep_w: float = 0.90
    c2_clean_ng_margin_floor: float = 1.25
    c2_warm_ng_invar_mult: float = 1.75
    c2_warm_gate_specific_mult: float = 2.50
    c2_warm_clean_ng_keep_mult: float = 2.25
    c2_early_ng_invar_mult: float = 1.35
    c2_early_gate_specific_mult: float = 1.75
    c2_early_clean_ng_keep_mult: float = 1.60
    c2_late_ng_invar_mult: float = 1.20
    c2_late_gate_specific_mult: float = 1.60
    c2_late_clean_ng_keep_mult: float = 1.40
    c2_no_gap_ng_invar_mult: float = 1.10
    c2_no_gap_gate_specific_mult: float = 1.35
    c2_no_gap_clean_ng_keep_mult: float = 1.20
    c2_best_checkpoint_require_pass_floors: bool = True
    c2_init_from_c1: bool = True

    # watermark detector warmup (recommended if det_acc ~50%)
    det_warmup_epochs: int = 2          # detector-focused warmup for first N epochs (0 disables)
    det_warmup_w_det: float = 6.0       # weight for BCE(wm_logit) during warmup
    det_warmup_k_factor: float = 1.75   # stronger C2-only positives during warmup
    det_post_w_det_early: float = 2.50  # FIX-1: was 1.25 — too low, wm_head died after warmup
    det_post_w_det_late: float = 2.50   # FIX-1: was 1.25 — caused gate collapse (tanh→0)
    gate_close_margin: float = 0.30   # FIX-2: target tanh(wm_logit_clean) < -margin (gate closed for clean)
    gate_close_w: float = 1.50        # FIX-2: weight of gate_close loss
    det_post_w_sep_early: float = 0.50  # margin separation weight for epochs 3..4
    det_post_w_sep_late: float = 1.10   # margin separation weight after early phase
    det_warm_unfreeze_layer4: bool = True  # let the last ResNet block adapt to watermark cues during warmup

    # optimization
    lr_c2: float = 1e-4
    lr_g: float = 1e-4
    wd: float = 1e-4
    ema_tau: float = 0.995

    # gray/RGB handling
    auto_switch: bool = True
    gray_like_eps: float = 0.010
    force_gray_output: bool = True

    # diagnostics (not in loss by default)
    white_thr: float = 0.70
    # leak-proof gate
    gate_strength: float = 2.10
    gn_groups: int = 32

    # stabilizers
    c2_grad_clip: float = 2.0
    wm_affine_l2: float = 0.0  # set to e.g. 1e-4 if you want to regularize

    # C1 guard rail (external frozen classifier; used to prevent watermark from degrading classification)
    c1_ckpt: Optional[Path] = None
    c1_guard_min_acc: float = 0.0
    c1_guard_max_drop: float = 0.05  # Keep C1 wm acc within this many points of C1 clean acc (0 disables)
    c1_guard_lambda: float = 0.20
    c1_guard_every: int = 1
    c1_guard_ce_margin: float = 0.0
    c1_brake_eps: float = 0.005

    # generator push (explicit prod-like support for watermarked path)
    g_push_start_epoch: int = 1
    g_push_margin_target: float = 0.25
    g_push_cls_lambda: float = 0.35
    g_push_det_lambda: float = 0.20
    g_push_margin_lambda: float = 0.30

    # ── Transfer-attack & diversity protection (SOTA-beating) ──
    transfer_lam: float = 0.50        # L_transfer: penalise WM detection on transplanted residual
    diversity_lam: float = 0.10       # L_diversity: cosine sim between WM residuals in batch
    ssim_lam: float = 0.60            # L_ssim: perceptual quality (was missing from L_g)
    transfer_start_epoch: int = 2     # begin transfer-attack sim after N epochs (split must exist first)

    # checkpoint / model-selection metadata for real-life prod eval
    best_realval_acc_both_floor: float = 0.70
    best_realval_det_acc_floor: float = 0.85
    best_realval_max_border_ratio: float = 0.10
    best_realval_gap_floor: float = 0.05
    best_realval_gate_specific_floor: float = 0.05



# -------------------------
# Trainer
# -------------------------

class WatermarkTrainer:
    def _apply_profile_defaults(self) -> None:
        cfg = self.cfg
        prof = str(getattr(cfg, "profile", "auto_bandmix") or "auto_bandmix").lower().strip()
        cfg.profile = prof
        if prof == "auto_bandmix":
            cfg.delta_hp_beta = 0.0
            cfg.delta_post_blur_k = 0
            cfg.delta_post_blur_mix = 0.0
            cfg.spec_lambda = max(float(getattr(cfg, "spec_lambda", 0.05) or 0.05), 0.05)
            cfg.skip_hp_mix = 0.25
            cfg.w64_post_blur_k = max(int(getattr(cfg, "w64_post_blur_k", 0) or 0), 5)
            cfg.w64_hp_k = max(int(getattr(cfg, "w64_hp_k", 0) or 0), 5)
            aw = float(getattr(cfg, "avoid_white_thr", 1.01) or 1.01)
            cfg.avoid_white_thr = (min(aw, 0.98) if (aw > 0.0 and aw <= 1.0) else 1.01)
            cfg.headroom_margin = min(float(getattr(cfg, "headroom_margin", 0.98) or 0.98), 0.98)

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self._apply_profile_defaults()

        if cfg.gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

        ensure_dir(cfg.out_root)
        self.art_root = cfg.out_root / "artifacts"
        self.ckpt_root = cfg.out_root / "checkpoints"
        ensure_dir(self.art_root)
        ensure_dir(self.ckpt_root)

        self.mode = str(getattr(cfg, "mode", "train") or "train").lower().strip()
        if self.mode not in ("train", "infer"):
            raise RuntimeError(f"Unknown mode={self.mode!r}. Use --mode train|infer")

        # Inference-only runtime init (no class folders, no C2, no optimizers).
        if self.mode == "infer":
            self._init_infer_runtime()
            return

        # classes
        self.classes = infer_classes(cfg.train_root)
        val_classes = infer_classes(cfg.val_root)
        if self.classes != val_classes:
            raise RuntimeError(f"Train/Val classes mismatch:\ntrain={self.classes}\nval={val_classes}")

        print("\n[CLASSES] Loaded and verified")
        print(f"  num_classes = {len(self.classes)}")
        for i, c in enumerate(self.classes):
            print(f"  {i:02d}: {c}")
        print()

        # transforms + datasets
        self.tfm = PadToSquareNoUpscale(size=cfg.image_size, pad_value=cfg.pad_value)
        self.train_ds = DiskClassFolderWithPathsAndMask(cfg.train_root, self.classes, self.tfm)
        self.val_ds = DiskClassFolderWithPathsAndMask(cfg.val_root, self.classes, self.tfm)

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )

        # Shuffled val probe loader (small, for quick sanity checks during training)
        self.val_probe_loader = DataLoader(
            self.val_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )

        # AE
        self.ae = load_ae(cfg.ae_ckpt, cfg.ae_module, cfg.ae_class, cfg.ae_py_path, self.device)

        # ROI + patterns
        self.mask_lat = MiniUNetMask(1024, mid=64).to(self.device)
        self.mask_64 = MiniUNetMask(512, mid=64).to(self.device)
        self.g_lat = GLat(1024).to(self.device)
        self.g_64 = G64(512).to(self.device)
        self.freq_ctrl = FreqController(in_dim=6, hidden=32).to(self.device)
        self.lowtex = LocalVariance(k=9).to(self.device)

        # C2 and EMA
        self.c2 = ResNet34LF_GN(len(self.classes), gate_strength=cfg.gate_strength, gn_groups=cfg.gn_groups).to(self.device)
        self.c2_ema = ResNet34LF_GN(len(self.classes), gate_strength=cfg.gate_strength, gn_groups=cfg.gn_groups).to(self.device)
        self.c2_ema.load_state_dict(self.c2.state_dict(), strict=True)

        # C1 guard rail (external frozen classifier). Optional.
        self.c1 = None
        self._c1_last = (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
        if getattr(cfg, "c1_ckpt", None):
            try:
                self.c1 = load_c1_classifier(
                    ckpt_path=Path(cfg.c1_ckpt),
                    num_classes=len(self.classes),
                    device=self.device,
                    gate_strength=cfg.gate_strength,
                    gn_groups=cfg.gn_groups,
                )
                print(f"[C1] guard rail loaded: {cfg.c1_ckpt}")
            except Exception as e:
                print(f"[C1] failed to load guard rail from {cfg.c1_ckpt} -> disabling. Error: {e}")
                self.c1 = None
        else:
            print("[C1] guard rail disabled (no --c1_ckpt provided)")

        # Warm-start C2 from the external C1 classifier when available:
        # copy backbone + classifier head, keep watermark detector/gate params fresh.
        if (self.c1 is not None) and bool(getattr(cfg, "c2_init_from_c1", True)):
            try:
                src_sd = unwrap(self.c1).state_dict()
                dst_sd = unwrap(self.c2).state_dict()
                copied = 0
                skipped = 0
                for k, v in src_sd.items():
                    if not k.startswith("base."):
                        continue
                    # Keep C2 classifier head fresh: copying base.fc from C1 preserves C1 class bias/collapse
                    # into epoch-1 validation.
                    if k.startswith("base.fc."):
                        skipped += 1
                        continue
                    if (k in dst_sd) and (tuple(dst_sd[k].shape) == tuple(v.shape)):
                        dst_sd[k] = v.detach().clone()
                        copied += 1
                    else:
                        skipped += 1
                unwrap(self.c2).load_state_dict(dst_sd, strict=False)
                unwrap(self.c2_ema).load_state_dict(unwrap(self.c2).state_dict(), strict=True)
                print(f"[C2 INIT] warm-started from C1 backbone only: copied={copied} skipped={skipped} (base.fc/wm_head/wm_affine kept fresh)")
            except Exception as e:
                print(f"[C2 INIT] warm-start from C1 failed; continuing from random init. Error: {e}")

        if cfg.use_dataparallel and self.device.type == "cuda" and torch.cuda.device_count() > 1:
            self.mask_lat = nn.DataParallel(self.mask_lat)
            self.mask_64 = nn.DataParallel(self.mask_64)
            self.g_lat = nn.DataParallel(self.g_lat)
            self.g_64 = nn.DataParallel(self.g_64)
            self.freq_ctrl = nn.DataParallel(self.freq_ctrl)
            self.c2 = nn.DataParallel(self.c2)
            self.c2_ema = nn.DataParallel(self.c2_ema)

        # optimizers
        g_params = list(unwrap(self.mask_lat).parameters()) + list(unwrap(self.mask_64).parameters()) + list(unwrap(self.g_lat).parameters()) + list(unwrap(self.g_64).parameters()) + list(unwrap(self.freq_ctrl).parameters())
        self.opt_g = torch.optim.AdamW(g_params, lr=cfg.lr_g, weight_decay=cfg.wd)

        self.opt_c2 = torch.optim.AdamW(unwrap(self.c2).parameters(), lr=cfg.lr_c2, weight_decay=cfg.wd)

        self.ctrl = Controller(
            eps=float(getattr(cfg, "ctrl_init_eps", 0.10) or 0.10),
            r_skip=float(getattr(cfg, "ctrl_init_r_skip", 0.66) or 0.66),
            ema_delta=0.0,
            ema_wm_gap=0.0,
        )
        self.ctrl.eps = float(min(max(self.ctrl.eps, 0.02), float(getattr(cfg, "ctrl_eps_max", 0.12) or 0.12)))
        self.ctrl.r_skip = float(min(max(self.ctrl.r_skip, float(getattr(cfg, "val_r_skip_min", 0.60) or 0.60)),
                                     float(getattr(cfg, "val_r_skip_max", 0.75) or 0.75)))

        # Optional train-time resume / warm-start:
        # --system_ckpt loads watermarking system (ROI + generators + controller),
        # --c2_eval_ckpt loads C2 / EMA / opt_c2,
        # --opt_g_ckpt loads opt_g.
        if getattr(cfg, "system_ckpt", None):
            self.load_system_ckpt(cfg.system_ckpt)
        else:
            print(f"[CTRL INIT] train-start eps={self.ctrl.eps:.4f} r_skip={self.ctrl.r_skip:.2f} (from CLI/defaults)")
        if getattr(cfg, "c2_eval_ckpt", None):
            self.load_c2_eval_ckpt(cfg.c2_eval_ckpt)
        if getattr(cfg, "opt_g_ckpt", None):
            self.load_opt_g_ckpt(cfg.opt_g_ckpt)

        # artifact folders
        ensure_dir(self.art_root / "collages")
        ensure_dir(self.art_root / "collages_pub")
        ensure_dir(self.art_root / "dataset_watermarked_full")
        print(f"[WM EXPORT] {'enabled' if bool(cfg.export_wm_dataset) else 'disabled'} -> {self.art_root / 'dataset_watermarked_full'}")
        if bool(getattr(cfg, "pub_collage_enable", True)) and int(getattr(cfg, "pub_collage_per_class", 0) or 0) > 0:
            print(f"[PUB COLLAGE] enabled -> {self.art_root / 'collages_pub'} | per_class={int(getattr(cfg, 'pub_collage_per_class', 10) or 10)}")
        else:
            print("[PUB COLLAGE] disabled")

        self._last_val_probe_stats: Dict[str, float] = {
            "acc_raw_g": float("nan"), "acc_both_g": float("nan"), "gap_g": float("nan"),
            "acc_raw_ng": float("nan"), "acc_both_ng": float("nan"), "gap_ng": float("nan"),
            "gate_specific_gap": float("nan"),
        }

        print(f"[RUN FILE] {Path(__file__).resolve() if '__file__' in globals() else 'unknown'}")
        print(
            f"[RUN CFG] gate_strength={self.cfg.gate_strength:.2f} train_prod_mix_prob={self.cfg.train_prod_mix_prob:.2f} "
            f"band_mid_floor={self.cfg.band_mid_floor:.2f} spec_lambda={self.cfg.spec_lambda:.3f} "
            f"val_probe_every={int(self.cfg.val_probe_every)} val_probe_batches={int(self.cfg.val_probe_batches)} "
            f"best_gap_floor={float(getattr(self.cfg, 'best_realval_gap_floor', 0.0)):.3f} "
            f"best_gs_floor={float(getattr(self.cfg, 'best_realval_gate_specific_floor', 0.0)):.3f}"
        )

        ensure_dir(self.ckpt_root / "meta")
        self._run_started_utc = utc_now_iso()
        self._trainer_file = Path(__file__).resolve() if "__file__" in globals() else None
        self._trainer_sha256 = sha256_file(self._trainer_file)
        self._ae_ckpt_sha256 = sha256_file(cfg.ae_ckpt)
        self._c1_ckpt_sha256 = sha256_file(cfg.c1_ckpt) if getattr(cfg, "c1_ckpt", None) else None
        self._last_val_stats: Optional[Dict[str, object]] = None
        self._last_realval_stats: Optional[Dict[str, object]] = None
        self._best_realval_record: Optional[Dict[str, object]] = None
        try:
            best_p = self.ckpt_root / "best_checkpoint_summary.json"
            if best_p.exists():
                self._best_realval_record = json.loads(best_p.read_text(encoding="utf-8"))
        except Exception:
            self._best_realval_record = None

        self._wm_export_manifests: Dict[Tuple[str, int], Dict[str, object]] = {}
        self._collage_export_manifests: Dict[Tuple[str, int], Dict[str, object]] = {}

        split_mode = "consistency" if bool(getattr(cfg, "clean_consistency", False)) else "keygap"
        print(f"[SPLIT MODE] clean_consistency={int(bool(getattr(cfg, 'clean_consistency', False)))} -> {split_mode}")

        # real-val automatic stop state
        self._real_val_bad_epochs = 0
        self._real_val_stop_reason: Optional[str] = None

    # ---------- inference runtime ----------

    def load_system_ckpt(self, system_ckpt: Path) -> None:
        """Load watermarking system weights (ROI masks + pattern generators + controller)."""
        system_ckpt = Path(system_ckpt)
        if not system_ckpt.exists():
            raise RuntimeError(f"System checkpoint not found: {system_ckpt}")
        ckpt = torch.load(system_ckpt, map_location=self.device)

        # Controller
        ctrl = ckpt.get("ctrl", {}) or {}
        for k in ("eps", "r_skip", "ema_delta", "ema_wm_gap", "ema_acc_gap_g", "ema_gate_spec_gap"):
            if k in ctrl:
                try:
                    setattr(self.ctrl, k, float(ctrl[k]))
                except Exception:
                    pass

        # Models
        if "mask_lat" in ckpt:
            unwrap(self.mask_lat).load_state_dict(ckpt["mask_lat"], strict=True)
        if "mask_64" in ckpt:
            unwrap(self.mask_64).load_state_dict(ckpt["mask_64"], strict=True)
        if "g_lat" in ckpt:
            unwrap(self.g_lat).load_state_dict(ckpt["g_lat"], strict=True)
        if "g_64" in ckpt:
            unwrap(self.g_64).load_state_dict(ckpt["g_64"], strict=True)
        if "freq_ctrl" in ckpt:
            unwrap(self.freq_ctrl).load_state_dict(ckpt["freq_ctrl"], strict=True)

        self._loaded_system_ckpt = str(system_ckpt)

        print(f"[CKPT] loaded system <- {system_ckpt}")
        print(f"[CTRL] eps={getattr(self.ctrl, 'eps', float('nan')):.4f} r_skip={getattr(self.ctrl, 'r_skip', float('nan')):.2f}")

    def load_c2_eval_ckpt(self, c2_ckpt: Path) -> None:
        """Load C2 / EMA / opt_c2 checkpoint for train-time resume."""
        c2_ckpt = Path(c2_ckpt)
        if not c2_ckpt.exists():
            raise RuntimeError(f"C2 checkpoint not found: {c2_ckpt}")
        ckpt = torch.load(c2_ckpt, map_location=self.device)

        if "c2" in ckpt:
            unwrap(self.c2).load_state_dict(ckpt["c2"], strict=True)
        if "c2_ema" in ckpt:
            unwrap(self.c2_ema).load_state_dict(ckpt["c2_ema"], strict=True)
        else:
            unwrap(self.c2_ema).load_state_dict(unwrap(self.c2).state_dict(), strict=True)
        if "opt_c2" in ckpt:
            self.opt_c2.load_state_dict(ckpt["opt_c2"])

        print(f"[CKPT] loaded c2_eval <- {c2_ckpt}")

    def load_opt_g_ckpt(self, g_ckpt: Path) -> None:
        """Load generator optimizer state for train-time resume."""
        g_ckpt = Path(g_ckpt)
        if not g_ckpt.exists():
            raise RuntimeError(f"Generator optimizer checkpoint not found: {g_ckpt}")
        ckpt = torch.load(g_ckpt, map_location=self.device)
        if "opt_g" not in ckpt:
            raise RuntimeError(f"Checkpoint does not contain opt_g: {g_ckpt}")
        self.opt_g.load_state_dict(ckpt["opt_g"])
        print(f"[CKPT] loaded opt_g <- {g_ckpt}")

    def _init_infer_runtime(self) -> None:
        """Init minimal runtime for applying watermark outside of training."""
        cfg = self.cfg

        # No class folders required.
        self.classes = []

        # transform
        self.tfm = PadToSquareNoUpscale(size=cfg.image_size, pad_value=cfg.pad_value)

        # AE
        self.ae = load_ae(cfg.ae_ckpt, cfg.ae_module, cfg.ae_class, cfg.ae_py_path, self.device).to(self.device)
        unwrap(self.ae).eval()

        # ROI + patterns (same arch as training)
        self.mask_lat = MiniUNetMask(1024, mid=64).to(self.device)
        self.mask_64 = MiniUNetMask(512, mid=64).to(self.device)
        self.g_lat = GLat(1024).to(self.device)
        self.g_64 = G64(512).to(self.device)
        self.freq_ctrl = FreqController(in_dim=6, hidden=32).to(self.device)
        self.lowtex = LocalVariance(k=9).to(self.device)

        # DP (optional)
        if cfg.use_dataparallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.mask_lat = nn.DataParallel(self.mask_lat)
            self.mask_64 = nn.DataParallel(self.mask_64)
            self.g_lat = nn.DataParallel(self.g_lat)
            self.g_64 = nn.DataParallel(self.g_64)
            self.freq_ctrl = nn.DataParallel(self.freq_ctrl)

        # controller
        self.ctrl = Controller()

        # artifacts / outputs
        ensure_dir(self.art_root / "inference")

        # load system ckpt (required)
        if not getattr(cfg, "system_ckpt", None):
            raise RuntimeError("Infer mode requires --system_ckpt (wm_system_eXXX.pth)")
        self.load_system_ckpt(cfg.system_ckpt)

        # eval mode
        unwrap(self.mask_lat).eval()
        unwrap(self.mask_64).eval()
        unwrap(self.g_lat).eval()
        unwrap(self.g_64).eval()
        unwrap(self.freq_ctrl).eval()
        unwrap(self.freq_ctrl).eval()

        # infer paths
        infer_paths: List[Path] = []
        if getattr(cfg, "infer_list", None):
            p_list = Path(cfg.infer_list)
            if not p_list.exists():
                raise RuntimeError(f"infer_list not found: {p_list}")
            with open(p_list, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    infer_paths.append(Path(s))
        elif getattr(cfg, "infer_root", None):
            infer_root = Path(cfg.infer_root)
            self._infer_root = infer_root
            ds = DiskImageFolderWithPathsAndMask(infer_root, self.tfm, max_images=int(getattr(cfg, "infer_max_images", 0) or 0))
            self.infer_loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)
            # output root
            out_root = Path(cfg.infer_out) if getattr(cfg, "infer_out", None) else (cfg.out_root / "inference")
            self.infer_out_root = out_root
            ensure_dir(out_root)
            return
        else:
            raise RuntimeError("Infer mode requires --infer_root or --infer_list")

        # list-based dataset
        ds = DiskPathListWithMask(infer_paths, self.tfm)
        self.infer_loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        out_root = Path(cfg.infer_out) if getattr(cfg, "infer_out", None) else (cfg.out_root / "inference")
        self.infer_out_root = out_root
        ensure_dir(out_root)

    def save_infer_batch(self, orig01: torch.Tensor, base01: torch.Tensor, wm01: torch.Tensor, valid_mask: torch.Tensor, paths: List[str]) -> None:
        """Save inference outputs. Always crops padding out of the exported image."""
        cfg = self.cfg
        fmt = str(getattr(cfg, "wm_export_format", "png") or "png").lower().strip()
        ext = "png" if fmt == "png" else "jpg"
        q = int(getattr(cfg, "wm_jpeg_quality", 92))

        suffix = str(getattr(cfg, "infer_suffix", "_watermarked") or "_watermarked")

        # optional extra outputs
        save_base = bool(getattr(cfg, "infer_save_base", False))
        save_diff = bool(getattr(cfg, "infer_save_diff", False))

        vm_cpu = valid_mask.detach().cpu()

        for i, (o, b, w, vm_i, pth) in enumerate(zip(orig01, base01, wm01, vm_cpu, paths)):
            src = Path(pth)
            # preserve relative structure if infer_root is set and the file is under it
            rel = None
            try:
                if hasattr(self, "_infer_root") and self._infer_root and src.is_absolute():
                    rel = src.relative_to(self._infer_root)
                elif hasattr(self, "_infer_root") and self._infer_root:
                    rel = src.relative_to(self._infer_root)
            except Exception:
                rel = None

            out_dir = self.infer_out_root
            if rel is not None and rel.parent != Path("."):
                out_dir = out_dir / rel.parent
            ensure_dir(out_dir)

            stem = src.stem
            out_wm = out_dir / f"{stem}{suffix}.{ext}"
            out_b = out_dir / f"{stem}_base.{ext}"
            out_d = out_dir / f"{stem}_diff.{ext}"

            def _crop(im01: torch.Tensor, vm_i: torch.Tensor) -> torch.Tensor:
                m2 = vm_i.squeeze(0)
                ys, xs = torch.where(m2 > 0.5)
                if ys.numel() == 0:
                    return im01
                y0 = int(ys.min().item());
                y1 = int(ys.max().item()) + 1
                x0 = int(xs.min().item());
                x1 = int(xs.max().item()) + 1
                y0 = max(0, y0);
                x0 = max(0, x0)
                y1 = min(im01.size(1), y1);
                x1 = min(im01.size(2), x1)
                if (y1 - y0) <= 0 or (x1 - x0) <= 0:
                    return im01
                return im01[:, y0:y1, x0:x1]

            def _save(im01: torch.Tensor, out_path: Path):
                im01 = im01.detach().clamp(0, 1)
                im01 = _crop(im01, vm_i)
                arr = (im01.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                pil = Image.fromarray(arr, mode="RGB")
                if fmt == "png":
                    pil.save(out_path, format="PNG", compress_level=3)
                else:
                    pil.save(out_path, format="JPEG", quality=q, subsampling=0, optimize=False)

            _save(w, out_wm)
            if save_base:
                _save(b, out_b)
            if save_diff:
                # visualize delta in luma (centered at 0.5)
                dy = self._luma01(w[None]) - self._luma01(o[None])
                dy = dy.squeeze(0)  # [1,H,W]
                clip = float(getattr(cfg, "wm_res_clip", 0.10) or 0.10)
                vis = (0.5 + dy / (2.0 * clip)).clamp(0, 1).repeat(3, 1, 1)
                _save(vis, out_d)

    @torch.no_grad()
    def infer(self) -> None:
        """Apply watermark to images in infer_root/infer_list and save results."""
        if getattr(self, "mode", "train") != "infer":
            print("[WARN] infer() called but cfg.mode != infer; running anyway.")
        if not hasattr(self, "infer_loader"):
            raise RuntimeError("Infer loader is not initialized.")

        self.ae.eval()
        unwrap(self.mask_lat).eval()
        unwrap(self.mask_64).eval()
        unwrap(self.g_lat).eval()
        unwrap(self.g_64).eval()

        it = 0
        for batch in self.infer_loader:
            xN, valid_mask, paths = batch
            xN = xN.to(self.device)
            valid_mask = valid_mask.to(self.device)

            x01 = self._to01(xN)

            syn = self.synth_variants_nograd(
                x01,
                valid_mask=valid_mask,
                epoch=0,
                variants=("base", "both"),
                k_factor=1.0,
                varpercent=False,
                mode="infer",
            )
            base01 = syn["base01"].detach()
            both01 = syn["both01"].detach()

            # enforce exact outside-valid copy (pad stays in tensors, but export is cropped)
            base01 = self._copy_outside_valid(base01, x01, valid_mask)
            both01 = self._copy_outside_valid(both01, x01, valid_mask)

            self.save_infer_batch(x01.detach(), base01, both01, valid_mask, paths)

            it += 1
            if (it % 10) == 0:
                print(f"[INFER] batches={it} images~{it * int(self.cfg.batch_size)} -> {self.infer_out_root}")

    # ---------- tensor helpers ----------

    @staticmethod
    def _to01(xN: torch.Tensor) -> torch.Tensor:
        return (xN + 1.0) * 0.5

    @staticmethod
    def _luma01(x01: torch.Tensor) -> torch.Tensor:
        return 0.299 * x01[:, 0:1] + 0.587 * x01[:, 1:2] + 0.114 * x01[:, 2:3]

    @staticmethod
    def _rgb_to_cbcr(x01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Approximate Cb/Cr from RGB in [0,1]. Returns (Cb, Cr), each [B,1,H,W]."""
        r, g, b = x01[:, 0:1], x01[:, 1:2], x01[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 0.564 * (b - y)
        cr = 0.713 * (r - y)
        return cb, cr

    @staticmethod
    def _force_gray(x01: torch.Tensor) -> torch.Tensor:
        y = 0.299 * x01[:, 0:1] + 0.587 * x01[:, 1:2] + 0.114 * x01[:, 2:3]
        return y.repeat(1, 3, 1, 1)

    def _colorfulness(self, x01: torch.Tensor) -> torch.Tensor:
        rg = (x01[:, 0:1] - x01[:, 1:2]).abs()
        rb = (x01[:, 0:1] - x01[:, 2:3]).abs()
        gb = (x01[:, 1:2] - x01[:, 2:3]).abs()
        c = (rg + rb + gb).mean(dim=(2, 3), keepdim=True) / 3.0
        return c

    def _gray_like_mask(self, x01: torch.Tensor) -> torch.Tensor:
        if not self.cfg.auto_switch:
            return torch.zeros((x01.size(0), 1, 1, 1), device=x01.device, dtype=x01.dtype)
        eps = float(self.cfg.gray_like_eps)
        c = self._colorfulness(x01)
        return (c < eps).to(x01.dtype)

    def _maybe_force_gray(self, out01: torch.Tensor, gray_mask: torch.Tensor) -> torch.Tensor:
        if not self.cfg.force_gray_output:
            return out01
        if gray_mask is None:
            return out01
        g = gray_mask
        if g.ndim != 4:
            g = g.view(-1, 1, 1, 1)
        if float(g.max().item()) <= 0.0:
            return out01
        out_gray = self._force_gray(out01)
        return out01 * (1.0 - g) + out_gray * g

    def _soft_roi_mask(self, m: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if m is None:
            return None
        k = int(getattr(self.cfg, "soft_roi_blur_k", 0) or 0)
        if k < 3:
            return m
        if (k % 2) == 0:
            k += 1
        s = F.avg_pool2d(m, k, 1, k // 2)
        return s.clamp(0.0, 1.0)

    def _masked_mean_per_image(self, t: torch.Tensor, m: Optional[torch.Tensor]) -> torch.Tensor:
        if m is None:
            return t.mean(dim=(1, 2, 3), keepdim=False)
        if m.dim() == 3:
            m = m[:, None, :, :]
        if m.shape[-2:] != t.shape[-2:]:
            m = F.interpolate(m.to(device=t.device, dtype=t.dtype), size=t.shape[-2:], mode="nearest")
        denom = m.sum(dim=(1, 2, 3)).clamp_min(1.0)
        num = (t * m).sum(dim=(1, 2, 3))
        return num / denom

    def _compute_bandmix(self, x01: torch.Tensor, base01: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        B = x01.size(0)
        y = self._luma01(x01)
        yb = self._luma01(x01)
        vm = valid_mask if valid_mask is not None else torch.ones((B, 1, y.size(-2), y.size(-1)), device=y.device, dtype=y.dtype)
        if vm.dim() == 3:
            vm = vm[:, None, :, :]
        tex = (y - F.avg_pool2d(y, 5, 1, 2)).abs()
        tex_m = self._masked_mean_per_image(tex, vm).view(B, 1)
        dark_thr = float(getattr(self.cfg, "bg_protect_thr", 0.06) or 0.06)
        white_thr = float(getattr(self.cfg, "avoid_white_thr", 1.01) or 1.01)
        if not (white_thr > 0.0 and white_thr <= 1.0):
            white_thr = 0.98
        dark_frac = self._masked_mean_per_image((y < dark_thr).to(y.dtype), vm).view(B, 1)
        bright_frac = self._masked_mean_per_image((y > white_thr).to(y.dtype), vm).view(B, 1)
        headroom = torch.minimum(1.0 - yb, yb)
        headroom_m = self._masked_mean_per_image(headroom, vm).view(B, 1)
        mean_luma = self._masked_mean_per_image(y, vm).view(B, 1)
        gray_m = self._gray_like_mask(x01).view(B, 1)
        feat = torch.cat([tex_m, dark_frac, bright_frac, headroom_m, mean_luma, gray_m], dim=1)
        logits = self.freq_ctrl(feat)
        temp = float(getattr(self.cfg, "freq_ctrl_temp", 1.0) or 1.0)
        temp = max(0.25, temp)
        w = F.softmax(logits / temp, dim=1)
        mid_floor = float(getattr(self.cfg, "band_mid_floor", 0.0) or 0.0)
        mid_floor = max(0.0, min(0.49, mid_floor))
        if mid_floor > 0.0:
            w_mid = mid_floor + (1.0 - 2.0 * mid_floor) * w[:, 1:2]
            w_low = 1.0 - w_mid
            w = torch.cat([w_low, w_mid], dim=1)
        dbg = {
            "band_low": float(w[:, 0].mean().detach().item()),
            "band_mid": float(w[:, 1].mean().detach().item()),
            "tex": float(tex_m.mean().detach().item()),
            "headroom": float(headroom_m.mean().detach().item()),
        }
        return w[:, 0:1].view(B, 1, 1, 1), w[:, 1:2].view(B, 1, 1, 1), dbg

    @staticmethod
    def _copy_outside_valid(out01: torch.Tensor, src01: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Replace pixels outside valid_mask with src01 (both in [0,1]).
        This prevents padded borders from becoming an information channel.
        """
        if valid_mask is None:
            return out01
        vm = valid_mask
        if vm.dim() == 3:
            vm = vm[:, None, :, :]
        if vm.size(2) != out01.size(2) or vm.size(3) != out01.size(3):
            vm = F.interpolate(vm.to(dtype=out01.dtype, device=out01.device), size=out01.shape[-2:], mode="nearest")
        vm3 = vm.repeat(1, out01.size(1), 1, 1)
        return out01 * vm3 + src01 * (1.0 - vm3)

    @staticmethod
    def _border_stats(res01: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> Tuple[float, float, float]:
        """Return (border_abs, valid_abs, border_ratio) for residual res01 in [0,1].
        res01: [B,3,H,W] (typically both01-base01).
        valid_mask: [B,1,H,W] in {0,1}.
        """
        if valid_mask is None:
            a = float(res01.abs().mean().item())
            return a, a, 1.0
        vm = valid_mask
        if vm.dim() == 3:
            vm = vm[:, None, :, :]
        vm3 = vm.repeat(1, res01.size(1), 1, 1)
        inv3 = (1.0 - vm3)
        # per-pixel magnitude averaged over channels
        mag = res01.abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
        inv = 1.0 - vm
        border_abs = float((mag * inv).sum().item() / inv.sum().clamp_min(1.0).item())
        valid_abs = float((mag * vm).sum().item() / vm.sum().clamp_min(1.0).item())
        border_ratio = float(border_abs / (valid_abs + 1e-8))
        return border_abs, valid_abs, border_ratio

    @staticmethod
    def _pad_mean01(x01: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> float:
        """Mean pixel value in padding area in [0,1] (averaged over B,C,H,W of padding)."""
        if valid_mask is None:
            return float('nan')
        vm = valid_mask
        if vm.dim() == 3:
            vm = vm[:, None, :, :]
        inv = (1.0 - vm)
        m3 = inv.repeat(1, x01.size(1), 1, 1)
        denom = m3.sum().clamp_min(1.0)
        return float((x01 * m3).sum().item() / denom.item())

    @staticmethod
    def _pad_meanN(xN: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> float:
        """Mean pixel value in padding area in [-1,1] (averaged over B,C,H,W of padding)."""
        if valid_mask is None:
            return float('nan')
        vm = valid_mask
        if vm.dim() == 3:
            vm = vm[:, None, :, :]
        inv = (1.0 - vm)
        m3 = inv.repeat(1, xN.size(1), 1, 1)
        denom = m3.sum().clamp_min(1.0)
        return float((xN * m3).sum().item() / denom.item())

    @staticmethod
    def _valid_frac(valid_mask: Optional[torch.Tensor]) -> float:
        if valid_mask is None:
            return 1.0
        return float(valid_mask.mean().item())

    @staticmethod
    def _rms_vis01(t: torch.Tensor) -> torch.Tensor:
        """Project multi-channel pattern to 1ch RMS and normalize to [0,1]."""
        # t: [B,C,H,W]
        v = torch.sqrt((t * t).mean(dim=1, keepdim=True) + 1e-12)
        lo = torch.quantile(v.flatten(1), 0.01, dim=1, keepdim=True).view(-1, 1, 1, 1)
        hi = torch.quantile(v.flatten(1), 0.99, dim=1, keepdim=True).view(-1, 1, 1, 1)
        return torch.clamp((v - lo) / (hi - lo + 1e-8), 0, 1)

    # ---------- top-k mask (valid-aware) ----------

    @staticmethod
    def topk_binary_mask(soft: torch.Tensor, keep: float, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """soft: [B,1,H,W], keep fraction in (0,1]. valid: [B,1,H,W] in {0,1} (optional)."""
        B, _, H, W = soft.shape
        flat = soft.view(B, -1)

        if valid is None:
            k = max(1, int(round(float(keep) * H * W)))
            out = torch.zeros_like(flat)
            for b in range(B):
                idx = torch.topk(flat[b], k, sorted=False).indices
                out[b, idx] = 1.0
            return out.view(B, 1, H, W)

        v = (valid.view(B, -1) > 0.5)
        flat_masked = flat.masked_fill(~v, -1e9)
        n_valid = v.sum(dim=1).clamp_min(1)
        k_each = torch.clamp((n_valid.float() * float(keep)).round().to(torch.int64), min=1)

        out = torch.zeros_like(flat)
        for b in range(B):
            k = int(k_each[b].item())
            idx = torch.topk(flat_masked[b], k, sorted=False).indices
            out[b, idx] = 1.0
        return out.view(B, 1, H, W)

    # ---------- teacher masks (dynamic sizes + valid-aware) ----------

    @torch.no_grad()
    def build_masks_teacher(
            self,
            x01: torch.Tensor,
            valid_mask: Optional[torch.Tensor],
            keep_lat: float,
            keep_64: float,
            lat_hw: Tuple[int, int],
            skip_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Texture-based teacher: latent prefers low texture, skip prefers high texture."""
        y = self._luma01(x01)
        var = self.lowtex(y)

        # normalize var per sample
        vmax = var.amax(dim=(2, 3), keepdim=True).clamp_min(1e-8)
        var_n = var / vmax

        # valid masks downsample
        if valid_mask is not None:
            vm_lat = F.interpolate(valid_mask, size=lat_hw, mode="nearest")
            vm_skip = F.interpolate(valid_mask, size=skip_hw, mode="nearest")
        else:
            vm_lat = vm_skip = None

        # low texture map (higher = lower texture)
        low = 1.0 - var_n
        low_lat = F.adaptive_avg_pool2d(low, lat_hw)
        hi_skip = F.adaptive_avg_pool2d(var_n, skip_hw)

        # topk: latent picks low texture => top-k on low_lat
        # skip picks high texture => top-k on hi_skip
        T_lat = self.topk_binary_mask(low_lat, keep_lat, vm_lat)
        T_skip = self.topk_binary_mask(hi_skip, keep_64, vm_skip)
        return T_lat, T_skip

    # ---------- ROI builder (trainable) ----------

    def build_roi_trainable(
            self,
            Z: torch.Tensor,
            S64: torch.Tensor,
            x01: torch.Tensor,
            valid_mask: Optional[torch.Tensor],
            epoch: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
        cfg = self.cfg

        P_lat_soft = self.mask_lat(Z)
        P_64_soft = self.mask_64(S64)

        lat_hw = (int(P_lat_soft.size(-2)), int(P_lat_soft.size(-1)))
        skip_hw = (int(P_64_soft.size(-2)), int(P_64_soft.size(-1)))

        # downsample valid mask
        vm_lat = F.interpolate(valid_mask, size=lat_hw, mode="nearest") if valid_mask is not None else None
        vm_skip = F.interpolate(valid_mask, size=skip_hw, mode="nearest") if valid_mask is not None else None

        # optional content mask to avoid placing ROI in near-black flat background
        cm_full = None
        if (valid_mask is not None) and (cfg.avoid_black_thr is not None) and (float(cfg.avoid_black_thr) > 0.0):
            y01 = _to_luma01(x01)  # [B,1,H,W]
            cm_full = (y01 > float(cfg.avoid_black_thr)).float() * valid_mask
            thr_w = float(getattr(cfg, 'avoid_white_thr', 1.01) or 1.01)
            if thr_w <= 1.0:
                cm_full = cm_full * (y01 < thr_w).float()

        else:
            cm_full = valid_mask

        cm_lat = F.interpolate(cm_full, size=lat_hw, mode="nearest") if cm_full is not None else None
        cm_skip = F.interpolate(cm_full, size=skip_hw, mode="nearest") if cm_full is not None else None

        with torch.no_grad():
            P_lat_h = self.topk_binary_mask(P_lat_soft, cfg.keep_lat, cm_lat)
            P_64_h = self.topk_binary_mask(P_64_soft, cfg.keep_64, cm_skip)

        # STE
        P_lat = P_lat_h + (P_lat_soft - P_lat_soft.detach())
        P_64 = P_64_h + (P_64_soft - P_64_soft.detach())

        # area loss (valid-aware)
        def mean_valid(m: torch.Tensor, vm: Optional[torch.Tensor]) -> torch.Tensor:
            if vm is None:
                return m.mean(dim=(2, 3))
            denom = vm.sum(dim=(2, 3)).clamp_min(1.0)
            return (m * vm).sum(dim=(2, 3)) / denom

        area_lat = mean_valid(P_lat_soft, vm_lat)
        area_64 = mean_valid(P_64_soft, vm_skip)
        L_area = (area_lat - cfg.keep_lat).abs().mean() + (area_64 - cfg.keep_64).abs().mean()

        # overlap loss in skip resolution
        P_lat_to_skip = F.interpolate(P_lat_soft, size=skip_hw, mode="bilinear", align_corners=False)
        overlap = P_lat_to_skip * P_64_soft
        if vm_skip is not None:
            overlap = overlap * vm_skip
        L_overlap = overlap.mean()

        # TV
        def tv(m: torch.Tensor, vm: Optional[torch.Tensor]) -> torch.Tensor:
            # simple, lightly valid-aware (mask diffs)
            dy = (m[:, :, 1:, :] - m[:, :, :-1, :]).abs()
            dx = (m[:, :, :, 1:] - m[:, :, :, :-1]).abs()
            if vm is None:
                return dy.mean() + dx.mean()
            vmy = vm[:, :, 1:, :] * vm[:, :, :-1, :]
            vmx = vm[:, :, :, 1:] * vm[:, :, :, :-1]
            return (dy * vmy).sum() / vmy.sum().clamp_min(1.0) + (dx * vmx).sum() / vmx.sum().clamp_min(1.0)

        L_tv = tv(P_lat_soft, vm_lat) + tv(P_64_soft, vm_skip)

        # sparsity + binarization
        if vm_lat is None:
            L_sparse = P_lat_soft.mean() + P_64_soft.mean()
            L_bin = (P_lat_soft * (1 - P_lat_soft)).mean() + (P_64_soft * (1 - P_64_soft)).mean()
        else:
            L_sparse = _masked_mean(P_lat_soft, vm_lat) + _masked_mean(P_64_soft, vm_skip)
            L_bin = _masked_mean(P_lat_soft * (1 - P_lat_soft), vm_lat) + _masked_mean(P_64_soft * (1 - P_64_soft), vm_skip)

        # texture prior
        with torch.no_grad():
            y = self._luma01(x01)
            var = self.lowtex(y)
            vmax = var.amax(dim=(2, 3), keepdim=True).clamp_min(1e-8)
            var_n = var / vmax
            low = 1.0 - var_n
            low_lat = F.adaptive_avg_pool2d(low, lat_hw)
            hi_skip = F.adaptive_avg_pool2d(var_n, skip_hw)

        # penalize selecting opposite texture
        tex_lat = P_lat_soft * (1.0 - low_lat)
        tex_skip = P_64_soft * (1.0 - hi_skip)
        if vm_lat is not None:
            tex_lat = tex_lat * vm_lat
            tex_skip = tex_skip * vm_skip
        L_tex = tex_lat.mean() + tex_skip.mean()

        # teacher warmup
        if cfg.roi_teach_epochs > 0 and epoch <= cfg.roi_teach_epochs:
            with torch.no_grad():
                T_lat, T_skip = self.build_masks_teacher(x01, valid_mask, cfg.keep_lat, cfg.keep_64, lat_hw, skip_hw)
            L_teacher = F.binary_cross_entropy(P_lat_soft, T_lat) + F.binary_cross_entropy(P_64_soft, T_skip)
        else:
            L_teacher = P_lat_soft.new_tensor(0.0)

        roi_losses = {
            "L_area": L_area,
            "L_overlap": L_overlap,
            "L_tv": L_tv,
            "L_sparse": L_sparse,
            "L_bin": L_bin,
            "L_tex": L_tex,
            "L_teacher": L_teacher,
        }
        roi_dbg = {
            "roi_lat_mean": float(area_lat.mean().item()),
            "roi_64_mean": float(area_64.mean().item()),
            "roi_overlap": float(L_overlap.item()),
        }
        return P_lat, P_64, roi_losses, roi_dbg

    @torch.no_grad()
    def build_roi_trainable_nograd(
            self, Z: torch.Tensor, S64: torch.Tensor, x01: torch.Tensor, valid_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        P_lat_soft = self.mask_lat(Z)
        P_64_soft = self.mask_64(S64)
        lat_hw = (int(P_lat_soft.size(-2)), int(P_lat_soft.size(-1)))
        skip_hw = (int(P_64_soft.size(-2)), int(P_64_soft.size(-1)))
        vm_lat = F.interpolate(valid_mask, size=lat_hw, mode="nearest") if valid_mask is not None else None
        vm_skip = F.interpolate(valid_mask, size=skip_hw, mode="nearest") if valid_mask is not None else None
        cm_full = valid_mask
        y01 = _to_luma01(x01)
        thr_b = float(getattr(self.cfg, "avoid_black_thr", 0.0) or 0.0)
        if thr_b > 0.0 and cm_full is not None:
            cm_full = cm_full * (y01 > thr_b).float()
        thr_w = float(getattr(self.cfg, "avoid_white_thr", 1.01) or 1.01)
        if thr_w <= 1.0 and cm_full is not None:
            cm_full = cm_full * (y01 < thr_w).float()
        cm_lat = F.interpolate(cm_full, size=lat_hw, mode="nearest") if cm_full is not None else vm_lat
        cm_skip = F.interpolate(cm_full, size=skip_hw, mode="nearest") if cm_full is not None else vm_skip
        P_lat = self.topk_binary_mask(P_lat_soft, self.cfg.keep_lat, cm_lat)
        P_64 = self.topk_binary_mask(P_64_soft, self.cfg.keep_64, cm_skip)
        return P_lat, P_64

    # ---------- luma-only embedding helper ----------

    def _embed_luma_only_gray(
            self,
            x01: torch.Tensor,
            base01: torch.Tensor,
            wm_lat: Optional[torch.Tensor],
            wm_skip: Optional[torch.Tensor],
            alpha_lat: float | torch.Tensor,
            alpha_skip: float | torch.Tensor,
            roi_lat: Optional[torch.Tensor],
            roi_skip: Optional[torch.Tensor],
            valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Embed watermark in AE latent/skip space but use the ORIGINAL image as the carrier.

        Variant 2: AE remains the watermark operator (enc/latent/skip -> embed_external_wm_gray),
        while the final residual is defined against the original luma and applied back onto x01.
        AE reconstruction (base01) stays available for diagnostics/collages only.
        Also performs *masked* residual shaping to avoid gray veils / background lift while preserving
        the overall clean→wm delta budget (RMS of delta on the valid region).
        """
        cfg = self.cfg

        def _ensure_mask(m: Optional[torch.Tensor], ref: torch.Tensor) -> Optional[torch.Tensor]:
            if m is None:
                return None
            if m.dim() == 3:
                m = m[:, None, :, :]
            if m.size(-2) != ref.size(-2) or m.size(-1) != ref.size(-1):
                m = F.interpolate(m.to(dtype=ref.dtype, device=ref.device), size=ref.shape[-2:], mode="nearest")
            return m.to(dtype=ref.dtype, device=ref.device)

        def _masked_box_blur(t: torch.Tensor, m: torch.Tensor, k: int) -> torch.Tensor:
            # Mask-aware box blur: blur(t*m) / blur(m)
            k = int(max(1, k))
            if k % 2 == 0:
                k += 1
            p = k // 2
            tm = t * m
            num = F.avg_pool2d(tm, k, 1, p)
            den = F.avg_pool2d(m, k, 1, p).clamp_min(1e-6)
            return num / den

        def _masked_mean(t: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            den = m.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
            return (t * m).sum(dim=(1, 2, 3), keepdim=True) / den

        def _masked_rms(t: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            den = m.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
            return torch.sqrt(((t * t) * m).sum(dim=(1, 2, 3), keepdim=True) / den + 1e-12)

        y_in = self._luma01(x01)
        y_wm = self.ae.embed_external_wm_gray(
            y_in,
            wm_lat=wm_lat,
            wm_skip=wm_skip,
            alpha_lat=alpha_lat,
            alpha_skip=alpha_skip,
            roi_lat_32=roi_lat,
            roi_skip_64=roi_skip,
            valid_mask=valid_mask,
        ).clamp(0, 1)
        y_carrier = self._luma01(x01)

        delta = (y_wm - y_carrier)

        # Build "change mask" cm: valid region AND optional content masks.
        cm = _ensure_mask(valid_mask, delta)
        if cm is None:
            cm = torch.ones_like(delta)

        # Optional: avoid embedding into near-black pixels (helps keep dark backgrounds clean).
        thr_b = float(getattr(cfg, "avoid_black_thr", 0.0) or 0.0)
        if thr_b > 0.0:
            yi = y_in
            if yi.size(-2) != delta.size(-2) or yi.size(-1) != delta.size(-1):
                yi = F.interpolate(yi, size=delta.shape[-2:], mode="bilinear", align_corners=False)
            cm = cm * (yi > thr_b).to(dtype=delta.dtype)

        # Optional: avoid embedding into near-white pixels (helps avoid highlight sparkle).
        thr_w = float(getattr(cfg, "avoid_white_thr", 1.01) or 1.01)
        if thr_w > 0.0 and thr_w <= 1.0:
            yi = y_in
            if yi.size(-2) != delta.size(-2) or yi.size(-1) != delta.size(-1):
                yi = F.interpolate(yi, size=delta.shape[-2:], mode="bilinear", align_corners=False)
            cm = cm * (yi < thr_w).to(dtype=delta.dtype)

        # If the usable embedding area collapses, fall back to the valid region instead of zeroing the watermark.
        vm_ref = _ensure_mask(valid_mask, delta)
        if vm_ref is None:
            vm_ref = torch.ones_like(delta)
        cm_sum = float(cm.sum().detach().item())
        vm_sum = float(vm_ref.sum().detach().item())
        if vm_sum > 0.0 and (cm_sum / (vm_sum + 1e-8)) < 0.01:
            cm = vm_ref

        # Always enforce: no residual outside cm.
        delta = delta * cm

        # --- delta shaping pipeline (prevents gray veils, preserves delta budget) ---
        # Cache pre-shaping budget (RMS in the valid region)
        rms_pre = _masked_rms(delta, cm).detach()
        mu_pre = _masked_mean(delta, cm).detach()

        # Minimum-strength hinge + adaptive alpha boost.
        # The scratch AE can be very insensitive to small latent/skip perturbations at the start.
        k_min = float(getattr(cfg, "min_delta_rms_k", 0.0) or 0.0)
        lam_min = float(getattr(cfg, "min_delta_lambda", 0.0) or 0.0)
        max_boost = float(getattr(cfg, "max_alpha_boost", 12.0) or 12.0)
        eps_eff = float(getattr(self.ctrl, "eps", 0.0) or 0.0)
        target = delta.new_tensor(k_min * eps_eff)
        rms_now = _masked_rms(delta, cm)
        boost = (target / (rms_now + 1e-12)).clamp(min=1.0, max=max_boost) if (k_min > 0.0 and eps_eff > 0.0) else delta.new_tensor(1.0)

        if float(boost.detach().max().item()) > 1.01 and ((wm_lat is not None) or (wm_skip is not None)):
            if isinstance(alpha_lat, (float, int)):
                alpha_lat_boost = float(alpha_lat) * float(boost.detach().mean().item())
            else:
                alpha_lat_boost = alpha_lat.to(device=delta.device, dtype=delta.dtype) * boost.to(device=delta.device, dtype=delta.dtype)
            if isinstance(alpha_skip, (float, int)):
                alpha_skip_boost = float(alpha_skip) * float(boost.detach().mean().item())
            else:
                alpha_skip_boost = alpha_skip.to(device=delta.device, dtype=delta.dtype) * boost.to(device=delta.device, dtype=delta.dtype)

            y_wm_boost = self.ae.embed_external_wm_gray(
                y_in,
                wm_lat=wm_lat,
                wm_skip=wm_skip,
                alpha_lat=alpha_lat_boost,
                alpha_skip=alpha_skip_boost,
                roi_lat_32=roi_lat,
                roi_skip_64=roi_skip,
                valid_mask=valid_mask,
            ).clamp(0, 1)
            delta = (y_wm_boost - y_carrier) * cm
            rms_pre = _masked_rms(delta, cm).detach()
            mu_pre = _masked_mean(delta, cm).detach()
            rms_now = _masked_rms(delta, cm)

        L_min = delta.new_tensor(0.0)
        self._last_embed_Lmin = L_min
        if lam_min > 0.0 and k_min > 0.0:
            L_min = lam_min * (F.relu(target - rms_now).pow(2)).mean()
        self._last_embed_Lmin = L_min

        # DC removal (kills global brightness lift)
        if bool(getattr(cfg, "delta_remove_dc", True)):
            delta = (delta - mu_pre) * cm

        auto_band = str(getattr(cfg, "profile", "legacy") or "legacy").lower().strip() == "auto_bandmix"

        # optional image-adaptive shaping / band selection
        beta = float(getattr(cfg, "delta_hp_beta", 0.0) or 0.0)
        bg_thr = float(getattr(cfg, "bg_protect_thr", 0.0) or 0.0)
        bg_scale = float(getattr(cfg, "bg_delta_scale", 1.0) or 1.0)
        band_low = delta.new_full((delta.size(0), 1, 1, 1), 0.5)
        band_mid = delta.new_full((delta.size(0), 1, 1, 1), 0.5)
        band_dbg = {"band_low": 0.5, "band_mid": 0.5, "tex": 0.0, "headroom": 0.0}
        e_low = delta.new_tensor(0.0)
        e_mid = delta.new_tensor(0.0)
        e_high = delta.new_tensor(0.0)
        if auto_band:
            band_low, band_mid, band_dbg = self._compute_bandmix(x01, x01, cm)
            k_low = int(getattr(cfg, "band_low_k", 11) or 11)
            k_mid = int(getattr(cfg, "band_mid_k", 5) or 5)
            if k_low < 3:
                k_low = 3
            if k_mid < 3:
                k_mid = 3
            if (k_low % 2) == 0:
                k_low += 1
            if (k_mid % 2) == 0:
                k_mid += 1
            if k_mid >= k_low:
                k_mid = max(3, k_low - 2)
                if (k_mid % 2) == 0:
                    k_mid -= 1
            low_raw = _masked_box_blur(delta, cm, k_low) * cm
            smooth = _masked_box_blur(delta, cm, k_mid) * cm
            mid_raw = (smooth - low_raw) * cm
            high = (delta - smooth) * cm
            e_low = _masked_mean(low_raw * low_raw, cm).detach()
            e_mid = _masked_mean(mid_raw * mid_raw, cm).detach()
            e_high = _masked_mean(high * high, cm).detach()
            mix_target_rms = _masked_rms(delta, cm).detach()
            if bool(getattr(cfg, "band_energy_norm", True)):
                low_rms = _masked_rms(low_raw, cm).detach().clamp_min(1e-12)
                mid_rms = _masked_rms(mid_raw, cm).detach().clamp_min(1e-12)
                low = (low_raw / low_rms) * cm
                mid = (mid_raw / mid_rms) * cm
                delta = (band_low * low + band_mid * mid) * cm
                mix_rms = _masked_rms(delta, cm).detach()
                max_gain = float(getattr(cfg, "band_norm_max_gain", 4.0) or 4.0)
                restore = (mix_target_rms / (mix_rms + 1e-12)).clamp(0.0, max_gain)
                delta = delta * restore
            else:
                delta = (band_low * low_raw + band_mid * mid_raw) * cm
            if bool(getattr(cfg, "delta_remove_dc", True)):
                mu_mid = _masked_mean(delta, cm).detach()
                delta = (delta - mu_mid) * cm
            beta = 0.0
        else:
            if bool(getattr(cfg, "freq_adapt", False)):
                # simple texture score in [0,1]: abs(highpass(original luma))
                k_tex = int(getattr(cfg, "freq_tex_window", 9) or 9)
                y_lp = _masked_box_blur(y_carrier, cm, k_tex)
                tex = (y_carrier - y_lp).abs()
                tex_s = tex.mean(dim=(1, 2, 3), keepdim=True)
                t = tex_s / (tex_s + 0.05)  # stable squash
                beta_min = float(getattr(cfg, "freq_beta_min", 0.50) or 0.50)
                beta_max = float(getattr(cfg, "freq_beta_max", 0.80) or 0.80)
                beta = (beta_max - t * (beta_max - beta_min)).clamp(min=0.0).mean().item()
                bg_s_min = float(getattr(cfg, "freq_bg_scale_min", 0.10) or 0.10)
                bg_s_max = float(getattr(cfg, "freq_bg_scale_max", 0.30) or 0.30)
                bg_scale = (bg_s_min + t * (bg_s_max - bg_s_min)).clamp(0.0, 1.0).mean().item()

            # Low-frequency suppression (kills "veil" while keeping energy)
            if beta > 0.0:
                k = int(getattr(cfg, "delta_hp_window", 9) or 9)
                low = _masked_box_blur(delta, cm, k)
                delta = (delta - beta * low) * cm
                # DC removal again (after filtering)
                if bool(getattr(cfg, "delta_remove_dc", True)):
                    mu_mid = _masked_mean(delta, cm).detach()
                    delta = (delta - mu_mid) * cm

        # Dark background protection (scale delta where the ORIGINAL carrier is dark)
        dark_abs = torch.zeros((delta.size(0), 1, 1, 1), device=delta.device, dtype=delta.dtype)
        if bg_thr > 0.0 and bg_scale < 1.0:
            dark = (y_carrier < bg_thr).to(dtype=delta.dtype)
            # only within cm
            dark = dark * cm
            bright = cm - dark
            factor = bright + dark * float(bg_scale)
            delta = delta * factor
            # diagnostic: mean |delta| on dark pixels
            den_d = dark.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
            dark_abs = (delta.abs() * dark).sum(dim=(1, 2, 3), keepdim=True) / den_d

        # Re-normalize RMS to preserve budget (delta magnitude stability)
        renorm_scale = torch.ones_like(rms_pre)
        if bool(getattr(cfg, "delta_renorm", True)):
            rms_post = _masked_rms(delta, cm).detach()
            renorm_scale = rms_pre / (rms_post + 1e-12)
            max_gain = float(getattr(cfg, "delta_renorm_max", 3.0) or 3.0)
            renorm_scale = renorm_scale.clamp(0.0, max_gain)
            delta = delta * renorm_scale
            # final DC removal (after renorm)
            if bool(getattr(cfg, "delta_remove_dc", True)):
                mu_post = _masked_mean(delta, cm).detach()
                delta = (delta - mu_post) * cm


        # Post-blur on delta to suppress grid / checkerboard artifacts (then re-normalize RMS).
        postk = int(getattr(cfg, "delta_post_blur_k", 0) or 0)
        postmix = float(getattr(cfg, "delta_post_blur_mix", 1.0))
        postmix = 0.0 if postmix < 0.0 else (1.0 if postmix > 1.0 else postmix)
        if (not auto_band) and postk >= 3 and postmix > 0.0:
            if (postk % 2) == 0:
                postk += 1
            delta_blur = _masked_box_blur(delta, cm, postk) * cm
            if postmix >= 1.0:
                delta = delta_blur
            else:
                delta = delta * (1.0 - postmix) + delta_blur * postmix
            if bool(getattr(cfg, "delta_renorm", True)):
                rms_post2 = _masked_rms(delta, cm).detach()
                ren2 = rms_pre / (rms_post2 + 1e-12)
                max_gain = float(getattr(cfg, "delta_renorm_max", 3.0) or 3.0)
                ren2 = ren2.clamp(0.0, max_gain)
                delta = delta * ren2
                # track combined renorm for logging
                renorm_scale = renorm_scale * ren2
                if bool(getattr(cfg, "delta_remove_dc", True)):
                    mu_post2 = _masked_mean(delta, cm).detach()
                    delta = (delta - mu_post2) * cm

        # Residual clamp (prevents rare spikes / halos).
        clip = float(getattr(cfg, "wm_res_clip", 0.0))
        mode = str(getattr(cfg, "wm_res_clip_mode", "tanh")).lower()
        if clip > 0.0 and mode not in ("none", "off", "0"):
            if mode == "tanh":
                delta = clip * torch.tanh(delta / clip)
            elif mode in ("hard", "clip"):
                delta = delta.clamp(-clip, clip)

        # Always enforce: no residual outside cm (after all ops).
        delta = delta * cm

        # Headroom clamp: prevent highlight/shadow saturation artifacts.
        # This keeps base+delta away from hard [0,1] clipping, which otherwise creates visible speckle/grids on bright regions.
        hm = float(getattr(cfg, "headroom_margin", 0.98) or 0.98)
        if 0.0 < hm < 1.0:
            yb = y_carrier
            if yb.size(-2) != delta.size(-2) or yb.size(-1) != delta.size(-1):
                yb = F.interpolate(yb, size=delta.shape[-2:], mode="bilinear", align_corners=False)
            pos_cap = (1.0 - yb) * hm
            neg_cap = yb * hm
            delta = torch.minimum(delta, pos_cap)
            delta = torch.maximum(delta, -neg_cap)
            delta = delta * cm


        # stash debug stats (for logging)
        try:
            mu_final = _masked_mean(delta, cm).detach()
            rms_post_final = _masked_rms(delta, cm).detach()
            self._last_embed_dbg = {
                "rms_pre": float(rms_pre.mean().item()),
                "rms_post": float(rms_post_final.mean().item()),
                "renorm_scale": float(renorm_scale.mean().item()),
                "delta_mean": float(mu_final.mean().item()),
                "delta_dark_abs": float(dark_abs.mean().item()),
                "hp_beta_eff": float(beta),
                "bg_scale_eff": float(bg_scale),
                "alpha_boost": float(boost.detach().mean().item()) if torch.is_tensor(boost) else float(boost),
                "band_low": float(band_dbg.get("band_low", 0.5)),
                "band_mid": float(band_dbg.get("band_mid", 0.5)),
                "band_tex": float(band_dbg.get("tex", 0.0)),
                "band_headroom": float(band_dbg.get("headroom", 0.0)),
                "band_e_low": float(e_low.mean().item()) if torch.is_tensor(e_low) else float(e_low),
                "band_e_mid": float(e_mid.mean().item()) if torch.is_tensor(e_mid) else float(e_mid),
                "band_e_high": float(e_high.mean().item()) if torch.is_tensor(e_high) else float(e_high),
                "cm_frac": float(cm_sum / (vm_sum + 1e-8)),
                "L_min": float(L_min.detach().mean().item()),
            }
        except Exception:
            self._last_embed_dbg = {}

        out = (x01 + delta.repeat(1, 3, 1, 1)).clamp(0, 1)
        return out

    # ---------- watermark synthesis (no-grad) ----------

    @torch.no_grad()
    def synth_variants_nograd(
            self,
            x01: torch.Tensor,
            valid_mask: Optional[torch.Tensor],
            epoch: int,
            variants=("base", "both"),
            k_factor: float = 1.0,
            varpercent: bool = True,
            mode: str = "train",
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        mode_l = str(mode).lower().strip()

        gray_mask = self._gray_like_mask(x01)

        base01 = self.ae.forward_plain(x01).clamp(0, 1)
        base01 = self._maybe_force_gray(base01, gray_mask)
        base01 = self._copy_outside_valid(base01, x01, valid_mask)
        out: Dict[str, torch.Tensor] = {"base01": base01}

        if variants == ("base",) or ("base" in variants and len(variants) == 1):
            return out

        # AE taps
        y_luma = self._luma01(x01)
        enc = self.ae.enc(y_luma)
        Z, S64 = enc["latent"], enc["s64"]

        # ROI
        P_lat, P_64 = self.build_roi_trainable_nograd(Z, S64, x01, valid_mask)

        # patterns
        z_pat = self.g_lat(Z)
        w64 = self.g_64(S64)

        # ROI + frequency shaping
        P_lat_e = self._soft_roi_mask(P_lat)
        P_64_e = self._soft_roi_mask(P_64)
        z_pat = F.avg_pool2d(z_pat * P_lat_e, 3, 1, 1)

        w64_roi = w64 * P_64_e
        # anti-grid: split into low/high in S64 domain with configurable kernel
        k_hp = int(getattr(cfg, 'w64_hp_k', 3) or 3)
        if k_hp < 3:
            k_hp = 3
        if (k_hp % 2) == 0:
            k_hp += 1
        w64_lp = F.avg_pool2d(w64_roi, k_hp, 1, k_hp // 2)
        w64_hp = w64_roi - w64_lp
        mix = float(min(max(cfg.skip_hp_mix, 0.0), 1.0))
        w64 = mix * w64_hp + (1.0 - mix) * w64_lp
        post_k = int(getattr(cfg, 'w64_post_blur_k', 0) or 0)
        if post_k >= 3:
            if (post_k % 2) == 0:
                post_k += 1
            w64 = F.avg_pool2d(w64, post_k, 1, post_k // 2)
            w64 = w64 * P_64_e
        # unit RMS
        def unit_rms(t: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            denom = m.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
            rms = torch.sqrt(((t * t) * m).sum(dim=(1, 2, 3), keepdim=True) / denom + 1e-12)
            return torch.nan_to_num(t / rms, nan=0.0, posinf=0.0, neginf=0.0)

        z_pat = unit_rms(z_pat, P_lat_e)
        w64 = unit_rms(w64, P_64_e)

        # budgets
        B = x01.size(0)
        eps = float(self.ctrl.eps)
        if mode_l != "train":
            ee = float(getattr(cfg, "eval_eps", 0.0) or 0.0)
            if ee > 0.0:
                eps = float(ee)
        A_lat = int(P_lat.size(-2) * P_lat.size(-1))
        A_64 = int(P_64.size(-2) * P_64.size(-1))
        area_lat = P_lat.flatten(1).sum(1).clamp_min(1.0)
        area_64 = P_64.flatten(1).sum(1).clamp_min(1.0)

        if varpercent:
            rmin = float(min(max(cfg.r_lat_min, 0.0), 1.0))
            rmax = float(min(max(cfg.r_lat_max, 0.0), 1.0))
            if rmax < rmin:
                rmax = rmin
            r_lat = rmin + torch.rand(B, 1, 1, 1, device=x01.device, dtype=x01.dtype) * (rmax - rmin)
            r_skip = 1.0 - r_lat
        else:
            rsk = float(self.ctrl.r_skip)
            if mode_l != "train":
                ers = float(getattr(cfg, "eval_r_skip", -1.0) or -1.0)
                if ers >= 0.0:
                    rsk = float(ers)
            r_skip_eff = float(min(max(rsk, cfg.val_r_skip_min), cfg.val_r_skip_max))
            r_lat_eff = 1.0 - r_skip_eff
            r_lat = torch.full((B, 1, 1, 1), r_lat_eff, device=x01.device, dtype=x01.dtype)
            r_skip = torch.full((B, 1, 1, 1), r_skip_eff, device=x01.device, dtype=x01.dtype)

        alpha_lat = eps * r_lat * (float(A_lat) / area_lat).sqrt().view(B, 1, 1, 1)
        alpha_skip = eps * r_skip * (float(A_64) / area_64).sqrt().view(B, 1, 1, 1)
        alpha_lat = alpha_lat * float(cfg.alpha_lat_gain)
        alpha_skip = alpha_skip * float(cfg.alpha_skip_gain)

        # apply k_factor (watermark strength augmentation)
        alpha_lat = alpha_lat * float(k_factor)
        alpha_skip = alpha_skip * float(k_factor)

        if "both" in variants:
            both01 = self._embed_luma_only_gray(
                x01=x01,
                base01=base01,
                wm_lat=z_pat,
                wm_skip=w64,
                alpha_lat=alpha_lat,
                alpha_skip=alpha_skip,
                roi_lat=P_lat_e,
                roi_skip=P_64_e,
                valid_mask=valid_mask,
            )
            both01 = self._maybe_force_gray(both01, gray_mask)
            both01 = self._copy_outside_valid(both01, x01, valid_mask)
            out["both01"] = torch.nan_to_num(both01, nan=0.0, posinf=1.0, neginf=0.0)

        if "lat" in variants:
            lat01 = self._embed_luma_only_gray(
                x01=x01,
                base01=base01,
                wm_lat=z_pat,
                wm_skip=None,
                alpha_lat=alpha_lat,
                alpha_skip=0.0,
                roi_lat=P_lat_e,
                roi_skip=None,
                valid_mask=valid_mask,
            )
            lat01 = self._maybe_force_gray(lat01, gray_mask)
            lat01 = self._copy_outside_valid(lat01, x01, valid_mask)
            out["lat01"] = torch.nan_to_num(lat01, nan=0.0, posinf=1.0, neginf=0.0)

        if "skip" in variants:
            skip01 = self._embed_luma_only_gray(
                x01=x01,
                base01=base01,
                wm_lat=None,
                wm_skip=w64,
                alpha_lat=0.0,
                alpha_skip=alpha_skip,
                roi_lat=None,
                roi_skip=P_64_e,
                valid_mask=valid_mask,
            )
            skip01 = self._maybe_force_gray(skip01, gray_mask)
            skip01 = self._copy_outside_valid(skip01, x01, valid_mask)
            out["skip01"] = torch.nan_to_num(skip01, nan=0.0, posinf=1.0, neginf=0.0)

        return out

    # ---------- generator step ----------

    def step_generator(self, batch, epoch: int, it: int) -> Dict[str, float]:
        xN, valid_mask, y, paths = batch
        xN = xN.to(self.device)
        valid_mask = valid_mask.to(self.device)
        y = y.to(self.device)

        x01 = self._to01(xN)
        gray_mask = self._gray_like_mask(x01)

        with torch.no_grad():
            base01 = self.ae.forward_plain(x01).clamp(0, 1)
            base01 = self._maybe_force_gray(base01, gray_mask)
            base01 = self._copy_outside_valid(base01, x01, valid_mask)

            y_luma = self._luma01(x01)
            enc = self.ae.enc(y_luma)
            Z = enc["latent"]
            S64 = enc["s64"]

        # ROI (trainable)
        P_lat, P_64, roi_losses, roi_dbg = self.build_roi_trainable(Z, S64, x01, valid_mask, epoch)

        # patterns
        z_pat = self.g_lat(Z)
        w64 = self.g_64(S64)

        # ROI + shaping
        P_lat_e = self._soft_roi_mask(P_lat)
        P_64_e = self._soft_roi_mask(P_64)
        z_pat = F.avg_pool2d(z_pat * P_lat_e, 3, 1, 1)

        w64_roi = w64 * P_64_e
        # anti-grid: split into low/high in S64 domain with configurable kernel
        k_hp = int(getattr(self.cfg, 'w64_hp_k', 3) or 3)
        if k_hp < 3:
            k_hp = 3
        if (k_hp % 2) == 0:
            k_hp += 1
        w64_lp = F.avg_pool2d(w64_roi, k_hp, 1, k_hp // 2)
        w64_hp = w64_roi - w64_lp
        mix = float(min(max(self.cfg.skip_hp_mix, 0.0), 1.0))
        w64 = mix * w64_hp + (1.0 - mix) * w64_lp
        post_k = int(getattr(self.cfg, 'w64_post_blur_k', 0) or 0)
        if post_k >= 3:
            if (post_k % 2) == 0:
                post_k += 1
            w64 = F.avg_pool2d(w64, post_k, 1, post_k // 2)
            w64 = w64 * P_64_e
        # unit RMS
        def unit_rms(t: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            denom = m.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
            rms = torch.sqrt(((t * t) * m).sum(dim=(1, 2, 3), keepdim=True) / denom + 1e-12)
            return torch.nan_to_num(t / rms, nan=0.0, posinf=0.0, neginf=0.0)

        z_pat = unit_rms(z_pat, P_lat_e)
        w64 = unit_rms(w64, P_64_e)

        # pattern vis (for collage)
        with torch.no_grad():
            z_vis = self._rms_vis01(z_pat.detach() * P_lat_e.detach())
            w_vis = self._rms_vis01(w64.detach() * P_64_e.detach())
            H, W = x01.shape[-2:]
            z_vis_up = F.interpolate(z_vis, size=(H, W), mode="nearest")
            w_vis_up = F.interpolate(w_vis, size=(H, W), mode="nearest")

        # budgets
        B = x01.size(0)
        eps = float(self.ctrl.eps)
        A_lat = int(P_lat.size(-2) * P_lat.size(-1))
        A_64 = int(P_64.size(-2) * P_64.size(-1))
        area_lat = P_lat.flatten(1).sum(1).clamp_min(1.0)
        area_64 = P_64.flatten(1).sum(1).clamp_min(1.0)

        rmin = float(min(max(self.cfg.r_lat_min, 0.0), 1.0))
        rmax = float(min(max(self.cfg.r_lat_max, 0.0), 1.0))
        if rmax < rmin:
            rmax = rmin

        r_lat = rmin + torch.rand(B, 1, 1, 1, device=self.device) * (rmax - rmin)
        r_skip = 1.0 - r_lat

        alpha_lat = eps * r_lat * (float(A_lat) / area_lat).sqrt().view(B, 1, 1, 1)
        alpha_skip = eps * r_skip * (float(A_64) / area_64).sqrt().view(B, 1, 1, 1)
        alpha_lat = alpha_lat * float(self.cfg.alpha_lat_gain)
        alpha_skip = alpha_skip * float(self.cfg.alpha_skip_gain)

        # --- branch energy diagnostics (latent vs skip64) + soft quota (prevents latent collapse) ---
        lat_quota_min = float(getattr(self.cfg, 'lat_quota_min', 0.0))
        lat_quota_lambda = float(getattr(self.cfg, 'lat_quota_lambda', 0.0))
        lat_quota_warmup = int(getattr(self.cfg, 'lat_quota_warmup_epochs', 0))

        # Estimate feature-space energy (after alpha scaling) inside the ROI.
        # Note: z_pat and w64 are unit-RMS normalized on their ROIs, so energies mainly reflect alpha^2.
        lat_ms = (z_pat * z_pat).mean(dim=1, keepdim=True)  # [B,1,hZ,wZ]
        sk_ms = (w64 * w64).mean(dim=1, keepdim=True)  # [B,1,hS,wS]
        denom_lat = P_lat_e.sum(dim=(2, 3)).clamp_min(1.0)
        denom_sk = P_64_e.sum(dim=(2, 3)).clamp_min(1.0)
        E_lat_s = (lat_ms * P_lat_e).sum(dim=(2, 3)) / denom_lat  # [B,1]
        E_sk_s = (sk_ms * P_64_e).sum(dim=(2, 3)) / denom_sk  # [B,1]
        alpha_lat2 = alpha_lat.view(B, 1).pow(2)
        alpha_skip2 = alpha_skip.view(B, 1).pow(2)
        E_lat_s = E_lat_s * alpha_lat2
        E_sk_s = E_sk_s * alpha_skip2
        p_lat_s = E_lat_s / (E_lat_s + E_sk_s + 1e-12)  # [B,1]
        p_lat_mean_t = p_lat_s.mean()
        p_s64_mean_t = (1.0 - p_lat_s).mean()

        L_quota = z_pat.new_tensor(0.0)
        if (lat_quota_lambda > 0.0) and (lat_quota_min > 0.0) and (epoch > lat_quota_warmup):
            shortfall = (lat_quota_min - p_lat_s).clamp_min(0.0)
            L_quota = lat_quota_lambda * (shortfall * shortfall).mean()

        # embed BOTH with gradients
        both01 = self._embed_luma_only_gray(
            x01=x01,
            base01=base01,
            wm_lat=z_pat,
            wm_skip=w64,
            alpha_lat=alpha_lat,
            alpha_skip=alpha_skip,
            roi_lat=P_lat_e,
            roi_skip=P_64_e,
            valid_mask=valid_mask,
        )
        both01 = self._maybe_force_gray(both01, gray_mask)
        both01 = self._copy_outside_valid(both01, x01, valid_mask)
        both01 = torch.nan_to_num(both01, nan=0.0, posinf=1.0, neginf=0.0)

        # ---- positive generator objective via frozen current C2 ----
        # Without this term the generator can collapse to the trivial identity solution
        # (both01 == x01), especially once clean/reference paths are moved to raw-original.
        L_push_cls = both01.new_tensor(0.0)
        L_push_det = both01.new_tensor(0.0)
        L_push_margin = both01.new_tensor(0.0)
        if epoch >= int(getattr(self.cfg, "g_push_start_epoch", 1) or 1):
            cleanN_push = self._apply_prod_padding_wipe(xN.detach(), valid_mask)
            bothN_push = self._apply_prod_padding_wipe(both01 * 2.0 - 1.0, valid_mask)

            c2m = unwrap(self.c2)
            was_training = c2m.training
            req_flags = [p.requires_grad for p in c2m.parameters()]
            try:
                c2m.eval()
                for p_ in c2m.parameters():
                    p_.requires_grad_(False)
                logits_clean_push, wm_clean_push, _ = c2m(cleanN_push, gate=True)
                logits_both_push, wm_both_push, _ = c2m(bothN_push, gate=True)
                L_push_cls = F.cross_entropy(logits_both_push, y)
                L_push_det = F.binary_cross_entropy_with_logits(wm_both_push, torch.ones_like(wm_both_push))
                m_clean_push = self._margin(logits_clean_push.detach(), y)
                m_both_push = self._margin(logits_both_push, y)
                margin_target = float(getattr(self.cfg, "g_push_margin_target", 0.25) or 0.25)
                L_push_margin = F.softplus(margin_target - (m_both_push - m_clean_push)).mean()
            finally:
                for p_, rf in zip(c2m.parameters(), req_flags):
                    p_.requires_grad_(rf)
                if was_training:
                    c2m.train()

        # ---- C1 guard rail (frozen external classifier) ----
        # Goal: watermark must NOT degrade C1 accuracy on watermarked images below a target.
        # Implementation: always compute C1 clean/wm metrics on schedule, and optionally apply
        # a differentiable CE-degradation penalty plus controller brake on eps.
        L_c1_guard = both01.new_tensor(0.0)
        c1_acc_clean = float("nan")
        c1_acc_wm = float("nan")
        c1_ce_clean = float("nan")
        c1_ce_wm = float("nan")
        c1_ce_diff = float("nan")
        c1_acc_delta = float("nan")
        c1_ce_delta = float("nan")

        if getattr(self, "c1", None) is not None:
            every = int(getattr(self.cfg, "c1_guard_every", 1) or 1)
            every = max(1, every)
            lam = float(getattr(self.cfg, "c1_guard_lambda", 0.0) or 0.0)
            margin_ce = float(getattr(self.cfg, "c1_guard_ce_margin", 0.0) or 0.0)
            min_acc_cfg = float(getattr(self.cfg, "c1_guard_min_acc", 0.0) or 0.0)

            if (it % every) == 0:
                # Use RAW original as the clean reference; AE baseline is diagnostic only.
                cleanN_c1 = self._apply_prod_padding_wipe(xN, valid_mask)
                bothN_c1 = self._apply_prod_padding_wipe(both01 * 2.0 - 1.0, valid_mask)

                logits_c1_clean = self.c1_logits(cleanN_c1)
                logits_c1_wm = self.c1_logits(bothN_c1)

                # metrics (no grad)
                with torch.no_grad():
                    c1_acc_clean = float((logits_c1_clean.argmax(1) == y).float().mean().item())
                    c1_acc_wm = float((logits_c1_wm.argmax(1) == y).float().mean().item())
                    ce_clean_t = F.cross_entropy(logits_c1_clean.detach(), y)
                    c1_ce_clean = float(ce_clean_t.item())

                # differentiable penalty (grad flows into both01 -> generator)
                ce_wm_t = F.cross_entropy(logits_c1_wm, y)
                c1_ce_wm = float(ce_wm_t.detach().item())
                ce_diff_t = ce_wm_t - ce_clean_t
                c1_ce_diff = float(ce_diff_t.detach().item())
                c1_acc_delta = float(c1_acc_wm - c1_acc_clean)
                c1_ce_delta = float(c1_ce_wm - c1_ce_clean)

                if (lam > 0.0) or (min_acc_cfg > 0.0):
                    drop_scale = 1.0
                    max_drop = float(getattr(self.cfg, "c1_guard_max_drop", 0.0) or 0.0)
                    if max_drop > 0.0:
                        drop = max(0.0, float(c1_acc_clean - c1_acc_wm))
                        if drop > max_drop:
                            drop_scale = 1.0 + (drop - max_drop) / max_drop
                    L_c1_guard = (lam * drop_scale) * F.relu(ce_diff_t - margin_ce).pow(2)

                self._c1_last = (c1_acc_clean, c1_acc_wm, c1_ce_clean, c1_ce_wm, c1_ce_diff, c1_acc_delta, c1_ce_delta)
            else:
                # report last cached values (if any)
                if hasattr(self, "_c1_last") and isinstance(self._c1_last, tuple):
                    if len(self._c1_last) == 7:
                        c1_acc_clean, c1_acc_wm, c1_ce_clean, c1_ce_wm, c1_ce_diff, c1_acc_delta, c1_ce_delta = self._c1_last
                    elif len(self._c1_last) == 5:
                        c1_acc_clean, c1_acc_wm, c1_ce_clean, c1_ce_wm, c1_ce_diff = self._c1_last
                        if (c1_acc_clean == c1_acc_clean) and (c1_acc_wm == c1_acc_wm):
                            c1_acc_delta = float(c1_acc_wm - c1_acc_clean)
                        if (c1_ce_clean == c1_ce_clean) and (c1_ce_wm == c1_ce_wm):
                            c1_ce_delta = float(c1_ce_wm - c1_ce_clean)

        # capture embed diagnostics from the *BOTH* pass (lat/skip probes below will overwrite)
        _edbg = getattr(self, "_last_embed_dbg", {}) or {}
        d_mean = float(_edbg.get("delta_mean", 0.0))
        rms_pre = float(_edbg.get("rms_pre", 0.0))
        rms_post = float(_edbg.get("rms_post", 0.0))
        rn_scale = float(_edbg.get("renorm_scale", 1.0))
        d_dark = float(_edbg.get("delta_dark_abs", 0.0))
        hp_beta_eff = float(_edbg.get("hp_beta_eff", 0.0))
        bg_scale_eff = float(_edbg.get("bg_scale_eff", 1.0))
        L_min_dbg = float(_edbg.get("L_min", 0.0))
        cm_frac_dbg = float(_edbg.get("cm_frac", 1.0))
        alpha_boost_dbg = float(_edbg.get("alpha_boost", 1.0))
        band_low_dbg = float(_edbg.get("band_low", 0.5))
        band_mid_dbg = float(_edbg.get("band_mid", 0.5))
        band_tex_dbg = float(_edbg.get("band_tex", 0.0))
        band_headroom_dbg = float(_edbg.get("band_headroom", 0.0))
        band_e_low = float(_edbg.get("band_e_low", 0.0))
        band_e_mid = float(_edbg.get("band_e_mid", 0.0))
        band_e_high = float(_edbg.get("band_e_high", 0.0))

        # shapley variants (no grad)
        with torch.no_grad():
            lat01 = self._embed_luma_only_gray(
                x01=x01,
                base01=base01,
                wm_lat=z_pat,
                wm_skip=None,
                alpha_lat=alpha_lat,
                alpha_skip=0.0,
                roi_lat=P_lat_e,
                roi_skip=None,
                valid_mask=valid_mask,
            )
            lat01 = self._maybe_force_gray(lat01, gray_mask)
            lat01 = self._copy_outside_valid(lat01, x01, valid_mask)

            skip01 = self._embed_luma_only_gray(
                x01=x01,
                base01=base01,
                wm_lat=None,
                wm_skip=w64,
                alpha_lat=0.0,
                alpha_skip=alpha_skip,
                roi_lat=None,
                roi_skip=P_64_e,
                valid_mask=valid_mask,
            )
            skip01 = self._maybe_force_gray(skip01, gray_mask)
            skip01 = self._copy_outside_valid(skip01, x01, valid_mask)

            # seismic diff for collage (masked)
            diff = (both01 - x01).mean(dim=1, keepdim=True)
            p = 99.0
            lo = torch.quantile(diff.flatten(1), (100 - p) / 200, dim=1, keepdim=True).view(B, 1, 1, 1)
            hi = torch.quantile(diff.flatten(1), (100 + p) / 200, dim=1, keepdim=True).view(B, 1, 1, 1)
            seismic01 = torch.clamp((diff - lo) / (hi - lo + 1e-8), 0, 1)

        # save WM images (audit)
        self.save_watermarked_batch(both01.detach(), valid_mask.detach(), paths, y, epoch)

        # quality metrics (mask-aware; pad excluded via valid_mask)
        PSNR_rgb = float(psnr_torch(both01.detach(), x01, valid_mask=valid_mask).item())
        PSNR = float(psnr_y_torch(both01.detach(), x01, valid_mask=valid_mask).item())  # PSNR_y
        SSIM = float(ssim_y_torch(both01.detach(), x01, valid_mask=valid_mask).item())  # SSIM_y (mask-aware)
        MAE = float(mae_torch(both01.detach(), x01, valid_mask=valid_mask).item())

        PSNR_base_rgb = float(psnr_torch(both01.detach(), base01.detach(), valid_mask=valid_mask).item())
        PSNR_base = float(psnr_y_torch(both01.detach(), base01.detach(), valid_mask=valid_mask).item())
        SSIM_base = float(ssim_y_torch(both01.detach(), base01.detach(), valid_mask=valid_mask).item())
        MAE_base = float(mae_torch(both01.detach(), base01.detach(), valid_mask=valid_mask).item())

        PSNR_ae_rgb = float(psnr_torch(base01.detach(), x01, valid_mask=valid_mask).item())
        PSNR_ae = float(psnr_y_torch(base01.detach(), x01, valid_mask=valid_mask).item())  # PSNR_y(AE vs x)
        SSIM_ae = float(ssim_y_torch(base01.detach(), x01, valid_mask=valid_mask).item())
        MAE_ae = float(mae_torch(base01.detach(), x01, valid_mask=valid_mask).item())

        # saturation diagnostics (valid region only)
        sat_x_hi = sat_hi_frac(x01.detach(), valid_mask)
        sat_base_hi = sat_hi_frac(base01.detach(), valid_mask)
        sat_wm_hi = sat_hi_frac(both01.detach(), valid_mask)

        skip_hw = (int(P_64.size(-2)), int(P_64.size(-1)))
        P_lat_to_skip_h = F.interpolate(P_lat, size=skip_hw, mode="nearest")
        roi_full_skip = torch.clamp(P_64 + P_lat_to_skip_h, 0, 1)
        roi_full = F.interpolate(roi_full_skip, size=x01.shape[-2:], mode="bilinear", align_corners=False)
        roi_union_mean = float(_masked_mean(roi_full, valid_mask).item())

        res = (both01 - x01)

        # chroma / object diagnostics (not part of loss; helps debug RGB datasets)
        rgb_mask = (1.0 - gray_mask.view(-1)).detach()
        if float(rgb_mask.sum().item()) > 0.5:
            cb_b, cr_b = self._rgb_to_cbcr(x01.detach())
            cb_w, cr_w = self._rgb_to_cbcr(both01.detach())
            diff_ch = (cb_w - cb_b).abs() + (cr_w - cr_b).abs()  # [B,1,H,W]
            per = _masked_mean_per_sample(diff_ch, valid_mask)  # [B]
            L_chroma = float(((per * rgb_mask).sum() / rgb_mask.sum().clamp_min(1.0)).item())
        else:
            L_chroma = 0.0

        y_in = self._luma01(x01.detach())
        obj = (y_in > float(getattr(self.cfg, 'white_thr', 0.70))).to(x01.dtype)
        if valid_mask is not None:
            obj = obj * valid_mask
        denom_obj = obj.sum().clamp_min(1.0)
        res1 = res.abs().mean(dim=1, keepdim=True)
        L_obj = float(((res1 * obj).sum() / denom_obj).item())
        vm3 = valid_mask.repeat(1, 3, 1, 1)

        delta_abs = float((res.abs() * vm3).sum().item() / vm3.sum().clamp_min(1.0).item())
        delta_max = float((res.abs() * vm3).amax().item())

        L_leak = _masked_mean(res.abs() * (1.0 - roi_full), valid_mask)

        # HF penalty (encourage smoothness) — masked
        dx = (res[:, :, :, 1:] - res[:, :, :, :-1]).abs()
        dy = (res[:, :, 1:, :] - res[:, :, :-1, :]).abs()
        vmx = vm3[:, :, :, 1:] * vm3[:, :, :, :-1]
        vmy = vm3[:, :, 1:, :] * vm3[:, :, :-1, :]
        L_hf = 0.5 * ((dx * vmx).sum() / vmx.sum().clamp_min(1.0) + (dy * vmy).sum() / vmy.sum().clamp_min(1.0))

        # --- clip-hit + spectral + ROI/BG quality diagnostics ---
        # Clip-hit (how often delta hits the residual clamp). Computed on luma residual.
        clip = float(getattr(self.cfg, 'wm_res_clip', 0.0))
        deltaY = (self._luma01(both01) - self._luma01(x01))  # [B,1,H,W]
        if clip > 0.0:
            thr_hit = 0.99 * clip
            clip_hit = (((deltaY.abs() >= thr_hit).to(deltaY.dtype) * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)).clamp(0, 1)
        else:
            clip_hit = deltaY.new_tensor(float('nan'))

        # Anti-grid: penalize block-boundary discontinuities (8x8-ish checkerboard)
        grid_lam = float(getattr(self.cfg, "grid_boundary_lambda", 0.0) or 0.0)
        if grid_lam > 0.0:
            hS, wS = int(P_64.shape[-2]), int(P_64.shape[-1])
            H, W = int(deltaY.shape[-2]), int(deltaY.shape[-1])
            blk_h = max(1, H // max(1, hS))
            blk_w = max(1, W // max(1, wS))
            pen_h = deltaY.new_tensor(0.0)
            pen_w = deltaY.new_tensor(0.0)
            if blk_h >= 2:
                rows = torch.arange(blk_h, H, blk_h, device=deltaY.device, dtype=torch.long)
                if rows.numel() > 0:
                    d = (deltaY[:, :, rows, :] - deltaY[:, :, rows - 1, :]).abs()
                    m = valid_mask[:, :, rows, :] * valid_mask[:, :, rows - 1, :]
                    pen_h = (d * m).sum() / (m.sum() + 1e-6)
            if blk_w >= 2:
                cols = torch.arange(blk_w, W, blk_w, device=deltaY.device, dtype=torch.long)
                if cols.numel() > 0:
                    d = (deltaY[:, :, :, cols] - deltaY[:, :, :, cols - 1]).abs()
                    m = valid_mask[:, :, :, cols] * valid_mask[:, :, :, cols - 1]
                    pen_w = (d * m).sum() / (m.sum() + 1e-6)
            L_grid = 0.5 * (pen_h + pen_w)
        else:
            L_grid = deltaY.new_tensor(0.0)

        # Low-frequency energy ratio (cheap proxy): energy(blur(delta)) / energy(delta)
        spec_k = int(getattr(self.cfg, 'spec_window', 9))
        if spec_k < 3:
            spec_k = 3
        if (spec_k % 2) == 0:
            spec_k += 1
        lp = F.avg_pool2d(deltaY, kernel_size=spec_k, stride=1, padding=spec_k // 2)
        e_lp = _masked_mean(lp * lp, valid_mask)
        e_tot = _masked_mean(deltaY * deltaY, valid_mask)
        lowfreq_ratio_t = e_lp / (e_tot + 1e-12)

        spec_lambda = float(getattr(self.cfg, 'spec_lambda', 0.0))
        spec_max = float(getattr(self.cfg, 'spec_lowfreq_max', 0.35))
        L_spec = deltaY.new_tensor(0.0)
        if spec_lambda > 0.0:
            spec_min = float(getattr(self.cfg, 'spec_lowfreq_min', 0.0) or 0.0)
            L_spec = spec_lambda * (F.relu(lowfreq_ratio_t - spec_max).pow(2) + F.relu(spec_min - lowfreq_ratio_t).pow(2))

        # ROI vs background quality (mask-aware).
        roi_m = (roi_full > 0.5).to(dtype=x01.dtype) * valid_mask
        bg_m = (valid_mask - roi_m).clamp(0.0, 1.0)

        PSNR_roi = float(psnr_y_torch(both01.detach(), x01.detach(), valid_mask=roi_m).item())
        SSIM_roi = float(ssim_y_torch(both01.detach(), x01.detach(), valid_mask=roi_m).item())
        MAE_roi = float(mae_torch(both01.detach(), x01.detach(), valid_mask=roi_m).item())

        PSNR_bg = float(psnr_y_torch(both01.detach(), x01.detach(), valid_mask=bg_m).item())
        SSIM_bg = float(ssim_y_torch(both01.detach(), x01.detach(), valid_mask=bg_m).item())
        MAE_bg = float(mae_torch(both01.detach(), x01.detach(), valid_mask=bg_m).item())

        # ROI losses aggregate
        cfg = self.cfg
        L_roi = (
                cfg.lam_area * roi_losses["L_area"]
                + cfg.lam_overlap * roi_losses["L_overlap"]
                + cfg.lam_tv * roi_losses["L_tv"]
                + cfg.lam_sparse * roi_losses["L_sparse"]
                + cfg.lam_bin * roi_losses["L_bin"]
                + cfg.lam_tex * roi_losses["L_tex"]
                + cfg.lam_teacher * roi_losses["L_teacher"]
        )

        # generator total
        push_cls_lam    = float(getattr(self.cfg, "g_push_cls_lambda",   0.35) or 0.35)
        push_det_lam    = float(getattr(self.cfg, "g_push_det_lambda",   0.15) or 0.15)
        push_margin_lam = float(getattr(self.cfg, "g_push_margin_lambda",0.20) or 0.20)
        transfer_lam    = float(getattr(self.cfg, "transfer_lam",         0.50) or 0.50)
        diversity_lam   = float(getattr(self.cfg, "diversity_lam",        0.10) or 0.10)
        ssim_lam        = float(getattr(self.cfg, "ssim_lam",             0.60) or 0.60)
        _t_start        = int(getattr(self.cfg, "transfer_start_epoch",   2)   or 2)

        # ── L_ssim: perceptual quality loss (was only diagnostic before) ──
        with torch.no_grad():
            _ssim_val = float(ssim_y_torch(both01.detach(), x01.detach(), valid_mask).item())
        L_ssim = (1.0 - ssim_y_torch(both01, x01, valid_mask)) * ssim_lam

        # ── L_transfer: simulate transfer/KNN attack ──
        # Take WM residual from img_A, apply to img_B; C2 must NOT detect it.
        # This forces WM to be content-entangled, not a global additive pattern.
        L_transfer = both01.new_tensor(0.0)
        if epoch >= _t_start and B > 1 and transfer_lam > 0.0:
            with torch.no_grad():
                delta = (both01.detach() - x01.detach())           # [B,C,H,W] WM residual
                idx_perm = torch.randperm(B, device=x01.device)
                transplant01 = (x01[idx_perm] + delta).clamp(0.0, 1.0)
                transplantN  = transplant01 * 2.0 - 1.0
            try:
                _, wm_transplant, _ = self.c2(transplantN, gate=True)
                t_zeros = torch.zeros_like(wm_transplant)          # must NOT detect WM
                L_transfer = F.binary_cross_entropy_with_logits(wm_transplant, t_zeros)
            except Exception:
                L_transfer = both01.new_tensor(0.0)

        # ── L_diversity: WM residuals in batch should NOT be cosine-similar ──
        # Prevents the network from learning a single universal WM pattern.
        L_diversity = both01.new_tensor(0.0)
        if diversity_lam > 0.0 and B > 1:
            res_flat = (both01 - x01).flatten(1)                   # [B, C*H*W]
            res_norm = F.normalize(res_flat, dim=1, eps=1e-8)
            sim_mat  = res_norm @ res_norm.T                       # [B, B] cosine similarity
            off_diag = sim_mat * (1.0 - torch.eye(B, device=sim_mat.device))
            L_diversity = off_diag.relu().mean() * diversity_lam

        L_g = (
                0.5 * L_leak
                + 0.5 * L_hf
                + L_roi
                + L_quota
                + L_spec
                + L_c1_guard
                + (grid_lam * L_grid)
                + (push_cls_lam * L_push_cls)
                + (push_det_lam * L_push_det)
                + (push_margin_lam * L_push_margin)
                + self._last_embed_Lmin
                + L_ssim                              # SOTA-FIX: perceptual quality
                + (transfer_lam * L_transfer)         # SOTA-FIX: transfer/KNN resistance
                + L_diversity                         # SOTA-FIX: content-unique WM patterns
        )

        self.opt_g.zero_grad(set_to_none=True)
        L_g.backward()
        torch.nn.utils.clip_grad_norm_(
            list(unwrap(self.mask_lat).parameters())
            + list(unwrap(self.mask_64).parameters())
            + list(unwrap(self.g_lat).parameters())
            + list(unwrap(self.g_64).parameters())
            + list(unwrap(self.freq_ctrl).parameters()),
            1.0,
        )
        self.opt_g.step()

        # optional train-debug collage (publication panels, disabled by default)
        if bool(getattr(self.cfg, "pub_collage_keep_train_debug", False)) and self.cfg.collage_every > 0 and (it % self.cfg.collage_every == 0):
            idx = 0
            try:
                self.save_pub_collage(
                    epoch=epoch,
                    cls_name=self.classes[int(y[idx].item())],
                    src_path=paths[idx],
                    orig01=x01[idx].detach(),
                    base01=base01[idx].detach(),
                    both01=both01.detach()[idx],
                    lat01=lat01[idx].detach(),
                    skip01=skip01[idx].detach(),
                    valid_mask01=valid_mask[idx].detach(),
                    root_name="collages_train_debug",
                )
            except Exception as e:
                print(f"[COLLAGE DBG] failed at E{epoch:02d} it {it:06d}: {e}", flush=True)

        # border/padding diagnostics (shortcut detection)
        with torch.no_grad():
            pad_mean_x01 = self._pad_mean01(x01.detach(), valid_mask)
            pad_mean_xN = self._pad_meanN(xN.detach(), valid_mask)
            valid_frac = self._valid_frac(valid_mask)
            border_abs, valid_abs_res, border_ratio = self._border_stats((both01.detach() - x01.detach()), valid_mask)
            border_abs_wm_ae, _, _ = self._border_stats((both01.detach() - base01.detach()), valid_mask)
            border_abs_wm_x = border_abs
        return {
            # losses
            "L_g": float(L_g.item()),
            "L_leak": float(L_leak.item()),
            "L_hf": float(L_hf.item()),
            "L_roi": float(L_roi.item()),
            "L_quota": float(L_quota.item()),
            "L_spec": float(L_spec.item()),
            "L_push_cls": float(L_push_cls.item()),
            "L_transfer": float(L_transfer.item()),    # transfer-attack resistance
            "L_diversity": float(L_diversity.item()),  # WM pattern diversity
            "L_ssim": float(L_ssim.item()),            # perceptual quality
            "L_push_det": float(L_push_det.item()),
            "L_push_margin": float(L_push_margin.item()),
            "L_grid": float(L_grid.item()),

            "L_c1_guard": float(L_c1_guard.item()),
            "c1_acc_clean": float(c1_acc_clean),
            "c1_acc_wm": float(c1_acc_wm),
            "c1_ce_clean": float(c1_ce_clean),
            "c1_ce_wm": float(c1_ce_wm),
            "c1_ce_diff": float(c1_ce_diff),
            "c1_acc_delta": float(c1_acc_delta),
            "c1_ce_delta": float(c1_ce_delta),

            # quality (wm vs x)
            "PSNR": PSNR,
            "PSNR_rgb": PSNR_rgb,
            "SSIM": SSIM,
            "MAE": MAE,

            # quality (wm vs AE baseline)
            "PSNR_base": PSNR_base,
            "PSNR_base_rgb": PSNR_base_rgb,
            "SSIM_base": SSIM_base,
            "MAE_base": MAE_base,

            # AE baseline (AE vs x)
            "PSNR_ae": PSNR_ae,
            "PSNR_ae_rgb": PSNR_ae_rgb,
            "SSIM_y_ae": SSIM_ae,
            "MAE_ae": MAE_ae,

            # saturation diagnostics (valid region only)
            "sat_x_hi": float(sat_x_hi),
            "sat_base_hi": float(sat_base_hi),
            "sat_wm_hi": float(sat_wm_hi),

            # watermark energy diagnostics
            "delta_abs": float(delta_abs),
            "delta_max": float(delta_max),
            "delta_mean": float(d_mean),
            "delta_rms_pre": float(rms_pre),
            "delta_rms_post": float(rms_post),
            "delta_rn": float(rn_scale),
            "delta_dark_abs": float(d_dark),
            "hp_beta_eff": float(hp_beta_eff),
            "bg_scale_eff": float(bg_scale_eff),
            "L_min": float(L_min_dbg),
            "cm_frac": float(cm_frac_dbg),
            "alpha_boost": float(alpha_boost_dbg),
            "band_low": float(band_low_dbg),
            "band_mid": float(band_mid_dbg),
            "band_tex": float(band_tex_dbg),
            "band_headroom": float(band_headroom_dbg),
            "band_e_low": float(band_e_low),
            "band_e_mid": float(band_e_mid),
            "band_e_high": float(band_e_high),

            "E_lat": float(E_lat_s.mean().detach().item()),
            "E_s64": float(E_sk_s.mean().detach().item()),
            "p_lat": float(p_lat_mean_t.detach().item()),
            "p_s64": float(p_s64_mean_t.detach().item()),
            "clip_hit": float(clip_hit.detach().item()),
            "lowfreq_ratio": float(lowfreq_ratio_t.detach().item()),

            "PSNR_roi": float(PSNR_roi),
            "SSIM_roi": float(SSIM_roi),
            "MAE_roi": float(MAE_roi),
            "PSNR_bg": float(PSNR_bg),
            "SSIM_bg": float(SSIM_bg),
            "MAE_bg": float(MAE_bg),

            # ROI diagnostics
            "P_lat_mean": float(roi_dbg["roi_lat_mean"]),
            "P_64_mean": float(roi_dbg["roi_64_mean"]),
            "roi_union_mean": float(roi_union_mean),
            "roi_lat_mean": float(roi_dbg["roi_lat_mean"]),
            "roi_64_mean": float(roi_dbg["roi_64_mean"]),
            "roi_overlap": float(roi_dbg["roi_overlap"]),

            # RGB diagnostics (mostly relevant if not grayscale-like)
            "L_chroma": float(L_chroma),
            "L_obj": float(L_obj),

            # padding / border diagnostics
            "pad_mean_x01": float(pad_mean_x01),
            "pad_mean_xN": float(pad_mean_xN),
            "valid_frac": float(valid_frac),
            "border_abs": float(border_abs),
            "valid_abs": float(valid_abs_res),
            "border_ratio": float(border_ratio),
            "border_abs_wm_x": float(border_abs_wm_x),
            "border_abs_wm_ae": float(border_abs_wm_ae),
        }

    # ---------- C2 step ----------

    def _margin(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        idx = y.view(-1, 1)
        z_y = z.gather(1, idx).squeeze(1)
        z2 = z.clone()
        z2.scatter_(1, idx, float("-inf"))
        z_oth = z2.max(1).values
        return z_y - z_oth

    def step_c2(self, batch, epoch: int) -> Dict[str, float]:
        xN, valid_mask, y, _ = batch
        xN = xN.to(self.device)
        valid_mask = valid_mask.to(self.device)
        y = y.to(self.device)

        x01 = self._to01(xN)
        # Detector warmup: optionally train wm_head only for the first N epochs.
        det_epochs = int(getattr(self.cfg, "det_warmup_epochs", 0) or 0)
        det_warm = (det_epochs > 0 and epoch <= det_epochs)

        # C2 training-time synthesis mix:
        #  - Most steps: augmented watermark strength (varpercent=True, k_factor in [0.8,1.2])
        #  - Some steps: prod-like synthesis (varpercent=False, k_factor=1.0) to match validate/REALVAL
        prod_p = float(getattr(self.cfg, "train_prod_mix_prob", 0.0))
        if det_warm:
            k_factor = float(getattr(self.cfg, "det_warmup_k_factor", 1.25) or 1.25)
            varpercent = True
        elif random.random() < prod_p:
            k_factor = 1.0
            varpercent = False
        else:
            k_factor = float(0.80 + 0.40 * random.random())
            varpercent = True

        syn = self.synth_variants_nograd(
            x01,
            valid_mask=valid_mask,
            epoch=epoch,
            variants=("base", "both"),
            k_factor=k_factor,
            varpercent=varpercent,
            mode="train",
        )
        cleanN = self._apply_prod_padding_wipe(xN, valid_mask)
        bothN = self._apply_prod_padding_wipe(syn["both01"] * 2 - 1, valid_mask)

        self.c2.train()

        logits_clean_ng, wm_clean, _ = self.c2(cleanN, gate=False)
        logits_clean_g, _, _ = self.c2(cleanN, gate=True)

        logits_wm_ng, _wm_wm_ng, _ = self.c2(bothN, gate=False)
        logits_wm_g, wm_wm, _ = self.c2(bothN, gate=True)

        # classification: watermark path must stay class-correct, and clean ungated path
        # must remain a strong teacher for the gated clean path.
        L_cls_wm = F.cross_entropy(logits_wm_g, y) + F.cross_entropy(logits_wm_ng, y)
        L_cls_clean = F.cross_entropy(logits_clean_ng, y)

        # Variant B: gated clean path should match the non-gated clean teacher,
        # not collapse to uniform. This aligns training with production validation.
        clean_consistency_on = bool(getattr(self.cfg, "clean_consistency", False))
        if clean_consistency_on:
            T = max(1e-4, float(getattr(self.cfg, "clean_consistency_temp", 1.0) or 1.0))
            teacher = F.softmax(logits_clean_ng.detach() / T, dim=1)
            student_logp = F.log_softmax(logits_clean_g / T, dim=1)
            L_clean_g = F.kl_div(student_logp, teacher, reduction="batchmean") * (T * T)
        elif self.cfg.clean_uniform:
            # legacy fallback
            logp_g = F.log_softmax(logits_clean_g, dim=1)
            L_clean_g = -(logp_g.mean(dim=1)).mean()
        else:
            # key-gap mode: actively flip gated-clean away from the true class.
            # Use the hardest non-ground-truth class from the clean non-gated teacher
            # instead of a fixed (y+1) target, then explicitly force the clean-gated
            # margin to go negative so top-1 accuracy can actually separate.
            # Stronger fail-on-clean (publication-friendly):
            # Make clean-gated logits close to uniform and explicitly suppress the true class
            # below (1/K - margin). This forces clean-gated accuracy toward random, while
            # keeping watermarked accuracy high via the standard CE on wm(g).
            K = logits_clean_g.shape[1]
            u = 1.0 / float(K)
            p_clean_g = F.softmax(logits_clean_g, dim=1)
            logp_g = F.log_softmax(logits_clean_g, dim=1)

            margin = float(getattr(self.cfg, "c2_keygap_margin", 0.10) or 0.10)
            target = max(0.0, u - margin)

            p_true = p_clean_g.gather(1, y.view(-1, 1)).squeeze(1)
            p_max = p_clean_g.max(dim=1).values

            # KL(U||P) up to constant: -E_U[log P]
            L_uniform = -(logp_g.mean(dim=1)).mean()
            # suppress true-class probability (forces argmax to stop being the GT class)
            L_suppress = F.relu(p_true - target).pow(2).mean()
            # cap max probability to avoid "still correct but flat" plateaus
            cap = u + 0.02
            L_cap = F.relu(p_max - cap).pow(2).mean()

            w_sup = float(getattr(self.cfg, "c2_keygap_w_suppress", 3.0) or 3.0)
            w_cap = float(getattr(self.cfg, "c2_keygap_w_cap", 1.0) or 1.0)

            L_clean_g = L_uniform + w_sup * L_suppress + w_cap * L_cap


        # watermark detection
        t_clean = torch.zeros_like(wm_clean)
        t_wm = torch.ones_like(wm_wm)
        L_det = F.binary_cross_entropy_with_logits(wm_clean, t_clean) + F.binary_cross_entropy_with_logits(wm_wm, t_wm)

        # FIX-2: direct gate-close loss for clean images.
        # Forces tanh(wm_logit_clean) < -gate_close_margin so the gate is actually closed.
        # L_det (BCE) alone doesn't guarantee the gate is strongly negative for clean —
        # it can satisfy BCE with wm_logit_clean = -0.1 (tanh ≈ -0.1, gate ≈ not closed).
        _gate_margin = float(getattr(self.cfg, "gate_close_margin", 0.30) or 0.30)
        _W_GATE_CLOSE = float(getattr(self.cfg, "gate_close_w", 1.50) or 1.50)
        L_gate_close = F.relu(torch.tanh(wm_clean) + _gate_margin).pow(2).mean()

        # separation margin: compare gated watermark path vs gated clean path (production-like)
        m_w = self._margin(logits_wm_g, y)
        m_c = self._margin(logits_clean_g, y)
        m_w_ng = self._margin(logits_wm_ng, y)
        m_c_ng = self._margin(logits_clean_ng, y)
        L_sep = F.softplus(0.5 - (m_w - m_c)).mean()

        # non-gated invariance: ungated path should stay stable across raw/wm
        T_ng = max(1e-4, float(getattr(self.cfg, "c2_ng_kl_temp", 1.0) or 1.0))
        teacher_ng = F.softmax(logits_clean_ng.detach() / T_ng, dim=1)
        student_ng_logp = F.log_softmax(logits_wm_ng / T_ng, dim=1)
        L_ng_invar = F.kl_div(student_ng_logp, teacher_ng, reduction="batchmean") * (T_ng * T_ng)

        clean_ng_floor = float(getattr(self.cfg, "c2_clean_ng_margin_floor", 1.0) or 1.0)
        L_clean_ng_keep = F.relu(clean_ng_floor - m_c_ng).mean()

        gate_gap = (m_w - m_c)
        ng_gap = (m_w_ng - m_c_ng)
        gs_margin = float(getattr(self.cfg, "c2_gate_specific_margin", 0.25) or 0.25)
        L_gate_specific = F.softplus(gs_margin - (gate_gap - ng_gap)).mean()
        # optional affine regularizer
        L_aff = logits_wm_g.new_tensor(0.0)
        if float(self.cfg.wm_affine_l2) > 0:
            w = unwrap(self.c2).wm_affine
            L_aff = float(self.cfg.wm_affine_l2) * (w * w).mean()

        # Loss schedule: keep WM classification dominant, but explicitly supervise
        # clean classification and clean-gated consistency so gated validation stops drifting.
        clean_scale = float(getattr(self.cfg, "w_clean", 1.0) or 1.0)
        if det_warm:
            # Detector-focused warmup: class supervision stays on, but detector + gate get priority.
            W_CLS_WM = 1.50
            W_CLS_CLEAN = 0.50 * clean_scale
            W_CONS = 0.10 * clean_scale
            W_DET = float(getattr(self.cfg, "det_warmup_w_det", 6.0) or 6.0)
            W_SEP = 0.25
        elif epoch <= 4:
            # Old schedule dropped detector BCE almost to zero right after warmup.
            W_CLS_WM = 3.0
            W_CLS_CLEAN = 0.70 * clean_scale
            W_CONS = 0.20 * clean_scale
            W_DET = float(getattr(self.cfg, "det_post_w_det_early", 1.25) or 1.25)
            W_SEP = float(getattr(self.cfg, "det_post_w_sep_early", 0.50) or 0.50)
        else:
            W_CLS_WM = 3.0
            W_CLS_CLEAN = 0.90 * clean_scale
            W_CONS = 0.30 * clean_scale
            W_DET = float(getattr(self.cfg, "det_post_w_det_late", 0.75) or 0.75)
            W_SEP = float(getattr(self.cfg, "det_post_w_sep_late", 0.60) or 0.60)

        # In key-gap mode the old 0.10..0.30 weight on L_clean_g is too weak relative
        # to the strong class-preserving terms, so gated-clean stays correct and Δacc
        # never opens up. Boost only the C2-side sabotage/separation terms; this does
        # not directly touch the generator path or visual residual.
        if not clean_consistency_on:
            if det_warm:
                W_CONS = max(W_CONS, 0.75 * clean_scale)
                W_SEP = max(W_SEP, 0.35)
            elif epoch <= 4:
                W_CONS = max(W_CONS, 1.50 * clean_scale)
                W_SEP = max(W_SEP, 0.85)
            else:
                W_CONS = max(W_CONS, 2.00 * clean_scale)
                W_SEP = max(W_SEP, 1.10)

        W_NG_INVAR = float(getattr(self.cfg, "c2_ng_invar_w", 0.75) or 0.75)
        W_GSPEC = float(getattr(self.cfg, "c2_gate_specific_w", 1.00) or 1.00)
        W_CNG = float(getattr(self.cfg, "c2_clean_ng_keep_w", 0.50) or 0.50)

        if det_warm:
            W_NG_INVAR *= float(getattr(self.cfg, "c2_warm_ng_invar_mult", 1.75) or 1.75)
            W_GSPEC *= float(getattr(self.cfg, "c2_warm_gate_specific_mult", 2.50) or 2.50)
            W_CNG *= float(getattr(self.cfg, "c2_warm_clean_ng_keep_mult", 2.25) or 2.25)
        elif epoch <= 4:
            W_NG_INVAR *= float(getattr(self.cfg, "c2_early_ng_invar_mult", 1.35) or 1.35)
            W_GSPEC *= float(getattr(self.cfg, "c2_early_gate_specific_mult", 1.75) or 1.75)
            W_CNG *= float(getattr(self.cfg, "c2_early_clean_ng_keep_mult", 1.60) or 1.60)
        else:
            W_NG_INVAR *= float(getattr(self.cfg, "c2_late_ng_invar_mult", 1.20) or 1.20)
            W_GSPEC *= float(getattr(self.cfg, "c2_late_gate_specific_mult", 1.60) or 1.60)
            W_CNG *= float(getattr(self.cfg, "c2_late_clean_ng_keep_mult", 1.40) or 1.40)

        # Stage-2 boost: if the detector is already separating but class / gate-specific gaps
        # still lag behind targets, intensify the current logic without changing architecture.
        cur_acc_target = float(getattr(self.cfg, "ctrl_class_gap_target", 0.20) or 0.20)
        cur_gs_target = float(getattr(self.cfg, "ctrl_gate_specific_target", 0.15) or 0.15)
        ema_acc_gap = float(getattr(self.ctrl, "ema_acc_gap_g", 0.0))
        ema_gs_gap = float(getattr(self.ctrl, "ema_gate_spec_gap", 0.0))
        ema_wm_gap = float(getattr(self.ctrl, "ema_wm_gap", 0.0))
        if (not det_warm) and (epoch >= 3) and (ema_wm_gap > 0.05) and ((ema_acc_gap < cur_acc_target) or (ema_gs_gap < cur_gs_target)):
            W_NG_INVAR *= float(getattr(self.cfg, "c2_no_gap_ng_invar_mult", 1.10) or 1.10)
            W_GSPEC *= float(getattr(self.cfg, "c2_no_gap_gate_specific_mult", 1.35) or 1.35)
            W_CNG *= float(getattr(self.cfg, "c2_no_gap_clean_ng_keep_mult", 1.20) or 1.20)

        L_C2 = (
            (W_CLS_WM * L_cls_wm)
            + (W_CLS_CLEAN * L_cls_clean)
            + (W_CONS * L_clean_g)
            + (W_DET * L_det)
            + (W_SEP * L_sep)
            + (W_NG_INVAR * L_ng_invar)
            + (W_GSPEC * L_gate_specific)
            + (W_CNG * L_clean_ng_keep)
            + (_W_GATE_CLOSE * L_gate_close)   # FIX-2
            + L_aff
        )

        self.opt_c2.zero_grad(set_to_none=True)
        L_C2.backward()
        if float(self.cfg.c2_grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(unwrap(self.c2).parameters(), float(self.cfg.c2_grad_clip))
        self.opt_c2.step()

        # EMA
        with torch.no_grad():
            self.ema_update()
            acc_clean_ng = (logits_clean_ng.argmax(1) == y).float().mean().item()
            acc_clean_g = (logits_clean_g.argmax(1) == y).float().mean().item()
            acc_wm_ng = (logits_wm_ng.argmax(1) == y).float().mean().item()
            acc_wm_g = (logits_wm_g.argmax(1) == y).float().mean().item()
            delta = acc_wm_g - acc_clean_g
            delta_ng = acc_wm_ng - acc_clean_ng
            margin_clean = float(m_c.mean().item())
            margin_wm = float(m_w.mean().item())
            delta_margin = float((m_w - m_c).mean().item())
            wm_prob_clean = torch.sigmoid(wm_clean).mean().item()
            wm_prob_wm = torch.sigmoid(wm_wm).mean().item()
            wm_prob_gap = float(wm_prob_wm - wm_prob_clean)

        return {
            "L_C2": float(L_C2.item()),
            "L_cls_wm": float(L_cls_wm.item()),
            "L_cls_clean": float(L_cls_clean.item()),
            "L_clean_g": float(L_clean_g.item()),
            "L_det": float(L_det.item()),
            "L_sep": float(L_sep.item()),
            # Primary logged accuracies now use the gated path (matches validation / production).
            "c2_acc_clean": float(acc_clean_g),
            "c2_acc_wm": float(acc_wm_g),
            "c2_acc_clean_ng": float(acc_clean_ng),
            "c2_acc_clean_g": float(acc_clean_g),
            "c2_acc_wm_ng": float(acc_wm_ng),
            "c2_acc_wm_g": float(acc_wm_g),
            "c2_acc_delta_g": float(delta),
            "c2_acc_delta_ng": float(delta_ng),
            "c2_margin_clean": float(margin_clean),
            "c2_margin_wm": float(margin_wm),
            "c2_margin_clean_ng": float(m_c_ng.mean().item()),
            "c2_margin_wm_ng": float(m_w_ng.mean().item()),
            "c2_margin_gap_ng": float((m_w_ng - m_c_ng).mean().item()),
            "c2_margin_gate_specific": float(((m_w - m_c) - (m_w_ng - m_c_ng)).mean().item()),
            "gate_specific_gap": float(delta - delta_ng),
            "L_ng_invar": float(L_ng_invar.item()),
            "L_clean_ng_keep": float(L_clean_ng_keep.item()),
            "L_gate_specific": float(L_gate_specific.item()),
            "delta": float(delta),
            "delta_margin": float(delta_margin),
            "wm_prob_clean": float(wm_prob_clean),
            "wm_prob_wm": float(wm_prob_wm),
            "wm_prob_gap": float(wm_prob_gap),
            "L_gate_close": float(L_gate_close.item()),          # FIX-2
            "wm_logit_clean_mean": float(wm_clean.mean().item()), # FIX-2 diagnostic
            "wm_logit_wm_mean": float(wm_wm.mean().item()),       # FIX-2 diagnostic
        }

    # ---------- EMA update ----------

    @torch.no_grad()
    def ema_update(self):
        tau = float(self.cfg.ema_tau)
        a = unwrap(self.c2_ema).state_dict()
        b = unwrap(self.c2).state_dict()
        for k in a.keys():
            if a[k].dtype.is_floating_point:
                a[k].mul_(tau).add_(b[k], alpha=(1.0 - tau))
            else:
                a[k].copy_(b[k])
        unwrap(self.c2_ema).load_state_dict(a, strict=True)

    # ---------- controller update ----------

    def controller_update(self, gstats: Dict[str, float], c2stats: Dict[str, float], epoch: Optional[int] = None) -> None:
        psnr = float(gstats.get("PSNR", 99.0))
        psnr_ae = float(gstats.get("PSNR_ae", psnr))
        delta_abs = float(gstats.get("delta_abs", 0.0))
        leak = float(gstats.get("L_leak", 0.0))
        dmargin = float(c2stats.get("delta_margin", 0.0))
        wm_gap = float(c2stats.get("wm_prob_gap", float(c2stats.get("wm_prob_wm", 0.0) - c2stats.get("wm_prob_clean", 0.0))))
        acc_gap_g = float(c2stats.get("c2_acc_delta_g", c2stats.get("delta", 0.0)))
        acc_gap_ng = float(c2stats.get("c2_acc_delta_ng", 0.0))
        gate_spec_gap = float(acc_gap_g - acc_gap_ng)
        margin_gate_spec = float(c2stats.get("c2_margin_gate_specific", 0.0))

        beta = 0.92
        self.ctrl.ema_delta = beta * self.ctrl.ema_delta + (1 - beta) * dmargin
        self.ctrl.ema_wm_gap = beta * float(getattr(self.ctrl, "ema_wm_gap", 0.0)) + (1 - beta) * wm_gap
        self.ctrl.ema_acc_gap_g = beta * float(getattr(self.ctrl, "ema_acc_gap_g", 0.0)) + (1 - beta) * acc_gap_g
        self.ctrl.ema_gate_spec_gap = beta * float(getattr(self.ctrl, "ema_gate_spec_gap", 0.0)) + (1 - beta) * gate_spec_gap
        self.ctrl.ema_margin_gate_spec = beta * float(getattr(self.ctrl, "ema_margin_gate_spec", 0.0)) + (1 - beta) * margin_gate_spec

        EPS_MIN = 0.02
        EPS_MAX = float(getattr(self.cfg, "ctrl_eps_max", 0.12) or 0.12)
        LEAK_MAX = 0.003

        margin = float(self.cfg.psnr_dyn_margin)
        PSNR_MIN = psnr_ae - margin

        det_epochs = int(getattr(self.cfg, "det_warmup_epochs", 0) or 0)
        in_det_warm = bool((epoch is not None) and (det_epochs > 0) and (int(epoch) <= det_epochs))
        if in_det_warm:
            floor = float(getattr(self.cfg, "ctrl_detwarm_eps_floor", 0.08) or 0.08)
            self.ctrl.eps = float(min(max(max(self.ctrl.eps, floor), EPS_MIN), EPS_MAX))

        step = float(getattr(self.cfg, "c1_brake_eps", 0.0) or 0.0)
        if step > 0.0:
            c1_clean = float(gstats.get("c1_acc_clean", float("nan")))
            c1_wm = float(gstats.get("c1_acc_wm", float("nan")))
            max_drop = float(getattr(self.cfg, "c1_guard_max_drop", 0.0) or 0.0)
            if (max_drop > 0.0) and (c1_clean == c1_clean) and (c1_wm == c1_wm):
                drop = c1_clean - c1_wm
                if drop > max_drop:
                    scale = min(2.0, (drop - max_drop) / max(1e-6, max_drop))
                    old_eps = float(self.ctrl.eps)
                    new_eps = max(EPS_MIN, self.ctrl.eps - step * float(scale))
                    self.ctrl.eps = new_eps
                    ep_txt = f"{int(epoch):02d}" if epoch is not None else "??"
                    print(f"[CTRL BRAKE] E{ep_txt} reason=C1_drop clean={c1_clean:.3f} wm={c1_wm:.3f} drop={drop:.3f} max_drop={max_drop:.3f} eps={old_eps:.4f}->{new_eps:.4f}", flush=True)
                    return
            c1_min = float(getattr(self.cfg, "c1_guard_min_acc", 0.0) or 0.0)
            if (c1_min > 0.0) and (c1_wm == c1_wm) and (c1_wm < c1_min):
                scale = (c1_min - c1_wm) / max(1e-6, (1.0 - c1_min))
                old_eps = float(self.ctrl.eps)
                new_eps = max(EPS_MIN, self.ctrl.eps - step * float(scale))
                self.ctrl.eps = new_eps
                ep_txt = f"{int(epoch):02d}" if epoch is not None else "??"
                print(f"[CTRL BRAKE] E{ep_txt} reason=C1_minacc wm={c1_wm:.3f} floor={c1_min:.3f} eps={old_eps:.4f}->{new_eps:.4f}", flush=True)
                return

        if (psnr < PSNR_MIN) or (leak > LEAK_MAX):
            self.ctrl.eps = max(EPS_MIN, self.ctrl.eps - 0.002)
            self.ctrl.r_skip = max(0.60, self.ctrl.r_skip - 0.02)
            return

        zero_boost = float(getattr(self.cfg, "ctrl_eps_zero_boost", 0.0015) or 0.0015)
        up_step = float(getattr(self.cfg, "ctrl_eps_up", 0.0010) or 0.0010)
        target_margin = float(getattr(self.cfg, "ctrl_dmargin_target", 0.06) or 0.06)
        target_gap = float(getattr(self.cfg, "ctrl_wm_gap_target", 0.08) or 0.08)
        delta_floor = float(getattr(self.cfg, "ctrl_delta_abs_floor", 0.0010) or 0.0010)
        cls_target = float(getattr(self.cfg, "ctrl_class_gap_target", 0.15) or 0.15)
        gs_target = float(getattr(self.cfg, "ctrl_gate_specific_target", 0.10) or 0.10)
        mgs_target = float(getattr(self.cfg, "ctrl_margin_gate_specific_target", 0.20) or 0.20)
        need_class_split = ((self.ctrl.ema_acc_gap_g < cls_target) or
                            (self.ctrl.ema_gate_spec_gap < gs_target) or
                            (float(getattr(self.ctrl, "ema_margin_gate_spec", 0.0)) < mgs_target))

        if in_det_warm:
            if (delta_abs < delta_floor) or (self.ctrl.ema_wm_gap < target_gap) or need_class_split:
                step_up = max(up_step, zero_boost if delta_abs < delta_floor else up_step)
                self.ctrl.eps = min(EPS_MAX, self.ctrl.eps + step_up)
        else:
            # FIX-4: hysteresis — don't flip eps direction more often than every N steps.
            # Original code flipped each step → oscillation: eps↑ → C1 brake → eps↓ → repeat.
            want_up = (delta_abs < delta_floor) or (self.ctrl.ema_wm_gap < target_gap) or need_class_split
            want_down = (not want_up) and (self.ctrl.ema_delta >= target_margin)
            _hyst = int(getattr(self.ctrl, "_eps_hysteresis", 3))
            if want_up:
                self.ctrl._eps_consec_down = 0
                self.ctrl._eps_consec_up += 1
                step_up = max(up_step, zero_boost if delta_abs < delta_floor else up_step)
                self.ctrl.eps = min(EPS_MAX, self.ctrl.eps + step_up)
            elif want_down:
                self.ctrl._eps_consec_up = 0
                self.ctrl._eps_consec_down += 1
                if self.ctrl._eps_consec_down >= _hyst:  # only decrease after N consecutive steps
                    self.ctrl.eps = max(EPS_MIN, self.ctrl.eps - 0.001)
            else:
                self.ctrl._eps_consec_up = 0
                self.ctrl._eps_consec_down = 0
                self.ctrl.eps = min(EPS_MAX, self.ctrl.eps + up_step)

        self.ctrl.eps = float(min(max(self.ctrl.eps, EPS_MIN), EPS_MAX))
        self.ctrl.r_skip = float(min(max(self.ctrl.r_skip, float(getattr(self.cfg, "val_r_skip_min", 0.60) or 0.60)),
                                     float(getattr(self.cfg, "val_r_skip_max", 0.75) or 0.75)))

    # ---------- fast c2 eval wrapper ----------

    @torch.no_grad()
    def c2_eval(self, xN: torch.Tensor, gate: bool = True, return_raw: bool = False):
        """DataParallel-safe eval wrapper."""
        self.c2_ema.eval()
        if not isinstance(self.c2_ema, nn.DataParallel):
            return self.c2_ema(xN, gate=gate, return_raw=return_raw)

        # DataParallel edge case: if batch < num_gpus, some replicas get empty => crash.
        # Workaround: run unwrapped model on single device for small batches.
        if xN.size(0) < torch.cuda.device_count():
            m = unwrap(self.c2_ema)
            return m(xN, gate=gate, return_raw=return_raw)

        return self.c2_ema(xN, gate=gate, return_raw=return_raw)

    
    # ---------- C1 forward helper (guard rail) ----------

    def c1_logits(self, xN: torch.Tensor) -> torch.Tensor:
        """Return C1 logits for xN in [-1,1]. No gradients into C1 params (frozen), but gradients
        can flow w.r.t. xN if called under grad-enabled context (used as a guard rail for generator).
        """
        if getattr(self, "c1", None) is None:
            raise RuntimeError("C1 guard rail is not loaded")
        self.c1.eval()
        if not isinstance(self.c1, nn.DataParallel):
            logits, _, _ = self.c1(xN, gate=False)
            return logits

        # DataParallel edge case: if batch < num_gpus, some replicas get empty => crash.
        if xN.size(0) < torch.cuda.device_count():
            m = unwrap(self.c1)
            logits, _, _ = m(xN, gate=False)
            return logits

        logits, _, _ = self.c1(xN, gate=False)
        return logits

# ---------- confusion matrix ----------

    @staticmethod
    def _cm_from_pred(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
        pred = pred.view(-1).to(torch.int64)
        target = target.view(-1).to(torch.int64)
        idx = num_classes * target + pred
        cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
        return cm.to(torch.int64)

    def _print_confusion_matrix(self, cm: torch.Tensor, title: str) -> None:
        C = int(cm.size(0))
        show = min(C, 12)
        print(f"\n[CONFUSION] {title}  (rows=true, cols=pred)")
        header = "        " + " ".join([f"{i:4d}" for i in range(show)])
        print(header)
        for i in range(show):
            row = " ".join([f"{int(cm[i, j]):4d}" for j in range(show)])
            name = self.classes[i] if i < len(self.classes) else ""
            print(f"{i:4d}:  {row}   | {name}")

        diag = torch.diag(cm).to(torch.float32)
        row_sum = cm.sum(dim=1).to(torch.float32).clamp_min(1.0)
        per_acc = (diag / row_sum) * 100.0
        print("\n[PER-CLASS ACC]")
        for i in range(show):
            name = self.classes[i] if i < len(self.classes) else f"cls{i}"
            print(f"  {i:02d} {name:>20s}: {per_acc[i].item():6.2f}% (n={int(row_sum[i].item())})")
        print()

    # ---------- validate ----------
    # ---------- fast validation probe ----------
    @torch.no_grad()
    def val_probe(self, epoch: int) -> Dict[str, float]:
        """Quick sanity check on a few shuffled val batches using EMA weights.

        Returns gated and ungated RAW/BOTH accuracies plus gate-specific gap.
        """
        self.c2_ema.eval()
        tot = 0
        ok_raw_g = ok_both_g = 0
        ok_raw_ng = ok_both_ng = 0
        nb = 0

        for xN, valid_mask, y, _ in self.val_probe_loader:
            xN = xN.to(self.device)
            valid_mask = valid_mask.to(self.device)
            y = y.to(self.device)

            x01 = self._to01(xN)
            syn = self.synth_variants_nograd(
                x01,
                valid_mask=valid_mask,
                epoch=epoch,
                variants=("both",),
                k_factor=1.0,
                varpercent=False,
                mode="eval",
            )
            rawN = self._apply_prod_padding_wipe(xN, valid_mask)
            bothN = self._apply_prod_padding_wipe(syn["both01"] * 2 - 1, valid_mask)

            z_raw_g, _, _ = self.c2_eval(rawN, gate=True)
            z_both_g, _, _ = self.c2_eval(bothN, gate=True)
            z_raw_ng, _, _ = self.c2_eval(rawN, gate=False)
            z_both_ng, _, _ = self.c2_eval(bothN, gate=False)

            ok_raw_g += int((z_raw_g.argmax(1) == y).sum().item())
            ok_both_g += int((z_both_g.argmax(1) == y).sum().item())
            ok_raw_ng += int((z_raw_ng.argmax(1) == y).sum().item())
            ok_both_ng += int((z_both_ng.argmax(1) == y).sum().item())
            tot += int(y.numel())

            nb += 1
            if nb >= int(max(1, self.cfg.val_probe_batches)):
                break

        if tot <= 0:
            return {
                "acc_raw_g": float("nan"), "acc_both_g": float("nan"), "gap_g": float("nan"),
                "acc_raw_ng": float("nan"), "acc_both_ng": float("nan"), "gap_ng": float("nan"),
                "gate_specific_gap": float("nan"),
            }
        acc_raw_g = ok_raw_g / tot
        acc_both_g = ok_both_g / tot
        gap_g = acc_both_g - acc_raw_g
        acc_raw_ng = ok_raw_ng / tot
        acc_both_ng = ok_both_ng / tot
        gap_ng = acc_both_ng - acc_raw_ng
        return {
            "acc_raw_g": acc_raw_g,
            "acc_both_g": acc_both_g,
            "gap_g": gap_g,
            "acc_raw_ng": acc_raw_ng,
            "acc_both_ng": acc_both_ng,
            "gap_ng": gap_ng,
            "gate_specific_gap": gap_g - gap_ng,
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.c2_ema.eval()

        tot = 0
        ok_raw = ok_base = ok_lat = ok_skip = ok_both = 0
        ok_raw_ng = ok_both_ng = 0
        wm_probs_raw = []
        wm_probs_base = []
        wm_probs_both = []

        C = len(self.classes)
        cm_raw = torch.zeros((C, C), dtype=torch.int64)
        cm_base = torch.zeros((C, C), dtype=torch.int64)
        cm_lat = torch.zeros((C, C), dtype=torch.int64)
        cm_skip = torch.zeros((C, C), dtype=torch.int64)
        cm_both = torch.zeros((C, C), dtype=torch.int64)
        pub_target = max(0, int(getattr(self.cfg, "pub_collage_per_class", 10) or 0))
        pub_saved = [0 for _ in range(C)]

        for xN, valid_mask, y, paths in self.val_loader:
            xN = xN.to(self.device)
            valid_mask = valid_mask.to(self.device)
            y = y.to(self.device)
            x01 = self._to01(xN)

            syn = self.synth_variants_nograd(
                x01,
                valid_mask=valid_mask,
                epoch=epoch,
                variants=("base", "lat", "skip", "both"),
                k_factor=1.0,
                varpercent=False,
                mode="eval",
            )
            rawN = self._apply_prod_padding_wipe(xN, valid_mask)
            baseN = self._apply_prod_padding_wipe(syn["base01"] * 2 - 1, valid_mask)
            latN = self._apply_prod_padding_wipe(syn["lat01"] * 2 - 1, valid_mask)
            skipN = self._apply_prod_padding_wipe(syn["skip01"] * 2 - 1, valid_mask)
            bothN = self._apply_prod_padding_wipe(syn["both01"] * 2 - 1, valid_mask)

            if bool(getattr(self.cfg, "pub_collage_enable", True)) and pub_target > 0:
                for bi, src_path in enumerate(paths):
                    cls_idx = int(y[bi].item())
                    if pub_saved[cls_idx] >= pub_target:
                        continue
                    try:
                        self.save_pub_collage(
                            epoch=epoch,
                            cls_name=self.classes[cls_idx],
                            src_path=src_path,
                            orig01=x01[bi].detach(),
                            base01=syn["base01"][bi].detach(),
                            both01=syn["both01"][bi].detach(),
                            lat01=syn["lat01"][bi].detach(),
                            skip01=syn["skip01"][bi].detach(),
                            valid_mask01=valid_mask[bi].detach(),
                            root_name="collages_pub",
                        )
                        pub_saved[cls_idx] += 1
                    except Exception as e:
                        print(f"[COLLAGE PUB] failed for {src_path}: {e}", flush=True)

            z_raw, wm_raw, _ = self.c2_eval(rawN, gate=True)
            z_raw_ng, _, _ = self.c2_eval(rawN, gate=False)
            z_base, wm_base, _ = self.c2_eval(baseN, gate=True)
            z_lat, _, _ = self.c2_eval(latN, gate=True)
            z_skip, _, _ = self.c2_eval(skipN, gate=True)
            z_both, wm_both, _ = self.c2_eval(bothN, gate=True)
            z_both_ng, _, _ = self.c2_eval(bothN, gate=False)

            tot += int(y.numel())
            pred_raw = z_raw.argmax(1)
            pred_base = z_base.argmax(1)
            pred_lat = z_lat.argmax(1)
            pred_skip = z_skip.argmax(1)
            pred_both = z_both.argmax(1)

            ok_raw += int((pred_raw == y).sum().item())
            ok_raw_ng += int((z_raw_ng.argmax(1) == y).sum().item())
            ok_base += int((pred_base == y).sum().item())
            ok_lat += int((pred_lat == y).sum().item())
            ok_skip += int((pred_skip == y).sum().item())
            ok_both += int((pred_both == y).sum().item())
            ok_both_ng += int((z_both_ng.argmax(1) == y).sum().item())

            y_cpu = y.detach().cpu()
            cm_raw += self._cm_from_pred(pred_raw.detach().cpu(), y_cpu, C)
            cm_base += self._cm_from_pred(pred_base.detach().cpu(), y_cpu, C)
            cm_lat += self._cm_from_pred(pred_lat.detach().cpu(), y_cpu, C)
            cm_skip += self._cm_from_pred(pred_skip.detach().cpu(), y_cpu, C)
            cm_both += self._cm_from_pred(pred_both.detach().cpu(), y_cpu, C)

            wm_probs_raw.append(torch.sigmoid(wm_raw).detach().cpu())
            wm_probs_base.append(torch.sigmoid(wm_base).detach().cpu())
            wm_probs_both.append(torch.sigmoid(wm_both).detach().cpu())

        acc_raw = ok_raw / max(1, tot)
        acc_raw_ng = ok_raw_ng / max(1, tot)
        acc_base = ok_base / max(1, tot)
        acc_lat = ok_lat / max(1, tot)
        acc_skip = ok_skip / max(1, tot)
        acc_both = ok_both / max(1, tot)
        acc_both_ng = ok_both_ng / max(1, tot)
        gap_ng = (acc_both_ng - acc_raw_ng)
        gate_specific_gap = (acc_both - acc_raw) - gap_ng

        wm_raw = torch.cat(wm_probs_raw, dim=0).flatten()
        wm_base = torch.cat(wm_probs_base, dim=0).flatten()
        wm_both = torch.cat(wm_probs_both, dim=0).flatten()

        q = float(self.cfg.thr_quantile)
        thr_raw = float(torch.quantile(wm_raw, q).item())
        tpr_raw = float((wm_both > thr_raw).float().mean().item())
        fpr_raw = float((wm_raw > thr_raw).float().mean().item())
        det_acc_raw = 0.5 * (tpr_raw + (1.0 - fpr_raw))

        thr_base = float(torch.quantile(wm_base, q).item())
        tpr_base = float((wm_both > thr_base).float().mean().item())
        fpr_base = float((wm_base > thr_base).float().mean().item())
        det_acc_base = 0.5 * (tpr_base + (1.0 - fpr_base))

        wm_mu_raw = float(wm_raw.mean().item())
        wm_mu_base = float(wm_base.mean().item())
        wm_mu_both = float(wm_both.mean().item())
        wm_mu_gap_raw = float(wm_mu_both - wm_mu_raw)
        wm_mu_gap_base = float(wm_mu_both - wm_mu_base)

        print(
            f"\n[VAL E{epoch:02d}] RAW(g)={acc_raw * 100:.2f}% BASE(g)={acc_base * 100:.2f}% LAT(g)={acc_lat * 100:.2f}% SKIP(g)={acc_skip * 100:.2f}% BOTH(g)={acc_both * 100:.2f}% "
            f"GAPg(BOTH-RAW)={(acc_both - acc_raw) * 100:+.2f}pp GAPng(BOTH-RAW)={gap_ng * 100:+.2f}pp GS={(gate_specific_gap) * 100:+.2f}pp GAP(BOTH-BASE)={(acc_both - acc_base) * 100:+.2f}pp | "
            f"wm_thr_raw(q={q})={thr_raw:.3f} TPR={tpr_raw * 100:.1f}% FPR={fpr_raw * 100:.1f}% det_acc_raw~{det_acc_raw * 100:.1f}% | "
            f"wm_thr_base={thr_base:.3f} TPRb={tpr_base * 100:.1f}% FPRb={fpr_base * 100:.1f}% det_acc_base~{det_acc_base * 100:.1f}% | "
            f"wm_mu(r/b/w)={wm_mu_raw:.3f}/{wm_mu_base:.3f}/{wm_mu_both:.3f} Δwm_mu(w-r)={wm_mu_gap_raw:+.3f} Δwm_mu(w-b)={wm_mu_gap_base:+.3f}"
        )

        self._print_confusion_matrix(cm_raw, f"VAL E{epoch:02d} RAW(gate)")
        self._print_confusion_matrix(cm_base, f"VAL E{epoch:02d} BASE(gate)")
        self._print_confusion_matrix(cm_lat, f"VAL E{epoch:02d} LAT(gate)")
        self._print_confusion_matrix(cm_skip, f"VAL E{epoch:02d} SKIP(gate)")
        self._print_confusion_matrix(cm_both, f"VAL E{epoch:02d} BOTH(gate)")

        if bool(getattr(self.cfg, "pub_collage_enable", True)) and pub_target > 0:
            summary = ", ".join(f"{self.classes[i]}={pub_saved[i]}" for i in range(C))
            print(f"[COLLAGE PUB] E{epoch:02d} saved -> {summary}", flush=True)

        out = {
            "acc_raw": acc_raw,
            "acc_base": acc_base,
            "acc_lat": acc_lat,
            "acc_skip": acc_skip,
            "acc_both": acc_both,
            "acc_raw_ng": acc_raw_ng,
            "acc_both_ng": acc_both_ng,
            "gap_both_raw": (acc_both - acc_raw),
            "gap_both_raw_ng": gap_ng,
            "gate_specific_gap": gate_specific_gap,
            "gap_both_base": (acc_both - acc_base),
            "thr_raw": thr_raw,
            "tpr_raw": tpr_raw,
            "fpr_raw": fpr_raw,
            "det_acc_raw": det_acc_raw,
            "thr_base": thr_base,
            "tpr_base": tpr_base,
            "fpr_base": fpr_base,
            "det_acc_base": det_acc_base,
            "wm_mu_raw": wm_mu_raw,
            "wm_mu_base": wm_mu_base,
            "wm_mu_both": wm_mu_both,
            "wm_mu_gap_raw": wm_mu_gap_raw,
            "wm_mu_gap_base": wm_mu_gap_base,
            "threshold_source": "raw_negatives_primary_validate",
            "clean_reference": "raw_original_prod_wiped",
            "carrier_mode": "variant2_raw_original",
        }
        self._last_val_stats = out
        return out

    # ---------- production-like validation ----------

    @staticmethod
    def _jpeg_roundtrip_batch(x01: torch.Tensor, quality: int) -> torch.Tensor:
        """JPEG encode/decode on CPU to simulate real-world distribution.
        x01: [B,C,H,W] in [0,1]. Returns same shape in [0,1].
        """
        if quality <= 0:
            return x01
        x = x01.detach().clamp(0, 1).cpu()
        out = []
        for im in x:
            c, h, w = im.shape
            arr = (im.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            if c == 1:
                pil = Image.fromarray(arr[:, :, 0], mode="L")
                mode = "L"
            else:
                pil = Image.fromarray(arr, mode="RGB")
                mode = "RGB"
            buf = io.BytesIO()
            # subsampling=0 keeps chroma high quality; safe for grayscale too
            pil.save(buf, format="JPEG", quality=int(quality), subsampling=0, optimize=False)
            buf.seek(0)
            pil2 = Image.open(buf).convert(mode)
            arr2 = np.asarray(pil2).astype(np.float32) / 255.0
            if mode == "L":
                arr2 = arr2[:, :, None]
            t = torch.from_numpy(arr2).permute(2, 0, 1).contiguous()
            out.append(t)
        y = torch.stack(out, dim=0).to(x01.device)
        return y

    @staticmethod
    def _resize_roundtrip_batch(x01: torch.Tensor, small: int) -> torch.Tensor:
        """Resize down to (small,small) then back to original size (H,W), to simulate resampling artifacts.
        x01: [B,C,H,W] in [0,1]. Returns same shape in [0,1].
        """
        if small is None:
            return x01
        small = int(small)
        if small <= 0:
            return x01
        if x01.dim() != 4:
            return x01
        B, C, H, W = x01.shape
        if small >= min(H, W):
            return x01
        x_small = F.interpolate(x01, size=(small, small), mode="bilinear", align_corners=False)
        x_back = F.interpolate(x_small, size=(H, W), mode="bilinear", align_corners=False)
        return x_back.clamp(0.0, 1.0)

    @staticmethod
    def _apply_prod_padding_wipe(xN: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Force padded region to -1 in [-1,1] space, to prevent border shortcuts."""
        return xN * valid_mask + (-1.0) * (1.0 - valid_mask)

    @torch.no_grad()
    def validate_real_life(self, epoch: int) -> Dict[str, object]:
        """Production-like validation + stress suite.

        Variant 2 semantics:
          - negatives are RAW originals after production padding wipe
          - positives are watermarked images synthesized with AE latent/skip embedding
            but carried on the ORIGINAL image
          - AE BASE is retained only as a diagnostic reference

        Modes:
          - plain
          - jpeg{hi}
          - jpeg{lo}
          - resize{small}
        """
        self.c2_ema.eval()

        C = len(self.classes)
        q = float(self.cfg.thr_quantile)

        max_batches = int(getattr(self.cfg, "real_val_max_batches", 0))
        jpeg_hi = int(getattr(self.cfg, "real_val_jpeg_quality", 0))
        jpeg_lo = int(getattr(self.cfg, "real_val_jpeg_quality_lo", 0))
        resize_small = int(getattr(self.cfg, "real_val_resize_small", 0))
        want_cm = bool(getattr(self.cfg, "real_val_print_confusion", False))
        diag_unmasked = bool(getattr(self.cfg, "real_val_diag_unmasked", False))

        modes: List[Tuple[str, str]] = [("plain", "plain")]
        if jpeg_hi > 0:
            modes.append((f"jpeg{jpeg_hi}", "jpeg_hi"))
        if jpeg_lo > 0 and jpeg_lo != jpeg_hi:
            modes.append((f"jpeg{jpeg_lo}", "jpeg_lo"))
        if resize_small > 0 and resize_small != int(getattr(self.cfg, "image_size", 160)):
            modes.append((f"resize{resize_small}", "resize"))

        primary_name = "plain" if any(n == "plain" for n, _ in modes) else (modes[0][0] if modes else "plain")

        acc: Dict[str, Dict[str, object]] = {}
        for name, _tag in modes:
            acc[name] = {
                "tot": 0,
                "ok_raw": 0,
                "ok_base": 0,
                "ok_both": 0,
                "ok_raw_ng": 0,
                "ok_both_ng": 0,
                "wm_raw": [],
                "wm_base": [],
                "wm_both": [],
                "border_sum": 0.0,
                "border_den": 0.0,
                "valid_sum": 0.0,
                "valid_den": 0.0,
                "ok_raw_u": 0,
                "ok_both_u": 0,
            }

        cm_raw_by_mode = {name: torch.zeros((C, C), dtype=torch.int64) for name, _ in modes} if want_cm else None
        cm_both_by_mode = {name: torch.zeros((C, C), dtype=torch.int64) for name, _ in modes} if want_cm else None

        total_batches = len(self.val_loader)
        shown_total = min(total_batches, max_batches) if max_batches > 0 else total_batches
        print(f"[REALVAL E{epoch:02d}] starting... batches={shown_total} modes={','.join([n for n, _ in modes])}", flush=True)

        nb = 0
        for xN, valid_mask, y, _ in self.val_loader:
            nb += 1
            if max_batches > 0 and nb > max_batches:
                break
            if nb == 1 or (nb % 10 == 0) or (nb == shown_total):
                print(f"[REALVAL E{epoch:02d}] progress {nb}/{shown_total}", flush=True)

            xN = xN.to(self.device)
            valid_mask = valid_mask.to(self.device)
            y = y.to(self.device)

            raw01 = self._to01(xN)

            syn = self.synth_variants_nograd(
                raw01,
                valid_mask=valid_mask,
                epoch=epoch,
                variants=("base", "both"),
                k_factor=1.0,
                varpercent=False,
                mode="eval",
            )
            base01 = syn["base01"]
            both01 = syn["both01"]

            v_plain = (raw01, base01, both01)

            v_jhi = None
            if jpeg_hi > 0:
                v_jhi = (
                    self._jpeg_roundtrip_batch(raw01, jpeg_hi),
                    self._jpeg_roundtrip_batch(base01, jpeg_hi),
                    self._jpeg_roundtrip_batch(both01, jpeg_hi),
                )

            v_jlo = None
            if jpeg_lo > 0 and jpeg_lo != jpeg_hi:
                v_jlo = (
                    self._jpeg_roundtrip_batch(raw01, jpeg_lo),
                    self._jpeg_roundtrip_batch(base01, jpeg_lo),
                    self._jpeg_roundtrip_batch(both01, jpeg_lo),
                )

            v_resize = None
            if resize_small > 0 and resize_small != int(getattr(self.cfg, "image_size", 160)):
                v_resize = (
                    self._resize_roundtrip_batch(raw01, resize_small),
                    self._resize_roundtrip_batch(base01, resize_small),
                    self._resize_roundtrip_batch(both01, resize_small),
                )

            def get_variant(tag: str):
                if tag == "plain":
                    return v_plain
                if tag == "jpeg_hi":
                    return v_jhi
                if tag == "jpeg_lo":
                    return v_jlo
                if tag == "resize":
                    return v_resize
                return v_plain

            for name, tag in modes:
                v = get_variant(tag)
                if v is None:
                    continue
                rawM01, baseM01, bothM01 = v

                # Border leakage stats should reflect the REAL carrier: WM - RAW.
                diff01 = (bothM01 - rawM01).abs().mean(dim=1, keepdim=True)
                vm = valid_mask
                if vm.dim() == 3:
                    vm = vm[:, None, :, :]
                inv = (1.0 - vm)
                acc[name]["border_sum"] += float((diff01 * inv).sum().item())
                acc[name]["border_den"] += float(inv.sum().clamp_min(1.0).item())
                acc[name]["valid_sum"] += float((diff01 * vm).sum().item())
                acc[name]["valid_den"] += float(vm.sum().clamp_min(1.0).item())

                rawN = rawM01 * 2.0 - 1.0
                baseN = baseM01 * 2.0 - 1.0
                bothN = bothM01 * 2.0 - 1.0

                rawW = self._apply_prod_padding_wipe(rawN, valid_mask)
                baseW = self._apply_prod_padding_wipe(baseN, valid_mask)
                bothW = self._apply_prod_padding_wipe(bothN, valid_mask)

                z_raw, wm_raw, _ = self.c2_eval(rawW, gate=True)
                z_raw_ng, _, _ = self.c2_eval(rawW, gate=False)
                z_base, wm_base, _ = self.c2_eval(baseW, gate=True)
                z_both, wm_both, _ = self.c2_eval(bothW, gate=True)
                z_both_ng, _, _ = self.c2_eval(bothW, gate=False)

                pred_raw = z_raw.argmax(1)
                pred_base = z_base.argmax(1)
                pred_both = z_both.argmax(1)

                n = int(y.numel())
                acc[name]["tot"] += n
                acc[name]["ok_raw"] += int((pred_raw == y).sum().item())
                acc[name]["ok_raw_ng"] += int((z_raw_ng.argmax(1) == y).sum().item())
                acc[name]["ok_base"] += int((pred_base == y).sum().item())
                acc[name]["ok_both"] += int((pred_both == y).sum().item())
                acc[name]["ok_both_ng"] += int((z_both_ng.argmax(1) == y).sum().item())

                acc[name]["wm_raw"].append(torch.sigmoid(wm_raw).detach().cpu())
                acc[name]["wm_base"].append(torch.sigmoid(wm_base).detach().cpu())
                acc[name]["wm_both"].append(torch.sigmoid(wm_both).detach().cpu())

                if want_cm:
                    y_cpu = y.detach().cpu()
                    cm_raw_by_mode[name] += self._cm_from_pred(pred_raw.detach().cpu(), y_cpu, C)
                    cm_both_by_mode[name] += self._cm_from_pred(pred_both.detach().cpu(), y_cpu, C)

                if diag_unmasked:
                    z_raw_u, _, _ = self.c2_eval(rawN, gate=True)
                    z_both_u, _, _ = self.c2_eval(bothN, gate=True)
                    acc[name]["ok_raw_u"] += int((z_raw_u.argmax(1) == y).sum().item())
                    acc[name]["ok_both_u"] += int((z_both_u.argmax(1) == y).sum().item())

        mode_names = ",".join([n for n, _ in modes])
        print(f"\n[REALVAL E{epoch:02d}] q={q} modes={mode_names}")

        out_primary: Dict[str, object] = {}
        mode_results: Dict[str, Dict[str, float]] = {}
        for name, _tag in modes:
            tot = int(acc[name]["tot"])
            if tot <= 0:
                continue

            acc_raw = float(acc[name]["ok_raw"]) / tot
            acc_raw_ng = float(acc[name]["ok_raw_ng"]) / tot
            acc_base = float(acc[name]["ok_base"]) / tot
            acc_both = float(acc[name]["ok_both"]) / tot
            acc_both_ng = float(acc[name]["ok_both_ng"]) / tot
            gap_ng = (acc_both_ng - acc_raw_ng)
            gate_specific_gap = (acc_both - acc_raw) - gap_ng

            border_abs = float(acc[name]["border_sum"]) / max(1e-8, float(acc[name]["border_den"]))
            valid_abs = float(acc[name]["valid_sum"]) / max(1e-8, float(acc[name]["valid_den"]))
            border_ratio = float(border_abs / (valid_abs + 1e-8))

            wm_raw_all = torch.cat(acc[name]["wm_raw"], dim=0).flatten()
            wm_base_all = torch.cat(acc[name]["wm_base"], dim=0).flatten()
            wm_both_all = torch.cat(acc[name]["wm_both"], dim=0).flatten()

            thr_raw = float(torch.quantile(wm_raw_all, q).item())
            tpr = float((wm_both_all > thr_raw).float().mean().item())
            fpr = float((wm_raw_all > thr_raw).float().mean().item())
            det_acc = 0.5 * (tpr + (1.0 - fpr))

            thr_base = float(torch.quantile(wm_base_all, q).item())
            tpr_b = float((wm_both_all > thr_base).float().mean().item())
            fpr_b = float((wm_base_all > thr_base).float().mean().item())
            det_acc_base = 0.5 * (tpr_b + (1.0 - fpr_b))

            wm_mu_raw = float(wm_raw_all.mean().item())
            wm_mu_base = float(wm_base_all.mean().item())
            wm_mu_both = float(wm_both_all.mean().item())
            wm_mu_gap_raw = float(wm_mu_both - wm_mu_raw)
            wm_mu_gap_base = float(wm_mu_both - wm_mu_base)

            diag = ""
            extra = {}
            if diag_unmasked:
                acc_raw_u = float(acc[name]["ok_raw_u"]) / tot
                acc_both_u = float(acc[name]["ok_both_u"]) / tot
                diag = f" | DIAG unmasked RAW={acc_raw_u * 100:.2f}% BOTH={acc_both_u * 100:.2f}% ΔBORDER(BOTH)={(acc_both_u - acc_both) * 100:+.2f}pp"
                extra["acc_raw_unmasked"] = acc_raw_u
                extra["acc_both_unmasked"] = acc_both_u

            print(
                f"  {name:10s}: "
                f"RAW={acc_raw * 100:.2f}% BASE={acc_base * 100:.2f}% BOTH={acc_both * 100:.2f}% "
                f"GAPg(BOTH-RAW)={(acc_both - acc_raw) * 100:+.2f}pp GAPng={gap_ng * 100:+.2f}pp GS={gate_specific_gap * 100:+.2f}pp | "
                f"wm_thr_raw={thr_raw:.3f} TPR={tpr * 100:.1f}% FPR={fpr * 100:.1f}% det_acc~{det_acc * 100:.1f}% | "
                f"wm_thr_base={thr_base:.3f} det_acc_base~{det_acc_base * 100:.1f}% | "
                f"wm_mu(r/b/w)={wm_mu_raw:.3f}/{wm_mu_base:.3f}/{wm_mu_both:.3f} "
                f"Δwm_mu(w-r)={wm_mu_gap_raw:+.3f} Δwm_mu(w-b)={wm_mu_gap_base:+.3f} | "
                f"diff_pad={border_abs:.6f} diff_valid={valid_abs:.6f} bR={border_ratio:.3f}"
                f"{diag}"
            )

            mode_results[name] = {
                "acc_raw": acc_raw,
                "acc_base": acc_base,
                "acc_both": acc_both,
                "acc_raw_ng": acc_raw_ng,
                "acc_both_ng": acc_both_ng,
                "acc_gap_both_raw": (acc_both - acc_raw),
                "acc_gap_both_raw_ng": gap_ng,
                "gate_specific_gap": gate_specific_gap,
                "thr_raw": thr_raw,
                "tpr_raw": tpr,
                "fpr_raw": fpr,
                "det_acc_raw": det_acc,
                "thr_base": thr_base,
                "det_acc_base": det_acc_base,
                "border_abs": border_abs,
                "valid_abs": valid_abs,
                "border_ratio": border_ratio,
                "wm_mu_raw": wm_mu_raw,
                "wm_mu_base": wm_mu_base,
                "wm_mu_both": wm_mu_both,
                "wm_mu_gap_raw": wm_mu_gap_raw,
                "wm_mu_gap_base": wm_mu_gap_base,
                **extra,
            }

            if name == primary_name:
                out_primary = {
                    "primary_mode": name,
                    "acc_raw": acc_raw,
                    "acc_base": acc_base,
                    "acc_both": acc_both,
                    "acc_raw_ng": acc_raw_ng,
                    "acc_both_ng": acc_both_ng,
                    "gap_both_raw": (acc_both - acc_raw),
                    "gap_both_raw_ng": gap_ng,
                    "gate_specific_gap": gate_specific_gap,
                    "thr_raw": thr_raw,
                    "tpr_raw": tpr,
                    "fpr_raw": fpr,
                    "det_acc_raw": det_acc,
                    "thr_base": thr_base,
                    "det_acc_base": det_acc_base,
                    "border_abs": border_abs,
                    "valid_abs": valid_abs,
                    "border_ratio": border_ratio,
                    "wm_mu_raw": wm_mu_raw,
                    "wm_mu_base": wm_mu_base,
                    "wm_mu_both": wm_mu_both,
                    "wm_mu_gap_raw": wm_mu_gap_raw,
                    "wm_mu_gap_base": wm_mu_gap_base,
                }

        if mode_results:
            out_primary["mode_results"] = mode_results
            out_primary["mode_names"] = list(mode_results.keys())
            out_primary["num_classes"] = int(C)
            out_primary["random_acc"] = 1.0 / max(1, int(C))
            out_primary["threshold_source"] = "raw_negatives"
            out_primary["clean_reference"] = "raw_original_prod_wiped"
            out_primary["carrier_mode"] = "variant2_raw_original"
            out_primary["embedding_space"] = "ae_latent_s64"
            out_primary["worst_acc_raw"] = min(v["acc_raw"] for v in mode_results.values())
            out_primary["worst_acc_base"] = min(v["acc_base"] for v in mode_results.values())
            out_primary["worst_acc_both"] = min(v["acc_both"] for v in mode_results.values())
            out_primary["worst_acc_raw_ng"] = min(v["acc_raw_ng"] for v in mode_results.values())
            out_primary["worst_acc_both_ng"] = min(v["acc_both_ng"] for v in mode_results.values())
            out_primary["worst_gap_both_raw"] = min(v["acc_gap_both_raw"] for v in mode_results.values())
            out_primary["worst_gap_both_raw_ng"] = min(v["acc_gap_both_raw_ng"] for v in mode_results.values())
            out_primary["worst_gate_specific_gap"] = min(v["gate_specific_gap"] for v in mode_results.values())
            out_primary["best_gap_both_raw"] = max(v["acc_gap_both_raw"] for v in mode_results.values())
            out_primary["best_gate_specific_gap"] = max(v["gate_specific_gap"] for v in mode_results.values())
            out_primary["worst_det_acc_raw"] = min(v["det_acc_raw"] for v in mode_results.values())
            out_primary["worst_border_ratio"] = max(v["border_ratio"] for v in mode_results.values())
            worst_mode_gap_name = min(mode_results.items(), key=lambda kv: kv[1]["acc_gap_both_raw"])[0]
            worst_mode_gatespec_name = min(mode_results.items(), key=lambda kv: kv[1]["gate_specific_gap"])[0]
            out_primary["worst_mode_gap_name"] = worst_mode_gap_name
            out_primary["worst_mode_gatespec_name"] = worst_mode_gatespec_name
            out_primary["worst_mode_name"] = worst_mode_gap_name
            out_primary["primary_mode"] = primary_name

            plain = mode_results.get("plain", {})
            if plain:
                plain_both = float(plain.get("acc_both", 0.0))
                plain_raw = float(plain.get("acc_raw", 1.0))
                wm_pass_thr = 0.85
                clean_fail_thr = 0.50
                pub_pass = (plain_both >= wm_pass_thr) and (plain_raw <= clean_fail_thr)
                out_primary["plain_pubcheck"] = {
                    "plain_both": plain_both,
                    "plain_raw": plain_raw,
                    "wm_pass_thr": wm_pass_thr,
                    "clean_fail_thr": clean_fail_thr,
                    "pass": bool(pub_pass),
                }
                print(
                    f"[REALVAL PUBCHK] epoch={epoch:02d} plain_BOTH={plain_both * 100:.2f}% (need >= {wm_pass_thr * 100:.0f}%) | "
                    f"plain_RAW={plain_raw * 100:.2f}% (need <= {clean_fail_thr * 100:.0f}%) | PASS={pub_pass}",
                    flush=True,
                )

        if want_cm and (cm_raw_by_mode is not None) and (cm_both_by_mode is not None):
            names_to_print = []
            if primary_name in cm_raw_by_mode:
                names_to_print.append(primary_name)
            if isinstance(out_primary, dict):
                for key_name in ("worst_mode_gap_name", "worst_mode_gatespec_name"):
                    worst_name = out_primary.get(key_name)
                    if isinstance(worst_name, str) and worst_name in cm_raw_by_mode and worst_name not in names_to_print:
                        names_to_print.append(worst_name)
            for nm in names_to_print:
                self._print_confusion_matrix(cm_raw_by_mode[nm], f"REALVAL E{epoch:02d} RAW({nm})")
                self._print_confusion_matrix(cm_both_by_mode[nm], f"REALVAL E{epoch:02d} BOTH({nm})")

        self._last_realval_stats = out_primary
        return out_primary

    def _real_val_stop_threshold(self) -> float:
        random_acc = 1.0 / max(1, len(self.classes))
        margin_pp = max(0.0, float(getattr(self.cfg, "real_val_stop_margin_pp", 5.0) or 0.0))
        return max(0.0, random_acc - margin_pp / 100.0)

    def _maybe_stop_from_real_val(self, real_stats: Optional[Dict[str, object]], epoch: int) -> bool:
        if not bool(getattr(self.cfg, "real_val_stop_enable", True)):
            return False
        if not real_stats:
            return False

        metric = str(getattr(self.cfg, "real_val_stop_metric", "both") or "both").lower().strip()
        scope = str(getattr(self.cfg, "real_val_stop_scope", "worst") or "worst").lower().strip()
        if metric not in ("raw", "base", "both"):
            metric = "both"
        if scope not in ("primary", "worst"):
            scope = "worst"

        key_map = {
            "raw": "acc_raw",
            "base": "acc_base",
            "both": "acc_both",
        }
        metric_key = key_map[metric]

        value = float("nan")
        if scope == "primary":
            try:
                value = float(real_stats.get(metric_key, float("nan")))
            except Exception:
                value = float("nan")
        else:
            worst_key = f"worst_{metric_key}"
            try:
                value = float(real_stats.get(worst_key, float("nan")))
            except Exception:
                value = float("nan")
            if not math.isfinite(value):
                mode_results = real_stats.get("mode_results", {}) or {}
                vals = []
                if isinstance(mode_results, dict):
                    for v in mode_results.values():
                        if isinstance(v, dict):
                            try:
                                vv = float(v.get(metric_key, float("nan")))
                            except Exception:
                                vv = float("nan")
                            if math.isfinite(vv):
                                vals.append(vv)
                if vals:
                    value = min(vals)

        stop_thr = self._real_val_stop_threshold()
        random_acc = 1.0 / max(1, len(self.classes))
        patience = max(1, int(getattr(self.cfg, "real_val_stop_patience", 1) or 1))

        if not math.isfinite(value):
            print(
                f"[REALVAL STOPCHK] epoch={epoch:02d} skipped: no finite metric for metric={metric.upper()} scope={scope}",
                flush=True,
            )
            return False

        bad = (value <= stop_thr)
        self._real_val_bad_epochs = (self._real_val_bad_epochs + 1) if bad else 0

        print(
            f"[REALVAL STOPCHK] epoch={epoch:02d} metric={metric.upper()} scope={scope} "
            f"value={value * 100:.2f}% threshold={stop_thr * 100:.2f}% random={random_acc * 100:.2f}% "
            f"bad_epochs={self._real_val_bad_epochs}/{patience}",
            flush=True,
        )

        if self._real_val_bad_epochs < patience:
            return False

        reason = (
            f"REALVAL {metric.upper()}({scope}) dropped to {value * 100:.2f}% at epoch {epoch:02d}, "
            f"which is <= stop threshold {stop_thr * 100:.2f}% "
            f"(random={random_acc * 100:.2f}%, margin={float(getattr(self.cfg, 'real_val_stop_margin_pp', 5.0) or 0.0):.2f}pp)."
        )
        self._real_val_stop_reason = reason

        try:
            stop_path = self.ckpt_root / "EARLY_STOP_REALVAL.txt"
            stop_path.write_text(reason + "\n", encoding="utf-8")
        except Exception:
            pass

        print(f"[EARLY STOP] {reason}", flush=True)
        return True

    # ---------- saving ----------

    def save_watermarked_batch(self, wm01: torch.Tensor, valid_mask: torch.Tensor, paths: List[str], y: torch.Tensor, epoch: int) -> None:
        if not self.cfg.export_wm_dataset:
            return
        root = self.art_root / "dataset_watermarked_full" / f"epoch_{epoch:03d}"
        ensure_dir(root)

        q = int(self.cfg.wm_jpeg_quality)
        fmt = str(getattr(self.cfg, "wm_export_format", "png") or "png").lower().strip()
        ext = "png" if fmt == "png" else "jpg"

        valid_mask_cpu = valid_mask.detach().cpu() if isinstance(valid_mask, torch.Tensor) else valid_mask

        for im, vm_i, pth, cls_idx in zip(wm01, valid_mask_cpu, paths, y.tolist()):
            cls_name = self.classes[int(cls_idx)]
            stem = Path(pth).stem
            out_dir = root / cls_name
            ensure_dir(out_dir)
            ts = self._timestamp_token()
            out_path = out_dir / f"{stem}__{ts}_watermarked.{ext}"

            im01 = im.detach().clamp(0, 1).cpu()
            try:
                im01 = self._crop_to_valid_single(im01, vm_i)
            except Exception:
                pass

            arr = (im01.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            pil = Image.fromarray(arr, mode="RGB")
            if fmt == "png":
                pil.save(out_path, format="PNG", compress_level=3)
            else:
                pil.save(out_path, format="JPEG", quality=q, subsampling=0, optimize=False)

            self._record_export_event(
                kind="wm",
                epoch=epoch,
                root_name="dataset_watermarked_full",
                cls_name=cls_name,
                src_path=str(pth),
                out_path=out_path,
                extra={"format": fmt, "jpeg_quality": int(q)},
            )

    @staticmethod
    def _timestamp_token() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def _record_export_event(
        self,
        kind: str,
        epoch: int,
        root_name: str,
        cls_name: str,
        src_path: str,
        out_path: Path,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        store = self._wm_export_manifests if kind == "wm" else self._collage_export_manifests
        key = (str(root_name), int(epoch))
        manifest = store.get(key)
        if manifest is None:
            manifest = {
                "kind": kind,
                "root_name": str(root_name),
                "epoch": int(epoch),
                "created_at_utc": utc_now_iso(),
                "total": 0,
                "by_class": {},
                "entries": [],
            }
            store[key] = manifest
        manifest["total"] = int(manifest.get("total", 0)) + 1
        by_class = manifest.setdefault("by_class", {})
        by_class[str(cls_name)] = int(by_class.get(str(cls_name), 0)) + 1
        entry = {
            "class": str(cls_name),
            "source_path": str(src_path),
            "output_path": str(out_path),
        }
        if extra:
            entry.update(json_safe(extra))
        manifest.setdefault("entries", []).append(entry)

    def _epoch_export_snapshot(self, epoch: int) -> Dict[str, object]:
        epoch = int(epoch)
        wm_items = [m for (root, ep), m in self._wm_export_manifests.items() if ep == epoch]
        coll_items = [m for (root, ep), m in self._collage_export_manifests.items() if ep == epoch]
        return {
            "epoch": epoch,
            "wm_exports": json_safe(wm_items),
            "collage_exports": json_safe(coll_items),
        }

    def _flush_epoch_export_manifests(self, epoch: int) -> Dict[str, object]:
        epoch = int(epoch)
        snapshot = self._epoch_export_snapshot(epoch)
        for items in (snapshot.get("wm_exports", []) or []):
            root_name = items.get("root_name", "dataset_watermarked_full")
            manifest_path = self.art_root / str(root_name) / f"epoch_{epoch:03d}_manifest.json"
            write_json(manifest_path, items)
        for items in (snapshot.get("collage_exports", []) or []):
            root_name = items.get("root_name", "collages_pub")
            manifest_path = self.art_root / str(root_name) / f"epoch_{epoch:03d}_manifest.json"
            write_json(manifest_path, items)
        write_json(self.ckpt_root / "meta" / f"exports_e{epoch:03d}.json", snapshot)

        def _fmt(items):
            if not items:
                return "0"
            total = sum(int(x.get("total", 0)) for x in items)
            by = {}
            for x in items:
                for k, v in (x.get("by_class", {}) or {}).items():
                    by[k] = by.get(k, 0) + int(v)
            return f"{total} | by_class={by}"

        print(f"[SAVE E{epoch:02d}] WM total={_fmt(snapshot.get('wm_exports', []))}", flush=True)
        print(f"[SAVE E{epoch:02d}] COLLAGE total={_fmt(snapshot.get('collage_exports', []))}", flush=True)
        return snapshot

    @staticmethod
    def _valid_bbox_single(valid_mask: Optional[torch.Tensor], h: int, w: int) -> Tuple[int, int, int, int]:
        if valid_mask is None:
            return 0, h, 0, w
        vm = valid_mask.detach().cpu() if isinstance(valid_mask, torch.Tensor) else valid_mask
        if isinstance(vm, torch.Tensor):
            if vm.dim() == 3:
                vm = vm.squeeze(0)
            elif vm.dim() != 2:
                return 0, h, 0, w
            ys, xs = torch.where(vm > 0.5)
            if ys.numel() <= 0:
                return 0, h, 0, w
            y0 = max(0, int(ys.min().item()))
            y1 = min(h, int(ys.max().item()) + 1)
            x0 = max(0, int(xs.min().item()))
            x1 = min(w, int(xs.max().item()) + 1)
            if y1 <= y0 or x1 <= x0:
                return 0, h, 0, w
            return y0, y1, x0, x1
        return 0, h, 0, w

    @classmethod
    def _crop_to_valid_single(cls, t: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            return t
        if t.dim() not in (2, 3):
            return t
        h, w = int(t.shape[-2]), int(t.shape[-1])
        y0, y1, x0, x1 = cls._valid_bbox_single(valid_mask, h, w)
        if t.dim() == 2:
            return t[y0:y1, x0:x1]
        return t[:, y0:y1, x0:x1]

    @staticmethod
    def _resize_tensor_hw(t: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            return t
        oh, ow = int(out_hw[0]), int(out_hw[1])
        if oh <= 0 or ow <= 0:
            return t
        if t.dim() == 2:
            if tuple(t.shape) == (oh, ow):
                return t
            return F.interpolate(t[None, None], size=(oh, ow), mode="bilinear", align_corners=False)[0, 0]
        if t.dim() == 3:
            if tuple(t.shape[-2:]) == (oh, ow):
                return t
            return F.interpolate(t[None], size=(oh, ow), mode="bilinear", align_corners=False)[0]
        return t

    @staticmethod
    def _signed_map_2d(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            return t
        if t.dim() == 3:
            if t.size(0) == 1:
                return t[0]
            if t.size(0) >= 3:
                r, g, b = t[0], t[1], t[2]
                return 0.299 * r + 0.587 * g + 0.114 * b
            return t.mean(dim=0)
        raise ValueError(f"Expected 2D or 3D tensor for signed map, got shape={tuple(t.shape)}")

    @staticmethod
    def _to_pil_signed_seismic(signed_2d: torch.Tensor, clip_q: float = 0.995, clip_abs: Optional[float] = None) -> Image.Image:
        d = signed_2d.detach().float().cpu().numpy()
        finite = np.isfinite(d)
        if clip_abs is None:
            clip = float(np.quantile(np.abs(d[finite]), clip_q)) if finite.any() else 1.0
        else:
            clip = float(clip_abs)
        clip = max(clip, 1e-6)
        x = np.clip(d / clip, -1.0, 1.0)
        pos = np.clip(x, 0.0, 1.0)
        neg = np.clip(-x, 0.0, 1.0)
        rgb = np.ones((d.shape[0], d.shape[1], 3), dtype=np.float32)
        rgb[..., 0] = 1.0 - neg
        rgb[..., 1] = 1.0 - np.maximum(pos, neg)
        rgb[..., 2] = 1.0 - pos
        arr = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    @staticmethod
    def _make_signed_reference_bar(width: int, height: int) -> Image.Image:
        width = max(8, int(width))
        height = max(8, int(height))
        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
        x = np.repeat(x, height, axis=0)
        pos = np.clip(x, 0.0, 1.0)
        neg = np.clip(-x, 0.0, 1.0)
        rgb = np.ones((height, width, 3), dtype=np.float32)
        rgb[..., 0] = 1.0 - neg
        rgb[..., 1] = 1.0 - np.maximum(pos, neg)
        rgb[..., 2] = 1.0 - pos
        arr = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    @torch.no_grad()
    def _make_pub_collage(self, orig01: torch.Tensor, base01: torch.Tensor, both01: torch.Tensor, lat01: torch.Tensor, skip01: torch.Tensor, valid_mask01: torch.Tensor) -> Tuple[Image.Image, Dict[str, object]]:
        orig = self._crop_to_valid_single(orig01.detach().clamp(0, 1).cpu(), valid_mask01)
        base = self._crop_to_valid_single(base01.detach().clamp(0, 1).cpu(), valid_mask01)
        both = self._crop_to_valid_single(both01.detach().clamp(0, 1).cpu(), valid_mask01)
        lat = self._crop_to_valid_single(lat01.detach().clamp(0, 1).cpu(), valid_mask01)
        skip = self._crop_to_valid_single(skip01.detach().clamp(0, 1).cpu(), valid_mask01)

        target_hw = tuple(int(v) for v in orig.shape[-2:])
        base = self._resize_tensor_hw(base, target_hw)
        both = self._resize_tensor_hw(both, target_hw)
        lat = self._resize_tensor_hw(lat, target_hw)
        skip = self._resize_tensor_hw(skip, target_hw)

        diff_map = self._signed_map_2d(both - orig)
        skip_map = self._signed_map_2d(skip - orig)
        lat_map = self._signed_map_2d(lat - orig)

        signed_abs_max = max(
            float(diff_map.detach().abs().max().item()),
            float(skip_map.detach().abs().max().item()),
            float(lat_map.detach().abs().max().item()),
            1e-6,
        )

        panels = [
            ("Original Image", self._to_pil_01(orig)),
            ("AE Reconstruction", self._to_pil_01(base)),
            ("Watermarked Image", self._to_pil_01(both)),
            ("Diff (WM - Original)", self._to_pil_signed_seismic(diff_map, clip_abs=signed_abs_max)),
            ("Skip64 Contribution", self._to_pil_signed_seismic(skip_map, clip_abs=signed_abs_max)),
            ("Latent Contribution", self._to_pil_signed_seismic(lat_map, clip_abs=signed_abs_max)),
        ]

        W = max(im.size[0] for _, im in panels)
        H = max(im.size[1] for _, im in panels)
        gap = 8
        label_h = 20
        legend_h = 42
        cols = 3
        rows = 2
        canvas_w = gap + cols * (W + gap)
        canvas_h = gap + rows * (H + label_h + gap) + legend_h
        grid = Image.new("RGB", (canvas_w, canvas_h), (12, 12, 12))
        draw = ImageDraw.Draw(grid)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        for idx, (label, im) in enumerate(panels):
            r = idx // cols
            c = idx % cols
            x = gap + c * (W + gap)
            y = gap + r * (H + label_h + gap)
            if im.size != (W, H):
                im = im.resize((W, H))
            if im.mode != "RGB":
                im = im.convert("RGB")
            draw.rectangle((x, y, x + W, y + label_h), fill=(24, 24, 24))
            draw.text((x + 5, y + 4), label, fill=(255, 255, 255), font=font)
            grid.paste(im, (x, y + label_h))

        # Shared signed reference bar for Diff / Skip / Latent panels
        bar_total_w = cols * W + (cols - 1) * gap
        bar = self._make_signed_reference_bar(width=bar_total_w, height=12)
        bar_x = gap
        bar_y = gap + rows * (H + label_h + gap)
        grid.paste(bar, (bar_x, bar_y + 14))
        draw.text((bar_x, bar_y), "Shared signed scale for Diff / Skip / Latent", fill=(235, 235, 235), font=font)
        draw.text((bar_x, bar_y + 28), f"negative perturbation", fill=(90, 170, 255), font=font)
        mid_x = bar_x + (bar_total_w // 2)
        draw.text((mid_x - 8, bar_y + 28), "0", fill=(240, 240, 240), font=font)
        pos_txt = f"positive perturbation   max|Δ|={signed_abs_max:.6f}"
        tx = max(bar_x + 4, bar_x + bar_total_w - 7 * len(pos_txt))
        draw.text((tx, bar_y + 28), pos_txt, fill=(255, 120, 120), font=font)

        meta = {
            "signed_scale_mode": "shared_per_collage",
            "signed_abs_max": float(signed_abs_max),
            "signed_color_convention": "blue_negative_red_positive",
            "legend_enabled": True,
        }
        return grid, meta

    @torch.no_grad()
    def save_pub_collage(
        self,
        epoch: int,
        cls_name: str,
        src_path: str,
        orig01: torch.Tensor,
        base01: torch.Tensor,
        both01: torch.Tensor,
        lat01: torch.Tensor,
        skip01: torch.Tensor,
        valid_mask01: torch.Tensor,
        root_name: str = "collages_pub",
    ) -> Optional[Path]:
        cls_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(cls_name)).strip("_") or "class"
        stem = Path(src_path).stem
        out_dir = self.art_root / root_name / f"epoch_{epoch:03d}" / cls_safe
        ensure_dir(out_dir)
        ts = self._timestamp_token()
        out_path = out_dir / f"{stem}__{ts}.png"
        grid, collage_meta = self._make_pub_collage(orig01, base01, both01, lat01, skip01, valid_mask01)
        grid.save(out_path, format="PNG", compress_level=3)
        self._record_export_event(
            kind="collage",
            epoch=epoch,
            root_name=root_name,
            cls_name=cls_safe,
            src_path=str(src_path),
            out_path=out_path,
            extra=collage_meta,
        )
        return out_path

    @staticmethod
    def _to_pil_01(x01: torch.Tensor) -> Image.Image:
        arr = (x01.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def _to_pil_gray01(x01_1c: torch.Tensor) -> Image.Image:
        a = (x01_1c.clamp(0, 1).squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(a, mode="L")

    @torch.no_grad()
    def save_collage_8(
            self,
            epoch: int,
            it: int,
            orig01: torch.Tensor,
            base01: torch.Tensor,
            both01: torch.Tensor,
            lat01: torch.Tensor,
            skip01: torch.Tensor,
            zpat_vis01: torch.Tensor,
            w64_vis01: torch.Tensor,
            seismic01: torch.Tensor,
            out_path: Path,
    ) -> None:
        tiles = [
            ("orig", self._to_pil_01(orig01)),
            ("ae_base", self._to_pil_01(base01)),
            ("wm_both", self._to_pil_01(both01)),
            ("wm_lat", self._to_pil_01(lat01)),
            ("wm_skip", self._to_pil_01(skip01)),
            ("diff", self._to_pil_gray01(seismic01)),
            ("z_pat(RMS)", self._to_pil_gray01(zpat_vis01)),
            ("w64(RMS)", self._to_pil_gray01(w64_vis01)),
        ]
        W, H = tiles[0][1].size
        grid = Image.new("RGB", (4 * W, 2 * H), (0, 0, 0))
        draw = ImageDraw.Draw(grid)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        for idx, (name, im) in enumerate(tiles):
            r = idx // 4
            c = idx % 4
            if im.mode != "RGB":
                im = im.convert("RGB")
            grid.paste(im, (c * W, r * H))
            draw.text((c * W + 5, r * H + 5), name, fill=(255, 255, 255), font=font)

        grid.save(out_path)

    # ---------- checkpoints ----------

    def _realval_suite_meta(self) -> Dict[str, object]:
        modes = ["plain"]
        q_hi = int(getattr(self.cfg, "real_val_jpeg_quality", 0) or 0)
        q_lo = int(getattr(self.cfg, "real_val_jpeg_quality_lo", 0) or 0)
        rs = int(getattr(self.cfg, "real_val_resize_small", 0) or 0)
        if q_hi > 0:
            modes.append(f"jpeg{q_hi}")
        if q_lo > 0 and q_lo != q_hi:
            modes.append(f"jpeg{q_lo}")
        if rs > 0 and rs != int(getattr(self.cfg, "image_size", 160)):
            modes.append(f"resize{rs}")
        return {
            "modes": modes,
            "primary_mode_policy": "plain_if_present_else_first",
            "threshold_source": "raw_negatives",
            "scope_for_selection": "worst",
        }

    def _common_checkpoint_meta(self, epoch: int) -> Dict[str, object]:
        eval_eps = float(getattr(self.cfg, "eval_eps", 0.0) or 0.0)
        eval_r_skip = float(getattr(self.cfg, "eval_r_skip", -1.0) or -1.0)
        recommended_eval_eps = eval_eps if eval_eps > 0.0 else float(getattr(self.ctrl, "eps", 0.0))
        recommended_eval_r_skip = eval_r_skip if eval_r_skip >= 0.0 else float(getattr(self.ctrl, "r_skip", 0.0))
        return {
            "format_version": 3,
            "created_at_utc": utc_now_iso(),
            "run_started_utc": self._run_started_utc,
            "epoch": int(epoch),
            "trainer_filename": (self._trainer_file.name if self._trainer_file is not None else None),
            "trainer_path": (str(self._trainer_file) if self._trainer_file is not None else None),
            "trainer_sha256": self._trainer_sha256,
            "classes": list(self.classes),
            "class_to_idx": {c: i for i, c in enumerate(self.classes)},
            "num_classes": int(len(self.classes)),
            "preprocess_name": "PadToSquareNoUpscale",
            "image_size": int(self.cfg.image_size),
            "pad_value": float(self.cfg.pad_value),
            "input_range": [-1.0, 1.0],
            "padding_wipe_value": -1.0,
            "auto_switch": bool(getattr(self.cfg, "auto_switch", True)),
            "gray_like_eps": float(getattr(self.cfg, "gray_like_eps", 0.01) or 0.01),
            "force_gray_output": bool(getattr(self.cfg, "force_gray_output", True)),
            "carrier_mode": "variant2_raw_original",
            "embedding_space": "ae_latent_s64",
            "clean_reference": "raw_original_prod_wiped",
            "wm_positive_domain": "watermarked_prod_wiped",
            "threshold_source": "raw_negatives",
            "ae_module": str(self.cfg.ae_module),
            "ae_class": str(self.cfg.ae_class),
            "ae_ckpt_name": Path(self.cfg.ae_ckpt).name,
            "ae_ckpt_path": str(self.cfg.ae_ckpt),
            "ae_ckpt_sha256": self._ae_ckpt_sha256,
            "c1_ckpt_name": (Path(self.cfg.c1_ckpt).name if getattr(self.cfg, "c1_ckpt", None) else None),
            "c1_ckpt_path": (str(self.cfg.c1_ckpt) if getattr(self.cfg, "c1_ckpt", None) else None),
            "c1_ckpt_sha256": self._c1_ckpt_sha256,
            "model_name": "ResNet34LF_GN",
            "gate_strength": float(self.cfg.gate_strength),
            "gn_groups": int(self.cfg.gn_groups),
            "train_prod_mix_prob": float(getattr(self.cfg, "train_prod_mix_prob", 0.0) or 0.0),
            "thr_quantile": float(getattr(self.cfg, "thr_quantile", 0.995) or 0.995),
            "clean_consistency": bool(getattr(self.cfg, "clean_consistency", False)),
            "w_clean": float(getattr(self.cfg, "w_clean", 1.0) or 1.0),
            "c2_keygap_margin": float(getattr(self.cfg, "c2_keygap_margin", 0.0) or 0.0),
            "c2_keygap_w_suppress": float(getattr(self.cfg, "c2_keygap_w_suppress", 0.0) or 0.0),
            "c2_keygap_w_cap": float(getattr(self.cfg, "c2_keygap_w_cap", 0.0) or 0.0),
            "c2_ng_invar_w": float(getattr(self.cfg, "c2_ng_invar_w", 0.0) or 0.0),
            "c2_ng_kl_temp": float(getattr(self.cfg, "c2_ng_kl_temp", 1.0) or 1.0),
            "c2_gate_specific_w": float(getattr(self.cfg, "c2_gate_specific_w", 0.0) or 0.0),
            "c2_gate_specific_margin": float(getattr(self.cfg, "c2_gate_specific_margin", 0.0) or 0.0),
            "c2_clean_ng_keep_w": float(getattr(self.cfg, "c2_clean_ng_keep_w", 0.0) or 0.0),
            "c2_clean_ng_margin_floor": float(getattr(self.cfg, "c2_clean_ng_margin_floor", 0.0) or 0.0),
            "g_push_start_epoch": int(getattr(self.cfg, "g_push_start_epoch", 1) or 1),
            "g_push_margin_target": float(getattr(self.cfg, "g_push_margin_target", 0.25) or 0.25),
            "g_push_cls_lambda": float(getattr(self.cfg, "g_push_cls_lambda", 0.35) or 0.35),
            "g_push_det_lambda": float(getattr(self.cfg, "g_push_det_lambda", 0.15) or 0.15),
            "g_push_margin_lambda": float(getattr(self.cfg, "g_push_margin_lambda", 0.20) or 0.20),
            "controller_state": {
                "eps": float(self.ctrl.eps),
                "r_skip": float(self.ctrl.r_skip),
                "ema_delta": float(self.ctrl.ema_delta),
                "ema_wm_gap": float(getattr(self.ctrl, "ema_wm_gap", 0.0)),
                "ema_acc_gap_g": float(getattr(self.ctrl, "ema_acc_gap_g", 0.0)),
                "ema_gate_spec_gap": float(getattr(self.ctrl, "ema_gate_spec_gap", 0.0)),
                "ema_margin_gate_spec": float(getattr(self.ctrl, "ema_margin_gate_spec", 0.0)),
            },
            "diagnostic_contract": {
                "primary_realval_mode": "plain_when_available",
                "worst_mode_gap_name": "mode with minimum gated gap(acc_both-acc_raw)",
                "worst_mode_gatespec_name": "mode with minimum gate_specific_gap",
                "batch_log_fields": ["ctrlGap", "ctrlAcc", "ctrlGS", "Δm(g/ng)", "mGS"],
            },
            "recommended_eval_eps": float(recommended_eval_eps),
            "recommended_eval_r_skip": float(recommended_eval_r_skip),
            "realval_suite": self._realval_suite_meta(),
            "selection_rule": {
                "metric": "worst_gap_both_raw + worst_gate_specific_gap",
                "acc_both_floor": float(getattr(self.cfg, "best_realval_acc_both_floor", 0.70) or 0.70),
                "det_acc_floor": float(getattr(self.cfg, "best_realval_det_acc_floor", 0.85) or 0.85),
                "max_border_ratio": float(getattr(self.cfg, "best_realval_max_border_ratio", 0.10) or 0.10),
                "gap_floor": float(getattr(self.cfg, "best_realval_gap_floor", 0.10) or 0.10),
                "gate_specific_floor": float(getattr(self.cfg, "best_realval_gate_specific_floor", 0.05) or 0.05),
            },
            "cfg_snapshot": json_safe(vars(self.cfg)),
        }

    def _selection_record_from_realval(
        self,
        epoch: int,
        p_sys: Path,
        p_c2: Path,
        p_g: Path,
        val_stats: Optional[Dict[str, object]],
        real_stats: Optional[Dict[str, object]],
    ) -> Optional[Dict[str, object]]:
        if not real_stats:
            return None

        def _f(d: Dict[str, object], key: str, default: float) -> float:
            try:
                v = float(d.get(key, default))
            except Exception:
                v = float(default)
            return v if math.isfinite(v) else float(default)

        worst_gap = _f(real_stats, "worst_gap_both_raw", _f(real_stats, "gap_both_raw", float("-inf")))
        worst_gap_ng = _f(real_stats, "worst_gap_both_raw_ng", _f(real_stats, "gap_both_raw_ng", float("-inf")))
        worst_gspec = _f(real_stats, "worst_gate_specific_gap", worst_gap - worst_gap_ng)
        worst_acc_both = _f(real_stats, "worst_acc_both", _f(real_stats, "acc_both", float("-inf")))
        worst_det = _f(real_stats, "worst_det_acc_raw", _f(real_stats, "det_acc_raw", float("-inf")))
        worst_border = _f(real_stats, "worst_border_ratio", _f(real_stats, "border_ratio", float("inf")))
        acc_floor = float(getattr(self.cfg, "best_realval_acc_both_floor", 0.70) or 0.70)
        det_floor = float(getattr(self.cfg, "best_realval_det_acc_floor", 0.85) or 0.85)
        border_cap = float(getattr(self.cfg, "best_realval_max_border_ratio", 0.10) or 0.10)
        gap_floor = float(getattr(self.cfg, "best_realval_gap_floor", 0.10) or 0.10)
        gspec_floor = float(getattr(self.cfg, "best_realval_gate_specific_floor", 0.05) or 0.05)
        passes = (worst_acc_both >= acc_floor) and (worst_det >= det_floor) and (worst_border <= border_cap) and (worst_gap >= gap_floor) and (worst_gspec >= gspec_floor)

        score = [
            int(bool(passes)),
            float(worst_gap),
            float(worst_gspec),
            float(worst_acc_both),
            float(worst_det),
            float(-worst_border),
        ]
        reason = (
            f"Selected by REALVAL worst gap(acc_both-acc_raw) and worst gate_specific_gap; "
            f"floors: acc_both>={acc_floor:.3f}, det_acc_raw>={det_floor:.3f}, border_ratio<={border_cap:.3f}, gap>={gap_floor:.3f}, gate_specific>={gspec_floor:.3f}."
        )
        return {
            "epoch": int(epoch),
            "passes_floors": bool(passes),
            "score": score,
            "reason": reason,
            "wm_system_ckpt": str(p_sys.name),
            "c2_eval_ckpt": str(p_c2.name),
            "opt_g_ckpt": str(p_g.name),
            "primary_mode": real_stats.get("primary_mode"),
            "worst_mode_gap_name": real_stats.get("worst_mode_gap_name"),
            "worst_mode_gatespec_name": real_stats.get("worst_mode_gatespec_name"),
            "worst_gap_both_raw": float(worst_gap),
            "worst_gap_both_raw_ng": float(worst_gap_ng),
            "worst_gate_specific_gap": float(worst_gspec),
            "worst_acc_both": float(worst_acc_both),
            "worst_det_acc_raw": float(worst_det),
            "worst_border_ratio": float(worst_border),
            "val_stats": json_safe(val_stats),
            "realval_stats": json_safe(real_stats),
        }

    def _maybe_update_best_checkpoint_summary(
        self,
        epoch: int,
        p_sys: Path,
        p_c2: Path,
        p_g: Path,
        val_stats: Optional[Dict[str, object]],
        real_stats: Optional[Dict[str, object]],
    ) -> None:
        rec = self._selection_record_from_realval(epoch, p_sys, p_c2, p_g, val_stats, real_stats)
        if rec is None:
            return

        if bool(getattr(self.cfg, "c2_best_checkpoint_require_pass_floors", True)) and (not bool(rec.get("passes_floors", False))):
            return

        cur = self._best_realval_record
        better = cur is None or tuple(rec["score"]) > tuple(cur["score"])
        if not better:
            return

        self._best_realval_record = rec
        write_json(self.ckpt_root / "best_checkpoint_summary.json", rec)
        print(
            f"[BEST CKPT] epoch={epoch:03d} gap={rec['worst_gap_both_raw'] * 100:.2f}pp "
            f"gap_ng={rec.get('worst_gap_both_raw_ng', float('nan')) * 100:.2f}pp "
            f"gs={rec.get('worst_gate_specific_gap', float('nan')) * 100:.2f}pp "
            f"acc_both={rec['worst_acc_both'] * 100:.2f}% det={rec['worst_det_acc_raw'] * 100:.2f}% "
            f"border={rec['worst_border_ratio']:.3f}",
            flush=True,
        )

    def save_checkpoint(
        self,
        epoch: int,
        val_stats: Optional[Dict[str, object]] = None,
        real_stats: Optional[Dict[str, object]] = None,
    ) -> None:
        export_snapshot = self._flush_epoch_export_manifests(epoch)
        common_meta = self._common_checkpoint_meta(epoch)
        common_meta["epoch_exports"] = export_snapshot
        common_meta["signed_legend_enabled"] = True
        common_meta["signed_scale_mode"] = "shared_per_collage"
        common_meta["signed_color_convention"] = "blue_negative_red_positive"

        ckpt_sys = {
            "format_version": 3,
            "kind": "wm_system",
            "epoch": int(epoch),
            "meta": {
                **common_meta,
                "checkpoint_role": "wm_system",
            },
            "cfg": vars(self.cfg),
            "ctrl": {
                "eps": float(self.ctrl.eps),
                "r_skip": float(self.ctrl.r_skip),
                "ema_delta": float(self.ctrl.ema_delta),
                "ema_wm_gap": float(getattr(self.ctrl, "ema_wm_gap", 0.0)),
                "ema_acc_gap_g": float(getattr(self.ctrl, "ema_acc_gap_g", 0.0)),
                "ema_gate_spec_gap": float(getattr(self.ctrl, "ema_gate_spec_gap", 0.0)),
                "ema_margin_gate_spec": float(getattr(self.ctrl, "ema_margin_gate_spec", 0.0)),
            },
            "mask_lat": unwrap(self.mask_lat).state_dict(),
            "mask_64": unwrap(self.mask_64).state_dict(),
            "g_lat": unwrap(self.g_lat).state_dict(),
            "g_64": unwrap(self.g_64).state_dict(),
            "freq_ctrl": unwrap(self.freq_ctrl).state_dict(),
            "last_val": val_stats,
            "last_realval": real_stats,
        }
        ckpt_c2 = {
            "format_version": 3,
            "kind": "c2_eval",
            "epoch": int(epoch),
            "meta": {
                **common_meta,
                "checkpoint_role": "c2_eval",
                "use_ema_for_eval": True,
                "recommended_model_key": "c2_ema",
                "recommended_selection_metric": "REALVAL worst gated gap(acc_both-acc_raw) plus gate_specific_gap, with floors on acc_both/det_acc_raw/border/gap/gate_specific",
            },
            "classes": list(self.classes),
            "class_to_idx": {c: i for i, c in enumerate(self.classes)},
            "num_classes": int(len(self.classes)),
            "image_size": int(self.cfg.image_size),
            "pad_value": float(self.cfg.pad_value),
            "preprocess_name": "PadToSquareNoUpscale",
            "input_range": "[-1,1]",
            "padding_wipe_value": -1.0,
            "model_name": "ResNet34LF_GN",
            "gate_strength": float(self.cfg.gate_strength),
            "gn_groups": int(self.cfg.gn_groups),
            "carrier_mode": "variant2_raw_original",
            "clean_negative_domain": "raw_original_prod_wiped",
            "wm_positive_domain": "watermarked_prod_wiped",
            "threshold_calibration": "quantile_on_raw_negatives",
            "c2": unwrap(self.c2).state_dict(),
            "c2_ema": unwrap(self.c2_ema).state_dict(),
            "opt_c2": self.opt_c2.state_dict(),
            "last_val": val_stats,
            "last_realval": real_stats,
        }
        ckpt_g = {
            "format_version": 3,
            "kind": "opt_g",
            "epoch": int(epoch),
            "meta": {
                **common_meta,
                "checkpoint_role": "opt_g",
            },
            "opt_g": self.opt_g.state_dict(),
        }

        p_sys = self.ckpt_root / f"wm_system_e{epoch:03d}.pth"
        p_c2 = self.ckpt_root / f"c2_eval_e{epoch:03d}.pth"
        p_g = self.ckpt_root / f"opt_g_e{epoch:03d}.pth"

        torch.save(ckpt_sys, p_sys)
        torch.save(ckpt_c2, p_c2)
        torch.save(ckpt_g, p_g)

        meta_root = self.ckpt_root / "meta"
        ensure_dir(meta_root)
        write_json(meta_root / f"wm_system_e{epoch:03d}.meta.json", {"checkpoint": p_sys.name, "meta": ckpt_sys["meta"], "last_val": val_stats, "last_realval": real_stats})
        write_json(meta_root / f"c2_eval_e{epoch:03d}.meta.json", {"checkpoint": p_c2.name, "meta": ckpt_c2["meta"], "last_val": val_stats, "last_realval": real_stats})
        write_json(meta_root / f"opt_g_e{epoch:03d}.meta.json", {"checkpoint": p_g.name, "meta": ckpt_g["meta"]})
        if val_stats is not None:
            write_json(meta_root / f"val_e{epoch:03d}.json", {"epoch": int(epoch), "val": val_stats})
        if real_stats is not None:
            write_json(meta_root / f"realval_e{epoch:03d}.json", {"epoch": int(epoch), "realval": real_stats})

        self._maybe_update_best_checkpoint_summary(epoch, p_sys, p_c2, p_g, val_stats, real_stats)

        print(f"[CKPT] system -> {p_sys}")
        print(f"[CKPT] c2_eval -> {p_c2}")
        print(f"[CKPT] opt_g -> {p_g}")

    # ---------- main train loop ----------

    
    def _set_c2_detwarm_mode(self, epoch: int) -> None:
        """Optionally freeze C2 backbone/fc and train watermark detector head only.
    
        IMPORTANT SAFETY FIX:
        det-warmup is only sensible when C2 already has a meaningful backbone
        (warm-start from C1 or resume from a prior C2 checkpoint). If C1 failed to
        load and there is no --c2_eval_ckpt, then C2 is random-initialized; freezing
        the backbone in epoch 1 trains only the classifier head on random features and
        validation collapses into a few classes.
        """
        cfg = self.cfg
        det_epochs = int(getattr(cfg, "det_warmup_epochs", 0) or 0)
        on = (det_epochs > 0 and epoch <= det_epochs)

        # Detwarm is safe only when C2 is not random-init.
        safe_detwarm = (
            (self.c1 is not None and bool(getattr(cfg, "c2_init_from_c1", True)))
            or bool(getattr(cfg, "c2_eval_ckpt", None))
        )
        if on and not safe_detwarm:
            if not getattr(self, "_detwarm_disabled_warned", False):
                print(
                    "[C2 DETWARM] disabled: no C1 warm-start and no --c2_eval_ckpt; "
                    "random-init C2 must train full from epoch 1"
                )
                self._detwarm_disabled_warned = True
            on = False
    
        if getattr(self, "_detwarm_on", None) == on:
            return
        self._detwarm_on = on
    
        m = unwrap(self.c2)
        if on:
            for p in m.parameters():
                p.requires_grad_(False)
            # Warm detector, gate affine, classifier head, and (optionally) the last feature block.
            # A fully frozen backbone tends to keep wm_head at ~0.5/0.5.
            if bool(getattr(cfg, "det_warm_unfreeze_layer4", True)):
                for p in m.base.layer4.parameters():
                    p.requires_grad_(True)
            for p in m.wm_head.parameters():
                p.requires_grad_(True)
            for p in m.base.fc.parameters():
                p.requires_grad_(True)
            m.wm_affine.requires_grad_(True)
            w_det = float(getattr(cfg, "det_warmup_w_det", 6.0) or 6.0)
            kf = float(getattr(cfg, "det_warmup_k_factor", 1.75) or 1.75)
            if bool(getattr(cfg, "det_warm_unfreeze_layer4", True)):
                print(f"[C2 DETWARM] epoch {epoch}: layer4 + base.fc + wm_head + wm_affine trainable (W_det={w_det:g}, k_factor={kf:g})")
            else:
                print(f"[C2 DETWARM] epoch {epoch}: backbone frozen, training base.fc + wm_head + wm_affine (W_det={w_det:g}, k_factor={kf:g})")
        else:
            for p in m.parameters():
                p.requires_grad_(True)
            print(f"[C2 DETWARM] epoch {epoch}: full C2 training")
    
    def train(self) -> None:
            it = 0
            for epoch in range(1, self.cfg.epochs + 1):
                self._set_c2_detwarm_mode(epoch)
                self.mask_lat.train()
                self.mask_64.train()
                self.g_lat.train()
                self.g_64.train()
                self.freq_ctrl.train()
                self.c2.train()

                # FIX-3: C2 trains on previous batch (1-step lag).
                # Generator output changes each step — training C2 on the same
                # batch causes co-adaptation instability and split collapse.
                _prev_c2_batch = None
                for batch in self.train_loader:
                    it += 1
                    gstats = self.step_generator(batch, epoch, it)
                    # Use previous batch for C2; first step falls back to current batch.
                    c2_batch = _prev_c2_batch if _prev_c2_batch is not None else batch
                    c2stats = self.step_c2(c2_batch, epoch)
                    _prev_c2_batch = batch
                    self.controller_update(gstats, c2stats, epoch=epoch)

                    if (it % max(1, self.cfg.print_every) == 0):
                        val_probe_stats = dict(getattr(self, "_last_val_probe_stats", {
                            "acc_raw_g": float("nan"), "acc_both_g": float("nan"), "gap_g": float("nan"),
                            "acc_raw_ng": float("nan"), "acc_both_ng": float("nan"), "gap_ng": float("nan"),
                            "gate_specific_gap": float("nan"),
                        }))
                        probe_every = int(getattr(self.cfg, 'val_probe_every', 0) or 0)
                        if probe_every > 0 and ((it == 1) or (it % probe_every == 0)):
                            val_probe_stats = self.val_probe(epoch)
                            self._last_val_probe_stats = dict(val_probe_stats)

                        print(
                            f"[E{epoch:02d} it {it:06d}] eps={self.ctrl.eps:.4f} r_skip={self.ctrl.r_skip:.2f} | "
                            f"SSIM(wm/x)={gstats.get('SSIM', 0.0):.4f} SSIM(ae/x)={gstats.get('SSIM_y_ae', 0.0):.4f} | "
                            f"PSNRy(wm/x)={gstats.get('PSNR', 0.0):.2f} PSNRy(ae/x)={gstats.get('PSNR_ae', 0.0):.2f} | "
                            f"MAE(wm/x)={gstats.get('MAE', 0.0):.5f} sat_hi(x/ae/wm)={gstats.get('sat_x_hi', 0.0):.3f}/{gstats.get('sat_base_hi', 0.0):.3f}/{gstats.get('sat_wm_hi', 0.0):.3f} | "
                            f"leak={gstats.get('L_leak', 0.0):.4f} "
                            f"pad01={gstats.get('pad_mean_x01', 0.0):.3f} padN={gstats.get('pad_mean_xN', 0.0):.3f} vFrac={gstats.get('valid_frac', 0.0):.3f} "
                            f"bAbs={gstats.get('border_abs', 0.0):.4f} bR={gstats.get('border_ratio', 0.0):.3f} "
                            f"dAbs={gstats.get('delta_abs', 0.0):.4f} dMax={gstats.get('delta_max', 0.0):.4f} dMu={gstats.get('delta_mean', 0.0):+.4f} rms={gstats.get('delta_rms_pre', 0.0):.4f}->{gstats.get('delta_rms_post', 0.0):.4f} rn={gstats.get('delta_rn', 0.0):.2f} dDark={gstats.get('delta_dark_abs', 0.0):.4f} "
                            f"band(l/m)={gstats.get('band_low', 0.0):.3f}/{gstats.get('band_mid', 0.0):.3f} E(l/m/h)={gstats.get('band_e_low', 0.0):.4f}/{gstats.get('band_e_mid', 0.0):.4f}/{gstats.get('band_e_high', 0.0):.4f} cmF={gstats.get('cm_frac', 1.0):.3f} Lmin={gstats.get('L_min', 0.0):.4f} aBoost={gstats.get('alpha_boost', 1.0):.2f} "
                            f"ROI(lat/64/u)={gstats.get('P_lat_mean', 0.0):.3f}/{gstats.get('P_64_mean', 0.0):.3f}/{gstats.get('roi_union_mean', 0.0):.3f} "
                            f"pLat={gstats.get('p_lat', 0.0):.3f} clipHit={gstats.get('clip_hit', 0.0):.3f} LF={gstats.get('lowfreq_ratio', 0.0):.3f} "
                            f"Qroi(P,S,M)={gstats.get('PSNR_roi', 0.0):.1f},{gstats.get('SSIM_roi', 0.0):.3f},{gstats.get('MAE_roi', 0.0):.4f} "
                            f"Qbg(P,S,M)={gstats.get('PSNR_bg', 0.0):.1f},{gstats.get('SSIM_bg', 0.0):.3f},{gstats.get('MAE_bg', 0.0):.4f} "
                            f"chroma={gstats.get('L_chroma', 0.0):.4f} obj={gstats.get('L_obj', 0.0):.4f} grid={gstats.get('L_grid', 0.0):.4f} push(c/d/m)={gstats.get('L_push_cls', 0.0):.3f}/{gstats.get('L_push_det', 0.0):.3f}/{gstats.get('L_push_margin', 0.0):.3f} | "
                            f"C2 clean(g/ng)={c2stats.get('c2_acc_clean_g', c2stats.get('c2_acc_clean', 0.0)):.3f}/{c2stats.get('c2_acc_clean_ng', 0.0):.3f} "
                            f"wm(g/ng)={c2stats.get('c2_acc_wm_g', c2stats.get('c2_acc_wm', 0.0)):.3f}/{c2stats.get('c2_acc_wm_ng', 0.0):.3f} "
                            f"Δacc(g/ng)={c2stats.get('c2_acc_delta_g', c2stats.get('delta', 0.0)):+.3f}/{c2stats.get('c2_acc_delta_ng', 0.0):+.3f} "
                            f"gateSpec={c2stats.get('gate_specific_gap', c2stats.get('c2_acc_delta_g', c2stats.get('delta', 0.0)) - c2stats.get('c2_acc_delta_ng', 0.0)):+.3f} "
                            f"m(c/w)={c2stats.get('c2_margin_clean', 0.0):+.3f}/{c2stats.get('c2_margin_wm', 0.0):+.3f} "
                            f"Δm(g/ng)={c2stats.get('delta_margin', 0.0):+.3f}/{c2stats.get('c2_margin_gap_ng', 0.0):+.3f} mGS={c2stats.get('c2_margin_gate_specific', 0.0):+.3f} "
                            f"wmP(c/w)={c2stats.get('wm_prob_clean', 0.0):.3f}/{c2stats.get('wm_prob_wm', 0.0):.3f} "
                            f"Δp={c2stats.get('wm_prob_gap', c2stats.get('wm_prob_wm', 0.0) - c2stats.get('wm_prob_clean', 0.0)):+.3f} "
                            f"gate(c/w)={c2stats.get('wm_logit_clean_mean', float('nan')):+.3f}/{c2stats.get('wm_logit_wm_mean', float('nan')):+.3f} Lgc={c2stats.get('L_gate_close', float('nan')):.4f} "
                            f"Ltrans={gstats.get('L_transfer', 0.0):.4f} Ldiv={gstats.get('L_diversity', 0.0):.4f} Lssim={gstats.get('L_ssim', 0.0):.4f} "
                            f"ctrlGap={getattr(self.ctrl, 'ema_wm_gap', 0.0):+.3f} ctrlAcc={getattr(self.ctrl, 'ema_acc_gap_g', 0.0):+.3f} ctrlGS={getattr(self.ctrl, 'ema_gate_spec_gap', 0.0):+.3f} | "
                            f"C1 clean={gstats.get('c1_acc_clean', float('nan')):.3f} wm={gstats.get('c1_acc_wm', float('nan')):.3f} Δacc={gstats.get('c1_acc_delta', float('nan')):+.3f} "
                            f"CE={gstats.get('c1_ce_clean', float('nan')):.3f}/{gstats.get('c1_ce_wm', float('nan')):.3f} ΔCE={gstats.get('c1_ce_delta', float('nan')):+.3f} Lc1={gstats.get('L_c1_guard', 0.0):.4f} | "
                            f"VALprobe g(raw/wm/gap)={val_probe_stats.get('acc_raw_g', float('nan')):.3f}/{val_probe_stats.get('acc_both_g', float('nan')):.3f}/{val_probe_stats.get('gap_g', float('nan')):+.3f} "
                            f"ng(raw/wm/gap)={val_probe_stats.get('acc_raw_ng', float('nan')):.3f}/{val_probe_stats.get('acc_both_ng', float('nan')):.3f}/{val_probe_stats.get('gap_ng', float('nan')):+.3f} "
                            f"gs={val_probe_stats.get('gate_specific_gap', float('nan')):+.3f}"
                        )

                val_stats = None
                real_stats = None

                if self.cfg.val_every > 0 and (epoch % self.cfg.val_every == 0):
                    print(f"[E{epoch:02d}] starting validate()", flush=True)
                    val_stats = self.validate(epoch)

                if int(getattr(self.cfg, 'real_val_every', 0)) > 0 and (epoch % int(self.cfg.real_val_every) == 0):
                    print(f"[E{epoch:02d}] starting validate_real_life()", flush=True)
                    real_stats = self.validate_real_life(epoch)

                print(
                    f"[SPLIT E{epoch:02d}][TRAIN] C2Δg={c2stats.get('c2_acc_delta_g', c2stats.get('delta', 0.0)):+.3f} "
                    f"C2Δng={c2stats.get('c2_acc_delta_ng', 0.0):+.3f} "
                    f"GS={c2stats.get('gate_specific_gap', c2stats.get('c2_acc_delta_g', c2stats.get('delta', 0.0)) - c2stats.get('c2_acc_delta_ng', 0.0)):+.3f} "
                    f"mGS={c2stats.get('c2_margin_gate_specific', 0.0):+.3f} "
                    f"C1Δ={gstats.get('c1_acc_delta', float('nan')):+.3f} "
                    f"C1ΔCE={gstats.get('c1_ce_delta', float('nan')):+.3f} ctrlGap={getattr(self.ctrl, 'ema_wm_gap', 0.0):+.3f} "
                    f"ctrlAcc={getattr(self.ctrl, 'ema_acc_gap_g', 0.0):+.3f} ctrlGS={getattr(self.ctrl, 'ema_gate_spec_gap', 0.0):+.3f}",
                    flush=True,
                )
                if val_stats is not None:
                    print(
                        f"[VAL E{epoch:02d}] RAW={val_stats.get('acc_raw', float('nan')) * 100:.2f}% BASE={val_stats.get('acc_base', float('nan')) * 100:.2f}% "
                        f"BOTH={val_stats.get('acc_both', float('nan')) * 100:.2f}% GAPg={val_stats.get('gap_both_raw', float('nan')) * 100:+.2f}pp "
                        f"GAPng={val_stats.get('gap_both_raw_ng', float('nan')) * 100:+.2f}pp GS={val_stats.get('gate_specific_gap', float('nan')) * 100:+.2f}pp "
                        f"det_raw={val_stats.get('det_acc_raw', float('nan')) * 100:.2f}% det_base={val_stats.get('det_acc_base', float('nan')) * 100:.2f}%",
                        flush=True,
                    )
                if real_stats is not None:
                    plain = ((real_stats.get('mode_results', {}) or {}).get('plain', {}) if isinstance(real_stats, dict) else {})
                    worst_gap_name = real_stats.get('worst_mode_gap_name') if isinstance(real_stats, dict) else None
                    worst_gs_name = real_stats.get('worst_mode_gatespec_name') if isinstance(real_stats, dict) else None
                    worst_gap = ((real_stats.get('mode_results', {}) or {}).get(worst_gap_name, {}) if (isinstance(real_stats, dict) and worst_gap_name) else {})
                    worst_gs = ((real_stats.get('mode_results', {}) or {}).get(worst_gs_name, {}) if (isinstance(real_stats, dict) and worst_gs_name) else {})
                    if plain:
                        print(
                            f"[REALVAL E{epoch:02d}][plain] RAW={plain.get('acc_raw', float('nan')) * 100:.2f}% BOTH={plain.get('acc_both', float('nan')) * 100:.2f}% "
                            f"GAPg={plain.get('acc_gap_both_raw', float('nan')) * 100:+.2f}pp GAPng={plain.get('acc_gap_both_raw_ng', float('nan')) * 100:+.2f}pp GS={plain.get('gate_specific_gap', float('nan')) * 100:+.2f}pp det={plain.get('det_acc_raw', float('nan')) * 100:.2f}% "
                            f"border={plain.get('border_ratio', float('nan')):.3f} thr={plain.get('thr_raw', float('nan')):.3f}",
                            flush=True,
                        )
                    if worst_gap:
                        print(
                            f"[REALVAL E{epoch:02d}][worst-gap] mode={worst_gap_name} RAW={worst_gap.get('acc_raw', float('nan')) * 100:.2f}% BOTH={worst_gap.get('acc_both', float('nan')) * 100:.2f}% "
                            f"GAPg={worst_gap.get('acc_gap_both_raw', float('nan')) * 100:+.2f}pp GAPng={worst_gap.get('acc_gap_both_raw_ng', float('nan')) * 100:+.2f}pp GS={worst_gap.get('gate_specific_gap', float('nan')) * 100:+.2f}pp det={worst_gap.get('det_acc_raw', float('nan')) * 100:.2f}% "
                            f"border={worst_gap.get('border_ratio', float('nan')):.3f}",
                            flush=True,
                        )
                    if worst_gs and worst_gs_name != worst_gap_name:
                        print(
                            f"[REALVAL E{epoch:02d}][worst-gs] mode={worst_gs_name} RAW={worst_gs.get('acc_raw', float('nan')) * 100:.2f}% BOTH={worst_gs.get('acc_both', float('nan')) * 100:.2f}% "
                            f"GAPg={worst_gs.get('acc_gap_both_raw', float('nan')) * 100:+.2f}pp GAPng={worst_gs.get('acc_gap_both_raw_ng', float('nan')) * 100:+.2f}pp GS={worst_gs.get('gate_specific_gap', float('nan')) * 100:+.2f}pp det={worst_gs.get('det_acc_raw', float('nan')) * 100:.2f}% "
                            f"border={worst_gs.get('border_ratio', float('nan')):.3f}",
                            flush=True,
                        )
                    pass_floors = (
                        float(real_stats.get('worst_acc_both', float('-inf'))) >= float(getattr(self.cfg, 'best_realval_acc_both_floor', 0.70) or 0.70)
                        and float(real_stats.get('worst_det_acc_raw', float('-inf'))) >= float(getattr(self.cfg, 'best_realval_det_acc_floor', 0.85) or 0.85)
                        and float(real_stats.get('worst_border_ratio', float('inf'))) <= float(getattr(self.cfg, 'best_realval_max_border_ratio', 0.10) or 0.10)
                        and float(real_stats.get('worst_gap_both_raw', float('-inf'))) >= float(getattr(self.cfg, 'best_realval_gap_floor', 0.10) or 0.10)
                        and float(real_stats.get('worst_gate_specific_gap', float('-inf'))) >= float(getattr(self.cfg, 'best_realval_gate_specific_floor', 0.05) or 0.05)
                    )
                    print(
                        f"[SELECT E{epoch:02d}] pass_floors={'yes' if pass_floors else 'no'} | "
                        f"acc_both={float(real_stats.get('worst_acc_both', float('nan'))) * 100:.2f}% "
                        f"det={float(real_stats.get('worst_det_acc_raw', float('nan'))) * 100:.2f}% "
                        f"gap={float(real_stats.get('worst_gap_both_raw', float('nan'))) * 100:+.2f}pp "
                        f"gs={float(real_stats.get('worst_gate_specific_gap', float('nan'))) * 100:+.2f}pp "
                        f"border={float(real_stats.get('worst_border_ratio', float('nan'))):.3f}",
                        flush=True,
                    )

                self.save_checkpoint(epoch, val_stats=val_stats, real_stats=real_stats)

                if self._maybe_stop_from_real_val(real_stats, epoch):
                    break


# -------------------------
# CLI
# -------------------------


def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", type=str, default="train", choices=["train", "infer"])
    ap.add_argument("--profile", type=str, default="auto_bandmix", choices=["auto_bandmix", "legacy"], help="auto_bandmix = system chooses low/mid band per image with minimal manual tuning")
    ap.add_argument("--train_root", type=str, default="")
    ap.add_argument("--val_root", type=str, default="")
    ap.add_argument("--infer_root", type=str, default="")
    ap.add_argument("--infer_list", type=str, default="")
    ap.add_argument("--infer_out", type=str, default="")
    ap.add_argument("--system_ckpt", type=str, default="")
    ap.add_argument("--out_root", type=str, required=True)

    ap.add_argument("--ae_ckpt", type=str, required=True)
    ap.add_argument("--ae_module", type=str, required=True)
    ap.add_argument("--ae_class", type=str, default="UniversalAutoEncoder")
    ap.add_argument("--ae_py_path", type=str, required=True)

    # C1 guard rail (optional, frozen external classifier)
    ap.add_argument("--c1_ckpt", type=str, default="", help="Path to frozen C1 checkpoint (.pth). If empty, C1 guard is disabled.")
    ap.add_argument("--c1_guard_min_acc", type=float, default=0.0, help="Target minimum C1 accuracy on watermarked images (used as a controller brake).")
    ap.add_argument("--c1_guard_max_drop", type=float, default=0.05, help="Max allowed (C1 clean acc - C1 wm acc) per batch. 0 disables.")
    ap.add_argument("--delta_post_blur_mix", type=float, default=0.35, help="Blend factor for post-blur on deltaY to kill block artifacts (0..1).")
    ap.add_argument("--grid_boundary_lambda", type=float, default=0.20, help="Penalty weight for block-boundary discontinuities (0 disables).")

    ap.add_argument("--c1_guard_lambda", type=float, default=0.20, help="Weight of differentiable C1 guard penalty (CE degradation). 0 disables penalty.")
    ap.add_argument("--c1_guard_every", type=int, default=1, help="Compute C1 guard penalty every N iters (>=1).")
    ap.add_argument("--c1_guard_ce_margin", type=float, default=0.0, help="Allowed CE(wm)-CE(clean) margin for C1 before penalty.")
    ap.add_argument("--c1_brake_eps", type=float, default=0.005, help="How strongly controller reduces eps when C1(wm) acc < c1_guard_min_acc.")
    ap.add_argument("--c2_init_from_c1", type=int, default=1, choices=[0, 1], help="Warm-start C2 backbone+classifier from --c1_ckpt when available (wm_head/wm_affine stay fresh).")
    ap.add_argument("--det_warmup_epochs", type=int, default=2, help="Detector-focused warmup for first N epochs (0 disables).")
    ap.add_argument("--det_warmup_w_det", type=float, default=6.0, help="Detector BCE weight during warmup.")
    ap.add_argument("--det_warmup_k_factor", type=float, default=1.75, help="Watermark strength multiplier for C2 synthesis during warmup.")
    ap.add_argument("--det_post_w_det_early", type=float, default=2.50, help="Detector BCE weight for epochs 3..4. FIX-1: raised from 1.25.")
    ap.add_argument("--det_post_w_det_late", type=float, default=2.50, help="Detector BCE weight after early phase.")
    ap.add_argument("--det_post_w_sep_early", type=float, default=0.50, help="Separation weight for epochs 3..4.")
    ap.add_argument("--det_post_w_sep_late", type=float, default=1.10, help="Separation weight after early phase.")
    ap.add_argument("--det_warm_unfreeze_layer4", type=int, default=1, choices=[0, 1], help="Unfreeze C2 layer4 during detector warmup.")

    ap.add_argument("--clean_consistency", type=int, default=0, choices=[0, 1], help="Use Variant-B clean gated consistency instead of pushing clean-gated logits to uniform.")
    ap.add_argument("--clean_consistency_temp", type=float, default=1.0, help="Temperature for clean gated consistency KL.")
    ap.add_argument("--w_clean", type=float, default=1.20, help="Global multiplier for clean CE plus clean-gated sabotage/consistency.")
    ap.add_argument("--c2_keygap_margin", type=float, default=0.22, help="Probability margin below 1/K for suppressing the true class in fail-clean mode.")
    ap.add_argument("--c2_keygap_w_suppress", type=float, default=4.0, help="Weight for suppressing the true class in fail-clean mode.")
    ap.add_argument("--c2_keygap_w_cap", type=float, default=1.5, help="Weight for capping max clean-gated probability in fail-clean mode.")
    ap.add_argument("--c2_ng_invar_w", type=float, default=1.00, help="Weight for non-gated invariance KL between raw and watermarked inputs.")
    ap.add_argument("--c2_ng_kl_temp", type=float, default=1.0, help="Temperature for non-gated invariance KL.")
    ap.add_argument("--c2_gate_specific_w", type=float, default=1.50, help="Weight for encouraging gated gap to exceed non-gated gap.")
    ap.add_argument("--c2_gate_specific_margin", type=float, default=0.35, help="Desired margin by which gated gap should exceed non-gated gap.")
    ap.add_argument("--c2_clean_ng_keep_w", type=float, default=0.90, help="Weight for keeping clean non-gated margin strong.")
    ap.add_argument("--c2_clean_ng_margin_floor", type=float, default=1.25, help="Minimum desired clean non-gated class margin.")
    ap.add_argument("--c2_warm_ng_invar_mult", type=float, default=1.75, help="Multiplier on non-gated invariance weight during detector warmup.")
    ap.add_argument("--c2_warm_gate_specific_mult", type=float, default=2.50, help="Multiplier on gate-specific loss during detector warmup.")
    ap.add_argument("--c2_warm_clean_ng_keep_mult", type=float, default=2.25, help="Multiplier on clean non-gated keep loss during detector warmup.")
    ap.add_argument("--c2_early_ng_invar_mult", type=float, default=1.35, help="Multiplier on non-gated invariance weight during early post-warmup epochs.")
    ap.add_argument("--c2_early_gate_specific_mult", type=float, default=1.75, help="Multiplier on gate-specific loss during early post-warmup epochs.")
    ap.add_argument("--c2_early_clean_ng_keep_mult", type=float, default=1.60, help="Multiplier on clean non-gated keep loss during early post-warmup epochs.")
    ap.add_argument("--c2_late_ng_invar_mult", type=float, default=1.20, help="Multiplier on non-gated invariance weight after the early phase.")
    ap.add_argument("--c2_late_gate_specific_mult", type=float, default=1.60, help="Multiplier on gate-specific loss after the early phase.")
    ap.add_argument("--c2_late_clean_ng_keep_mult", type=float, default=1.40, help="Multiplier on clean non-gated keep loss after the early phase.")
    ap.add_argument("--c2_no_gap_ng_invar_mult", type=float, default=1.10, help="Extra multiplier on non-gated invariance when detector is alive but class/gate-specific gaps lag targets.")
    ap.add_argument("--c2_no_gap_gate_specific_mult", type=float, default=1.35, help="Extra multiplier on gate-specific loss when detector is alive but class/gate-specific gaps lag targets.")
    ap.add_argument("--c2_no_gap_clean_ng_keep_mult", type=float, default=1.20, help="Extra multiplier on clean non-gated keep loss when detector is alive but class/gate-specific gaps lag targets.")
    ap.add_argument("--g_push_start_epoch", type=int, default=1, help="Enable generator push losses from this epoch onward.")
    ap.add_argument("--g_push_margin_target", type=float, default=0.25, help="Desired margin advantage of WM over RAW in generator push mode.")
    ap.add_argument("--g_push_cls_lambda",    type=float, default=0.35, help="Weight for generator push CE on WM images.")
    ap.add_argument("--transfer_lam",         type=float, default=0.50, help="Weight for L_transfer (KNN/transfer attack resistance).")
    ap.add_argument("--diversity_lam",        type=float, default=0.10, help="Weight for L_diversity (WM pattern uniqueness per image).")
    ap.add_argument("--ssim_lam",             type=float, default=0.60, help="Weight for L_ssim perceptual quality loss.")
    ap.add_argument("--transfer_start_epoch", type=int,   default=2,    help="Epoch to start transfer-attack simulation (split must be stable).")
    ap.add_argument("--g_push_det_lambda", type=float, default=0.20, help="Weight for generator push detector BCE on WM images.")
    ap.add_argument("--g_push_margin_lambda", type=float, default=0.30, help="Weight for generator push margin objective.")
    ap.add_argument("--ctrl_init_eps", type=float, default=0.10, help="Initial controller eps for train mode when no system checkpoint is loaded.")
    ap.add_argument("--ctrl_init_r_skip", type=float, default=0.66, help="Initial controller r_skip for train mode when no system checkpoint is loaded.")
    ap.add_argument("--ctrl_dmargin_target", type=float, default=0.06, help="Target EMA delta-margin for the eps controller. Lower keeps eps from running away.")
    ap.add_argument("--ctrl_wm_gap_target", type=float, default=0.08, help="Target EMA gap between wm_head(wm) and wm_head(clean).")
    ap.add_argument("--ctrl_detwarm_eps_floor", type=float, default=0.08, help="Minimum eps while detector warmup is active.")
    ap.add_argument("--ctrl_delta_abs_floor", type=float, default=0.0010, help="If delta_abs falls below this, controller pushes eps up.")
    ap.add_argument("--ctrl_eps_max", type=float, default=0.12, help="Hard cap for controller eps.")
    ap.add_argument("--ctrl_eps_up", type=float, default=0.0010, help="Normal eps growth step when EMA delta-margin is below target.")
    ap.add_argument("--ctrl_eps_zero_boost", type=float, default=0.0015, help="Extra eps growth only when watermark amplitude nearly vanishes.")
    ap.add_argument("--ctrl_class_gap_target", type=float, default=0.20, help="Desired EMA gated class gap before eps is allowed to relax.")
    ap.add_argument("--ctrl_gate_specific_target", type=float, default=0.15, help="Desired EMA gate-specific gap before eps is allowed to relax.")
    ap.add_argument("--ctrl_margin_gate_specific_target", type=float, default=0.20, help="Desired EMA margin-based gate-specific gap before eps is allowed to relax.")
    ap.add_argument("--c2_eval_ckpt", type=str, default="", help="Optional c2_eval_eXXX.pth to resume C2 / EMA / opt_c2 in train mode.")
    ap.add_argument("--opt_g_ckpt", type=str, default="", help="Optional opt_g_eXXX.pth to resume opt_g in train mode.")

    ap.add_argument("--gate_strength", type=float, default=2.10, help="Gate strength for the protected classifier.")
    ap.add_argument("--gpus", type=str, default="")
    ap.add_argument("--no_dataparallel", action="store_true")

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--image_size", type=int, default=160)
    ap.add_argument("--pad_value", type=float, default=0.0)

    ap.add_argument("--train_prod_mix_prob", type=float, default=0.70,
                    help="Probability to train C2 on prod-like synthesis (k_factor=1,varpercent=False)")
    ap.add_argument("--wm_res_clip", type=float, default=0.10, help="Clamp luma residual delta magnitude (in [0,1] luma units). 0 disables.")
    ap.add_argument("--wm_res_clip_mode", type=str, default="tanh", choices=["tanh", "hard", "none"], help="Residual clamp mode: tanh|hard|none")

    ap.add_argument("--lat_quota_min", type=float, default=0.20, help="Soft minimum fraction of watermark energy in latent branch (0 disables).")
    ap.add_argument("--lat_quota_lambda", type=float, default=0.10, help="Weight of latent quota penalty.")
    ap.add_argument("--lat_quota_warmup_epochs", type=int, default=2, help="Enable latent quota after this many warmup epochs.")

    ap.add_argument("--spec_window", type=int, default=9, help="Window size for low-frequency energy estimate (odd preferred).")
    ap.add_argument("--spec_lowfreq_max", type=float, default=0.55, help="Upper bound for low-frequency ratio (band control).")
    ap.add_argument("--spec_lowfreq_min", type=float, default=0.25, help="Lower bound for low-frequency ratio (band control).")
    ap.add_argument("--spec_lambda", type=float, default=0.05, help="Weight of low-frequency spectral penalty (0 disables).")
    ap.add_argument("--band_energy_norm", type=int, default=1, choices=[0, 1], help="Normalize low/mid branches before auto band mixing so controller weights reflect actual energy.")
    ap.add_argument("--band_norm_max_gain", type=float, default=4.0, help="Clamp RMS restoration gain after branch normalization in auto bandmix.")
    ap.add_argument("--band_mid_floor", type=float, default=0.20, help="Optional minimum mid-band share in auto bandmix (0 disables).")

    ap.add_argument("--lr_c2", type=float, default=1e-4)
    ap.add_argument("--lr_g", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)

    ap.add_argument("--alpha_lat_gain", type=float, default=1.0)
    ap.add_argument("--alpha_skip_gain", type=float, default=1.0)

    # anti-grid controls (skip feature domain)
    ap.add_argument("--skip_hp_mix", type=float, default=0.50, help="Mix of HP vs LP in skip (S64) watermark: 1=HP-only (grid prone), 0=LP-only.")
    ap.add_argument("--w64_hp_k", type=int, default=5, help="Kernel size for lowpass/highpass split in S64 domain (odd >=3).")
    ap.add_argument("--w64_post_blur_k", type=int, default=3, help="Extra blur in S64 domain to suppress grid (0 disables).")

    ap.add_argument("--print_every", type=int, default=1)
    ap.add_argument("--val_probe_every", type=int, default=5, help="Run a quick val probe every N train iterations (0 disables)")
    ap.add_argument("--val_probe_batches", type=int, default=8, help="How many val batches to use for the probe")
    ap.add_argument("--white_thr", type=float, default=0.70, help="Bright-region threshold for L_obj diagnostic (in [0,1])")
    ap.add_argument("--val_every", type=int, default=1)
    ap.add_argument("--collage_every", type=int, default=50)
    ap.add_argument("--pub_collage_enable", type=int, default=1, choices=[0, 1], help="Save publication collages from validation each epoch")
    ap.add_argument("--pub_collage_per_class", type=int, default=10, help="How many publication collages to save per class per epoch")
    ap.add_argument("--pub_collage_keep_train_debug", type=int, default=0, choices=[0, 1], help="Also save train-time debug collages using publication panels")

    ap.add_argument("--export_wm_dataset", action="store_true", default=True)
    ap.add_argument("--no_export_wm_dataset", dest="export_wm_dataset", action="store_false")

    ap.add_argument("--wm_jpeg_quality", type=int, default=92)
    ap.add_argument("--wm_export_format", type=str, default="png", choices=["png", "jpg"])
    ap.add_argument("--avoid_black_thr", type=float, default=0.0)

    ap.add_argument("--avoid_white_thr", type=float, default=1.01, help="If <=1.0, exclude near-white pixels (luma) from ROI+embedding (helps avoid highlight speckle/saturation).")

    # eval/infer profile overrides (optional)
    ap.add_argument("--eval_eps", type=float, default=0.0)
    ap.add_argument("--eval_r_skip", type=float, default=-1.0)

    # delta shaping (prevents gray veils / background lift)
    ap.add_argument("--delta_remove_dc", type=int, default=1, choices=[0, 1])
    ap.add_argument("--delta_hp_beta", type=float, default=0.70)
    ap.add_argument("--delta_hp_window", type=int, default=9)
    ap.add_argument("--delta_post_blur_k", type=int, default=3, help="Post-blur kernel on deltaY to suppress grid (0 disables).")
    ap.add_argument("--bg_protect_thr", type=float, default=0.06)
    ap.add_argument("--bg_delta_scale", type=float, default=0.20)
    ap.add_argument("--delta_renorm", type=int, default=1, choices=[0, 1])
    ap.add_argument("--delta_renorm_max", type=float, default=3.0)
    ap.add_argument("--headroom_margin", type=float, default=0.98, help="Headroom margin for saturation-safe delta clamp (0.98 keeps 2%% margin from [0,1] limits).")
    ap.add_argument("--min_delta_rms_k", type=float, default=0.015, help="Minimum delta RMS as a fraction of eps to prevent zero-watermark collapse.")
    ap.add_argument("--min_delta_lambda", type=float, default=0.20, help="Weight for minimum-delta hinge loss.")
    ap.add_argument("--max_alpha_boost", type=float, default=12.0, help="Maximum adaptive boost for alpha when AE response is too weak.")
    ap.add_argument("--freq_adapt", type=int, default=0, choices=[0, 1])
    ap.add_argument("--freq_beta_min", type=float, default=0.50)
    ap.add_argument("--freq_beta_max", type=float, default=0.80)
    ap.add_argument("--freq_bg_scale_min", type=float, default=0.10)
    ap.add_argument("--freq_bg_scale_max", type=float, default=0.30)
    ap.add_argument("--freq_tex_window", type=int, default=9)

    # inference output
    ap.add_argument("--infer_suffix", type=str, default="_watermarked")
    ap.add_argument("--infer_save_base", action="store_true")
    ap.add_argument("--infer_save_diff", action="store_true")
    ap.add_argument("--infer_max_images", type=int, default=0)

    ap.add_argument("--real_val_every", type=int, default=1, help="Run production-like validation every N epochs (0 disables)")
    ap.add_argument("--real_val_max_batches", type=int, default=0, help="Limit real-val to N batches (0 = all)")
    ap.add_argument("--real_val_jpeg_quality", type=int, default=92, help="JPEG roundtrip quality for real-val (0 disables)")
    ap.add_argument("--real_val_jpeg_quality_lo", type=int, default=85, help="Additional lower JPEG quality stress test for real-val (0 disables)")
    ap.add_argument("--real_val_resize_small", type=int, default=144, help="Resize roundtrip small side for real-val stress (0 disables)")
    ap.add_argument("--real_val_print_confusion", type=int, default=1, choices=[0, 1], help="Print confusion matrices for REALVAL (1=on)")
    ap.add_argument("--real_val_diag_unmasked", type=int, default=1, choices=[0, 1], help="Also compute unmasked acc (shortcut diagnostic) (1=on)")
    ap.add_argument("--real_val_stop_enable", type=int, default=1, choices=[0, 1], help="Auto-stop training when REALVAL classification drops below (random - margin_pp).")
    ap.add_argument("--real_val_stop_margin_pp", type=float, default=5.0, help="Absolute percentage-point margin below random accuracy used for REALVAL auto-stop.")
    ap.add_argument("--real_val_stop_patience", type=int, default=1, help="Number of consecutive bad REALVAL checks before stopping.")
    ap.add_argument("--real_val_stop_metric", type=str, default="both", choices=["raw", "base", "both"], help="Which REALVAL classification metric to monitor for auto-stop.")
    ap.add_argument("--real_val_stop_scope", type=str, default="worst", choices=["primary", "worst"], help="Use primary mode only or the worst mode across REALVAL stress suite for auto-stop.")
    ap.add_argument("--best_realval_acc_both_floor", type=float, default=0.70, help="Floor on worst REALVAL acc_both for selecting the best checkpoint.")
    ap.add_argument("--best_realval_det_acc_floor", type=float, default=0.85, help="Floor on worst REALVAL detector accuracy (raw-threshold) for selecting the best checkpoint.")
    ap.add_argument("--best_realval_max_border_ratio", type=float, default=0.10, help="Maximum allowed worst REALVAL border_ratio for selecting the best checkpoint.")
    ap.add_argument("--best_realval_gap_floor", type=float, default=0.05, help="Minimum required worst REALVAL gated gap(acc_both-acc_raw) for selecting the best checkpoint.")
    ap.add_argument("--best_realval_gate_specific_floor", type=float, default=0.05, help="Minimum required worst REALVAL gate-specific gap = gated_gap - ungated_gap.")

    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    seed_everything(args.seed)

    mode = str(getattr(args, "mode", "train") or "train").lower().strip()
    if mode not in ("train", "infer"):
        raise SystemExit(f"--mode must be train|infer (got {mode!r})")

    # conditional required args
    if mode == "train":
        if not args.train_root or not args.val_root:
            raise SystemExit("Train mode requires --train_root and --val_root")
    else:
        if not args.system_ckpt:
            raise SystemExit("Infer mode requires --system_ckpt (wm_system_eXXX.pth)")
        if (not args.infer_root) and (not args.infer_list):
            raise SystemExit("Infer mode requires --infer_root or --infer_list")

    train_root = Path(args.train_root) if args.train_root else Path(".")
    val_root = Path(args.val_root) if args.val_root else Path(".")
    infer_root = Path(args.infer_root) if args.infer_root else None
    infer_list = Path(args.infer_list) if args.infer_list else None
    infer_out = Path(args.infer_out) if args.infer_out else None
    system_ckpt = Path(args.system_ckpt) if args.system_ckpt else None
    c2_eval_ckpt = Path(args.c2_eval_ckpt) if args.c2_eval_ckpt else None
    opt_g_ckpt = Path(args.opt_g_ckpt) if args.opt_g_ckpt else None

    cfg = TrainConfig(
        train_root=train_root,
        val_root=val_root,
        mode=mode,
        system_ckpt=system_ckpt,
        c2_eval_ckpt=c2_eval_ckpt,
        opt_g_ckpt=opt_g_ckpt,
        infer_root=infer_root,
        infer_list=infer_list,
        infer_out=infer_out,
        infer_suffix=str(args.infer_suffix),
        infer_save_base=bool(args.infer_save_base),
        infer_save_diff=bool(args.infer_save_diff),
        infer_max_images=int(args.infer_max_images),
        profile=str(args.profile),
        eval_eps=float(args.eval_eps),
        eval_r_skip=float(args.eval_r_skip),
        delta_remove_dc=bool(int(args.delta_remove_dc)),
        delta_hp_beta=float(args.delta_hp_beta),
        delta_hp_window=int(args.delta_hp_window),
        delta_post_blur_k=int(args.delta_post_blur_k),
        delta_post_blur_mix=float(args.delta_post_blur_mix),
        grid_boundary_lambda=float(args.grid_boundary_lambda),
        bg_protect_thr=float(args.bg_protect_thr),
        bg_delta_scale=float(args.bg_delta_scale),
        delta_renorm=bool(int(args.delta_renorm)),
        delta_renorm_max=float(args.delta_renorm_max),
        headroom_margin=float(args.headroom_margin),
        avoid_white_thr=float(args.avoid_white_thr),
        min_delta_rms_k=float(args.min_delta_rms_k),
        min_delta_lambda=float(args.min_delta_lambda),
        max_alpha_boost=float(args.max_alpha_boost),
        freq_adapt=bool(int(args.freq_adapt)),
        freq_beta_min=float(args.freq_beta_min),
        freq_beta_max=float(args.freq_beta_max),
        freq_bg_scale_min=float(args.freq_bg_scale_min),
        freq_bg_scale_max=float(args.freq_bg_scale_max),
        freq_tex_window=int(args.freq_tex_window),
        out_root=Path(args.out_root),
        ae_ckpt=Path(args.ae_ckpt),
        ae_module=args.ae_module,
        ae_class=args.ae_class,
        ae_py_path=Path(args.ae_py_path),
        c1_ckpt=(Path(args.c1_ckpt) if args.c1_ckpt else None),
        c1_guard_min_acc=float(args.c1_guard_min_acc),
        c1_guard_max_drop=float(args.c1_guard_max_drop),
        c1_guard_lambda=float(args.c1_guard_lambda),
        c1_guard_every=int(args.c1_guard_every),
        c1_guard_ce_margin=float(args.c1_guard_ce_margin),
        c1_brake_eps=float(args.c1_brake_eps),
        ctrl_init_eps=float(args.ctrl_init_eps),
        ctrl_init_r_skip=float(args.ctrl_init_r_skip),
        ctrl_dmargin_target=float(args.ctrl_dmargin_target),
        ctrl_wm_gap_target=float(args.ctrl_wm_gap_target),
        ctrl_detwarm_eps_floor=float(args.ctrl_detwarm_eps_floor),
        ctrl_delta_abs_floor=float(args.ctrl_delta_abs_floor),
        ctrl_eps_max=float(args.ctrl_eps_max),
        ctrl_eps_up=float(args.ctrl_eps_up),
        ctrl_eps_zero_boost=float(args.ctrl_eps_zero_boost),
        ctrl_class_gap_target=float(args.ctrl_class_gap_target),
        ctrl_gate_specific_target=float(args.ctrl_gate_specific_target),
        ctrl_margin_gate_specific_target=float(args.ctrl_margin_gate_specific_target),
        c2_init_from_c1=bool(int(args.c2_init_from_c1)),
        det_warmup_epochs=int(args.det_warmup_epochs),
        det_warmup_w_det=float(args.det_warmup_w_det),
        det_warmup_k_factor=float(args.det_warmup_k_factor),
        det_post_w_det_early=float(args.det_post_w_det_early),
        det_post_w_det_late=float(args.det_post_w_det_late),
        det_post_w_sep_early=float(args.det_post_w_sep_early),
        det_post_w_sep_late=float(args.det_post_w_sep_late),
        det_warm_unfreeze_layer4=bool(int(args.det_warm_unfreeze_layer4)),
        clean_consistency=bool(int(args.clean_consistency)),
        clean_consistency_temp=float(args.clean_consistency_temp),
        w_clean=float(args.w_clean),
        c2_keygap_margin=float(args.c2_keygap_margin),
        c2_keygap_w_suppress=float(args.c2_keygap_w_suppress),
        c2_keygap_w_cap=float(args.c2_keygap_w_cap),
        c2_ng_invar_w=float(args.c2_ng_invar_w),
        c2_ng_kl_temp=float(args.c2_ng_kl_temp),
        c2_gate_specific_w=float(args.c2_gate_specific_w),
        c2_gate_specific_margin=float(args.c2_gate_specific_margin),
        c2_clean_ng_keep_w=float(args.c2_clean_ng_keep_w),
        c2_clean_ng_margin_floor=float(args.c2_clean_ng_margin_floor),
        c2_warm_ng_invar_mult=float(args.c2_warm_ng_invar_mult),
        c2_warm_gate_specific_mult=float(args.c2_warm_gate_specific_mult),
        c2_warm_clean_ng_keep_mult=float(args.c2_warm_clean_ng_keep_mult),
        c2_early_ng_invar_mult=float(args.c2_early_ng_invar_mult),
        c2_early_gate_specific_mult=float(args.c2_early_gate_specific_mult),
        c2_early_clean_ng_keep_mult=float(args.c2_early_clean_ng_keep_mult),
        c2_late_ng_invar_mult=float(args.c2_late_ng_invar_mult),
        c2_late_gate_specific_mult=float(args.c2_late_gate_specific_mult),
        c2_late_clean_ng_keep_mult=float(args.c2_late_clean_ng_keep_mult),
        c2_no_gap_ng_invar_mult=float(args.c2_no_gap_ng_invar_mult),
        c2_no_gap_gate_specific_mult=float(args.c2_no_gap_gate_specific_mult),
        c2_no_gap_clean_ng_keep_mult=float(args.c2_no_gap_clean_ng_keep_mult),
        c2_best_checkpoint_require_pass_floors=True,
        g_push_start_epoch=int(args.g_push_start_epoch),
        g_push_margin_target=float(args.g_push_margin_target),
        g_push_cls_lambda=float(args.g_push_cls_lambda),
        g_push_det_lambda=float(args.g_push_det_lambda),
        g_push_margin_lambda=float(args.g_push_margin_lambda),
        gate_strength=float(args.gate_strength),
        gpus=args.gpus,
        use_dataparallel=(not args.no_dataparallel),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        image_size=int(args.image_size),
        pad_value=float(args.pad_value),
        train_prod_mix_prob=float(args.train_prod_mix_prob),
        wm_res_clip=float(args.wm_res_clip),
        wm_res_clip_mode=str(args.wm_res_clip_mode),
        lat_quota_min=float(args.lat_quota_min),
        lat_quota_lambda=float(args.lat_quota_lambda),
        lat_quota_warmup_epochs=int(args.lat_quota_warmup_epochs),
        spec_window=int(args.spec_window),
        spec_lowfreq_max=float(args.spec_lowfreq_max),
        spec_lowfreq_min=float(args.spec_lowfreq_min),
        spec_lambda=float(args.spec_lambda),
        band_energy_norm=bool(int(args.band_energy_norm)),
        band_norm_max_gain=float(args.band_norm_max_gain),
        band_mid_floor=float(args.band_mid_floor),
        lr_c2=float(args.lr_c2),
        lr_g=float(args.lr_g),
        wd=float(args.wd),
        alpha_lat_gain=float(args.alpha_lat_gain),
        alpha_skip_gain=float(args.alpha_skip_gain),
        skip_hp_mix=float(args.skip_hp_mix),
        w64_hp_k=int(args.w64_hp_k),
        w64_post_blur_k=int(args.w64_post_blur_k),
        print_every=int(args.print_every),
        val_probe_every=int(args.val_probe_every),
        val_probe_batches=int(args.val_probe_batches),
        white_thr=float(args.white_thr),
        val_every=int(args.val_every),
        collage_every=int(args.collage_every),
        pub_collage_enable=bool(int(args.pub_collage_enable)),
        pub_collage_per_class=int(args.pub_collage_per_class),
        pub_collage_keep_train_debug=bool(int(args.pub_collage_keep_train_debug)),
        export_wm_dataset=bool(args.export_wm_dataset),
        wm_jpeg_quality=int(args.wm_jpeg_quality),
        wm_export_format=str(args.wm_export_format),
        avoid_black_thr=float(args.avoid_black_thr),
        real_val_every=int(args.real_val_every),
        real_val_max_batches=int(args.real_val_max_batches),
        real_val_jpeg_quality=int(args.real_val_jpeg_quality),
        real_val_jpeg_quality_lo=int(args.real_val_jpeg_quality_lo),
        real_val_resize_small=int(args.real_val_resize_small),
        real_val_print_confusion=bool(int(args.real_val_print_confusion)),
        real_val_diag_unmasked=bool(int(args.real_val_diag_unmasked)),
        real_val_stop_enable=bool(int(args.real_val_stop_enable)),
        real_val_stop_margin_pp=float(args.real_val_stop_margin_pp),
        real_val_stop_patience=int(args.real_val_stop_patience),
        real_val_stop_metric=str(args.real_val_stop_metric),
        real_val_stop_scope=str(args.real_val_stop_scope),
        best_realval_acc_both_floor=float(args.best_realval_acc_both_floor),
        best_realval_det_acc_floor=float(args.best_realval_det_acc_floor),
        best_realval_max_border_ratio=float(args.best_realval_max_border_ratio),
        best_realval_gap_floor=float(args.best_realval_gap_floor),
        best_realval_gate_specific_floor=float(args.best_realval_gate_specific_floor),
    )
    return cfg


def main():
    cfg = parse_args()
    t = WatermarkTrainer(cfg)
    if str(getattr(cfg, "mode", "train") or "train").lower().strip() == "infer":
        t.infer()
    else:
        t.train()


if __name__ == "__main__":
    main()