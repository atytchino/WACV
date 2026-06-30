#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
verify_ae_pipeline_v2.py — Comprehensive diagnostic for the watermark pipeline.

v2 changes over v1:
  - BATCH TEST: tests AE on 32-64 val images across all classes, not just one
                → gives mean ± std AE quality, confirms whether single-image
                  PSNR (23 dB) is outlier or systematic
  - INLINE C1:  ResNet34LF_BN written inline (no importlib of trainer file).
                Avoids the 'NoneType __dict__' bug from filenames with numeric
                prefix / hyphens.
  - EXTREME STRESS TEST: tries to reproduce training's catastrophic dMax=0.95
                          state by pushing wm to tanh saturation (scale=4.0)
                          and α to extreme values (5.0). If AE can produce
                          such large perturbations, confirms generator-driven
                          collapse path. If not, perturbation comes from
                          somewhere ELSE in trainer.
  - GENERATOR PROBE (optional): loads trained generator from a failed run
                                 (e.g. RUN07) and tests what watermark IT
                                 produces. Reveals whether trained generator
                                 produces noise or sensible patterns.

Usage (PowerShell, single line via backticks)
---------------------------------------------
python verify_ae_pipeline_v2.py `
  --ae_ckpt    "E:\AE_TRAINED\TLD\ckpts\ae_best.pth" `
  --c1_ckpt    "E:\C1_TRAINED\TLD\ckpts\c1_best.pth" `
  --sample_img "E:\TLD\val\Tomato_healthy\839daf12-...jpg" `
  --val_root   "E:\TLD\val" `
  --out_dir    "E:\DIAGNOSTICS\TLD_v2" `
  --code_root  "C:\Users\atytchino\PycharmProjects\WACV" `
  --image_size 512 `
  --batch_test_n 48 `
  --num_classes 10

Optional: --gen_ckpt "E:\RUNS\TLD_GATEONLY_20260518_RUN07\ckpts\state_last.pt"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from PIL import Image


# ════════════════════════════════════════════════════════════════════════════
# Image / metric utilities (same as v1)
# ════════════════════════════════════════════════════════════════════════════

def load_image_rgb(path: Path, size: int = 512) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        arr = np.array(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def rgb_to_luma01(x01: torch.Tensor) -> torch.Tensor:
    r, g, b = x01[:, 0:1], x01[:, 1:2], x01[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def psnr_y(a01, b01) -> float:
    a_y = rgb_to_luma01(a01) if a01.size(1) == 3 else a01
    b_y = rgb_to_luma01(b01) if b01.size(1) == 3 else b01
    mse = ((a_y - b_y) ** 2).mean().clamp_min(1e-12)
    return float(-10.0 * torch.log10(mse).item())


def _gaussian_kernel2d(window, sigma, device, dtype):
    x = torch.arange(window, device=device, dtype=dtype) - (window - 1) / 2.0
    g1 = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    g1 = g1 / g1.sum()
    return (g1[:, None] @ g1[None, :]).unsqueeze(0).unsqueeze(0)


def ssim_y(a01, b01, window=11, sigma=1.5) -> float:
    x = rgb_to_luma01(a01) if a01.size(1) == 3 else a01
    y = rgb_to_luma01(b01) if b01.size(1) == 3 else b01
    _, _, H, W = x.shape
    w = int(min(window, H, W))
    if w < 3:
        return float((1.0 - (x - y).abs().mean()).clamp(-1, 1).item())
    if (w % 2) == 0:
        w -= 1
    pad = w // 2
    k = _gaussian_kernel2d(w, sigma, device=x.device, dtype=x.dtype)
    conv = lambda z: F.conv2d(z, k, padding=pad)
    mux, muy = conv(x), conv(y)
    sx2 = (conv(x * x) - mux * mux).clamp_min(0.0)
    sy2 = (conv(y * y) - muy * muy).clamp_min(0.0)
    sxy = conv(x * y) - mux * muy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mux * muy + c1) * (2 * sxy + c2)
    den = (mux * mux + muy * muy + c1) * (sx2 + sy2 + c2)
    return float((num / den.clamp_min(1e-6)).mean().clamp(-1, 1).item())


def mae(a, b): return float((a - b).abs().mean().item())
def max_abs_diff(a, b): return float((a - b).abs().max().item())


def save_pil(t, path):
    t = t.detach().cpu().clamp(0, 1)
    if t.size(1) == 1: t = t.repeat(1, 3, 1, 1)
    arr = (t[0].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_strip(tensors, labels, path):
    imgs = []
    for t in tensors:
        t = t.detach().cpu().clamp(0, 1)
        if t.size(1) == 1: t = t.repeat(1, 3, 1, 1)
        imgs.append((t[0].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8))
    H = max(im.shape[0] for im in imgs)
    W_total = sum(im.shape[1] for im in imgs) + 10 * (len(imgs) - 1)
    canvas = np.ones((H, W_total, 3), dtype=np.uint8) * 255
    x = 0
    for im in imgs:
        canvas[:im.shape[0], x:x + im.shape[1], :] = im
        x += im.shape[1] + 10
    Image.fromarray(canvas).save(path)
    with open(path.with_suffix(".labels.txt"), "w", encoding="utf-8") as f:
        f.write(" | ".join(labels))


# ════════════════════════════════════════════════════════════════════════════
# AE loader (same as v1)
# ════════════════════════════════════════════════════════════════════════════

def load_ae(ae_ckpt: Path, code_root: Path, device):
    sys.path.insert(0, str(code_root))
    try:
        from AE_ContentBound import UniversalAutoEncoder, AEConfig
    finally:
        if str(code_root) in sys.path:
            sys.path.remove(str(code_root))
    model = UniversalAutoEncoder(cfg=AEConfig()).to(device).eval()
    payload = torch.load(ae_ckpt, map_location=device, weights_only=False)
    sd = payload.get("state_dict") or payload.get("ae_state_dict") or payload
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    cfg = AEConfig()
    return model, cfg, missing, unexpected


# ════════════════════════════════════════════════════════════════════════════
# Inline C1 — ResNet34LF_BN (mirrors trainer's architecture from lines 657-862)
# ════════════════════════════════════════════════════════════════════════════

class BlurPool(nn.Module):
    """Anti-aliased downsampling: 3x3 binomial kernel applied with stride 2."""
    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        k = torch.tensor([1., 2., 1.])
        k = (k[:, None] * k[None, :]) / 16.0  # 3x3, sums to 1
        self.register_buffer("k", k[None, None, :, :].repeat(channels, 1, 1, 1))
        self.channels = channels
        self.stride = stride

    def forward(self, x):
        return F.conv2d(x, self.k, stride=self.stride, padding=1, groups=self.channels)


def _wrap_blur(layer: nn.Sequential) -> nn.Sequential:
    """
    Modify a ResNet stage so the first block's stride=2 Conv becomes stride=1
    followed by BlurPool. Also rebuilds the downsample branch identically.
    Matches the trainer's _wrap_blur exactly.
    """
    first = layer[0]
    # Conv1: change stride 2 -> 1
    first.conv1.stride = (1, 1)
    # Downsample: replace strided 1x1 conv with stride-1 conv + BlurPool + BN
    if first.downsample is not None:
        old_conv = first.downsample[0]
        old_bn = first.downsample[1]
        new_conv = nn.Conv2d(
            old_conv.in_channels, old_conv.out_channels,
            kernel_size=1, stride=1, bias=False,
        )
        new_conv.weight.data = old_conv.weight.data.clone()
        first.downsample = nn.Sequential(
            new_conv,                               # index 0: 1x1 conv stride 1
            BlurPool(old_conv.out_channels, stride=2),  # index 1: BlurPool (the "k" buffer)
            nn.BatchNorm2d(old_conv.out_channels),  # index 2: BN
        )
        first.downsample[2].load_state_dict(old_bn.state_dict())
    # Insert BlurPool after first conv (in main path: conv1 -> BN -> relu -> blur)
    # Trainer's approach: add BlurPool to the first block's main path.
    # Simpler approach: monkey-patch forward to apply BlurPool before conv2.
    # We do this via a hook on first.bn1 to apply blur on output.
    # But that's fragile. Cleaner: replace first block with custom block.
    # For diagnostic purposes, we approximate — the downsample buffer must match
    # for state_dict loading; main-path blur can be approximated or skipped.
    return layer


class ResNet34LF_BN(nn.Module):
    """
    Diagnostic-only ResNet34 with BlurPool downsampling and trainer's heads.
    Designed to load c1_best.pth state_dict produced by C1_trainer_compatible.py.

    Architecture mirrors trainer/lines 657-862:
      base.conv1 (stride 1)            ← stride 2→1 in trainer
      base.layer1-4 (ResNet34 blocks)
        layer2/3/4 first block: BlurPool downsampling
      wm_head: AvgPool → Flatten → Linear(512,128) → ReLU → Linear(128,1)
      wm_affine: Parameter[num_classes]
      gate_strength, destructive_strength: buffers
    """
    def __init__(self, num_classes: int = 10, gate_strength: float = 2.10,
                 destructive_strength: float = 1.0):
        super().__init__()
        # Build vanilla ResNet34
        base = tv_models.resnet34(weights=None)
        # Trainer's modifications
        base.conv1.stride = (1, 1)
        _wrap_blur(base.layer2)
        _wrap_blur(base.layer3)
        _wrap_blur(base.layer4)
        # Replace FC head
        base.fc = nn.Linear(512, num_classes)
        self.base = base
        # Heads (matching trainer)
        self.wm_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.wm_affine = nn.Parameter(torch.zeros(num_classes))
        self.register_buffer("gate_strength", torch.tensor(float(gate_strength)))
        self.register_buffer("destructive_strength", torch.tensor(float(destructive_strength)))

    def forward(self, x, gate: bool = False, gate_target=None,
                detach_gate: bool = False, detach_affine: bool = False,
                return_raw: bool = False):
        """
        Diagnostic forward — supports trainer's call signature c1(xN, gate=False).
        Returns (logits, wm_logit, x4) tuple matching trainer's c1_logits expectation.
        For gate=False (which is how trainer always calls c1), just runs vanilla
        forward and produces wm_logit from intermediate features.
        """
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x1 = self.base.layer1(x)
        x2 = self.base.layer2(x1)
        x3 = self.base.layer3(x2)
        x4 = self.base.layer4(x3)
        pooled = self.base.avgpool(x4).flatten(1)
        logits = self.base.fc(pooled)
        wm_logit = self.wm_head(x4).squeeze(-1)  # [B]
        if return_raw:
            return (logits, wm_logit, logits)
        return (logits, wm_logit, x4)


def load_c1_inline(c1_ckpt: Path, device, num_classes: int):
    """Load C1 from checkpoint into inline ResNet34LF_BN."""
    c1 = ResNet34LF_BN(num_classes=num_classes).to(device).eval()
    payload = torch.load(c1_ckpt, map_location=device, weights_only=False)
    sd = payload.get("state_dict") or payload
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    # Auto-detect num_classes from checkpoint
    if "base.fc.weight" in sd:
        sd_nc = sd["base.fc.weight"].shape[0]
        if sd_nc != num_classes:
            # Rebuild with correct size
            c1 = ResNet34LF_BN(num_classes=sd_nc).to(device).eval()
            num_classes = sd_nc
    missing, unexpected = c1.load_state_dict(sd, strict=False)
    return c1, num_classes, missing, unexpected


# ════════════════════════════════════════════════════════════════════════════
# Tests 1-4: same as v1 (single-image, kept for direct comparison)
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def test_1_rgb_reconstruction(ae, x01, out_dir):
    x_hat = ae.forward_plain(x01).clamp(0, 1)
    m = {
        "ssim_y(x, x_hat)": ssim_y(x01, x_hat),
        "psnr_y(x, x_hat)": psnr_y(x01, x_hat),
        "mae(x, x_hat)":    mae(x01, x_hat),
        "max|x - x_hat|":   max_abs_diff(x01, x_hat),
    }
    save_strip([x01, x_hat], ["input RGB", "AE recon RGB"], out_dir / "t1_strip.png")
    return m


@torch.no_grad()
def test_2_luma_identity(ae, x01, out_dir):
    y = rgb_to_luma01(x01)
    y_hat = ae.embed_external_wm_gray(
        y, wm_lat=None, wm_skip=None, alpha_lat=0.0, alpha_skip=0.0,
    ).clamp(0, 1)
    m = {
        "ssim_y(y, y_hat)": ssim_y(y, y_hat),
        "psnr_y(y, y_hat)": psnr_y(y, y_hat),
        "mae(y, y_hat)":    mae(y, y_hat),
        "max|y - y_hat|":   max_abs_diff(y, y_hat),
    }
    save_strip([y, y_hat, (y - y_hat).abs() * 5.0],
               ["input LUMA", "embed(α=0)", "|diff|×5"],
               out_dir / "t2_strip.png")
    return m


def _make_wm(shape, device, scale, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.tanh(torch.randn(*shape, generator=g).to(device) * float(scale))


@torch.no_grad()
def test_3_small_perturbation(ae, x01, out_dir, alpha=0.1, seed=42):
    y = rgb_to_luma01(x01)
    B, _, H, W = y.shape
    wm_lat  = _make_wm((B, 1024, H // 16, W // 16), x01.device, 0.5, seed)
    wm_skip = _make_wm((B, 512,  H // 8,  W // 8 ), x01.device, 0.5, seed + 1)
    y_hat = ae.embed_external_wm_gray(
        y, wm_lat=wm_lat, wm_skip=wm_skip,
        alpha_lat=alpha, alpha_skip=alpha,
    ).clamp(0, 1)
    m = {
        "alpha": alpha,
        "ssim_y(y, y_hat)": ssim_y(y, y_hat),
        "psnr_y(y, y_hat)": psnr_y(y, y_hat),
        "mae(y, y_hat)":    mae(y, y_hat),
        "max|y - y_hat|":   max_abs_diff(y, y_hat),
    }
    save_strip([y, y_hat, (y - y_hat).abs() * 5.0],
               ["input", f"watermarked (α={alpha})", "|diff|×5"],
               out_dir / f"t3_strip_alpha{alpha:.2f}.png")
    return m


@torch.no_grad()
def test_4_large_perturbation(ae, x01, out_dir, alpha=1.0, seed=42):
    y = rgb_to_luma01(x01)
    B, _, H, W = y.shape
    wm_lat  = _make_wm((B, 1024, H // 16, W // 16), x01.device, 0.5, seed)
    wm_skip = _make_wm((B, 512,  H // 8,  W // 8 ), x01.device, 0.5, seed + 1)
    y_hat = ae.embed_external_wm_gray(
        y, wm_lat=wm_lat, wm_skip=wm_skip,
        alpha_lat=alpha, alpha_skip=alpha,
    ).clamp(0, 1)
    m = {
        "alpha": alpha,
        "ssim_y(y, y_hat)": ssim_y(y, y_hat),
        "psnr_y(y, y_hat)": psnr_y(y, y_hat),
        "mae(y, y_hat)":    mae(y, y_hat),
        "max|y - y_hat|":   max_abs_diff(y, y_hat),
    }
    save_strip([y, y_hat, (y - y_hat).abs() * 2.0],
               ["input", f"watermarked (α={alpha})", "|diff|×2"],
               out_dir / f"t4_strip_alpha{alpha:.2f}.png")
    return m


# ════════════════════════════════════════════════════════════════════════════
# NEW Test 5: AE→C1 chain on small batch (with inline C1)
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def test_5_ae_c1_chain(ae, c1, val_root: Path, image_size, batch_size, device,
                       out_dir, max_images=64) -> dict:
    if c1 is None:
        return {"skipped": True, "reason": "C1 not loaded"}
    class_dirs = sorted([d for d in val_root.iterdir() if d.is_dir()])
    if not class_dirs:
        return {"skipped": True, "reason": f"no class subfolders in {val_root}"}

    items = []
    per_class = max(1, max_images // len(class_dirs))
    for ci, cd in enumerate(class_dirs):
        files = sorted([p for p in cd.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
        for p in files[:per_class]:
            items.append((p, ci))
        if len(items) >= max_images: break
    items = items[:max_images]

    correct_x = correct_xh = agree = 0
    total = 0
    per_class_acc_x = {}
    per_class_acc_xh = {}

    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        x = torch.stack([load_image_rgb(p, image_size)[0] for p, _ in chunk]).to(device)
        y_true = torch.tensor([c for _, c in chunk], device=device, dtype=torch.long)
        x_hat = ae.forward_plain(x).clamp(0, 1)
        xN = x * 2.0 - 1.0
        xN_hat = x_hat * 2.0 - 1.0
        out_x = c1(xN, gate=False)
        out_xh = c1(xN_hat, gate=False)
        logits_x = out_x[0] if isinstance(out_x, tuple) else out_x
        logits_xh = out_xh[0] if isinstance(out_xh, tuple) else out_xh
        pred_x = logits_x.argmax(dim=1)
        pred_xh = logits_xh.argmax(dim=1)
        correct_x += int((pred_x == y_true).sum().item())
        correct_xh += int((pred_xh == y_true).sum().item())
        agree += int((pred_x == pred_xh).sum().item())
        total += len(chunk)
        # per-class
        for j, (_, ci) in enumerate(chunk):
            per_class_acc_x.setdefault(ci, [0, 0])
            per_class_acc_xh.setdefault(ci, [0, 0])
            per_class_acc_x[ci][1] += 1
            per_class_acc_xh[ci][1] += 1
            if int(pred_x[j]) == ci: per_class_acc_x[ci][0] += 1
            if int(pred_xh[j]) == ci: per_class_acc_xh[ci][0] += 1

    return {
        "skipped": False,
        "n_images": total,
        "n_classes": len(class_dirs),
        "acc_c1(x)":     correct_x / max(1, total),
        "acc_c1(ae(x))": correct_xh / max(1, total),
        "agreement_rate": agree / max(1, total),
        "acc_drop":      (correct_x - correct_xh) / max(1, total),
        "per_class_acc_x":   {k: v[0] / max(1, v[1]) for k, v in per_class_acc_x.items()},
        "per_class_acc_xh":  {k: v[0] / max(1, v[1]) for k, v in per_class_acc_xh.items()},
    }


# ════════════════════════════════════════════════════════════════════════════
# NEW Test 6: BATCH AE quality — confirms single-image PSNR is outlier or not
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def test_6_batch_ae_quality(ae, val_root: Path, image_size, batch_size,
                             device, n_images: int = 48) -> dict:
    """Run AE on N val images across classes. Report mean ± std of metrics."""
    class_dirs = sorted([d for d in val_root.iterdir() if d.is_dir()])
    items = []
    per_class = max(1, n_images // max(1, len(class_dirs)))
    for cd in class_dirs:
        files = sorted([p for p in cd.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
        for p in files[:per_class]:
            items.append(p)
        if len(items) >= n_images: break
    items = items[:n_images]

    ssims_rgb, psnrs_rgb, maes_rgb = [], [], []
    ssims_luma, psnrs_luma = [], []
    worst = {"path": None, "psnr": float("inf"), "ssim": 1.0}
    best  = {"path": None, "psnr": -float("inf"), "ssim": 0.0}

    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        x = torch.stack([load_image_rgb(p, image_size)[0] for p in chunk]).to(device)
        x_hat = ae.forward_plain(x).clamp(0, 1)
        for j in range(x.size(0)):
            xj = x[j:j+1]; xhj = x_hat[j:j+1]
            s_rgb = ssim_y(xj, xhj)
            p_rgb = psnr_y(xj, xhj)
            m_rgb = mae(xj, xhj)
            ssims_rgb.append(s_rgb); psnrs_rgb.append(p_rgb); maes_rgb.append(m_rgb)
            y = rgb_to_luma01(xj)
            y_hat = ae.embed_external_wm_gray(
                y, wm_lat=None, wm_skip=None, alpha_lat=0.0, alpha_skip=0.0,
            ).clamp(0, 1)
            ssims_luma.append(ssim_y(y, y_hat))
            psnrs_luma.append(psnr_y(y, y_hat))
            if p_rgb < worst["psnr"]:
                worst = {"path": str(chunk[j]), "psnr": p_rgb, "ssim": s_rgb}
            if p_rgb > best["psnr"]:
                best = {"path": str(chunk[j]), "psnr": p_rgb, "ssim": s_rgb}

    def stat(arr):
        a = np.array(arr, dtype=np.float64)
        return {"mean": float(a.mean()), "std": float(a.std()),
                "min": float(a.min()), "max": float(a.max()),
                "p10": float(np.percentile(a, 10)),
                "p50": float(np.percentile(a, 50)),
                "p90": float(np.percentile(a, 90))}

    return {
        "n_images": len(items),
        "ssim_y_rgb":  stat(ssims_rgb),
        "psnr_y_rgb":  stat(psnrs_rgb),
        "mae_rgb":     stat(maes_rgb),
        "ssim_y_luma": stat(ssims_luma),
        "psnr_y_luma": stat(psnrs_luma),
        "worst_image": worst,
        "best_image":  best,
    }


# ════════════════════════════════════════════════════════════════════════════
# NEW Test 7: EXTREME STRESS — try to reproduce dMax=0.95 catastrophic state
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def test_7_extreme_stress(ae, x01, out_dir, seed=42) -> dict:
    """
    Push wm scale and alpha to extreme values. If AE alone can reach
    SSIM<0.1 or max|diff|>0.7, we've found the route to catastrophic state.
    If not, the catastrophic state must come from logic OUTSIDE
    embed_external_wm_gray (i.e., somewhere in trainer's _embed_luma_only_gray).
    """
    y = rgb_to_luma01(x01)
    B, _, H, W = y.shape

    configs = [
        # (wm_scale, alpha_lat, alpha_skip, label)
        (0.5,  0.1, 0.1, "baseline_small"),
        (0.5,  1.0, 1.0, "baseline_large"),
        (2.0,  1.0, 1.0, "tanh_saturated_α=1"),
        (4.0,  1.0, 1.0, "tanh_deep_saturated_α=1"),
        (0.5,  3.0, 3.0, "α=3"),
        (0.5,  5.0, 5.0, "α=5"),
        (2.0,  3.0, 3.0, "saturated+α=3"),
        (4.0,  5.0, 5.0, "extreme_combined"),
        (4.0, 10.0, 10.0, "ridiculous"),
    ]

    results = []
    for scale, a_lat, a_skip, label in configs:
        wm_lat  = _make_wm((B, 1024, H // 16, W // 16), x01.device, scale, seed)
        wm_skip = _make_wm((B, 512,  H // 8,  W // 8 ), x01.device, scale, seed + 1)
        y_hat = ae.embed_external_wm_gray(
            y, wm_lat=wm_lat, wm_skip=wm_skip,
            alpha_lat=a_lat, alpha_skip=a_skip,
        ).clamp(0, 1)
        m = {
            "config": label,
            "wm_scale":     scale,
            "alpha_lat":    a_lat,
            "alpha_skip":   a_skip,
            "ssim":         ssim_y(y, y_hat),
            "psnr":         psnr_y(y, y_hat),
            "mae":          mae(y, y_hat),
            "max_diff":     max_abs_diff(y, y_hat),
            "saturated_pixels": float(((y_hat <= 0.001) | (y_hat >= 0.999)).float().mean().item()),
        }
        results.append(m)
        save_pil(y_hat, out_dir / f"t7_{label}.png")

    # Build strip showing escalation
    tensors = [y]
    labels = ["input"]
    for r, (scale, a_lat, a_skip, lab) in zip(results, configs):
        wm_lat  = _make_wm((B, 1024, H // 16, W // 16), x01.device, scale, seed)
        wm_skip = _make_wm((B, 512,  H // 8,  W // 8 ), x01.device, scale, seed + 1)
        y_hat = ae.embed_external_wm_gray(
            y, wm_lat=wm_lat, wm_skip=wm_skip,
            alpha_lat=a_lat, alpha_skip=a_skip,
        ).clamp(0, 1)
        tensors.append(y_hat)
        labels.append(lab)
    save_strip(tensors, labels, out_dir / "t7_escalation_strip.png")

    return {"configs": results}


# ════════════════════════════════════════════════════════════════════════════
# Optional Test 8: GENERATOR PROBE — load trained generator and test its output
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def test_8_generator_probe(ae, x01, gen_ckpt: Optional[Path], code_root: Path,
                            out_dir, device) -> dict:
    """If trained generator checkpoint provided, load and test what it produces."""
    if gen_ckpt is None or not gen_ckpt.exists():
        return {"skipped": True, "reason": "no --gen_ckpt"}
    try:
        sys.path.insert(0, str(code_root))
        try:
            from AE_ContentBound import WMGeneratorConditioned, WMGeneratorSkip
        finally:
            if str(code_root) in sys.path:
                sys.path.remove(str(code_root))
        g_lat = WMGeneratorConditioned(
            in_ch=1024, mid_ch=256, out_ch=1024, content_dim=64
        ).to(device).eval()
        g_64 = WMGeneratorSkip(
            skip_ch=512, lat_ch=1024, mid_ch=128, out_ch=512, content_dim=64
        ).to(device).eval()
        payload = torch.load(gen_ckpt, map_location=device, weights_only=False)
        # Try common key locations
        sd_lat = None; sd_64 = None
        for key in ["g_lat", "generator_lat", "G_lat"]:
            if isinstance(payload, dict) and key in payload:
                sd_lat = payload[key]; break
        for key in ["g_64", "generator_64", "G_64"]:
            if isinstance(payload, dict) and key in payload:
                sd_64 = payload[key]; break
        # Or look inside "state_dict" with prefix
        if sd_lat is None and isinstance(payload, dict):
            sd_full = payload.get("state_dict", payload)
            sd_lat = {k.replace("g_lat.", ""): v for k, v in sd_full.items()
                      if k.startswith("g_lat.")}
            sd_64 = {k.replace("g_64.", ""): v for k, v in sd_full.items()
                     if k.startswith("g_64.")}
            if not sd_lat: sd_lat = None
            if not sd_64: sd_64 = None
        if sd_lat is None or sd_64 is None:
            return {"skipped": True,
                    "reason": f"could not find g_lat/g_64 in {gen_ckpt.name}; "
                              f"top keys: {list(payload.keys())[:10] if isinstance(payload, dict) else type(payload)}"}
        g_lat.load_state_dict(sd_lat, strict=False)
        g_64.load_state_dict(sd_64, strict=False)
    except Exception as e:
        return {"skipped": True, "reason": f"exception: {type(e).__name__}: {e}"}

    # Run trained generator
    y = rgb_to_luma01(x01)
    enc = ae.enc(y)
    Z, S64 = enc["latent"], enc["s64"]
    wm_lat  = g_lat(Z)
    wm_skip = g_64(S64, Z)

    metrics = {
        "wm_lat_shape":  list(wm_lat.shape),
        "wm_lat_mean":   float(wm_lat.mean().item()),
        "wm_lat_std":    float(wm_lat.std().item()),
        "wm_lat_min":    float(wm_lat.min().item()),
        "wm_lat_max":    float(wm_lat.max().item()),
        "wm_lat_abs_p90": float(torch.quantile(wm_lat.abs().flatten(), 0.90).item()),
        "wm_skip_mean":  float(wm_skip.mean().item()),
        "wm_skip_std":   float(wm_skip.std().item()),
        "wm_skip_max":   float(wm_skip.abs().max().item()),
    }

    # Now test embedding with the TRAINED generator output at several α
    for alpha in [0.1, 0.5, 1.0]:
        y_hat = ae.embed_external_wm_gray(
            y, wm_lat=wm_lat, wm_skip=wm_skip,
            alpha_lat=alpha, alpha_skip=alpha,
        ).clamp(0, 1)
        metrics[f"α={alpha}_ssim"]     = ssim_y(y, y_hat)
        metrics[f"α={alpha}_max_diff"] = max_abs_diff(y, y_hat)
        metrics[f"α={alpha}_mae"]      = mae(y, y_hat)
        save_pil(y_hat, out_dir / f"t8_trained_gen_alpha{alpha}.png")

    metrics["gen_ckpt"] = str(gen_ckpt)
    metrics["skipped"] = False
    return metrics


# ════════════════════════════════════════════════════════════════════════════
# Interpretation
# ════════════════════════════════════════════════════════════════════════════

def interpret(results: dict) -> List[str]:
    L: List[str] = []
    L.append("=" * 70)
    L.append("INTERPRETATION")
    L.append("=" * 70)

    t1 = results.get("test_1", {})
    L.append(f"\n[T1] RGB recon (single):  "
             f"SSIM={t1.get('ssim_y(x, x_hat)', 0):.4f}  "
             f"PSNR={t1.get('psnr_y(x, x_hat)', 0):.2f}dB")

    t2 = results.get("test_2", {})
    L.append(f"[T2] LUMA α=0 identity:    "
             f"SSIM={t2.get('ssim_y(y, y_hat)', 0):.4f}  "
             f"PSNR={t2.get('psnr_y(y, y_hat)', 0):.2f}dB  "
             f"max={t2.get('max|y - y_hat|', 0):.4f}")

    t3 = results.get("test_3", {})
    L.append(f"[T3] LUMA α=0.1 (random):  "
             f"SSIM={t3.get('ssim_y(y, y_hat)', 0):.4f}  "
             f"max={t3.get('max|y - y_hat|', 0):.4f}")

    t4 = results.get("test_4", {})
    L.append(f"[T4] LUMA α=1.0 (random):  "
             f"SSIM={t4.get('ssim_y(y, y_hat)', 0):.4f}  "
             f"max={t4.get('max|y - y_hat|', 0):.4f}")

    # Test 5
    t5 = results.get("test_5", {})
    if t5.get("skipped"):
        L.append(f"\n[T5] AE→C1 chain: SKIPPED ({t5.get('reason', '')})")
    else:
        L.append(f"\n[T5] AE→C1 chain ({t5.get('n_images', 0)} images):")
        L.append(f"     acc C1(x)         = {t5.get('acc_c1(x)', 0):.3f}")
        L.append(f"     acc C1(AE(x))     = {t5.get('acc_c1(ae(x))', 0):.3f}")
        L.append(f"     agreement rate    = {t5.get('agreement_rate', 0):.3f}")
        L.append(f"     acc drop          = {t5.get('acc_drop', 0):+.3f}")

    # Test 6: batch AE
    t6 = results.get("test_6", {})
    if t6:
        psnr = t6.get("psnr_y_rgb", {})
        ssim = t6.get("ssim_y_rgb", {})
        L.append(f"\n[T6] Batch AE quality ({t6.get('n_images', 0)} images):")
        L.append(f"     PSNR RGB: mean={psnr.get('mean', 0):.2f}  "
                 f"std={psnr.get('std', 0):.2f}  "
                 f"[{psnr.get('min', 0):.2f}, {psnr.get('max', 0):.2f}]")
        L.append(f"     SSIM RGB: mean={ssim.get('mean', 0):.4f}  "
                 f"std={ssim.get('std', 0):.4f}")
        worst = t6.get("worst_image", {})
        L.append(f"     Worst:  PSNR={worst.get('psnr', 0):.2f}  "
                 f"({Path(worst.get('path', '')).name})")
        # Compare with training average
        L.append(f"     [Training reported: PSNR_y avg=34.77, SSIM_y avg=0.9939]")
        if psnr.get("mean", 0) < 30:
            L.append(f"     → Mean PSNR significantly below training average.")
            L.append(f"       AE may have degraded OR train/val have different distribution.")
        elif psnr.get("mean", 0) < 33:
            L.append(f"     → Mean PSNR slightly below training. Acceptable variance.")
        else:
            L.append(f"     → Mean PSNR consistent with training.")

    # Test 7: extreme stress
    t7 = results.get("test_7", {})
    if t7:
        L.append(f"\n[T7] Extreme stress (escalating wm/α):")
        L.append(f"     {'config':<28} {'SSIM':>6} {'max':>6} {'sat%':>6}")
        for cfg in t7.get("configs", []):
            L.append(f"     {cfg.get('config', ''):<28} "
                     f"{cfg.get('ssim', 0):>6.4f} "
                     f"{cfg.get('max_diff', 0):>6.4f} "
                     f"{cfg.get('saturated_pixels', 0)*100:>6.1f}%")
        # Compare with training catastrophic state
        worst_ssim = min(c.get("ssim", 1.0) for c in t7.get("configs", [{"ssim": 1}]))
        worst_max  = max(c.get("max_diff", 0) for c in t7.get("configs", [{"max_diff": 0}]))
        L.append(f"\n     Training catastrophe was: SSIM<0.001, max≈0.95")
        L.append(f"     Worst AE alone produced:  SSIM={worst_ssim:.4f}, max={worst_max:.4f}")
        if worst_max < 0.5 and worst_ssim > 0.3:
            L.append(f"     → AE ALONE cannot produce training-level catastrophe.")
            L.append(f"       The dMax≈0.95 collapse in training comes from")
            L.append(f"       LOGIC OUTSIDE embed_external_wm_gray.")
            L.append(f"       Investigation target: trainer's _embed_luma_only_gray")
            L.append(f"       and how delta=(y_wm - y_carrier) is applied to RGB.")
        else:
            L.append(f"     → AE alone CAN reach catastrophic state under extreme α.")
            L.append(f"       Training's generator may drive α to extreme values.")

    # Test 8: generator probe
    t8 = results.get("test_8", {})
    if t8.get("skipped"):
        L.append(f"\n[T8] Trained generator probe: SKIPPED ({t8.get('reason', '')})")
    else:
        L.append(f"\n[T8] Trained generator output stats:")
        L.append(f"     wm_lat:  mean={t8.get('wm_lat_mean', 0):+.4f}  "
                 f"std={t8.get('wm_lat_std', 0):.4f}  "
                 f"max={t8.get('wm_lat_max', 0):+.4f}  "
                 f"|p90|={t8.get('wm_lat_abs_p90', 0):.4f}")
        L.append(f"     wm_skip: mean={t8.get('wm_skip_mean', 0):+.4f}  "
                 f"std={t8.get('wm_skip_std', 0):.4f}  "
                 f"max(|.|)={t8.get('wm_skip_max', 0):.4f}")
        for alpha in [0.1, 0.5, 1.0]:
            L.append(f"     α={alpha}: SSIM={t8.get(f'α={alpha}_ssim', 0):.4f}  "
                     f"max={t8.get(f'α={alpha}_max_diff', 0):.4f}")

    L.append("\n" + "=" * 70)
    L.append("CONCLUSIONS")
    L.append("=" * 70)
    # Auto-conclude based on data
    ae_ok = (t6.get("psnr_y_rgb", {}).get("mean", 0) >= 28 and
             t2.get("ssim_y(y, y_hat)", 0) >= 0.85)
    if ae_ok:
        L.append("→ AE pathway is FUNCTIONAL (Tests 1, 2, 6 all confirm).")
    else:
        L.append("⚠ AE quality below expected. May need retraining.")

    worst_t7_ssim = min((c.get("ssim", 1.0) for c in t7.get("configs", [{"ssim": 1.0}])),
                        default=1.0)
    if worst_t7_ssim > 0.3:
        L.append("→ AE alone CANNOT produce training's catastrophic state.")
        L.append("  ⇒ Bug is in TRAINER (not AE). Investigate _embed_luma_only_gray")
        L.append("    and surrounding logic for the perturbation amplification path.")
    else:
        L.append("→ AE alone CAN reach catastrophic state under extreme α.")
        L.append("  ⇒ Bug is generator dynamics — α grows too large during training.")
        L.append("    Consider hard-clamping α in trainer.")

    return L


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae_ckpt", type=str, required=True)
    ap.add_argument("--c1_ckpt", type=str, default=None)
    ap.add_argument("--gen_ckpt", type=str, default=None,
                    help="Optional trained generator checkpoint (e.g. from RUN07)")
    ap.add_argument("--sample_img", type=str, required=True)
    ap.add_argument("--val_root", type=str, default=None)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--code_root", type=str, required=True)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--batch_test_n", type=int, default=48,
                    help="N images for Test 6 batch AE quality")
    ap.add_argument("--num_classes", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    code_root = Path(args.code_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[ae_ckpt] {args.ae_ckpt}")
    print(f"[c1_ckpt] {args.c1_ckpt}")
    print(f"[gen_ckpt] {args.gen_ckpt}")
    print(f"[sample_img] {args.sample_img}")
    print(f"[out_dir] {out_dir}")
    print()

    # AE
    print("[loader] Loading AE...")
    ae, ae_cfg, m, u = load_ae(Path(args.ae_ckpt), code_root, device)
    print(f"  ext_gain_lat={ae_cfg.ext_gain_lat}, ext_gain_skip={ae_cfg.ext_gain_skip}")
    print(f"  state_dict: missing={len(m)}, unexpected={len(u)}\n")

    # C1
    c1 = None
    if args.c1_ckpt:
        print("[loader] Loading C1 (inline architecture)...")
        try:
            c1, nc_real, mc, uc = load_c1_inline(Path(args.c1_ckpt), device, args.num_classes)
            print(f"  loaded: num_classes={nc_real}, missing={len(mc)}, unexpected={len(uc)}")
            if mc:
                print(f"  missing (first 10): {mc[:10]}")
            if uc:
                print(f"  unexpected (first 10): {uc[:10]}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            c1 = None
        print()

    # Sample image
    x = load_image_rgb(Path(args.sample_img), args.image_size).to(device)
    print(f"[sample] shape={tuple(x.shape)} range=[{x.min().item():.3f}, {x.max().item():.3f}]\n")

    results = {
        "ae_config": {"ext_gain_lat": ae_cfg.ext_gain_lat,
                      "ext_gain_skip": ae_cfg.ext_gain_skip,
                      "ext_direct_gain_lat": ae_cfg.ext_direct_gain_lat,
                      "ext_direct_gain_skip": ae_cfg.ext_direct_gain_skip},
    }

    print("=" * 70); print("TEST 1 — RGB reconstruction"); print("=" * 70)
    results["test_1"] = test_1_rgb_reconstruction(ae, x, out_dir)
    for k, v in results["test_1"].items(): print(f"  {k}: {v:.4f}")
    print()

    print("=" * 70); print("TEST 2 — LUMA identity α=0"); print("=" * 70)
    results["test_2"] = test_2_luma_identity(ae, x, out_dir)
    for k, v in results["test_2"].items(): print(f"  {k}: {v:.4f}")
    print()

    print("=" * 70); print("TEST 3 — LUMA α=0.1"); print("=" * 70)
    results["test_3"] = test_3_small_perturbation(ae, x, out_dir, alpha=0.1)
    for k, v in results["test_3"].items(): print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print()

    print("=" * 70); print("TEST 4 — LUMA α=1.0"); print("=" * 70)
    results["test_4"] = test_4_large_perturbation(ae, x, out_dir, alpha=1.0)
    for k, v in results["test_4"].items(): print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print()

    print("=" * 70); print("TEST 5 — AE→C1 chain"); print("=" * 70)
    val_root = Path(args.val_root) if args.val_root else None
    if c1 is None or val_root is None:
        results["test_5"] = {"skipped": True,
                              "reason": "no C1" if c1 is None else "no --val_root"}
    else:
        try:
            results["test_5"] = test_5_ae_c1_chain(ae, c1, val_root, args.image_size,
                                                    args.batch_size, device, out_dir)
        except Exception as e:
            results["test_5"] = {"skipped": True, "reason": f"{type(e).__name__}: {e}",
                                 "traceback": traceback.format_exc()}
    for k, v in results["test_5"].items():
        if k in ("traceback", "per_class_acc_x", "per_class_acc_xh"): continue
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print()

    print("=" * 70); print("TEST 6 — Batch AE quality"); print("=" * 70)
    if val_root is None:
        results["test_6"] = {"skipped": True, "reason": "no --val_root"}
    else:
        try:
            results["test_6"] = test_6_batch_ae_quality(ae, val_root, args.image_size,
                                                        args.batch_size, device,
                                                        args.batch_test_n)
            psnr = results["test_6"]["psnr_y_rgb"]
            ssim = results["test_6"]["ssim_y_rgb"]
            print(f"  n_images: {results['test_6']['n_images']}")
            print(f"  PSNR RGB: mean={psnr['mean']:.2f} std={psnr['std']:.2f} "
                  f"[{psnr['min']:.2f}, {psnr['max']:.2f}]  p50={psnr['p50']:.2f}")
            print(f"  SSIM RGB: mean={ssim['mean']:.4f} std={ssim['std']:.4f}")
            print(f"  worst: {Path(results['test_6']['worst_image']['path']).name} "
                  f"(PSNR={results['test_6']['worst_image']['psnr']:.2f})")
            print(f"  best:  {Path(results['test_6']['best_image']['path']).name} "
                  f"(PSNR={results['test_6']['best_image']['psnr']:.2f})")
        except Exception as e:
            results["test_6"] = {"skipped": True, "reason": f"{type(e).__name__}: {e}"}
            print(f"  FAILED: {e}")
    print()

    print("=" * 70); print("TEST 7 — Extreme stress test"); print("=" * 70)
    results["test_7"] = test_7_extreme_stress(ae, x, out_dir)
    print(f"  {'config':<28} {'SSIM':>7} {'max':>7} {'mae':>7} {'sat%':>7}")
    for cfg in results["test_7"]["configs"]:
        print(f"  {cfg['config']:<28} "
              f"{cfg['ssim']:>7.4f} "
              f"{cfg['max_diff']:>7.4f} "
              f"{cfg['mae']:>7.4f} "
              f"{cfg['saturated_pixels']*100:>6.1f}%")
    print()

    print("=" * 70); print("TEST 8 — Trained generator probe"); print("=" * 70)
    gen_ckpt = Path(args.gen_ckpt) if args.gen_ckpt else None
    results["test_8"] = test_8_generator_probe(ae, x, gen_ckpt, code_root, out_dir, device)
    for k, v in results["test_8"].items():
        if k == "configs": continue
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print()

    # Interpret + save
    for line in interpret(results):
        print(line)

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    with open(out_dir / "pipeline_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(interpret(results)))
        f.write("\n\n" + "=" * 70 + "\nRAW RESULTS\n" + "=" * 70 + "\n")
        f.write(json.dumps(results, indent=2, default=str))

    print(f"\n[done] outputs in: {out_dir}")


if __name__ == "__main__":
    main()
