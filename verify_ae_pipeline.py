#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
verify_ae_pipeline.py — Diagnostic for UniversalAutoEncoder + watermark pipeline.

Runs 5 systematic tests to isolate whether AE is the source of color watermark
training failures, or whether the bug is elsewhere (loss balance, gain settings,
etc.). After running, you'll have CONCRETE NUMBERS instead of theories.

Tests
-----
1. RGB reconstruction quality      — does ae.forward_plain(x) match x?
2. LUMA identity (alpha=0)         — does embed_external_wm_gray(y, wm=0) ≈ y?
                                     [CRITICAL: tests trainer's exact code path]
3. LUMA small perturbation         — alpha=0.1, magnitude of pixel change?
4. LUMA large perturbation         — alpha=1.0, does decoder collapse?
5. AE→C1 chain accuracy (optional) — does AE-reconstructed image still classify?

Outputs
-------
- Console: structured numerical results for all 5 tests
- File: results.json with all metrics
- Files: visualization images (originals + reconstructions side-by-side)
- File: pipeline_report.txt with interpretation

Usage (PowerShell, single line via backticks)
---------------------------------------------
python verify_ae_pipeline.py `
  --ae_ckpt    "E:\AE_TRAINED\TLD\ckpts\ae_best.pth" `
  --c1_ckpt    "E:\C1_TRAINED\TLD\ckpts\c1_best.pth" `
  --sample_img "E:\TLD\val\Tomato_healthy\839daf12-8d8b-478e-b214-2d5dd0fa509a___GH_HL-Leaf-445_1_JPG.rf.6739906b2ae104a1dab2d3652316defb.jpg" `
  --val_root   "E:\TLD\val" `
  --out_dir    "E:\DIAGNOSTICS\TLD" `
  --code_root  "C:\Users\atytchino\PycharmProjects\WACV" `
  --image_size 512 `
  --batch_size 8

Requirements
------------
- AE_ContentBound.py must be importable from --code_root
- C1 checkpoint must match ResNet34LF_BN architecture (compatible with trainer);
  if it doesn't load, Test 5 is skipped gracefully (Tests 1-4 still run).
- PyTorch ≥ 2.0
- PIL, numpy
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
from PIL import Image


# ════════════════════════════════════════════════════════════════════════════
# Utilities
# ════════════════════════════════════════════════════════════════════════════

def load_image_rgb(path: Path, size: int = 512) -> torch.Tensor:
    """Load image as RGB tensor in [0,1], shape [1,3,size,size]."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        arr = np.array(im, dtype=np.float32) / 255.0  # HxWx3
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return x


def rgb_to_luma01(x01: torch.Tensor) -> torch.Tensor:
    """RGB [0,1] -> luma Y [0,1] via BT.601 (matches AE's rgb_to_ycbcr)."""
    r = x01[:, 0:1]
    g = x01[:, 1:2]
    b = x01[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def psnr_y(a01: torch.Tensor, b01: torch.Tensor) -> float:
    """PSNR on luma channel in dB. Tensors in [0,1]."""
    a_y = rgb_to_luma01(a01) if a01.size(1) == 3 else a01
    b_y = rgb_to_luma01(b01) if b01.size(1) == 3 else b01
    mse = ((a_y - b_y) ** 2).mean().clamp_min(1e-12)
    return float(-10.0 * torch.log10(mse).item())


def _gaussian_kernel2d(window: int, sigma: float, device, dtype) -> torch.Tensor:
    x = torch.arange(window, device=device, dtype=dtype) - (window - 1) / 2.0
    g1 = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    g1 = g1 / g1.sum()
    return (g1[:, None] @ g1[None, :]).unsqueeze(0).unsqueeze(0)


def ssim_y(a01: torch.Tensor, b01: torch.Tensor, window: int = 11, sigma: float = 1.5) -> float:
    """SSIM on luma channel. Tensors in [0,1]."""
    x = rgb_to_luma01(a01) if a01.size(1) == 3 else a01
    y = rgb_to_luma01(b01) if b01.size(1) == 3 else b01
    _, _, H, W = x.shape
    w = int(min(window, H, W))
    if w < 3:
        l1 = (x - y).abs().mean()
        return float((1.0 - l1).clamp(-1, 1).item())
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


def mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().item())


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def save_pil(tensor01: torch.Tensor, path: Path) -> None:
    """Save a [1,C,H,W] tensor in [0,1] to disk as PNG."""
    t = tensor01.detach().cpu().clamp(0, 1)
    if t.size(1) == 1:
        t = t.repeat(1, 3, 1, 1)
    arr = (t[0].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_strip(tensors01: List[torch.Tensor], labels: List[str], path: Path) -> None:
    """Save horizontal strip of images with labels above each."""
    imgs = []
    for t in tensors01:
        t = t.detach().cpu().clamp(0, 1)
        if t.size(1) == 1:
            t = t.repeat(1, 3, 1, 1)
        arr = (t[0].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        imgs.append(arr)
    H = max(im.shape[0] for im in imgs)
    W_total = sum(im.shape[1] for im in imgs) + 10 * (len(imgs) - 1)
    canvas = np.ones((H, W_total, 3), dtype=np.uint8) * 255
    x = 0
    for im in imgs:
        canvas[:im.shape[0], x:x + im.shape[1], :] = im
        x += im.shape[1] + 10
    Image.fromarray(canvas).save(path)
    # also save labels in a sibling .txt file
    txt = path.with_suffix(".labels.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(" | ".join(labels))


# ════════════════════════════════════════════════════════════════════════════
# AE loader
# ════════════════════════════════════════════════════════════════════════════

def load_ae(ae_ckpt: Path, code_root: Path, device: torch.device):
    """Import UniversalAutoEncoder from code_root and load checkpoint."""
    sys.path.insert(0, str(code_root))
    try:
        from AE_ContentBound import UniversalAutoEncoder, AEConfig
    finally:
        if str(code_root) in sys.path:
            sys.path.remove(str(code_root))

    cfg = AEConfig()
    model = UniversalAutoEncoder(cfg=cfg).to(device).eval()

    payload = torch.load(ae_ckpt, map_location=device, weights_only=False)
    sd = payload.get("state_dict") or payload.get("ae_state_dict") or payload
    # strip "module." prefix if any (from DataParallel)
    sd_clean = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith("module.") else k
        sd_clean[nk] = v
    missing, unexpected = model.load_state_dict(sd_clean, strict=False)
    return model, cfg, missing, unexpected


# ════════════════════════════════════════════════════════════════════════════
# C1 loader — best-effort, falls back gracefully if unavailable
# ════════════════════════════════════════════════════════════════════════════

def try_load_c1(c1_ckpt: Optional[Path], code_root: Path, device: torch.device, num_classes: int):
    """
    Try to load C1 from trainer's module by importing it via importlib.
    If that fails for any reason, return None — Test 5 is skipped.
    """
    if c1_ckpt is None or not c1_ckpt.exists():
        return None, "no c1_ckpt provided"

    # Find the trainer file in code_root
    trainer_files = sorted(code_root.glob("*Trainer*MULTIBIT*.py"))
    if not trainer_files:
        return None, f"no Trainer*MULTIBIT*.py found in {code_root}"
    trainer_path = trainer_files[-1]  # most recent by name

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("wm_trainer_mod", trainer_path)
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(code_root))
        try:
            spec.loader.exec_module(mod)
        finally:
            if str(code_root) in sys.path:
                sys.path.remove(str(code_root))
        ResNet34LF_BN = getattr(mod, "ResNet34LF_BN", None)
        if ResNet34LF_BN is None:
            return None, f"ResNet34LF_BN not found in {trainer_path.name}"
        c1 = ResNet34LF_BN(num_classes=num_classes).to(device).eval()
        payload = torch.load(c1_ckpt, map_location=device, weights_only=False)
        sd = payload.get("state_dict") or payload
        sd_clean = {}
        for k, v in sd.items():
            nk = k[7:] if k.startswith("module.") else k
            sd_clean[nk] = v
        missing, unexpected = c1.load_state_dict(sd_clean, strict=False)
        return c1, f"loaded (missing={len(missing)} unexpected={len(unexpected)})"
    except Exception as e:
        return None, f"exception: {type(e).__name__}: {e}"


# ════════════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def test_1_rgb_reconstruction(ae, x01: torch.Tensor, out_dir: Path) -> dict:
    """Test 1: full RGB forward through forward_plain."""
    x_hat = ae.forward_plain(x01).clamp(0, 1)
    metrics = {
        "ssim_y(x, x_hat)": ssim_y(x01, x_hat),
        "psnr_y(x, x_hat)": psnr_y(x01, x_hat),
        "mae(x, x_hat)":    mae(x01, x_hat),
        "max|x - x_hat|":   max_abs_diff(x01, x_hat),
    }
    save_pil(x01, out_dir / "t1_input_rgb.png")
    save_pil(x_hat, out_dir / "t1_recon_rgb.png")
    save_strip([x01, x_hat], ["input RGB", "AE recon RGB"], out_dir / "t1_strip.png")
    return metrics


@torch.no_grad()
def test_2_luma_identity(ae, x01: torch.Tensor, out_dir: Path) -> dict:
    """Test 2 (CRITICAL): luma identity through embed_external_wm_gray(alpha=0)."""
    y = rgb_to_luma01(x01)  # [1,1,H,W]
    # This is THE exact call trainer makes (with zero watermark — should be identity-like)
    y_hat = ae.embed_external_wm_gray(
        y,
        wm_lat=None,
        wm_skip=None,
        alpha_lat=0.0,
        alpha_skip=0.0,
        roi_lat_32=None,
        roi_skip_64=None,
        valid_mask=None,
    ).clamp(0, 1)
    metrics = {
        "ssim_y(y, y_hat)":   ssim_y(y, y_hat),
        "psnr_y(y, y_hat)":   psnr_y(y, y_hat),
        "mae(y, y_hat)":      mae(y, y_hat),
        "max|y - y_hat|":     max_abs_diff(y, y_hat),
    }
    save_pil(y, out_dir / "t2_input_luma.png")
    save_pil(y_hat, out_dir / "t2_recon_luma_alpha0.png")
    save_strip([y, y_hat, (y - y_hat).abs() * 5.0],
               ["input LUMA", "embed_external_wm_gray(α=0)", "|diff| ×5"],
               out_dir / "t2_strip.png")
    return metrics


@torch.no_grad()
def _make_wm(shape, device, scale: float, seed: int) -> torch.Tensor:
    """Generate tanh-bounded random watermark tensor matching generator output style."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(*shape, generator=g)
    raw = raw.to(device=device, dtype=torch.float32) * float(scale)
    return torch.tanh(raw)


@torch.no_grad()
def test_3_small_perturbation(ae, x01: torch.Tensor, out_dir: Path,
                              alpha: float = 0.1, seed: int = 42) -> dict:
    """Test 3: small alpha — should produce subtle perceptible watermark."""
    y = rgb_to_luma01(x01)
    B, _, H, W = y.shape
    # Latent/skip shapes for /16 and /8 downsampling
    H_lat, W_lat = H // 16, W // 16
    H_s64, W_s64 = H // 8, W // 8
    wm_lat = _make_wm((B, 1024, H_lat, W_lat), x01.device, scale=0.5, seed=seed)
    wm_skip = _make_wm((B, 512, H_s64, W_s64), x01.device, scale=0.5, seed=seed + 1)
    y_hat = ae.embed_external_wm_gray(
        y,
        wm_lat=wm_lat,
        wm_skip=wm_skip,
        alpha_lat=alpha,
        alpha_skip=alpha,
    ).clamp(0, 1)
    metrics = {
        "alpha":              alpha,
        "ssim_y(y, y_hat)":   ssim_y(y, y_hat),
        "psnr_y(y, y_hat)":   psnr_y(y, y_hat),
        "mae(y, y_hat)":      mae(y, y_hat),
        "max|y - y_hat|":     max_abs_diff(y, y_hat),
    }
    save_pil(y_hat, out_dir / f"t3_recon_luma_alpha{alpha:.2f}.png")
    save_strip([y, y_hat, (y - y_hat).abs() * 5.0],
               ["input LUMA", f"watermarked (α={alpha})", "|diff| ×5"],
               out_dir / f"t3_strip_alpha{alpha:.2f}.png")
    return metrics


@torch.no_grad()
def test_4_large_perturbation(ae, x01: torch.Tensor, out_dir: Path,
                              alpha: float = 1.0, seed: int = 42) -> dict:
    """Test 4: large alpha — does decoder collapse?"""
    y = rgb_to_luma01(x01)
    B, _, H, W = y.shape
    H_lat, W_lat = H // 16, W // 16
    H_s64, W_s64 = H // 8, W // 8
    wm_lat = _make_wm((B, 1024, H_lat, W_lat), x01.device, scale=0.5, seed=seed)
    wm_skip = _make_wm((B, 512, H_s64, W_s64), x01.device, scale=0.5, seed=seed + 1)
    y_hat = ae.embed_external_wm_gray(
        y,
        wm_lat=wm_lat,
        wm_skip=wm_skip,
        alpha_lat=alpha,
        alpha_skip=alpha,
    ).clamp(0, 1)
    metrics = {
        "alpha":              alpha,
        "ssim_y(y, y_hat)":   ssim_y(y, y_hat),
        "psnr_y(y, y_hat)":   psnr_y(y, y_hat),
        "mae(y, y_hat)":      mae(y, y_hat),
        "max|y - y_hat|":     max_abs_diff(y, y_hat),
    }
    save_pil(y_hat, out_dir / f"t4_recon_luma_alpha{alpha:.2f}.png")
    save_strip([y, y_hat, (y - y_hat).abs() * 2.0],
               ["input LUMA", f"watermarked (α={alpha})", "|diff| ×2"],
               out_dir / f"t4_strip_alpha{alpha:.2f}.png")
    return metrics


@torch.no_grad()
def test_5_ae_c1_chain(ae, c1, val_root: Path, image_size: int,
                       batch_size: int, device: torch.device,
                       out_dir: Path, max_images: int = 64) -> dict:
    """Test 5: AE preserves features C1 needs."""
    if c1 is None:
        return {"skipped": True, "reason": "C1 not loaded"}

    # Build a small val batch by scanning subfolders (class folders)
    class_dirs = sorted([d for d in val_root.iterdir() if d.is_dir()])
    if not class_dirs:
        return {"skipped": True, "reason": f"no class subfolders in {val_root}"}

    items: List[Tuple[Path, int]] = []
    for ci, cd in enumerate(class_dirs):
        files = sorted(
            [p for p in cd.iterdir()
             if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        )
        for p in files[:max(1, max_images // max(1, len(class_dirs)))]:
            items.append((p, ci))
            if len(items) >= max_images:
                break
        if len(items) >= max_images:
            break

    if not items:
        return {"skipped": True, "reason": "no images found"}

    correct_x, correct_xhat = 0, 0
    total = 0
    agree = 0  # both predict same class

    # Process in batches of batch_size
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        x = torch.stack([load_image_rgb(p, image_size)[0] for p, _ in chunk]).to(device)
        y_true = torch.tensor([c for _, c in chunk], device=device, dtype=torch.long)

        # Forward through AE
        x_hat = ae.forward_plain(x).clamp(0, 1)

        # Trainer's normalization: xN = x01 * 2 - 1
        xN = x * 2.0 - 1.0
        xN_hat = x_hat * 2.0 - 1.0

        # C1 forward (gate=False per trainer's c1_logits call pattern)
        out_x = c1(xN, gate=False)
        out_xh = c1(xN_hat, gate=False)
        logits_x = out_x[0] if isinstance(out_x, tuple) else out_x
        logits_xh = out_xh[0] if isinstance(out_xh, tuple) else out_xh
        pred_x = logits_x.argmax(dim=1)
        pred_xh = logits_xh.argmax(dim=1)

        correct_x += int((pred_x == y_true).sum().item())
        correct_xhat += int((pred_xh == y_true).sum().item())
        agree += int((pred_x == pred_xh).sum().item())
        total += len(chunk)

    return {
        "skipped": False,
        "n_images": total,
        "n_classes": len(class_dirs),
        "acc_c1(x)":       correct_x / max(1, total),
        "acc_c1(ae(x))":   correct_xhat / max(1, total),
        "agreement_rate":  agree / max(1, total),
        "acc_drop":        (correct_x - correct_xhat) / max(1, total),
    }


# ════════════════════════════════════════════════════════════════════════════
# Reporting
# ════════════════════════════════════════════════════════════════════════════

def interpret(results: dict) -> List[str]:
    """Human-readable interpretation of results."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("INTERPRETATION")
    lines.append("=" * 70)

    t1 = results.get("test_1", {})
    s1 = t1.get("ssim_y(x, x_hat)", 0.0)
    p1 = t1.get("psnr_y(x, x_hat)", 0.0)
    lines.append(f"\n[Test 1] RGB reconstruction: SSIM={s1:.4f}, PSNR={p1:.2f} dB")
    if s1 >= 0.95 and p1 >= 30:
        lines.append("  ✓ AE's RGB pathway looks healthy.")
    elif s1 >= 0.85:
        lines.append("  ! Mediocre — AE may have degraded since training.")
    else:
        lines.append("  ✗ FAIL — RGB reconstruction broken. Check checkpoint integrity.")

    t2 = results.get("test_2", {})
    s2 = t2.get("ssim_y(y, y_hat)", 0.0)
    p2 = t2.get("psnr_y(y, y_hat)", 0.0)
    mx2 = t2.get("max|y - y_hat|", 1.0)
    lines.append(f"\n[Test 2] LUMA identity (α=0): SSIM={s2:.4f}, "
                 f"PSNR={p2:.2f} dB, max|diff|={mx2:.4f}")
    if s2 >= 0.97 and p2 >= 32:
        lines.append("  ✓ LUMA pathway works correctly with zero watermark.")
        lines.append("    This means AE is NOT the root cause of color failures.")
    elif s2 >= 0.85:
        lines.append("  ! LUMA reconstruction OK but worse than RGB. Possibly")
        lines.append("    AE was trained with chroma helping reconstruction —")
        lines.append("    pure luma path may have artifacts.")
    else:
        lines.append("  ✗ FAIL — LUMA pathway broken even with zero watermark.")
        lines.append("    This is the smoking gun. AE needs investigation/retraining.")

    t3 = results.get("test_3", {})
    s3 = t3.get("ssim_y(y, y_hat)", 0.0)
    mx3 = t3.get("max|y - y_hat|", 1.0)
    lines.append(f"\n[Test 3] LUMA small α=0.1: SSIM={s3:.4f}, max|diff|={mx3:.4f}")
    if s3 >= 0.85 and mx3 <= 0.20:
        lines.append("  ✓ Small watermark produces subtle perturbation as expected.")
    elif s3 >= 0.50:
        lines.append("  ! Even small α produces noticeable change. ext_gain_lat=16 might")
        lines.append("    be too aggressive — consider reducing in AEConfig.")
    else:
        lines.append("  ✗ Small α=0.1 already destroys image. Latent gain WAY too high.")
        lines.append("    REDUCE ext_gain_lat from 16.0 to ~2.0 in AEConfig and retest.")

    t4 = results.get("test_4", {})
    s4 = t4.get("ssim_y(y, y_hat)", 0.0)
    mx4 = t4.get("max|y - y_hat|", 1.0)
    lines.append(f"\n[Test 4] LUMA large α=1.0: SSIM={s4:.4f}, max|diff|={mx4:.4f}")
    if s4 <= 0.20 and mx4 >= 0.40:
        lines.append("  → α=1.0 saturates: AE decoder receives out-of-distribution")
        lines.append("    latent and produces high-amplitude noise. This EXPLAINS")
        lines.append("    why generator with strong gradient drives α toward 1 → cliff.")
        lines.append("    Generator never finds α=0.05-0.20 sweet spot during training.")
    elif s4 >= 0.50:
        lines.append("  ✓ Even α=1.0 preserves structure. Latent gain reasonably scaled.")

    t5 = results.get("test_5", {})
    if t5.get("skipped"):
        lines.append(f"\n[Test 5] AE→C1 chain: SKIPPED ({t5.get('reason', 'unknown')})")
    else:
        a_x = t5.get("acc_c1(x)", 0.0)
        a_xh = t5.get("acc_c1(ae(x))", 0.0)
        ag = t5.get("agreement_rate", 0.0)
        n = t5.get("n_images", 0)
        lines.append(f"\n[Test 5] AE→C1 chain ({n} images):")
        lines.append(f"  acc C1(x)         = {a_x:.3f}")
        lines.append(f"  acc C1(ae(x))     = {a_xh:.3f}")
        lines.append(f"  agreement rate    = {ag:.3f}")
        lines.append(f"  acc drop          = {(a_x - a_xh):.3f}")
        if (a_x - a_xh) <= 0.05 and ag >= 0.85:
            lines.append("  ✓ AE reconstruction preserves C1-relevant features.")
        elif (a_x - a_xh) <= 0.15:
            lines.append("  ! Moderate drop. C1 still works on AE output but loses some.")
        else:
            lines.append("  ✗ Significant accuracy drop. AE degrades classification features.")
            lines.append("    This contributes to C1 guard rail instability in watermark trainer.")

    # Overall recommendation
    lines.append("\n" + "=" * 70)
    lines.append("OVERALL RECOMMENDATION")
    lines.append("=" * 70)
    if s2 < 0.85:
        lines.append("→ AE LUMA pathway is broken. Retrain AE with luma-supervised loss.")
    elif s3 < 0.50 or (s4 < 0.20 and mx4 >= 0.40):
        lines.append("→ AE pathway works, but ext_gain_lat/ext_gain_skip in AEConfig is")
        lines.append("  too aggressive. Try reducing ext_gain_lat from 16.0 to 2.0-4.0")
        lines.append("  (config change in AE_ContentBound.py — NO retrain needed).")
        lines.append("  Then re-run watermark training. This may be the root cause of")
        lines.append("  all color watermark catastrophic failures.")
    else:
        lines.append("→ AE looks fine. Problem is in watermark trainer's loss balance")
        lines.append("  or generator dynamics — not in AE. Next step: stripped diagnostic")
        lines.append("  trainer with minimum loss terms to isolate which loss term")
        lines.append("  destabilizes training.")

    return lines


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae_ckpt", type=str, required=True)
    ap.add_argument("--c1_ckpt", type=str, default=None)
    ap.add_argument("--sample_img", type=str, required=True,
                    help="One sample image for Tests 1-4")
    ap.add_argument("--val_root", type=str, default=None,
                    help="Val root with class subfolders (for Test 5). Optional.")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--code_root", type=str, required=True,
                    help="Folder containing AE_ContentBound.py and trainer")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_classes", type=int, default=10,
                    help="Used for C1 architecture; mismatch is OK with strict=False")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    code_root = Path(args.code_root)
    ae_ckpt = Path(args.ae_ckpt)
    sample_img = Path(args.sample_img)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[ae_ckpt] {ae_ckpt}")
    print(f"[sample_img] {sample_img}")
    print(f"[out_dir] {out_dir}")
    print()

    # ── Load AE ─────────────────────────────────────────────────────────
    print("[loader] Loading AE...")
    ae, ae_cfg, missing, unexpected = load_ae(ae_ckpt, code_root, device)
    print(f"  AEConfig: gn_groups={ae_cfg.gn_groups}, mult={ae_cfg.mult}")
    print(f"  ext_gain_lat={ae_cfg.ext_gain_lat}, ext_gain_skip={ae_cfg.ext_gain_skip}")
    print(f"  state_dict load: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"  unexpected keys (first 5): {unexpected[:5]}")
    print()

    # ── Load sample image ──────────────────────────────────────────────
    print(f"[loader] Loading sample image at {args.image_size}x{args.image_size}...")
    x = load_image_rgb(sample_img, args.image_size).to(device)
    print(f"  shape: {tuple(x.shape)}, dtype: {x.dtype}, range: [{x.min().item():.3f}, {x.max().item():.3f}]")
    print()

    # ── Try to load C1 ─────────────────────────────────────────────────
    c1 = None
    if args.c1_ckpt:
        print("[loader] Trying to load C1...")
        c1, msg = try_load_c1(Path(args.c1_ckpt), code_root, device, args.num_classes)
        print(f"  result: {msg}")
        print()

    # ── Run tests ──────────────────────────────────────────────────────
    results: dict = {
        "ae_ckpt": str(ae_ckpt),
        "sample_img": str(sample_img),
        "image_size": args.image_size,
        "device": str(device),
        "ae_config": {
            "ext_gain_lat": ae_cfg.ext_gain_lat,
            "ext_gain_skip": ae_cfg.ext_gain_skip,
            "ext_direct_gain_lat": ae_cfg.ext_direct_gain_lat,
            "ext_direct_gain_skip": ae_cfg.ext_direct_gain_skip,
            "mult": ae_cfg.mult,
            "gn_groups": ae_cfg.gn_groups,
        },
        "load_status": {
            "ae_missing_keys": len(missing),
            "ae_unexpected_keys": len(unexpected),
        },
    }

    print("=" * 70)
    print("TEST 1 — RGB reconstruction (forward_plain)")
    print("=" * 70)
    t0 = time.time()
    r1 = test_1_rgb_reconstruction(ae, x, out_dir)
    results["test_1"] = r1
    for k, v in r1.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  [time: {time.time() - t0:.2f}s]")
    print()

    print("=" * 70)
    print("TEST 2 — LUMA identity via embed_external_wm_gray(α=0) [CRITICAL]")
    print("=" * 70)
    t0 = time.time()
    r2 = test_2_luma_identity(ae, x, out_dir)
    results["test_2"] = r2
    for k, v in r2.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  [time: {time.time() - t0:.2f}s]")
    print()

    print("=" * 70)
    print("TEST 3 — LUMA small perturbation (α=0.1)")
    print("=" * 70)
    t0 = time.time()
    r3 = test_3_small_perturbation(ae, x, out_dir, alpha=0.1)
    results["test_3"] = r3
    for k, v in r3.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  [time: {time.time() - t0:.2f}s]")
    print()

    print("=" * 70)
    print("TEST 4 — LUMA large perturbation (α=1.0)")
    print("=" * 70)
    t0 = time.time()
    r4 = test_4_large_perturbation(ae, x, out_dir, alpha=1.0)
    results["test_4"] = r4
    for k, v in r4.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  [time: {time.time() - t0:.2f}s]")
    print()

    print("=" * 70)
    print("TEST 5 — AE→C1 chain accuracy")
    print("=" * 70)
    t0 = time.time()
    val_root = Path(args.val_root) if args.val_root else None
    if c1 is None:
        r5 = {"skipped": True, "reason": "C1 not loaded"}
    elif val_root is None:
        r5 = {"skipped": True, "reason": "no --val_root"}
    else:
        try:
            r5 = test_5_ae_c1_chain(ae, c1, val_root, args.image_size,
                                    args.batch_size, device, out_dir)
        except Exception as e:
            r5 = {"skipped": True, "reason": f"exception: {e}",
                  "traceback": traceback.format_exc()}
    results["test_5"] = r5
    for k, v in r5.items():
        if k == "traceback":
            continue
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  [time: {time.time() - t0:.2f}s]")
    print()

    # ── Interpretation ─────────────────────────────────────────────────
    interp_lines = interpret(results)
    for line in interp_lines:
        print(line)

    # ── Save reports ──────────────────────────────────────────────────
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    with open(out_dir / "pipeline_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(interp_lines))
        f.write("\n\n")
        f.write("=" * 70 + "\n")
        f.write("RAW RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(json.dumps(results, indent=2, default=str))

    print(f"\n[done] All outputs saved to: {out_dir}")
    print(f"  - results.json")
    print(f"  - pipeline_report.txt")
    print(f"  - t1_strip.png, t2_strip.png, t3_strip_alpha0.10.png, t4_strip_alpha1.00.png")
    print(f"  - t1_input_rgb.png, t1_recon_rgb.png")
    print(f"  - t2_input_luma.png, t2_recon_luma_alpha0.png")
    print(f"  - t3_recon_luma_alpha0.10.png, t4_recon_luma_alpha1.00.png")


if __name__ == "__main__":
    main()
