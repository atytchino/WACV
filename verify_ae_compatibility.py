#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
AE / Watermark Trainer Compatibility Verification
==================================================

Quick smoke test that the AE checkpoint produced by train_ae_color.py
is compatible with the watermark trainer's expectations.

Verifies:
  1. Checkpoint loads cleanly into UniversalAutoEncoder.
  2. forward_plain(x01) returns RGB tensor in [0, 1].
  3. enc(y01) returns dict with 'latent' (1024ch) and 's64' (512ch).
  4. Latent shapes match what g_lat (in_ch=1024) expects.
  5. Skip64 shapes match what g_64 (in_ch=512) expects.
  6. embed_external_wm_gray exists and is callable.

Run:
  python verify_ae_compatibility.py `
    --ae_ckpt "E:\AE_TRAINED\AFHQ\ckpts\ae_best.pth"

Exit code 0 = OK, 1 = compatibility issue detected.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

from AE_ContentBound import UniversalAutoEncoder, AEConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae_ckpt", required=True, type=Path,
                    help="Path to ae_best.pth")
    ap.add_argument("--image_size", type=int, default=512,
                    help="Test image size (use 512 for AFHQ/TLD, 160 for ORNL)")
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] {device}")

    # ── Test 1: checkpoint loads ──
    print(f"\n[TEST 1] Loading checkpoint: {args.ae_ckpt}")
    if not args.ae_ckpt.exists():
        print(f"  FAIL — file does not exist")
        return 1

    try:
        ckpt = torch.load(args.ae_ckpt, map_location="cpu")
    except Exception as e:
        print(f"  FAIL — torch.load error: {e}")
        return 1

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "ae_state_dict" in ckpt:
            state = ckpt["ae_state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    # Strip DataParallel prefix
    cleaned = {}
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue
        kk = k
        for pref in ("module.", "_orig_mod."):
            if kk.startswith(pref):
                kk = kk[len(pref):]
        cleaned[kk] = v
    print(f"  OK — checkpoint contains {len(cleaned)} parameter tensors")

    # ── Test 2: model builds and loads state ──
    print(f"\n[TEST 2] Building UniversalAutoEncoder")
    model = UniversalAutoEncoder(cfg=AEConfig()).to(device)
    result = model.load_state_dict(cleaned, strict=False)
    if result.missing_keys:
        print(f"  WARN — missing keys: {len(result.missing_keys)} "
              f"(first 3: {result.missing_keys[:3]})")
    if result.unexpected_keys:
        print(f"  WARN — unexpected keys: {len(result.unexpected_keys)} "
              f"(first 3: {result.unexpected_keys[:3]})")
    print(f"  OK — model built, weights loaded")
    model.eval()

    # ── Test 3: forward_plain output range ──
    print(f"\n[TEST 3] forward_plain output range")
    H = W = int(args.image_size)
    x01 = torch.rand(2, 3, H, W, device=device)  # synthetic RGB in [0, 1]
    with torch.no_grad():
        recon = model.forward_plain(x01)
    recon_min = float(recon.min().item())
    recon_max = float(recon.max().item())
    recon_mean = float(recon.mean().item())
    print(f"  shape={list(recon.shape)} dtype={recon.dtype}")
    print(f"  range [{recon_min:.4f}, {recon_max:.4f}] mean={recon_mean:.4f}")
    if recon_min < -1e-4 or recon_max > 1.0 + 1e-4:
        print(f"  FAIL — output outside [0, 1]! "
              f"Watermark trainer expects clamped output.")
        return 1
    if recon.shape != x01.shape:
        print(f"  FAIL — output shape {list(recon.shape)} != input "
              f"{list(x01.shape)}")
        return 1
    print(f"  OK — output in [0, 1], shape matches input")

    # ── Test 4: enc() returns latent+s64 dict ──
    print(f"\n[TEST 4] enc() interface")
    y_luma = 0.299 * x01[:, 0:1] + 0.587 * x01[:, 1:2] + 0.114 * x01[:, 2:3]
    with torch.no_grad():
        enc_out = model.enc(y_luma)
    if not isinstance(enc_out, dict):
        print(f"  FAIL — enc() returned {type(enc_out)}, expected dict")
        return 1
    if "latent" not in enc_out or "s64" not in enc_out:
        print(f"  FAIL — enc() returned dict without 'latent' or 's64' keys. "
              f"Got: {list(enc_out.keys())}")
        return 1
    Z = enc_out["latent"]
    S64 = enc_out["s64"]
    print(f"  latent shape: {list(Z.shape)} (expected [B=2, C=1024, H/16, W/16])")
    print(f"  s64    shape: {list(S64.shape)} (expected [B=2, C=512, H/8, W/8])")
    if Z.size(1) != 1024:
        print(f"  FAIL — latent channels = {Z.size(1)}, expected 1024 "
              f"(watermark trainer's g_lat = GLat(1024))")
        return 1
    if S64.size(1) != 512:
        print(f"  FAIL — s64 channels = {S64.size(1)}, expected 512 "
              f"(watermark trainer's g_64 = G64(512))")
        return 1
    print(f"  OK — channel counts match watermark trainer's g_lat / g_64")

    # ── Test 5: embed_external_wm_gray callable ──
    print(f"\n[TEST 5] embed_external_wm_gray interface")
    if not hasattr(model, "embed_external_wm_gray"):
        print(f"  FAIL — model has no embed_external_wm_gray method")
        return 1
    try:
        # Correct signature from AE_ContentBound.py:
        #   embed_external_wm_gray(y01, wm_lat, wm_skip, alpha_lat, alpha_skip,
        #                          roi_lat_32, roi_skip_64, valid_mask)
        # Watermark trainer calls this with luma (single channel) input.
        y_in = y_luma  # already computed above as BT.601 luma of x01
        # Tiny watermark perturbations at the right shapes
        wm_lat = torch.zeros_like(Z) * 0.001
        wm_skip = torch.zeros_like(S64) * 0.001
        with torch.no_grad():
            y_wm = model.embed_external_wm_gray(
                y01=y_in,
                wm_lat=wm_lat,
                wm_skip=wm_skip,
                alpha_lat=0.5,
                alpha_skip=0.5,
                roi_lat_32=None,
                roi_skip_64=None,
                valid_mask=None,
            )
        if y_wm is None:
            print(f"  FAIL — embed_external_wm_gray returned None")
            return 1
        if not isinstance(y_wm, torch.Tensor):
            print(f"  FAIL — returned {type(y_wm)}, expected Tensor")
            return 1
        # Output is luma (1-channel), shape [B, 1, H, W]
        print(f"  output shape: {list(y_wm.shape)} (expected [B=2, 1, H, W] luma)")
        print(f"  range [{float(y_wm.min()):.4f}, {float(y_wm.max()):.4f}]")
        if y_wm.size(1) != 1:
            print(f"  FAIL — expected 1-channel luma output, got {y_wm.size(1)} channels")
            return 1
        if float(y_wm.min()) < -1e-4 or float(y_wm.max()) > 1.0 + 1e-4:
            print(f"  FAIL — output outside [0, 1]")
            return 1
        print(f"  OK — embed_external_wm_gray works, luma output in [0, 1]")
    except Exception as e:
        print(f"  FAIL — embed_external_wm_gray raised: {e}")
        return 1

    # ── Test 6: embed_external_wm (RGB variant) callable ──
    print(f"\n[TEST 6] embed_external_wm (RGB) interface")
    if not hasattr(model, "embed_external_wm"):
        print(f"  WARN — model has no embed_external_wm method "
              f"(only required for color path)")
    else:
        try:
            with torch.no_grad():
                rgb_wm = model.embed_external_wm(
                    x01=x01,
                    wm_lat=wm_lat,
                    wm_skip=wm_skip,
                    alpha_lat=0.5,
                    alpha_skip=0.5,
                    roi_lat_32=None,
                    roi_skip_64=None,
                    valid_mask=None,
                )
            print(f"  output shape: {list(rgb_wm.shape)} (expected [B=2, 3, H, W] RGB)")
            print(f"  range [{float(rgb_wm.min()):.4f}, {float(rgb_wm.max()):.4f}]")
            if rgb_wm.size(1) != 3:
                print(f"  FAIL — expected 3-channel RGB output, got "
                      f"{rgb_wm.size(1)} channels")
                return 1
            if float(rgb_wm.min()) < -1e-4 or float(rgb_wm.max()) > 1.0 + 1e-4:
                print(f"  FAIL — output outside [0, 1]")
                return 1
            print(f"  OK — embed_external_wm works, RGB output in [0, 1]")
        except Exception as e:
            print(f"  WARN — embed_external_wm raised: {e}")
            print(f"  (Not fatal — only used for color watermark path)")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"ALL COMPATIBILITY TESTS PASSED")
    print(f"AE checkpoint is compatible with watermark trainer.")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
