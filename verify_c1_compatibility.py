#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
C1 / Watermark Trainer Compatibility Verification
===================================================

Standalone script that verifies a C1 checkpoint is compatible with the
watermark trainer's load_c1_classifier function. Run this AFTER training
C1 with C1_trainer_compatible.py and BEFORE launching watermark training,
to catch architecture mismatches early (e.g. vanilla ResNet34 instead of
ResNet34LF_BN with BlurPool).

Verifies:
  1. Checkpoint file exists and loads cleanly.
  2. Required buffer keys present: wm_affine, gate_strength, destructive_strength.
  3. BlurPool buffers present in downsample paths (base.layerN.0.downsample.1.k).
  4. BatchNorm at position 2 of downsample (not position 1 as in vanilla ResNet).
  5. ResNet34LF_BN model builds AND loads state with minimal missing/unexpected.
  6. Forward pass works on synthetic input.
  7. (Optional) Inference on a real image returns valid class probabilities.

Usage:
  # Verify single checkpoint
  python verify_c1_compatibility.py --c1_ckpt "E:\C1_TRAINED\TLD\ckpts\c1_best.pth"

  # Verify both at once
  python verify_c1_compatibility.py `
    --c1_ckpt "E:\C1_TRAINED\TLD\ckpts\c1_best.pth" `
    --c1_ckpt_extra "E:\C1_TRAINED\AFHQ\ckpts\c1_best.pth"

  # With image-level smoke test
  python verify_c1_compatibility.py `
    --c1_ckpt "E:\C1_TRAINED\TLD\ckpts\c1_best.pth" `
    --test_image "E:\TLD\val\Tomato_healthy\some_image.jpg"

Exit code 0 = all checks PASS, 1 = at least one FAIL.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as tv_models
from PIL import Image


# ============================================================================
# Architecture (must match trainer's ResNet34LF_BN exactly, lines 765-859)
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


class ResNet34LF_BN(nn.Module):
    """EXACT copy of trainer's ResNet34LF_BN (lines 765-859)."""

    def __init__(self, num_classes: int, gate_strength: float = 2.10):
        super().__init__()
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

    def forward(self, x, gate=True, gate_target=None, detach_gate=False,
                detach_affine=False, return_raw=False):
        if gate_target is not None:
            raise RuntimeError("gate_target forbidden (label leakage).")

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
# Verification routines
# ============================================================================

# Keys that the watermark trainer's ResNet34LF_BN wrapper REQUIRES.
# Missing any of these means architecture mismatch with vanilla ResNet34
# OR missing wm_head module (forward signature mismatch).
REQUIRED_KEYS = {
    "gate_strength",
    "destructive_strength",
    "wm_affine",  # Parameter shape [num_classes], not scalar buffer
    "wm_head.2.weight",  # Linear(512, 128) inside wm_head Sequential
    "wm_head.2.bias",
    "wm_head.4.weight",  # Linear(128, 1) inside wm_head Sequential
    "wm_head.4.bias",
    "base.layer2.0.downsample.1.k",  # BlurPool buffer
    "base.layer3.0.downsample.1.k",
    "base.layer4.0.downsample.1.k",
    "base.layer2.0.downsample.2.weight",  # BN at position 2 (after BlurPool)
    "base.layer2.0.downsample.2.running_mean",
    "base.layer2.0.downsample.2.running_var",
    "base.layer3.0.downsample.2.weight",
    "base.layer4.0.downsample.2.weight",
}

# Keys that should NOT be present (indicate vanilla ResNet34 = mismatch)
FORBIDDEN_KEYS = {
    "base.layer2.0.downsample.1.weight",  # BN at position 1 = vanilla layout
    "base.layer2.0.downsample.1.running_mean",
}


def verify_checkpoint(ckpt_path: Path, test_image: Optional[Path],
                      device: torch.device) -> bool:
    """Run all 7 checks on one checkpoint. Return True if all pass."""
    print(f"\n{'=' * 70}")
    print(f"Verifying: {ckpt_path}")
    print(f"{'=' * 70}")

    # ── Check 1: file exists & loads ──
    print("\n[CHECK 1] File exists and loads")
    if not ckpt_path.exists():
        print(f"  FAIL — file not found: {ckpt_path}")
        return False
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"  FAIL — torch.load error: {e}")
        return False
    print(f"  OK — loaded ({ckpt_path.stat().st_size / 1e6:.1f} MB)")

    # Extract state_dict
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model" in ckpt:
            sd = ckpt["model"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    # Strip prefixes
    cleaned = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        kk = k
        for pref in ("module.", "_orig_mod."):
            if kk.startswith(pref):
                kk = kk[len(pref):]
        cleaned[kk] = v
    sd = cleaned
    print(f"  OK — state dict has {len(sd)} parameter tensors")

    # Print metadata if available
    meta_keys = ("epoch", "best_val_acc", "num_classes", "arch")
    if isinstance(ckpt, dict):
        meta_strs = []
        for k in meta_keys:
            if k in ckpt:
                v = ckpt[k]
                if isinstance(v, float):
                    meta_strs.append(f"{k}={v:.4f}")
                elif isinstance(v, list):
                    meta_strs.append(f"{k}=[{len(v)} items]")
                else:
                    meta_strs.append(f"{k}={v}")
        if meta_strs:
            print(f"  metadata: {', '.join(meta_strs)}")

    # ── Check 2: required keys present ──
    print("\n[CHECK 2] Required keys (trainer-compatible architecture)")
    missing = [k for k in REQUIRED_KEYS if k not in sd]
    if missing:
        print(f"  FAIL — {len(missing)} required keys missing:")
        for k in missing:
            print(f"    ✗ {k}")
        print(f"  This indicates VANILLA ResNet34 was used instead of "
              f"ResNet34LF_BN with BlurPool.")
        print(f"  Retrain using C1_trainer_compatible.py.")
        return False
    print(f"  OK — all {len(REQUIRED_KEYS)} required keys present")

    # ── Check 3: forbidden keys absent ──
    print("\n[CHECK 3] Forbidden keys (vanilla ResNet34 signature)")
    present_forbidden = [k for k in FORBIDDEN_KEYS if k in sd]
    if present_forbidden:
        print(f"  FAIL — vanilla-architecture keys found:")
        for k in present_forbidden:
            print(f"    ✗ {k}")
        print(f"  This is a VANILLA ResNet34 checkpoint, not BlurPool-augmented.")
        return False
    print(f"  OK — no forbidden vanilla-layout keys present")

    # ── Check 4: infer num_classes ──
    print("\n[CHECK 4] Output dimension (num_classes)")
    fc_key = "base.fc.weight"
    if fc_key not in sd:
        print(f"  FAIL — final classifier weight '{fc_key}' missing")
        return False
    n_classes_state = sd[fc_key].size(0)
    n_classes_ckpt = ckpt.get("num_classes", None) if isinstance(ckpt, dict) else None
    if n_classes_ckpt is not None and n_classes_ckpt != n_classes_state:
        print(f"  WARN — metadata num_classes={n_classes_ckpt} but state has "
              f"{n_classes_state} output classes; using state-derived value")
    print(f"  OK — num_classes = {n_classes_state}")

    # ── Check 5: build model and load state ──
    print(f"\n[CHECK 5] Build ResNet34LF_BN({n_classes_state}) and load state")
    try:
        model = ResNet34LF_BN(num_classes=n_classes_state, gate_strength=2.10).to(device)
    except Exception as e:
        print(f"  FAIL — model construction error: {e}")
        return False

    result = model.load_state_dict(sd, strict=False)
    n_missing = len(result.missing_keys)
    n_unexpected = len(result.unexpected_keys)

    # On the compatible architecture, missing/unexpected should be near zero.
    # Tolerance: a few keys (e.g. from torch.compile or DataParallel) can drift.
    if n_missing > 5 or n_unexpected > 5:
        print(f"  FAIL — too many key mismatches: "
              f"missing={n_missing} unexpected={n_unexpected}")
        if n_missing:
            print(f"  missing (first 10): {result.missing_keys[:10]}")
        if n_unexpected:
            print(f"  unexpected (first 10): {result.unexpected_keys[:10]}")
        return False
    if n_missing > 0 or n_unexpected > 0:
        print(f"  OK (with minor diffs) — missing={n_missing} unexpected={n_unexpected}")
        if n_missing:
            print(f"  missing keys: {result.missing_keys}")
        if n_unexpected:
            print(f"  unexpected keys: {result.unexpected_keys}")
    else:
        print(f"  OK — clean load (0 missing, 0 unexpected)")

    # ── Check 6: forward pass on synthetic input ──
    # CRITICAL: Trainer feeds C1 with xN in range [-1, 1] (not [0, 1]).
    print("\n[CHECK 6] Forward pass test (trainer's calling pattern + input range)")
    model.eval()
    try:
        # Use [-1, 1] range matching trainer's xN = x01 * 2 - 1
        x = torch.rand(2, 3, 512, 512, device=device) * 2.0 - 1.0
        with torch.no_grad():
            # Trainer calls: self.c1(xN, gate=False) and unpacks 3-tuple
            result = model(x, gate=False)
        if not isinstance(result, tuple) or len(result) != 3:
            print(f"  FAIL — expected 3-tuple return, got {type(result).__name__}")
            return False
        logits, wm_logit, x4 = result
        if logits.shape != (2, n_classes_state):
            print(f"  FAIL — expected logits shape [2, {n_classes_state}], "
                  f"got {list(logits.shape)}")
            return False
        if not torch.isfinite(logits).all():
            print(f"  FAIL — logits contain NaN/Inf")
            return False
        if not torch.isfinite(wm_logit).all():
            print(f"  FAIL — wm_logit contains NaN/Inf")
            return False
        probs = F.softmax(logits, dim=1)
        if not torch.allclose(probs.sum(dim=1), torch.ones(2, device=device), atol=1e-4):
            print(f"  FAIL — softmax doesn't sum to 1")
            return False
        print(f"  OK — 3-tuple returned (input range [-1, 1])")
        print(f"     logits   shape={list(logits.shape)} range=[{float(logits.min()):.3f}, {float(logits.max()):.3f}]")
        print(f"     wm_logit shape={list(wm_logit.shape)} range=[{float(wm_logit.min()):.3f}, {float(wm_logit.max()):.3f}]")
        print(f"     x4       shape={list(x4.shape)}")

        # Also test with gate=True (full forward with destructive logic)
        with torch.no_grad():
            logits_g, _, _ = model(x, gate=True)
        if not torch.isfinite(logits_g).all():
            print(f"  FAIL — gate=True forward produces NaN/Inf")
            return False
        print(f"  OK — gate=True forward also produces finite logits")
    except Exception as e:
        print(f"  FAIL — forward error: {type(e).__name__}: {e}")
        return False

    # ── Check 7: optional real image inference ──
    if test_image is not None:
        print(f"\n[CHECK 7] Real image inference: {test_image}")
        if not test_image.exists():
            print(f"  SKIP — image not found")
        else:
            try:
                pil = Image.open(test_image).convert("RGB")
                # Apply same [-1, 1] normalization as trainer uses (xN = x01*2-1)
                tfm = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),  # [0, 1]
                    transforms.Lambda(lambda t: t * 2.0 - 1.0),  # → [-1, 1]
                ])
                x = tfm(pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _, _ = model(x, gate=False)
                    probs = F.softmax(logits, dim=1)[0]
                top_p, top_idx = probs.max(dim=0)
                # Look up class name if available
                cls_name = "?"
                if isinstance(ckpt, dict) and "classes" in ckpt:
                    classes = ckpt["classes"]
                    if 0 <= int(top_idx) < len(classes):
                        cls_name = classes[int(top_idx)]
                print(f"  predicted class index = {int(top_idx)} "
                      f"({cls_name}) confidence = {float(top_p):.4f}")
                if not torch.isfinite(logits).all():
                    print(f"  FAIL — non-finite logits on real image")
                    return False
                print(f"  OK")
            except Exception as e:
                print(f"  FAIL — real-image inference error: {e}")
                return False
    else:
        print("\n[CHECK 7] Real image inference (SKIPPED — no --test_image provided)")

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1_ckpt", required=True, type=Path,
                    help="Path to first c1_best.pth to verify")
    ap.add_argument("--c1_ckpt_extra", type=Path, default=None,
                    help="Optional second checkpoint to verify in one run")
    ap.add_argument("--test_image", type=Path, default=None,
                    help="Optional real image to run inference on")
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] {device}")

    all_ok = True
    targets: List[Path] = [args.c1_ckpt]
    if args.c1_ckpt_extra is not None:
        targets.append(args.c1_ckpt_extra)

    results = {}
    for ck in targets:
        ok = verify_checkpoint(ck, args.test_image, device)
        results[str(ck)] = ok
        all_ok = all_ok and ok

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for path, ok in results.items():
        status = "PASS ✓" if ok else "FAIL ✗"
        print(f"  {status}  {path}")

    if all_ok:
        print(f"\nAll checkpoints are compatible with the watermark trainer.")
        print(f"You can proceed with watermark training.")
        return 0
    else:
        print(f"\nOne or more checkpoints failed compatibility checks.")
        print(f"Retrain failing checkpoints using C1_trainer_compatible.py.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
