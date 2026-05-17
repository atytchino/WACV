#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Standalone Watermark Extraction (Inference)
===========================================

Loads a trained decoder checkpoint and extracts the embedded N-bit message
from one or more watermarked images. This script is DELIBERATELY independent
of the watermark trainer and the C2 classifier — it touches ONLY the
StandaloneDecoder weights from a checkpoint produced by the multi-bit
trainer.

Why this script matters (paper context):
  Reviewer 2 of the ECCV submission argued the proposed system was not a
  watermarking system in the traditional sense because there was no
  recoverable signal independent of the trained classifier. This script
  is the operational rebuttal: third parties run it, recover the bits,
  and verify the watermark — no access to C2 or generator weights needed.

Usage:
  # Single image extraction
  python inference_extract.py `
    --decoder_ckpt "E:\RUNS\AFHQ_MULTIBIT_..\checkpoints\wm_system_e012.pth" `
    --image "watermarked_test.jpg" `
    --n_bits 32

  # Batch extraction over a folder, save CSV
  python inference_extract.py `
    --decoder_ckpt "...\wm_system_e012.pth" `
    --image_dir "E:\AFHQ\val\cat" `
    --out_csv "results.csv" `
    --n_bits 32

  # Verify against expected message (returns exit code 0 if match)
  python inference_extract.py `
    --decoder_ckpt "...\wm_system_e012.pth" `
    --image "watermarked_test.jpg" `
    --expected_bits "10110100110010101011000110100110"

  # Visualize residual side-by-side (requires --clean for comparison)
  python inference_extract.py `
    --decoder_ckpt "...\wm_system_e012.pth" `
    --image "watermarked.jpg" `
    --clean "original.jpg" `
    --save_collage "comparison.png"

Output:
  - Per-image: extracted bit string, detection probability, bit accuracy
    if --expected_bits or --ground_truth_csv given.
  - Batch: CSV with columns [filepath, bits, detection_prob, n_bits].
  - Optional: side-by-side collage PNG with (clean | WM | residual x10).

Exit code:
  0 = all extractions successful AND (if expected given) bits match.
  1 = checkpoint / image load error.
  2 = bits mismatch when --expected_bits provided.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from wm_decoder import StandaloneDecoder


# =================================================================
# Decoder loading from various checkpoint formats
# =================================================================

def load_decoder_from_checkpoint(
        ckpt_path: Path,
        n_bits: int = 32,
        in_channels: int = 3,
        arch: str = "resnet34",
        device: torch.device = torch.device("cpu"),
) -> StandaloneDecoder:
    """Load a StandaloneDecoder from a checkpoint produced by the multi-bit trainer.

    Handles two checkpoint formats:
      A) Full system checkpoint (wm_system_eXXX.pth) — extracts 'wm_decoder' key.
      B) Standalone decoder checkpoint (decoder_eXXX.pth) — uses entire state dict.

    Strips DataParallel 'module.' prefix automatically.

    Args:
        ckpt_path: path to .pth file on disk.
        n_bits: number of watermark bits the decoder was trained for.
        in_channels: 1 (grayscale) or 3 (color). For multi-bit trainer this is 3.
        arch: backbone architecture (must match training: 'resnet34' default).
        device: target device.

    Returns:
        StandaloneDecoder in eval mode, weights loaded, on the requested device.

    Raises:
        FileNotFoundError: if checkpoint file is missing.
        KeyError: if neither 'wm_decoder' nor a direct decoder state is found.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[LOAD] checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Determine where the decoder state lives in the checkpoint
    decoder_state = None
    saved_n_bits = None

    if isinstance(ckpt, dict):
        # Format A: full system checkpoint
        if "wm_decoder" in ckpt:
            decoder_state = ckpt["wm_decoder"]
            saved_n_bits = ckpt.get("wm_n_bits", None)
            print(f"[LOAD] format=system_checkpoint key=wm_decoder")
        # Format B: standalone decoder checkpoint
        elif "state_dict" in ckpt and any(
                k.startswith(("backbone", "bit_head", "detect_head"))
                for k in ckpt["state_dict"].keys()
        ):
            decoder_state = ckpt["state_dict"]
            saved_n_bits = ckpt.get("n_bits", None)
            print(f"[LOAD] format=standalone_decoder")
        # Format C: raw state dict (decoder keys at top level)
        elif any(k.startswith(("backbone", "bit_head")) for k in ckpt.keys()):
            decoder_state = ckpt
            print(f"[LOAD] format=raw_state_dict")
        else:
            raise KeyError(
                f"Checkpoint has no 'wm_decoder' key and doesn't look like a "
                f"standalone decoder state. Top-level keys: {list(ckpt.keys())[:20]}"
            )
    else:
        # Direct tensor dict
        decoder_state = ckpt

    # Use bit count from checkpoint if available — overrides --n_bits CLI arg
    if saved_n_bits is not None and saved_n_bits != n_bits:
        print(f"[LOAD] checkpoint declares n_bits={saved_n_bits}, "
              f"overriding CLI value {n_bits}")
        n_bits = int(saved_n_bits)

    # Strip DataParallel 'module.' prefix if present
    cleaned = {}
    for k, v in decoder_state.items():
        if not isinstance(v, torch.Tensor):
            continue
        k_clean = k
        for prefix in ("module.", "_orig_mod."):
            if k_clean.startswith(prefix):
                k_clean = k_clean[len(prefix):]
        cleaned[k_clean] = v

    # Build the decoder with matching config
    decoder = StandaloneDecoder(
        n_bits=n_bits,
        in_channels=in_channels,
        arch=arch,
        hidden_dim=256,
    )

    result = decoder.load_state_dict(cleaned, strict=False)
    if result.missing_keys:
        print(f"[LOAD] WARNING: {len(result.missing_keys)} missing keys "
              f"(first 5: {result.missing_keys[:5]})")
    if result.unexpected_keys:
        print(f"[LOAD] WARNING: {len(result.unexpected_keys)} unexpected keys "
              f"(first 5: {result.unexpected_keys[:5]})")
    n_loaded = len(cleaned) - len(result.unexpected_keys)
    print(f"[LOAD] OK — {n_loaded} parameter tensors loaded, n_bits={n_bits}")

    decoder = decoder.to(device).eval()
    return decoder


# =================================================================
# Image I/O and preprocessing
# =================================================================

def load_image_rgb(image_path: Path, target_size: Optional[Tuple[int, int]] = None
                   ) -> torch.Tensor:
    """Load an image from disk as a [1, 3, H, W] float tensor in [0, 1].

    Always converts to RGB (3 channels) for consistency with decoder input.

    Args:
        image_path: path to JPG/PNG/etc.
        target_size: (H, W) optional resize. If None, native size is used.

    Returns:
        Tensor of shape [1, 3, H, W] in [0, 1].
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        if target_size is not None:
            tfm = transforms.Compose([
                transforms.Resize(target_size,
                                  interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ])
        else:
            tfm = transforms.ToTensor()
        x = tfm(im)  # [3, H, W]
    return x.unsqueeze(0)  # [1, 3, H, W]


def parse_bit_string(s: str) -> torch.Tensor:
    """Parse a bit string like '10110100' into a tensor of {0, 1}.

    Accepts:
      - '10110100' (sequence of 0/1 characters)
      - '10110100,01010101,...' (comma-separated, joined)
      - whitespace-separated ('1011 0100 ...')
    """
    s = s.replace(",", "").replace(" ", "").replace("_", "").strip()
    bad = set(s) - {"0", "1"}
    if bad:
        raise ValueError(f"Bit string contains non-binary chars: {bad}")
    return torch.tensor([int(c) for c in s], dtype=torch.float32)


def format_bit_string(bits: torch.Tensor, group: int = 8) -> str:
    """Format bit tensor as readable string, e.g., '10110100 11001010'."""
    bits = bits.detach().cpu().int().tolist()
    if group <= 0:
        return "".join(str(b) for b in bits)
    groups = []
    for i in range(0, len(bits), group):
        groups.append("".join(str(b) for b in bits[i:i + group]))
    return " ".join(groups)


# =================================================================
# Inference core
# =================================================================

@torch.no_grad()
def extract_bits(decoder: StandaloneDecoder, image: torch.Tensor
                 ) -> Tuple[torch.Tensor, float]:
    """Run decoder on a single image and return predicted bits + detection prob.

    Args:
        decoder: trained StandaloneDecoder.
        image: [1, 3, H, W] tensor in [0, 1].

    Returns:
        bits: [n_bits] tensor in {0, 1}.
        detection_prob: float in [0, 1] from auxiliary detection head.
    """
    image = image.to(next(decoder.parameters()).device)
    bit_logits, detect_logits = decoder.forward_full(image)
    bits = (bit_logits > 0.0).float().squeeze(0)
    detect_prob = torch.sigmoid(detect_logits).squeeze().item()
    return bits, detect_prob


# =================================================================
# Visualization: side-by-side collage
# =================================================================

def make_comparison_collage(
        clean_path: Path,
        wm_path: Path,
        out_path: Path,
        residual_scale: float = 10.0,
        bits: Optional[torch.Tensor] = None,
        detection_prob: Optional[float] = None,
) -> None:
    """Render a (clean | watermarked | residual x10) side-by-side PNG.

    Args:
        clean_path: path to clean / original image.
        wm_path: path to watermarked image.
        out_path: output PNG path.
        residual_scale: amplification factor for visualizing residual.
        bits: optional decoded bit string to annotate.
        detection_prob: optional detection probability to annotate.
    """
    clean_im = Image.open(clean_path).convert("RGB")
    wm_im = Image.open(wm_path).convert("RGB")

    # Resize to match if shapes differ
    if clean_im.size != wm_im.size:
        wm_im = wm_im.resize(clean_im.size, Image.Resampling.BILINEAR)

    W, H = clean_im.size

    # Compute residual
    clean_arr = np.asarray(clean_im, dtype=np.float32) / 255.0
    wm_arr = np.asarray(wm_im, dtype=np.float32) / 255.0
    diff = (wm_arr - clean_arr) * residual_scale
    residual_arr = np.clip(0.5 + diff, 0, 1)
    residual_im = Image.fromarray((residual_arr * 255).astype(np.uint8), mode="RGB")

    # Layout: 3 columns + header + footer
    margin = 12
    header_h = 36
    footer_h = 60 if bits is not None else 0
    canvas_w = 3 * W + 4 * margin
    canvas_h = header_h + H + margin + footer_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(245, 245, 248))
    draw = ImageDraw.Draw(canvas)

    try:
        font_hdr = ImageFont.truetype("arial.ttf", 18)
        font_body = ImageFont.truetype("arial.ttf", 13)
    except (OSError, IOError):
        font_hdr = ImageFont.load_default()
        font_body = ImageFont.load_default()

    # Column headers
    headers = ["Clean (original)", "Watermarked",
               f"Residual x{residual_scale:.0f}"]
    for col, header in enumerate(headers):
        x = margin + col * (W + margin) + W // 2 - 80
        draw.text((x, 8), header, fill=(30, 30, 40), font=font_hdr)

    # Images
    y_img = header_h
    for col, im in enumerate([clean_im, wm_im, residual_im]):
        x = margin + col * (W + margin)
        canvas.paste(im, (x, y_img))

    # Footer: extracted bits + detection
    if bits is not None:
        bits_str = format_bit_string(bits, group=8)
        det_str = (f"det_prob={detection_prob:.3f}" if detection_prob is not None
                   else "")
        text = f"Extracted bits: {bits_str}    {det_str}"
        draw.text((margin, header_h + H + 6), text,
                  fill=(40, 40, 50), font=font_body)

    canvas.save(out_path, format="PNG", optimize=True)
    print(f"[COLLAGE] saved -> {out_path}")


# =================================================================
# Main
# =================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Standalone watermark bit extraction from images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--decoder_ckpt", required=True, type=Path,
                    help="Path to system checkpoint (.pth) containing decoder weights")
    ap.add_argument("--n_bits", type=int, default=32,
                    help="Number of watermark bits (overridden by checkpoint if saved)")
    ap.add_argument("--arch", type=str, default="resnet34",
                    choices=["resnet18", "resnet34"],
                    help="Decoder backbone architecture (must match training)")
    ap.add_argument("--in_channels", type=int, default=3, choices=[1, 3])
    ap.add_argument("--image_size", type=int, default=0,
                    help="Resize images to this square size before extraction. "
                         "0 = use native size.")
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cuda", "cpu"])

    # Single image or batch mode
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--image", type=Path,
                     help="Path to a single watermarked image")
    grp.add_argument("--image_dir", type=Path,
                     help="Folder of watermarked images to process in batch")

    # Optional verification
    ap.add_argument("--expected_bits", type=str, default="",
                    help="Expected bit string for verification "
                         "(e.g., '10110100...'). Exit code 2 if mismatch.")
    ap.add_argument("--ground_truth_csv", type=Path, default=None,
                    help="CSV with columns [filepath, bits] for batch verification")

    # Output options
    ap.add_argument("--out_csv", type=Path, default=None,
                    help="Save batch results as CSV")
    ap.add_argument("--clean", type=Path, default=None,
                    help="Clean image path for side-by-side collage")
    ap.add_argument("--save_collage", type=Path, default=None,
                    help="Save side-by-side collage PNG (requires --clean)")
    ap.add_argument("--residual_scale", type=float, default=10.0)

    # Quiet mode
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-image logging in batch mode")

    args = ap.parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] {device}")

    # Load decoder
    try:
        decoder = load_decoder_from_checkpoint(
            ckpt_path=args.decoder_ckpt,
            n_bits=args.n_bits,
            in_channels=args.in_channels,
            arch=args.arch,
            device=device,
        )
    except (FileNotFoundError, KeyError) as e:
        print(f"[ERROR] {e}")
        return 1

    target_size = (args.image_size, args.image_size) if args.image_size > 0 else None

    # =================== Single-image mode ===================
    if args.image is not None:
        try:
            img = load_image_rgb(args.image, target_size=target_size)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            return 1

        bits, detection_prob = extract_bits(decoder, img)
        bits_str = format_bit_string(bits, group=8)

        print()
        print(f"[RESULT] image       : {args.image}")
        print(f"[RESULT] n_bits      : {bits.numel()}")
        print(f"[RESULT] extracted   : {bits_str}")
        print(f"[RESULT] detection   : prob={detection_prob:.4f} "
              f"({'WATERMARKED' if detection_prob > 0.5 else 'CLEAN'})")

        # Verification against expected bits
        exit_code = 0
        if args.expected_bits:
            try:
                expected = parse_bit_string(args.expected_bits)
            except ValueError as e:
                print(f"[ERROR] {e}")
                return 1
            if expected.numel() != bits.numel():
                print(f"[ERROR] expected_bits length {expected.numel()} "
                      f"!= decoded length {bits.numel()}")
                return 2
            n_correct = int((expected.cpu() == bits.cpu()).sum().item())
            bit_acc = n_correct / expected.numel()
            print(f"[RESULT] expected    : {format_bit_string(expected, group=8)}")
            print(f"[RESULT] bit_accuracy: {bit_acc:.4f} ({n_correct}/{expected.numel()})")
            if n_correct == expected.numel():
                print(f"[RESULT] verdict     : MATCH (all bits correct)")
            else:
                print(f"[RESULT] verdict     : MISMATCH ({expected.numel() - n_correct} "
                      f"bit errors)")
                exit_code = 2

        # Optional side-by-side collage
        if args.save_collage is not None:
            if args.clean is None:
                print(f"[WARN] --save_collage given without --clean; "
                      f"skipping collage")
            else:
                try:
                    make_comparison_collage(
                        clean_path=args.clean,
                        wm_path=args.image,
                        out_path=args.save_collage,
                        residual_scale=args.residual_scale,
                        bits=bits,
                        detection_prob=detection_prob,
                    )
                except Exception as e:
                    print(f"[WARN] collage generation failed: {e}")

        return exit_code

    # =================== Batch mode ===================
    img_dir = args.image_dir
    if not img_dir.exists() or not img_dir.is_dir():
        print(f"[ERROR] image_dir not found or not a directory: {img_dir}")
        return 1

    # Collect images
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images: List[Path] = []
    for p in img_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            images.append(p)
    images.sort()
    if not images:
        print(f"[ERROR] no images found under {img_dir}")
        return 1
    print(f"[BATCH] processing {len(images)} images from {img_dir}")

    # Optional ground-truth CSV for verification
    ground_truth = {}
    if args.ground_truth_csv is not None:
        if not args.ground_truth_csv.exists():
            print(f"[ERROR] ground truth CSV not found: {args.ground_truth_csv}")
            return 1
        with open(args.ground_truth_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ground_truth[row["filepath"]] = row["bits"]
        print(f"[BATCH] loaded ground truth for {len(ground_truth)} images")

    # Process and collect
    results = []
    n_verified = 0
    n_correct_total = 0
    n_bit_total = 0

    for i, img_path in enumerate(images):
        try:
            img = load_image_rgb(img_path, target_size=target_size)
        except Exception as e:
            print(f"[BATCH] [{i + 1}/{len(images)}] FAIL load {img_path.name}: {e}")
            results.append({
                "filepath": str(img_path),
                "bits": "",
                "detection_prob": "",
                "n_bits": 0,
                "error": str(e),
            })
            continue

        bits, detection_prob = extract_bits(decoder, img)
        bits_str = "".join(str(int(b)) for b in bits.cpu().tolist())

        row = {
            "filepath": str(img_path),
            "bits": bits_str,
            "detection_prob": f"{detection_prob:.6f}",
            "n_bits": int(bits.numel()),
            "error": "",
        }

        if ground_truth:
            key_candidates = [str(img_path), img_path.name, str(img_path.relative_to(img_dir))]
            gt_str = None
            for k in key_candidates:
                if k in ground_truth:
                    gt_str = ground_truth[k]
                    break
            if gt_str is not None:
                gt_bits = parse_bit_string(gt_str)
                if gt_bits.numel() == bits.numel():
                    n_correct = int((gt_bits.cpu() == bits.cpu()).sum().item())
                    row["bit_accuracy"] = f"{n_correct / gt_bits.numel():.4f}"
                    n_verified += 1
                    n_correct_total += n_correct
                    n_bit_total += gt_bits.numel()

        results.append(row)

        if not args.quiet:
            print(f"[BATCH] [{i + 1}/{len(images)}] {img_path.name} "
                  f"det={detection_prob:.3f} bits={bits_str[:16]}...")

    # Summary
    print()
    print(f"[SUMMARY] processed {len(results)} images")
    if n_verified > 0:
        avg_bit_acc = n_correct_total / max(1, n_bit_total)
        print(f"[SUMMARY] verified  {n_verified} against ground truth; "
              f"avg bit_accuracy={avg_bit_acc:.4f}")

    # Save CSV
    if args.out_csv is not None:
        fieldnames = ["filepath", "bits", "detection_prob", "n_bits", "error"]
        if any("bit_accuracy" in r for r in results):
            fieldnames.append("bit_accuracy")
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                # Fill missing optional columns
                for fn in fieldnames:
                    r.setdefault(fn, "")
                writer.writerow(r)
        print(f"[SUMMARY] CSV written -> {args.out_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
