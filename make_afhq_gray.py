#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
make_afhq_gray.py — convert the whole AFHQ dataset to grayscale on disk.

WHY on disk (not on-the-fly): so that ALL THREE trainers — train_ae_color.py,
C1_trainer_compatible.py, and the Stage-3 interlocked trainer — run UNCHANGED,
just pointed at the new root. No trainer edits, no risk to the interlocked four.
The gate then genuinely operates on grayscale content end to end (AE latent, C1
judging, C2 gating are all trained on gray), which is what a real "gray split"
requires — not a colour system evaluated in gray.

WHAT it does: walks train/ and val/ under --src, mirrors the class-folder
structure under --dst, and writes each image converted to grayscale but SAVED AS
3-CHANNEL RGB (gray replicated across R=G=B). 3-channel is REQUIRED because:
  - the AE's forward_plain does RGB->luma internally (expects 3ch input);
  - C1 / C2 are 3-channel ResNet34LF_BN;
  - wm_dataset_configs afhq entries are in_channels=3.
A true 1-channel save would break all of them. Gray-replicated-RGB is exactly how
ORNL_LOWMEM is stored ("RGB-replicated grayscale"), so this makes AFHQ-gray match
the ORNL convention the pipeline already handles.

Conversion uses ITU-R BT.601 luma (L = 0.299R + 0.587G + 0.114B), the same
coefficients the AE uses internally, so the gray content is consistent with what
the luma AE would see.

USAGE
  python make_afhq_gray.py --src "E:\AFHQ" --dst "E:\AFHQ_GRAY"
  # then train pointing at E:\AFHQ_GRAY\train and E:\AFHQ_GRAY\val

Idempotent-ish: skips a target file that already exists unless --overwrite.
Verifies counts at the end.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def convert_one(src_path: Path, dst_path: Path, overwrite: bool) -> str:
    if dst_path.exists() and not overwrite:
        return "skip"
    try:
        with Image.open(src_path) as im:
            # 'L' applies BT.601 luma; convert back to RGB replicates it across channels
            gray_rgb = im.convert("L").convert("RGB")
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            # preserve format by extension; JPEG quality high to avoid adding artifacts
            ext = dst_path.suffix.lower()
            if ext in (".jpg", ".jpeg"):
                gray_rgb.save(dst_path, quality=95, subsampling=0)
            else:
                gray_rgb.save(dst_path)
        return "ok"
    except Exception as e:
        print(f"  [ERR] {src_path}: {type(e).__name__}: {e}", flush=True)
        return "err"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help=r"AFHQ root (contains train\ and val\)")
    ap.add_argument("--dst", required=True, type=Path, help=r"output root for the gray copy")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    if not a.src.exists():
        raise SystemExit(f"[FATAL] src not found: {a.src}")

    grand = {"ok": 0, "skip": 0, "err": 0}
    for split in a.splits:
        src_split = a.src / split
        dst_split = a.dst / split
        if not src_split.exists():
            print(f"[WARN] {src_split} does not exist — skipping", flush=True)
            continue
        classes = sorted([d.name for d in src_split.iterdir() if d.is_dir()])
        print(f"\n[{split}] classes: {classes}", flush=True)
        for cls in classes:
            src_cls = src_split / cls
            n = {"ok": 0, "skip": 0, "err": 0}
            files = [f for f in src_cls.iterdir() if f.suffix.lower() in IMG_EXT]
            for i, f in enumerate(files, 1):
                r = convert_one(f, dst_split / cls / f.name, a.overwrite)
                n[r] += 1
                grand[r] += 1
                if i % 500 == 0:
                    print(f"    {split}/{cls}: {i}/{len(files)}", flush=True)
            print(f"  [{split}/{cls}] ok={n['ok']} skip={n['skip']} err={n['err']} "
                  f"(source files={len(files)})", flush=True)

    print(f"\n[DONE] total ok={grand['ok']} skip={grand['skip']} err={grand['err']}", flush=True)

    # sanity: counts match per split/class
    print("\n[VERIFY] source vs dest counts:", flush=True)
    all_ok = True
    for split in a.splits:
        src_split, dst_split = a.src / split, a.dst / split
        if not src_split.exists():
            continue
        for cls in sorted([d.name for d in src_split.iterdir() if d.is_dir()]):
            ns = len([f for f in (src_split / cls).iterdir() if f.suffix.lower() in IMG_EXT])
            dcls = dst_split / cls
            nd = len([f for f in dcls.iterdir() if f.suffix.lower() in IMG_EXT]) if dcls.exists() else 0
            flag = "OK" if ns == nd else "MISMATCH"
            if ns != nd:
                all_ok = False
            print(f"  {split}/{cls}: src={ns} dst={nd} [{flag}]", flush=True)
    print(f"\n[VERIFY] {'ALL COUNTS MATCH' if all_ok else 'COUNT MISMATCH — investigate'}", flush=True)
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
