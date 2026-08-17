#!/usr/bin/env python
"""
make_contrast_dataset.py  —  standalone, self-contained

Creates contrast-enhanced copies of UNMARKED originals, to test whether an
attacker can pass the gate by applying an arbitrary global contrast operation
(i.e. whether the watermark is "just contrast enhancement").

It does NOT touch the trainer, the harness, or any model — pure image processing.

INPUT   : a dump_pairs 'orig' tree   <in>/<class>/*.png   (unmarked originals)
OUTPUT  : <out>/<variant>/<class>/*.png   for each contrast variant

Variants (all mean-preserving where possible, matching what the mark appears
to do — raise std, keep mean):
    hist_eq        global histogram equalization (per-channel on Y)
    clahe          adaptive (CLAHE), clipLimit 2.0, 8x8 tiles
    contrast_1p3   linear contrast x1.3 about the per-image mean
    contrast_1p6   linear contrast x1.6 about the per-image mean

Usage:
    python make_contrast_dataset.py --in  E:\\ATTACK_DATA\\AFHQ_256_smooth\\orig ^
                                    --out E:\\ATTACK_DATA\\AFHQ_contrast

Then point contrast_attack.py at each  <out>\\<variant>  as its --data.
"""
import argparse, os
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False


def _to_float(im):      # PIL RGB -> HxWx3 float [0,1]
    return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0

def _to_pil(a):         # float [0,1] -> PIL
    return Image.fromarray(np.clip(a * 255.0, 0, 255).astype("uint8"), "RGB")


def contrast_linear(a, k):
    """Mean-preserving linear contrast: push pixels away from the per-image mean.
    This is exactly the operation that raises std while keeping mean fixed —
    the thing the AFHQ mark visually resembles."""
    m = a.mean(axis=(0, 1), keepdims=True)
    return np.clip(m + (a - m) * k, 0.0, 1.0)


def hist_eq(a):
    """Global histogram equalization on the luma channel, chroma preserved."""
    if HAVE_CV2:
        bgr = cv2.cvtColor((a * 255).astype("uint8"), cv2.COLOR_RGB2YCrCb)
        bgr[:, :, 0] = cv2.equalizeHist(bgr[:, :, 0])
        out = cv2.cvtColor(bgr, cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255.0
        return out
    # numpy fallback: equalize each channel
    out = np.zeros_like(a)
    for c in range(3):
        ch = (a[:, :, c] * 255).astype("uint8").ravel()
        hist, _ = np.histogram(ch, 256, (0, 256))
        cdf = hist.cumsum(); cdf = 255 * cdf / cdf[-1]
        out[:, :, c] = np.interp(ch, np.arange(256), cdf).reshape(a.shape[:2]) / 255.0
    return np.clip(out, 0, 1)


def clahe(a):
    """CLAHE on luma; requires cv2. Falls back to hist_eq if cv2 missing."""
    if not HAVE_CV2:
        return hist_eq(a)
    ycc = cv2.cvtColor((a * 255).astype("uint8"), cv2.COLOR_RGB2YCrCb)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycc[:, :, 0] = cl.apply(ycc[:, :, 0])
    out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255.0
    return out


VARIANTS = {
    "hist_eq":      hist_eq,
    "clahe":        clahe,
    "contrast_1p3": lambda a: contrast_linear(a, 1.3),
    "contrast_1p6": lambda a: contrast_linear(a, 1.6),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True,
                    help="orig tree: <in>/<class>/*.png (unmarked originals)")
    ap.add_argument("--out", required=True, help="output root")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS.keys()),
                    help="subset of: " + ", ".join(VARIANTS.keys()))
    ap.add_argument("--max_per_class", type=int, default=0,
                    help="0 = all; else cap images per class")
    args = ap.parse_args()

    inp = Path(args.inp); out = Path(args.out)
    if not inp.exists():
        raise SystemExit(f"[ERR] input not found: {inp}")
    if not HAVE_CV2:
        print("[WARN] cv2 not available — CLAHE falls back to hist_eq; "
              "hist_eq uses a numpy fallback. Install opencv-python for exact CLAHE.")

    classes = sorted([d.name for d in inp.iterdir() if d.is_dir()])
    print(f"[DATA] {inp}  classes={classes}")
    print(f"[VARIANTS] {args.variants}")

    total = 0
    for var in args.variants:
        fn = VARIANTS[var]
        for cls in classes:
            src_dir = inp / cls
            dst_dir = out / var / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            files = sorted(src_dir.glob("*.png")) + sorted(src_dir.glob("*.jpg"))
            if args.max_per_class:
                files = files[:args.max_per_class]
            for f in files:
                a = _to_float(Image.open(f))
                b = fn(a)
                _to_pil(b).save(dst_dir / f.name)
                total += 1
            print(f"  {var}/{cls}: {len(files)} images")
    print(f"[DONE] wrote {total} images to {out}")
    print("Next: run contrast_attack.py --data <out>/<variant> for each variant.")


if __name__ == "__main__":
    main()
