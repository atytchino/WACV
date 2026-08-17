#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
imperceptibility.py  --  PSNR / SSIM / LPIPS of the watermarked vs original images.

Reviewers expect numeric invisibility. This measures, over a dump_pairs folder
(orig/ + wm/), the standard three:
    PSNR  (dB, higher = more invisible; <30 usually visible, 35-40 borderline, >40 near-invisible)
    SSIM  (structural similarity, 1.0 = identical)
    LPIPS (perceptual distance, lower = more similar; needs the lpips package)

Prints per-dataset means +/- std and writes a JSON. Run it on each dataset's
dump_pairs to fill the imperceptibility row of the 3-domain table.

USAGE (WACV venv, project dir):
    & $WPY imperceptibility.py --data "E:\\ATTACK_DATA\\TLD_eps010_smooth" `
        --out "E:\\RUNS\\EVAL\\imperceptibility_TLD.json" [--gray]

Notes:
  - SSIM uses torchmetrics if available, else a small local implementation.
  - LPIPS uses the lpips package if importable; otherwise it is skipped (reported null).
  - --data is the SAME dump_pairs folder the harness uses.
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dump_pairs folder (orig/ + wm/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gray", action="store_true")
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from torchvision.io import read_image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(args.data)
    orig_root = data / "orig"; wm_root = data / "wm"
    if not orig_root.exists() or not wm_root.exists():
        sys.exit(f"[FATAL] expected {orig_root} and {wm_root}")

    # gather matched pairs
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    pairs = []
    for cls_dir in sorted([d for d in orig_root.iterdir() if d.is_dir()]):
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() not in exts:
                continue
            wf = wm_root / cls_dir.name / f.name
            if wf.exists():
                pairs.append((f, wf))
    if args.max_images:
        pairs = pairs[:args.max_images]
    if not pairs:
        sys.exit(f"[FATAL] no matched pairs under {data}")
    print(f"[data] {len(pairs)} pairs")

    def load01(p):
        t = read_image(str(p)).float() / 255.0    # C,H,W
        if t.shape[0] == 4:
            t = t[:3]
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        return t.unsqueeze(0).to(device)           # 1,3,H,W

    # ---- metrics ----
    def psnr(a, b):
        mse = F.mse_loss(a, b, reduction="none").mean(dim=[1, 2, 3])
        return (10 * torch.log10(1.0 / mse.clamp_min(1e-10)))

    # SSIM: try torchmetrics, else local
    ssim_fn = None
    try:
        from torchmetrics.functional import structural_similarity_index_measure as tm_ssim
        def ssim_fn(a, b):
            return torch.stack([tm_ssim(a[i:i+1], b[i:i+1], data_range=1.0) for i in range(a.shape[0])])
        print("[ssim] using torchmetrics")
    except Exception:
        def _gauss(win, sigma, device):
            import math
            xs = torch.arange(win, dtype=torch.float32, device=device) - win // 2
            g = torch.exp(-(xs ** 2) / (2 * sigma ** 2)); g /= g.sum()
            k = torch.outer(g, g)
            return k.view(1, 1, win, win)
        def ssim_fn(a, b, win=11, sigma=1.5):
            ch = a.shape[1]
            k = _gauss(win, sigma, a.device).repeat(ch, 1, 1, 1)
            pad = win // 2
            mu_a = F.conv2d(a, k, padding=pad, groups=ch)
            mu_b = F.conv2d(b, k, padding=pad, groups=ch)
            mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
            sa = F.conv2d(a * a, k, padding=pad, groups=ch) - mu_a2
            sb = F.conv2d(b * b, k, padding=pad, groups=ch) - mu_b2
            sab = F.conv2d(a * b, k, padding=pad, groups=ch) - mu_ab
            C1, C2 = 0.01 ** 2, 0.03 ** 2
            s = ((2 * mu_ab + C1) * (2 * sab + C2)) / ((mu_a2 + mu_b2 + C1) * (sa + sb + C2))
            return s.mean(dim=[1, 2, 3])
        print("[ssim] using local implementation")

    # LPIPS: optional
    lpips_fn = None
    try:
        import lpips as lpips_pkg
        _lp = lpips_pkg.LPIPS(net="alex").to(device)
        def lpips_fn(a, b):
            # lpips expects [-1,1]
            return _lp(a * 2 - 1, b * 2 - 1).flatten()
        print("[lpips] using lpips(alex)")
    except Exception as e:
        print(f"[lpips] unavailable ({e}); skipping LPIPS")

    psnr_all, ssim_all, lpips_all = [], [], []
    B = args.batch_size
    buf_o, buf_w = [], []

    def flush():
        if not buf_o:
            return
        a = torch.cat(buf_o, 0); b = torch.cat(buf_w, 0)
        if args.gray:
            # convert both to gray then back to 3ch for fair metric on the luma mark
            def g(x):
                y = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
                return y.repeat(1, 3, 1, 1)
            a, b = g(a), g(b)
        psnr_all.extend(psnr(a, b).tolist())
        ssim_all.extend(ssim_fn(a, b).tolist())
        if lpips_fn is not None:
            lpips_all.extend(lpips_fn(a, b).tolist())
        buf_o.clear(); buf_w.clear()

    for i, (op, wp) in enumerate(pairs):
        o = load01(op); w = load01(wp)
        # match sizes (dump pairs are same size, but guard)
        if o.shape[-2:] != w.shape[-2:]:
            w = F.interpolate(w, size=o.shape[-2:], mode="bilinear", align_corners=False)
        buf_o.append(o); buf_w.append(w)
        if len(buf_o) >= B:
            flush()
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(pairs)}")
    flush()

    def stat(xs):
        if not xs:
            return None
        import statistics as st
        return {"mean": st.mean(xs), "std": (st.pstdev(xs) if len(xs) > 1 else 0.0),
                "min": min(xs), "max": max(xs), "n": len(xs)}

    res = {
        "dataset": data.name,
        "n_pairs": len(pairs),
        "psnr_db": stat(psnr_all),
        "ssim": stat(ssim_all),
        "lpips": stat(lpips_all),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    print("\n" + "=" * 52)
    print(f"IMPERCEPTIBILITY — {data.name}  ({len(pairs)} pairs)")
    print("=" * 52)
    if res["psnr_db"]:
        print(f"  PSNR  {res['psnr_db']['mean']:6.2f} dB  (+/- {res['psnr_db']['std']:.2f})")
    if res["ssim"]:
        print(f"  SSIM  {res['ssim']['mean']:6.4f}     (+/- {res['ssim']['std']:.4f})")
    if res["lpips"]:
        print(f"  LPIPS {res['lpips']['mean']:6.4f}     (+/- {res['lpips']['std']:.4f})")
    else:
        print("  LPIPS  (unavailable)")
    print(f"\n[out] wrote {args.out}")


if __name__ == "__main__":
    main()
