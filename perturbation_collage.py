#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
perturbation_collage.py  --  publication heat-map collages for the gate-only smooth mark.

For each image it renders a horizontal panel:

    [ ORIGINAL | WATERMARKED | delta+  | delta-  | |delta| ]

  - ORIGINAL / WATERMARKED : the real pair (the smooth content-adaptive mark)
  - delta+  (positive perturbation)  : where the mark ADDS luma  (red, diverging)
  - delta-  (negative perturbation)  : where the mark REMOVES luma (blue, diverging)
  - |delta| (magnitude)              : where the mark is strongest (hot = strong)

Diverging palette by default (blue = negative, white ~ 0, red = positive) so a reviewer
sees at a glance that the perturbation flows around texture (content-adaptive) rather than
sitting on a fixed grid. Writes 10-15 collages PER CLASS (configurable) plus one combined
contact sheet per class.

Reuses the harness loader (build_trainer) and the SAME watermark generation
(synth_variants_nograd), so the collages show exactly the frozen system's mark.

USAGE (Machine B, WACV venv, from the project dir):
    $WPY = "C:\\Users\\atytchino.GIGABYTE.000\\PycharmProjects\\WACV\\.venv\\Scripts\\python.exe"
    cd C:\\Users\\atytchino.GIGABYTE.000\\PycharmProjects\\WACV
    & $WPY perturbation_collage.py --trainer .\\20260728-Trainer_MULTIBIT_v17_TILEGRID.py `
        --system_ckpt "E:\\RUNS\\TLD_smooth_eps010_s0\\checkpoints\\wm_system_e008.pth" `
        --c2_ckpt     "E:\\RUNS\\TLD_smooth_eps010_s0\\checkpoints\\c2_eval_e008.pth" `
        --out "E:\\RUNS\\FIGS\\TLD_collages" --per_class 12

Options:
    --per_class N     collages per class (default 12)
    --amp AMP         delta amplification for VISUALISATION only (default auto per-image);
                      pass a float (e.g. 8) to fix the gain so classes are comparable.
    --gray            grayscale datasets (ORNL): renders single-channel originals correctly.
    --palette {diverging,fire}   heat style (default diverging).
    --contact         also write one contact sheet per class (all its panels stacked).
"""

import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--system_ckpt", required=True)
    ap.add_argument("--c2_ckpt", required=True)
    ap.add_argument("--out", required=True, help="output dir for collages")
    ap.add_argument("--per_class", type=int, default=12)
    ap.add_argument("--amp", type=float, default=0.0,
                    help="fixed delta gain for visualisation (0 = auto per-image)")
    ap.add_argument("--gray", action="store_true")
    ap.add_argument("--palette", choices=["diverging", "fire"], default="diverging")
    ap.add_argument("--contact", action="store_true")
    ap.add_argument("--max_batches", type=int, default=0, help="0 = whole val set")
    args = ap.parse_args()

    # heavy imports here so --help stays fast
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    import importlib.util as ilu

    proj = Path(args.trainer).resolve().parent
    evp = proj / "eval_wm_system.py"
    if not evp.exists():
        sys.exit(f"[FATAL] eval_wm_system.py not found next to trainer at {evp}")
    spec = ilu.spec_from_file_location("eval_wm_system", str(evp))
    EM = ilu.module_from_spec(spec)
    spec.loader.exec_module(EM)

    print("[load] installing torch.load compat + trainer module ...")
    EM._install_torch_load_compat()
    TM = EM.load_trainer_module(str(args.trainer))
    print("[load] rebuilding system from checkpoints ...")
    scratch = proj / "_collage_scratch"
    scratch.mkdir(exist_ok=True)
    tr, _ = EM.build_trainer(TM, str(args.system_ckpt), str(args.c2_ckpt),
                             overrides={"out_root": scratch})
    device = tr.device
    classes = list(getattr(tr, "classes", None)
                   or getattr(getattr(tr, "val_ds", None), "classes", []) or [])
    print(f"[load] READY. device={device} classes={len(classes)}")

    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)

    # ---------- rendering helpers ----------
    def to01_img(t01):
        if t01.dim() == 4:
            t01 = t01[0]
        a = (t01.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
        if a.shape[2] == 1:
            a = np.repeat(a, 3, axis=2)
        return a  # H,W,3 uint8

    def diverging(dn):
        """dn in [-1,1] (H,W) -> RGB uint8, blue=neg, white~0, red=pos."""
        H, W = dn.shape
        rgb = np.ones((H, W, 3), dtype="float32")
        pos = np.clip(dn, 0, 1)
        neg = np.clip(-dn, 0, 1)
        # positive -> toward red: keep R, drop G,B
        rgb[..., 1] -= pos
        rgb[..., 2] -= pos
        # negative -> toward blue: keep B, drop R,G
        rgb[..., 0] -= neg
        rgb[..., 1] -= neg
        return (np.clip(rgb, 0, 1) * 255).astype("uint8")

    def fire(mag):
        """mag in [0,1] (H,W) -> black->red->yellow->white."""
        H, W = mag.shape
        r = np.clip(mag * 3, 0, 1)
        g = np.clip(mag * 3 - 1, 0, 1)
        b = np.clip(mag * 3 - 2, 0, 1)
        return (np.stack([r, g, b], axis=2) * 255).astype("uint8")

    def signed_pos(dn):
        return diverging(np.clip(dn, 0, 1))   # only positive part shown, on white

    def signed_neg(dn):
        return diverging(np.clip(dn, -1, 0))  # only negative part shown, on white

    def label_strip(width, texts, cell_w, h=22):
        img = Image.new("RGB", (width, h), (0, 0, 0))
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("consola.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        for i, t in enumerate(texts):
            d.text((i * cell_w + 6, 3), t, fill=(0, 255, 102), font=font)
        return img

    def _signed_luma(delta_i):
        """[.,C,H,W] or [C,H,W] signed residual -> H,W signed luma-ish numpy."""
        d = delta_i
        if d.dim() == 4:
            d = d[0]
        d = d.detach().cpu().numpy()
        return d.mean(axis=0)                     # H,W signed

    def _colorbar(width, height=24):
        """Horizontal -1..+1 diverging legend strip with tick labels."""
        # gradient row
        grad = np.linspace(-1, 1, width, dtype="float32")[None, :].repeat(height, axis=0)
        if args.palette == "fire":
            bar = fire(np.abs(grad))
        else:
            bar = diverging(grad)
        img = Image.fromarray(bar, "RGB")
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("consola.ttf", 12)
        except Exception:
            font = ImageFont.load_default()
        # tick labels: -1 (left), 0 (mid), +1 (right)
        d.text((4, height // 2 - 7), "-1", fill=(255, 255, 255), font=font)
        d.text((width // 2 - 6, height // 2 - 7), "0", fill=(0, 0, 0), font=font)
        d.text((width - 20, height // 2 - 7), "+1", fill=(255, 255, 255), font=font)
        return img

    def make_panel(x01_i, lat01_i, skip01_i, both01_i):
        """
        One horizontal panel showing the SEPARATE contribution of each gate path,
        the combined net perturbation, AND the LAT-minus-SKIP difference:

          [ ORIGINAL | WATERMARKED | NET (lat+skip) | LAT only | SKIP only | LAT - SKIP ]

        - NET       = (both - orig): the single combined perturbation image the user
          reads (one diverging map; blue = mark REMOVED luma, red = ADDED).
        - LAT only  = (lat_only  - orig): the LATENT path's contribution alone.
        - SKIP only = (skip_only - orig): the SKIP64 path's contribution alone.
        - LAT-SKIP  = (LAT delta) - (SKIP delta): where the two paths DIFFER. The
          two paths look similar because both are FiLM-conditioned on the same
          content_vec(Z) (so both trace the object), but they are NOT identical —
          this cell shows the residual structure that distinguishes them, and the
          gate-accuracy numbers confirm the functional difference (LAT ~59% vs
          SKIP ~87% gate-acc on TLD).

        NET/LAT/SKIP share ONE normalization (net's max) so they are comparable and
        visibly sum to NET. LAT-SKIP is normalized on its OWN max so the (smaller)
        difference structure is visible rather than washed out. A -1..+1 colorbar
        is drawn under the panel.
        """
        orig = to01_img(x01_i)
        wm = to01_img(both01_i)

        d_net = _signed_luma(both01_i - x01_i)
        d_lat = _signed_luma(lat01_i - x01_i)
        d_skip = _signed_luma(skip01_i - x01_i)
        d_diff = d_lat - d_skip                    # where the two paths differ

        # shared scale for net/lat/skip: fixed --amp if given, else net's own max
        m = args.amp if args.amp > 0 else (float(np.abs(d_net).max()) or 1e-6)
        net_n = np.clip(d_net / m, -1, 1)
        lat_n = np.clip(d_lat / m, -1, 1)
        skip_n = np.clip(d_skip / m, -1, 1)
        # LAT-SKIP on its OWN max so the difference is visible (it's smaller)
        md = float(np.abs(d_diff).max()) or 1e-6
        diff_n = np.clip(d_diff / md, -1, 1)

        if args.palette == "fire":
            net_c = fire(np.abs(net_n)); lat_c = fire(np.abs(lat_n))
            skip_c = fire(np.abs(skip_n)); diff_c = fire(np.abs(diff_n))
        else:
            net_c = diverging(net_n)
            lat_c = diverging(lat_n)
            skip_c = diverging(skip_n)
            diff_c = diverging(diff_n)

        cells = [orig, wm, net_c, lat_c, skip_c, diff_c]
        H, W = orig.shape[:2]
        strip = np.concatenate(cells, axis=1)
        pil = Image.fromarray(strip, "RGB")
        labels = ["ORIGINAL", "WATERMARKED", "NET (lat+skip)",
                  "LAT only", "SKIP only", "LAT - SKIP"]
        lab = label_strip(pil.width, labels, cell_w=W)
        bar = _colorbar(pil.width, height=24)
        out = Image.new("RGB", (pil.width, lab.height + pil.height + bar.height),
                        (0, 0, 0))
        out.paste(lab, (0, 0))
        out.paste(pil, (0, lab.height))
        out.paste(bar, (0, lab.height + pil.height))
        return out

    # ---------- iterate val set, group by class ----------
    per_class_target = int(args.per_class)
    made = {}                       # class -> count
    panels_by_class = {}            # class -> [PIL]
    mb = args.max_batches or None

    print(f"[run] generating up to {per_class_target} collages per class ...")
    for xN, valid_mask, y, paths in EM._iter_val(tr, mb):
        x01 = tr._to01(xN)
        if args.gray:
            x01 = EM.to_gray01(x01)
        syn = tr.synth_variants_nograd(
            x01, valid_mask=valid_mask, epoch=0,
            variants=("base", "lat", "skip", "both"), k_factor=1.0,
            varpercent=False, mode="eval",
        )
        both01 = syn["both01"].clamp(0, 1)
        lat01 = syn["lat01"].clamp(0, 1)
        skip01 = syn["skip01"].clamp(0, 1)
        b = int(x01.shape[0])
        for i in range(b):
            yi = int(y[i].item()) if hasattr(y[i], "item") else int(y[i])
            cname = classes[yi] if (classes and 0 <= yi < len(classes)) else f"class_{yi}"
            if made.get(cname, 0) >= per_class_target:
                continue
            panel = make_panel(x01[i:i+1], lat01[i:i+1], skip01[i:i+1], both01[i:i+1])
            cdir = out_root / cname
            cdir.mkdir(parents=True, exist_ok=True)
            idx = made.get(cname, 0)
            stem = Path(str(paths[i])).stem if (paths is not None and i < len(paths) and paths[i]) else f"img{idx:03d}"
            panel.save(str(cdir / f"{stem}_collage.png"))
            panels_by_class.setdefault(cname, []).append(panel)
            made[cname] = idx + 1
        # stop early if every known class is full
        if classes and all(made.get(c, 0) >= per_class_target for c in classes):
            break

    total = sum(made.values())
    print(f"[done] wrote {total} collages across {len(made)} classes -> {out_root}")
    for c, n in sorted(made.items()):
        print(f"    {c}: {n}")

    # ---------- optional contact sheets ----------
    if args.contact:
        print("[contact] building one contact sheet per class ...")
        for cname, panels in panels_by_class.items():
            if not panels:
                continue
            w = panels[0].width
            gap = 6
            H = sum(p.height for p in panels) + gap * (len(panels) - 1)
            sheet = Image.new("RGB", (w, H), (0, 0, 0))
            yoff = 0
            for p in panels:
                sheet.paste(p, (0, yoff))
                yoff += p.height + gap
            sheet.save(str(out_root / f"{cname}_contact.png"))
            print(f"    contact: {cname} ({len(panels)} panels)")

    print("[all done]")


if __name__ == "__main__":
    main()
