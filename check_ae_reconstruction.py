#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_ae_reconstruction.py — can the TLD-trained AE be reused for AFHQ?

EVAL-ONLY. Trains nothing, touches none of the 4 interlocked files. It imports the
trainer, loads the TLD AE via the trainer's own load_ae(), runs `forward_plain` on a
folder of images, and measures reconstruction with the trainer's own psnr_y_torch /
ssim_y_torch. The number decides whether the AFHQ pipeline needs 3 jobs (AE→C1→Stage-3)
or 2 (reuse this AE → C1→Stage-3).

The gate generators operate in this AE's latent, so if the AE cannot reconstruct AFHQ
its latents are meaningless there and a fresh AFHQ AE is required. Rule of thumb:
SSIM_y >~ 0.95 on AFHQ ≈ as good as its 0.994 on leaves → reuse is plausible;
much lower → train an AFHQ AE.

USAGE
  python check_ae_reconstruction.py ^
    --trainer  "C:\\...\\20260707-Trainer_MULTIBIT_v13.py" ^
    --ae_ckpt  "E:\\AE_TRAINED\\TLD\\ckpts\\ae_best.pth" ^
    --ae_py_path "C:\\Users\\atytchino\\PycharmProjects\\WACV" ^
    --data "E:\\AFHQ\\val" --image_size 512 --max_images 400 ^
    --out "E:\\RUNS\\EVAL\\afhq_ae_recon.json"

Compare against the SAME command pointed at "E:\\TLD\\val" to get an apples-to-apples
reference number on the AE's home domain.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


def P(m=""):
    print(m, flush=True)


def load_trainer(trainer_path):
    p = Path(trainer_path)
    if not p.exists():
        raise SystemExit(f"[FATAL] trainer not found: {p}")
    # torch.load compat (same reason as the eval harness: v13's loader predates the
    # PyTorch 2.6 weights_only flip and the ckpt embeds pathlib paths)
    import pathlib
    try:
        torch.serialization.add_safe_globals(
            [pathlib.WindowsPath, pathlib.PosixPath, pathlib.PurePath, pathlib.Path])
    except Exception:
        pass
    if not getattr(torch.load, "_wm_compat", False):
        _orig = torch.load
        def _load(*a, **k):
            k.setdefault("weights_only", False)
            return _orig(*a, **k)
        _load._wm_compat = True
        torch.load = _load

    sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location("wacv_trainer_ae", str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wacv_trainer_ae"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--ae_ckpt", required=True)
    ap.add_argument("--ae_py_path", required=True)
    ap.add_argument("--ae_module", default="AE_ContentBound")
    ap.add_argument("--ae_class", default="UniversalAutoEncoder")
    ap.add_argument("--data", required=True, help="an ImageFolder root (e.g. E:\\AFHQ\\val)")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--max_images", type=int, default=400)
    ap.add_argument("--gray", action="store_true", help="also measure on grayscale-converted input")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P(f"[ENV] torch {torch.__version__} | device {device}")

    TM = load_trainer(a.trainer)
    ae = TM.load_ae(Path(a.ae_ckpt), a.ae_module, a.ae_class, Path(a.ae_py_path), device)
    ae.eval()
    P(f"[AE] loaded {a.ae_ckpt}")

    # dataloader — reuse torchvision ImageFolder; num_workers=0 (importlib + Windows spawn)
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    tf = transforms.Compose([
        transforms.Resize((a.image_size, a.image_size)),
        transforms.ToTensor(),  # [0,1]
    ])
    ds = datasets.ImageFolder(a.data, transform=tf)
    P(f"[DATA] {a.data}: {len(ds)} images, {len(ds.classes)} classes {ds.classes}")
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0,
                    generator=torch.Generator().manual_seed(1234))

    def to_gray01(x):
        y = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        return y.repeat(1, 3, 1, 1)

    @torch.no_grad()
    def measure(gray):
        n = 0
        psnr_sum = ssim_sum = 0.0
        for x01, _ in dl:
            x01 = x01.to(device)
            if gray:
                x01 = to_gray01(x01)
            recon = ae.forward_plain(x01).clamp(0, 1)
            b = x01.shape[0]
            psnr_sum += float(TM.psnr_y_torch(recon, x01).item()) * b
            ssim_sum += float(TM.ssim_y_torch(recon, x01).item()) * b
            n += b
            if a.max_images and n >= a.max_images:
                break
        return n, psnr_sum / n, ssim_sum / n

    P("\n" + "=" * 66)
    P("AE RECONSTRUCTION on this dataset (higher = AE is at home here)")
    P("=" * 66)
    n, psnr, ssim = measure(gray=False)
    P(f"  colour : {n} images | PSNR_y={psnr:6.2f} dB | SSIM_y={ssim:.4f}")
    out = {"data": a.data, "n": n, "psnr_y": psnr, "ssim_y": ssim}
    if a.gray:
        ng, pg, sg = measure(gray=True)
        P(f"  gray   : {ng} images | PSNR_y={pg:6.2f} dB | SSIM_y={sg:.4f}")
        out["psnr_y_gray"] = pg
        out["ssim_y_gray"] = sg

    P("")
    if ssim >= 0.95:
        P(f"  => SSIM_y {ssim:.4f} ≥ 0.95: the TLD AE reconstructs this domain well.")
        P("     REUSE is plausible — the AFHQ pipeline may need only 2 jobs (C1 → Stage-3).")
        P("     Confirm by eye on a few reconstructions before committing.")
    elif ssim >= 0.90:
        P(f"  => SSIM_y {ssim:.4f} in [0.90,0.95): borderline. Reuse is risky; the gate")
        P("     lives in this latent, so marginal reconstruction may still hurt it.")
        P("     Safer to train an AFHQ AE (3 jobs).")
    else:
        P(f"  => SSIM_y {ssim:.4f} < 0.90: the TLD AE does NOT transfer to this domain.")
        P("     Train a dedicated AFHQ AE — the AFHQ pipeline needs 3 jobs (AE → C1 → Stage-3).")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        P(f"\n[OUT] {a.out}")


if __name__ == "__main__":
    main()
