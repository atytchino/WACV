#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
measure_latent_diversity.py — how much is there to bind to?

CONTEXT. The overnight eval proved transplant succeeds ~99% and UNI is rejected: the
detector keys on the MAGNITUDE/COHERENCE of a perturbation, not on its binding to a
carrier. A code audit then found there is NO dedicated content-encoder in the trainer
at all — the generators (g_lat/g_64, cond_lat/cond_64) are conditioned on the AE LATENT
and on the message, nothing else. So the ONLY channel through which content can shape δ
is the AE latent. Content-binding is therefore bounded by how DISTINGUISHABLE those
latents are across images: if two images have near-identical latents, their δ must be
near-identical too, and transplant between them cannot be blocked by ANY architecture —
the information simply is not there.

This script measures that ceiling, per dataset, with NO training. It runs each image
through the AE encoder, collects the latent, and reports how spread out the latents are:
  - mean pairwise cosine distance (higher = more distinguishable content)
  - within-class vs between-class separation (does class structure survive in the latent?)
  - effective dimensionality (PCA participation ratio — how many directions the latents
    actually use; a low number means the latents live on a thin manifold = little to bind)

Reading:
  HIGH diversity  -> content-binding is FEASIBLE here; a Level-2/3 redesign has signal
                     to work with. This is the dataset to prototype the redesign on.
  LOW diversity   -> content-binding is near-IMPOSSIBLE here regardless of method; the
                     images are too alike for δ to carry carrier-specific information.
                     Transplant ~100% is expected and is a property of the DATA, not a bug.

Run it on TLD, AFHQ, ORNL and compare. The prediction to test: ORNL (feature-poor,
near-identical 3D-print frames) collapses; AFHQ (cat/dog/wild) is widest; TLD sits
between — and if TLD is closer to ORNL than expected, that alone explains why transplant
was trivial and tells us to prototype any redesign on AFHQ, not TLD.

USAGE
  python measure_latent_diversity.py ^
    --trainer  "C:\\...\\20260707-Trainer_MULTIBIT_v13.py" ^
    --ae_ckpt  "E:\\AE_TRAINED\\TLD\\ckpts\\ae_best.pth" ^
    --ae_py_path "C:\\Users\\atytchino\\PycharmProjects\\WACV" ^
    --data "E:\\TLD\\val" --image_size 512 --per_class 40 ^
    --out "E:\\RUNS\\EVAL\\tld_latent_div.json"

NOTE the AE need NOT be the one trained on this dataset — the point is to measure how
distinguishable the images ARE, and a fixed encoder is a fair common yardstick. Use the
TLD AE for ALL THREE so the comparison is apples-to-apples (same encoder, different data).
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
    spec = importlib.util.spec_from_file_location("wacv_trainer_div", str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wacv_trainer_div"] = mod
    spec.loader.exec_module(mod)
    return mod


def get_latent(ae, x01):
    """Return a flat per-image latent vector. Tries enc(); falls back gracefully.

    The AE's first conv expects ONE channel (it is a luma/grayscale AE), so collapse
    RGB to BT.601 luma first. This matches how the trainer feeds the encoder.
    """
    if x01.shape[1] == 3:
        x01 = 0.299 * x01[:, 0:1] + 0.587 * x01[:, 1:2] + 0.114 * x01[:, 2:3]
    out = None
    for name in ("enc", "encode"):
        fn = getattr(ae, name, None)
        if callable(fn):
            out = fn(x01)
            break
    if out is None:
        raise SystemExit("[FATAL] AE exposes neither enc() nor encode().")
    # enc may return a dict {'latent','s64'} or a tensor
    if isinstance(out, dict):
        z = out.get("latent", None)
        if z is None:
            z = next(iter(out.values()))
    else:
        z = out
    return z.flatten(1)  # [B, D]


def pca_participation_ratio(Z):
    """Effective dimensionality: (sum λ)^2 / sum(λ^2). ~1 = one direction; =D = isotropic."""
    Zc = Z - Z.mean(0, keepdim=True)
    # covariance eigen-spectrum via SVD of centered data
    try:
        s = torch.linalg.svdvals(Zc.float())
    except Exception:
        return float("nan")
    lam = (s ** 2)
    pr = (lam.sum() ** 2) / (lam.pow(2).sum() + 1e-12)
    return float(pr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--ae_ckpt", required=True)
    ap.add_argument("--ae_py_path", required=True)
    ap.add_argument("--ae_module", default="AE_ContentBound")
    ap.add_argument("--ae_class", default="UniversalAutoEncoder")
    ap.add_argument("--data", required=True)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--per_class", type=int, default=40, help="images sampled per class")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TM = load_trainer(a.trainer)
    ae = TM.load_ae(Path(a.ae_ckpt), a.ae_module, a.ae_class, Path(a.ae_py_path), device)
    ae.eval()

    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.Resize((a.image_size, a.image_size)),
                             transforms.ToTensor()])
    ds = datasets.ImageFolder(a.data, transform=tf)
    P(f"[DATA] {a.data}: {len(ds)} images, {len(ds.classes)} classes {ds.classes}")

    # sample per_class images from each class
    from collections import defaultdict
    idx_by_cls = defaultdict(list)
    for i, (_, y) in enumerate(ds.samples):
        idx_by_cls[y].append(i)
    g = torch.Generator().manual_seed(1234)
    picked = []
    labels = []
    for c, idxs in idx_by_cls.items():
        idxs_t = torch.tensor(idxs)
        perm = idxs_t[torch.randperm(len(idxs), generator=g)][:a.per_class]
        picked += perm.tolist()
        labels += [c] * len(perm)
    labels = torch.tensor(labels)

    # encode
    Z = []
    with torch.no_grad():
        for k in range(0, len(picked), 8):
            batch = torch.stack([ds[i][0] for i in picked[k:k + 8]]).to(device)
            z = get_latent(ae, batch).cpu()
            Z.append(z)
    Z = torch.cat(Z, 0)  # [N, D]
    N, D = Z.shape
    P(f"[LATENT] N={N} images | latent dim D={D}")

    # L2-normalize for cosine geometry
    Zn = Z / (Z.norm(dim=1, keepdim=True) + 1e-12)

    # mean pairwise cosine distance (1 - cos sim), sampled to keep it cheap
    m = min(N, 400)
    sel = torch.randperm(N, generator=g)[:m]
    S = Zn[sel]
    cos = S @ S.t()
    iu = torch.triu_indices(m, m, offset=1)
    pair_cos = cos[iu[0], iu[1]]
    mean_cos_dist = float((1 - pair_cos).mean())

    # within- vs between-class cosine distance
    lab = labels[sel]
    same = lab.unsqueeze(0) == lab.unsqueeze(1)
    same_pairs = same[iu[0], iu[1]]
    within = float((1 - pair_cos[same_pairs]).mean()) if same_pairs.any() else float("nan")
    between = float((1 - pair_cos[~same_pairs]).mean()) if (~same_pairs).any() else float("nan")

    pr = pca_participation_ratio(Z)

    P("\n" + "=" * 68)
    P("AE-LATENT DIVERSITY  (how much carrier-specific signal δ could bind to)")
    P("=" * 68)
    P(f"  mean pairwise cosine distance : {mean_cos_dist:.4f}   (higher = more distinguishable)")
    P(f"  within-class  cosine distance : {within:.4f}")
    P(f"  between-class cosine distance : {between:.4f}   (gap = class structure in latent)")
    P(f"  class separation (btw - wth)  : {between - within:+.4f}")
    P(f"  effective dim (PCA part.ratio): {pr:.1f} of {D}   ({100 * pr / D:.1f}% of dims used)")
    P("")
    if mean_cos_dist < 0.05:
        P("  => VERY LOW diversity: images are near-identical in latent space. δ cannot")
        P("     carry carrier-specific info here — transplant ~100% is a DATA property,")
        P("     and NO architecture (Level 2 or 3) can bind content on this dataset.")
    elif mean_cos_dist < 0.15:
        P("  => LOW diversity: content-binding will be hard; a redesign has thin signal.")
    else:
        P("  => HEALTHY diversity: content-binding is FEASIBLE — prototype the redesign here.")

    out = {"data": a.data, "n": N, "latent_dim": D,
           "mean_cos_dist": mean_cos_dist, "within_class": within,
           "between_class": between, "class_sep": between - within,
           "eff_dim": pr, "eff_dim_frac": pr / D}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        P(f"\n[OUT] {a.out}")


if __name__ == "__main__":
    main()
