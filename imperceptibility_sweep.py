#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
imperceptibility_sweep.py — find the invisibility / gate frontier on a trained system.

WHAT IT DOES (phase 1, post-hoc, no retraining)
    Takes an already-dumped orig/wm pair set (delta = wm - orig, the REAL learned
    watermark), and scales delta by a factor alpha < 1 to make the mark weaker /
    less visible. At each alpha it measures THREE axes at once so the whole
    trade-off is visible in one table:

      INVISIBILITY   PSNR, SSIM, LPIPS   between x and x_a = x + alpha*delta
      GATE           GAPg = Acc(x_a) - Acc(x)   through C2 (the whole point)
      FORGERY        CONS-accept on a transplanted mark   through cons_det

    Andrey's target: keep the gate at or above ~35-40 while pushing PSNR/LPIPS as
    high (as invisible) as possible, without crossing the cliff where the gate
    collapses. The sweep shows exactly where that cliff is.

    It also writes side-by-side PNG samples (x | x_a | 10x|delta|) at a few alphas
    so the mark can be judged by eye — the metric is only a guide, the eye decides.

IMPORTANT CAVEAT
    cons_det was trained on the NATIVE mark (alpha=1). At alpha<1 it sees a
    weakened-own mark, so its forgery column is informative about detector
    sensitivity but is NOT a trained operating point — the honest per-alpha
    forgery number comes later from a retrain at the chosen alpha (phase 2).
    The gate and the invisibility metrics ARE exact at every alpha.

USAGE (Machine B, WACV .venv — call by full path so lpips/torchmetrics resolve):
    $WPY = "C:\Users\atytchino.GIGABYTE.000\PycharmProjects\WACV\.venv\Scripts\python.exe"
    cd  C:\Users\atytchino.GIGABYTE.000\PycharmProjects\WACV
    & $WPY imperceptibility_sweep.py --trainer .\20260726-Trainer_MULTIBIT_v15_L2NEG.py `
        --system_ckpt "E:\RUNS\TLD_L2neg_full_s0\checkpoints\wm_system_e008.pth" `
        --c2_ckpt     "E:\RUNS\TLD_L2neg_full_s0\checkpoints\c2_eval_e008.pth" `
        --data        "E:\ATTACK_DATA\TLD_v15full" `
        --alphas      "1.0,0.8,0.7,0.6,0.5,0.4,0.3,0.25,0.2" `
        --sample_alphas "1.0,0.5,0.3" `
        --out_json    "E:\RUNS\EVAL\imperceptibility_TLD_v15full.json" `
        --sample_dir  "E:\RUNS\EVAL\vis_samples_TLD_v15full"
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

# reuse the harness's proven loaders (same directory)
import importlib.util as _ilu


def _load_module(here: Path, name: str):
    cand = here / name
    if not cand.exists():
        raise SystemExit(f"[FATAL] {name} must sit next to this script: {cand} not found")
    spec = _ilu.spec_from_file_location(name.replace(".py", ""), str(cand))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _list_pairs(data_root: Path):
    orig_root, wm_root = data_root / "orig", data_root / "wm"
    if not orig_root.exists() or not wm_root.exists():
        raise SystemExit(f"[FATAL] expected {orig_root} and {wm_root}")
    pairs = []
    for cls_dir in sorted([d for d in orig_root.iterdir() if d.is_dir()]):
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                wm_f = wm_root / cls_dir.name / f.name
                if wm_f.exists():
                    pairs.append((cls_dir.name, f.stem, f, wm_f))
    if not pairs:
        raise SystemExit(f"[FATAL] no matched pairs under {data_root}")
    return pairs


def _load01(path: Path, device):
    from torchvision.io import read_image
    from torchvision.transforms.functional import convert_image_dtype
    t = read_image(str(path))
    t = convert_image_dtype(t, torch.float32)
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    elif t.shape[0] == 4:
        t = t[:3]
    return t.unsqueeze(0).to(device)


# ---------------- invisibility metrics ----------------
def _psnr(x, y):
    mse = torch.mean((x - y) ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
    return (10.0 * torch.log10(1.0 / mse))  # [B], images in [0,1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--system_ckpt", required=True)
    ap.add_argument("--c2_ckpt", required=True)
    ap.add_argument("--data", required=True, help=r"dumped pair set: orig\ and wm\ subdirs")
    ap.add_argument("--alphas", default="1.0,0.8,0.7,0.6,0.5,0.4,0.3,0.25,0.2")
    ap.add_argument("--sample_alphas", default="1.0,0.5,0.3")
    ap.add_argument("--max_images", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_json", default="")
    ap.add_argument("--sample_dir", default="")
    ap.add_argument("--n_samples", type=int, default=8, help="how many example images to dump per sampled alpha")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    EM = _load_module(here, "eval_wm_system.py")

    # PyTorch >= 2.6 defaults torch.load to weights_only=True, but the trainer's own
    # load_system_ckpt() calls torch.load without it and the ckpt stores pathlib paths.
    # eval_wm_system ships the exact compat shim; install it before instantiating.
    try:
        EM._install_torch_load_compat()
    except Exception as _e:
        print(f"[WARN] could not install torch.load compat ({_e}); proceeding")

    TM = EM.load_trainer_module(args.trainer)   # module, not the path string (matches the harness)
    scratch = str(Path(args.out_json).parent / "_sweep_scratch") if args.out_json else str(Path.cwd() / "_sweep_scratch")
    tr, _ = EM.build_trainer(TM, args.system_ckpt, args.c2_ckpt, overrides={"out_root": Path(scratch)})
    for m in (getattr(tr, "ae", None), getattr(tr, "c2", None)):
        try: m.eval()
        except Exception: pass
    device = tr.device
    try:
        classes = list(tr.class_names)
    except Exception:
        classes = sorted({c for c, *_ in _list_pairs(Path(args.data))})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    cons = getattr(tr, "cons_det", None)
    if cons is not None:
        try: cons.eval()
        except Exception: pass
    print(f"[SWEEP] device={device}  cons_det={'yes' if cons is not None else 'no'}  classes={len(classes)}")

    # LPIPS (optional but expected present in the WACV .venv)
    lpips_fn = None
    try:
        import lpips as _lp
        lpips_fn = _lp.LPIPS(net="alex").to(device).eval()
        print("[SWEEP] LPIPS(alex) ready")
    except Exception as e:
        print(f"[SWEEP] LPIPS unavailable ({e}); reporting PSNR + SSIM only")

    # SSIM via torchmetrics (functional)
    ssim_fn = None
    try:
        from torchmetrics.functional import structural_similarity_index_measure as _ssim
        ssim_fn = _ssim
        print("[SWEEP] SSIM (torchmetrics) ready")
    except Exception as e:
        print(f"[SWEEP] SSIM unavailable ({e})")

    rng = random.Random(args.seed)
    pairs = _list_pairs(Path(args.data))
    if args.max_images and args.max_images < len(pairs):
        pairs = rng.sample(pairs, args.max_images)
    N = len(pairs)
    print(f"[DATA] {N} pairs from {args.data}")

    # preload originals, deltas, labels once
    origs, deltas, labels = [], [], []
    for cls, stem, op, wp in pairs:
        x = _load01(op, device)
        w = _load01(wp, device)
        origs.append(x); deltas.append(w - x); labels.append(class_to_idx.get(cls, 0))
    labels_t = torch.tensor(labels, dtype=torch.long)

    # transplant partners (random derangement) for the forgery column
    perm = list(range(N))
    rng.shuffle(perm)
    for i in range(N):
        if perm[i] == i:
            perm[i] = (i + 1) % N

    B = args.batch_size

    @torch.no_grad()
    def gate_acc(imgs01):
        """accuracy of C2 gate over a list of [1,3,H,W] tensors."""
        corr = 0
        for s in range(0, len(imgs01), B):
            xb = torch.cat(imgs01[s:s + B], 0).clamp(0, 1)
            xN = xb * 2 - 1
            try: xN = tr._apply_prod_padding_wipe(xN, None)
            except Exception: pass
            z, _, _ = tr.c2_eval(xN, gate=True)
            lb = labels_t[s:s + len(imgs01[s:s + B])]
            corr += int((z.argmax(1).cpu() == lb).sum())
        return 100.0 * corr / len(imgs01)

    @torch.no_grad()
    def cons_accept(imgs01):
        """fraction cons_det calls native (logit>0). None if no detector."""
        if cons is None:
            return None
        acc = 0
        for s in range(0, len(imgs01), B):
            xb = torch.cat(imgs01[s:s + B], 0).clamp(0, 1)
            logit = cons(xb * 2 - 1).flatten()
            acc += int((logit > 0).sum())
        return 100.0 * acc / len(imgs01)

    @torch.no_grad()
    def invis(imgs_a, imgs_x):
        """mean PSNR/SSIM/LPIPS between scaled and original."""
        ps, ss, lp, n = 0.0, 0.0, 0.0, 0
        for s in range(0, len(imgs_a), B):
            a = torch.cat(imgs_a[s:s + B], 0).clamp(0, 1)
            x = torch.cat(imgs_x[s:s + B], 0).clamp(0, 1)
            ps += float(_psnr(a, x).sum())
            if ssim_fn is not None:
                ss += float(ssim_fn(a, x, data_range=1.0)) * a.shape[0]
            if lpips_fn is not None:
                lp += float(lpips_fn(a * 2 - 1, x * 2 - 1).sum())
            n += a.shape[0]
        return ps / n, (ss / n if ssim_fn is not None else None), (lp / n if lpips_fn is not None else None)

    # baseline: gate on originals (alpha-independent)
    acc_orig = gate_acc(origs)
    print(f"[BASE] gate accuracy on originals = {acc_orig:.2f}%")

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    sample_set = {float(a) for a in args.sample_alphas.split(",") if a.strip()}
    rows = []
    print("\n" + "=" * 92)
    print(f"{'alpha':>6} | {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} | {'gate_acc':>9} {'GAPg':>8} | {'cons_acc':>9} {'forgery':>8}")
    print("-" * 92)
    for a in alphas:
        xa = [ (origs[i] + a * deltas[i]) for i in range(N) ]
        psnr, ssim, lp = invis(xa, origs)
        g_acc = gate_acc(xa)
        gapg = g_acc - acc_orig
        # forgery: transplant this alpha's delta onto a different image, ask cons_det
        forged = [ (origs[i] + a * deltas[perm[i]]) for i in range(N) ] if cons is not None else None
        cons_native = cons_accept(xa)            # should stay high (own mark accepted)
        cons_forge = cons_accept(forged) if forged is not None else None  # should stay low
        rows.append({
            "alpha": a, "psnr": psnr, "ssim": ssim, "lpips": lp,
            "gate_acc": g_acc, "gapg": gapg,
            "cons_accept_native": cons_native, "cons_forgery": cons_forge,
        })
        ss_s = f"{ssim:7.4f}" if ssim is not None else "   n/a "
        lp_s = f"{lp:7.4f}" if lp is not None else "   n/a "
        cn_s = f"{cons_native:8.2f}%" if cons_native is not None else "    n/a "
        cf_s = f"{cons_forge:7.2f}%" if cons_forge is not None else "   n/a "
        print(f"{a:6.2f} | {psnr:7.2f} {ss_s} {lp_s} | {g_acc:8.2f}% {gapg:+8.2f} | {cn_s} {cf_s}")

        # dump eye-check samples at requested alphas
        if a in sample_set and args.sample_dir:
            from torchvision.utils import save_image
            outdir = Path(args.sample_dir); outdir.mkdir(parents=True, exist_ok=True)
            for k in range(min(args.n_samples, N)):
                x = origs[k].clamp(0, 1)
                xw = (origs[k] + a * deltas[k]).clamp(0, 1)
                dd = (10.0 * (a * deltas[k]).abs()).clamp(0, 1)
                strip = torch.cat([x, xw, dd], dim=3)  # side by side
                save_image(strip, str(outdir / f"a{a:.2f}_{pairs[k][1]}.png"))
    print("=" * 92)

    # find the frontier: highest-PSNR alpha whose gate is still >= 40 and >= 35
    def frontier(thr):
        ok = [r for r in rows if r["gapg"] >= thr]
        return max(ok, key=lambda r: r["psnr"]) if ok else None
    for thr in (40.0, 35.0):
        f = frontier(thr)
        if f:
            print(f"[FRONTIER] gate >= {thr:.0f}: best alpha={f['alpha']:.2f}  "
                  f"PSNR={f['psnr']:.2f}  GAPg={f['gapg']:+.2f}"
                  + (f"  LPIPS={f['lpips']:.4f}" if f['lpips'] is not None else ""))
        else:
            print(f"[FRONTIER] gate >= {thr:.0f}: no alpha in the swept range keeps the gate this high")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(
            {"data": str(args.data), "n": N, "acc_orig": acc_orig, "rows": rows}, indent=2), encoding="utf-8")
        print(f"[OUT] wrote {args.out_json}")
    if args.sample_dir:
        print(f"[OUT] eye-check strips (original | scaled | 10x|delta|) in {args.sample_dir}")


if __name__ == "__main__":
    main()
