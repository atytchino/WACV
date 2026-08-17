#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
attack_harness_extended.py  --  robustness + kNN-forgery battery for the frozen system.

This is the LAST tool. It adds the reviewer-expected ROBUSTNESS axis and the honest
kNN FORGERY stress test on top of the existing forgery harness. It reuses the same
loader and the same judge/cons_judge as attack_harness.py, so numbers are comparable.

Two axes:

  ROBUSTNESS  (does the GENUINE watermark survive normal processing? gate should HOLD)
    We take the real watermarked images (wm = orig + delta) and DEGRADE them, then check
    the gate still classifies them (high gate-acc = robust). Reported as gate-acc RETAINED
    (higher = better), unlike forgery (lower = better).
      jpeg_q90 / q75 / q50      JPEG re-compression at quality 90/75/50
      blur_gauss                Gaussian blur (denoising attack)
      blur_median               median filter (denoising attack)
      noise_gauss               additive Gaussian noise (sigma configurable)
      resize_rt                 downsample to 0.5x and back (resize round-trip)
      crop_pad                  center-crop 87.5% and pad back

  FORGERY  (kNN / nearest-neighbour transplant -- the honest content-binding stress test)
    Instead of a RANDOM partner, lift delta from the visually NEAREST watermarked image
    (nearest in pixel L2 over a downsampled thumbnail). If content-binding is real, even
    the closest foreign delta should NOT pass the blind consistency detector.
      knn_transplant            delta from the nearest-neighbour image (by content)

USAGE (Machine B, WACV venv, from the project dir):
    $WPY = "...\\.venv\\Scripts\\python.exe"
    cd ...\\WACV
    & $WPY attack_harness_extended.py --trainer .\\20260728-Trainer_MULTIBIT_v17_TILEGRID.py `
        --system_ckpt "E:\\RUNS\\TLD_smooth_eps010_s0\\checkpoints\\wm_system_e008.pth" `
        --c2_ckpt     "E:\\RUNS\\TLD_smooth_eps010_s0\\checkpoints\\c2_eval_e008.pth" `
        --data        "E:\\ATTACK_DATA\\TLD_eps010_smooth" `
        --out         "E:\\RUNS\\EVAL\\attack_TLD_extended.json"

Notes:
  - --data must be a dump_pairs folder (orig/<class>/*.png + wm/<class>/*.png), the same
    input attack_harness.py takes. Generate it with eval_wm_system.py --mode dump_pairs.
  - --gray for ORNL (grayscale).
  - --noise_sigma controls the additive-noise strength (default 0.03 in [0,1] luma).
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--system_ckpt", required=True)
    ap.add_argument("--c2_ckpt", required=True)
    ap.add_argument("--data", required=True, help="dump_pairs folder (orig/ + wm/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gray", action="store_true")
    ap.add_argument("--noise_sigma", type=float, default=0.03)
    ap.add_argument("--max_images", type=int, default=0, help="0 = all pairs")
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    import importlib.util as ilu

    proj = Path(args.trainer).resolve().parent

    # --- reuse attack_harness.py's own helpers by importing it as a module ---
    ahp = proj / "attack_harness.py"
    if not ahp.exists():
        sys.exit(f"[FATAL] attack_harness.py not found at {ahp} (needed for shared helpers)")
    spec = ilu.spec_from_file_location("attack_harness", str(ahp))
    AH = ilu.module_from_spec(spec)
    spec.loader.exec_module(AH)

    # eval_wm_system loader (same path the harness uses)
    EM = AH._load_eval_module(proj) if hasattr(AH, "_load_eval_module") else None
    if EM is None:
        evp = proj / "eval_wm_system.py"
        spec2 = ilu.spec_from_file_location("eval_wm_system", str(evp))
        EM = ilu.module_from_spec(spec2); spec2.loader.exec_module(EM)

    print("[load] building system from checkpoints ...")
    EM._install_torch_load_compat()
    TM = EM.load_trainer_module(str(args.trainer))
    scratch = proj / "_ext_scratch"; scratch.mkdir(exist_ok=True)
    tr, _ = EM.build_trainer(TM, str(args.system_ckpt), str(args.c2_ckpt),
                             overrides={"out_root": scratch})
    device = tr.device
    print(f"[load] READY device={device}")

    # --- load the orig/wm pairs exactly like the base harness ---
    data_root = Path(args.data)
    pairs = AH._list_pairs(data_root)  # list of (orig_path, wm_path, class_name)
    if args.max_images:
        pairs = pairs[:args.max_images]
    if not pairs:
        sys.exit(f"[FATAL] no orig/wm pairs under {data_root}")
    classes = sorted({c for c, _s, _o, _w in pairs})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"[data] {len(pairs)} pairs, {len(classes)} classes")

    def load01(p):
        return AH._load01(Path(p), device)  # 1,3,H,W in [0,1]

    origs, wms, deltas, labels, clss = [], [], [], [], []
    for c, _stem, op, wp in pairs:
        o = load01(op); w = load01(wp)
        if args.gray:
            o = EM.to_gray01(o); w = EM.to_gray01(w)
        origs.append(o); wms.append(w)
        deltas.append((w - o))
        labels.append(class_to_idx[c]); clss.append(c)
    N = len(origs)
    labels_t = torch.tensor(labels, device=device)

    judge = AH.Judge(tr, class_to_idx) if hasattr(AH, "Judge") else None
    cons_judge = AH.ConsistencyJudge(tr) if hasattr(AH, "ConsistencyJudge") else None
    if judge is None:
        sys.exit("[FATAL] could not construct Judge from attack_harness.py")

    B = args.batch_size

    def judge_stack(imgs):
        correct = 0; wm_sum = 0.0
        for s in range(0, len(imgs), B):
            xb = torch.cat(imgs[s:s+B], 0).clamp(0, 1)
            lb = labels_t[s:s+B]
            c, wm = judge.score01(xb, lb)
            correct += int(c.sum()); wm_sum += float(wm.sum())
        return 100.0 * correct / len(imgs), wm_sum / len(imgs)

    def cons_stack(imgs):
        if cons_judge is None or not getattr(cons_judge, "available", False):
            return None
        acc = 0
        for s in range(0, len(imgs), B):
            xb = torch.cat(imgs[s:s+B], 0).clamp(0, 1)
            r, _ = cons_judge.accept_rate(xb)
            acc += r * xb.shape[0]
        return 100.0 * acc / len(imgs)

    # baselines
    print("\n[SCORING] sanity ...")
    acc_orig, _ = judge_stack(origs)
    acc_wm, _ = judge_stack(wms)
    print(f"  originals   gate-acc {acc_orig:6.2f}%")
    print(f"  watermarked gate-acc {acc_wm:6.2f}%   (robustness baseline = this should be retained)")
    cons_wm = cons_stack(wms)
    if cons_wm is not None:
        print(f"  watermarked cons-accept {cons_wm:6.2f}%")

    results = {
        "dataset": data_root.name, "n": N, "classes": classes,
        "acc_originals": acc_orig, "acc_watermarked": acc_wm,
        "robustness": {}, "forgery": {},
    }

    # ---------------- degradation ops (operate on wm images) ----------------
    def jpeg(x01, q):
        """JPEG round-trip via PIL, per image."""
        from PIL import Image
        import io
        out = []
        for k in range(x01.shape[0]):
            arr = (x01[k].clamp(0,1).detach().cpu().permute(1,2,0).numpy()*255).astype("uint8")
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, 2)
            im = Image.fromarray(arr, "RGB")
            buf = io.BytesIO(); im.save(buf, format="JPEG", quality=int(q)); buf.seek(0)
            im2 = Image.open(buf).convert("RGB")
            a2 = np.asarray(im2).astype("float32")/255.0
            t = torch.from_numpy(a2).permute(2,0,1).unsqueeze(0).to(x01.device)
            if x01.shape[1] == 1:
                t = t.mean(dim=1, keepdim=True)
            out.append(t)
        return torch.cat(out, 0)

    def gauss_kernel(sigma, ch, device):
        r = max(1, int(round(3*sigma)))
        xs = torch.arange(-r, r+1, dtype=torch.float32, device=device)
        k1 = torch.exp(-(xs**2)/(2*sigma*sigma)); k1 = k1/k1.sum()
        k2 = torch.outer(k1, k1)
        return k2.view(1,1,2*r+1,2*r+1).repeat(ch,1,1,1), r

    def blur_gauss(x01, sigma=1.2):
        ch = x01.shape[1]
        k, r = gauss_kernel(sigma, ch, x01.device)
        return F.conv2d(F.pad(x01, (r,r,r,r), mode="reflect"), k, groups=ch).clamp(0,1)

    def blur_median(x01, ksize=3):
        ch = x01.shape[1]; r = ksize//2
        xp = F.pad(x01, (r,r,r,r), mode="reflect")
        patches = xp.unfold(2, ksize, 1).unfold(3, ksize, 1)   # B,C,H,W,k,k
        med = patches.contiguous().view(*patches.shape[:4], -1).median(dim=-1).values
        return med.clamp(0,1)

    def noise_gauss(x01, sigma):
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        n = torch.randn(x01.shape, generator=g).to(x01.device) * sigma
        return (x01 + n).clamp(0,1)

    def resize_rt(x01, scale=0.5):
        H, W = x01.shape[-2:]
        small = F.interpolate(x01, scale_factor=scale, mode="bilinear", align_corners=False)
        back = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)
        return back.clamp(0,1)

    def crop_pad(x01, keep=0.875):
        H, W = x01.shape[-2:]
        ch, cw = int(H*keep), int(W*keep)
        top, left = (H-ch)//2, (W-cw)//2
        cropped = x01[..., top:top+ch, left:left+cw]
        back = F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)
        return back.clamp(0,1)

    def apply_op(op):
        """Apply a degradation op to every wm image, return list of 1,C,H,W tensors."""
        out = []
        for s in range(0, N, B):
            xb = torch.cat(wms[s:s+B], 0)
            yb = op(xb)
            for k in range(yb.shape[0]):
                out.append(yb[k:k+1])
        return out

    def record_robust(name, op):
        imgs = apply_op(op)
        acc, _ = judge_stack(imgs)
        retained = 100.0 * acc / max(acc_wm, 1e-6)   # % of the clean-wm gate-acc kept
        results["robustness"][name] = {"gate_acc": acc, "retained_pct": retained}
        verdict = "ROBUST" if retained >= 80 else ("FRAGILE" if retained < 50 else "OK")
        print(f"  {name:14s}  gate-acc {acc:6.2f}%   retained {retained:6.1f}%  [{verdict}]")

    print("\n[ROBUSTNESS] genuine watermark under processing (gate should HOLD; higher = better)")
    record_robust("jpeg_q90",    lambda x: jpeg(x, 90))
    record_robust("jpeg_q75",    lambda x: jpeg(x, 75))
    record_robust("jpeg_q50",    lambda x: jpeg(x, 50))
    record_robust("blur_gauss",  lambda x: blur_gauss(x, 1.2))
    record_robust("blur_median", lambda x: blur_median(x, 3))
    record_robust("noise_gauss", lambda x: noise_gauss(x, args.noise_sigma))
    record_robust("resize_rt",   lambda x: resize_rt(x, 0.5))
    record_robust("crop_pad",    lambda x: crop_pad(x, 0.875))

    # ---------------- kNN forgery ----------------
    print("\n[FORGERY] kNN transplant -- delta from the NEAREST image by content (blind cons is the honest number)")
    # nearest neighbour by L2 over a small thumbnail of the ORIGINALS
    thumbs = []
    for o in origs:
        t = F.interpolate(o, size=(16, 16), mode="bilinear", align_corners=False).flatten()
        thumbs.append(t)
    T = torch.stack(thumbs, 0)                      # N, D
    # pairwise distances (N is val-set sized; fine on GPU)
    d2 = torch.cdist(T, T)                          # N,N
    d2.fill_diagonal_(float("inf"))
    nn_idx = torch.argmin(d2, dim=1).tolist()       # nearest neighbour per image

    imgs = [(origs[i] + deltas[nn_idx[i]]).clamp(0, 1) for i in range(N)]
    acc, _ = judge_stack(imgs)
    rate = AH._forgery_rate(acc_orig, acc, acc_wm) if hasattr(AH, "_forgery_rate") else (100.0*acc/max(acc_wm,1e-6))
    cons_rate = cons_stack(imgs)
    entry = {"gate_acc": acc, "forgery_success": rate}
    line = f"  knn_transplant    C2-forgery {rate:6.2f}%"
    if cons_rate is not None:
        entry["forgery_via_cons"] = cons_rate
        cv = "BLOCKED" if cons_rate < 20 else ("FORGED" if cons_rate > 70 else "PARTIAL")
        line += f"   |   CONS-forgery {cons_rate:6.2f}% [{cv}]"
    results["forgery"]["knn_transplant"] = entry
    print(line)

    # ---------------- summary + save ----------------
    print("\n" + "=" * 74)
    print(f"EXTENDED SUMMARY — {data_root.name}")
    print("=" * 74)
    print("  ROBUSTNESS (retained % of clean-wm gate-acc; >=80 robust):")
    for k, v in results["robustness"].items():
        print(f"    {k:14s} {v['retained_pct']:6.1f}%")
    if results["forgery"]:
        kv = results["forgery"]["knn_transplant"]
        cons = kv.get("forgery_via_cons")
        print(f"  kNN FORGERY (blind cons): {cons:.2f}%" if cons is not None
              else f"  kNN FORGERY (C2): {kv['forgery_success']:.2f}%")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OUT] wrote {args.out}")


if __name__ == "__main__":
    main()
