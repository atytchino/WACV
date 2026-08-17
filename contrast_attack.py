#!/usr/bin/env python
"""
contrast_attack.py  -  standalone contrast-forgery test

Question: "Is the watermark just contrast enhancement?" - can an attacker pass
the gate by applying an arbitrary GLOBAL contrast operation to an UNMARKED image,
without ever seeing a real watermark?

If contrast-enhanced ORIGINALS do NOT pass the gate and are REJECTED by the
consistency detector, the mark is a specific content-adaptive pattern, not
reproducible by any global contrast op - the concern becomes a PASSED test.

Reuses attack_harness.py's own loaders and judges (Judge, ConsistencyJudge) so
numbers are directly comparable. Does NOT modify any existing file.

Two ways to supply contrast-enhanced images:
  (A) --contrast_root <dir>  tree <variant>/<class>/*.png from make_contrast_dataset.py
  (B) (default)              variants generated on the fly from the dump originals

It also scores unmarked originals (FLOOR - should fail) and genuine watermarked
(CEILING - should pass) so the contrast rows sit between a clear floor and ceiling.

Usage (on the fly):
  python contrast_attack.py --trainer <v17.py> --system_ckpt <e008> --c2_ckpt <e008> \
      --data E:\ATTACK_DATA\AFHQ_256_smooth --out E:\RUNS\EVAL\contrast_attack_AFHQ.json
Usage (pre-made dataset):
  ... --contrast_root E:\ATTACK_DATA\AFHQ_contrast
"""
import argparse, json, sys
from pathlib import Path


def fmt(v):
    return "  n/a " if v is None else f"{v:6.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--system_ckpt", required=True)
    ap.add_argument("--c2_ckpt", required=True)
    ap.add_argument("--data", required=True, help="dump_pairs folder (orig/ + wm/)")
    ap.add_argument("--contrast_root", default="", help="optional <variant>/<class>/*.png tree")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--gray", action="store_true")
    ap.add_argument("--max_images", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    import torch
    import importlib.util as ilu

    proj = Path(args.trainer).resolve().parent
    ahp = proj / "attack_harness.py"
    if not ahp.exists():
        sys.exit(f"[FATAL] attack_harness.py not found at {ahp}")
    spec = ilu.spec_from_file_location("attack_harness", str(ahp))
    AH = ilu.module_from_spec(spec); spec.loader.exec_module(AH)

    EM = AH._load_eval_module(proj) if hasattr(AH, "_load_eval_module") else None
    if EM is None:
        evp = proj / "eval_wm_system.py"
        spec2 = ilu.spec_from_file_location("eval_wm_system", str(evp))
        EM = ilu.module_from_spec(spec2); spec2.loader.exec_module(EM)

    print("[load] building system from checkpoints ...")
    EM._install_torch_load_compat()
    TM = EM.load_trainer_module(str(args.trainer))
    scratch = proj / "_contrast_scratch"; scratch.mkdir(exist_ok=True)
    tr, _ = EM.build_trainer(TM, str(args.system_ckpt), str(args.c2_ckpt),
                             overrides={"out_root": scratch})
    device = tr.device
    print(f"[load] READY device={device}")

    data_root = Path(args.data)
    pairs = AH._list_pairs(data_root)
    if args.max_images:
        pairs = pairs[:args.max_images]
    if not pairs:
        sys.exit(f"[FATAL] no orig/wm pairs under {data_root}")
    classes = sorted({c for c, _s, _o, _w in pairs})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"[data] {len(pairs)} reference pairs, {len(classes)} classes")

    def load01(p):
        x = AH._load01(Path(p), device)
        if args.gray:
            x = EM.to_gray01(x)
        return x

    origs, wms, labels, orig_paths = [], [], [], []
    for c, _stem, op, wp in pairs:
        origs.append(load01(op)); wms.append(load01(wp))
        labels.append(class_to_idx[c]); orig_paths.append((c, op))
    labels_t = torch.tensor(labels, device=device)

    judge = AH.Judge(tr, class_to_idx) if hasattr(AH, "Judge") else None
    cons_judge = AH.ConsistencyJudge(tr) if hasattr(AH, "ConsistencyJudge") else None
    if judge is None:
        sys.exit("[FATAL] could not construct Judge from attack_harness.py")
    B = args.batch_size

    def judge_stack(imgs, labs):
        correct = 0
        for s in range(0, len(imgs), B):
            xb = torch.cat(imgs[s:s+B], 0).clamp(0, 1)
            lb = labs[s:s+B]
            c, _wm = judge.score01(xb, lb)
            correct += int(c.sum())
        return 100.0 * correct / len(imgs)

    def cons_stack(imgs):
        if cons_judge is None or not getattr(cons_judge, "available", False):
            return None
        acc = 0.0
        for s in range(0, len(imgs), B):
            xb = torch.cat(imgs[s:s+B], 0).clamp(0, 1)
            r, _ = cons_judge.accept_rate(xb)
            acc += r * xb.shape[0]
        return 100.0 * acc / len(imgs)

    def contrast_linear(x, k):
        m = x.mean(dim=(2, 3), keepdim=True)
        return (m + (x - m) * k).clamp(0, 1)

    def hist_eq(x):
        a = (x.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        out = np.zeros_like(a)
        for c in range(a.shape[2]):
            ch = a[:, :, c].ravel()
            hist, _ = np.histogram(ch, 256, (0, 256))
            cdf = hist.cumsum().astype(float); cdf = 255 * cdf / max(cdf[-1], 1)
            out[:, :, c] = np.interp(ch, np.arange(256), cdf).reshape(a.shape[:2])
        t = torch.from_numpy(out.astype("float32") / 255.0).permute(2, 0, 1).unsqueeze(0).to(x.device)
        return t.clamp(0, 1)

    on_the_fly = {
        "contrast_1p3": lambda x: contrast_linear(x, 1.3),
        "contrast_1p6": lambda x: contrast_linear(x, 1.6),
        "contrast_2p0": lambda x: contrast_linear(x, 2.0),
        "hist_eq":      hist_eq,
    }

    print("\n[SCORING] reference floor/ceiling ...")
    acc_orig = judge_stack(origs, labels_t)
    acc_wm   = judge_stack(wms, labels_t)
    cons_orig = cons_stack(origs)
    cons_wm   = cons_stack(wms)
    print(f"  originals    gate-acc {acc_orig:6.2f}%   cons-accept {fmt(cons_orig)}   (FLOOR - should be low)")
    print(f"  watermarked  gate-acc {acc_wm:6.2f}%   cons-accept {fmt(cons_wm)}   (CEILING - should be high)")

    results = {
        "dataset": data_root.name, "n": len(origs),
        "floor_originals": {"gate_acc": acc_orig, "cons_accept": cons_orig},
        "ceiling_watermarked": {"gate_acc": acc_wm, "cons_accept": cons_wm},
        "contrast": {},
    }

    print("\n[ATTACKS] contrast enhancement of UNMARKED originals")
    print("  high gate-acc AND high cons-accept = the mark IS reproducible by contrast (bad)")
    print("  low  gate-acc OR  low  cons-accept = contrast is NOT the watermark (good)\n")

    def report(name, imgs):
        ga = judge_stack(imgs, labels_t)
        ca = cons_stack(imgs)
        verdict = "BLOCKED"
        if ga >= 50.0 and (ca is None or ca >= 50.0):
            verdict = "FORGED"
        elif ga >= 50.0 or (ca is not None and ca >= 50.0):
            verdict = "PARTIAL"
        print(f"  {name:14s} gate-acc {ga:6.2f}%   cons-accept {fmt(ca)}   [{verdict}]")
        results["contrast"][name] = {"gate_acc": ga, "cons_accept": ca, "verdict": verdict}

    if args.contrast_root:
        root = Path(args.contrast_root)
        variants = sorted([d.name for d in root.iterdir() if d.is_dir()])
        for var in variants:
            imgs = []
            for c, op in orig_paths:
                f = root / var / c / Path(op).name
                if f.exists():
                    imgs.append(load01(f))
            if imgs:
                report(var, imgs)
    else:
        for name, fn in on_the_fly.items():
            imgs = [fn(o) for o in origs]
            report(name, imgs)

    print("\n" + "=" * 74)
    print(f"SUMMARY - contrast attack on {data_root.name}")
    print("=" * 74)
    print(f"  FLOOR   originals    gate-acc {acc_orig:.2f}%")
    print(f"  CEILING watermarked  gate-acc {acc_wm:.2f}%   cons-accept {fmt(cons_wm)}")
    for name, r in results["contrast"].items():
        print(f"  {name:14s} gate-acc {r['gate_acc']:6.2f}%   cons-accept {fmt(r['cons_accept'])}   [{r['verdict']}]")
    allblocked = all(r["verdict"] == "BLOCKED" for r in results["contrast"].values())
    if allblocked:
        print("\n  => ALL contrast variants BLOCKED. Arbitrary contrast enhancement does NOT")
        print("     reproduce the watermark: the mark is a specific content-adaptive pattern.")
    else:
        print("\n  => Some contrast variants pass - consider Path B (retrain w/ contrast penalty).")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OUT] wrote {args.out}")


if __name__ == "__main__":
    main()
