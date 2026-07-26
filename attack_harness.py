#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
attack_harness.py — forgery-attack benchmark over dumped original/watermarked pairs.

PURPOSE
    Measure how easily the gate (C2) is fooled by forged watermarks, per attack,
    per dataset. This is the BEFORE/AFTER instrument: run it now on the current
    system to get the baseline (transplant is expected to succeed ~99%), then run
    the SAME tool on a Level-2 system later — the drop in forgery success is the
    result. One metric, one interface, both roles.

WHAT IT DOES
    1. Loads original/watermarked PNG pairs from a dump dir produced by
       eval_wm_system.py --mode dump_pairs:
           <data>/orig/<class>/<stem>.png
           <data>/wm/<class>/<stem>.png
       Same <class>/<stem> in both trees = one pair. The residual is recomputed
       on the fly: delta = wm - orig.
    2. Loads the dataset's C2 detector from its checkpoint (the judge). This reuses
       eval_wm_system's own trainer-loading so the scoring path is identical to the
       trainer's val.
    3. Runs each attack, feeds the forged image through C2 (gate=True), and reports
       FORGERY SUCCESS RATE = where the forged image's gate-accuracy lands between
       originals (0% = fully blocked) and genuine watermarked (100% = fully forged).
       This is exactly the metric run_transplant uses, applied to every attack.

ATTACKS
    passthrough_orig   sanity: originals (should be ~0% — blocked)
    passthrough_wm     sanity: genuine watermarked (should be ~100% — passes)
    transplant_same    delta from another image of the SAME class -> orig_B
    transplant_cross   delta from another image of a DIFFERENT class -> orig_B
    sign_flip          -delta on orig_B
    scaled_0.5/1.5/2.0 alpha * delta on orig_B (alpha sweep)
    universal          the MEAN delta over the whole set, applied to every orig
                       (degenerate control — expected to be REJECTED ~<5%)

USAGE
    python attack_harness.py \
        --trainer ".\20260707-Trainer_MULTIBIT_v13.py" \
        --system_ckpt "E:\RUNS\AFHQ160_tile64_s0\checkpoints\wm_system_e008.pth" \
        --c2_ckpt     "E:\RUNS\AFHQ160_tile64_s0\checkpoints\c2_eval_e008.pth" \
        --data        "E:\ATTACK_DATA\AFHQ_color" \
        --out         "E:\RUNS\EVAL\attack_afhq_color.json"

    Optional: --max_images N (subsample for a quick pass), --batch_size B,
              --seed S (controls the random pairing for transplant/sign/scale).

NOTES
    - Images are loaded from PNG (fast, no val loader needed); only the C2 detector
      comes from the checkpoint. This keeps the harness dataset-agnostic and quick.
    - The harness imports eval_wm_system.py to reuse build_trainer / load_trainer_module
      and the trainer's c2_eval + _apply_prod_padding_wipe, so judging matches the
      trainer exactly. Keep attack_harness.py next to eval_wm_system.py.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

# reuse the proven loading + scoring plumbing from the eval wrapper
import importlib.util as _ilu


def _load_eval_module(here: Path):
    cand = here / "eval_wm_system.py"
    if not cand.exists():
        raise SystemExit(f"[FATAL] eval_wm_system.py must sit next to this script: {cand} not found")
    spec = _ilu.spec_from_file_location("eval_wm_system", str(cand))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- IO
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _list_pairs(data_root: Path):
    """Return [(class, stem, orig_path, wm_path)] for every matched pair."""
    orig_root = data_root / "orig"
    wm_root = data_root / "wm"
    if not orig_root.exists() or not wm_root.exists():
        raise SystemExit(f"[FATAL] expected {orig_root} and {wm_root}")
    pairs = []
    for cls_dir in sorted([d for d in orig_root.iterdir() if d.is_dir()]):
        cls = cls_dir.name
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() not in IMG_EXT:
                continue
            wm_f = wm_root / cls / f.name
            if wm_f.exists():
                pairs.append((cls, f.stem, f, wm_f))
    if not pairs:
        raise SystemExit(f"[FATAL] no matched pairs under {data_root}")
    return pairs


def _load01(path: Path, device):
    """Load an image as a [1,C,H,W] float tensor in [0,1]."""
    from torchvision.io import read_image
    from torchvision.transforms.functional import convert_image_dtype
    t = read_image(str(path))                       # uint8 [C,H,W]
    t = convert_image_dtype(t, torch.float32)       # [0,1]
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    elif t.shape[0] == 4:
        t = t[:3]
    return t.unsqueeze(0).to(device)


# --------------------------------------------------------------------------- judge
class Judge:
    """Wraps the trainer's C2 so a batch of [0,1] images -> (gate-correct, wm-score)."""

    def __init__(self, tr, class_to_idx):
        self.tr = tr
        self.class_to_idx = class_to_idx

    @torch.no_grad()
    def score01(self, x01, labels):
        """x01: [B,3,H,W] in [0,1]; labels: LongTensor[B]. Returns (correct[B] bool, wm[B])."""
        tr = self.tr
        xN = (x01 * 2 - 1)
        # match the trainer's production padding wipe path (valid_mask=None -> all valid)
        try:
            xN = tr._apply_prod_padding_wipe(xN, None)
        except Exception:
            pass
        z, wm, _ = tr.c2_eval(xN, gate=True)
        correct = (z.argmax(1).cpu() == labels.cpu())
        return correct, wm.detach().float().cpu()


# --------------------------------------------------------------------------- attacks
def _forgery_rate(acc_orig, acc_attack, acc_wm):
    """Where the attack lands between originals (blocked) and genuine wm (passes)."""
    span = max(acc_wm - acc_orig, 1e-6)
    return 100.0 * (acc_attack - acc_orig) / span


def run(args):
    here = Path(__file__).resolve().parent
    EM = _load_eval_module(here)
    EM._install_torch_load_compat()
    TM = EM.load_trainer_module(args.trainer)

    scratch = args.out_root or (str(Path(args.out).parent / "_atk_scratch") if args.out
                                else str(Path.cwd() / "_atk_scratch"))
    tr, ck = EM.build_trainer(TM, args.system_ckpt, args.c2_ckpt,
                              overrides={"out_root": Path(scratch)})
    for m in (getattr(tr, "ae", None), getattr(tr, "c2", None)):
        try:
            m.eval()
        except Exception:
            pass
    device = tr.device

    data_root = Path(args.data)
    pairs = _list_pairs(data_root)
    classes = sorted({c for (c, *_ ) in pairs})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"[DATA] {data_root.name}: {len(pairs)} pairs, {len(classes)} classes {classes}")

    rng = random.Random(args.seed)
    if args.max_images and args.max_images < len(pairs):
        pairs = rng.sample(pairs, args.max_images)
        print(f"[DATA] subsampled to {len(pairs)} pairs (seed {args.seed})")

    judge = Judge(tr, class_to_idx)

    # -------- load everything we need once: orig01, wm01, label, class --------
    # (dumped images are small — 160px — so this fits comfortably.)
    origs, wms, labels, clss = [], [], [], []
    for (cls, stem, of, wf) in pairs:
        origs.append(_load01(of, device))
        wms.append(_load01(wf, device))
        labels.append(class_to_idx[cls])
        clss.append(cls)
    labels_t = torch.tensor(labels, dtype=torch.long)
    N = len(origs)

    # residuals
    deltas = [w - o for (w, o) in zip(wms, origs)]
    mean_delta = torch.stack([d.squeeze(0) for d in deltas], 0).mean(0, keepdim=True)  # [1,3,H,W]

    # per-class index lists (for same/cross transplant partner selection)
    by_class = {}
    for i, c in enumerate(clss):
        by_class.setdefault(c, []).append(i)
    all_idx = list(range(N))

    def partner_same(i):
        pool = [j for j in by_class[clss[i]] if j != i]
        return rng.choice(pool) if pool else i

    def partner_cross(i):
        pool = [j for j in all_idx if clss[j] != clss[i]]
        return rng.choice(pool) if pool else i

    # -------- batched judging helper --------
    def judge_stack(imgs, labs):
        correct_total = 0
        wm_sum = 0.0
        B = args.batch_size
        for s in range(0, len(imgs), B):
            xb = torch.cat(imgs[s:s + B], 0)
            lb = labs[s:s + B]
            correct, wm = judge.score01(xb.clamp(0, 1), lb)
            correct_total += int(correct.sum())
            wm_sum += float(wm.sum())
        n = len(imgs)
        return 100.0 * correct_total / n, wm_sum / n

    # -------- baselines (sanity) --------
    print("\n[SCORING] originals and genuine watermarked (sanity) ...")
    acc_orig, wm_orig = judge_stack(origs, labels_t)
    acc_wm, wm_wm = judge_stack(wms, labels_t)
    print(f"  originals   gate-acc {acc_orig:6.2f}%   wm-score {wm_orig:+.3f}")
    print(f"  watermarked gate-acc {acc_wm:6.2f}%   wm-score {wm_wm:+.3f}")

    results = {
        "dataset": data_root.name,
        "n": N,
        "classes": classes,
        "acc_originals": acc_orig,
        "acc_watermarked": acc_wm,
        "wm_score_orig": wm_orig,
        "wm_score_wm": wm_wm,
        "attacks": {},
    }

    def record(name, imgs):
        acc, wm = judge_stack(imgs, labels_t)
        rate = _forgery_rate(acc_orig, acc, acc_wm)
        results["attacks"][name] = {"gate_acc": acc, "wm_score": wm, "forgery_success": rate}
        verdict = "BLOCKED" if rate < 20 else ("FORGED" if rate > 70 else "PARTIAL")
        print(f"  {name:18s}  gate-acc {acc:6.2f}%   wm {wm:+.3f}   "
              f"forgery {rate:6.2f}%   [{verdict}]")
        return rate

    # -------- attacks --------
    print("\n[ATTACKS] forgery success: 0 = blocked, 100 = binding provides no defense")

    # transplant same-class
    imgs = []
    for i in range(N):
        j = partner_same(i)
        imgs.append((origs[i] + deltas[j]).clamp(0, 1))
    record("transplant_same", imgs)

    # transplant cross-class
    imgs = []
    for i in range(N):
        j = partner_cross(i)
        imgs.append((origs[i] + deltas[j]).clamp(0, 1))
    record("transplant_cross", imgs)

    # sign flip (own residual, negated)
    imgs = [(origs[i] - deltas[i]).clamp(0, 1) for i in range(N)]
    record("sign_flip", imgs)

    # scaled (own residual, alpha sweep)
    for alpha in (0.5, 1.5, 2.0):
        imgs = [(origs[i] + alpha * deltas[i]).clamp(0, 1) for i in range(N)]
        record(f"scaled_{alpha}", imgs)

    # universal (mean residual on every original) — degenerate control
    imgs = [(origs[i] + mean_delta).clamp(0, 1) for i in range(N)]
    record("universal", imgs)

    # -------- summary --------
    print("\n" + "=" * 66)
    print(f"SUMMARY — {data_root.name}  (originals {acc_orig:.1f}% -> watermarked {acc_wm:.1f}%)")
    print("=" * 66)
    for name, r in results["attacks"].items():
        print(f"  {name:18s}  forgery success {r['forgery_success']:6.2f}%")
    print("  (transplant/sign/scaled high = mark is transferable; "
          "universal should be low)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n[OUT] wrote {args.out}")

    return results


def main():
    ap = argparse.ArgumentParser(description="Forgery-attack harness over dumped orig/wm pairs")
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--system_ckpt", required=True)
    ap.add_argument("--c2_ckpt", required=True)
    ap.add_argument("--data", required=True, help=r"dump root with orig\ and wm\ trees")
    ap.add_argument("--out", default="", help="optional JSON output path")
    ap.add_argument("--out_root", default="", help="scratch dir for trainer bookkeeping")
    ap.add_argument("--max_images", type=int, default=0, help="0 = all pairs")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
