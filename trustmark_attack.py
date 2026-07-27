#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
trustmark_attack.py — transplant-forgery benchmark for the TrustMark baseline.

PURPOSE (the "vs" column of the paper's forgery table)
    Our Level-2 system drops blind transplant forgery to ~13-16%. This script
    measures the SAME attack family against TrustMark, judged through
    TRUSTMARK'S OWN DECODER (not our C2) — the fair test for their method.
    Expected result: transplanted TrustMark residuals still decode ⇒ forgery
    ~high, showing the whole additive-watermark class shares the weakness that
    only our consistency detector closes.

WHAT IT DOES
    1. Loads N originals from an orig/<class>/*.png tree (any ATTACK_DATA dump).
    2. Embeds TrustMark (one shared secret) → wm_i, residual δ_i = wm_i − orig_i.
    3. Runs: passthrough_wm (sanity), transplant_same, transplant_cross,
       sign_flip, scaled_{0.5,1.5,2.0}, universal (mean δ), plus a
       false-positive check on plain originals.
    4. ACCEPTED = decode says watermark PRESENT and the secret matches the
       canonical stored secret (TrustMark truncates to its capacity, so the
       canonical value is taken from a genuine roundtrip, not from the CLI arg).
    5. forgery = acceptance rate on forged images. JSON out.

USAGE (Machine A, inside the isolated trustmark venv, full python.exe path):
    C:\venvs\trustmark\Scripts\python.exe trustmark_attack.py ^
        --data "E:\ATTACK_DATA\AFHQ_color" --max_images 300 ^
        --out "E:\RUNS\EVAL\trustmark_attack_afhq.json"

    AFHQ is TrustMark's native domain (natural images) — the fair arena.
    --max_images 300 keeps the CPU runtime reasonable (~encode 300 + ~2400
    decodes); raise it later for the paper number if wanted.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_origs(root: Path, max_images: int, rng: random.Random):
    orig_root = root / "orig"
    if not orig_root.exists():
        raise SystemExit(f"[FATAL] expected {orig_root} (an ATTACK_DATA-style dump)")
    items = []  # (class, path)
    for cls_dir in sorted(d for d in orig_root.iterdir() if d.is_dir()):
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                items.append((cls_dir.name, f))
    if not items:
        raise SystemExit(f"[FATAL] no images under {orig_root}")
    if max_images and max_images < len(items):
        items = rng.sample(items, max_images)
    return items


def to_arr(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB"), dtype=np.float32)


def to_img(a: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help=r"root containing orig\<class>\*.png")
    ap.add_argument("--max_images", type=int, default=300)
    ap.add_argument("--secret", default="WACV2026")
    ap.add_argument("--variant", default="Q", help="TrustMark model_type (Q/B/C)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from trustmark import TrustMark
    tm = TrustMark(verbose=False, model_type=args.variant)
    rng = random.Random(args.seed)

    items = load_origs(Path(args.data), args.max_images, rng)
    print(f"[DATA] {len(items)} originals from {args.data}")

    # ---- embed once, derive residuals, capture the canonical stored secret ----
    print("[EMBED] TrustMark encoding (one shared secret) ...")
    origs, wms, deltas, clss = [], [], [], []
    canonical = None
    for i, (cls, p) in enumerate(items):
        cover = Image.open(p).convert("RGB")
        stego = tm.encode(cover, args.secret)
        o, w = to_arr(cover), to_arr(stego)
        origs.append(o); wms.append(w); deltas.append(w - o); clss.append(cls)
        if canonical is None:
            dec, present, _conf = tm.decode(stego)
            if not present:
                raise SystemExit("[FATAL] genuine stego does not decode — check the install")
            canonical = dec
            print(f"[EMBED] canonical stored secret: {canonical!r} (capacity-truncated from {args.secret!r})")
        if (i + 1) % 50 == 0:
            print(f"    encoded {i + 1}/{len(items)}")
    N = len(origs)
    mean_delta = np.mean(np.stack(deltas, 0), axis=0)

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

    def accepted(img_arr) -> bool:
        dec, present, _conf = tm.decode(to_img(img_arr))
        return bool(present) and (dec == canonical)

    def rate(imgs, tag):
        n_acc = 0
        for k, a in enumerate(imgs):
            n_acc += int(accepted(a))
            if (k + 1) % 100 == 0:
                print(f"    [{tag}] decoded {k + 1}/{len(imgs)}")
        return 100.0 * n_acc / len(imgs)

    results = {"data": str(args.data), "n": N, "variant": args.variant,
               "canonical_secret": canonical, "attacks": {}}

    def record(name, imgs):
        r = rate(imgs, name)
        results["attacks"][name] = r
        verdict = "BLOCKED" if r < 20 else ("FORGED" if r > 70 else "PARTIAL")
        print(f"  {name:18s}  decode-accept {r:6.2f}%   [{verdict}]")
        return r

    print("\n[SANITY]")
    record("passthrough_wm", wms)                    # expect ~100
    record("false_pos_orig", origs)                  # expect ~0

    print("\n[ATTACKS] acceptance by TrustMark's OWN decoder (high = forgeable)")
    record("transplant_same", [origs[i] + deltas[partner_same(i)] for i in range(N)])
    record("transplant_cross", [origs[i] + deltas[partner_cross(i)] for i in range(N)])
    record("sign_flip", [origs[i] - deltas[i] for i in range(N)])
    for a in (0.5, 1.5, 2.0):
        record(f"scaled_{a}", [origs[i] + a * deltas[i] for i in range(N)])
    record("universal", [origs[i] + mean_delta for i in range(N)])

    print("\n" + "=" * 60)
    print(f"SUMMARY — TrustMark-{args.variant} on {Path(args.data).name} (n={N})")
    print("=" * 60)
    for k, v in results["attacks"].items():
        print(f"  {k:18s}  {v:6.2f}%")
    print("  (transplant high = TrustMark's mark is transferable — the class-wide")
    print("   weakness; our Level-2 consistency detector is what closes it)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n[OUT] wrote {args.out}")


if __name__ == "__main__":
    main()
