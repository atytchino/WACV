#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_curve.py  --  build the "original vs watermarked accuracy split vs epoch" data.

TWO MODES:

  (A) --from_log  (fast, no GPU): parse a console.log and pull the split per epoch.
      REALVAL lines give the clean GAPg but only every real_val_every epochs; the
      end-of-epoch VALprobe gap fills the gaps. Writes a CSV: epoch, gapg, source.

  (B) --from_ckpts (clean, uses GPU): for a publication-quality curve, re-run the
      REAL validation split on EVERY saved checkpoint wm_system_eNN.pth in a run
      folder. This gives a clean REALVAL GAPg at every epoch, not just every 4th.
      (Slower: one eval pass per epoch, but each is quick at 160/256.)

USAGE (from the WACV project dir, WACV venv):
  # fast, from an existing log:
  & $WPY split_curve.py --from_log "E:\\RUNS\\TLD_smooth_eps010_s0\\console.log" `
        --out "E:\\RUNS\\FIGS\\tld_split_curve.csv"

  # clean, recomputed per checkpoint (needs trainer + the run's checkpoints dir):
  & $WPY split_curve.py --from_ckpts "E:\\RUNS\\TLD_smooth_eps010_s0" `
        --trainer .\\20260728-Trainer_MULTIBIT_v17_TILEGRID.py `
        --out "E:\\RUNS\\FIGS\\tld_split_curve_clean.csv" [--gray]

  # multiple logs into one CSV (for a 3-dataset overlay plot):
  & $WPY split_curve.py --from_log log1 log2 log3 --labels TLD AFHQ ORNL `
        --out combined.csv

Output CSV columns: dataset, epoch, gapg, raw_acc, wm_acc, source
Plot it with any tool; each dataset is a series (gapg vs epoch).
"""

import argparse
import csv
import re
import sys
from pathlib import Path

# REALVAL E08][plain] RAW=43.16% BOTH=89.49% GAPg=+46.33pp
RE_REALVAL = re.compile(
    r"\[REALVAL E(\d+)\]\[plain\][^\n]*?RAW=([\-\d.]+)%[^\n]*?BOTH=([\-\d.]+)%[^\n]*?GAPg=([+\-\d.]+)pp")
# fallback: end-of-epoch VALprobe g(raw/wm/gap)=0.500/0.562/+0.062  -> gap as fraction
# the wrapped log may put the "VALprobe" label on a separate line, so match the
# metric line directly. (?<![a-z]) stops it matching the per-iter "ng(raw/wm/gap)".
RE_VALPROBE = re.compile(
    r"(?<![a-z])g\(raw/wm/gap\)=([\-\d.]+)/([\-\d.]+)/([+\-\d.]+)")
# epoch marker to attribute VALprobe to an epoch: [E08 it ...]
RE_EPOCH = re.compile(r"\[E(\d+)\s+it")


def parse_log(path):
    """Return dict epoch -> (gapg, raw, wm, source). Prefers REALVAL; fills with last
    VALprobe seen in each epoch."""
    rows = {}
    cur_epoch = None
    last_vp = {}   # epoch -> (gap_pp, raw_pct, wm_pct)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            me = RE_EPOCH.search(line)
            if me:
                cur_epoch = int(me.group(1))
            mr = RE_REALVAL.search(line)
            if mr:
                e = int(mr.group(1))
                raw = float(mr.group(2)); both = float(mr.group(3)); gap = float(mr.group(4))
                rows[e] = (gap, raw, both, "REALVAL")
                continue
            mv = RE_VALPROBE.search(line)
            if mv and cur_epoch is not None:
                raw = float(mv.group(1)) * 100
                wm = float(mv.group(2)) * 100
                gap = float(mv.group(3)) * 100
                last_vp[cur_epoch] = (gap, raw, wm)
    # fill epochs that have no REALVAL with the last VALprobe of that epoch
    for e, (gap, raw, wm) in last_vp.items():
        if e not in rows:
            rows[e] = (gap, raw, wm, "VALprobe")
    n_realval = sum(1 for v in rows.values() if v[3] == "REALVAL")
    print(f"    [parse] {Path(path).name}: {len(rows)} epochs "
          f"({n_realval} REALVAL, {len(last_vp)} epochs had VALprobe)")
    if not rows:
        print("    [parse][warn] matched 0 lines -- check the log format or encoding.")
    return rows


def recompute_from_ckpts(run_dir, trainer, gray):
    """Clean curve: run REAL validation split on every wm_system_eNN.pth."""
    import torch
    import importlib.util as ilu
    proj = Path(trainer).resolve().parent
    evp = proj / "eval_wm_system.py"
    spec = ilu.spec_from_file_location("eval_wm_system", str(evp))
    EM = ilu.module_from_spec(spec); spec.loader.exec_module(EM)
    EM._install_torch_load_compat()
    TM = EM.load_trainer_module(str(trainer))

    ckdir = Path(run_dir) / "checkpoints"
    sys_ckpts = sorted(ckdir.glob("wm_system_e*.pth"))
    if not sys_ckpts:
        sys.exit(f"[FATAL] no wm_system_e*.pth in {ckdir}")
    rows = {}
    scratch = proj / "_curve_scratch"; scratch.mkdir(exist_ok=True)
    for sysck in sys_ckpts:
        m = re.search(r"wm_system_e(\d+)\.pth", sysck.name)
        if not m:
            continue
        e = int(m.group(1))
        c2ck = ckdir / f"c2_eval_e{e:03d}.pth"
        if not c2ck.exists():
            # try non-zero-padded
            c2ck = ckdir / f"c2_eval_e{e}.pth"
        if not c2ck.exists():
            print(f"[skip] E{e}: no matching c2_eval ckpt")
            continue
        print(f"[eval] E{e}: {sysck.name} ...")
        try:
            tr, _ = EM.build_trainer(TM, str(sysck), str(c2ck),
                                     overrides={"out_root": scratch})
            # use the module's own real-val split if exposed; else a quick gate pass
            # run_realval-style: compare gate-acc on originals vs watermarked over val
            raw, wm, gap = _quick_split(EM, tr, gray)
            rows[e] = (gap, raw, wm, "REALVAL_recomputed")
        except Exception as ex:
            print(f"[warn] E{e} failed: {ex}")
    return rows


def _quick_split(EM, tr, gray):
    """Gate-acc on originals vs watermarked over the val set -> (raw%, wm%, gap_pp)."""
    import torch
    n = 0; ok_raw = 0; ok_wm = 0
    for xN, valid_mask, y, _paths in EM._iter_val(tr, None):
        x01 = tr._to01(xN)
        if gray:
            x01 = EM.to_gray01(x01)
        syn = tr.synth_variants_nograd(
            x01, valid_mask=valid_mask, epoch=0,
            variants=("base", "both"), k_factor=1.0, varpercent=False, mode="eval")
        both01 = syn["both01"].clamp(0, 1)
        rawN = tr._apply_prod_padding_wipe(x01 * 2 - 1, valid_mask)
        bothN = tr._apply_prod_padding_wipe(both01 * 2 - 1, valid_mask)
        z_raw, _, _ = tr.c2_eval(rawN, gate=True)
        z_wm, _, _ = tr.c2_eval(bothN, gate=True)
        yb = y.cpu()
        ok_raw += int((z_raw.argmax(1).cpu() == yb).sum())
        ok_wm += int((z_wm.argmax(1).cpu() == yb).sum())
        n += yb.numel()
    raw = 100.0 * ok_raw / max(n, 1)
    wm = 100.0 * ok_wm / max(n, 1)
    return raw, wm, wm - raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_log", nargs="*", default=None, help="one or more console.log paths")
    ap.add_argument("--from_ckpts", default=None, help="a run folder (has checkpoints/)")
    ap.add_argument("--trainer", default=None, help="v17 trainer (needed for --from_ckpts)")
    ap.add_argument("--labels", nargs="*", default=None, help="dataset labels matching --from_log order")
    ap.add_argument("--gray", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    all_rows = []   # (dataset, epoch, gapg, raw, wm, source)

    if args.from_ckpts:
        if not args.trainer:
            sys.exit("[FATAL] --from_ckpts needs --trainer")
        ds = Path(args.from_ckpts).name
        rows = recompute_from_ckpts(args.from_ckpts, args.trainer, args.gray)
        for e in sorted(rows):
            gap, raw, wm, src = rows[e]
            all_rows.append((ds, e, gap, raw, wm, src))

    if args.from_log:
        labels = args.labels or [Path(p).parent.name for p in args.from_log]
        for path, label in zip(args.from_log, labels):
            rows = parse_log(path)
            for e in sorted(rows):
                gap, raw, wm, src = rows[e]
                all_rows.append((label, e, gap, raw, wm, src))

    if not all_rows:
        sys.exit("[FATAL] nothing parsed. Pass --from_log <paths> or --from_ckpts <run>.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "epoch", "gapg", "raw_acc", "wm_acc", "source"])
        for r in all_rows:
            w.writerow(r)

    print(f"[out] wrote {len(all_rows)} rows -> {args.out}")
    # quick text preview
    cur = None
    for ds, e, gap, raw, wm, src in all_rows:
        if ds != cur:
            print(f"\n{ds}:")
            cur = ds
        print(f"  E{e:02d}  gap {gap:+6.2f}pp  (raw {raw:.1f} -> wm {wm:.1f})  [{src}]")


if __name__ == "__main__":
    main()
