#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inspect_ckpt_and_api.py — READ-ONLY introspection. Changes nothing, trains nothing.

Purpose: the eval harness (eval_wm_system.py) was written against an OLDER copy of
the trainer, so its assumptions need verifying against the REAL v13 + the REAL
checkpoints before we waste time debugging by traceback. This script answers:

  1. what is actually inside wm_system_eXXX.pth and c2_eval_eXXX.pth
  2. what the embedded cfg really contains (field names may differ in v13)
  3. what E6's own recorded metrics are  (confirms GAPg/BER without a GPU run)
  4. whether TrainConfig accepts the fields the harness sets
  5. whether the trainer exposes the methods the harness calls
  6. what the val_loader batch signature really is (source-level, nothing executed)

Nothing is imported unless --trainer is given, and even then only module-level
definitions run (main() is guarded by __name__).

USAGE
  python inspect_ckpt_and_api.py ^
     --trainer     "C:\\Users\\atytchino\\PycharmProjects\\WACV\\20260707-Trainer_MULTIBIT_v13.py" ^
     --system_ckpt "E:\\RUNS\\TLD_tile64_tl05_s0\\checkpoints\\wm_system_e006.pth" ^
     --c2_ckpt     "E:\\RUNS\\TLD_tile64_tl05_s0\\checkpoints\\c2_eval_e006.pth" ^
     --out "E:\\RUNS\\EVAL\\introspect.txt"

Then paste the output (or attach the .txt) back into the chat.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

import torch

OUT = io.StringIO()


def P(msg=""):
    print(msg, flush=True)
    OUT.write(str(msg) + "\n")


def H(title):
    P("\n" + "=" * 76)
    P(title)
    P("=" * 76)


def safe_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def describe(v, depth=0, max_items=8):
    """One-line description of an arbitrary checkpoint value."""
    if torch.is_tensor(v):
        return f"Tensor{tuple(v.shape)} {v.dtype}"
    if isinstance(v, dict):
        # a state_dict?
        if v and all(torch.is_tensor(x) for x in v.values()):
            n = sum(x.numel() for x in v.values())
            keys = list(v.keys())
            return (f"state_dict: {len(v)} tensors, {n:,} params | "
                    f"first={keys[0]} last={keys[-1]}")
        return f"dict({len(v)} keys)"
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__}(len={len(v)})"
    s = repr(v)
    return s if len(s) <= 90 else s[:87] + "..."


# ------------------------------------------------------------------ checkpoints
def dump_ckpt(path, label):
    H(f"{label}: {path}")
    p = Path(path)
    if not p.exists():
        P("  !! FILE NOT FOUND")
        return None
    P(f"  size: {p.stat().st_size / 1e6:.1f} MB")
    ck = safe_load(path)
    if not isinstance(ck, dict):
        P(f"  top-level is not a dict but {type(ck)}")
        return ck
    P(f"  TOP-LEVEL KEYS ({len(ck)}):")
    for k in ck:
        P(f"    {k:28s} -> {describe(ck[k])}")
    return ck


def dump_metrics(ck):
    """The checkpoint records its own last_val / last_realval -> confirms E6 numbers."""
    H("E6 RECORDED METRICS (straight from the checkpoint — no GPU needed)")
    for key in ("last_val", "last_realval"):
        blob = ck.get(key)
        P(f"\n  [{key}]")
        if blob is None:
            P("    (absent)")
            continue
        if isinstance(blob, dict):
            for k, v in blob.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    P(f"    {k:34s} = {v}")
                else:
                    P(f"    {k:34s} : {describe(v)}")
        else:
            P(f"    {describe(blob)}")


# ------------------------------------------------------------------ config
HARNESS_NEEDS = [
    # the harness sets these
    "mode", "system_ckpt", "c2_eval_ckpt", "pub_collage_enable", "wm_export_sample_every",
    # the harness reports / relies on these
    "n_bits", "msg_bridge", "tile_eps", "transfer_lam", "image_size",
    "decoder_norm", "alpha_msg", "msg_inject", "gate_strength",
    "train_root", "val_root", "ae_ckpt", "c1_ckpt", "batch_size", "num_workers",
]


def dump_cfg(ck):
    H("EMBEDDED cfg  (what the harness rebuilds the config from)")
    cfg = ck.get("cfg")
    if not isinstance(cfg, dict):
        P(f"  !! no usable 'cfg' in checkpoint (got {type(cfg)}) — the harness CANNOT rebuild config")
        return None
    P(f"  cfg has {len(cfg)} keys\n")
    P("  -- fields the harness depends on --")
    for k in HARNESS_NEEDS:
        if k in cfg:
            P(f"    {k:26s} = {describe(cfg[k])}")
        else:
            P(f"    {k:26s} = <<< MISSING >>>")
    P("\n  -- all cfg keys (sorted) --")
    keys = sorted(cfg.keys())
    for i in range(0, len(keys), 4):
        P("    " + "  ".join(f"{k:<26s}" for k in keys[i:i + 4]))
    return cfg


# ------------------------------------------------------------------ trainer API
SOURCE_PATTERNS = {
    "val_loader iteration": r"for\s+(.+?)\s+in\s+self\.val_loader",
    "c2_eval signature": r"def\s+c2_eval\s*\([^)]*\)",
    "c1 attribute assign": r"self\.c1\s*=\s*.+",
    "c1 call sites": r"self\.c1\((.*?)\)",
    "trainer class": r"class\s+WatermarkTrainer",
    "config class": r"class\s+TrainConfig",
    "load_system_ckpt": r"def\s+load_system_ckpt\s*\([^)]*\)",
    "load_c2_eval_ckpt": r"def\s+load_c2_eval_ckpt\s*\([^)]*\)",
    "synth_variants_nograd": r"def\s+synth_variants_nograd\s*\(",
    "_tile_reader assign": r"self\._tile_reader\s*=\s*.+",
    "tile reader call": r"self\._tile_reader\((.*?)\)",
    "msg_bits key": r"syn\[.msg_bits.\]",
}


def scan_source(trainer_path):
    H("TRAINER SOURCE SCAN (v13 reality vs the harness's assumptions)")
    p = Path(trainer_path)
    if not p.exists():
        P("  !! trainer not found")
        return
    src = p.read_text(encoding="utf-8", errors="replace")
    P(f"  file: {p.name}  ({len(src):,} chars, {src.count(chr(10)):,} lines)")
    for label, pat in SOURCE_PATTERNS.items():
        hits = re.findall(pat, src)
        P(f"\n  [{label}]  matches={len(hits)}")
        seen = []
        for h in hits:
            h = h.strip() if isinstance(h, str) else str(h)
            if h and h not in seen:
                seen.append(h)
            if len(seen) >= 4:
                break
        for s in seen:
            P(f"      {s[:100]}")


def import_and_check(trainer_path, cfg_from_ckpt):
    H("TrainConfig COMPATIBILITY (importing the real trainer)")
    import importlib.util
    import dataclasses
    p = Path(trainer_path)
    sys.path.insert(0, str(p.parent))
    try:
        spec = importlib.util.spec_from_file_location("wacv_trainer_probe", str(p))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["wacv_trainer_probe"] = mod
        spec.loader.exec_module(mod)
    except Exception as e:
        P(f"  !! import FAILED: {type(e).__name__}: {e}")
        P("     (the harness imports the trainer the same way — this would break it too)")
        return

    P("  import: OK")
    TC = getattr(mod, "TrainConfig", None)
    WT = getattr(mod, "WatermarkTrainer", None)
    P(f"  TrainConfig present     : {TC is not None}")
    P(f"  WatermarkTrainer present: {WT is not None}")
    if TC is None:
        return

    fields = {f.name for f in dataclasses.fields(TC)}
    P(f"  TrainConfig fields: {len(fields)}")

    P("\n  -- harness-required fields present in TrainConfig? --")
    for k in HARNESS_NEEDS:
        P(f"    {k:26s} : {'OK' if k in fields else '<<< ABSENT — harness would crash >>>'}")

    if cfg_from_ckpt:
        ck_keys = set(cfg_from_ckpt)
        dropped = sorted(ck_keys - fields)
        defaulted = sorted(fields - ck_keys)
        P(f"\n  -- checkpoint cfg keys NOT in TrainConfig (harness drops these): {len(dropped)}")
        for k in dropped[:20]:
            P(f"      {k}")
        P(f"  -- TrainConfig fields NOT in checkpoint cfg (take defaults): {len(defaulted)}")
        for k in defaulted[:20]:
            P(f"      {k}")

    if WT is not None:
        P("\n  -- methods the harness calls --")
        for m in ("_to01", "_luma01", "_apply_prod_padding_wipe", "synth_variants_nograd",
                  "c2_eval", "load_system_ckpt", "load_c2_eval_ckpt", "validate"):
            P(f"    {m:28s} : {'OK' if hasattr(WT, m) else '<<< ABSENT >>>'}")


def sweep_epochs(ckpt_dir):
    """Read every wm_system_eXXX.pth in a folder and tabulate its recorded last_val.

    This is the ONLY way to get a per-epoch table that is guaranteed to come from a
    SINGLE run: console logs from different runs look alike and are trivial to mix up.
    No GPU, no data, seconds.
    """
    H(f"PER-EPOCH SWEEP: {ckpt_dir}")
    d = Path(ckpt_dir)
    if not d.exists():
        P("  !! folder not found")
        return
    files = sorted(d.glob("wm_system_e*.pth"))
    if not files:
        P("  !! no wm_system_e*.pth found")
        return
    P(f"  found {len(files)} checkpoint(s)\n")
    hdr = (f"  {'ep':>3} {'BER%':>7} {'msgAcc%':>8} {'GAPg':>8} {'RAW%':>7} {'BOTH%':>7} "
           f"{'GAPng':>7} {'det%':>7} {'dWmMu':>7} {'C1raw%':>7} {'C1wm%':>7}")
    P(hdr)
    P("  " + "-" * (len(hdr) - 2))
    rows = []
    for f in files:
        try:
            ck = safe_load(f)
        except Exception as e:
            P(f"  {f.name}: LOAD FAILED {e}")
            continue
        ep = ck.get("epoch", "?")
        lv = ck.get("last_val") or {}
        if not lv:
            P(f"  {ep:>3}  (no last_val recorded)")
            continue

        def g(k, mul=100.0, default=float("nan")):
            v = lv.get(k, default)
            try:
                return float(v) * mul
            except Exception:
                return float("nan")

        row = dict(
            epoch=ep, ber=g("val_ber"), msg=g("val_msg_acc"), gapg=g("gap_both_raw"),
            raw=g("acc_raw"), both=g("acc_both"), gapng=g("gap_both_raw_ng"),
            det=g("det_acc_raw"), dmu=g("wm_mu_gap_raw", 1.0),
            c1r=g("val_c1_acc_raw"), c1w=g("val_c1_acc_both"),
        )
        rows.append(row)
        P(f"  {str(ep):>3} {row['ber']:>7.3f} {row['msg']:>8.2f} {row['gapg']:>+8.2f} "
          f"{row['raw']:>7.2f} {row['both']:>7.2f} {row['gapng']:>+7.2f} {row['det']:>7.2f} "
          f"{row['dmu']:>+7.3f} {row['c1r']:>7.2f} {row['c1w']:>7.2f}")

    # correlation gate-strength vs BER, computed WITHIN this single run
    post = [r for r in rows if isinstance(r["epoch"], int) and r["epoch"] >= 4]
    if len(post) >= 3:
        import math

        def pearson(x, y):
            n = len(x)
            mx, my = sum(x) / n, sum(y) / n
            num = sum((a - mx) * (b - my) for a, b in zip(x, y))
            den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
            return num / den if den else float("nan")

        def spearman(x, y):
            def rank(v):
                order = sorted(range(len(v)), key=lambda i: v[i])
                r = [0] * len(v)
                for pos, i in enumerate(order):
                    r[i] = pos + 1
                return r
            return pearson(rank(x), rank(y))

        dmu = [r["dmu"] for r in post]
        ber = [r["ber"] for r in post]
        P(f"\n  POST-CONVERGENCE (epochs >= 4, n={len(post)}) — SAME RUN, no mixing:")
        P(f"    Pearson (dWmMu, BER)  = {pearson(dmu, ber):+.3f}")
        P(f"    Spearman(dWmMu, BER)  = {spearman(dmu, ber):+.3f}")
        P("    (a strong positive value supports gate-growth interference as the drift cause)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system_ckpt", default="")
    ap.add_argument("--c2_ckpt", default="")
    ap.add_argument("--trainer", default="")
    ap.add_argument("--sweep", default="", help="checkpoints dir -> per-epoch table from ONE run")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    P("inspect_ckpt_and_api.py — read-only. Nothing is trained or modified.")
    P(f"torch {torch.__version__} | python {sys.version.split()[0]}")

    if a.sweep:
        sweep_epochs(a.sweep)

    ck = None
    cfg = None
    if a.system_ckpt:
        ck = dump_ckpt(a.system_ckpt, "SYSTEM CHECKPOINT (watermark generator)")
        if isinstance(ck, dict):
            cfg = dump_cfg(ck)
            dump_metrics(ck)
            ctrl = ck.get("ctrl")
            if isinstance(ctrl, dict):
                H("CONTROLLER STATE restored by load_system_ckpt (drives the watermark amplitude)")
                for k, v in ctrl.items():
                    P(f"  {k:28s} = {v}")

    if a.c2_ckpt:
        ck2 = dump_ckpt(a.c2_ckpt, "C2 CHECKPOINT (the gate)")
        if isinstance(ck2, dict):
            H("C2 CHECKPOINT — non-tensor metadata")
            for k, v in ck2.items():
                if isinstance(v, dict) and v and all(torch.is_tensor(x) for x in v.values()):
                    continue
                if k == "meta" and isinstance(v, dict):
                    P(f"  [meta]")
                    for mk, mv in v.items():
                        P(f"    {mk:32s} = {describe(mv)}")
                    continue
                P(f"  {k:28s} = {describe(v)}")

    if a.trainer:
        scan_source(a.trainer)
        import_and_check(a.trainer, cfg)

    H("DONE")
    P("Paste this output (or attach the --out file) back into the chat.")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(OUT.getvalue(), encoding="utf-8")
        print(f"\n[OUT] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
