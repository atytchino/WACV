#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
patch_L_transfer.py — ONE change, ONE variable: repair L_transfer.

WHY
---
The overnight eval measured forgery success of 99.0-100.7% at every epoch: pasting
image A's watermark residual onto image B passes the gate as well as a real mark.
The cause is in L_transfer itself — the loss that was supposed to BE this defense.

Two distinct defects, both in the same block:

  1. NO GRADIENT REACHES THE GENERATOR.
     The transplant is built entirely inside `with torch.no_grad()` from a detached
     delta, so transplantN carries no grad_fn. Only C2's wm_head ever learns from
     this loss. The generator (g_lat / g_64 / ContentEncoder / FiLM) is never
     pushed to make delta content-specific — so the watermark was never
     content-entangled, despite the comment claiming exactly that.
     Side effect: it also explains why transfer_lam 0.25/0.35/0.50 produced
     identical BER curves (the A1 sweep) — the loss is nearly inert.

  2. THE PERMUTATION HAS FIXED POINTS.
     `torch.randperm(B)` maps i -> i for an expected 1 element per batch, for ANY B.
     When that happens transplant01[i] == both01[i] — a LEGITIMATE watermark — and
     the loss tells C2 "do not detect this", fighting the main detection objective
     on ~1 of every 4 samples at B=4. This is the likely reason a higher
     transfer_lam gave a slightly WEAKER gate at every epoch.

WHAT THE FIX DOES
-----------------
Lets the gradient through to the generator and guarantees a derangement, so the
generator's objective becomes exactly what the paper claims:
    "delta must be DETECTED on its own content and NOT detected on other content"
which is content-entanglement.

WATCH DURING THE RUN
--------------------
The generator can now satisfy L_transfer by driving delta -> 0 (undetectable
everywhere), which would kill the gate. Monitor GAPg: if it collapses, lower
transfer_lam rather than reverting the fix.

USAGE
  python patch_L_transfer.py --trainer "C:\\...\\20260707-Trainer_MULTIBIT_v13.py"
  python patch_L_transfer.py --trainer "..." --revert
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

OLD = '''        L_transfer = both01.new_tensor(0.0)
        if epoch >= _t_start and B > 1 and transfer_lam > 0.0:
            with torch.no_grad():
                delta = (both01.detach() - x01.detach())  # [B,C,H,W] WM residual
                idx_perm = torch.randperm(B, device=x01.device)
                transplant01 = (x01[idx_perm] + delta).clamp(0.0, 1.0)
                transplantN = transplant01 * 2.0 - 1.0
            try:
                _, wm_transplant, _ = self.c2(transplantN, gate=True)
                t_zeros = torch.zeros_like(wm_transplant)  # must NOT detect WM
                L_transfer = F.binary_cross_entropy_with_logits(wm_transplant, t_zeros)
            except Exception:
                L_transfer = both01.new_tensor(0.0)'''

NEW = '''        L_transfer = both01.new_tensor(0.0)
        if epoch >= _t_start and B > 1 and transfer_lam > 0.0:
            # FIX-1 (gradient): the transplant must stay DIFFERENTIABLE w.r.t. the
            # generator. The previous version built it under torch.no_grad() from a
            # detached delta, so only C2's wm_head learned from this loss and the
            # generator was never pushed to make delta content-specific -- i.e. the
            # watermark was never content-entangled at all. Keep x01 detached (it is
            # data), but let delta carry gradient back into g_lat / g_64 / FiLM.
            delta = both01 - x01.detach()                      # differentiable
            # FIX-2 (derangement): torch.randperm has an expected 1 fixed point per
            # batch for ANY B. A fixed point makes transplant01[i] == both01[i] -- a
            # LEGITIMATE watermark -- and the loss then says "do not detect this",
            # fighting the detection objective on ~1 in 4 samples at B=4. A random
            # cyclic shift by 1..B-1 is a guaranteed derangement.
            _shift = int(torch.randint(1, B, (1,), device=x01.device).item())
            idx_perm = (torch.arange(B, device=x01.device) + _shift) % B
            transplant01 = (x01[idx_perm].detach() + delta).clamp(0.0, 1.0)
            transplantN = transplant01 * 2.0 - 1.0
            if not transplantN.requires_grad:
                print("[L_TRANSFER][WARN] no gradient path to the generator -- "
                      "the fix is not active", flush=True)
            try:
                _, wm_transplant, _ = self.c2(transplantN, gate=True)
                t_zeros = torch.zeros_like(wm_transplant)  # must NOT detect WM
                L_transfer = F.binary_cross_entropy_with_logits(wm_transplant, t_zeros)
            except Exception:
                L_transfer = both01.new_tensor(0.0)'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--revert", action="store_true", help="restore the newest .bak_Ltransfer_*")
    a = ap.parse_args()

    p = Path(a.trainer)
    if not p.exists():
        raise SystemExit(f"[FATAL] not found: {p}")

    if a.revert:
        baks = sorted(p.parent.glob(p.name + ".bak_Ltransfer_*"))
        if not baks:
            raise SystemExit("[FATAL] no backup found")
        shutil.copy2(baks[-1], p)
        print(f"[REVERT] restored from {baks[-1].name}")
        return

    src = p.read_text(encoding="utf-8")

    if "FIX-1 (gradient)" in src:
        raise SystemExit("[SKIP] already patched (found the FIX-1 marker).")

    n = src.count(OLD)
    if n == 0:
        print("[FATAL] the expected L_transfer block was not found verbatim.")
        print("        Your v13 may differ from the copy this patch was written against.")
        print("        Locate it with:")
        print('          Select-String -Path <trainer> -Pattern "L_transfer: simulate" -Context 0,16')
        print("        and paste the block back into the chat so the patch can be re-aimed.")
        raise SystemExit(1)
    if n > 1:
        raise SystemExit(f"[FATAL] the block appears {n} times — refusing to guess.")

    bak = p.with_suffix(p.suffix + f".bak_Ltransfer_{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(p, bak)
    p.write_text(src.replace(OLD, NEW), encoding="utf-8")

    print(f"[OK] patched : {p.name}")
    print(f"[OK] backup  : {bak.name}")
    print()
    print("Changed, and ONLY this:")
    print("  1. gradient now reaches the generator (no_grad / detach removed)")
    print("  2. permutation is now a guaranteed derangement (no self-transplants)")
    print()
    print("Everything else — flags, capacity, tile_eps, transfer_lam — is untouched,")
    print("so the next run isolates exactly one variable: does a working L_transfer")
    print("actually buy content-entanglement?")
    print()
    print("WATCH: the generator can now satisfy this loss by driving delta -> 0,")
    print("       which would kill the gate. If GAPg collapses, lower transfer_lam")
    print("       rather than reverting the fix.")


if __name__ == "__main__":
    main()
