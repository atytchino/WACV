#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_wm_system.py — EVALUATION-ONLY harness for the frozen watermarking system.

NOTHING here trains, and NOTHING here modifies the four interlocked files
(trainer / wm_message.py / wm_decoder.py / wm_tile.py). This script only
IMPORTS the trainer and reuses its own proven code paths:
    _to01 / _luma01 / _apply_prod_padding_wipe / synth_variants_nograd / c2_eval
so that whatever it measures is measured exactly the way validate() measures it.

The training config is recovered FROM THE CHECKPOINT ITSELF (the trainer saves
`cfg: vars(self.cfg)` inside wm_system_eXXX.pth), so there is no chance of a
flag mismatch between this harness and the run that produced the checkpoint.

MODES
-----
  transplant : *** THE LOAD-BEARING TEST ***
               Take the watermark residual d_A = wm_A - A from image A and paste
               it onto a DIFFERENT image B: B' = B + d_A. Then ask C2's gate:
               does B' get access?
                 - acc(B') ~ acc(wm_B)  -> forgery SUCCEEDS -> content-binding FAILS
                 - acc(B') ~ acc(B)     -> forgery FAILS     -> content-binding WORKS
               NOTE: the *message* tile is a fixed +/-eps pixel pattern and is NOT
               content-bound (anyone can copy it). The forgery defense lives in the
               GATE generator (g_lat/g_64, FiLM-conditioned on ContentEncoder).
               So this test is measured on the GATE, which is also BLIND -> the
               comparison against blind baselines is fair, and no blind-decoder
               (C0) work is required first.

  validate   : the standalone validator demo. Originals -> gate destroys them;
               watermarked -> gate lets them through. Blind (no originals needed
               by the gate/detector). Also reports the C1 naive-classifier
               baseline, which is what proves the watermark does not damage the
               image.

  grayscale  : same as `validate`, but the input is converted to grayscale
               ON THE FLY (no grayscale dataset is ever written to disk --
               it is a deterministic transform). Reports the C1 baseline too, so
               a BOTH drop can be attributed to the classifier's colour blindness
               rather than to the watermark (the same defensive move as GAPng).

  hist       : per-image bit-error histogram -> tells us the ECC strength `t`
               needed, instead of inferring it from the mean.

  all        : run every mode.

USAGE
-----
  python eval_wm_system.py ^
      --trainer      "C:\\Users\\atytchino\\PycharmProjects\\WACV\\20260707-Trainer_MULTIBIT_v13.py" ^
      --system_ckpt  "E:\\RUNS\\TLD_tile64_tl05_s0\\checkpoints\\wm_system_e006.pth" ^
      --c2_ckpt      "E:\\RUNS\\TLD_tile64_tl05_s0\\checkpoints\\c2_eval_e006.pth" ^
      --mode all --max_batches 60 ^
      --out "E:\\RUNS\\EVAL\\e006_eval.json"
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import torch

_TM = None  # set by load_trainer_module; used by the imperceptibility metric helpers


# ----------------------------------------------------------------------------- utils
def _p(msg: str) -> None:
    print(msg, flush=True)


def _install_torch_load_compat():
    """PyTorch >= 2.6 flipped `torch.load`'s default to weights_only=True.

    The trainer's own load_system_ckpt() (v13 line 1893) calls torch.load WITHOUT
    that argument, and our checkpoints embed `cfg` containing pathlib.WindowsPath
    objects -> UnpicklingError. That makes load_system_ckpt broken on torch >= 2.6
    for ANY use (including --system_ckpt resume), independently of this harness.

    We fix it from the outside: `torch` is a singleton module, so re-binding
    torch.load here also applies inside the trainer. The trainer file is NOT
    touched. weights_only=False is safe here: these checkpoints are the user's own
    training artefacts, not untrusted downloads.
    """
    try:
        import pathlib
        torch.serialization.add_safe_globals(
            [pathlib.WindowsPath, pathlib.PosixPath, pathlib.PurePath, pathlib.Path]
        )
    except Exception:
        pass  # older torch has no add_safe_globals; the wrapper below covers it

    if getattr(torch.load, "_wm_compat", False):
        return
    _orig = torch.load

    def _load(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig(*a, **kw)

    _load._wm_compat = True
    torch.load = _load
    _p("[COMPAT] torch.load patched to weights_only=False (trainer's loader predates the "
       "PyTorch 2.6 default flip; trainer file untouched)")


def load_trainer_module(trainer_path: str):
    """Import the trainer by path.

    The trainer's filename starts with a digit (20260707-...), so it cannot be
    imported with a normal `import` statement -- importlib by file location is
    mandatory here.
    """
    p = Path(trainer_path)
    if not p.exists():
        raise SystemExit(f"[FATAL] trainer not found: {p}")
    # the trainer's own directory must be importable (wm_tile, wm_message, AE...)
    sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location("wacv_trainer", str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wacv_trainer"] = mod
    spec.loader.exec_module(mod)
    global _TM
    _TM = mod
    return mod


def build_trainer(TM, system_ckpt: str, c2_ckpt: str, overrides: dict):
    """Rebuild the exact training config from the checkpoint, then instantiate.

    The trainer auto-loads both checkpoints in __init__ when cfg.system_ckpt /
    cfg.c2_eval_ckpt are set, so we do not re-implement any loading logic.
    """
    try:
        ck = torch.load(system_ckpt, map_location="cpu", weights_only=False)
    except TypeError:                      # torch < 2.4 has no weights_only kwarg
        ck = torch.load(system_ckpt, map_location="cpu")

    cfg_dict = dict(ck.get("cfg", {}) or {})
    if not cfg_dict:
        raise SystemExit("[FATAL] checkpoint has no embedded 'cfg' -- cannot rebuild config.")

    field_names = {f.name for f in dataclasses.fields(TM.TrainConfig)}
    kept = {k: v for k, v in cfg_dict.items() if k in field_names}
    dropped = sorted(set(cfg_dict) - field_names)
    if dropped:
        _p(f"[CFG] ignoring {len(dropped)} checkpoint key(s) not in TrainConfig: {dropped[:6]}{'...' if len(dropped) > 6 else ''}")

    # mode=train so that C2 is actually constructed (infer mode skips C2 entirely,
    # and without C2 there is no gate to measure).
    kept["mode"] = "train"
    kept["system_ckpt"] = Path(system_ckpt)
    kept["c2_eval_ckpt"] = Path(c2_ckpt)
    # --- keep the harness side-effect free -------------------------------------
    # cfg["out_root"] points at the ORIGINAL run folder; instantiating the trainer
    # derives ckpt_root from it. Never let an eval run write anywhere near the
    # frozen reference run.
    kept["pub_collage_enable"] = False
    kept["wm_export_sample_every"] = 0
    kept["export_wm_dataset"] = False
    kept["collage_every"] = 0
    kept["real_val_every"] = 0
    # --- Windows + importlib: MUST be 0 -----------------------------------------
    # DataLoader workers start via `spawn` on Windows; the child re-imports the
    # parent module to unpickle the dataset. We loaded the trainer under the
    # synthetic name "wacv_trainer" (its real filename starts with a digit, so it
    # is not importable normally) -> the child raises
    #   ModuleNotFoundError: No module named 'wacv_trainer'
    # num_workers=0 keeps everything in-process. The trainer already guards
    # persistent_workers/prefetch_factor on num_workers>0, so this is safe, and
    # its own infer_loader hardcodes 0 for the same reason. Eval is GPU-bound
    # anyway, so nothing is lost.
    kept["num_workers"] = 0
    kept.update(overrides)                      # out_root override arrives here

    cfg = TM.TrainConfig(**kept)
    _p(f"[CFG] rebuilt from checkpoint: n_bits={getattr(cfg, 'n_bits', '?')} "
       f"msg_bridge={getattr(cfg, 'msg_bridge', '?')} tile_eps={getattr(cfg, 'tile_eps', '?')} "
       f"transfer_lam={getattr(cfg, 'transfer_lam', '?')} image_size={getattr(cfg, 'image_size', '?')}")
    _p(f"[CFG] eval overrides in cfg: eval_eps={getattr(cfg, 'eval_eps', '?')} "
       f"eval_r_skip={getattr(cfg, 'eval_r_skip', '?')}  "
       f"(0 / -1 mean 'use the controller state restored from the checkpoint')")
    _p(f"[CFG] out_root redirected to: {cfg.out_root}")
    tr = TM.WatermarkTrainer(cfg)
    _p(f"[CKPT] epoch in checkpoint: {ck.get('epoch', '?')}")
    _p(f"[CTRL] restored: eps={getattr(tr.ctrl, 'eps', '?'):.4f} "
       f"r_skip={getattr(tr.ctrl, 'r_skip', '?'):.4f}")
    return tr, ck


def print_reference(ck):
    """The checkpoint carries the metrics of the very validate() that produced it.

    Printing them lets us CROSS-CHECK the harness: if our `validate` mode does not
    land on these numbers, the harness is not faithful to the training-time path.
    """
    lv = ck.get("last_val")
    if not isinstance(lv, dict):
        return None
    _p("\n[REFERENCE] metrics recorded inside this checkpoint (what validate() measured at save time):")
    for k in ("acc_raw", "acc_both", "gap_both_raw", "gap_both_raw_ng", "det_acc_raw",
              "val_ber", "val_msg_acc", "val_c1_acc_raw", "val_c1_acc_both", "val_c1_drop",
              "wm_mu_gap_raw"):
        if k in lv:
            v = lv[k]
            _p(f"    {k:20s} = {v * 100:8.3f}%" if isinstance(v, float) and abs(v) <= 1.5
               else f"    {k:20s} = {v}")
    return lv


def to_gray01(x01: torch.Tensor) -> torch.Tensor:
    """Deterministic BT.601 luma, replicated to 3 channels. Never materialised to disk."""
    y = 0.299 * x01[:, 0:1] + 0.587 * x01[:, 1:2] + 0.114 * x01[:, 2:3]
    return y.repeat(1, 3, 1, 1)


def _logits(out):
    """C1 / C2 return (logits, wm_logit, x4); normalise to logits."""
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def c1_logits(tr, xN):
    """Run the NAIVE classifier.

    CRITICAL: ResNet34LF_BN.forward has `gate: bool = True` as its DEFAULT, and the
    trainer itself always calls C1 as `self.c1(xN, gate=False)`. C1 is the *naive*
    safety-circuit baseline — gating it would silently produce wrong numbers.
    """
    return _logits(tr.c1(xN, gate=False))


def _synth(tr, x01, valid_mask, variants=("base", "both")):
    return tr.synth_variants_nograd(
        x01, valid_mask=valid_mask, epoch=0,
        variants=variants, k_factor=1.0, varpercent=False, mode="eval",
    )


def _shuffled_val_loader(tr, seed=1234):
    """A class-balanced subset needs shuffling.

    The trainer builds val_loader with shuffle=False and ImageFolder walks the
    classes alphabetically, so the first N batches are ALL ONE CLASS. Any
    --max_batches run against the unshuffled loader measures a single class and
    is not comparable to the checkpoint's full-val numbers.
    """
    from torch.utils.data import DataLoader
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        tr.val_ds,
        batch_size=int(getattr(tr.cfg, "batch_size", 4)),
        shuffle=True,
        num_workers=0,
        pin_memory=(tr.device.type == "cuda"),
        generator=g,
    )


def _iter_val(tr, max_batches):
    """The trainer's val_loader yields (xN, valid_mask, y, paths) -- in that order."""
    for i, batch in enumerate(tr.val_loader):
        if max_batches and i >= max_batches:
            break
        if len(batch) == 4:
            xN, valid_mask, y, paths = batch
        else:
            raise SystemExit(f"[FATAL] unexpected val batch arity: {len(batch)}")
        yield (xN.to(tr.device), valid_mask.to(tr.device), y.to(tr.device), paths)


# ----------------------------------------------------------------------------- modes
@torch.no_grad()
def run_transplant(tr, max_batches=None, gray=False):
    """*** LOAD-BEARING *** Does a foreign watermark residual buy access?"""
    _p("\n" + "=" * 78)
    _p("TRANSPLANT / CONTENT-BINDING TEST" + ("  [grayscale]" if gray else ""))
    _p("  paste d_A (from image A) onto image B -> does B' pass the gate?")
    _p("=" * 78)

    n = 0
    ok_raw = ok_both = ok_forg = 0
    wm_raw_s = wm_both_s = wm_forg_s = 0.0
    for xN, valid_mask, y, _paths in _iter_val(tr, max_batches):
        x01 = tr._to01(xN)
        if gray:
            x01 = to_gray01(x01)
            xN = x01 * 2.0 - 1.0
        if x01.shape[0] < 2:
            continue                                   # need >=2 images to swap residuals

        syn = _synth(tr, x01, valid_mask, variants=("base", "both"))
        both01 = syn["both01"].clamp(0, 1)

        delta = both01 - x01                           # the full watermark residual
        delta_foreign = torch.roll(delta, shifts=1, dims=0)   # image i gets image (i-1)'s residual
        forged01 = (x01 + delta_foreign).clamp(0, 1)   # <- the forgery

        rawN = tr._apply_prod_padding_wipe(xN, valid_mask)
        bothN = tr._apply_prod_padding_wipe(both01 * 2 - 1, valid_mask)
        forgN = tr._apply_prod_padding_wipe(forged01 * 2 - 1, valid_mask)

        z_raw, wm_raw, _ = tr.c2_eval(rawN, gate=True)
        z_both, wm_both, _ = tr.c2_eval(bothN, gate=True)
        z_forg, wm_forg, _ = tr.c2_eval(forgN, gate=True)

        b = int(y.numel())
        n += b
        ok_raw += int((z_raw.argmax(1) == y).sum())
        ok_both += int((z_both.argmax(1) == y).sum())
        ok_forg += int((z_forg.argmax(1) == y).sum())
        wm_raw_s += float(wm_raw.mean()) * b
        wm_both_s += float(wm_both.mean()) * b
        wm_forg_s += float(wm_forg.mean()) * b

    if n == 0:
        raise SystemExit("[FATAL] no batches evaluated.")

    a_raw, a_both, a_forg = 100 * ok_raw / n, 100 * ok_both / n, 100 * ok_forg / n
    d_raw, d_both, d_forg = wm_raw_s / n, wm_both_s / n, wm_forg_s / n

    _p(f"  images                    : {n}")
    _p(f"  gate acc | originals      : {a_raw:6.2f}%   (destroyed = gate working)")
    _p(f"  gate acc | watermarked    : {a_both:6.2f}%   (legit access)")
    _p(f"  gate acc | TRANSPLANTED   : {a_forg:6.2f}%   <-- the answer")
    _p(f"  detector mu | orig/wm/forged : {d_raw:.3f} / {d_both:.3f} / {d_forg:.3f}")

    # Where does the forgery land between "original" and "legit watermarked"?
    span = max(a_both - a_raw, 1e-6)
    success = 100.0 * (a_forg - a_raw) / span          # 0% = fully blocked, 100% = fully forged
    _p(f"\n  FORGERY SUCCESS RATE      : {success:6.2f}%   "
       f"(0 = content-binding blocks it, 100 = binding provides no defense)")
    if success < 20:
        _p("  => content-binding HOLDS. The forgery narrative is supported.")
    elif success > 70:
        _p("  => content-binding DOES NOT HOLD. The forgery axis must be rethought.")
    else:
        _p("  => PARTIAL. Binding helps but does not block; report honestly.")

    return {"mode": "transplant" + ("_gray" if gray else ""), "n": n,
            "acc_originals": a_raw, "acc_watermarked": a_both, "acc_transplanted": a_forg,
            "det_mu_orig": d_raw, "det_mu_wm": d_both, "det_mu_forged": d_forg,
            "forgery_success_rate": success}


@torch.no_grad()
def run_validate(tr, max_batches=None, gray=False):
    """The validator demo: originals destroyed, watermarked working. Fully blind."""
    tag = "GRAYSCALE VALIDATOR" if gray else "VALIDATOR"
    _p("\n" + "=" * 78)
    _p(tag + "  (gate + detector are blind: no originals required)")
    _p("=" * 78)

    n = 0
    ok_raw = ok_both = 0
    ok_raw_ng = ok_both_ng = 0
    ok_c1_raw = ok_c1_both = 0
    have_c1 = hasattr(tr, "c1") and tr.c1 is not None
    for xN, valid_mask, y, _paths in _iter_val(tr, max_batches):
        x01 = tr._to01(xN)
        if gray:
            x01 = to_gray01(x01)
            xN = x01 * 2.0 - 1.0

        syn = _synth(tr, x01, valid_mask, variants=("base", "both"))
        both01 = syn["both01"].clamp(0, 1)

        rawN = tr._apply_prod_padding_wipe(xN, valid_mask)
        bothN = tr._apply_prod_padding_wipe(both01 * 2 - 1, valid_mask)

        z_raw, _, _ = tr.c2_eval(rawN, gate=True)
        z_both, _, _ = tr.c2_eval(bothN, gate=True)
        z_raw_ng, _, _ = tr.c2_eval(rawN, gate=False)
        z_both_ng, _, _ = tr.c2_eval(bothN, gate=False)

        b = int(y.numel())
        n += b
        ok_raw += int((z_raw.argmax(1) == y).sum())
        ok_both += int((z_both.argmax(1) == y).sum())
        ok_raw_ng += int((z_raw_ng.argmax(1) == y).sum())
        ok_both_ng += int((z_both_ng.argmax(1) == y).sum())

        if have_c1:
            try:
                ok_c1_raw += int((c1_logits(tr, rawN).argmax(1) == y).sum())
                ok_c1_both += int((c1_logits(tr, bothN).argmax(1) == y).sum())
            except Exception as e:
                have_c1 = False
                _p(f"  [warn] C1 baseline unavailable: {e}")

    a_raw, a_both = 100 * ok_raw / n, 100 * ok_both / n
    a_raw_ng, a_both_ng = 100 * ok_raw_ng / n, 100 * ok_both_ng / n
    _p(f"  images                       : {n}")
    _p(f"  C2 gated   | originals       : {a_raw:6.2f}%")
    _p(f"  C2 gated   | watermarked     : {a_both:6.2f}%")
    _p(f"  GAPg (watermarked-originals) : {a_both - a_raw:+6.2f}pp   <-- the access split")
    _p(f"  GAPng (non-gated branch)     : {a_both_ng - a_raw_ng:+6.2f}pp   "
       f"(~0 proves a destructive gate, not a broken classifier)")
    out = {"mode": "validate" + ("_gray" if gray else ""), "n": n,
           "acc_originals": a_raw, "acc_watermarked": a_both,
           "GAPg": a_both - a_raw, "GAPng": a_both_ng - a_raw_ng}

    if have_c1:
        c1_raw, c1_both = 100 * ok_c1_raw / n, 100 * ok_c1_both / n
        _p(f"  C1 naive   | originals       : {c1_raw:6.2f}%   <-- the ceiling for this data")
        _p(f"  C1 naive   | watermarked     : {c1_both:6.2f}%   (drop {c1_both - c1_raw:+.2f}pp "
           f"=> the watermark does not damage the image)")
        out.update({"c1_originals": c1_raw, "c1_watermarked": c1_both,
                    "c1_drop": c1_both - c1_raw})
        if gray:
            _p(f"\n  READ THIS: if BOTH ({a_both:.2f}%) ~ C1 ceiling ({c1_raw:.2f}%), the gate is")
            _p( "  delivering its FULL potential and only the ceiling moved -- i.e. any drop is")
            _p( "  the classifier's colour blindness, NOT the watermark.")
    return out


@torch.no_grad()
def run_hist(tr, max_batches=None):
    """Per-image bit-error histogram -> sizes the ECC correction strength t."""
    _p("\n" + "=" * 78)
    _p("BIT-ERROR HISTOGRAM  (sizes the ECC strength t)")
    _p("=" * 78)
    if getattr(tr, "_tile_reader", None) is None:
        _p("  [skip] this checkpoint was not trained with --msg_bridge tile.")
        return {"mode": "hist", "skipped": True}

    n_bits = int(getattr(tr.cfg, "n_bits", 0))
    errs = Counter()
    n = 0
    for xN, valid_mask, y, _paths in _iter_val(tr, max_batches):
        x01 = tr._to01(xN)
        syn = _synth(tr, x01, valid_mask, variants=("base", "both"))
        both01 = syn["both01"].clamp(0, 1)
        true_bits = syn["msg_bits"]

        # NOTE: this read is NON-BLIND (it subtracts the original). That is the
        # current reference reader; the blind number will come from C0.
        resid = tr._luma01(both01) - tr._luma01(x01)
        pred = (tr._tile_reader(resid) > 0).float()
        per_img = (pred != true_bits).sum(dim=1).long()
        for e in per_img.tolist():
            errs[int(e)] += 1
        n += int(per_img.numel())

    total_err = sum(k * v for k, v in errs.items())
    clean = errs.get(0, 0)
    _p(f"  images            : {n}   (n_bits={n_bits})")
    _p(f"  BER               : {100.0 * total_err / max(n * n_bits, 1):.3f}%")
    _p(f"  msgAcc (0 errors) : {100.0 * clean / max(n, 1):.2f}%")
    _p("\n  errors/image   images    cumulative msgAcc if ECC corrects t>=this")
    cum = 0
    for k in sorted(errs):
        cum += errs[k]
        _p(f"    t={k:<3d}        {errs[k]:>6d}      {100.0 * cum / n:6.2f}%")
    _p("\n  => pick the smallest t whose cumulative msgAcc is acceptable (~99%).")
    _p("     TrustMark precedent: BCH_5 on 100 raw bits -> ~70 protected.")
    return {"mode": "hist", "n": n, "n_bits": n_bits,
            "ber": 100.0 * total_err / max(n * n_bits, 1),
            "msg_acc": 100.0 * clean / max(n, 1),
            "hist": {str(k): v for k, v in sorted(errs.items())}}


@torch.no_grad()
def run_realval(tr, epoch=6):
    """REALVAL on THIS checkpoint: a fixed post-hoc threshold applied to an
    independent pass, under JPEG-hi / JPEG-lo / resize.

    Why this is needed: real_val_every=4, so REALVAL only ran at E4 and E8 —
    the E6 reference checkpoint carries last_realval=None. Without this, the
    "the split survives JPEG/resize" claim is not established for the operating
    point we actually ship.
    """
    _p("\n" + "=" * 78)
    _p("REALVAL on this checkpoint (independent pass, post-hoc threshold)")
    _p("=" * 78)
    if not hasattr(tr, "validate_real_life"):
        _p("  [skip] trainer has no validate_real_life()")
        return {"mode": "realval", "skipped": True}
    stats = tr.validate_real_life(int(epoch))
    out = {"mode": "realval"}
    if isinstance(stats, dict):
        for k, v in stats.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                out[k] = v
    return out


@torch.no_grad()
def run_gray_attack(tr, max_batches=None):
    """Grayscale conversion as an ATTACK (distinct from watermarking B/W data).

    Scenario: the image was watermarked in colour, then someone converts it to
    black & white. Does the mark survive?

    Theory says it survives EXACTLY: the watermark is a luma offset added equally
    to R,G,B, so
        gray(x + d) = 0.299(R+d) + 0.587(G+d) + 0.114(B+d)
                    = gray(x) + d * (0.299 + 0.587 + 0.114)
                    = gray(x) + d
    i.e. greyscaling is a no-op for a luma-only watermark. This measures whether
    that holds in practice (clamping and the padding wipe could still bite).
    """
    _p("\n" + "=" * 78)
    _p("GRAYSCALE CONVERSION AS AN ATTACK  (watermark in colour -> convert to B/W)")
    _p("=" * 78)
    n = 0
    ok_raw = ok_both = 0
    bit_err = 0
    bits_total = 0
    has_tiles = getattr(tr, "_tile_reader", None) is not None
    for xN, valid_mask, y, _paths in _iter_val(tr, max_batches):
        x01 = tr._to01(xN)
        syn = _synth(tr, x01, valid_mask, variants=("base", "both"))
        both01 = syn["both01"].clamp(0, 1)

        # --- the attack ---
        both_gray = to_gray01(both01)
        raw_gray = to_gray01(x01)

        rawN = tr._apply_prod_padding_wipe(raw_gray * 2 - 1, valid_mask)
        bothN = tr._apply_prod_padding_wipe(both_gray * 2 - 1, valid_mask)
        z_raw, _, _ = tr.c2_eval(rawN, gate=True)
        z_both, _, _ = tr.c2_eval(bothN, gate=True)

        b = int(y.numel())
        n += b
        ok_raw += int((z_raw.argmax(1) == y).sum())
        ok_both += int((z_both.argmax(1) == y).sum())

        if has_tiles:
            resid = tr._luma01(both_gray) - tr._luma01(raw_gray)
            pred = (tr._tile_reader(resid) > 0).float()
            bit_err += int((pred != syn["msg_bits"]).sum())
            bits_total += int(syn["msg_bits"].numel())

    a_raw, a_both = 100 * ok_raw / n, 100 * ok_both / n
    _p(f"  images                     : {n}")
    _p(f"  gate acc | gray originals  : {a_raw:6.2f}%")
    _p(f"  gate acc | gray watermarked: {a_both:6.2f}%")
    _p(f"  GAPg after greyscaling     : {a_both - a_raw:+6.2f}pp")
    out = {"mode": "gray_attack", "n": n, "acc_originals": a_raw,
           "acc_watermarked": a_both, "GAPg": a_both - a_raw}
    if bits_total:
        ber = 100.0 * bit_err / bits_total
        _p(f"  BER after greyscaling      : {ber:6.3f}%   "
           f"(theory: unchanged — greyscaling is a no-op for a luma watermark)")
        out["ber"] = ber
    return out


@torch.no_grad()
def run_universal(tr, max_batches=None):
    """UNI attack — the airtight test.

    Build ONE universal residual by averaging the watermark residual over many
    images, then paste that single δ_universal onto every fresh image and ask the
    gate for access.

    This is stronger evidence than transplant: in transplant δ is a REAL residual
    from a real image, so a defender could argue "the mark is content-bound, it
    just happens to transfer between similar leaves". Here δ_universal belongs to
    NO image — it is the mean of many. If it still opens the gate, there is no
    content-binding of any kind: a single fixed key admits everything, which is
    the worst case for a DRM gate and a clean, unambiguous architectural finding.

    Two passes: pass 1 accumulates the mean residual; pass 2 applies it.
    """
    _p("\n" + "=" * 78)
    _p("UNIVERSAL-KEY (UNI) ATTACK  — one averaged δ, belonging to no image")
    _p("=" * 78)

    # pass 1: accumulate the mean residual
    delta_sum = None
    n_acc = 0
    for xN, valid_mask, y, _paths in _iter_val(tr, max_batches):
        x01 = tr._to01(xN)
        syn = _synth(tr, x01, valid_mask, variants=("base", "both"))
        both01 = syn["both01"].clamp(0, 1)
        d = (both01 - x01).sum(dim=0, keepdim=True)
        delta_sum = d if delta_sum is None else delta_sum + d
        n_acc += x01.shape[0]
    if n_acc == 0:
        raise SystemExit("[FATAL] no images.")
    delta_universal = delta_sum / n_acc                     # [1,C,H,W] — belongs to no image
    _p(f"  built δ_universal from {n_acc} images | "
       f"mean|δ|={delta_universal.abs().mean():.5f} max|δ|={delta_universal.abs().max():.4f}")

    # pass 2: apply the universal δ to fresh images
    n = 0
    ok_raw = ok_uni = 0
    du_raw = du_uni = 0.0
    for xN, valid_mask, y, _paths in _iter_val(tr, max_batches):
        x01 = tr._to01(xN)
        uni01 = (x01 + delta_universal.to(x01.device)).clamp(0, 1)

        rawN = tr._apply_prod_padding_wipe(xN, valid_mask)
        uniN = tr._apply_prod_padding_wipe(uni01 * 2 - 1, valid_mask)
        z_raw, wm_raw, _ = tr.c2_eval(rawN, gate=True)
        z_uni, wm_uni, _ = tr.c2_eval(uniN, gate=True)

        b = int(y.numel())
        n += b
        ok_raw += int((z_raw.argmax(1) == y).sum())
        ok_uni += int((z_uni.argmax(1) == y).sum())
        du_raw += float(wm_raw.mean()) * b
        du_uni += float(wm_uni.mean()) * b

    a_raw, a_uni = 100 * ok_raw / n, 100 * ok_uni / n
    _p(f"  images                       : {n}")
    _p(f"  gate acc | originals         : {a_raw:6.2f}%")
    _p(f"  gate acc | + universal δ     : {a_uni:6.2f}%   <-- the answer")
    _p(f"  detector mu | orig / uni     : {du_raw / n:.3f} / {du_uni / n:.3f}")
    # reuse the same 0..1 normalisation as transplant: 0 = blocked, 100 = fully admitted
    # (denominator is the legit watermarked ceiling, which we approximate by the
    #  best gate accuracy this checkpoint reaches; use acc on watermarked from validate
    #  if available, else fall back to reporting the raw gate accuracy)
    _p(f"\n  UNIVERSAL-KEY ADMISSION      : {a_uni:6.2f}%  gate accuracy with a key that")
    _p( "                                 belongs to NO image.")
    if a_uni > 60:
        _p("  => a single fixed δ admits arbitrary images. NO content-binding of any kind.")
    elif a_uni < 20:
        _p("  => the universal key is REJECTED — some content-dependence exists.")
    else:
        _p("  => partial admission; report honestly.")
    return {"mode": "universal", "n": n, "acc_originals": a_raw,
            "acc_universal": a_uni, "det_mu_orig": du_raw / n, "det_mu_uni": du_uni / n}


@torch.no_grad()
def run_imperceptibility(tr, max_batches=None, gray=False):
    """The mandatory imperceptibility table: PSNR / SSIM / LPIPS / Linf / L2 on VAL.

    Why this exists: the trainer computes PSNR/SSIM only INSIDE the training step on
    TRAIN batches (line ~3630), so they never reach val_stats — the ablation SSIM
    numbers are TRAIN metrics, unusable as-is, and LPIPS exists nowhere. This measures
    watermarked-vs-original distortion on the VAL set, reusing the trainer's own
    psnr/ssim funcs so the numbers match its conventions, and adds LPIPS + Linf/L2.
    """
    tag = "IMPERCEPTIBILITY" + ("  [grayscale]" if gray else "")
    _p("\n" + "=" * 78)
    _p(tag + "  (watermarked vs original, VAL set)")
    _p("=" * 78)

    # LPIPS is a separate package; degrade gracefully if absent
    lpips_fn = None
    try:
        import lpips as _lpips
        lpips_fn = _lpips.LPIPS(net="alex").to(tr.device).eval()
        _p("  LPIPS: using alexnet backbone")
    except Exception as e:
        _p(f"  LPIPS: unavailable ({type(e).__name__}) — install with `pip install lpips`; "
           "reporting PSNR/SSIM/Linf/L2 only")

    n = 0
    psnr_sum = psnr_y_sum = ssim_sum = 0.0
    lpips_sum = 0.0
    linf_sum = l2_sum = 0.0
    for xN, valid_mask, y, _paths in _iter_val(tr, max_batches):
        x01 = tr._to01(xN)
        if gray:
            x01 = to_gray01(x01)
        syn = _synth(tr, x01, valid_mask, variants=("base", "both"))
        both01 = syn["both01"].clamp(0, 1)

        b = int(x01.shape[0])
        n += b
        psnr_sum += float(TM_psnr(tr, both01, x01)) * b
        psnr_y_sum += float(TM_psnr_y(tr, both01, x01)) * b
        ssim_sum += float(TM_ssim_y(tr, both01, x01)) * b

        d = (both01 - x01)
        linf_sum += float(d.abs().flatten(1).max(dim=1).values.mean()) * b
        l2_sum += float(d.flatten(1).norm(dim=1).mean()) * b

        if lpips_fn is not None:
            # LPIPS wants [-1,1]
            lp = lpips_fn(both01 * 2 - 1, x01 * 2 - 1)
            lpips_sum += float(lp.mean()) * b

    out = {"mode": "imperceptibility" + ("_gray" if gray else ""), "n": n,
           "psnr": psnr_sum / n, "psnr_y": psnr_y_sum / n, "ssim_y": ssim_sum / n,
           "linf": linf_sum / n, "l2": l2_sum / n}
    _p(f"  images        : {n}")
    _p(f"  PSNR   (RGB)  : {out['psnr']:6.2f} dB")
    _p(f"  PSNR_y (luma) : {out['psnr_y']:6.2f} dB")
    _p(f"  SSIM_y        : {out['ssim_y']:.4f}")
    _p(f"  Linf          : {out['linf']:.4f}   (max per-pixel change; tile_eps is 0.03)")
    _p(f"  L2 (per image): {out['l2']:.3f}")
    if lpips_fn is not None:
        out["lpips"] = lpips_sum / n
        _p(f"  LPIPS (alex)  : {out['lpips']:.4f}   (lower = more imperceptible)")
    _p("\n  Reference points: PSNR>40 dB / SSIM>0.98 / LPIPS<0.01 read as 'invisible';")
    _p("  30-40 dB / 0.95-0.98 as 'very good'. Report these against the baselines' own numbers.")
    return out


def TM_psnr(tr, a, b):
    return _TM.psnr_torch(a, b).item()


def TM_psnr_y(tr, a, b):
    return _TM.psnr_y_torch(a, b).item()


def TM_ssim_y(tr, a, b):
    return _TM.ssim_y_torch(a, b).item()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Evaluation-only harness (no training, no code changes).")
    ap.add_argument("--trainer", required=True, help="path to 20260707-Trainer_MULTIBIT_v13.py")
    ap.add_argument("--system_ckpt", required=True, help="wm_system_eXXX.pth (the watermark generator)")
    ap.add_argument("--c2_ckpt", required=True, help="c2_eval_eXXX.pth (the conditional classifier)")
    ap.add_argument("--mode", default="all",
                    choices=["all", "transplant", "validate", "grayscale", "hist",
                             "transplant_gray", "realval", "gray_attack", "universal",
                             "imperceptibility", "imperceptibility_gray"])
    ap.add_argument("--cross_class", action="store_true",
                    help="shuffle the val loader so transplant swaps residuals ACROSS classes "
                         "(the honest paper number; default is same-class within-batch roll)")
    ap.add_argument("--max_batches", type=int, default=0, help="0 = whole val set")
    ap.add_argument("--epoch", type=int, default=6, help="epoch label for REALVAL output")
    ap.add_argument("--out_root", default="",
                    help="scratch dir for the trainer's own bookkeeping. Defaults to a "
                         "sibling of --out. NEVER point this at the reference run folder.")
    ap.add_argument("--out", default="", help="optional JSON output path")
    args = ap.parse_args()

    mb = args.max_batches or None
    _install_torch_load_compat()
    TM = load_trainer_module(args.trainer)

    scratch = args.out_root or str(Path(args.out).parent / "_harness_scratch"
                                   if args.out else Path.cwd() / "_harness_scratch")
    tr, ck = build_trainer(TM, args.system_ckpt, args.c2_ckpt,
                           overrides={"out_root": Path(scratch)})
    ref = print_reference(ck)

    if mb:
        try:
            tr.val_loader = _shuffled_val_loader(tr)
            _p(f"\n[SUBSET] --max_batches {args.max_batches} -> switched to a SHUFFLED val "
               f"loader (seed 1234) so the subset spans all classes.")
            _p("         Without this the trainer's shuffle=False loader would hand back the "
               "first N images,\n         which ImageFolder orders by class — i.e. a single-class "
               "sample, not comparable to the\n         checkpoint's full-val record.")
        except Exception as e:
            _p(f"[SUBSET] could not build a shuffled loader ({e}); subset will be class-biased.")
    elif args.cross_class:
        try:
            tr.val_loader = _shuffled_val_loader(tr)
            _p("\n[CROSS-CLASS] shuffled val loader (seed 1234): transplant now swaps residuals "
               "ACROSS classes,\n              since consecutive images in a batch are no longer "
               "the same class. This is the\n              honest paper number for the transplant "
               "attack.")
        except Exception as e:
            _p(f"[CROSS-CLASS] could not build a shuffled loader ({e}); staying same-class.")

    for m in (tr.ae, tr.c2):
        try:
            m.eval()
        except Exception:
            pass

    results = []
    if args.mode in ("all", "transplant"):
        results.append(run_transplant(tr, mb, gray=False))
    if args.mode in ("all", "validate"):
        results.append(run_validate(tr, mb, gray=False))
    if args.mode in ("all", "grayscale"):
        results.append(run_validate(tr, mb, gray=True))
    if args.mode in ("all", "transplant_gray"):
        results.append(run_transplant(tr, mb, gray=True))
    if args.mode in ("all", "gray_attack"):
        results.append(run_gray_attack(tr, mb))
    if args.mode in ("all", "universal"):
        results.append(run_universal(tr, mb))
    if args.mode in ("all", "imperceptibility"):
        results.append(run_imperceptibility(tr, mb, gray=False))
    if args.mode in ("all", "imperceptibility_gray"):
        results.append(run_imperceptibility(tr, mb, gray=True))
    if args.mode in ("all", "hist"):
        results.append(run_hist(tr, mb))
    if args.mode in ("all", "realval"):
        results.append(run_realval(tr, args.epoch))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        _p(f"\n[OUT] wrote {args.out}")

    # ---- self-test: does the harness reproduce the checkpoint's own record? ----
    if ref and any(r.get("mode") == "validate" for r in results):
        v = next(r for r in results if r.get("mode") == "validate")
        _p("\n" + "=" * 78)
        _p("HARNESS SELF-TEST  (our numbers vs the checkpoint's own last_val)")
        _p("=" * 78)
        pairs = [("acc_originals", "acc_raw"), ("acc_watermarked", "acc_both"),
                 ("GAPg", "gap_both_raw"), ("GAPng", "gap_both_raw_ng")]
        worst = 0.0
        for ours_k, ref_k in pairs:
            if ours_k in v and ref_k in ref:
                a, b = float(v[ours_k]), float(ref[ref_k]) * 100.0
                worst = max(worst, abs(a - b))
                _p(f"  {ours_k:16s} harness={a:7.2f}  checkpoint={b:7.2f}  diff={a - b:+.2f}pp")
        _p(f"\n  worst |diff| = {worst:.2f}pp")
        n_eval = int(v.get("n", 0))
        if mb:
            _p(f"  NOTE: this pass saw {n_eval} of 1075 val images (--max_batches {args.max_batches}).")
            _p("        Sampling error alone can move BOTH by several pp. A clean verdict needs")
            _p("        the FULL val set — drop --max_batches.")
        if worst <= 1.0:
            _p("  => harness reproduces the training-time validate() path. Numbers are trustworthy.")
        elif mb and worst <= 20.0:
            _p("  => inconclusive on a subset; re-run without --max_batches before judging.")
        else:
            _p("  => MISMATCH. Do NOT trust the other modes until this is explained.")
            _p("     (some drift is expected: the message bits are re-drawn at random each run)")


if __name__ == "__main__":
    main()
