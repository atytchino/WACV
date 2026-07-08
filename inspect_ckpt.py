#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""inspect_ckpt.py — determine what resolution/data AE and C1 were trained on.

Three levels of evidence, from cheapest to most decisive:

  A) METADATA (always runs):
     - dumps every non-tensor entry stored inside the .pth (config dicts,
       argparse Namespaces, epoch counters, notes — whatever the original
       trainer saved alongside the weights);
     - scans the checkpoint directory and its parent for sidecar files
       (*.json / *.yaml / *.yml / *.txt / *.log / *.md) and prints the small
       ones — training configs usually live there;
     - summarizes tensor shapes (first conv, fc head, param counts).
     NOTE: pure conv weights do NOT encode training resolution (the nets are
     fully convolutional / adaptive-pooled), so if A comes up empty, use B/C.

  B) AE BEHAVIORAL PROBE (--probe_images <dir>):
     Runs AE.forward_plain on the same images resized to 160/256/512 and
     reports reconstruction PSNR/consistency per size. An autoencoder
     reconstructs best near its training resolution; a clear PSNR peak is
     strong evidence.

  C) C1 BEHAVIORAL PROBE (--c1_val_root <dir with class subfolders>):
     Loads the C1 backbone into a plain torchvision ResNet34 (wm_head and
     other extras skipped) and measures classification accuracy per size.
     The training resolution shows markedly higher accuracy.

Usage examples (PowerShell, from C:\\Users\\atytchino\\PycharmProjects\\WACV):

  # Level A only — metadata for both checkpoints:
  python inspect_ckpt.py --ckpt "E:\\AE_TRAINED\\TLD\\ckpts\\ae_best.pth" "E:\\C1_TRAINED\\TLD\\ckpts\\c1_best.pth"

  # A + B (AE probe on a few validation images):
  python inspect_ckpt.py --ckpt "E:\\AE_TRAINED\\TLD\\ckpts\\ae_best.pth" ^
      --probe_images "E:\\TLD\\val" --ae_py_path "C:\\Users\\atytchino\\PycharmProjects\\WACV"

  # A + C (C1 probe on the labeled val set):
  python inspect_ckpt.py --ckpt "E:\\C1_TRAINED\\TLD\\ckpts\\c1_best.pth" ^
      --c1_val_root "E:\\TLD\\val"

All probes are read-only: nothing is written, no checkpoint is modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SIDE_EXTS = {".json", ".yaml", ".yml", ".txt", ".log", ".md", ".cfg", ".ini"}
SIDE_MAX_BYTES = 64 * 1024  # print sidecar files up to 64 KB


# ----------------------------------------------------------------------
# Level A — metadata
# ----------------------------------------------------------------------

def _load_ckpt(path: Path):
    """Load a trusted local checkpoint, tolerating both old and new torch."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # torch < 2.0 has no weights_only kwarg
        return torch.load(path, map_location="cpu")
    except Exception as e1:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e2:
            raise RuntimeError(f"failed to load: {e1!r} / {e2!r}")


def _fmt_val(v, depth=0, max_items=40):
    pad = "  " * depth
    if torch.is_tensor(v):
        return f"<tensor {tuple(v.shape)} {v.dtype}>"
    if isinstance(v, dict):
        if depth > 4:
            return f"<dict with {len(v)} keys>"
        lines = []
        for i, (k, vv) in enumerate(v.items()):
            if i >= max_items:
                lines.append(f"{pad}  ... ({len(v) - max_items} more keys)")
                break
            lines.append(f"{pad}  {k}: {_fmt_val(vv, depth + 1)}")
        return "{\n" + "\n".join(lines) + f"\n{pad}}}"
    if isinstance(v, (list, tuple)):
        if len(v) > 12:
            return f"<{type(v).__name__} len={len(v)}: {list(v[:6])!r} ...>"
        return repr(v)
    if hasattr(v, "__dict__") and not isinstance(v, (str, bytes)):
        # argparse.Namespace, dataclasses, custom cfg objects
        try:
            return f"<{type(v).__name__}> " + _fmt_val(vars(v), depth)
        except Exception:
            return repr(v)[:200]
    s = repr(v)
    return s if len(s) <= 300 else s[:300] + "...(truncated)"


def _tensor_summary(sd: dict, title: str):
    n_tensors = 0
    n_params = 0
    interesting = []
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        n_tensors += 1
        n_params += v.numel()
        lk = k.lower()
        if (
            v.dim() == 4 and ("conv1" in lk or lk.endswith("e0.0.weight") or ".e0." in lk)
        ) or lk.endswith("fc.weight") or "wm_head" in lk or "content_enc" in lk:
            interesting.append((k, tuple(v.shape)))
    print(f"    [{title}] tensors={n_tensors} params={n_params/1e6:.2f}M")
    for k, shp in interesting[:20]:
        print(f"      {k}: {shp}")
    # fc.weight rows = num_classes for a classifier
    for k, v in sd.items():
        if torch.is_tensor(v) and k.lower().endswith("fc.weight") and v.dim() == 2:
            print(f"      -> fc suggests num_classes={v.shape[0]} (in_features={v.shape[1]})")


def inspect_metadata(path: Path):
    print("=" * 78)
    print(f"CKPT: {path}")
    print("=" * 78)
    if not path.exists():
        print("  !! file not found")
        return
    try:
        obj = _load_ckpt(path)
    except RuntimeError as e:
        print(f"  !! {e}")
        return

    if isinstance(obj, dict):
        tensor_keys = [k for k, v in obj.items() if torch.is_tensor(v)]
        meta_keys = [k for k in obj.keys() if k not in tensor_keys]
        print(f"  top-level keys: {list(obj.keys())[:30]}")
        # Non-tensor metadata — this is where training config usually hides
        printed_meta = False
        for k in meta_keys:
            v = obj[k]
            # a nested dict of tensors is a state_dict, summarize instead
            if isinstance(v, dict) and v and all(torch.is_tensor(x) for x in v.values()):
                _tensor_summary(v, f"state_dict '{k}'")
                continue
            if isinstance(v, dict) and any(kk in str(v.keys()).lower() for kk in ("exp_avg", "step", "param_groups")):
                print(f"  [{k}]: <optimizer state, skipped>")
                continue
            print(f"  [{k}]: {_fmt_val(v, 1)}")
            printed_meta = True
        if tensor_keys:
            _tensor_summary({k: obj[k] for k in tensor_keys}, "flat state_dict")
        if not printed_meta:
            print("  (no non-tensor metadata stored in this checkpoint — "
                  "the trainer saved weights only; check sidecar files below "
                  "or run the behavioral probes)")
    else:
        print(f"  checkpoint object type: {type(obj)} — trying vars()/summary")
        try:
            print(_fmt_val(obj, 1))
        except Exception as e:
            print(f"  !! cannot introspect: {e}")

    # Sidecar files next to the checkpoint (and one directory up)
    print("-" * 78)
    print("  SIDECAR SCAN (ckpt dir + parent):")
    seen = set()
    for d in (path.parent, path.parent.parent):
        if not d.exists() or d in seen:
            continue
        seen.add(d)
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in SIDE_EXTS:
                size = f.stat().st_size
                print(f"    {f}  ({size} bytes)")
                if size <= SIDE_MAX_BYTES:
                    try:
                        text = f.read_text(encoding="utf-8", errors="replace")
                        for line in text.splitlines():
                            print(f"      | {line}")
                    except Exception as e:
                        print(f"      !! unreadable: {e}")
                else:
                    print("      (too large to print; open manually — likely a log)")
    print()


# ----------------------------------------------------------------------
# Shared image loading for probes
# ----------------------------------------------------------------------

def _gather_images(root: Path, per_dir: int, max_total: int):
    """Collect up to max_total image paths; if root has class subfolders,
    take up to per_dir from each (with the folder name as label)."""
    items = []  # (path, label or None)
    subdirs = [d for d in sorted(root.iterdir()) if d.is_dir()] if root.is_dir() else []
    if subdirs:
        for d in subdirs:
            n = 0
            for f in sorted(d.rglob("*")):
                if f.suffix.lower() in IMG_EXTS:
                    items.append((f, d.name))
                    n += 1
                    if n >= per_dir:
                        break
    else:
        for f in sorted(root.rglob("*")):
            if f.suffix.lower() in IMG_EXTS:
                items.append((f, None))
    return items[:max_total]


def _load_batch(paths, size: int, device):
    from PIL import Image
    import numpy as np
    ts = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        arr = np.asarray(im).astype("float32") / 255.0
        ts.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(ts, 0).to(device)


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a - b) ** 2).mean().item())
    if mse <= 1e-12:
        return 99.0
    import math
    return 10.0 * math.log10(1.0 / mse)


# ----------------------------------------------------------------------
# Level B — AE reconstruction probe
# ----------------------------------------------------------------------

def probe_ae(ckpt_path: Path, images_root: Path, ae_py_path: str,
             ae_module: str, ae_class: str, sizes, n_images: int, device):
    print("=" * 78)
    print(f"AE BEHAVIORAL PROBE: {ckpt_path}")
    print("=" * 78)
    sys.path.insert(0, ae_py_path)
    try:
        mod = __import__(ae_module)
        AE = getattr(mod, ae_class)
    except Exception as e:
        print(f"  !! cannot import {ae_module}.{ae_class} from {ae_py_path}: {e}")
        return
    ae = AE()
    obj = _load_ckpt(ckpt_path)
    sd = obj
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "ae", "ema"):
            if key in obj and isinstance(obj[key], dict):
                sd = obj[key]
                break
    cleaned = {}
    for k, v in (sd.items() if isinstance(sd, dict) else []):
        if not torch.is_tensor(v):
            continue
        for pref in ("module.", "_orig_mod."):
            if k.startswith(pref):
                k = k[len(pref):]
        cleaned[k] = v
    res = ae.load_state_dict(cleaned, strict=False)
    print(f"  load_state_dict: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    if res.missing_keys[:5]:
        print(f"    missing (first 5): {res.missing_keys[:5]}")
    ae.eval().to(device)

    items = _gather_images(images_root, per_dir=2, max_total=n_images)
    if not items:
        print(f"  !! no images found under {images_root}")
        return
    paths = [p for p, _ in items]
    print(f"  probing {len(paths)} images at sizes {list(sizes)}")
    print(f"  {'size':>6} | {'PSNR(recon,in) dB':>18} | verdict hint")
    results = {}
    with torch.no_grad():
        for s in sizes:
            try:
                x = _load_batch(paths, s, device)
                y = ae.forward_plain(x).clamp(0, 1)
                results[s] = _psnr(y, x)
                print(f"  {s:>6} | {results[s]:>18.2f} |")
            except Exception as e:
                print(f"  {s:>6} | {'ERROR':>18} | {e}")
    if results:
        best = max(results, key=results.get)
        spread = max(results.values()) - min(results.values())
        print(f"  -> best reconstruction at {best}px (spread {spread:.2f} dB).")
        if spread < 1.0:
            print("     Spread < 1 dB: AE generalizes across sizes; resolution "
                  "evidence weak — rely on metadata/sidecars or the C1 probe.")
        else:
            print(f"     Clear peak: AE was most likely trained at ~{best}px input.")
    print()


# ----------------------------------------------------------------------
# Level C — C1 classification probe
# ----------------------------------------------------------------------

def _load_class_from_file(py_path: str, class_name: str):
    """Import a class from an arbitrary .py file (filenames with dashes/digits
    are fine — importlib is used instead of the import statement)."""
    import importlib.util
    p = Path(py_path).resolve()
    sys.path.insert(0, str(p.parent))  # so wm_decoder / wm_message resolve
    spec = importlib.util.spec_from_file_location("wm_trainer_mod_for_probe", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


def _build_c1_trainer_identical(num_classes: int):
    """Self-contained replica of the trainer's ResNet34LF_BN: stride-1 conv1,
    BlurPool downsamples in layer2-4, deep-feature wm_head, gate buffers.
    State-dict layout matches checkpoints saved with
    arch='ResNet34LF_BN_trainer_identical' — no trainer import required
    (importing the 6800-line trainer pulls its whole dependency chain and is
    fragile; this replica is copied verbatim from the class definition)."""
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models

    class BlurPool(nn.Module):
        def __init__(self, ch: int, filt=(1, 2, 1)):
            super().__init__()
            f = torch.tensor(filt, dtype=torch.float32)
            k = (f[:, None] * f[None, :])
            k = (k / k.sum()).view(1, 1, 3, 3).repeat(ch, 1, 1, 1)
            self.register_buffer("k", k)
            self.groups = ch

        def forward(self, x):
            return F.conv2d(x, self.k, stride=2, padding=1, groups=self.groups)

    class ResNet34LF_BN(nn.Module):
        def __init__(self, n_cls: int, gate_strength: float = 2.10):
            super().__init__()
            self.register_buffer('gate_strength', torch.tensor(float(gate_strength)))
            self.register_buffer('destructive_strength', torch.tensor(1.0))
            base = models.resnet34(weights=None)  # BatchNorm2d, as in the ckpt
            base.conv1.stride = (1, 1)
            self.base = base
            self._wrap_blur(base.layer2)
            self._wrap_blur(base.layer3)
            self._wrap_blur(base.layer4)
            self.wm_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
            )
            self.base.fc = nn.Linear(self.base.fc.in_features, n_cls)
            self.wm_affine = nn.Parameter(torch.zeros(n_cls))

        @staticmethod
        def _wrap_blur(layer):
            for b in layer:
                if b.downsample is not None:
                    seq = []
                    for sm in b.downsample:
                        if isinstance(sm, nn.Conv2d) and sm.stride == (2, 2):
                            sm.stride = (1, 1)
                            seq += [sm, BlurPool(sm.out_channels)]
                        else:
                            seq.append(sm)
                    b.downsample = nn.Sequential(*seq)

        def forward(self, x, gate: bool = False):
            # Probe uses the RAW classifier path only (gate=False semantics).
            x = self.base.relu(self.base.bn1(self.base.conv1(x)))
            x = self.base.maxpool(x)
            x1 = self.base.layer1(x)
            x2 = self.base.layer2(x1)
            x3 = self.base.layer3(x2)
            x4 = self.base.layer4(x3)
            pooled = self.base.avgpool(x4).flatten(1)
            logits_raw = self.base.fc(pooled)
            wm_logit = self.wm_head(x4).squeeze(1)
            return logits_raw, wm_logit, x4

    return ResNet34LF_BN(num_classes)


def probe_c1(ckpt_path: Path, val_root: Path, sizes, per_class: int, device,
             trainer_py: str = ""):
    print("=" * 78)
    print(f"C1 BEHAVIORAL PROBE: {ckpt_path}")
    print("=" * 78)
    import torchvision

    obj = _load_ckpt(ckpt_path)
    sd = obj
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "c1"):
            if key in obj and isinstance(obj[key], dict):
                sd = obj[key]
                break

    keys = [k for k in (sd.keys() if isinstance(sd, dict) else [])]
    has_base_prefix = any(str(k).startswith(("base.", "module.base.")) for k in keys)

    cleaned = {}
    num_classes = None
    for k, v in (sd.items() if isinstance(sd, dict) else []):
        if not torch.is_tensor(v):
            continue
        kc = k
        # For trainer-identical checkpoints KEEP base./wm_head/wm_affine keys —
        # the replica class has those attributes.
        prefixes = ("module.", "_orig_mod.") if has_base_prefix else ("module.", "_orig_mod.", "base.", "backbone.")
        for pref in prefixes:
            if kc.startswith(pref):
                kc = kc[len(pref):]
        if (not has_base_prefix) and (kc.startswith("wm_head") or kc.startswith("wm_affine")):
            continue
        cleaned[kc] = v
        if kc.endswith("fc.weight") and v.dim() == 2:
            num_classes = int(v.shape[0])
    classes = sorted([d.name for d in val_root.iterdir() if d.is_dir()])
    if num_classes is None:
        num_classes = len(classes)
    print(f"  num_classes from ckpt fc: {num_classes}; val folders: {len(classes)}")
    if len(classes) != num_classes:
        print("  !! class count mismatch — label order may be wrong; "
              "accuracy is still comparable ACROSS sizes.")

    trainer_forward = False
    if has_base_prefix:
        try:
            model = _build_c1_trainer_identical(num_classes)
            trainer_forward = True
            print("  using BUILT-IN trainer-identical ResNet34LF_BN replica "
                  "(stride-1 conv1 + BlurPool; no trainer import needed)")
        except Exception as e:
            print(f"  !! built-in replica failed ({e}); falling back to torchvision")
            has_base_prefix = False
    if not has_base_prefix:
        import torchvision as _tv
        print("  !! WARNING: torchvision ResNet34 fallback — structurally WRONG for "
              "ResNet34LF checkpoints; per-size comparison unreliable.")
        model = _tv.models.resnet34(weights=None, num_classes=num_classes)
    res = model.load_state_dict(cleaned, strict=False)
    print(f"  load_state_dict: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    if res.missing_keys[:5]:
        print(f"    missing (first 5): {res.missing_keys[:5]}")
    if res.unexpected_keys[:5]:
        print(f"    unexpected (first 5): {res.unexpected_keys[:5]}")
    load_ok = (len(res.missing_keys) <= 5) and (len(res.unexpected_keys) <= 5)
    if not load_ok:
        print("  !! LOAD SANITY FAILED — the accuracies below are NOT informative "
              "(weights did not land in the model). Paste the key lists above and "
              "we adapt the loader.")
    model.eval().to(device)

    cls_to_idx = {c: i for i, c in enumerate(classes)}
    items = _gather_images(val_root, per_dir=per_class, max_total=per_class * max(1, len(classes)))
    labeled = [(p, cls_to_idx[lab]) for p, lab in items if lab in cls_to_idx]
    if not labeled:
        print(f"  !! no labeled images under {val_root} (expected class subfolders)")
        return
    print(f"  probing {len(labeled)} images at sizes {list(sizes)}; input mapped to [-1,1] "
          "(matches how the WM trainer feeds C1)")
    print(f"  {'size':>6} | {'accuracy':>9}")
    accs = {}
    with torch.no_grad():
        for s in sizes:
            ok = 0
            bs = 8
            for i in range(0, len(labeled), bs):
                chunk = labeled[i:i + bs]
                x = _load_batch([p for p, _ in chunk], s, device) * 2.0 - 1.0
                out = model(x, gate=False) if trainer_forward else model(x)
                logits = out[0] if isinstance(out, tuple) else out
                pred = logits.argmax(1).cpu()
                ok += int((pred == torch.tensor([y for _, y in chunk])).sum())
            accs[s] = ok / len(labeled)
            print(f"  {s:>6} | {accs[s]:>9.3f}")
    if accs:
        chance = 1.0 / max(1, num_classes)
        if all(a <= chance + 0.10 for a in accs.values()):
            print(f"  -> ALL sizes are at/near chance ({chance:.3f}): weights not "
                  "loaded or preprocessing mismatch. RESULT NOT INFORMATIVE — "
                  "do not draw resolution conclusions from this run.")
        else:
            best = max(accs, key=accs.get)
            print(f"  -> best accuracy at {best}px; C1 was most likely trained at ~{best}px.")
    print()


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True, help="One or more .pth files to inspect (level A)")
    ap.add_argument("--probe_images", type=str, default="", help="Image dir for the AE probe (level B); e.g. E:\\TLD\\val")
    ap.add_argument("--ae_py_path", type=str, default=".", help="Dir containing AE_ContentBound.py")
    ap.add_argument("--ae_module", type=str, default="AE_ContentBound")
    ap.add_argument("--ae_class", type=str, default="UniversalAutoEncoder")
    ap.add_argument("--c1_val_root", type=str, default="", help="Labeled val dir (class subfolders) for the C1 probe (level C)")
    ap.add_argument("--trainer_py", type=str, default="", help="Path to the v9 trainer .py; loads the trainer-identical ResNet34LF_BN class for a structurally correct C1 probe (REQUIRED for reliable results with arch=ResNet34LF_BN_trainer_identical)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[160, 256, 512])
    ap.add_argument("--n_images", type=int, default=16, help="Images for the AE probe")
    ap.add_argument("--per_class", type=int, default=8, help="Images per class for the C1 probe")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpts = [Path(p) for p in args.ckpt]
    for p in ckpts:
        inspect_metadata(p)

    if args.probe_images:
        ae_ckpts = [p for p in ckpts if "ae" in p.name.lower()] or ckpts[:1]
        for p in ae_ckpts:
            probe_ae(p, Path(args.probe_images), args.ae_py_path,
                     args.ae_module, args.ae_class, args.sizes, args.n_images, args.device)

    if args.c1_val_root:
        c1_ckpts = [p for p in ckpts if "c1" in p.name.lower()] or ckpts[-1:]
        for p in c1_ckpts:
            probe_c1(p, Path(args.c1_val_root), args.sizes, args.per_class, args.device,
                     trainer_py=args.trainer_py)


if __name__ == "__main__":
    main()
