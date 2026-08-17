#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wm_gui.py  --  GUI Watermarker + Validator for the WACV gate-only smooth-mark system.

Green-on-black terminal aesthetic, button-driven. Two tools in one window (tabs):

  WATERMARKER
    - Load Image           -> pick any image file
    - Apply Watermark      -> runs the full C2+G+AE system, shows original | watermarked
    - Show Delta (heat)     -> shows the perturbation heat-map (diverging: blue -, red +)
    - Save Watermarked     -> write the watermarked PNG

  VALIDATOR
    - Single Image mode:
        Load & Validate    -> feed one image through C2 gate; verdict in green-on-black:
                              "WATERMARK DETECTED  -> classified as <class> (conf X%)"
                              vs "NO WATERMARK  -> gate output collapsed (conf X%)"
        (optional) tick "Auto-watermark first" so you can hand it a CLEAN image and it
        will make the marked version and validate BOTH, showing the contrast.
    - Batch mode:
        Select Folder      -> folder of images; runs the whole folder through the gate,
                              reports split accuracy: originals vs watermarked, in the
                              green-on-black console.

The checkpoint picker at the top lets one app serve TLD / AFHQ / ORNL: point it at that
dataset's frozen wm_system_e008.pth + c2_eval_e008.pth and it reloads.

USAGE (Machine B, from the WACV project dir, using the WACV venv so torch/lpips/AE import):
    $WPY = "C:\\Users\\atytchino.GIGABYTE.000\\PycharmProjects\\WACV\\.venv\\Scripts\\python.exe"
    cd C:\\Users\\atytchino.GIGABYTE.000\\PycharmProjects\\WACV
    & $WPY wm_gui.py --trainer .\\20260728-Trainer_MULTIBIT_v17_TILEGRID.py `
        --system_ckpt "E:\\RUNS\\TLD_smooth_eps010_s0\\checkpoints\\wm_system_e008.pth" `
        --c2_ckpt     "E:\\RUNS\\TLD_smooth_eps010_s0\\checkpoints\\c2_eval_e008.pth"

You can also launch it with NO checkpoint args and use the "Load Checkpoints" button.

NOTE: this is a gate-only system (decoder off). "Watermark detected" is decided by the
GATE, not by a message decoder: a genuinely watermarked image is classified correctly by
C2, an unmarked (or forged) image collapses toward chance. That IS the access-control gate.
"""

import os
import sys
import argparse
import threading
import traceback
from pathlib import Path

# --- Tkinter (stdlib) ---
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# --- lazy heavy imports (torch etc.) happen inside the loader thread ---
torch = None
np = None
Image = None
ImageTk = None
eval_mod = None   # the eval_wm_system module, imported like the harness does


# ----------------------------- theme ---------------------------------------
BG    = "#000000"   # black
FG    = "#00ff66"   # phosphor green
FG_DIM= "#00aa44"
ACCENT= "#00ff66"
WARN  = "#ffcc00"
ERR   = "#ff5555"
FONT_MONO = ("Consolas", 11)
FONT_MONO_BIG = ("Consolas", 15, "bold")
FONT_MONO_SM  = ("Consolas", 9)


# ----------------------- model backend (loads once) ------------------------
class Backend:
    """Wraps the trainer/system load and the two operations (watermark, gate-verdict)."""

    def __init__(self):
        self.tr = None
        self.classes = []
        self.device = None
        self.trainer_path = None
        self.system_ckpt = None
        self.c2_ckpt = None
        self.loaded = False

    def load(self, trainer_path, system_ckpt, c2_ckpt, log):
        """Heavy load. Call from a worker thread. `log` is a callable(str)."""
        global torch, np, Image, eval_mod
        log("importing torch / numpy / PIL ...")
        import torch as _torch
        import numpy as _np
        from PIL import Image as _Image
        torch = _torch; np = _np; Image = _Image

        log("loading eval_wm_system module (shared loader with the harness) ...")
        # import eval_wm_system.py from the project dir, the same way the harness does
        import importlib.util as ilu
        proj = Path(trainer_path).resolve().parent
        evp = proj / "eval_wm_system.py"
        if not evp.exists():
            raise FileNotFoundError(f"eval_wm_system.py not found next to trainer at {evp}")
        spec = ilu.spec_from_file_location("eval_wm_system", str(evp))
        eval_mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(eval_mod)

        log("installing torch.load compat shim ...")
        eval_mod._install_torch_load_compat()

        log(f"loading trainer module: {Path(trainer_path).name} ...")
        TM = eval_mod.load_trainer_module(str(trainer_path))

        log("rebuilding system from checkpoints (this constructs C2, G, AE) ...")
        # keep the eval side-effect free; out_root override to a scratch dir
        scratch = proj / "_gui_scratch"
        scratch.mkdir(exist_ok=True)
        tr, _ = eval_mod.build_trainer(
            TM, str(system_ckpt), str(c2_ckpt),
            overrides={"out_root": scratch},
        )
        self.tr = tr
        self.device = tr.device
        self.classes = list(getattr(tr, "classes", None)
                             or getattr(getattr(tr, "val_ds", None), "classes", [])
                             or [])
        if not self.classes:
            log("[warn] class names not found; will show numeric class ids")
        self.trainer_path = trainer_path
        self.system_ckpt = system_ckpt
        self.c2_ckpt = c2_ckpt
        self.loaded = True
        log(f"READY. device={self.device}  classes={len(self.classes)}")

    # ---- image helpers ----
    def _pil_to_tensor01(self, pil_img):
        """PIL RGB -> [1,3,H,W] float in [0,1] on device, sized to the model's image_size."""
        size = int(getattr(self.tr.cfg, "image_size", 256))
        pil_img = pil_img.convert("RGB").resize((size, size), Image.BICUBIC)
        arr = np.asarray(pil_img).astype("float32") / 255.0        # H,W,3
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)      # 1,3,H,W
        return t.to(self.device)

    def _tensor01_to_pil(self, t01):
        """[1,3,H,W] or [3,H,W] in [0,1] -> PIL RGB."""
        if t01.dim() == 4:
            t01 = t01[0]
        arr = (t01.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy() * 255.0)
        arr = arr.astype("uint8")
        return Image.fromarray(arr, "RGB")

    def _valid_mask_for(self, x01):
        b, _, h, w = x01.shape
        return torch.ones((b, 1, h, w), device=x01.device)

    def apply_watermark(self, pil_img):
        """Return (orig_pil, wm_pil, delta_heat_pil). Runs the full system."""
        x01 = self._pil_to_tensor01(pil_img)
        vm = self._valid_mask_for(x01)
        syn = self.tr.synth_variants_nograd(
            x01, valid_mask=vm, epoch=0,
            variants=("base", "both"), k_factor=1.0, varpercent=False, mode="eval",
        )
        both01 = syn["both01"].clamp(0, 1)
        delta = (both01 - x01)[0]                      # 3,H,W  signed
        orig_pil = self._tensor01_to_pil(x01)
        wm_pil = self._tensor01_to_pil(both01)
        heat_pil = self._delta_heatmap(delta)
        return orig_pil, wm_pil, heat_pil

    def _delta_heatmap(self, delta_chw):
        """Signed per-pixel delta -> diverging heat map PIL (blue = -, red = +, black ~ 0)."""
        d = delta_chw.detach().cpu().numpy()           # 3,H,W
        d = d.mean(axis=0)                             # H,W  (luma-ish average)
        m = float(np.abs(d).max()) or 1e-6
        dn = d / m                                     # [-1,1]
        H, W = dn.shape
        rgb = np.zeros((H, W, 3), dtype="float32")
        pos = dn > 0
        neg = dn < 0
        rgb[..., 0] = np.where(pos, dn, 0.0)           # red   = positive
        rgb[..., 2] = np.where(neg, -dn, 0.0)          # blue  = negative
        # a little green on strong magnitudes so it reads on black, matching the theme
        rgb[..., 1] = np.abs(dn) * 0.25
        rgb = (np.clip(rgb, 0, 1) * 255).astype("uint8")
        return Image.fromarray(rgb, "RGB")

    def gate_verdict(self, pil_img):
        """
        Feed one image through the C2 gate. Returns dict:
            {class_idx, class_name, conf, margin, detected(bool), probs}
        'detected' = the gate produced a confident, non-collapsed decision.
        """
        x01 = self._pil_to_tensor01(pil_img)
        vm = self._valid_mask_for(x01)
        xN = x01 * 2 - 1
        try:
            xN = self.tr._apply_prod_padding_wipe(xN, vm)
        except Exception:
            pass
        z, _, _ = self.tr.c2_eval(xN, gate=True)        # [1, num_classes] logits
        probs = torch.softmax(z, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
        conf = float(conf.item()); idx = int(idx.item())
        # margin = top1 - top2 (collapse indicator: near 0 => flat/chance)
        srt, _ = torch.sort(probs, descending=True)
        margin = float((srt[0] - srt[1]).item()) if probs.numel() > 1 else float(srt[0])
        n = probs.numel()
        chance = 1.0 / max(n, 1)
        # detected if the gate is meaningfully above chance AND has a real margin
        detected = (conf >= max(2.0 * chance, 0.35)) and (margin >= 0.10)
        name = self.classes[idx] if (self.classes and 0 <= idx < len(self.classes)) else f"class_{idx}"
        return dict(class_idx=idx, class_name=name, conf=conf, margin=margin,
                    detected=detected, chance=chance, n=n)

    def batch_split(self, folder, log, progress=None):
        """
        Run every image in `folder` (recursively) through the gate TWICE:
        once as-is (as 'original') and once after watermarking ('watermarked').
        Report gate-accuracy split. If the folder is an ImageFolder with class
        subdirs, we also score classification correctness; otherwise we just
        report detected-rate.
        """
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        files = [p for p in Path(folder).rglob("*") if p.suffix.lower() in exts]
        if not files:
            log("[warn] no image files found in that folder.")
            return
        log(f"scanning {len(files)} images ...")
        # class from parent dir name if it matches known classes
        known = set(self.classes)
        orig_detected = 0
        wm_detected = 0
        orig_correct = 0
        wm_correct = 0
        have_labels = 0
        for i, fp in enumerate(files):
            try:
                pil = Image.open(str(fp))
            except Exception:
                continue
            label = fp.parent.name if fp.parent.name in known else None
            # original
            vo = self.gate_verdict(pil)
            # watermarked
            _, wm_pil, _ = self.apply_watermark(pil)
            vw = self.gate_verdict(wm_pil)
            orig_detected += int(vo["detected"])
            wm_detected += int(vw["detected"])
            if label is not None:
                have_labels += 1
                orig_correct += int(vo["class_name"] == label)
                wm_correct += int(vw["class_name"] == label)
            if progress:
                progress((i + 1) / len(files))
            if (i + 1) % 25 == 0:
                log(f"  ... {i+1}/{len(files)}")
        n = len(files)
        log("")
        log("=" * 52)
        log(f"BATCH SPLIT  ({n} images)")
        log("=" * 52)
        log(f"  ORIGINALS   gate-detected: {100.0*orig_detected/n:6.2f}%   (expect LOW)")
        log(f"  WATERMARKED gate-detected: {100.0*wm_detected/n:6.2f}%   (expect HIGH)")
        if have_labels:
            log(f"  ORIGINALS   classified-correct: {100.0*orig_correct/have_labels:6.2f}%")
            log(f"  WATERMARKED classified-correct: {100.0*wm_correct/have_labels:6.2f}%")
            split = 100.0 * (wm_correct - orig_correct) / have_labels
            log(f"  => ACCURACY SPLIT (wm - orig): {split:+.2f} pp")
        else:
            split = 100.0 * (wm_detected - orig_detected) / n
            log(f"  => DETECTED SPLIT (wm - orig): {split:+.2f} pp")
            log("  (no class subfolders found -> detected-rate split only)")
        log("=" * 52)


# ------------------------------- GUI ---------------------------------------
class App(tk.Tk):
    def __init__(self, args):
        super().__init__()
        self.title("WACV  --  Watermarker + Validator")
        self.configure(bg=BG)
        self.geometry("1080x760")
        self.minsize(900, 640)

        self.backend = Backend()
        self._orig_pil = None
        self._wm_pil = None
        self._heat_pil = None
        self._imgrefs = []   # keep PhotoImage refs alive

        self._build_style()
        self._build_header(args)
        self._build_tabs()
        self._build_console()

        # if checkpoints were passed on the CLI, load them now
        if args.trainer and args.system_ckpt and args.c2_ckpt:
            self._start_load(args.trainer, args.system_ckpt, args.c2_ckpt)

    # ---- style ----
    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background="#111111", foreground=FG,
                        font=FONT_MONO, padding=(16, 6))
        style.map("TNotebook.Tab",
                  background=[("selected", "#003311")],
                  foreground=[("selected", FG)])
        style.configure("TFrame", background=BG)
        style.configure("Green.TButton", background="#002211", foreground=FG,
                        font=FONT_MONO, borderwidth=1, focusthickness=1,
                        focuscolor=FG)
        style.map("Green.TButton",
                  background=[("active", "#004422"), ("pressed", "#006633")],
                  foreground=[("active", FG)])
        style.configure("Green.TCheckbutton", background=BG, foreground=FG,
                        font=FONT_MONO_SM)
        style.map("Green.TCheckbutton", background=[("active", BG)],
                  foreground=[("active", FG)])

    def _mkbtn(self, parent, text, cmd, width=20):
        return ttk.Button(parent, text=text, command=cmd, style="Green.TButton", width=width)

    # ---- header: checkpoint picker + status ----
    def _build_header(self, args):
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 4))

        tk.Label(hdr, text="WACV  GATE-ONLY  SMOOTH-MARK", bg=BG, fg=FG,
                 font=FONT_MONO_BIG).pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="no model loaded")
        tk.Label(hdr, textvariable=self.status_var, bg=BG, fg=WARN,
                 font=FONT_MONO_SM).pack(side=tk.RIGHT)

        row = tk.Frame(self, bg=BG)
        row.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 6))
        self._mkbtn(row, "Load Checkpoints", self._pick_and_load, width=18).pack(side=tk.LEFT)
        self.ckpt_lbl = tk.Label(row, text="(none)", bg=BG, fg=FG_DIM, font=FONT_MONO_SM)
        self.ckpt_lbl.pack(side=tk.LEFT, padx=10)
        self._cli_args = args

    # ---- tabs ----
    def _build_tabs(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=4)
        self._build_watermarker_tab()
        self._build_validator_tab()

    def _build_watermarker_tab(self):
        tab = tk.Frame(self.nb, bg=BG)
        self.nb.add(tab, text="  WATERMARKER  ")

        btns = tk.Frame(tab, bg=BG); btns.pack(side=tk.TOP, fill=tk.X, pady=6)
        self._mkbtn(btns, "Load Image", self._wm_load_image).pack(side=tk.LEFT, padx=4)
        self._mkbtn(btns, "Apply Watermark", self._wm_apply).pack(side=tk.LEFT, padx=4)
        self._mkbtn(btns, "Show Delta (heat)", self._wm_show_heat).pack(side=tk.LEFT, padx=4)
        self._mkbtn(btns, "Save Watermarked", self._wm_save).pack(side=tk.LEFT, padx=4)

        canvas = tk.Frame(tab, bg=BG); canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.wm_panels = []
        for title in ("ORIGINAL", "WATERMARKED", "DELTA (heat)"):
            col = tk.Frame(canvas, bg=BG, highlightbackground=FG_DIM, highlightthickness=1)
            col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
            tk.Label(col, text=title, bg=BG, fg=FG, font=FONT_MONO).pack(side=tk.TOP)
            lbl = tk.Label(col, bg=BG)
            lbl.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.wm_panels.append(lbl)

    def _build_validator_tab(self):
        tab = tk.Frame(self.nb, bg=BG)
        self.nb.add(tab, text="  VALIDATOR  ")

        # mode row
        mode = tk.Frame(tab, bg=BG); mode.pack(side=tk.TOP, fill=tk.X, pady=6)
        self.auto_wm = tk.BooleanVar(value=True)
        ttk.Checkbutton(mode, text="Auto-watermark a clean image first (show contrast)",
                        variable=self.auto_wm, style="Green.TCheckbutton").pack(side=tk.LEFT, padx=4)

        btns = tk.Frame(tab, bg=BG); btns.pack(side=tk.TOP, fill=tk.X, pady=4)
        self._mkbtn(btns, "Single: Load & Validate", self._val_single, width=24).pack(side=tk.LEFT, padx=4)
        self._mkbtn(btns, "Batch: Select Folder", self._val_batch, width=22).pack(side=tk.LEFT, padx=4)

        # big verdict banner
        self.verdict_var = tk.StringVar(value="awaiting input ...")
        self.verdict_lbl = tk.Label(tab, textvariable=self.verdict_var, bg=BG, fg=FG,
                                    font=FONT_MONO_BIG, wraplength=1000, justify="left")
        self.verdict_lbl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=8)

        # small preview
        self.val_preview = tk.Label(tab, bg=BG)
        self.val_preview.pack(side=tk.TOP, pady=4)

    # ---- console ----
    def _build_console(self):
        wrap = tk.Frame(self, bg=BG); wrap.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False,
                                                padx=10, pady=(4, 10))
        tk.Label(wrap, text="console", bg=BG, fg=FG_DIM, font=FONT_MONO_SM).pack(side=tk.TOP, anchor="w")
        self.console = tk.Text(wrap, height=9, bg="#001100", fg=FG, insertbackground=FG,
                               font=FONT_MONO_SM, borderwidth=0, highlightthickness=1,
                               highlightbackground=FG_DIM)
        self.console.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.console.configure(state=tk.DISABLED)

    def log(self, msg):
        def _do():
            self.console.configure(state=tk.NORMAL)
            self.console.insert(tk.END, str(msg) + "\n")
            self.console.see(tk.END)
            self.console.configure(state=tk.DISABLED)
        try:
            self.after(0, _do)
        except Exception:
            print(msg)

    # ---- checkpoint load ----
    def _pick_and_load(self):
        trainer = filedialog.askopenfilename(
            title="Select trainer .py (v17)",
            filetypes=[("Python", "*.py"), ("All", "*.*")])
        if not trainer:
            return
        sysck = filedialog.askopenfilename(
            title="Select wm_system_e008.pth",
            filetypes=[("Checkpoint", "*.pth"), ("All", "*.*")])
        if not sysck:
            return
        c2ck = filedialog.askopenfilename(
            title="Select c2_eval_e008.pth",
            filetypes=[("Checkpoint", "*.pth"), ("All", "*.*")])
        if not c2ck:
            return
        self._start_load(trainer, sysck, c2ck)

    def _start_load(self, trainer, sysck, c2ck):
        self.status_var.set("loading model ...")
        self.ckpt_lbl.configure(text=Path(sysck).parent.parent.name)
        def worker():
            try:
                self.backend.load(trainer, sysck, c2ck, self.log)
                self.after(0, lambda: self.status_var.set(
                    f"READY  |  {self.backend.device}  |  {len(self.backend.classes)} classes"))
            except Exception as e:
                self.log("[ERROR] " + repr(e))
                self.log(traceback.format_exc())
                self.after(0, lambda: self.status_var.set("load FAILED (see console)"))
        threading.Thread(target=worker, daemon=True).start()

    def _require_model(self):
        if not self.backend.loaded:
            messagebox.showwarning("No model", "Load checkpoints first (Load Checkpoints button).")
            return False
        return True

    # ---- image display helper ----
    def _show_in(self, label_widget, pil_img, box=320):
        img = pil_img.copy()
        img.thumbnail((box, box), Image.NEAREST)
        from PIL import ImageTk as _ImageTk
        ph = _ImageTk.PhotoImage(img)
        self._imgrefs.append(ph)
        # keep the ref list from growing unbounded
        if len(self._imgrefs) > 12:
            self._imgrefs = self._imgrefs[-12:]
        label_widget.configure(image=ph)
        label_widget.image = ph

    # ---- WATERMARKER actions ----
    def _wm_load_image(self):
        if not self._require_model():
            return
        fp = filedialog.askopenfilename(
            title="Load image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All", "*.*")])
        if not fp:
            return
        try:
            self._orig_pil = Image.open(fp).convert("RGB")
            self._wm_pil = None; self._heat_pil = None
            self._show_in(self.wm_panels[0], self._orig_pil)
            self.wm_panels[1].configure(image="")
            self.wm_panels[2].configure(image="")
            self.log(f"loaded image: {Path(fp).name}  ({self._orig_pil.size[0]}x{self._orig_pil.size[1]})")
        except Exception as e:
            messagebox.showerror("Load failed", repr(e))

    def _wm_apply(self):
        if not self._require_model():
            return
        if self._orig_pil is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        def worker():
            try:
                self.log("applying watermark (C2+G+AE) ...")
                o, w, h = self.backend.apply_watermark(self._orig_pil)
                self._orig_pil, self._wm_pil, self._heat_pil = o, w, h
                self.after(0, lambda: (self._show_in(self.wm_panels[0], o),
                                       self._show_in(self.wm_panels[1], w)))
                self.log("watermark applied. (smooth content-adaptive mark, no tile)")
            except Exception as e:
                self.log("[ERROR] " + repr(e)); self.log(traceback.format_exc())
        threading.Thread(target=worker, daemon=True).start()

    def _wm_show_heat(self):
        if self._heat_pil is None:
            messagebox.showinfo("No delta", "Apply the watermark first.")
            return
        self._show_in(self.wm_panels[2], self._heat_pil)
        self.log("delta heat-map: red = +delta, blue = -delta, dark = ~0")

    def _wm_save(self):
        if self._wm_pil is None:
            messagebox.showinfo("Nothing to save", "Apply the watermark first.")
            return
        fp = filedialog.asksaveasfilename(
            title="Save watermarked PNG", defaultextension=".png",
            filetypes=[("PNG", "*.png")])
        if not fp:
            return
        try:
            self._wm_pil.save(fp)
            self.log(f"saved watermarked image -> {fp}")
            messagebox.showinfo("Saved", f"Watermarked image saved:\n{fp}")
        except Exception as e:
            messagebox.showerror("Save failed", repr(e))

    # ---- VALIDATOR actions ----
    def _set_verdict(self, text, color):
        self.verdict_var.set(text)
        self.verdict_lbl.configure(fg=color)

    def _fmt_verdict(self, v, tag):
        if v["detected"]:
            return (f"[{tag}]  WATERMARK DETECTED\n"
                    f"       classified as: {v['class_name']}   (conf {100*v['conf']:.1f}%, "
                    f"margin {100*v['margin']:.1f}%)")
        else:
            return (f"[{tag}]  NO WATERMARK  --  gate output collapsed\n"
                    f"       top guess {v['class_name']} at only {100*v['conf']:.1f}% "
                    f"(chance {100*v['chance']:.1f}%, margin {100*v['margin']:.1f}%)")

    def _val_single(self):
        if not self._require_model():
            return
        fp = filedialog.askopenfilename(
            title="Image to validate",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All", "*.*")])
        if not fp:
            return
        def worker():
            try:
                pil = Image.open(fp).convert("RGB")
                if self.auto_wm.get():
                    self.log(f"validating '{Path(fp).name}' as CLEAN, then watermarking and re-validating ...")
                    v_clean = self.backend.gate_verdict(pil)
                    _, wm_pil, _ = self.backend.apply_watermark(pil)
                    v_wm = self.backend.gate_verdict(wm_pil)
                    txt = (self._fmt_verdict(v_clean, "ORIGINAL") + "\n\n"
                           + self._fmt_verdict(v_wm, "WATERMARKED"))
                    # color: green if the contrast is correct (clean collapses, wm detected)
                    good = (not v_clean["detected"]) and v_wm["detected"]
                    color = FG if good else WARN
                    self.after(0, lambda: (self._set_verdict(txt, color),
                                           self._show_in(self.val_preview, wm_pil, box=260)))
                    self.log(self._fmt_verdict(v_clean, "ORIGINAL").replace("\n", " | "))
                    self.log(self._fmt_verdict(v_wm, "WATERMARKED").replace("\n", " | "))
                else:
                    self.log(f"validating '{Path(fp).name}' as-is ...")
                    v = self.backend.gate_verdict(pil)
                    txt = self._fmt_verdict(v, "INPUT")
                    color = FG if v["detected"] else WARN
                    self.after(0, lambda: (self._set_verdict(txt, color),
                                           self._show_in(self.val_preview, pil, box=260)))
                    self.log(txt.replace("\n", " | "))
            except Exception as e:
                self.log("[ERROR] " + repr(e)); self.log(traceback.format_exc())
                self.after(0, lambda: self._set_verdict("validation FAILED (see console)", ERR))
        threading.Thread(target=worker, daemon=True).start()

    def _val_batch(self):
        if not self._require_model():
            return
        folder = filedialog.askdirectory(title="Select folder of images")
        if not folder:
            return
        def worker():
            try:
                self._set_verdict("running batch split ...", FG)
                self.backend.batch_split(folder, self.log)
                self.after(0, lambda: self._set_verdict(
                    "batch done -- see console for the split.", FG))
            except Exception as e:
                self.log("[ERROR] " + repr(e)); self.log(traceback.format_exc())
                self.after(0, lambda: self._set_verdict("batch FAILED (see console)", ERR))
        threading.Thread(target=worker, daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", default=None, help="path to v17 trainer .py")
    ap.add_argument("--system_ckpt", default=None, help="wm_system_e008.pth")
    ap.add_argument("--c2_ckpt", default=None, help="c2_eval_e008.pth")
    args = ap.parse_args()
    app = App(args)
    app.mainloop()


if __name__ == "__main__":
    main()
