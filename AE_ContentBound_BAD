#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Size-agnostic, anti-checkerboard AutoEncoder.

Main fixes vs the first draft:
- mixed gray/color batches now work per-sample (no batch-average gate bug)
- forward_plain accepts 1ch or 3ch input
- chroma branch is residual/stable and no longer artificially clipped to +/-0.25
- `enc()` is no longer hard-coded under no_grad (watermark trainer can still freeze it externally)
- added `forward()` alias for generic PyTorch usage

Interface kept compatible with the watermark pipeline:
    enc(y01) -> {"latent": Z, "s64": S64}
    forward_plain(x01) -> rgb01
    embed_external_wm_gray(...)
    embed_external_wm(...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_pad(x: torch.Tensor, pad: Tuple[int, int, int, int], mode: str = "reflect") -> torch.Tensor:
    if min(x.shape[-2], x.shape[-1]) < 2 and mode == "reflect":
        mode = "replicate"
    return F.pad(x, pad, mode=mode)


def pad_to_multiple(x: torch.Tensor, mult: int = 16, mode: str = "reflect") -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    _, _, h, w = x.shape
    pad_h = (mult - (h % mult)) % mult
    pad_w = (mult - (w % mult)) % mult
    pt = pad_h // 2
    pb = pad_h - pt
    pl = pad_w // 2
    pr = pad_w - pl
    pad = (pl, pr, pt, pb)
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)
    return _safe_pad(x, pad, mode=mode), pad


def unpad(x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
    pl, pr, pt, pb = pad
    if (pl, pr, pt, pb) == (0, 0, 0, 0):
        return x
    h = x.shape[-2]
    w = x.shape[-1]
    return x[..., pt:h - pb if pb > 0 else h, pl:w - pr if pr > 0 else w]


def rgb_to_ycbcr(x01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r, g, b = x01[:, 0:1], x01[:, 1:2], x01[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.564 * (b - y)
    cr = 0.713 * (r - y)
    return y, cb, cr


def ycbcr_to_rgb(y: torch.Tensor, cb: torch.Tensor, cr: torch.Tensor) -> torch.Tensor:
    r = y + (1.0 / 0.713) * cr
    b = y + (1.0 / 0.564) * cb
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    x = torch.cat([r, g, b], dim=1)
    return x.clamp(0.0, 1.0)


def colorfulness_score(x01: torch.Tensor) -> torch.Tensor:
    rg = (x01[:, 0:1] - x01[:, 1:2]).abs()
    rb = (x01[:, 0:1] - x01[:, 2:3]).abs()
    gb = (x01[:, 1:2] - x01[:, 2:3]).abs()
    return (rg + rb + gb).mean(dim=(2, 3), keepdim=True) / 3.0


def is_grayscale_like(x01: torch.Tensor, eps: float = 0.010) -> torch.Tensor:
    return (colorfulness_score(x01) < eps).to(dtype=x01.dtype)


def _ensure_rgb_input(x01: torch.Tensor) -> torch.Tensor:
    if x01.dim() != 4:
        raise ValueError("Expected [B,C,H,W] input")
    if x01.size(1) == 3:
        return x01
    if x01.size(1) == 1:
        return x01.repeat(1, 3, 1, 1)
    raise ValueError(f"Expected 1 or 3 channels, got {x01.size(1)}")


def _down_mask(m: Optional[torch.Tensor], size_hw: Tuple[int, int], ref: torch.Tensor) -> Optional[torch.Tensor]:
    if m is None:
        return None
    if m.dim() == 3:
        m = m[:, None, :, :]
    if m.shape[-2:] != size_hw:
        m = F.interpolate(m.to(device=ref.device, dtype=ref.dtype), size=size_hw, mode="nearest")
    return m.to(device=ref.device, dtype=ref.dtype)


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, groups: int = 32, act: str = "silu"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        g = min(groups, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        self.gn = nn.GroupNorm(g, out_ch)
        self.act = nn.SiLU(inplace=True) if act == "silu" else nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, ch: int, groups: int = 32):
        super().__init__()
        self.c1 = ConvGNAct(ch, ch, 3, 1, 1, groups=groups, act="silu")
        self.c2 = ConvGNAct(ch, ch, 3, 1, 1, groups=groups, act="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.c2(self.c1(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 32):
        super().__init__()
        self.c1 = ConvGNAct(in_ch, out_ch, 3, 1, 1, groups=groups, act="silu")
        self.rb = ResBlock(out_ch, groups=groups)

    def forward(self, x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
        x = self.c1(x)
        return self.rb(x)

# ═══════════════════════════════════════════════════════════════════════════
# Learned Instance Binding: ContentEncoder + FiLM
# ─────────────────────────────────────────────────────────────────────────
# Instead of a hash function, the system LEARNS which semantic features of
# each image to bind the watermark to. The ContentEncoder maps the AE latent
# to a per-image content vector. FiLM (Feature-wise Linear Modulation) uses
# that vector to scale+shift intermediate activations of the WM generator,
# making the WM pattern a learned function of the image's semantic content.
#
# Trained end-to-end through L_transfer and L_diversity — the encoder learns
# to extract features that make WM maximally content-specific.
# ═══════════════════════════════════════════════════════════════════════════

class ContentEncoder(nn.Module):
    """Learns which image features to bind the WM to.

    Input : AE latent Z  [B, 1024, H/16, W/16]  (frozen, .detach())
    Output: content_vec  [B, content_dim=64]  — per-image learned signature

    Trained via L_transfer: if two images get the same WM residual (no
    content-binding), C2 detects transplanted WM → gradient flows back through
    FiLM → ContentEncoder learns to make WM image-specific.
    """
    def __init__(self, in_ch: int = 1024, content_dim: int = 64, groups: int = 8):
        super().__init__()
        mid = max(64, in_ch // 8)
        g1 = min(groups, in_ch);  g1 = g1 if in_ch % g1 == 0 else 1
        g2 = min(groups, mid);    g2 = g2 if mid  % g2 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1, bias=False),
            nn.GroupNorm(g1, mid),
            nn.GELU(),
            nn.Conv2d(mid, mid, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(g2, mid),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(mid, content_dim),
            nn.Tanh(),                     # bounded: content_vec ∈ (-1, +1)^D
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return self.net(Z)                 # [B, content_dim]


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: γ·h + β per channel.

    Applies a per-image affine transform to intermediate feature maps,
    conditioning the WM generator output on the image's content vector.
    γ (scale) initialized near 1, β (shift) near 0 — neutral at start.
    """
    def __init__(self, content_dim: int, feature_ch: int):
        super().__init__()
        self.to_gamma = nn.Linear(content_dim, feature_ch)
        self.to_beta  = nn.Linear(content_dim, feature_ch)
        # init: near-identity transform at training start
        nn.init.zeros_(self.to_gamma.weight);  nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight);   nn.init.zeros_(self.to_beta.bias)

    def forward(self, h: torch.Tensor, content_vec: torch.Tensor) -> torch.Tensor:
        # h: [B, C, H, W],  content_vec: [B, content_dim]
        B = h.size(0)
        gamma = self.to_gamma(content_vec).view(B, -1, 1, 1)  # scale ≈ 1
        beta  = self.to_beta(content_vec).view(B, -1, 1, 1)   # shift ≈ 0
        return h * gamma + beta


class WMGeneratorConditioned(nn.Module):
    """Content-conditioned WM generator replacing the unconditional G_lat / G_64.

    Architecture:
        Z (AE latent, frozen)
          → ContentEncoder (trainable) → content_vec [B, 64]
          → Conv1 → FiLM(content_vec) → Conv2 → FiLM(content_vec) → out

    The WM residual is now a learned function of the image's semantic content:
        wm_pattern = f_θ(Z, g_φ(Z))
    where g_φ is the ContentEncoder and f_θ is the FiLM-conditioned generator.
    Both θ and φ are trained jointly through the full pipeline loss.
    """
    def __init__(self,
                 in_ch: int = 1024,   # latent channels (AE e4 output)
                 mid_ch: int = 256,
                 out_ch: int = 1,     # 1 for Y-channel WM, 1024 for latent WM
                 content_dim: int = 64,
                 groups: int = 32):
        super().__init__()
        self.content_enc = ContentEncoder(in_ch=in_ch, content_dim=content_dim)
        gm = min(groups, mid_ch); gm = gm if mid_ch % gm == 0 else 1
        self.conv1   = ConvGNAct(in_ch, mid_ch, 3, 1, 1, groups=gm, act="silu")
        self.film1   = FiLM(content_dim, mid_ch)
        self.res1    = ResBlock(mid_ch, groups=gm)
        self.conv2   = ConvGNAct(mid_ch, mid_ch, 3, 1, 1, groups=gm, act="silu")
        self.film2   = FiLM(content_dim, mid_ch)
        self.out_conv = nn.Conv2d(mid_ch, out_ch, 1)
        self.out_act  = nn.Tanh()

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # Content vector from frozen latent (no grad to AE encoder)
        cv = self.content_enc(Z.detach())      # [B, content_dim]

        h = self.conv1(Z)                      # [B, mid_ch, H, W]
        h = self.film1(h, cv)                  # content-modulate
        h = self.res1(h)
        h = self.conv2(h)
        h = self.film2(h, cv)                  # content-modulate again
        return self.out_act(self.out_conv(h))  # [B, out_ch, H, W]


class WMGeneratorSkip(nn.Module):
    """Content-conditioned WM generator for skip64 connections.

    Identical to WMGeneratorConditioned but designed for s64 feature maps
    [B, 512, H/8, W/8] instead of the deeper latent [B, 1024, H/16, W/16].
    The ContentEncoder still reads from the deeper latent Z for a richer
    semantic signature, while the generator operates on skip features.
    """
    def __init__(self,
                 skip_ch: int = 512,   # s64 channels
                 lat_ch: int = 1024,   # latent channels for content enc
                 mid_ch: int = 128,
                 out_ch: int = 512,
                 content_dim: int = 64,
                 groups: int = 32):
        super().__init__()
        self.content_enc = ContentEncoder(in_ch=lat_ch, content_dim=content_dim)
        gm = min(groups, mid_ch); gm = gm if mid_ch % gm == 0 else 1
        self.conv1    = ConvGNAct(skip_ch, mid_ch, 3, 1, 1, groups=gm, act="silu")
        self.film1    = FiLM(content_dim, mid_ch)
        self.conv2    = ConvGNAct(mid_ch, mid_ch, 3, 1, 1, groups=gm, act="silu")
        self.film2    = FiLM(content_dim, mid_ch)
        self.out_conv = nn.Conv2d(mid_ch, out_ch, 1)
        self.out_act  = nn.Tanh()

    def forward(self, skip: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        cv = self.content_enc(Z.detach())       # semantic from deep latent
        h  = self.conv1(skip)
        h  = self.film1(h, cv)
        h  = self.conv2(h)
        h  = self.film2(h, cv)
        return self.out_act(self.out_conv(h))

# ─────────────────────────────────────────────────────────────────────────
# Usage in Trainer (replaces g_lat / g_64 instantiation):
#
#   self.g_lat = WMGeneratorConditioned(
#       in_ch=1024, mid_ch=256, out_ch=1024, content_dim=64)
#   self.g_64  = WMGeneratorSkip(
#       skip_ch=512, lat_ch=1024, mid_ch=128, out_ch=512, content_dim=64)
#
# Forward call in step_generator() stays identical:
#   wm_lat = self.g_lat(ae_out["latent"])      # was: self.g_lat(Z)
#   wm_64  = self.g_64(ae_out["s64"], ae_out["latent"])  # new skip signature
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class AEConfig:
    mult: int = 16
    gn_groups: int = 32
    gray_like_eps: float = 0.010
    chroma_res_scale: float = 0.25
    chroma_abs_clip: float = 0.75
    ext_gain_lat: float = 16.0
    ext_gain_skip: float = 12.0
    ext_direct_gain_lat: float = 0.040
    ext_direct_gain_skip: float = 0.050
    ext_direct_blur_k: int = 5


class UniversalAutoEncoder(nn.Module):
    def __init__(self, cfg: Optional[AEConfig] = None):
        super().__init__()
        self.cfg = cfg or AEConfig()
        g = self.cfg.gn_groups

        # Y encoder
        self.e0 = ConvGNAct(1, 64, 3, 1, 1, groups=g)
        self.e1 = nn.Sequential(ConvGNAct(64, 128, 3, 2, 1, groups=g), ResBlock(128, groups=g))     # /2
        self.e2 = nn.Sequential(ConvGNAct(128, 256, 3, 2, 1, groups=g), ResBlock(256, groups=g))    # /4
        self.e3 = nn.Sequential(ConvGNAct(256, 512, 3, 2, 1, groups=g), ResBlock(512, groups=g))    # /8  -> s64
        self.e4 = nn.Sequential(ConvGNAct(512, 1024, 3, 2, 1, groups=g), ResBlock(1024, groups=g))  # /16 -> latent

        # Y decoder (UNet-style, no deconv)
        self.d4 = UpBlock(1024, 512, groups=g)              # -> /8
        self.d3 = UpBlock(512 + 512, 256, groups=g)         # -> /4
        self.d2 = UpBlock(256 + 256, 128, groups=g)         # -> /2
        self.d1 = UpBlock(128 + 128, 64, groups=g)          # -> /1
        self.d0 = nn.Sequential(ConvGNAct(64 + 64, 64, 3, 1, 1, groups=g), nn.Conv2d(64, 1, 1))

        # Light chroma residual branch (CbCr)
        self.ce0 = ConvGNAct(2, 32, 3, 1, 1, groups=g)
        self.ce1 = nn.Sequential(ConvGNAct(32, 64, 3, 2, 1, groups=g), ResBlock(64, groups=g))
        self.ce2 = nn.Sequential(ConvGNAct(64, 128, 3, 2, 1, groups=g), ResBlock(128, groups=g))
        self.cd2 = UpBlock(128, 64, groups=g)
        self.cd1 = UpBlock(64 + 64, 32, groups=g)
        self.cd0 = nn.Sequential(ConvGNAct(32 + 32, 32, 3, 1, 1, groups=g), nn.Conv2d(32, 2, 1))

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        return self.forward_plain(x01)

    def enc(self, y01: torch.Tensor) -> Dict[str, torch.Tensor]:
        if y01.dim() == 3:
            y01 = y01[:, None, :, :]
        x = y01.clamp(0, 1)
        x, _pad = pad_to_multiple(x, self.cfg.mult, mode="reflect")
        h0 = self.e0(x)
        h1 = self.e1(h0)
        h2 = self.e2(h1)
        h3 = self.e3(h2)
        h4 = self.e4(h3)
        return {"latent": h4, "s64": h3}

    def forward_plain(self, x01: torch.Tensor) -> torch.Tensor:
        x01 = _ensure_rgb_input(x01).clamp(0, 1)
        gmask = is_grayscale_like(x01, eps=self.cfg.gray_like_eps)  # [B,1,1,1]

        y, cb, cr = rgb_to_ycbcr(x01)
        y_hat = self._forward_y(y)

        has_color = bool((gmask < 0.5).any().item())
        if has_color:
            cc_hat = self._forward_c(torch.cat([cb, cr], dim=1))
            cb_hat, cr_hat = cc_hat[:, 0:1], cc_hat[:, 1:2]
            cb_hat = cb_hat * (1.0 - gmask)
            cr_hat = cr_hat * (1.0 - gmask)
        else:
            cb_hat = torch.zeros_like(cb)
            cr_hat = torch.zeros_like(cr)

        out_color = ycbcr_to_rgb(y_hat, cb_hat, cr_hat)
        out_gray = y_hat.repeat(1, 3, 1, 1)
        return (out_color * (1.0 - gmask) + out_gray * gmask).clamp(0, 1)

    def _proj_external_map(self, w: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
        """Deterministic signed projection from multi-channel watermark to 1-channel map.

        Important subtlety:
        using a plain *mean* across 512/1024 channels shrinks the signal roughly by 1/sqrt(C),
        which can make the direct bridge numerically vanish. Here we use an alternating-sign sum
        normalized by sqrt(C), then ROI-normalize to unit RMS so the bridge has a stable scale.
        """
        if w is None:
            return None
        C = int(w.size(1))
        signs = torch.ones((1, C, 1, 1), device=w.device, dtype=w.dtype)
        if C > 1:
            signs[:, 1::2] = -1.0
        # signed sum with variance-preserving normalization (instead of plain mean)
        p = (w * signs).sum(dim=1, keepdim=True) / max(float(C) ** 0.5, 1.0)
        if roi is not None:
            p = p * roi
            denom = roi.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
            rms = torch.sqrt(((p * p) * roi).sum(dim=(2, 3), keepdim=True) / denom + 1e-8)
            p = p / rms.clamp_min(1e-4)
            p = p * roi
        else:
            rms = torch.sqrt((p * p).mean(dim=(2, 3), keepdim=True) + 1e-8)
            p = p / rms.clamp_min(1e-4)
        return p

    def _blur_map(self, x: torch.Tensor, k: int) -> torch.Tensor:
        k = int(max(1, k))
        if k <= 1:
            return x
        if (k % 2) == 0:
            k += 1
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)

    def embed_external_wm_gray(
        self,
        y01: torch.Tensor,
        wm_lat: Optional[torch.Tensor] = None,
        wm_skip: Optional[torch.Tensor] = None,
        alpha_lat: float | torch.Tensor = 0.0,
        alpha_skip: float | torch.Tensor = 0.0,
        roi_lat_32: Optional[torch.Tensor] = None,
        roi_skip_64: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if y01.dim() == 3:
            y01 = y01[:, None, :, :]
        y01 = y01.clamp(0, 1)
        B, _, H, W = y01.shape

        y_pad, pad = pad_to_multiple(y01, self.cfg.mult, mode="reflect")
        h0 = self.e0(y_pad)
        h1 = self.e1(h0)
        h2 = self.e2(h1)
        s64 = self.e3(h2)
        lat = self.e4(s64)

        vm_lat = _down_mask(valid_mask, lat.shape[-2:], lat)
        vm_s64 = _down_mask(valid_mask, s64.shape[-2:], s64)

        roi_lat = torch.ones((B, 1, *lat.shape[-2:]), device=lat.device, dtype=lat.dtype) if roi_lat_32 is None else _down_mask(roi_lat_32, lat.shape[-2:], lat)
        roi_s64 = torch.ones((B, 1, *s64.shape[-2:]), device=s64.device, dtype=s64.dtype) if roi_skip_64 is None else _down_mask(roi_skip_64, s64.shape[-2:], s64)
        if vm_lat is not None:
            roi_lat = roi_lat * vm_lat
        if vm_s64 is not None:
            roi_s64 = roi_s64 * vm_s64

        lat_wm = lat
        s64_wm = s64

        a_lat = None
        a_skip = None
        if wm_lat is not None and float(torch.as_tensor(alpha_lat).abs().max().item()) > 0:
            if isinstance(alpha_lat, (float, int)):
                a_lat = torch.full((B, 1, 1, 1), float(alpha_lat), device=lat.device, dtype=lat.dtype)
            else:
                a_lat = alpha_lat.to(device=lat.device, dtype=lat.dtype).view(B, 1, 1, 1)
            gain_lat = float(getattr(self.cfg, "ext_gain_lat", 16.0) or 16.0)
            lat_wm = lat + gain_lat * a_lat * roi_lat * wm_lat.to(device=lat.device, dtype=lat.dtype)

        if wm_skip is not None and float(torch.as_tensor(alpha_skip).abs().max().item()) > 0:
            if isinstance(alpha_skip, (float, int)):
                a_skip = torch.full((B, 1, 1, 1), float(alpha_skip), device=s64.device, dtype=s64.dtype)
            else:
                a_skip = alpha_skip.to(device=s64.device, dtype=s64.dtype).view(B, 1, 1, 1)
            gain_skip = float(getattr(self.cfg, "ext_gain_skip", 12.0) or 12.0)
            s64_wm = s64 + gain_skip * a_skip * roi_s64 * wm_skip.to(device=s64.device, dtype=s64.dtype)

        y_hat = self._decode_y(lat_wm, s64_wm, h2, h1, h0)
        y_hat = unpad(y_hat, pad)
        if y_hat.shape[-2:] != (H, W):
            y_hat = F.interpolate(y_hat, size=(H, W), mode="bilinear", align_corners=False)

        # Direct external bridge: if the decoder is locally insensitive to latent/skip perturbations,
        # add a small, smooth image-space residual derived deterministically from the external maps.
        direct = torch.zeros_like(y_hat)
        blur_k = int(getattr(self.cfg, "ext_direct_blur_k", 5) or 5)
        if wm_lat is not None and a_lat is not None:
            p_lat = self._proj_external_map(wm_lat.to(device=lat.device, dtype=lat.dtype), roi_lat)
            p_lat = F.interpolate(p_lat, size=(H, W), mode="bilinear", align_corners=False)
            p_lat = self._blur_map(p_lat, blur_k)
            direct = direct + float(getattr(self.cfg, "ext_direct_gain_lat", 0.040) or 0.040) * a_lat.mean(dim=(2,3), keepdim=True) * p_lat
        if wm_skip is not None and a_skip is not None:
            p_skip = self._proj_external_map(wm_skip.to(device=s64.device, dtype=s64.dtype), roi_s64)
            p_skip = F.interpolate(p_skip, size=(H, W), mode="bilinear", align_corners=False)
            p_skip = self._blur_map(p_skip, blur_k)
            direct = direct + float(getattr(self.cfg, "ext_direct_gain_skip", 0.050) or 0.050) * a_skip.mean(dim=(2,3), keepdim=True) * p_skip

        if valid_mask is not None:
            vm_out = _down_mask(valid_mask, (H, W), y_hat)
            if vm_out is not None:
                direct = direct * vm_out

        y_hat = (y_hat + direct).clamp(0, 1)
        return y_hat

    def embed_external_wm(self, x01: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x01 = _ensure_rgb_input(x01).clamp(0, 1)
        y, cb, cr = rgb_to_ycbcr(x01)
        y_hat = self.embed_external_wm_gray(y, *args, **kwargs)
        gmask = is_grayscale_like(x01, eps=self.cfg.gray_like_eps)
        out_gray = y_hat.repeat(1, 3, 1, 1)
        out_color = ycbcr_to_rgb(y_hat, cb, cr)
        return (out_color * (1.0 - gmask) + out_gray * gmask).clamp(0, 1)

    def _forward_y(self, y01: torch.Tensor) -> torch.Tensor:
        y01 = y01.clamp(0, 1)
        y_pad, pad = pad_to_multiple(y01, self.cfg.mult, mode="reflect")
        h0 = self.e0(y_pad)
        h1 = self.e1(h0)
        h2 = self.e2(h1)
        s64 = self.e3(h2)
        lat = self.e4(s64)
        y_hat = self._decode_y(lat, s64, h2, h1, h0)
        y_hat = unpad(y_hat, pad)
        if y_hat.shape[-2:] != y01.shape[-2:]:
            y_hat = F.interpolate(y_hat, size=y01.shape[-2:], mode="bilinear", align_corners=False)
        return y_hat.clamp(0, 1)

    def _forward_c(self, cc: torch.Tensor) -> torch.Tensor:
        cc_pad, pad = pad_to_multiple(cc, self.cfg.mult, mode="reflect")
        h0 = self.ce0(cc_pad)
        h1 = self.ce1(h0)
        h2 = self.ce2(h1)
        u2 = self.cd2(h2, size_hw=h1.shape[-2:])
        u1 = self.cd1(torch.cat([u2, h1], dim=1), size_hw=h0.shape[-2:])
        res = self.cd0(torch.cat([u1, h0], dim=1)).tanh() * float(self.cfg.chroma_res_scale)
        cc_hat = (cc_pad + res).clamp(-float(self.cfg.chroma_abs_clip), float(self.cfg.chroma_abs_clip))
        cc_hat = unpad(cc_hat, pad)
        if cc_hat.shape[-2:] != cc.shape[-2:]:
            cc_hat = F.interpolate(cc_hat, size=cc.shape[-2:], mode="bilinear", align_corners=False)
        return cc_hat

    def _decode_y(self, lat: torch.Tensor, s64: torch.Tensor, h2: torch.Tensor, h1: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        u4 = self.d4(lat, size_hw=s64.shape[-2:])
        u3 = self.d3(torch.cat([u4, s64], dim=1), size_hw=h2.shape[-2:])
        u2 = self.d2(torch.cat([u3, h2], dim=1), size_hw=h1.shape[-2:])
        u1 = self.d1(torch.cat([u2, h1], dim=1), size_hw=h0.shape[-2:])
        y_hat = self.d0(torch.cat([u1, h0], dim=1))
        return y_hat.sigmoid()
