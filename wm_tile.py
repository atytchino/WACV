# -*- coding: utf-8 -*-
"""
wm_tile.py — PATCH 2026-07-07 — structure-preserving spatial-tiling message path.

Proof-of-mechanism for the multi-bit channel. The concat/FiLM latent path +
direct bridge could push watermark ENERGY into pixels (detector reached 92.9%
TPR) but the AE direct bridge collapses the 64 message channels into a single
pixel map (_proj_external_map sign-sum + blur), destroying per-bit structure →
BER stayed 50%.

This module bypasses the AE entirely for the message: it tiles the bits across
non-overlapping spatial patches of the luma image and adds a small constant
luma offset per patch (+eps for bit=1, -eps for bit=0). A matching reader
averages each patch and thresholds. No channel collapse, no low-pass blur, so
per-bit structure survives to pixels by construction.

Start config: 16 bits on a 4x4 grid (comfortable patch size ~40x40 px at 160).
Scale n_bits later as an ablation. This is HiDDeN/MBRS-style spatial payload
injection — defensible against reviewers and comparable to the paper's own
baselines.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def _grid_for_bits(n_bits: int) -> Tuple[int, int]:
    """Smallest near-square grid (rows, cols) with rows*cols >= n_bits."""
    r = int(math.floor(math.sqrt(n_bits)))
    while r > 1 and (n_bits % r != 0):
        r -= 1
    c = int(math.ceil(n_bits / max(r, 1)))
    # ensure capacity
    while r * c < n_bits:
        c += 1
    return r, c


def tile_message_map(signed_bits: torch.Tensor, H: int, W: int,
                     grid: Tuple[int, int] | None = None) -> torch.Tensor:
    """Build a [B,1,H,W] luma-offset map from signed bits ({-1,+1}).

    Each bit occupies one non-overlapping spatial patch; its value is a constant
    ±1 across that patch. Returns a unit-scale map (values in {-1,+1}); the
    caller multiplies by the desired amplitude (eps).

    Args:
        signed_bits: [B, n_bits] in {-1,+1}.
        H, W: target spatial size.
        grid: (rows, cols). If None, derived from n_bits.
    """
    B, n_bits = signed_bits.shape
    if grid is None:
        grid = _grid_for_bits(n_bits)
    gr, gc = grid
    assert gr * gc >= n_bits, f"grid {gr}x{gc} too small for {n_bits} bits"

    # place bits into a [B, gr*gc] buffer (pad unused cells with 0 = no signal)
    buf = signed_bits.new_zeros(B, gr * gc)
    buf[:, :n_bits] = signed_bits
    buf = buf.view(B, 1, gr, gc)  # [B,1,gr,gc]

    # upsample to full resolution with NEAREST so each patch is a hard constant
    # (no interpolation blur across bit boundaries — structure preserved).
    m = F.interpolate(buf, size=(H, W), mode="nearest")  # [B,1,H,W]
    return m


class TileMessageReader(nn.Module):
    """Position-aware reader: average each spatial patch of the (wm - carrier)
    luma residual and produce a per-bit logit. Deterministic pooling, no BN.

    Because the encoder writes a constant ±eps per patch, the mean of each patch
    of the residual is a direct estimate of that bit's sign. We expose it as a
    logit (scaled mean) so BCE-with-logits works unchanged.
    """
    def __init__(self, n_bits: int, grid: Tuple[int, int] | None = None,
                 logit_scale: float = 50.0):
        super().__init__()
        self.n_bits = int(n_bits)
        self.grid = grid if grid is not None else _grid_for_bits(n_bits)
        self.logit_scale = float(logit_scale)

    def forward(self, residual_luma: torch.Tensor) -> torch.Tensor:
        """residual_luma: [B,1,H,W] (wm luma - carrier luma). Returns [B,n_bits] logits."""
        B = residual_luma.shape[0]
        gr, gc = self.grid
        # adaptive average pool to grid → [B,1,gr,gc]
        pooled = F.adaptive_avg_pool2d(residual_luma, output_size=(gr, gc))
        flat = pooled.view(B, gr * gc)[:, :self.n_bits]  # [B,n_bits]
        return flat * self.logit_scale
