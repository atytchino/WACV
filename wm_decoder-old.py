"""Standalone watermark decoder.

This module defines a watermark extraction network that is architecturally
INDEPENDENT from the C2 classifier and the destructive gate mechanism.

Design rationale (responding to ECCV-2026 review feedback):
- A separate decoder network with bit extraction is a recoverable signal
  in the traditional watermarking sense. It satisfies the requirement
  that third parties can verify watermarks without access to the trained
  classifier weights.
- The C2 classifier with destructive gate becomes a SECONDARY mechanism
  (checkpoint protection), not the primary watermarking proof.

The decoder is trained jointly with the encoder but uses a SEPARATE optimizer
state so it can be evaluated independently. It loads from its own checkpoint
file (decoder_*.pth) for inference and ablation studies.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# -------------------------
# Backbone variants
# -------------------------

def _build_resnet_backbone(arch: str, in_channels: int) -> nn.Module:
    """Build a ResNet feature extractor with adjusted first conv.

    Args:
        arch: 'resnet18' or 'resnet34'.
        in_channels: 1 for grayscale, 3 for color.

    Returns:
        nn.Module with .fc replaced by Identity. Output is [B, 512] features.
    """
    if arch == 'resnet18':
        model = torchvision.models.resnet18(weights=None)
        feat_dim = 512
    elif arch == 'resnet34':
        model = torchvision.models.resnet34(weights=None)
        feat_dim = 512
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    # Adjust first conv for grayscale input if needed
    if in_channels != 3:
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

    # Replace classifier head with Identity — we will attach our own heads
    model.fc = nn.Identity()
    model._feat_dim = feat_dim
    return model


# -------------------------
# Standalone decoder
# -------------------------

class StandaloneDecoder(nn.Module):
    """Independent watermark decoder.

    Takes a watermarked image, outputs N bit logits. This is the network that
    third parties run to verify a watermark — fulfilling the standalone
    detection requirement from review feedback.

    Architecture: ResNet34 backbone -> 512d feature -> N-bit logit head.
    ResNet34 chosen for architectural consistency with the C1 / C2 classifiers
    (also ResNet34), enabling backbone warm-start from a pretrained C1 and
    providing ample capacity for 32+ bit message extraction.

    The decoder is trained with BCE loss against the ground-truth message
    used by the encoder. Bit accuracy is the primary metric.

    Args:
        n_bits: number of bits in the watermark payload.
        in_channels: 1 for grayscale, 3 for color.
        arch: backbone architecture ('resnet18' or 'resnet34'; default resnet34).
        hidden_dim: dimensionality of the hidden layer in the bit head.
    """

    def __init__(
        self,
        n_bits: int = 32,
        in_channels: int = 3,
        arch: str = 'resnet34',
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.n_bits = n_bits
        self.in_channels = in_channels
        self.arch = arch

        self.backbone = _build_resnet_backbone(arch, in_channels)
        feat_dim = self.backbone._feat_dim

        # Bit extraction head — two-layer MLP for non-linearity
        self.bit_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_bits),
        )

        # Detection head (binary watermark-present-or-not) — auxiliary
        # signal for training stability and additional metric
        self.detect_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract backbone features without applying heads.

        Useful for ablation studies, intermediate analysis, and computing
        features for KNN attack evaluation.
        """
        return self.backbone(image)

    def forward(
        self, image: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor:
        """Extract bit logits from image.

        Args:
            image: [B, C, H, W] in [0, 1].
            return_features: if True, also return backbone features.

        Returns:
            bit_logits: [B, n_bits] raw logits (apply sigmoid to get probabilities).
            (Optionally) features: [B, feat_dim] backbone activations.
        """
        feat = self.backbone(image)  # [B, feat_dim]
        bit_logits = self.bit_head(feat)
        if return_features:
            return bit_logits, feat
        return bit_logits

    def forward_full(self, image: torch.Tensor):
        """Return both bit logits AND detection logit.

        Returns:
            bit_logits: [B, n_bits]
            detect_logit: [B, 1] — auxiliary watermark-present detector
        """
        feat = self.backbone(image)
        return self.bit_head(feat), self.detect_head(feat)

    @torch.no_grad()
    def decode(self, image: torch.Tensor) -> torch.Tensor:
        """Hard decode to binary bits.

        Args:
            image: [B, C, H, W].

        Returns:
            [B, n_bits] binary tensor in {0, 1}.
        """
        self.eval()
        logits = self.forward(image)
        return (logits > 0.0).float()


# -------------------------
# Loss helpers
# -------------------------

def decoder_loss(
    predicted_logits: torch.Tensor,
    true_bits: torch.Tensor,
    detection_logits: Optional[torch.Tensor] = None,
    is_watermarked: Optional[torch.Tensor] = None,
    detection_weight: float = 0.5,
) -> torch.Tensor:
    """Compute decoder training loss.

    Args:
        predicted_logits: [B, n_bits] decoder output for watermarked images.
        true_bits: [B, n_bits] ground-truth message.
        detection_logits: [B, 1] optional auxiliary detection head output.
        is_watermarked: [B] in {0, 1} ground-truth for detection.
        detection_weight: weight for auxiliary detection loss.

    Returns:
        Scalar loss tensor.
    """
    # Primary: BCE on bits
    bit_loss = F.binary_cross_entropy_with_logits(predicted_logits, true_bits)

    if detection_logits is not None and is_watermarked is not None:
        det_loss = F.binary_cross_entropy_with_logits(
            detection_logits.squeeze(-1), is_watermarked.float()
        )
        return bit_loss + detection_weight * det_loss
    return bit_loss


# -------------------------
# Split-channel decoder (Option B: layered watermarking)
# -------------------------

class SplitChannelDecoder(nn.Module):
    """Two-head watermark decoder for layered (latent / skip64) encoding.

    Extends the single-channel concept: instead of one bit_head extracting
    N bits from a single message channel, this decoder has TWO heads:
      - head_lat:   extracts the first n_bits_lat bits (carried by latent path)
      - head_skip:  extracts the next n_bits_skip bits (carried by skip64 path)

    The total payload is (n_bits_lat + n_bits_skip). Each half is conditioned
    by a SEPARATE MessageEmbedding in the encoder, so the two halves are
    semantically independent — they encode different bits and are extracted
    independently.

    Design rationale:
      - Latent path (1024 ch, /16 spatial) is low-frequency, robust to attacks.
      - Skip64 path (512 ch, /8 spatial) is higher-frequency, more precise but
        more fragile.
      - Splitting the message lets us report per-channel bit accuracy and
        demonstrate layered robustness — bits in the latent half should survive
        stronger attacks than bits in the skip64 half.

    Compatible with the existing inference_extract.py loader because it uses
    the same backbone + same checkpoint conventions; the only difference is
    two separate head modules instead of one.

    Args:
        n_bits_lat: number of bits assigned to the latent channel.
        n_bits_skip: number of bits assigned to the skip64 channel.
        in_channels: 1 (grayscale) or 3 (color). 3 for the trainer (RGB output).
        arch: backbone architecture (default resnet34, matches C1/C2).
        hidden_dim: hidden width of each bit head.
    """

    def __init__(
        self,
        n_bits_lat: int = 64,
        n_bits_skip: int = 64,
        in_channels: int = 3,
        arch: str = 'resnet34',
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.n_bits_lat = n_bits_lat
        self.n_bits_skip = n_bits_skip
        self.n_bits_total = n_bits_lat + n_bits_skip
        self.in_channels = in_channels
        self.arch = arch

        self.backbone = _build_resnet_backbone(arch, in_channels)
        feat_dim = self.backbone._feat_dim

        # Head for bits stored in the latent path
        self.bit_head_lat = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_bits_lat),
        )

        # Head for bits stored in the skip64 path
        self.bit_head_skip = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_bits_skip),
        )

        # Auxiliary watermark-present detection head
        self.detect_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    # ---------- Convenience read-only attributes ----------

    @property
    def n_bits(self) -> int:
        """Total bit count — for compatibility with code that expects .n_bits."""
        return self.n_bits_total

    # ---------- Forward variants ----------

    def features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract backbone features without applying heads."""
        return self.backbone(image)

    def forward(
        self, image: torch.Tensor, return_split: bool = False
    ):
        """Extract bit logits.

        Args:
            image: [B, C, H, W] in [0, 1].
            return_split: if True, return (logits_lat, logits_skip) tuple.
                          if False, return concatenated [B, n_bits_total].

        Returns:
            By default: [B, n_bits_total] logits (concat of [lat, skip]).
            If return_split=True: (logits_lat [B, n_bits_lat], logits_skip [B, n_bits_skip]).
        """
        feat = self.backbone(image)
        logits_lat = self.bit_head_lat(feat)
        logits_skip = self.bit_head_skip(feat)
        if return_split:
            return logits_lat, logits_skip
        return torch.cat([logits_lat, logits_skip], dim=1)

    def forward_full(self, image: torch.Tensor):
        """Return all heads: bit logits (both) + detection.

        Returns:
            logits_lat:   [B, n_bits_lat]
            logits_skip:  [B, n_bits_skip]
            detect_logit: [B, 1]
        """
        feat = self.backbone(image)
        logits_lat = self.bit_head_lat(feat)
        logits_skip = self.bit_head_skip(feat)
        detect_logit = self.detect_head(feat)
        return logits_lat, logits_skip, detect_logit

    @torch.no_grad()
    def decode(self, image: torch.Tensor) -> torch.Tensor:
        """Hard decode to binary bits (concatenated lat+skip).

        Returns:
            [B, n_bits_total] tensor in {0, 1}.
        """
        self.eval()
        logits_lat, logits_skip = self.forward(image, return_split=True)
        bits_lat = (logits_lat > 0.0).float()
        bits_skip = (logits_skip > 0.0).float()
        return torch.cat([bits_lat, bits_skip], dim=1)


def split_decoder_loss(
    logits_lat: torch.Tensor,
    logits_skip: torch.Tensor,
    true_bits_lat: torch.Tensor,
    true_bits_skip: torch.Tensor,
    detection_logits: Optional[torch.Tensor] = None,
    is_watermarked: Optional[torch.Tensor] = None,
    detection_weight: float = 0.5,
    lat_weight: float = 1.0,
    skip_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Compute split-channel decoder training loss.

    Returns a dict with both the total loss and the individual components,
    enabling separate logging of latent-vs-skip64 channel performance.

    Args:
        logits_lat: [B, n_bits_lat] decoder output for latent-half bits.
        logits_skip: [B, n_bits_skip] decoder output for skip64-half bits.
        true_bits_lat: [B, n_bits_lat] ground-truth bits assigned to latent.
        true_bits_skip: [B, n_bits_skip] ground-truth bits assigned to skip64.
        detection_logits: [B, 1] optional auxiliary watermark detection.
        is_watermarked: [B] in {0, 1} ground-truth for detection.
        detection_weight: weight for auxiliary detection loss.
        lat_weight: weight for latent-half BCE.
        skip_weight: weight for skip64-half BCE.

    Returns:
        Dict with keys: 'total', 'lat', 'skip', 'detect'. All scalar tensors.
    """
    L_lat = F.binary_cross_entropy_with_logits(logits_lat, true_bits_lat)
    L_skip = F.binary_cross_entropy_with_logits(logits_skip, true_bits_skip)
    L_total = lat_weight * L_lat + skip_weight * L_skip

    L_det = torch.zeros((), device=logits_lat.device, dtype=logits_lat.dtype)
    if detection_logits is not None and is_watermarked is not None:
        L_det = F.binary_cross_entropy_with_logits(
            detection_logits.squeeze(-1), is_watermarked.float()
        )
        L_total = L_total + detection_weight * L_det

    return {
        'total': L_total,
        'lat': L_lat.detach(),
        'skip': L_skip.detach(),
        'detect': L_det.detach(),
    }


# -------------------------
# Warm-start helper
# -------------------------

def warm_start_from_c1(decoder, c1_ckpt_path: str,
                       verbose: bool = True) -> int:
    """Initialize decoder backbone from a pre-trained C1 ResNet34 checkpoint.

    Both C1 and StandaloneDecoder use ResNet34 with 3-channel input, so backbone
    weights are directly transferable. Only the final classification head differs
    (C1 -> num_classes, decoder -> n_bits) and is skipped during transfer.

    This saves 1-2 epochs of feature-learning training time and is a standard
    practice in watermarking literature when a classifier is already trained
    on the same domain.

    Args:
        decoder: StandaloneDecoder instance to initialize.
        c1_ckpt_path: path to C1 checkpoint (.pth file).
        verbose: if True, print transfer summary.

    Returns:
        Number of parameter tensors successfully transferred.
    """
    if decoder.arch != 'resnet34':
        if verbose:
            print(f"[DECODER WARMSTART] Skipped: decoder arch is {decoder.arch}, "
                  f"warm-start requires resnet34")
        return 0

    try:
        ckpt = torch.load(c1_ckpt_path, map_location='cpu')
    except (FileNotFoundError, OSError) as e:
        if verbose:
            print(f"[DECODER WARMSTART] Failed to load {c1_ckpt_path}: {e}")
        return 0

    # Extract state dict — C1 checkpoints may have different key conventions
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            c1_state = ckpt['state_dict']
        elif 'model' in ckpt:
            c1_state = ckpt['model']
        elif 'c1' in ckpt:
            c1_state = ckpt['c1']
        else:
            c1_state = ckpt
    else:
        c1_state = ckpt

    # Strip prefixes that may come from DataParallel / module wrapping
    cleaned = {}
    for k, v in c1_state.items():
        if not isinstance(v, torch.Tensor):
            continue
        k_clean = k
        for prefix in ('module.', '_orig_mod.', 'backbone.', 'base.'):
            if k_clean.startswith(prefix):
                k_clean = k_clean[len(prefix):]
        # Skip fc head — different output shape
        if k_clean.startswith('fc.') or k_clean.startswith('wm_head'):
            continue
        cleaned[k_clean] = v

    # Load with strict=False to ignore missing/extra keys
    result = decoder.backbone.load_state_dict(cleaned, strict=False)
    n_loaded = len(cleaned) - len(result.unexpected_keys)

    if verbose:
        print(f"[DECODER WARMSTART] from {c1_ckpt_path}")
        print(f"  Transferred: {n_loaded} tensors")
        if result.missing_keys:
            print(f"  Missing (will train from scratch): {len(result.missing_keys)} tensors")
        if result.unexpected_keys:
            print(f"  Unused C1 keys: {len(result.unexpected_keys)} tensors")
    return n_loaded
