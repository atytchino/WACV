"""Watermark message utilities.

Standalone module for:
  - Random message generation
  - Bit accuracy computation
  - Message-to-embedding projection (FiLM-style conditioning)

Designed to be independent of trainer specifics so it can be reused by
inference scripts, evaluation scripts, and ablation studies.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Bit generation utilities
# -------------------------

def random_message_bits(batch_size: int, n_bits: int, device: torch.device,
                        seed_per_batch: bool = False) -> torch.Tensor:
    """Generate a random bit message for a batch of images.

    Args:
        batch_size: number of images in the batch.
        n_bits: number of bits per message.
        device: target device for the tensor.
        seed_per_batch: if True, use a deterministic seed for reproducibility.

    Returns:
        Tensor of shape [batch_size, n_bits] with values in {0, 1}.
    """
    if seed_per_batch:
        gen = torch.Generator(device='cpu').manual_seed(0)
        bits = torch.randint(0, 2, (batch_size, n_bits), generator=gen).to(device)
    else:
        bits = torch.randint(0, 2, (batch_size, n_bits), device=device)
    return bits.float()


def bits_to_signed(bits: torch.Tensor) -> torch.Tensor:
    """Map bits in {0, 1} to {-1, +1} for symmetric embedding.

    Symmetric encoding tends to train more stably because the projection
    has zero mean.
    """
    return bits * 2.0 - 1.0


def signed_to_bits(signed: torch.Tensor) -> torch.Tensor:
    """Inverse of bits_to_signed for hard decoding."""
    return (signed > 0).float()


# -------------------------
# Accuracy metrics
# -------------------------

def bit_accuracy(predicted_logits: torch.Tensor, true_bits: torch.Tensor) -> float:
    """Compute fraction of bits correctly predicted.

    Args:
        predicted_logits: [B, n_bits] raw logits from decoder.
        true_bits: [B, n_bits] ground-truth bits in {0, 1}.

    Returns:
        Scalar in [0, 1] representing average bit accuracy across the batch.
    """
    with torch.no_grad():
        predicted = (predicted_logits > 0.0).float()
        correct = (predicted == true_bits).float()
        return correct.mean().item()


def message_accuracy(predicted_logits: torch.Tensor, true_bits: torch.Tensor) -> float:
    """Compute fraction of messages where ALL bits are correctly predicted.

    Stricter metric than bit_accuracy — useful for variable-length payload
    where partial decoding is useless.
    """
    with torch.no_grad():
        predicted = (predicted_logits > 0.0).float()
        all_correct = (predicted == true_bits).all(dim=1).float()
        return all_correct.mean().item()


# -------------------------
# Message embedding (encoder side)
# -------------------------

class MessageEmbedding(nn.Module):
    """Project a bit message to a feature embedding for generator conditioning.

    Used by the encoder/generator to inject the message signal into the
    watermark generation process. Embedding dimensionality should match the
    feature channel count of the latent / skip64 path where it is injected.

    Architecture: simple MLP with one hidden layer. Kept lightweight so it
    adds minimal parameters compared to the full generator.
    """

    def __init__(self, n_bits: int = 32, embed_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.n_bits = n_bits
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(n_bits, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        """Project bits to embedding.

        Args:
            bits: [B, n_bits] in {0, 1}.

        Returns:
            [B, embed_dim] feature embedding.
        """
        signed = bits_to_signed(bits)  # symmetric input is more stable
        return self.net(signed)


# -------------------------
# Conditioned residual branch (generator side)
# -------------------------

class ConditionedResidualBranch(nn.Module):
    """Generate a feature-map residual conditioned on a message embedding.

    Lives in parallel with the existing GLat / G64 watermark refiners. The
    existing refiners produce a content-only residual; this branch produces
    a message-specific additive component. Their sum is the final residual.

    Architecture: FiLM-style modulation of a small conv stack. The message
    embedding controls per-channel scale and shift, allowing the network to
    'paint' the message pattern across the feature map without changing the
    spatial resolution or feature dimensionality.

    Args:
        in_ch: feature channels of the latent / skip64 path being modulated.
        msg_dim: dimensionality of the message embedding.
        mid: internal channel count (keep small to limit parameters).
    """

    def __init__(self, in_ch: int, msg_dim: int = 256, mid: int = 32):
        super().__init__()
        self.in_ch = in_ch
        self.mid = mid

        # Reduce input channels (mirrors _MiniUNetWM structure)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, 1, 0, bias=False),
            nn.GroupNorm(min(8, mid), mid),
            nn.SiLU(inplace=True),
        )

        # FiLM projection: msg_emb -> per-channel (gamma, beta)
        self.film = nn.Linear(msg_dim, 2 * mid)
        # Initialize FiLM to start as identity (gamma=0, beta=0 after init)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        # Conv stack after FiLM modulation
        self.conv = nn.Sequential(
            nn.Conv2d(mid, mid, 3, 1, 1, bias=False),
            nn.GroupNorm(min(8, mid), mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, mid, 3, 1, 1, bias=False),
            nn.GroupNorm(min(8, mid), mid),
            nn.SiLU(inplace=True),
        )

        # Project back to input channel count
        self.out = nn.Conv2d(mid, in_ch, 1, 1, 0)
        # Initialize output to zero so the branch starts as identity (zero residual)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, msg_emb: torch.Tensor) -> torch.Tensor:
        """Generate message-conditioned residual.

        Args:
            x: [B, in_ch, H, W] feature map (latent or skip64).
            msg_emb: [B, msg_dim] message embedding.

        Returns:
            [B, in_ch, H, W] additive residual to be summed with the existing
            watermark residual from GLat / G64.
        """
        h = self.reduce(x)  # [B, mid, H, W]

        # FiLM modulation
        film_params = self.film(msg_emb)  # [B, 2*mid]
        gamma, beta = film_params.chunk(2, dim=1)  # [B, mid] each
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, mid, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        h = h * (1.0 + gamma) + beta  # FiLM applied
        h = self.conv(h)
        return self.out(h)  # [B, in_ch, H, W]


# -------------------------
# Utility for shortening / padding messages (for future variable length)
# -------------------------

def pack_message_with_length(payload_bits: torch.Tensor, max_bits: int,
                              length_header_bits: int = 4) -> torch.Tensor:
    """Prepare a message with a length header for variable-length payload.

    NOT used in the current pipeline (we use fixed-length). Provided as a
    forward-compatible utility for future variable-length extension.

    Args:
        payload_bits: [B, L] where L <= 2 ** length_header_bits - 1
        max_bits: total bit budget.
        length_header_bits: bits reserved for length prefix.

    Returns:
        [B, max_bits] packed message: [length_header | payload | padding].
    """
    B, L = payload_bits.shape
    max_payload = 2 ** length_header_bits - 1
    if L > max_payload:
        raise ValueError(f"Payload length {L} exceeds max {max_payload}")

    # Encode length as binary
    length_header = []
    for i in range(length_header_bits):
        bit = (L >> i) & 1
        length_header.append(torch.full((B, 1), float(bit), device=payload_bits.device))
    length_tensor = torch.cat(length_header, dim=1)

    # Pad payload
    pad_len = max_bits - length_header_bits - L
    if pad_len < 0:
        raise ValueError(f"max_bits {max_bits} too small for {length_header_bits}+{L}")
    padding = torch.zeros(B, pad_len, device=payload_bits.device)

    return torch.cat([length_tensor, payload_bits, padding], dim=1)
