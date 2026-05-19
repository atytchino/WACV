"""Per-dataset configuration for multi-dataset training.

Centralizes the differences between grayscale ORNL, AFHQ (color animal), and
Tomato Leaf datasets so the main trainer stays dataset-agnostic.

Each config returns a dict with:
  - in_channels: 1 (grayscale) or 3 (color)
  - num_classes: number of classification classes
  - image_size: (H, W) input size expected by the pipeline
  - default_batch_size, default_num_workers
  - decoder_arch: which backbone size for the decoder
  - decoder_n_bits: number of watermark bits to embed
  - notes: human-readable description

Also provides class-discovery helpers that:
  - Infer class names from subfolder structure
  - Validate train/val class alignment case-insensitively
  - Normalize class names to a consistent form

Designed to be loaded once at trainer startup, used everywhere downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


# =================================================================
# Dataset configuration dataclass
# =================================================================

@dataclass
class DatasetConfig:
    """All hyperparameters that depend on the dataset."""

    name: str
    in_channels: int                  # 1 or 3
    num_classes: int
    image_size: Tuple[int, int]       # (H, W)
    default_batch_size: int
    default_num_workers: int
    decoder_arch: str                 # 'resnet18' or 'resnet34'
    decoder_n_bits: int               # bits in the watermark payload
    msg_embed_dim: int                # message embedding dimensionality
    notes: str = ""

    # Per-dataset hyperparameter overrides (optional)
    gate_strength: float = 2.10
    ssim_lam: float = 0.60
    transfer_lam: float = 0.50
    diversity_lam: float = 0.10
    decoder_loss_weight: float = 1.0   # weight of L_decoder in total loss
    decoder_warmup_epochs: int = 1     # epochs before decoder loss kicks in

    # Option B: split-channel encoding (set use_split_channels=True to enable)
    # Total payload = decoder_n_bits_lat + decoder_n_bits_skip.
    # If use_split_channels=False (default), the single-channel decoder is used
    # with decoder_n_bits as the total bit count.
    use_split_channels: bool = False
    decoder_n_bits_lat: int = 0        # bits encoded in latent path
    decoder_n_bits_skip: int = 0       # bits encoded in skip64 path


# =================================================================
# Concrete configurations
# =================================================================

DATASETS: Dict[str, DatasetConfig] = {

    'ornl_grayscale': DatasetConfig(
        name='ornl_grayscale',
        in_channels=3,                # dataset loader returns RGB-replicated grayscale
        num_classes=4,
        image_size=(120, 160),
        default_batch_size=16,
        default_num_workers=2,
        decoder_arch='resnet34',
        decoder_n_bits=64,            # SoTA-comparable payload (HiDDeN: 30-64)
        msg_embed_dim=256,            # smaller embed for low-texture grayscale
        notes="Original grayscale 3D-printing dataset. Featureless, small images.",
        gate_strength=2.10,
        ssim_lam=0.60,
        transfer_lam=0.50,
        diversity_lam=0.10,
        decoder_loss_weight=1.0,
        decoder_warmup_epochs=1,
    ),

    'afhq_color': DatasetConfig(
        name='afhq_color',
        in_channels=3,
        num_classes=3,                # cat, dog, wild — inferred at runtime
        image_size=(512, 512),        # native AFHQ resolution
        default_batch_size=4,         # small due to 512x512 memory footprint
        default_num_workers=4,
        decoder_arch='resnet34',
        decoder_n_bits=64,            # SoTA-comparable payload
        msg_embed_dim=384,            # larger embed — color textures support more signal
        notes="AFHQ animal faces (cat/dog/wild). Native 512x512 RGB JPG.",
        gate_strength=2.10,
        ssim_lam=0.80,                # tighter visual quality target on color
        transfer_lam=0.50,
        diversity_lam=0.10,
        decoder_loss_weight=1.0,
        decoder_warmup_epochs=1,
    ),

    'tomato_leaf': DatasetConfig(
        name='tomato_leaf',
        in_channels=3,
        num_classes=10,               # actual count inferred from disk at runtime
        image_size=(512, 512),        # native TLD resolution
        default_batch_size=4,
        default_num_workers=4,
        decoder_arch='resnet34',
        decoder_n_bits=64,            # SoTA-comparable payload
        msg_embed_dim=384,            # larger embed for color leaf textures
        notes="Tomato Leaf Disease (TLD). Native 512x512 RGB JPG.",
        gate_strength=2.10,
        ssim_lam=0.80,
        transfer_lam=0.50,
        diversity_lam=0.10,
        decoder_loss_weight=1.0,
        decoder_warmup_epochs=1,
    ),
}


# =================================================================
# Config lookup
# =================================================================

def get_config(dataset_key: str) -> DatasetConfig:
    """Look up a dataset config by key.

    Raises:
        KeyError if the dataset is not registered.
    """
    if dataset_key not in DATASETS:
        valid = ', '.join(sorted(DATASETS.keys()))
        raise KeyError(
            f"Unknown dataset '{dataset_key}'. Valid options: {valid}"
        )
    return DATASETS[dataset_key]


def list_datasets() -> Dict[str, str]:
    """Return mapping of dataset_key -> notes for CLI help / display."""
    return {k: v.notes for k, v in DATASETS.items()}


# =================================================================
# Class discovery & validation helpers (case-insensitive)
# =================================================================

def _normalize_class_name(name: str) -> str:
    """Normalize a class name for case-insensitive comparison.

    Currently: lowercase + strip whitespace. Does NOT modify the original
    name returned to the trainer — only used for comparison.
    """
    return name.strip().lower()


def infer_classes_from_subfolders(root: Path,
                                  sort_alphabetical: bool = True) -> List[str]:
    """Infer class names from immediate subfolder names of `root`.

    Subfolder names are taken AS-IS (original case preserved). The trainer
    will use these strings as canonical class labels.

    Args:
        root: dataset root (e.g., E:/AFHQ/train).
        sort_alphabetical: if True, return classes in case-insensitive
            alphabetical order — ensures deterministic class index mapping
            even across operating systems with different sort conventions.

    Returns:
        List of class name strings.

    Raises:
        RuntimeError if no class subfolders are found.
        FileNotFoundError if root does not exist.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    classes = [p.name for p in root.iterdir() if p.is_dir()]
    if not classes:
        raise RuntimeError(
            f"No class subfolders found in: {root}\n"
            f"Expected structure: {root}/<class_name>/*.jpg"
        )

    if sort_alphabetical:
        classes.sort(key=_normalize_class_name)
    return classes


def verify_class_alignment(train_root: Path, val_root: Path,
                            case_insensitive: bool = True) -> List[str]:
    """Verify that train and val have matching class folder structure.

    Class names must match between train and val. By default the comparison
    is case-insensitive — 'Cat' in train and 'cat' in val are considered the
    same class (the train spelling is used as canonical).

    Args:
        train_root: path to training data root.
        val_root: path to validation data root.
        case_insensitive: if True (default), 'Cat' and 'cat' match.

    Returns:
        List of canonical class names (using train_root spelling).

    Raises:
        RuntimeError with a detailed diff message if classes do not align.
    """
    train_classes = infer_classes_from_subfolders(Path(train_root))
    val_classes = infer_classes_from_subfolders(Path(val_root))

    if case_insensitive:
        train_norm = {_normalize_class_name(c): c for c in train_classes}
        val_norm = {_normalize_class_name(c): c for c in val_classes}

        # Compare normalized sets
        only_train = set(train_norm.keys()) - set(val_norm.keys())
        only_val = set(val_norm.keys()) - set(train_norm.keys())

        if only_train or only_val:
            lines = [
                f"[CLASS MISMATCH] train and val have different classes:",
                f"  train_root : {train_root}",
                f"  val_root   : {val_root}",
                f"  train classes ({len(train_classes)}): {sorted(train_classes)}",
                f"  val classes   ({len(val_classes)}):   {sorted(val_classes)}",
            ]
            if only_train:
                lines.append(f"  In train but NOT in val: {sorted(only_train)}")
            if only_val:
                lines.append(f"  In val but NOT in train: {sorted(only_val)}")
            lines.append("  Fix: make subfolder names match (case-insensitive).")
            raise RuntimeError("\n".join(lines))

        # Counts must match
        if len(train_classes) != len(val_classes):
            raise RuntimeError(
                f"Class count mismatch: train has {len(train_classes)}, "
                f"val has {len(val_classes)}. Train: {train_classes}; "
                f"Val: {val_classes}"
            )

        # Use train spelling as canonical — sorted alphabetically (case-insensitive)
        canonical = sorted(train_classes, key=_normalize_class_name)
        return canonical
    else:
        # Strict matching — exact case
        if train_classes != val_classes:
            raise RuntimeError(
                f"[CLASS MISMATCH] (strict case-sensitive):\n"
                f"  train: {train_classes}\n"
                f"  val:   {val_classes}"
            )
        return list(train_classes)


def class_to_index_map(classes: List[str]) -> Dict[str, int]:
    """Build a class-name -> integer-index mapping.

    Used by datasets to convert subfolder names to integer labels.
    """
    return {c: i for i, c in enumerate(classes)}
