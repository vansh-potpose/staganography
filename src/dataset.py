"""
dataset.py — Dataset loading and preprocessing for image steganography.

Provides a custom PyTorch Dataset class that:
  1. Loads images from a specified directory (including subdirectories).
  2. Resizes and normalizes images to a consistent format.
  3. Generates random binary messages for each image.
  4. Supports automatic train/val splitting from a single directory.
"""

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

from src.config import IMAGE_SIZE, MESSAGE_LENGTH, BATCH_SIZE


# ============================================================================
# Supported image file extensions
# ============================================================================
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def is_image_file(filename: str) -> bool:
    """Check if a file has a supported image extension."""
    return os.path.splitext(filename)[1].lower() in SUPPORTED_EXTENSIONS


class StegDataset(Dataset):
    """
    Custom dataset for image steganography training and evaluation.

    For each sample, the dataset provides:
      - cover_image: A normalized RGB image tensor of shape (3, H, W)
      - message: A random binary tensor of shape (MESSAGE_LENGTH,)

    Args:
        data_dir (str): Path to directory containing images.
        image_size (int): Target spatial size (images are resized to image_size x image_size).
        message_length (int): Number of bits in the hidden message.
        train (bool): If True, applies data augmentation (random horizontal flip).
        image_paths (list): Optional pre-computed list of image paths (overrides data_dir scan).
    """

    def __init__(
        self,
        data_dir: str = None,
        image_size: int = IMAGE_SIZE,
        message_length: int = MESSAGE_LENGTH,
        train: bool = True,
        image_paths: list = None,
    ):
        super().__init__()
        self.message_length = message_length
        self.train = train

        # Use provided image paths or scan directory
        if image_paths is not None:
            self.image_paths = image_paths
        elif data_dir is not None:
            self.image_paths = []
            for root, dirs, files in os.walk(data_dir):
                for fname in files:
                    if is_image_file(fname):
                        self.image_paths.append(os.path.join(root, fname))
            self.image_paths = sorted(self.image_paths)
        else:
            raise ValueError("Either data_dir or image_paths must be provided.")

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No supported image files found in '{data_dir}' or its subdirectories. "
                f"Supported extensions: {SUPPORTED_EXTENSIONS}"
            )

        # Define image transforms
        transform_list = [
            transforms.Resize((image_size, image_size)),
        ]
        if train:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
            transform_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))
        transform_list.append(transforms.ToTensor())  # Converts to [0, 1] range

        self.transform = transforms.Compose(transform_list)

        print(f"[Dataset] Loaded {len(self.image_paths)} images (train={train})")

    def __len__(self) -> int:
        """Return the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        Load a single sample.

        Returns:
            cover_image (Tensor): Shape (3, image_size, image_size), values in [0, 1].
            message (Tensor): Shape (message_length,), binary values {0, 1} as float.
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        cover_image = self.transform(image)

        # Generate a random binary message
        message = torch.randint(0, 2, (self.message_length,)).float()

        return cover_image, message


def get_data_loaders(
    data_dir: str = None,
    train_dir: str = None,
    val_dir: str = None,
    image_size: int = IMAGE_SIZE,
    message_length: int = MESSAGE_LENGTH,
    batch_size: int = BATCH_SIZE,
    val_split: float = 0.1,
    num_workers: int = 4,
):
    """
    Create training and validation DataLoaders.

    Supports two modes:
      1. Single directory mode: Provide `data_dir` and `val_split` to auto-split.
      2. Separate directories: Provide `train_dir` and optionally `val_dir`.

    Args:
        data_dir (str): Path to a single directory (splits into train/val automatically).
        train_dir (str): Path to training images directory.
        val_dir (str): Path to validation images directory.
        image_size (int): Target image size.
        message_length (int): Hidden message length in bits.
        batch_size (int): Batch size for DataLoaders.
        val_split (float): Fraction of data to use for validation (only for data_dir mode).
        num_workers (int): Number of worker processes for data loading.

    Returns:
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader (may be None).
    """
    # Determine which mode to use
    effective_dir = data_dir or train_dir
    if effective_dir is None:
        raise ValueError("Either data_dir or train_dir must be provided.")

    if data_dir is not None and val_dir is None:
        # Single directory mode: auto-split into train/val
        all_paths = []
        for root, dirs, files in os.walk(data_dir):
            for fname in files:
                if is_image_file(fname):
                    all_paths.append(os.path.join(root, fname))
        all_paths = sorted(all_paths)

        if len(all_paths) == 0:
            raise FileNotFoundError(f"No images found in '{data_dir}'")

        # Split into train and val
        random.shuffle(all_paths)
        split_idx = int(len(all_paths) * (1 - val_split))
        train_paths = all_paths[:split_idx]
        val_paths = all_paths[split_idx:]

        print(f"[DataLoader] Auto-split: {len(train_paths)} train, {len(val_paths)} val")

        train_dataset = StegDataset(
            image_paths=train_paths,
            image_size=image_size,
            message_length=message_length,
            train=True,
        )
        val_dataset = StegDataset(
            image_paths=val_paths,
            image_size=image_size,
            message_length=message_length,
            train=False,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        return train_loader, val_loader

    else:
        # Separate directories mode
        train_dataset = StegDataset(
            data_dir=effective_dir,
            image_size=image_size,
            message_length=message_length,
            train=True,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )

        val_loader = None
        if val_dir is not None:
            val_dataset = StegDataset(
                data_dir=val_dir,
                image_size=image_size,
                message_length=message_length,
                train=False,
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
            )

        return train_loader, val_loader
