"""
dataset.py — Dataset loading and preprocessing for image steganography.

Provides a custom PyTorch Dataset class that:
  1. Loads images from a specified directory.
  2. Resizes and normalizes images to a consistent format.
  3. Generates random binary messages for each image.
"""

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
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
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = IMAGE_SIZE,
        message_length: int = MESSAGE_LENGTH,
        train: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.message_length = message_length
        self.train = train

        # Collect all valid image file paths
        self.image_paths = sorted([
            os.path.join(data_dir, fname)
            for fname in os.listdir(data_dir)
            if is_image_file(fname)
        ])

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No supported image files found in '{data_dir}'. "
                f"Supported extensions: {SUPPORTED_EXTENSIONS}"
            )

        # Define image transforms
        transform_list = [
            transforms.Resize((image_size, image_size)),
        ]
        if train:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transform_list.append(transforms.ToTensor())  # Converts to [0, 1] range

        self.transform = transforms.Compose(transform_list)

        print(f"[Dataset] Loaded {len(self.image_paths)} images from '{data_dir}' "
              f"(train={train})")

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
        # Load and transform the cover image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        cover_image = self.transform(image)

        # Generate a random binary message
        message = torch.randint(0, 2, (self.message_length,)).float()

        return cover_image, message


def get_data_loaders(
    train_dir: str,
    val_dir: str = None,
    image_size: int = IMAGE_SIZE,
    message_length: int = MESSAGE_LENGTH,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 4,
):
    """
    Create training and (optionally) validation DataLoaders.

    Args:
        train_dir (str): Path to training images directory.
        val_dir (str, optional): Path to validation images directory.
        image_size (int): Target image size.
        message_length (int): Hidden message length in bits.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of worker processes for data loading.

    Returns:
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader or None): Validation data loader if val_dir is provided.
    """
    # Training dataset and loader
    train_dataset = StegDataset(
        data_dir=train_dir,
        image_size=image_size,
        message_length=message_length,
        train=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Validation dataset and loader (optional)
    val_loader = None
    if val_dir is not None:
        val_dataset = StegDataset(
            data_dir=val_dir,
            image_size=image_size,
            message_length=message_length,
            train=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader
