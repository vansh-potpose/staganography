"""
train_model.py — CLI entry point for training the steganography model.

Usage:
    python scripts/train_model.py --data_dir <path_to_images> [OPTIONS]

Example:
    python scripts/train_model.py --data_dir data/train2017 --epochs 100 --batch_size 16

This script:
  1. Parses command-line arguments for training configuration.
  2. Creates training (and optional validation) data loaders.
  3. Runs the full training loop with distortion-aware noise injection.
  4. Saves checkpoints to the specified directory.
"""

import sys
import os
import argparse

# Add project root to Python path so we can import from src/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.dataset import get_data_loaders
from src.train import train
from src.config import (
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, MESSAGE_LENGTH,
    IMAGE_SIZE, CHECKPOINT_DIR, DEVICE,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the Robust Image Steganography model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to the directory containing training images.",
    )

    # Optional arguments
    parser.add_argument(
        "--val_dir", type=str, default=None,
        help="Path to the directory containing validation images.",
    )
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help="Training batch size.",
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--message_length", type=int, default=MESSAGE_LENGTH,
        help="Length of the hidden binary message in bits.",
    )
    parser.add_argument(
        "--image_size", type=int, default=IMAGE_SIZE,
        help="Size to resize images to (square).",
    )
    parser.add_argument(
        "--save_dir", type=str, default=CHECKPOINT_DIR,
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of data loading workers.",
    )

    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()

    print("\n" + "=" * 70)
    print("  ROBUST IMAGE STEGANOGRAPHY — TRAINING SCRIPT")
    print("=" * 70)
    print(f"  Data directory:     {args.data_dir}")
    print(f"  Validation dir:     {args.val_dir or 'None (training only)'}")
    print(f"  Epochs:             {args.epochs}")
    print(f"  Batch size:         {args.batch_size}")
    print(f"  Learning rate:      {args.lr}")
    print(f"  Message length:     {args.message_length} bits")
    print(f"  Image size:         {args.image_size}×{args.image_size}")
    print(f"  Checkpoint dir:     {args.save_dir}")
    print(f"  Device:             {DEVICE}")
    print("=" * 70 + "\n")

    # Validate data directory exists
    if not os.path.isdir(args.data_dir):
        print(f"ERROR: Data directory '{args.data_dir}' does not exist.")
        sys.exit(1)

    if args.val_dir and not os.path.isdir(args.val_dir):
        print(f"ERROR: Validation directory '{args.val_dir}' does not exist.")
        sys.exit(1)

    # Create data loaders
    print("[1/2] Creating data loaders...")
    train_loader, val_loader = get_data_loaders(
        train_dir=args.data_dir,
        val_dir=args.val_dir,
        image_size=args.image_size,
        message_length=args.message_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Start training
    print("[2/2] Starting training...\n")
    encoder, decoder, history = train(
        train_loader=train_loader,
        val_loader=val_loader,
        message_length=args.message_length,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=DEVICE,
        checkpoint_dir=args.save_dir,
    )

    print("\nTraining complete! Checkpoints saved to:", args.save_dir)


if __name__ == "__main__":
    main()
