"""
test_robustness.py — CLI entry point for robustness testing.

Usage:
    python scripts/test_robustness.py --checkpoint <path> --data_dir <path> [OPTIONS]

Example:
    python scripts/test_robustness.py \
        --checkpoint checkpoints/best_model.pth \
        --data_dir data/val2017 \
        --output_dir results/
"""

import sys
import os
import argparse

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch

from src.config import DEVICE, MESSAGE_LENGTH, IMAGE_SIZE, BATCH_SIZE, RESULTS_DIR
from src.model import Encoder, Decoder
from src.dataset import get_data_loaders
from src.evaluate import evaluate_robustness
from src.utils import load_checkpoint


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test robustness of the trained steganography model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to the directory containing test images.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=RESULTS_DIR,
        help="Directory to save evaluation results and charts.",
    )
    parser.add_argument(
        "--message_length", type=int, default=MESSAGE_LENGTH,
        help="Message length (must match the trained model).",
    )
    parser.add_argument(
        "--image_size", type=int, default=IMAGE_SIZE,
        help="Image size (must match the trained model).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of data loading workers.",
    )

    return parser.parse_args()


def main():
    """Main robustness testing entry point."""
    args = parse_args()

    print("\n" + "=" * 70)
    print("  ROBUST IMAGE STEGANOGRAPHY — ROBUSTNESS TESTING")
    print("=" * 70)
    print(f"  Checkpoint:     {args.checkpoint}")
    print(f"  Test data:      {args.data_dir}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Message length: {args.message_length} bits")
    print(f"  Image size:     {args.image_size}×{args.image_size}")
    print(f"  Device:         {DEVICE}")
    print("=" * 70 + "\n")

    # Validate inputs
    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: Checkpoint file '{args.checkpoint}' not found.")
        sys.exit(1)

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: Test data directory '{args.data_dir}' does not exist.")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize models
    print("[1/3] Loading model...")
    encoder = Encoder(message_length=args.message_length).to(DEVICE)
    decoder = Decoder(message_length=args.message_length).to(DEVICE)

    # Load checkpoint
    encoder, decoder, epoch, metrics = load_checkpoint(encoder, decoder, args.checkpoint)
    print(f"       Loaded checkpoint from epoch {epoch}")

    # Create test data loader (use data_dir as single directory, no val split needed)
    print("[2/3] Creating test data loader...")
    test_loader, _ = get_data_loaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        message_length=args.message_length,
        batch_size=args.batch_size,
        val_split=0.0,  # Use all data for testing
        num_workers=args.num_workers,
    )

    # Run robustness evaluation
    print("[3/3] Running robustness evaluation...\n")
    results = evaluate_robustness(
        encoder=encoder,
        decoder=decoder,
        test_loader=test_loader,
        device=DEVICE,
        output_dir=args.output_dir,
    )

    # Summary
    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print(f"  Results saved to: {args.output_dir}")
    print(f"  - robustness_chart.png     (bar chart)")
    print(f"  - sample_comparisons.png   (cover vs stego)")
    print("=" * 70)


if __name__ == "__main__":
    main()
