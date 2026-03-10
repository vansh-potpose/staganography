#!/usr/bin/env python3
"""
Quick training script with all improvements enabled.
Run this to train the model with the new architecture and settings.
"""

import sys
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, '.')

from src.config import (
    BATCH_SIZE, DEVICE, NUM_EPOCHS, CHECKPOINT_DIR,
    MESSAGE_LENGTH, IMAGE_SIZE
)
from src.dataset import StegDataset
from src.train import train


def main():
    print("\n" + "="*70)
    print("ROBUST IMAGE STEGANOGRAPHY - TRAINING WITH IMPROVEMENTS")
    print("="*70)
    
    # Check device
    print(f"Device: {DEVICE}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    # Create datasets
    print("\n[Dataset] Loading training data...")
    try:
        train_dataset = StegDataset(
            'data/train',
            image_size=IMAGE_SIZE,
            message_length=MESSAGE_LENGTH,
            train=True
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Make sure your data is in the 'data/train' directory")
        return
    
    print("[Dataset] Loading validation data...")
    try:
        val_dataset = StegDataset(
            'data/val',
            image_size=IMAGE_SIZE,
            message_length=MESSAGE_LENGTH,
            train=False
        )
    except FileNotFoundError:
        print("WARNING: Validation data not found. Using training data for validation.")
        val_dataset = train_dataset
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 2-4 if on Linux/Mac for parallel loading
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\n[DataLoader] Training batches: {len(train_loader)}")
    print(f"[DataLoader] Validation batches: {len(val_loader)}")
    
    # Train with warmup!
    print(f"\n[Training] Starting training with improvements...")
    print(f"[Training] Warmup phase: 10 epochs (NO noise)")
    print(f"[Training] Robustness phase: {NUM_EPOCHS - 10} epochs (WITH noise)")
    
    encoder, decoder, history = train(
        train_loader=train_loader,
        val_loader=val_loader,
        message_length=MESSAGE_LENGTH,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        checkpoint_dir=CHECKPOINT_DIR,
        warmup_epochs=10,  # KEY: Warmup without noise
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best model saved to: {CHECKPOINT_DIR}/best_model.pth")
    print(f"Latest model saved to: {CHECKPOINT_DIR}/latest_model.pth")
    print("\nKey Improvements Applied:")
    print("✓ Decoder: Multi-scale attention (instead of global pooling)")
    print("✓ Capacity: Hidden channels 128 (2x larger)")
    print("✓ Batch Size: 32 (instead of 16)")
    print("✓ Warmup: 10 epochs without noise (curriculum learning)")
    print("✓ Loss Weights: Message loss 2x (was 1x), Image loss 0.5x (was 0.7x)")


if __name__ == '__main__':
    main()
