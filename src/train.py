"""
train.py — Training loop for the steganography encoder-decoder system.

This module implements the complete training procedure:
  1. Forward pass: cover image + message → encoder → stego image
  2. Noise injection: stego image → random distortion → noised stego
  3. Message extraction: noised stego → decoder → decoded message
  4. Loss computation: image fidelity loss + message recovery loss
  5. Backward pass: gradient update via Adam optimizer
  6. Logging: metrics printed every epoch with checkpoint saving

The training loop supports:
  - Distortion-aware training via the CombinedNoiseLayer
  - Periodic validation evaluation
  - Best-model checkpointing based on validation bit accuracy
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    CHECKPOINT_DIR, MESSAGE_LENGTH,
    WARMUP_EPOCHS, NOISE_RAMP_EPOCHS,
)
from src.model import Encoder, Decoder
from src.noise_layers import CombinedNoiseLayer
from src.losses import StegLoss
from src.utils import compute_psnr, compute_ssim, compute_bit_accuracy, save_checkpoint, load_checkpoint


def train_one_epoch(
    encoder: Encoder,
    decoder: Decoder,
    noise_layer: CombinedNoiseLayer,
    criterion: StegLoss,
    optimizer: optim.Optimizer,
    train_loader,
    device: torch.device,
    epoch: int,
    warmup_epochs: int = 0,
    noise_strength: float = 1.0,
):
    """
    Train for one epoch.

    Args:
        encoder: Encoder network.
        decoder: Decoder network.
        noise_layer: Combined noise layer for distortion-aware training.
        criterion: Combined loss function.
        optimizer: Adam optimizer for both encoder and decoder.
        train_loader: DataLoader providing (cover_image, message) batches.
        device: Device to run training on (CPU or CUDA).
        epoch: Current epoch number (for display).
        warmup_epochs: Number of initial epochs to train without noise.

    Returns:
        epoch_metrics (dict): Average metrics over the epoch.
    """
    encoder.train()
    decoder.train()
    noise_layer.train()

    # Accumulators for epoch-level metrics
    total_loss_sum = 0.0
    image_loss_sum = 0.0
    message_loss_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    bit_acc_sum = 0.0
    num_batches = 0

    # Noise type distribution tracking
    noise_counts = {}

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    for cover_images, messages in progress_bar:
        # Move data to device
        cover_images = cover_images.to(device)
        messages = messages.to(device)

        # ---- Forward Pass ----
        # Step 1: Encode the message into the cover image
        stego_images = encoder(cover_images, messages)

        # Step 2: Apply random distortion (skip during warmup phase)
        if epoch > warmup_epochs:
            # Use noise after warmup period with progressive strength
            noised_images, noise_name = noise_layer(stego_images, cover_images, noise_strength=noise_strength)
        else:
            # No noise during warmup - let model learn clean embedding first
            noised_images, noise_name = stego_images, "warmup_no_noise"

        # Track noise distribution
        noise_counts[noise_name] = noise_counts.get(noise_name, 0) + 1

        # Step 3: Decode the message from the (possibly noised) stego image
        decoded_messages = decoder(noised_images)

        # Step 4: Compute combined loss
        loss_dict = criterion(cover_images, stego_images, messages, decoded_messages)
        total_loss = loss_dict["total"]

        # ---- Backward Pass ----
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)

        optimizer.step()

        # ---- Metrics ----
        with torch.no_grad():
            psnr = compute_psnr(cover_images, stego_images)
            ssim = compute_ssim(cover_images, stego_images)
            bit_acc = compute_bit_accuracy(messages, decoded_messages)

        total_loss_sum += loss_dict["total"].item()
        image_loss_sum += loss_dict["image_total"]
        message_loss_sum += loss_dict["message"]
        psnr_sum += psnr
        ssim_sum += ssim
        bit_acc_sum += bit_acc
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "PSNR": f"{psnr:.1f}",
            "BitAcc": f"{bit_acc:.3f}",
            "noise": noise_name,
        })

    # Compute epoch averages
    epoch_metrics = {
        "total_loss": total_loss_sum / max(num_batches, 1),
        "image_loss": image_loss_sum / max(num_batches, 1),
        "message_loss": message_loss_sum / max(num_batches, 1),
        "psnr": psnr_sum / max(num_batches, 1),
        "ssim": ssim_sum / max(num_batches, 1),
        "bit_accuracy": bit_acc_sum / max(num_batches, 1),
        "noise_distribution": noise_counts,
    }

    return epoch_metrics


@torch.no_grad()
def validate(
    encoder: Encoder,
    decoder: Decoder,
    val_loader,
    criterion: StegLoss,
    device: torch.device,
):
    """
    Evaluate the model on the validation set (without noise).

    Args:
        encoder: Encoder network.
        decoder: Decoder network.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        device: Device to run evaluation on.

    Returns:
        val_metrics (dict): Average validation metrics.
    """
    encoder.eval()
    decoder.eval()

    total_loss_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    bit_acc_sum = 0.0
    num_batches = 0

    for cover_images, messages in val_loader:
        cover_images = cover_images.to(device)
        messages = messages.to(device)

        # Forward pass (no noise during validation)
        stego_images = encoder(cover_images, messages)
        decoded_messages = decoder(stego_images)

        # Compute loss
        loss_dict = criterion(cover_images, stego_images, messages, decoded_messages)

        # Compute metrics
        psnr = compute_psnr(cover_images, stego_images)
        ssim = compute_ssim(cover_images, stego_images)
        bit_acc = compute_bit_accuracy(messages, decoded_messages)

        total_loss_sum += loss_dict["total"].item()
        psnr_sum += psnr
        ssim_sum += ssim
        bit_acc_sum += bit_acc
        num_batches += 1

    val_metrics = {
        "total_loss": total_loss_sum / max(num_batches, 1),
        "psnr": psnr_sum / max(num_batches, 1),
        "ssim": ssim_sum / max(num_batches, 1),
        "bit_accuracy": bit_acc_sum / max(num_batches, 1),
    }

    return val_metrics


def train(
    train_loader,
    val_loader=None,
    message_length: int = MESSAGE_LENGTH,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    device: torch.device = DEVICE,
    checkpoint_dir: str = CHECKPOINT_DIR,
    resume_checkpoint: str = None,
    warmup_epochs: int = WARMUP_EPOCHS,
    ramp_epochs: int = NOISE_RAMP_EPOCHS,
):
    """
    Full training procedure for the steganography system.

    Args:
        train_loader: Training DataLoader.
        val_loader: Optional validation DataLoader.
        message_length: Length of the hidden binary message.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for Adam optimizer.
        device: Device to train on.
        checkpoint_dir: Directory to save checkpoints.
        resume_checkpoint: Path to checkpoint to resume from.
        warmup_epochs: Number of epochs to train without noise regularization.

    Returns:
        encoder: Trained encoder network.
        decoder: Trained decoder network.
        history: Dictionary of training history (metrics over epochs).
    """
    print("=" * 70)
    print("  ROBUST IMAGE STEGANOGRAPHY — TRAINING")
    print("=" * 70)
    print(f"  Device:          {device}")
    print(f"  Message length:  {message_length} bits")
    print(f"  Epochs:          {num_epochs}")
    print(f"  Learning rate:   {learning_rate}")
    print(f"  Checkpoint dir:  {checkpoint_dir}")
    print("=" * 70)

    # Initialize models
    encoder = Encoder(message_length=message_length).to(device)
    decoder = Decoder(message_length=message_length).to(device)
    noise_layer = CombinedNoiseLayer().to(device)
    criterion = StegLoss().to(device)

    # Optimizer for both encoder and decoder jointly
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    # Learning rate scheduler: reduce LR when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )

    # Resume from checkpoint if specified
    start_epoch = 1
    best_bit_accuracy = 0.0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"\n[Resume] Loading checkpoint from '{resume_checkpoint}'...")
        encoder, decoder, start_epoch_loaded, loaded_metrics = load_checkpoint(
            encoder, decoder, resume_checkpoint, optimizer
        )
        start_epoch = start_epoch_loaded + 1
        best_bit_accuracy = loaded_metrics.get("bit_accuracy", 0.0)
        print(f"[Resume] Resuming from epoch {start_epoch}")
        print(f"[Resume] Previous best bit accuracy: {best_bit_accuracy:.4f}")
    elif resume_checkpoint:
        print(f"\n[Warning] Resume checkpoint '{resume_checkpoint}' not found. Starting from scratch.")

    # Training history
    history = {"train": [], "val": []}

    for epoch in range(start_epoch, num_epochs + 1):
        start_time = time.time()

        # Compute progressive noise strength for this epoch
        if epoch <= warmup_epochs:
            noise_strength = 0.0
        elif epoch <= warmup_epochs + ramp_epochs:
            noise_strength = (epoch - warmup_epochs) / ramp_epochs
        else:
            noise_strength = 1.0

        # ---- Training ----
        train_metrics = train_one_epoch(
            encoder, decoder, noise_layer, criterion, optimizer,
            train_loader, device, epoch, warmup_epochs=warmup_epochs,
            noise_strength=noise_strength,
        )
        history["train"].append(train_metrics)

        # ---- Validation ----
        val_metrics = None
        if val_loader is not None:
            val_metrics = validate(encoder, decoder, val_loader, criterion, device)
            history["val"].append(val_metrics)

        # Update learning rate scheduler
        scheduler.step(train_metrics["total_loss"])

        elapsed = time.time() - start_time

        # ---- Logging ----
        print(f"\nEpoch {epoch}/{num_epochs} ({elapsed:.1f}s)")
        print(f"  Train | Loss: {train_metrics['total_loss']:.4f} | "
              f"PSNR: {train_metrics['psnr']:.2f} dB | "
              f"SSIM: {train_metrics['ssim']:.4f} | "
              f"BitAcc: {train_metrics['bit_accuracy']:.4f}")

        if val_metrics:
            print(f"  Val   | Loss: {val_metrics['total_loss']:.4f} | "
                  f"PSNR: {val_metrics['psnr']:.2f} dB | "
                  f"SSIM: {val_metrics['ssim']:.4f} | "
                  f"BitAcc: {val_metrics['bit_accuracy']:.4f}")

        # ---- Checkpointing ----
        # Save latest checkpoint every epoch
        save_checkpoint(
            encoder, decoder, optimizer, epoch, train_metrics,
            os.path.join(checkpoint_dir, "latest_model.pth"),
        )

        # Save best model based on bit accuracy
        current_bit_acc = (val_metrics or train_metrics)["bit_accuracy"]
        if current_bit_acc > best_bit_accuracy:
            best_bit_accuracy = current_bit_acc
            save_checkpoint(
                encoder, decoder, optimizer, epoch, train_metrics,
                os.path.join(checkpoint_dir, "best_model.pth"),
            )
            print(f"  ★ New best bit accuracy: {best_bit_accuracy:.4f}")

        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_checkpoint(
                encoder, decoder, optimizer, epoch, train_metrics,
                os.path.join(checkpoint_dir, f"epoch_{epoch}.pth"),
            )

    print("\n" + "=" * 70)
    print(f"  Training complete! Best bit accuracy: {best_bit_accuracy:.4f}")
    print("=" * 70)

    return encoder, decoder, history
