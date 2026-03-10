"""
train.py — Improved training loop for the steganography encoder-decoder system.

Key improvements:
  1. Progressive noise schedule (warmup → ramp → full)
  2. CosineAnnealingWarmRestarts scheduler for smoother LR decay
  3. Higher gradient clipping threshold (5.0 instead of 1.0)
  4. Noise strength ramping for gradual robustness learning
"""

import os
import time
import math
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


def get_noise_strength(epoch: int, warmup_epochs: int, ramp_epochs: int) -> float:
    """
    Compute noise strength for the current epoch.

    Schedule:
      - Epochs 1 to warmup_epochs: 0.0 (no noise, learn clean embedding)
      - Epochs warmup+1 to warmup+ramp: linear ramp from 0.0 to 1.0
      - Epochs after warmup+ramp: 1.0 (full noise)

    Returns:
        noise_strength: float from 0.0 to 1.0
    """
    if epoch <= warmup_epochs:
        return 0.0
    elif epoch <= warmup_epochs + ramp_epochs:
        progress = (epoch - warmup_epochs) / ramp_epochs
        return progress
    else:
        return 1.0


def train_one_epoch(
    encoder: Encoder,
    decoder: Decoder,
    noise_layer: CombinedNoiseLayer,
    criterion: StegLoss,
    optimizer: optim.Optimizer,
    train_loader,
    device: torch.device,
    epoch: int,
    warmup_epochs: int = WARMUP_EPOCHS,
    ramp_epochs: int = NOISE_RAMP_EPOCHS,
):
    """Train for one epoch with progressive noise."""
    encoder.train()
    decoder.train()
    noise_layer.train()

    # Compute noise strength for this epoch
    noise_strength = get_noise_strength(epoch, warmup_epochs, ramp_epochs)

    # Accumulators
    total_loss_sum = 0.0
    image_loss_sum = 0.0
    message_loss_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    bit_acc_sum = 0.0
    num_batches = 0
    noise_counts = {}

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    for cover_images, messages in progress_bar:
        cover_images = cover_images.to(device)
        messages = messages.to(device)

        # ---- Forward Pass ----
        stego_images = encoder(cover_images, messages)

        # Apply noise with progressive strength
        noised_images, noise_name = noise_layer(stego_images, cover_images,
                                                 noise_strength=noise_strength)
        noise_counts[noise_name] = noise_counts.get(noise_name, 0) + 1

        # Decode (outputs logits)
        decoded_logits = decoder(noised_images)

        # Compute loss (BCEWithLogitsLoss handles logits)
        loss_dict = criterion(cover_images, stego_images, messages, decoded_logits)
        total_loss = loss_dict["total"]

        # ---- Backward Pass ----
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping (more permissive = 5.0)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)

        optimizer.step()

        # ---- Metrics ----
        with torch.no_grad():
            psnr = compute_psnr(cover_images, stego_images)
            ssim = compute_ssim(cover_images, stego_images)
            # Apply sigmoid to logits for bit accuracy computation
            decoded_probs = torch.sigmoid(decoded_logits)
            bit_acc = compute_bit_accuracy(messages, decoded_probs)

        total_loss_sum += loss_dict["total"].item()
        image_loss_sum += loss_dict["image_total"]
        message_loss_sum += loss_dict["message"]
        psnr_sum += psnr
        ssim_sum += ssim
        bit_acc_sum += bit_acc
        num_batches += 1

        progress_bar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "PSNR": f"{psnr:.1f}",
            "BitAcc": f"{bit_acc:.3f}",
            "noise": noise_name,
            "ns": f"{noise_strength:.2f}",
        })

    epoch_metrics = {
        "total_loss": total_loss_sum / max(num_batches, 1),
        "image_loss": image_loss_sum / max(num_batches, 1),
        "message_loss": message_loss_sum / max(num_batches, 1),
        "psnr": psnr_sum / max(num_batches, 1),
        "ssim": ssim_sum / max(num_batches, 1),
        "bit_accuracy": bit_acc_sum / max(num_batches, 1),
        "noise_distribution": noise_counts,
        "noise_strength": noise_strength,
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
    """Evaluate the model on the validation set (without noise)."""
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

        stego_images = encoder(cover_images, messages)
        decoded_logits = decoder(stego_images)

        loss_dict = criterion(cover_images, stego_images, messages, decoded_logits)

        psnr = compute_psnr(cover_images, stego_images)
        ssim = compute_ssim(cover_images, stego_images)
        decoded_probs = torch.sigmoid(decoded_logits)
        bit_acc = compute_bit_accuracy(messages, decoded_probs)

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
    Full training procedure with progressive noise schedule.

    Noise schedule:
      - Epochs 1-5: No noise (warmup)
      - Epochs 6-25: Linear noise ramp (0→100%)
      - Epochs 26+: Full noise strength
    """
    print("=" * 70)
    print("  ROBUST IMAGE STEGANOGRAPHY — IMPROVED TRAINING")
    print("=" * 70)
    print(f"  Device:          {device}")
    print(f"  Message length:  {message_length} bits")
    print(f"  Epochs:          {num_epochs}")
    print(f"  Learning rate:   {learning_rate}")
    print(f"  Warmup epochs:   {warmup_epochs}")
    print(f"  Noise ramp:      {ramp_epochs} epochs")
    print(f"  Checkpoint dir:  {checkpoint_dir}")
    print("=" * 70)

    # Initialize models
    encoder = Encoder(message_length=message_length).to(device)
    decoder = Decoder(message_length=message_length).to(device)
    noise_layer = CombinedNoiseLayer().to(device)
    criterion = StegLoss().to(device)

    # Print model sizes
    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"\n  Encoder params:  {enc_params:,}")
    print(f"  Decoder params:  {dec_params:,}")
    print(f"  Total params:    {enc_params + dec_params:,}")
    print("=" * 70)

    # Optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    # CosineAnnealingWarmRestarts: smooth LR decay with periodic restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6,
    )

    # Resume from checkpoint
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

    history = {"train": [], "val": []}

    for epoch in range(start_epoch, num_epochs + 1):
        start_time = time.time()

        # Get current noise strength for logging
        noise_strength = get_noise_strength(epoch, warmup_epochs, ramp_epochs)

        # ---- Training ----
        train_metrics = train_one_epoch(
            encoder, decoder, noise_layer, criterion, optimizer,
            train_loader, device, epoch,
            warmup_epochs=warmup_epochs, ramp_epochs=ramp_epochs,
        )
        history["train"].append(train_metrics)

        # ---- Validation ----
        val_metrics = None
        if val_loader is not None:
            val_metrics = validate(encoder, decoder, val_loader, criterion, device)
            history["val"].append(val_metrics)

        # Update LR scheduler
        scheduler.step()

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        # ---- Logging ----
        print(f"\nEpoch {epoch}/{num_epochs} ({elapsed:.1f}s) | LR: {current_lr:.6f} | "
              f"Noise: {noise_strength:.2f}")
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
        save_checkpoint(
            encoder, decoder, optimizer, epoch, train_metrics,
            os.path.join(checkpoint_dir, "latest_model.pth"),
        )

        current_bit_acc = (val_metrics or train_metrics)["bit_accuracy"]
        if current_bit_acc > best_bit_accuracy:
            best_bit_accuracy = current_bit_acc
            save_checkpoint(
                encoder, decoder, optimizer, epoch, train_metrics,
                os.path.join(checkpoint_dir, "best_model.pth"),
            )
            print(f"  ★ New best bit accuracy: {best_bit_accuracy:.4f}")

        if epoch % 10 == 0:
            save_checkpoint(
                encoder, decoder, optimizer, epoch, train_metrics,
                os.path.join(checkpoint_dir, f"epoch_{epoch}.pth"),
            )

    print("\n" + "=" * 70)
    print(f"  Training complete! Best bit accuracy: {best_bit_accuracy:.4f}")
    print("=" * 70)

    return encoder, decoder, history
