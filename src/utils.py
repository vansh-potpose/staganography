"""
utils.py — Utility functions for the steganography project.

Provides helper functions for:
  - Computing evaluation metrics (PSNR, SSIM, bit accuracy)
  - Image visualization and comparison
  - Checkpoint saving and loading
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ============================================================================
# Metric Computation
# ============================================================================

def compute_psnr(cover: torch.Tensor, stego: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between cover and stego images.

    Args:
        cover (Tensor): Cover image, shape (B, 3, H, W) or (3, H, W), range [0, 1].
        stego (Tensor): Stego image, same shape as cover.

    Returns:
        psnr (float): Average PSNR over the batch in dB.
    """
    cover_np = cover.detach().cpu().numpy()
    stego_np = stego.detach().cpu().numpy()

    if cover_np.ndim == 3:
        # Single image
        cover_np = np.transpose(cover_np, (1, 2, 0))  # (H, W, C)
        stego_np = np.transpose(stego_np, (1, 2, 0))
        return peak_signal_noise_ratio(cover_np, stego_np, data_range=1.0)
    else:
        # Batch of images
        psnr_values = []
        for i in range(cover_np.shape[0]):
            c = np.transpose(cover_np[i], (1, 2, 0))
            s = np.transpose(stego_np[i], (1, 2, 0))
            psnr_values.append(peak_signal_noise_ratio(c, s, data_range=1.0))
        return np.mean(psnr_values)


def compute_ssim(cover: torch.Tensor, stego: torch.Tensor) -> float:
    """
    Compute Structural Similarity Index between cover and stego images.

    Args:
        cover (Tensor): Cover image, shape (B, 3, H, W) or (3, H, W), range [0, 1].
        stego (Tensor): Stego image, same shape as cover.

    Returns:
        ssim (float): Average SSIM over the batch.
    """
    cover_np = cover.detach().cpu().numpy()
    stego_np = stego.detach().cpu().numpy()

    if cover_np.ndim == 3:
        cover_np = np.transpose(cover_np, (1, 2, 0))
        stego_np = np.transpose(stego_np, (1, 2, 0))
        return structural_similarity(cover_np, stego_np, data_range=1.0, channel_axis=2)
    else:
        ssim_values = []
        for i in range(cover_np.shape[0]):
            c = np.transpose(cover_np[i], (1, 2, 0))
            s = np.transpose(stego_np[i], (1, 2, 0))
            ssim_values.append(structural_similarity(c, s, data_range=1.0, channel_axis=2))
        return np.mean(ssim_values)


def compute_bit_accuracy(original: torch.Tensor, decoded: torch.Tensor) -> float:
    """
    Compute bit-level accuracy between original and decoded messages.

    Args:
        original (Tensor): Original binary message, shape (B, L), values in {0, 1}.
        decoded (Tensor): Decoded message probabilities, shape (B, L), values in [0, 1].

    Returns:
        accuracy (float): Fraction of correctly decoded bits (0.0 to 1.0).
    """
    # Apply sigmoid to convert logits to probabilities, then threshold at 0.5
    decoded_bits = (torch.sigmoid(decoded.detach()) > 0.5).float()
    original_bits = original.detach()

    # Compare bit by bit
    correct = (decoded_bits == original_bits).float()
    return correct.mean().item()


# ============================================================================
# Visualization
# ============================================================================

def visualize_results(
    cover: torch.Tensor,
    stego: torch.Tensor,
    save_path: str = None,
    num_images: int = 4,
):
    """
    Create a side-by-side comparison of cover and stego images.

    Displays cover images, stego images, and amplified difference maps
    to visualize the embedding distortion.

    Args:
        cover (Tensor): Cover images, shape (B, 3, H, W).
        stego (Tensor): Stego images, shape (B, 3, H, W).
        save_path (str, optional): Path to save the figure. If None, displays interactively.
        num_images (int): Number of images to display (from the batch).
    """
    num_images = min(num_images, cover.shape[0])

    fig, axes = plt.subplots(3, num_images, figsize=(4 * num_images, 12))

    for i in range(num_images):
        # Convert tensors to numpy for display
        cover_img = cover[i].detach().cpu().numpy().transpose(1, 2, 0)
        stego_img = stego[i].detach().cpu().numpy().transpose(1, 2, 0)

        # Amplified difference (×10 for visibility)
        diff = np.abs(cover_img - stego_img) * 10.0
        diff = np.clip(diff, 0.0, 1.0)

        # Compute per-image metrics
        psnr_val = peak_signal_noise_ratio(cover_img, stego_img, data_range=1.0)
        ssim_val = structural_similarity(cover_img, stego_img, data_range=1.0, channel_axis=2)

        # Plot
        axes[0, i].imshow(cover_img)
        axes[0, i].set_title("Cover", fontsize=10)
        axes[0, i].axis("off")

        axes[1, i].imshow(stego_img)
        axes[1, i].set_title(f"Stego\nPSNR={psnr_val:.1f}dB", fontsize=10)
        axes[1, i].axis("off")

        axes[2, i].imshow(diff)
        axes[2, i].set_title(f"Diff (×10)\nSSIM={ssim_val:.4f}", fontsize=10)
        axes[2, i].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Utils] Saved visualization to '{save_path}'")
    else:
        plt.show()

    plt.close(fig)


# ============================================================================
# Checkpoint Management
# ============================================================================

def save_checkpoint(
    encoder,
    decoder,
    optimizer,
    epoch: int,
    metrics: dict,
    save_path: str,
):
    """
    Save a training checkpoint.

    Args:
        encoder: Encoder network.
        decoder: Decoder network.
        optimizer: Training optimizer.
        epoch (int): Current epoch number.
        metrics (dict): Dictionary of current metrics.
        save_path (str): Path to save the checkpoint file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, save_path)
    print(f"[Checkpoint] Saved to '{save_path}' (epoch {epoch})")


def load_checkpoint(encoder, decoder, checkpoint_path: str, optimizer=None):
    """
    Load a training checkpoint.

    Args:
        encoder: Encoder network (will be updated in-place).
        decoder: Decoder network (will be updated in-place).
        checkpoint_path (str): Path to the checkpoint file.
        optimizer: Optional optimizer to restore state.

    Returns:
        encoder: Encoder with loaded weights.
        decoder: Decoder with loaded weights.
        epoch (int): The epoch the checkpoint was saved at.
        metrics (dict): The metrics at checkpoint time.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})

    print(f"[Checkpoint] Loaded from '{checkpoint_path}' (epoch {epoch})")
    return encoder, decoder, epoch, metrics
