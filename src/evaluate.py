"""
evaluate.py — Evaluation and robustness testing for the steganography system.

This module provides comprehensive evaluation capabilities:
  1. Baseline evaluation: PSNR, SSIM, and bit accuracy without attacks.
  2. Individual attack evaluation: Test robustness against each attack type.
  3. Result visualization: Generate comparison images and metric tables.

Usage:
  from src.evaluate import evaluate_robustness
  results = evaluate_robustness(encoder, decoder, test_loader, device)
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.config import DEVICE, RESULTS_DIR
from src.noise_layers import (
    Identity, JpegCompression, GaussianNoise, Cropout, GaussianBlur,
)
from src.utils import compute_psnr, compute_ssim, compute_bit_accuracy, visualize_results


# ============================================================================
# Attack Configurations for Testing
# ============================================================================

ATTACK_CONFIGS = {
    "no_attack": {
        "layer_class": Identity,
        "params": {},
        "description": "No attack (baseline)",
    },
    "jpeg_q90": {
        "layer_class": JpegCompression,
        "params": {"quality_min": 90, "quality_max": 90},
        "description": "JPEG Compression (QF=90)",
    },
    "jpeg_q70": {
        "layer_class": JpegCompression,
        "params": {"quality_min": 70, "quality_max": 70},
        "description": "JPEG Compression (QF=70)",
    },
    "jpeg_q50": {
        "layer_class": JpegCompression,
        "params": {"quality_min": 50, "quality_max": 50},
        "description": "JPEG Compression (QF=50)",
    },
    "noise_001": {
        "layer_class": GaussianNoise,
        "params": {"sigma_min": 0.01, "sigma_max": 0.01},
        "description": "Gaussian Noise (σ=0.01)",
    },
    "noise_003": {
        "layer_class": GaussianNoise,
        "params": {"sigma_min": 0.03, "sigma_max": 0.03},
        "description": "Gaussian Noise (σ=0.03)",
    },
    "noise_005": {
        "layer_class": GaussianNoise,
        "params": {"sigma_min": 0.05, "sigma_max": 0.05},
        "description": "Gaussian Noise (σ=0.05)",
    },
    "cropout_10": {
        "layer_class": Cropout,
        "params": {"cropout_min": 0.10, "cropout_max": 0.10},
        "description": "Cropout (10% area)",
    },
    "cropout_20": {
        "layer_class": Cropout,
        "params": {"cropout_min": 0.20, "cropout_max": 0.20},
        "description": "Cropout (20% area)",
    },
    "cropout_30": {
        "layer_class": Cropout,
        "params": {"cropout_min": 0.30, "cropout_max": 0.30},
        "description": "Cropout (30% area)",
    },
    "blur_k3": {
        "layer_class": GaussianBlur,
        "params": {"kernel_sizes": [3], "sigma_min": 1.0, "sigma_max": 1.0},
        "description": "Gaussian Blur (kernel=3)",
    },
    "blur_k5": {
        "layer_class": GaussianBlur,
        "params": {"kernel_sizes": [5], "sigma_min": 1.5, "sigma_max": 1.5},
        "description": "Gaussian Blur (kernel=5)",
    },
    "blur_k7": {
        "layer_class": GaussianBlur,
        "params": {"kernel_sizes": [7], "sigma_min": 2.0, "sigma_max": 2.0},
        "description": "Gaussian Blur (kernel=7)",
    },
}


@torch.no_grad()
def evaluate_single_attack(
    encoder,
    decoder,
    test_loader,
    attack_layer,
    device: torch.device = DEVICE,
    needs_cover: bool = False,
):
    """
    Evaluate the model under a single attack type.

    Args:
        encoder: Trained encoder network.
        decoder: Trained decoder network.
        test_loader: DataLoader with (cover_image, message) pairs.
        attack_layer: Distortion layer to apply to stego images.
        device: Device to run evaluation on.
        needs_cover: Whether the attack needs the cover image (e.g., Cropout).

    Returns:
        metrics (dict): Dictionary with 'psnr', 'ssim', 'bit_accuracy' averages.
    """
    encoder.eval()
    decoder.eval()
    attack_layer.train()  # Enable randomness in noise layers

    psnr_list, ssim_list, bit_acc_list = [], [], []

    for cover_images, messages in test_loader:
        cover_images = cover_images.to(device)
        messages = messages.to(device)

        # Generate stego images
        stego_images = encoder(cover_images, messages)

        # Compute image quality metrics (before attack)
        psnr = compute_psnr(cover_images, stego_images)
        ssim = compute_ssim(cover_images, stego_images)

        # Apply attack
        if needs_cover:
            attacked_images = attack_layer(stego_images, cover_images, noise_strength=1.0)
        else:
            attacked_images = attack_layer(stego_images, noise_strength=1.0)

        # Decode message from attacked image (decoder outputs logits)
        decoded_logits = decoder(attacked_images)
        decoded_probs = torch.sigmoid(decoded_logits)

        # Compute bit accuracy (after attack)
        bit_acc = compute_bit_accuracy(messages, decoded_probs)

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        bit_acc_list.append(bit_acc)

    return {
        "psnr": np.mean(psnr_list),
        "ssim": np.mean(ssim_list),
        "bit_accuracy": np.mean(bit_acc_list),
    }


def evaluate_robustness(
    encoder,
    decoder,
    test_loader,
    device: torch.device = DEVICE,
    output_dir: str = RESULTS_DIR,
):
    """
    Run comprehensive robustness evaluation against all attack types.

    Tests the model under each configured attack and generates:
      - A summary table printed to the console.
      - A results dictionary for further analysis.
      - A bar chart visualization of bit accuracy across attacks.

    Args:
        encoder: Trained encoder.
        decoder: Trained decoder.
        test_loader: Test DataLoader.
        device: Computation device.
        output_dir: Directory to save results.

    Returns:
        all_results (dict): Attack name → metrics dictionary.
    """
    print("\n" + "=" * 70)
    print("  ROBUSTNESS EVALUATION")
    print("=" * 70)

    all_results = {}

    for attack_name, config in tqdm(ATTACK_CONFIGS.items(), desc="Attacks"):
        # Create the attack layer
        attack_layer = config["layer_class"](**config["params"]).to(device)
        needs_cover = config["layer_class"] == Cropout

        # Run evaluation
        metrics = evaluate_single_attack(
            encoder, decoder, test_loader, attack_layer, device, needs_cover,
        )
        all_results[attack_name] = {**metrics, "description": config["description"]}

    # ---- Print Results Table ----
    print("\n" + "-" * 70)
    print(f"{'Attack':<30} {'PSNR (dB)':>10} {'SSIM':>8} {'Bit Acc (%)':>12}")
    print("-" * 70)

    for attack_name, metrics in all_results.items():
        desc = metrics["description"]
        print(f"{desc:<30} {metrics['psnr']:>10.2f} {metrics['ssim']:>8.4f} "
              f"{metrics['bit_accuracy'] * 100:>11.2f}%")

    print("-" * 70)

    # ---- Generate Visualization ----
    _plot_robustness_chart(all_results, output_dir)

    # ---- Save Sample Images ----
    _save_sample_comparisons(encoder, decoder, test_loader, device, output_dir)

    return all_results


def _plot_robustness_chart(all_results: dict, output_dir: str):
    """
    Create a bar chart showing bit accuracy for each attack type.

    Args:
        all_results (dict): Results from evaluate_robustness.
        output_dir (str): Directory to save the chart.
    """
    descriptions = [r["description"] for r in all_results.values()]
    bit_accuracies = [r["bit_accuracy"] * 100 for r in all_results.values()]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Color bars based on accuracy thresholds
    colors = []
    for acc in bit_accuracies:
        if acc >= 95:
            colors.append("#2ecc71")   # Green: excellent
        elif acc >= 90:
            colors.append("#f1c40f")   # Yellow: good
        elif acc >= 80:
            colors.append("#e67e22")   # Orange: acceptable
        else:
            colors.append("#e74c3c")   # Red: poor

    bars = ax.barh(descriptions, bit_accuracies, color=colors, edgecolor="white", height=0.6)

    # Add value labels on bars
    for bar, acc in zip(bars, bit_accuracies):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Bit Accuracy (%)", fontsize=12)
    ax.set_title("Message Recovery Accuracy Under Various Attacks", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.axvline(x=90, color="gray", linestyle="--", alpha=0.5, label="90% threshold")
    ax.legend(fontsize=10)
    ax.invert_yaxis()

    plt.tight_layout()

    chart_path = os.path.join(output_dir, "robustness_chart.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"\n[Evaluation] Saved robustness chart to '{chart_path}'")
    plt.close(fig)


def _save_sample_comparisons(encoder, decoder, test_loader, device, output_dir):
    """
    Save sample cover vs. stego image comparisons.

    Args:
        encoder: Trained encoder.
        decoder: Trained decoder.
        test_loader: Test DataLoader.
        device: Computation device.
        output_dir: Directory to save comparisons.
    """
    # Get a single batch
    cover_images, messages = next(iter(test_loader))
    cover_images = cover_images.to(device)
    messages = messages.to(device)

    with torch.no_grad():
        stego_images = encoder(cover_images, messages)

    comparison_path = os.path.join(output_dir, "sample_comparisons.png")
    visualize_results(cover_images, stego_images, save_path=comparison_path, num_images=4)
