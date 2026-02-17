"""
noise_layers.py — Differentiable distortion layers for robust steganography training.

This module implements various image distortion layers that are inserted between
the encoder and decoder during training. Each distortion simulates a real-world
image processing operation, forcing the encoder to embed messages that survive
these attacks.

Supported Distortions:
  1. Identity (no distortion)
  2. JPEG compression (differentiable approximation)
  3. Gaussian noise
  4. Cropout (replace random region with cover image content)
  5. Gaussian blur

All layers are designed to be differentiable (or use straight-through estimators)
so that gradients can flow back through the noise layer to the encoder.
"""

import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    JPEG_QUALITY_MIN, JPEG_QUALITY_MAX,
    GAUSSIAN_NOISE_MIN, GAUSSIAN_NOISE_MAX,
    CROPOUT_MIN_RATIO, CROPOUT_MAX_RATIO,
    BLUR_KERNEL_SIZES, BLUR_SIGMA_MIN, BLUR_SIGMA_MAX,
    NOISE_PROBABILITY,
)


class Identity(nn.Module):
    """No distortion — passes the stego image through unchanged."""

    def forward(self, stego_image: torch.Tensor, cover_image: torch.Tensor = None) -> torch.Tensor:
        return stego_image


class JpegCompression(nn.Module):
    """
    Differentiable JPEG compression simulation.

    Since real JPEG compression involves non-differentiable quantization,
    we approximate it by:
      1. Adding scaled uniform noise to simulate quantization artifacts.
      2. Applying a low-pass filter to simulate DCT coefficient truncation.

    The strength of the distortion is controlled by the quality factor (QF).
    Lower QF = more aggressive compression = more distortion.

    Args:
        quality_min (int): Minimum JPEG quality factor.
        quality_max (int): Maximum JPEG quality factor.
    """

    def __init__(self, quality_min: int = JPEG_QUALITY_MIN, quality_max: int = JPEG_QUALITY_MAX):
        super().__init__()
        self.quality_min = quality_min
        self.quality_max = quality_max

    def forward(self, stego_image: torch.Tensor, cover_image: torch.Tensor = None) -> torch.Tensor:
        """
        Apply simulated JPEG compression.

        Args:
            stego_image (Tensor): Input stego image, shape (B, 3, H, W).
            cover_image (Tensor): Unused (kept for consistent interface).

        Returns:
            Distorted image tensor.
        """
        if not self.training:
            return stego_image

        # Randomly sample a quality factor
        quality = random.randint(self.quality_min, self.quality_max)

        # Compression strength: lower quality → higher noise scale
        noise_scale = (100 - quality) / 100.0 * 0.1

        # Add quantization noise (straight-through estimator for gradients)
        noise = torch.randn_like(stego_image) * noise_scale
        noised = stego_image + noise

        # Simulate frequency truncation with a small Gaussian blur
        if quality < 80:
            kernel_size = 3
            sigma = (100 - quality) / 100.0
            noised = self._gaussian_blur(noised, kernel_size, sigma)

        return torch.clamp(noised, 0.0, 1.0)

    @staticmethod
    def _gaussian_blur(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        """Apply Gaussian blur using a separable convolution."""
        channels = x.shape[1]

        # Create 1D Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device)
        coords -= kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Create 2D kernel via outer product
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)  # (K, K)
        kernel_2d = kernel_2d.expand(channels, 1, -1, -1)  # (C, 1, K, K)

        padding = kernel_size // 2
        return F.conv2d(x, kernel_2d, padding=padding, groups=channels)


class GaussianNoise(nn.Module):
    """
    Additive Gaussian noise distortion.

    Adds zero-mean Gaussian noise with a randomly sampled standard deviation.
    Fully differentiable via the reparameterization trick.

    Args:
        sigma_min (float): Minimum noise standard deviation.
        sigma_max (float): Maximum noise standard deviation.
    """

    def __init__(self, sigma_min: float = GAUSSIAN_NOISE_MIN, sigma_max: float = GAUSSIAN_NOISE_MAX):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def forward(self, stego_image: torch.Tensor, cover_image: torch.Tensor = None) -> torch.Tensor:
        if not self.training:
            return stego_image

        # Randomly sample noise standard deviation
        sigma = random.uniform(self.sigma_min, self.sigma_max)

        # Add Gaussian noise
        noise = torch.randn_like(stego_image) * sigma
        noised = stego_image + noise

        return torch.clamp(noised, 0.0, 1.0)


class Cropout(nn.Module):
    """
    Cropout distortion layer.

    Randomly selects a rectangular region and replaces it with the corresponding
    region from the cover image. This simulates partial image loss/replacement.

    The crop area ratio is randomly sampled between cropout_min and cropout_max.
    This layer is differentiable because it uses a binary mask.

    Args:
        cropout_min (float): Minimum fraction of area to crop out.
        cropout_max (float): Maximum fraction of area to crop out.
    """

    def __init__(self, cropout_min: float = CROPOUT_MIN_RATIO, cropout_max: float = CROPOUT_MAX_RATIO):
        super().__init__()
        self.cropout_min = cropout_min
        self.cropout_max = cropout_max

    def forward(self, stego_image: torch.Tensor, cover_image: torch.Tensor = None) -> torch.Tensor:
        if not self.training or cover_image is None:
            return stego_image

        batch_size, channels, height, width = stego_image.shape

        # Randomly determine crop area
        crop_ratio = random.uniform(self.cropout_min, self.cropout_max)
        crop_height = int(height * math.sqrt(crop_ratio))
        crop_width = int(width * math.sqrt(crop_ratio))

        # Ensure minimum size
        crop_height = max(1, min(crop_height, height))
        crop_width = max(1, min(crop_width, width))

        # Random position for the crop rectangle
        top = random.randint(0, height - crop_height)
        left = random.randint(0, width - crop_width)

        # Create a binary mask (1 = keep stego, 0 = replace with cover)
        mask = torch.ones_like(stego_image)
        mask[:, :, top:top + crop_height, left:left + crop_width] = 0.0

        # Apply mask: keep stego outside crop, use cover inside crop
        result = stego_image * mask + cover_image * (1.0 - mask)

        return result


class GaussianBlur(nn.Module):
    """
    Gaussian blur distortion layer.

    Applies a Gaussian blur filter with randomly sampled kernel size and sigma.
    Implemented as a depthwise convolution, which is fully differentiable.

    Args:
        kernel_sizes (list): List of possible kernel sizes (must be odd).
        sigma_min (float): Minimum blur sigma.
        sigma_max (float): Maximum blur sigma.
    """

    def __init__(
        self,
        kernel_sizes: list = None,
        sigma_min: float = BLUR_SIGMA_MIN,
        sigma_max: float = BLUR_SIGMA_MAX,
    ):
        super().__init__()
        self.kernel_sizes = kernel_sizes or BLUR_KERNEL_SIZES
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def forward(self, stego_image: torch.Tensor, cover_image: torch.Tensor = None) -> torch.Tensor:
        if not self.training:
            return stego_image

        # Randomly sample blur parameters
        kernel_size = random.choice(self.kernel_sizes)
        sigma = random.uniform(self.sigma_min, self.sigma_max)

        return self._apply_blur(stego_image, kernel_size, sigma)

    @staticmethod
    def _apply_blur(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        """Apply Gaussian blur using depthwise separable convolution."""
        channels = x.shape[1]

        # Create 1D Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device)
        coords -= kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (coords / max(sigma, 1e-6)) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Create 2D kernel
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.expand(channels, 1, -1, -1)

        padding = kernel_size // 2
        return F.conv2d(x, kernel_2d, padding=padding, groups=channels)


class CombinedNoiseLayer(nn.Module):
    """
    Combined noise layer that randomly selects one distortion per forward pass.

    During training, with probability `noise_probability`, a random distortion
    is applied to the stego image. Otherwise, the identity is applied.

    This randomized schedule forces the encoder to embed robustly against
    all supported attack types simultaneously.

    Args:
        noise_probability (float): Probability of applying a distortion (vs. identity).
    """

    def __init__(self, noise_probability: float = NOISE_PROBABILITY):
        super().__init__()
        self.noise_probability = noise_probability

        # Register all distortion layers
        self.noise_layers = nn.ModuleList([
            Identity(),
            JpegCompression(),
            GaussianNoise(),
            Cropout(),
            GaussianBlur(),
        ])

        # Layer names for logging
        self.layer_names = ["identity", "jpeg", "gaussian_noise", "cropout", "gaussian_blur"]

    def forward(self, stego_image: torch.Tensor, cover_image: torch.Tensor = None):
        """
        Apply a random distortion during training.

        Args:
            stego_image (Tensor): Encoder output, shape (B, 3, H, W).
            cover_image (Tensor): Original cover image (needed for Cropout).

        Returns:
            noised_image (Tensor): Distorted stego image.
            noise_name (str): Name of the applied distortion (for logging).
        """
        if not self.training or random.random() > self.noise_probability:
            return stego_image, "identity"

        # Randomly select a distortion (excluding Identity at index 0)
        idx = random.randint(1, len(self.noise_layers) - 1)
        noise_layer = self.noise_layers[idx]
        noise_name = self.layer_names[idx]

        noised_image = noise_layer(stego_image, cover_image)

        return noised_image, noise_name
