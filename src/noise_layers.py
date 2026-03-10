"""
noise_layers.py — Differentiable distortion layers for robust steganography training.

Improvements:
  - Added noise_strength parameter for progressive noise ramping
  - Gentler default noise intensities
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

    def forward(self, stego_image: torch.Tensor, cover_image: torch.Tensor = None,
                noise_strength: float = 1.0) -> torch.Tensor:
        return stego_image


class JpegCompression(nn.Module):
    """
    Differentiable JPEG compression simulation.

    Approximates JPEG artifacts by adding scaled noise and optional blur.
    The strength is controlled by the quality factor.
    """

    def __init__(self, quality_min: int = JPEG_QUALITY_MIN, quality_max: int = JPEG_QUALITY_MAX):
        super().__init__()
        self.quality_min = quality_min
        self.quality_max = quality_max

    def forward(self, stego_image: torch.Tensor, cover_image: torch.Tensor = None,
                noise_strength: float = 1.0) -> torch.Tensor:
        if not self.training:
            return stego_image

        # Interpolate quality based on noise_strength (higher strength = lower quality)
        effective_min = int(self.quality_max - noise_strength * (self.quality_max - self.quality_min))
        quality = random.randint(effective_min, self.quality_max)

        noise_scale = (100 - quality) / 100.0 * 0.1
        noise = torch.randn_like(stego_image) * noise_scale
        noised = stego_image + noise

        if quality < 80:
            kernel_size = 3
            sigma = (100 - quality) / 100.0
            noised = self._gaussian_blur(noised, kernel_size, sigma)

        return torch.clamp(noised, 0.0, 1.0)

    @staticmethod
    def _gaussian_blur(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        channels = x.shape[1]
        coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device)
        coords -= kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (coords / max(sigma, 1e-6)) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.expand(channels, 1, -1, -1)
        padding = kernel_size // 2
        return F.conv2d(x, kernel_2d, padding=padding, groups=channels)


class GaussianNoise(nn.Module):
    """Additive Gaussian noise with strength scaling."""

    def __init__(self, sigma_min: float = GAUSSIAN_NOISE_MIN, sigma_max: float = GAUSSIAN_NOISE_MAX):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def forward(self, stego_image: torch.Tensor, cover_image: torch.Tensor = None,
                noise_strength: float = 1.0) -> torch.Tensor:
        if not self.training:
            return stego_image

        # Scale sigma range by noise_strength
        effective_max = self.sigma_min + noise_strength * (self.sigma_max - self.sigma_min)
        sigma = random.uniform(self.sigma_min, effective_max)

        noise = torch.randn_like(stego_image) * sigma
        noised = stego_image + noise
        return torch.clamp(noised, 0.0, 1.0)


class Cropout(nn.Module):
    """Cropout distortion with strength scaling."""

    def __init__(self, cropout_min: float = CROPOUT_MIN_RATIO, cropout_max: float = CROPOUT_MAX_RATIO):
        super().__init__()
        self.cropout_min = cropout_min
        self.cropout_max = cropout_max

    def forward(self, stego_image: torch.Tensor, cover_image: torch.Tensor = None,
                noise_strength: float = 1.0) -> torch.Tensor:
        if not self.training or cover_image is None:
            return stego_image

        batch_size, channels, height, width = stego_image.shape

        # Scale crop ratio by noise_strength
        effective_max = self.cropout_min + noise_strength * (self.cropout_max - self.cropout_min)
        crop_ratio = random.uniform(self.cropout_min, effective_max)
        crop_height = int(height * math.sqrt(crop_ratio))
        crop_width = int(width * math.sqrt(crop_ratio))

        crop_height = max(1, min(crop_height, height))
        crop_width = max(1, min(crop_width, width))

        top = random.randint(0, height - crop_height)
        left = random.randint(0, width - crop_width)

        mask = torch.ones_like(stego_image)
        mask[:, :, top:top + crop_height, left:left + crop_width] = 0.0

        result = stego_image * mask + cover_image * (1.0 - mask)
        return result


class GaussianBlur(nn.Module):
    """Gaussian blur distortion with strength scaling."""

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

    def forward(self, stego_image: torch.Tensor, cover_image: torch.Tensor = None,
                noise_strength: float = 1.0) -> torch.Tensor:
        if not self.training:
            return stego_image

        # Scale sigma by noise_strength
        effective_max = self.sigma_min + noise_strength * (self.sigma_max - self.sigma_min)
        kernel_size = random.choice(self.kernel_sizes)
        sigma = random.uniform(self.sigma_min, effective_max)

        return self._apply_blur(stego_image, kernel_size, sigma)

    @staticmethod
    def _apply_blur(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        channels = x.shape[1]
        coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device)
        coords -= kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (coords / max(sigma, 1e-6)) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.expand(channels, 1, -1, -1)
        padding = kernel_size // 2
        return F.conv2d(x, kernel_2d, padding=padding, groups=channels)


class CombinedNoiseLayer(nn.Module):
    """
    Combined noise layer with progressive noise ramping.

    Uses a noise_strength parameter (0.0 to 1.0) that controls:
      - The probability of applying any noise
      - The intensity of each noise type

    This enables smooth progressive noise increase during training.
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

        self.layer_names = ["identity", "jpeg", "gaussian_noise", "cropout", "gaussian_blur"]

    def forward(self, stego_image: torch.Tensor, cover_image: torch.Tensor = None,
                noise_strength: float = 1.0):
        """
        Apply a random distortion during training with controlled strength.

        Args:
            stego_image: Encoder output, shape (B, 3, H, W).
            cover_image: Original cover image (needed for Cropout).
            noise_strength: Float from 0.0 to 1.0 controlling noise intensity.

        Returns:
            noised_image: Distorted stego image.
            noise_name: Name of the applied distortion (for logging).
        """
        # Scale the probability of applying noise by noise_strength
        effective_probability = self.noise_probability * noise_strength

        if not self.training or random.random() > effective_probability:
            return stego_image, "identity"

        # Randomly select a distortion (excluding Identity at index 0)
        idx = random.randint(1, len(self.noise_layers) - 1)
        noise_layer = self.noise_layers[idx]
        noise_name = self.layer_names[idx]

        noised_image = noise_layer(stego_image, cover_image, noise_strength=noise_strength)

        return noised_image, noise_name
