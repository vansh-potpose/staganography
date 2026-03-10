"""
losses.py — Improved loss functions for steganography training.

Key improvements:
  1. BCEWithLogitsLoss instead of BCELoss (numerically stable)
  2. Confidence penalty to push predictions away from 0.5
  
Total loss = λ_image * L_image + λ_message * L_message + λ_confidence * L_confidence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import LAMBDA_IMAGE, LAMBDA_MESSAGE


class SSIMLoss(nn.Module):
    """
    Differentiable Structural Similarity Index (SSIM) loss.
    Loss = 1 - SSIM, so minimizing this maximizes structural similarity.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer("window", self._create_window(window_size, sigma))

    @staticmethod
    def _create_window(window_size: int, sigma: float) -> torch.Tensor:
        """Create a 2D Gaussian kernel."""
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_2d = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)
        return gauss_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        channels = img1.shape[1]
        window = self.window.expand(channels, -1, -1, -1).to(img1.device)
        padding = self.window_size // 2

        mu1 = F.conv2d(img1, window, padding=padding, groups=channels)
        mu2 = F.conv2d(img2, window, padding=padding, groups=channels)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 ** 2, window, padding=padding, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 ** 2, window, padding=padding, groups=channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channels) - mu12

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        numerator = (2 * mu12 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = numerator / denominator

        return 1.0 - ssim_map.mean()


class StegLoss(nn.Module):
    """
    Improved combined steganography loss function.

    Changes from original:
      - Uses BCEWithLogitsLoss (numerically stable, handles logit outputs)
      - Adds confidence penalty pushing predictions away from 0.5

    Args:
        lambda_image (float): Weight for image loss.
        lambda_message (float): Weight for message loss.
        lambda_confidence (float): Weight for confidence penalty.
    """

    def __init__(
        self,
        lambda_image: float = LAMBDA_IMAGE,
        lambda_message: float = LAMBDA_MESSAGE,
        lambda_confidence: float = 0.1,
    ):
        super().__init__()
        self.lambda_image = lambda_image
        self.lambda_message = lambda_message
        self.lambda_confidence = lambda_confidence

        # Sub-losses
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss()
        # BCEWithLogitsLoss: decoder outputs logits, sigmoid is applied internally
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        cover_image: torch.Tensor,
        stego_image: torch.Tensor,
        original_message: torch.Tensor,
        decoded_logits: torch.Tensor,
    ) -> dict:
        """
        Compute the combined loss.

        Args:
            cover_image: Original cover image, shape (B, 3, H, W).
            stego_image: Generated stego image, shape (B, 3, H, W).
            original_message: Original binary message, shape (B, L).
            decoded_logits: Decoder output LOGITS (not probabilities), shape (B, L).

        Returns:
            loss_dict with 'total', 'image_mse', 'image_ssim', 'image_total', 'message'.
        """
        # Image losses
        mse = self.mse_loss(stego_image, cover_image)
        ssim = self.ssim_loss(stego_image, cover_image)
        image_loss = mse + ssim

        # Message loss (BCEWithLogitsLoss applies sigmoid internally)
        message_loss = self.bce_loss(decoded_logits, original_message)

        # Confidence penalty: penalize logits near 0 (i.e., probabilities near 0.5)
        # This encourages the model to make confident predictions
        confidence_penalty = torch.mean(torch.exp(-decoded_logits.abs()))

        # Total weighted loss
        total_loss = (
            self.lambda_image * image_loss
            + self.lambda_message * message_loss
            + self.lambda_confidence * confidence_penalty
        )

        return {
            "total": total_loss,
            "image_mse": mse.item(),
            "image_ssim": ssim.item(),
            "image_total": (self.lambda_image * image_loss).item(),
            "message": (self.lambda_message * message_loss).item(),
        }
