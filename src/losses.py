"""
losses.py — Loss functions for steganography training.

The total loss combines two objectives:
  1. Image Loss: Ensures the stego image is visually similar to the cover image.
     - MSE loss for pixel-level fidelity.
     - SSIM loss for structural similarity preservation.
  2. Message Loss: Ensures the hidden message can be accurately recovered.
     - Binary Cross-Entropy (BCE) between original and decoded messages.

Total loss = λ_image * L_image + λ_message * L_message
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import LAMBDA_IMAGE, LAMBDA_MESSAGE


class SSIMLoss(nn.Module):
    """
    Differentiable Structural Similarity Index (SSIM) loss.

    SSIM measures the perceived quality of an image by comparing luminance,
    contrast, and structure. The loss is defined as (1 - SSIM), so minimizing
    this loss maximizes structural similarity.

    Uses a Gaussian window for local statistics computation.

    Args:
        window_size (int): Size of the Gaussian window (must be odd).
        sigma (float): Standard deviation of the Gaussian window.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma

        # Pre-compute Gaussian window
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
        """
        Compute SSIM loss between two images.

        Args:
            img1, img2 (Tensor): Images of shape (B, C, H, W), values in [0, 1].

        Returns:
            loss (Tensor): Scalar SSIM loss = 1 - mean(SSIM).
        """
        channels = img1.shape[1]

        # Expand window to all channels
        window = self.window.expand(channels, -1, -1, -1).to(img1.device)
        padding = self.window_size // 2

        # Compute local means
        mu1 = F.conv2d(img1, window, padding=padding, groups=channels)
        mu2 = F.conv2d(img2, window, padding=padding, groups=channels)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2

        # Compute local variances and covariance
        sigma1_sq = F.conv2d(img1 ** 2, window, padding=padding, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 ** 2, window, padding=padding, groups=channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channels) - mu12

        # Stability constants
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # SSIM formula
        numerator = (2 * mu12 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = numerator / denominator

        # Return loss (1 - SSIM, averaged over all pixels and channels)
        return 1.0 - ssim_map.mean()


class StegLoss(nn.Module):
    """
    Combined steganography loss function.

    Balances two objectives:
      - Image fidelity: MSE + SSIM loss between cover and stego images.
      - Message recovery: BCE loss between original and decoded messages.

    Args:
        lambda_image (float): Weight for the image loss term.
        lambda_message (float): Weight for the message loss term.
    """

    def __init__(
        self,
        lambda_image: float = LAMBDA_IMAGE,
        lambda_message: float = LAMBDA_MESSAGE,
    ):
        super().__init__()
        self.lambda_image = lambda_image
        self.lambda_message = lambda_message

        # Sub-losses
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss()
        self.bce_loss = nn.BCELoss()

    def forward(
        self,
        cover_image: torch.Tensor,
        stego_image: torch.Tensor,
        original_message: torch.Tensor,
        decoded_message: torch.Tensor,
    ) -> dict:
        """
        Compute the combined loss.

        Args:
            cover_image (Tensor): Original cover image, shape (B, 3, H, W).
            stego_image (Tensor): Generated stego image, shape (B, 3, H, W).
            original_message (Tensor): Original binary message, shape (B, L).
            decoded_message (Tensor): Decoded message probabilities, shape (B, L).

        Returns:
            loss_dict (dict): Dictionary containing:
                - 'total': Combined total loss (scalar).
                - 'image_mse': MSE component of image loss.
                - 'image_ssim': SSIM component of image loss.
                - 'image_total': Weighted total image loss.
                - 'message': Weighted message loss.
        """
        # Image losses
        mse = self.mse_loss(stego_image, cover_image)
        ssim = self.ssim_loss(stego_image, cover_image)
        image_loss = mse + ssim  # Combined pixel + structural loss

        # Message loss
        message_loss = self.bce_loss(decoded_message, original_message)

        # Total weighted loss
        total_loss = self.lambda_image * image_loss + self.lambda_message * message_loss

        return {
            "total": total_loss,
            "image_mse": mse.item(),
            "image_ssim": ssim.item(),
            "image_total": (self.lambda_image * image_loss).item(),
            "message": (self.lambda_message * message_loss).item(),
        }
