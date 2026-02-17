"""
model.py — Encoder and Decoder networks for image steganography.

Architecture Overview:
  - Encoder: Takes a cover image (3×H×W) and a binary message (L bits),
    produces a stego image (3×H×W) with the message hidden inside.
  - Decoder: Takes a (possibly distorted) stego image and extracts
    the hidden binary message.

Both networks use convolutional architectures with batch normalization
and ReLU activations. The encoder uses residual learning (output = input + residual)
to ensure minimal visible modification to the cover image.
"""

import torch
import torch.nn as nn

from src.config import MESSAGE_LENGTH, IMAGE_CHANNELS


class ConvBNReLU(nn.Module):
    """
    A convenience building block: Conv2D → BatchNorm → ReLU.

    All convolutions use 3×3 kernels with padding=1 to maintain
    spatial dimensions throughout the network.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    """
    Steganographic Encoder Network.

    Takes a cover image and a binary message, and produces a stego image
    that visually resembles the cover but contains the hidden message.

    Architecture:
        1. Initial feature extraction from the cover image (3 Conv-BN-ReLU blocks).
        2. Message expansion: the 1D message vector is spatially replicated to
           match the image dimensions and concatenated with the feature maps.
        3. Post-fusion processing (2 Conv-BN-ReLU blocks + 1 Conv2D output layer).
        4. Residual addition: stego_image = cover_image + encoder_residual.

    Args:
        message_length (int): Length of the binary message in bits.
        hidden_channels (int): Number of feature channels in hidden layers.
    """

    def __init__(self, message_length: int = MESSAGE_LENGTH, hidden_channels: int = 64):
        super().__init__()
        self.message_length = message_length

        # Stage 1: Extract features from the cover image
        self.image_pre = nn.Sequential(
            ConvBNReLU(IMAGE_CHANNELS, hidden_channels),
            ConvBNReLU(hidden_channels, hidden_channels),
            ConvBNReLU(hidden_channels, hidden_channels),
        )

        # Stage 2: Process fused image features + message
        # Input channels = hidden_channels + message_length (after concatenation)
        self.image_post = nn.Sequential(
            ConvBNReLU(hidden_channels + message_length, hidden_channels),
            ConvBNReLU(hidden_channels, hidden_channels),
        )

        # Stage 3: Final convolution to produce 3-channel residual
        self.final_conv = nn.Conv2d(hidden_channels, IMAGE_CHANNELS, kernel_size=1)

    def forward(self, cover_image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            cover_image (Tensor): Cover image, shape (B, 3, H, W), values in [0, 1].
            message (Tensor): Binary message, shape (B, L), values in {0, 1}.

        Returns:
            stego_image (Tensor): Stego image with hidden message, shape (B, 3, H, W).
        """
        batch_size = cover_image.shape[0]
        height, width = cover_image.shape[2], cover_image.shape[3]

        # Extract image features
        image_features = self.image_pre(cover_image)  # (B, 64, H, W)

        # Expand message to spatial dimensions: (B, L) → (B, L, H, W)
        # Each bit is replicated across the entire spatial grid
        message_expanded = message.unsqueeze(-1).unsqueeze(-1)  # (B, L, 1, 1)
        message_expanded = message_expanded.expand(-1, -1, height, width)  # (B, L, H, W)

        # Concatenate image features and expanded message along channel dim
        fused = torch.cat([image_features, message_expanded], dim=1)  # (B, 64+L, H, W)

        # Process fused features
        post_features = self.image_post(fused)  # (B, 64, H, W)

        # Generate residual and add to cover image
        residual = self.final_conv(post_features)  # (B, 3, H, W)
        stego_image = cover_image + residual

        # Clamp to valid image range [0, 1]
        stego_image = torch.clamp(stego_image, 0.0, 1.0)

        return stego_image


class Decoder(nn.Module):
    """
    Steganographic Decoder Network.

    Takes a (possibly distorted) stego image and extracts the hidden binary message.

    Architecture:
        1. Five Conv-BN-ReLU blocks for deep feature extraction.
        2. Global Average Pooling to aggregate spatial features into a fixed vector.
        3. Fully connected layer mapping to message length.
        4. Sigmoid activation for per-bit probabilities.

    Args:
        message_length (int): Length of the binary message in bits.
        hidden_channels (int): Number of feature channels in hidden layers.
    """

    def __init__(self, message_length: int = MESSAGE_LENGTH, hidden_channels: int = 64):
        super().__init__()
        self.message_length = message_length

        # Feature extraction backbone (5 layers for sufficient capacity)
        self.features = nn.Sequential(
            ConvBNReLU(IMAGE_CHANNELS, hidden_channels),
            ConvBNReLU(hidden_channels, hidden_channels),
            ConvBNReLU(hidden_channels, hidden_channels),
            ConvBNReLU(hidden_channels, hidden_channels),
            ConvBNReLU(hidden_channels, hidden_channels),
        )

        # Global average pooling: (B, hidden_channels, H, W) → (B, hidden_channels)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected output: (B, hidden_channels) → (B, message_length)
        self.fc = nn.Linear(hidden_channels, message_length)

    def forward(self, stego_image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            stego_image (Tensor): Stego image (possibly distorted),
                                   shape (B, 3, H, W), values in [0, 1].

        Returns:
            decoded_message (Tensor): Per-bit probabilities,
                                       shape (B, L), values in [0, 1].
        """
        # Extract features
        features = self.features(stego_image)  # (B, 64, H, W)

        # Global average pooling
        pooled = self.global_pool(features)  # (B, 64, 1, 1)
        pooled = pooled.squeeze(-1).squeeze(-1)  # (B, 64)

        # Predict message bits
        decoded_message = torch.sigmoid(self.fc(pooled))  # (B, L)

        return decoded_message


# ============================================================================
# Utility: Create encoder-decoder pair
# ============================================================================
def create_model(message_length: int = MESSAGE_LENGTH, hidden_channels: int = 64):
    """
    Factory function to create an encoder-decoder pair.

    Args:
        message_length (int): Length of the hidden binary message.
        hidden_channels (int): Number of channels in hidden layers.

    Returns:
        encoder (Encoder): The encoder network.
        decoder (Decoder): The decoder network.
    """
    encoder = Encoder(message_length=message_length, hidden_channels=hidden_channels)
    decoder = Decoder(message_length=message_length, hidden_channels=hidden_channels)

    # Print model sizes for reference
    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"[Model] Encoder parameters: {enc_params:,}")
    print(f"[Model] Decoder parameters: {dec_params:,}")
    print(f"[Model] Total parameters:   {enc_params + dec_params:,}")

    return encoder, decoder
