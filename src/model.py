"""
model.py — Improved Encoder and Decoder networks for image steganography.

Key improvements over the original:
  - Encoder: Message pre-processing MLP, deeper post-fusion, smooth residual scaling
  - Decoder: Multi-scale feature extraction with downsampling, channel attention (SE),
    dual pooling (avg+max), deeper FC head
  - No sigmoid in decoder (moved to loss function for numerical stability)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import MESSAGE_LENGTH, IMAGE_CHANNELS


class ConvBNReLU(nn.Module):
    """Conv2D → BatchNorm → ReLU with optional stride for downsampling."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # Squeeze: global average pooling
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        # Excitation: FC → sigmoid
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResBlock(nn.Module):
    """Residual block with two Conv-BN-ReLU layers and optional SE attention."""

    def __init__(self, channels: int, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels) if use_se else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + residual)


class Encoder(nn.Module):
    """
    Improved Steganographic Encoder.

    Key improvements:
      1. Message pre-processing MLP for richer bit representations
      2. Deeper post-fusion with residual blocks
      3. Smooth residual scaling (tanh * scale_factor) instead of raw clamp
    """

    def __init__(self, message_length: int = MESSAGE_LENGTH, hidden_channels: int = 128):
        super().__init__()
        self.message_length = message_length

        # Message pre-processing: transform bits into richer representations
        self.message_mlp = nn.Sequential(
            nn.Linear(message_length, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Stage 1: Extract features from the cover image
        self.image_pre = nn.Sequential(
            ConvBNReLU(IMAGE_CHANNELS, hidden_channels),
            ConvBNReLU(hidden_channels, hidden_channels),
            ResBlock(hidden_channels, use_se=True),
        )

        # Stage 2: Process fused image features + processed message
        # Input channels = hidden_channels (image) + hidden_channels (message)
        self.image_post = nn.Sequential(
            ConvBNReLU(hidden_channels * 2, hidden_channels),
            ResBlock(hidden_channels, use_se=True),
            ResBlock(hidden_channels, use_se=True),
        )

        # Stage 3: Generate 3-channel residual with tanh for bounded output
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, IMAGE_CHANNELS, kernel_size=1),
            nn.Tanh(),
        )

        # Learnable residual scaling factor (starts small)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, cover_image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        batch_size = cover_image.shape[0]
        height, width = cover_image.shape[2], cover_image.shape[3]

        # Extract image features
        image_features = self.image_pre(cover_image)  # (B, C, H, W)

        # Pre-process message through MLP, then expand spatially
        message_processed = self.message_mlp(message)  # (B, C)
        message_expanded = message_processed.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        message_expanded = message_expanded.expand(-1, -1, height, width)  # (B, C, H, W)

        # Concatenate image features and processed message
        fused = torch.cat([image_features, message_expanded], dim=1)  # (B, 2C, H, W)

        # Process fused features
        post_features = self.image_post(fused)  # (B, C, H, W)

        # Generate bounded residual and add to cover image
        residual = self.final_conv(post_features)  # (B, 3, H, W), range [-1, 1]
        stego_image = cover_image + self.scale * residual

        # Clamp to valid range (soft via tanh scaling, so rarely triggered)
        stego_image = torch.clamp(stego_image, 0.0, 1.0)

        return stego_image


class Decoder(nn.Module):
    """
    Improved Steganographic Decoder.

    Key improvements:
      1. Multi-scale feature extraction with downsampling
      2. SE channel attention at each scale
      3. Dual pooling (average + max) for richer feature vectors
      4. Deeper FC head with dropout
      5. No sigmoid — outputs raw logits (sigmoid applied in loss)
    """

    def __init__(self, message_length: int = MESSAGE_LENGTH, hidden_channels: int = 128):
        super().__init__()
        self.message_length = message_length

        # Scale 1: Full resolution feature extraction
        self.scale1 = nn.Sequential(
            ConvBNReLU(IMAGE_CHANNELS, hidden_channels),
            ResBlock(hidden_channels, use_se=True),
        )

        # Scale 2: Downsample by 2x
        self.scale2 = nn.Sequential(
            ConvBNReLU(hidden_channels, hidden_channels * 2, stride=2),
            ResBlock(hidden_channels * 2, use_se=True),
        )

        # Scale 3: Downsample by another 2x
        self.scale3 = nn.Sequential(
            ConvBNReLU(hidden_channels * 2, hidden_channels * 2, stride=2),
            ResBlock(hidden_channels * 2, use_se=True),
        )

        # Feature dimensions after dual pooling at each scale:
        # Scale1: hidden_channels * 2 (avg + max pooling)
        # Scale2: hidden_channels * 2 * 2
        # Scale3: hidden_channels * 2 * 2
        fc_input_size = hidden_channels * 2 + hidden_channels * 2 * 2 + hidden_channels * 2 * 2
        # = hidden_channels * 10

        # FC head for message prediction (outputs logits, no sigmoid)
        self.fc_head = nn.Sequential(
            nn.Linear(fc_input_size, hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels * 4, hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels * 2, message_length),
        )

    def _dual_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Apply both average and max global pooling, concatenate results."""
        avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
        mx = F.adaptive_max_pool2d(x, 1).flatten(1)
        return torch.cat([avg, mx], dim=1)

    def forward(self, stego_image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Returns raw logits (NOT probabilities).
        Apply sigmoid externally for inference, or use BCEWithLogitsLoss for training.
        """
        # Multi-scale feature extraction
        f1 = self.scale1(stego_image)   # (B, C, H, W)
        f2 = self.scale2(f1)            # (B, 2C, H/2, W/2)
        f3 = self.scale3(f2)            # (B, 2C, H/4, W/4)

        # Dual pooling at each scale
        p1 = self._dual_pool(f1)  # (B, 2C)
        p2 = self._dual_pool(f2)  # (B, 4C)
        p3 = self._dual_pool(f3)  # (B, 4C)

        # Concatenate multi-scale features
        combined = torch.cat([p1, p2, p3], dim=1)  # (B, 10C)

        # Predict message bits (logits)
        logits = self.fc_head(combined)  # (B, message_length)

        return logits


# ============================================================================
# Utility: Create encoder-decoder pair
# ============================================================================
def create_model(message_length: int = MESSAGE_LENGTH, hidden_channels: int = 128):
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
