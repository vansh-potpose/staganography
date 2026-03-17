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

    def __init__(self, message_length: int = MESSAGE_LENGTH, hidden_channels: int = 128):
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
    Steganographic Decoder Network (Improved Architecture).

    Takes a (possibly distorted) stego image and extracts the hidden binary message.
    
    This improved decoder uses:
      1. Higher capacity (128 hidden channels instead of 64)
      2. Multi-scale feature aggregation with skip connections
      3. Spatial attention to focus on message-bearing regions
      4. Better feature representation before message prediction

    Args:
        message_length (int): Length of the binary message in bits.
        hidden_channels (int): Number of feature channels in hidden layers.
    """

    def __init__(self, message_length: int = MESSAGE_LENGTH, hidden_channels: int = 128):
        super().__init__()
        self.message_length = message_length

        # Stage 1: Initial feature extraction
        self.conv1 = ConvBNReLU(IMAGE_CHANNELS, hidden_channels)
        
        # Stage 2: Deeper feature extraction
        self.conv2 = ConvBNReLU(hidden_channels, hidden_channels)
        self.conv3 = ConvBNReLU(hidden_channels, hidden_channels)
        
        # Stage 3: Very deep feature extraction
        self.conv4 = ConvBNReLU(hidden_channels, hidden_channels * 2)
        self.conv5 = ConvBNReLU(hidden_channels * 2, hidden_channels * 2)
        
        # Spatial attention: learns which regions are important
        self.attention = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global pooling + FC layers for message prediction
        # Input features: attention-weighted high features (hidden_channels*2) + skip features (hidden_channels)
        fc_input_size = hidden_channels * 3
        
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, message_length)
        )

    def forward(self, stego_image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with improved feature extraction and spatial awareness.

        Args:
            stego_image (Tensor): Stego image, shape (B, 3, H, W), values in [0, 1].

        Returns:
            decoded_message (Tensor): Per-bit probabilities, shape (B, L).
        """
        batch_size = stego_image.shape[0]
        
        # Extract multi-level features
        f1 = self.conv1(stego_image)  # (B, hidden_channels, H, W)
        f2 = self.conv2(f1)            # (B, hidden_channels, H, W)
        f3 = self.conv3(f2)            # (B, hidden_channels, H, W)
        f4 = self.conv4(f3)            # (B, hidden_channels*2, H, W)
        f5 = self.conv5(f4)            # (B, hidden_channels*2, H, W)
        
        # Compute spatial attention weights
        attention_weights = self.attention(f5)  # (B, 1, H, W)
        
        # Apply attention to features and pool
        f5_attention = f5 * attention_weights  # Spatial weighting
        f5_pooled = nn.functional.adaptive_avg_pool2d(f5_attention, (1, 1))  # (B, hidden_channels*2, 1, 1)
        f5_pooled = f5_pooled.squeeze(3).squeeze(2)  # (B, hidden_channels*2)
        
        # Pool skip-connection features
        f3_pooled = nn.functional.adaptive_avg_pool2d(f3, (1, 1))  # (B, hidden_channels, 1, 1)
        f3_pooled = f3_pooled.squeeze(3).squeeze(2)  # (B, hidden_channels)
        
        # Concatenate multi-scale features
        combined_features = torch.cat([f5_pooled, f3_pooled], dim=1)  # (B, hidden_channels*3)
        
        # Predict message bits (output raw logits — sigmoid applied in loss function)
        decoded_logits = self.fc_layers(combined_features)  # (B, message_length)
        
        return decoded_logits


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
