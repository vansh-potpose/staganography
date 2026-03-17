"""
config.py — Centralized configuration for the steganography project.

All hyperparameters, paths, and training settings are defined here
to avoid hard-coded values throughout the codebase.
"""

import os


# ============================================================================
# Image Settings
# ============================================================================
IMAGE_SIZE = 128          # Input images resized to IMAGE_SIZE x IMAGE_SIZE
IMAGE_CHANNELS = 3        # RGB images

# ============================================================================
# Message Settings
# ============================================================================
MESSAGE_LENGTH = 30       # Number of hidden bits to embed

# ============================================================================
# Training Hyperparameters
# ============================================================================
BATCH_SIZE = 32           # Training batch size (increased for better stability)
LEARNING_RATE = 1e-3      # Adam optimizer learning rate
NUM_EPOCHS = 100          # Total training epochs
WEIGHT_DECAY = 1e-5       # L2 regularization

# Loss weights (message loss is critical for accuracy)
LAMBDA_IMAGE = 0.5        # Weight for image reconstruction loss (reduced)
LAMBDA_MESSAGE = 2.0      # Weight for message recovery loss (increased)

# ============================================================================
# Noise Layer Settings
# ============================================================================
NOISE_PROBABILITY = 0.8   # Probability of applying noise during training

# JPEG compression
JPEG_QUALITY_MIN = 50     # Minimum JPEG quality factor
JPEG_QUALITY_MAX = 100    # Maximum JPEG quality factor

# Gaussian noise
GAUSSIAN_NOISE_MIN = 0.01  # Minimum noise standard deviation
GAUSSIAN_NOISE_MAX = 0.05  # Maximum noise standard deviation

# Cropout
CROPOUT_MIN_RATIO = 0.1   # Minimum crop area ratio
CROPOUT_MAX_RATIO = 0.3   # Maximum crop area ratio

# Gaussian blur
BLUR_KERNEL_SIZES = [3, 5, 7]   # Possible blur kernel sizes
BLUR_SIGMA_MIN = 0.5            # Minimum blur sigma
BLUR_SIGMA_MAX = 2.0            # Maximum blur sigma

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# Create directories if they don't exist
for directory in [CHECKPOINT_DIR, RESULTS_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# Device Configuration
# ============================================================================
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
