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
MESSAGE_LENGTH = 64       # Number of hidden bits to embed (8 bytes)

# ============================================================================
# Training Hyperparameters
# ============================================================================
BATCH_SIZE = 32           # Training batch size
LEARNING_RATE = 5e-4      # Adam optimizer learning rate (reduced for stability)
NUM_EPOCHS = 100          # Total training epochs
WEIGHT_DECAY = 1e-5       # L2 regularization

# Loss weights — message recovery is the primary objective
LAMBDA_IMAGE = 0.3        # Weight for image reconstruction loss
LAMBDA_MESSAGE = 3.0      # Weight for message recovery loss

# ============================================================================
# Training Schedule
# ============================================================================
WARMUP_EPOCHS = 5         # Epochs without noise (learn clean embedding first)
NOISE_RAMP_EPOCHS = 20    # Epochs to linearly ramp noise to full strength

# ============================================================================
# Noise Layer Settings
# ============================================================================
NOISE_PROBABILITY = 0.5   # Max probability of applying noise during training

# JPEG compression
JPEG_QUALITY_MIN = 50     # Minimum JPEG quality factor
JPEG_QUALITY_MAX = 100    # Maximum JPEG quality factor

# Gaussian noise (gentler range)
GAUSSIAN_NOISE_MIN = 0.005  # Minimum noise standard deviation
GAUSSIAN_NOISE_MAX = 0.03   # Maximum noise standard deviation (reduced from 0.05)

# Cropout (gentler range)
CROPOUT_MIN_RATIO = 0.05   # Minimum crop area ratio
CROPOUT_MAX_RATIO = 0.2    # Maximum crop area ratio (reduced from 0.3)

# Gaussian blur
BLUR_KERNEL_SIZES = [3, 5]        # Possible blur kernel sizes (removed 7)
BLUR_SIGMA_MIN = 0.5              # Minimum blur sigma
BLUR_SIGMA_MAX = 1.5              # Maximum blur sigma (reduced from 2.0)

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
