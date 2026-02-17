# Robust Image Steganography Using Deep Learning

A PyTorch implementation of a CNN-based encoder–decoder steganography system with distortion-aware training for robustness against JPEG compression, noise, cropping, and blur.

## Features

- **Encoder–Decoder Architecture**: CNN-based networks for message embedding and extraction
- **Distortion-Aware Training**: Differentiable noise layers (JPEG, Gaussian noise, crop, blur)
- **Comprehensive Evaluation**: PSNR, SSIM, and bit accuracy metrics under various attacks
- **Modular Design**: Clean, well-documented, extensible codebase

## Installation

```bash
# Clone or navigate to the project directory
cd steganography

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Dataset Setup

Download one of the supported datasets:

### COCO 2017 (Recommended)
```bash
# Download training images (~18GB)
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d data/

# Download validation images (~1GB)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d data/
```

### Using Custom Images
Place your images in any directory and specify the path via `--data_dir`.

## Training

```bash
python scripts/train_model.py \
    --data_dir data/train2017 \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.001 \
    --message_length 30 \
    --image_size 128
```

### Training Options

| Flag              | Default | Description                    |
|-------------------|---------|--------------------------------|
| `--data_dir`      | required| Path to training images        |
| `--epochs`        | 100     | Number of training epochs      |
| `--batch_size`    | 16      | Batch size                     |
| `--lr`            | 0.001   | Learning rate                  |
| `--message_length`| 30      | Hidden message length (bits)   |
| `--image_size`    | 128     | Input image size               |
| `--save_dir`      | checkpoints | Checkpoint save directory  |

## Robustness Testing

```bash
python scripts/test_robustness.py \
    --checkpoint checkpoints/best_model.pth \
    --data_dir data/val2017 \
    --output_dir results/
```

## Project Structure

```
├── PROJECT_REPORT.md       # Academic project report
├── README.md               # This file
├── requirements.txt        # Dependencies
├── src/
│   ├── config.py           # Configuration & hyperparameters
│   ├── dataset.py          # Dataset loading
│   ├── model.py            # Encoder & Decoder networks
│   ├── noise_layers.py     # Differentiable distortion layers
│   ├── losses.py           # Loss functions
│   ├── train.py            # Training loop
│   ├── evaluate.py         # Evaluation & metrics
│   └── utils.py            # Utilities
└── scripts/
    ├── train_model.py      # Training entry point
    └── test_robustness.py  # Testing entry point
```

## Expected Performance

| Condition           | PSNR (dB) | SSIM  | Bit Accuracy (%) |
|---------------------|-----------|-------|-------------------|
| No attack           | 36–40     | 0.95+ | 99.5+             |
| JPEG (QF=70)        | —         | —     | 93–96             |
| Gaussian noise      | —         | —     | 91–95             |
| Cropout (20%)       | —         | —     | 90–94             |
| Gaussian blur       | —         | —     | 92–96             |

## License

This project is for academic and educational purposes only.
