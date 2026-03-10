"""
Script to regenerate the Robust_Steganography_Complete.ipynb notebook
with the updated source code from all modules.
"""

import json
import os

# Read all source files
SRC_DIR = r"d:\Pracice\htlm\staganography\src"

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

config_py = read_file(os.path.join(SRC_DIR, "config.py"))
dataset_py = read_file(os.path.join(SRC_DIR, "dataset.py"))
model_py = read_file(os.path.join(SRC_DIR, "model.py"))
noise_layers_py = read_file(os.path.join(SRC_DIR, "noise_layers.py"))
losses_py = read_file(os.path.join(SRC_DIR, "losses.py"))
utils_py = read_file(os.path.join(SRC_DIR, "utils.py"))
train_py = read_file(os.path.join(SRC_DIR, "train.py"))
evaluate_py = read_file(os.path.join(SRC_DIR, "evaluate.py"))


def make_markdown_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines,
    }

def make_code_cell(source_lines, outputs=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": source_lines,
    }

def file_to_writefile_lines(module_path, content):
    lines = [f"%%writefile {module_path}\n"]
    for line in content.split('\n'):
        lines.append(line + "\n")
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    return lines


cells = []

# =====================================================================
# Cell 1: Title
# =====================================================================
cells.append(make_markdown_cell([
    "# 🔐 Robust Image Steganography - Complete Self-Contained Notebook\n",
    "\n",
    "**Improved architecture for 80%+ bit accuracy with 64-bit messages on COCO 2017!**\n",
    "\n",
    "---\n",
    "\n",
    "## 📋 What This Does:\n",
    "\n",
    "✅ **Creates all source code automatically** (no cloning needed)  \n",
    "✅ **Downloads COCO 2017 from Kaggle** via kagglehub (auto-cached)  \n",
    "✅ **Trains encoder-decoder** with progressive distortion-aware learning  \n",
    "✅ **Tests robustness** against JPEG/noise/crop/blur attacks  \n",
    "✅ **Generates visualizations** and downloadable results  \n",
    "\n",
    "---\n",
    "\n",
    "### 🚀 Quick Start:\n",
    "\n",
    "1. **Enable GPU**: `Runtime` → `Change runtime type` → `T4 GPU`  \n",
    "2. **Run all**: `Runtime` → `Run all` (or press Ctrl+F9)  \n",
    "3. **Wait**: ~5-8 hours (T4 GPU) for 100 epochs on COCO 2017  \n",
    "4. **Download**: Results will be zipped automatically  \n",
    "\n",
    "---\n",
    "\n",
    "### 📊 Expected Performance (64-bit messages, COCO 2017):\n",
    "\n",
    "| Metric | Target | Typical |\n",
    "|--------|--------|--------|\n",
    "| **PSNR** | >30 dB | 32-38 dB |\n",
    "| **SSIM** | >0.90 | 0.92-0.96 |\n",
    "| **Bit Accuracy (no attack)** | >95% | 95-99% |\n",
    "| **Bit Accuracy (JPEG QF=70)** | >80% | 82-90% |\n",
    "| **Bit Accuracy (noise σ=0.03)** | >80% | 82-90% |\n",
    "\n",
    "### 💡 Key Improvements:\n",
    "- **64-bit messages** (up from 30 bits = 8 bytes per message)\n",
    "- **ResBlocks + SE attention** in both encoder and decoder\n",
    "- **Multi-scale decoder** with downsampling and dual pooling\n",
    "- **BCEWithLogitsLoss** for numerical stability\n",
    "- **Progressive noise schedule** (warmup → ramp → full)\n",
    "- **Smooth residual scaling** via learnable tanh\n",
    "\n",
    "---",
]))

# =====================================================================
# Cell 2: Install Dependencies
# =====================================================================
cells.append(make_markdown_cell([
    "## 📦 Step 1: Install Dependencies & Setup"
]))

cells.append(make_code_cell([
    "# Install required packages\n",
    "!pip install -q torch torchvision Pillow scikit-image matplotlib tqdm kagglehub\n",
    "\n",
    "import os\n",
    "import sys\n",
    "\n",
    "# Create project structure\n",
    "os.makedirs('src', exist_ok=True)\n",
    "os.makedirs('data', exist_ok=True)\n",
    "os.makedirs('checkpoints', exist_ok=True)\n",
    "os.makedirs('results', exist_ok=True)\n",
    "os.makedirs('logs', exist_ok=True)\n",
    "\n",
    "# Ensure the project root is in the Python path\n",
    "if '.' not in sys.path:\n",
    "    sys.path.insert(0, '.')\n",
    "\n",
    'print("✅ Dependencies installed and directories created!")',
]))

# =====================================================================
# Cell 3: Create source files
# =====================================================================
cells.append(make_markdown_cell([
    "## 📝 Step 2: Create Source Code Modules\n",
    "\n",
    "The following cells write all the Python source code files."
]))

# __init__.py
cells.append(make_code_cell([
    "%%writefile src/__init__.py\n",
    "# Steganography source package\n",
]))

# config.py
cells.append(make_markdown_cell(["### `config.py` — Hyperparameters & Settings"]))
cells.append(make_code_cell(file_to_writefile_lines("src/config.py", config_py)))

# dataset.py
cells.append(make_markdown_cell(["### `dataset.py` — Dataset Loading & Preprocessing"]))
cells.append(make_code_cell(file_to_writefile_lines("src/dataset.py", dataset_py)))

# model.py
cells.append(make_markdown_cell(["### `model.py` — Improved Encoder & Decoder Networks"]))
cells.append(make_code_cell(file_to_writefile_lines("src/model.py", model_py)))

# noise_layers.py
cells.append(make_markdown_cell(["### `noise_layers.py` — Differentiable Distortion Layers"]))
cells.append(make_code_cell(file_to_writefile_lines("src/noise_layers.py", noise_layers_py)))

# losses.py
cells.append(make_markdown_cell(["### `losses.py` — Loss Functions"]))
cells.append(make_code_cell(file_to_writefile_lines("src/losses.py", losses_py)))

# utils.py
cells.append(make_markdown_cell(["### `utils.py` — Utility Functions"]))
cells.append(make_code_cell(file_to_writefile_lines("src/utils.py", utils_py)))

# train.py
cells.append(make_markdown_cell(["### `train.py` — Training Loop"]))
cells.append(make_code_cell(file_to_writefile_lines("src/train.py", train_py)))

# evaluate.py
cells.append(make_markdown_cell(["### `evaluate.py` — Evaluation & Robustness Testing"]))
cells.append(make_code_cell(file_to_writefile_lines("src/evaluate.py", evaluate_py)))

# =====================================================================
# Cell 4: Download COCO 2017 Dataset
# =====================================================================
cells.append(make_markdown_cell([
    "## 📥 Step 3: Download COCO 2017 Dataset\n",
    "\n",
    "Downloads COCO 2017 images via Kaggle using `kagglehub`.\n",
    "First time may take 10-15 min. Subsequent runs use the cached version.",
]))

cells.append(make_code_cell([
    "import kagglehub\n",
    "import os\n",
    "\n",
    "# Download COCO 2017 dataset from Kaggle (cached after first download)\n",
    "print('Downloading COCO 2017 dataset from Kaggle...')\n",
    "print('(This may take 10-15 minutes on first run, cached for future runs)')\n",
    "print()\n",
    "\n",
    "dataset_path = kagglehub.dataset_download('sabahesaraki/2017-2017')\n",
    "print(f'Dataset downloaded to: {dataset_path}')\n",
    "\n",
    "# Find the directory with training images\n",
    "# COCO 2017 structure typically has train2017/ and val2017/ subdirectories\n",
    "TRAIN_DIR = None\n",
    "VAL_DIR = None\n",
    "DATA_DIR = None\n",
    "\n",
    "for root, dirs, files in os.walk(dataset_path):\n",
    "    basename = os.path.basename(root)\n",
    "    img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]\n",
    "    \n",
    "    if basename == 'train2017' and len(img_files) > 100:\n",
    "        TRAIN_DIR = root\n",
    "        print(f'Found train2017: {len(img_files)} images in {root}')\n",
    "    elif basename == 'val2017' and len(img_files) > 100:\n",
    "        VAL_DIR = root\n",
    "        print(f'Found val2017: {len(img_files)} images in {root}')\n",
    "    elif len(img_files) > 1000 and DATA_DIR is None:\n",
    "        DATA_DIR = root\n",
    "        print(f'Found images dir: {len(img_files)} images in {root}')\n",
    "\n",
    "# Determine which mode to use\n",
    "if TRAIN_DIR:\n",
    "    print(f'\\nUsing separate train/val directories')\n",
    "    USE_SEPARATE_DIRS = True\n",
    "elif DATA_DIR:\n",
    "    print(f'\\nUsing single directory with auto-split')\n",
    "    USE_SEPARATE_DIRS = False\n",
    "else:\n",
    "    # Fallback: use top-level path\n",
    "    DATA_DIR = dataset_path\n",
    "    USE_SEPARATE_DIRS = False\n",
    "    print(f'\\nFallback: using dataset root: {DATA_DIR}')\n",
    "\n",
    "print('\\n✅ COCO 2017 dataset ready!')",
]))

# =====================================================================
# Cell 5: Reload modules & Setup
# =====================================================================
cells.append(make_markdown_cell([
    "## 🔧 Step 4: Import & Verify Modules"
]))

cells.append(make_code_cell([
    "# Force reload of all source modules (important in Colab)\n",
    "import importlib\n",
    "import src.config\n",
    "importlib.reload(src.config)\n",
    "import src.dataset\n",
    "importlib.reload(src.dataset)\n",
    "import src.model\n",
    "importlib.reload(src.model)\n",
    "import src.noise_layers\n",
    "importlib.reload(src.noise_layers)\n",
    "import src.losses\n",
    "importlib.reload(src.losses)\n",
    "import src.utils\n",
    "importlib.reload(src.utils)\n",
    "import src.train\n",
    "importlib.reload(src.train)\n",
    "import src.evaluate\n",
    "importlib.reload(src.evaluate)\n",
    "\n",
    "from src.config import *\n",
    "from src.model import create_model, Encoder, Decoder\n",
    "\n",
    "import torch\n",
    "\n",
    "# Quick smoke test\n",
    "print(f'Device: {DEVICE}')\n",
    "print(f'Message length: {MESSAGE_LENGTH} bits')\n",
    "print(f'Image size: {IMAGE_SIZE}x{IMAGE_SIZE}')\n",
    "print(f'Batch size: {BATCH_SIZE}')\n",
    "print(f'Learning rate: {LEARNING_RATE}')\n",
    "print(f'Lambda image: {LAMBDA_IMAGE}')\n",
    "print(f'Lambda message: {LAMBDA_MESSAGE}')\n",
    "print()\n",
    "\n",
    "# Test model creation\n",
    "enc = Encoder(message_length=MESSAGE_LENGTH)\n",
    "dec = Decoder(message_length=MESSAGE_LENGTH)\n",
    "\n",
    "x = torch.randn(2, 3, 64, 64)\n",
    "m = torch.randint(0, 2, (2, MESSAGE_LENGTH)).float()\n",
    "stego = enc(x, m)\n",
    "logits = dec(stego)\n",
    "\n",
    "print(f'Encoder output shape: {stego.shape}  (expected: [2, 3, 64, 64])')\n",
    "print(f'Decoder output shape: {logits.shape}  (expected: [2, {MESSAGE_LENGTH}])')\n",
    "print(f'Encoder params: {sum(p.numel() for p in enc.parameters()):,}')\n",
    "print(f'Decoder params: {sum(p.numel() for p in dec.parameters()):,}')\n",
    "print()\n",
    "print('✅ Model forward pass verified!')\n",
    "\n",
    "del enc, dec, x, m, stego, logits  # Clean up\n",
    "torch.cuda.empty_cache() if torch.cuda.is_available() else None",
]))

# =====================================================================
# Cell 6: Load Data
# =====================================================================
cells.append(make_markdown_cell([
    "## 📂 Step 5: Load & Prepare COCO 2017 Dataset"
]))

cells.append(make_code_cell([
    "from src.dataset import get_data_loaders\n",
    "\n",
    "# Create data loaders based on detected directory structure\n",
    "if USE_SEPARATE_DIRS:\n",
    "    # Separate train/val directories detected\n",
    "    print(f'Loading from separate directories:')\n",
    "    print(f'  Train: {TRAIN_DIR}')\n",
    "    print(f'  Val:   {VAL_DIR}')\n",
    "    train_loader, val_loader = get_data_loaders(\n",
    "        train_dir=TRAIN_DIR,\n",
    "        val_dir=VAL_DIR,\n",
    "        image_size=IMAGE_SIZE,\n",
    "        batch_size=BATCH_SIZE,\n",
    "        message_length=MESSAGE_LENGTH,\n",
    "        num_workers=2,\n",
    "    )\n",
    "else:\n",
    "    # Single directory — auto-split into train/val\n",
    "    print(f'Loading from single directory with 90/10 split:')\n",
    "    print(f'  Data: {DATA_DIR}')\n",
    "    train_loader, val_loader = get_data_loaders(\n",
    "        data_dir=DATA_DIR,\n",
    "        image_size=IMAGE_SIZE,\n",
    "        batch_size=BATCH_SIZE,\n",
    "        message_length=MESSAGE_LENGTH,\n",
    "        val_split=0.1,\n",
    "        num_workers=2,\n",
    "    )\n",
    "\n",
    "print(f'\\nTraining batches: {len(train_loader)}')\n",
    "print(f'Validation batches: {len(val_loader)}')\n",
    "\n",
    "# Verify a batch\n",
    "sample_images, sample_messages = next(iter(train_loader))\n",
    "print(f'\\nSample batch:')\n",
    "print(f'  Images shape: {sample_images.shape}')\n",
    "print(f'  Messages shape: {sample_messages.shape}')\n",
    "print(f'  Image range: [{sample_images.min():.3f}, {sample_images.max():.3f}]')\n",
    "print(f'  Message values: {sample_messages[0, :10].tolist()}')\n",
    "print('\\n✅ COCO 2017 data loaded successfully!')",
]))

# =====================================================================
# Cell 7: Train
# =====================================================================
cells.append(make_markdown_cell([
    "## 🏋️ Step 6: Train the Model\n",
    "\n",
    "Training uses a **progressive noise schedule**:\n",
    "- **Epochs 1-5**: No noise (warmup — learn clean embedding)\n",
    "- **Epochs 6-25**: Linearly increasing noise strength (0% → 100%)\n",
    "- **Epochs 26-100**: Full noise strength\n",
    "\n",
    "This allows the model to first learn a strong embedding, then gradually adapt to attacks.\n",
    "\n",
    "**⏱️ Expected time**: ~3-5 min/epoch on T4 GPU with COCO 2017 (~118K images)",
]))

cells.append(make_code_cell([
    "from src.train import train\n",
    "\n",
    "# Train the model!\n",
    "encoder, decoder, history = train(\n",
    "    train_loader=train_loader,\n",
    "    val_loader=val_loader,\n",
    "    message_length=MESSAGE_LENGTH,\n",
    "    num_epochs=NUM_EPOCHS,\n",
    "    learning_rate=LEARNING_RATE,\n",
    "    device=DEVICE,\n",
    "    checkpoint_dir=CHECKPOINT_DIR,\n",
    "    warmup_epochs=WARMUP_EPOCHS,\n",
    "    ramp_epochs=NOISE_RAMP_EPOCHS,\n",
    ")\n",
    "\n",
    "print('\\n✅ Training complete!')",
]))

# =====================================================================
# Cell 8: Evaluate
# =====================================================================
cells.append(make_markdown_cell([
    "## 📊 Step 7: Evaluate Robustness"
]))

cells.append(make_code_cell([
    "from src.evaluate import evaluate_robustness\n",
    "\n",
    "# Run comprehensive robustness evaluation\n",
    "results = evaluate_robustness(\n",
    "    encoder=encoder,\n",
    "    decoder=decoder,\n",
    "    test_loader=val_loader,\n",
    "    device=DEVICE,\n",
    "    output_dir=RESULTS_DIR,\n",
    ")\n",
    "\n",
    "print('\\n✅ Evaluation complete!')",
]))

# =====================================================================
# Cell 9: Visualize Training History
# =====================================================================
cells.append(make_markdown_cell([
    "## 📈 Step 8: Visualize Training Progress"
]))

cells.append(make_code_cell([
    "import matplotlib.pyplot as plt\n",
    "\n",
    "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
    "\n",
    "epochs = range(1, len(history['train']) + 1)\n",
    "\n",
    "# Total Loss\n",
    "axes[0, 0].plot(epochs, [m['total_loss'] for m in history['train']], label='Train', color='#3498db')\n",
    "if history['val']:\n",
    "    axes[0, 0].plot(epochs, [m['total_loss'] for m in history['val']], label='Val', color='#e74c3c')\n",
    "axes[0, 0].set_title('Total Loss', fontweight='bold')\n",
    "axes[0, 0].set_xlabel('Epoch')\n",
    "axes[0, 0].legend()\n",
    "axes[0, 0].grid(True, alpha=0.3)\n",
    "\n",
    "# Bit Accuracy\n",
    "axes[0, 1].plot(epochs, [m['bit_accuracy'] for m in history['train']], label='Train', color='#2ecc71')\n",
    "if history['val']:\n",
    "    axes[0, 1].plot(epochs, [m['bit_accuracy'] for m in history['val']], label='Val', color='#e74c3c')\n",
    "axes[0, 1].axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='80% target')\n",
    "axes[0, 1].set_title('Bit Accuracy', fontweight='bold')\n",
    "axes[0, 1].set_xlabel('Epoch')\n",
    "axes[0, 1].set_ylim(0.4, 1.05)\n",
    "axes[0, 1].legend()\n",
    "axes[0, 1].grid(True, alpha=0.3)\n",
    "\n",
    "# PSNR\n",
    "axes[1, 0].plot(epochs, [m['psnr'] for m in history['train']], label='Train', color='#9b59b6')\n",
    "if history['val']:\n",
    "    axes[1, 0].plot(epochs, [m['psnr'] for m in history['val']], label='Val', color='#e74c3c')\n",
    "axes[1, 0].set_title('PSNR (dB)', fontweight='bold')\n",
    "axes[1, 0].set_xlabel('Epoch')\n",
    "axes[1, 0].legend()\n",
    "axes[1, 0].grid(True, alpha=0.3)\n",
    "\n",
    "# Noise Strength\n",
    "noise_strengths = [m.get('noise_strength', 0) for m in history['train']]\n",
    "axes[1, 1].plot(epochs, noise_strengths, color='#e67e22', linewidth=2)\n",
    "axes[1, 1].fill_between(epochs, noise_strengths, alpha=0.3, color='#e67e22')\n",
    "axes[1, 1].set_title('Noise Strength Schedule', fontweight='bold')\n",
    "axes[1, 1].set_xlabel('Epoch')\n",
    "axes[1, 1].set_ylim(-0.05, 1.1)\n",
    "axes[1, 1].grid(True, alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(RESULTS_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "print('\\n✅ Training history plotted!')",
]))

# =====================================================================
# Cell 10: Sample Visualizations
# =====================================================================
cells.append(make_markdown_cell([
    "## 🖼️ Step 9: Visual Comparison (Cover vs Stego)"
]))

cells.append(make_code_cell([
    "from src.utils import visualize_results\n",
    "\n",
    "# Get a batch for visualization\n",
    "sample_covers, sample_msgs = next(iter(val_loader))\n",
    "sample_covers = sample_covers.to(DEVICE)\n",
    "sample_msgs = sample_msgs.to(DEVICE)\n",
    "\n",
    "with torch.no_grad():\n",
    "    sample_stegos = encoder(sample_covers, sample_msgs)\n",
    "    sample_logits = decoder(sample_stegos)\n",
    "    sample_decoded = torch.sigmoid(sample_logits)\n",
    "\n",
    "# Compute per-sample metrics\n",
    "from src.utils import compute_psnr, compute_ssim, compute_bit_accuracy\n",
    "print(f'Batch PSNR: {compute_psnr(sample_covers, sample_stegos):.2f} dB')\n",
    "print(f'Batch SSIM: {compute_ssim(sample_covers, sample_stegos):.4f}')\n",
    "print(f'Batch Bit Accuracy: {compute_bit_accuracy(sample_msgs, sample_decoded):.4f}')\n",
    "\n",
    "# Visualize\n",
    "visualize_results(\n",
    "    sample_covers, sample_stegos,\n",
    "    save_path=os.path.join(RESULTS_DIR, 'final_comparison.png'),\n",
    "    num_images=4,\n",
    ")\n",
    "\n",
    "print('\\n✅ Visual comparison saved!')",
]))

# =====================================================================
# Cell 11: Package Results
# =====================================================================
cells.append(make_markdown_cell([
    "## 📦 Step 10: Package Results for Download"
]))

cells.append(make_code_cell([
    "import shutil\n",
    "\n",
    "# Create a zip file with results\n",
    "zip_name = 'steganography_results'\n",
    "\n",
    "# List all result files\n",
    "print('Result files:')\n",
    "for root, dirs, files in os.walk('results'):\n",
    "    for f in files:\n",
    "        path = os.path.join(root, f)\n",
    "        size = os.path.getsize(path) / 1024\n",
    "        print(f'  {path} ({size:.1f} KB)')\n",
    "\n",
    "# Create zip\n",
    "shutil.make_archive(zip_name, 'zip', '.', 'results')\n",
    "print(f'\\n✅ Results packaged as {zip_name}.zip')\n",
    "\n",
    "# Download in Colab\n",
    "try:\n",
    "    from google.colab import files\n",
    "    files.download(f'{zip_name}.zip')\n",
    "    print('📥 Download started!')\n",
    "except ImportError:\n",
    "    print('Not in Colab - find the zip file in the project directory.')",
]))

# =====================================================================
# Cell 12: Summary
# =====================================================================
cells.append(make_markdown_cell([
    "## ✅ Done!\n",
    "\n",
    "### Key Results:\n",
    "- **checkpoints/best_model.pth** — Best trained model weights\n",
    "- **results/robustness_chart.png** — Robustness evaluation chart\n",
    "- **results/sample_comparisons.png** — Cover vs stego comparison\n",
    "- **results/training_history.png** — Training curves\n",
    "\n",
    "### Architecture Summary:\n",
    "- **Encoder**: Message MLP → Feature extraction → ResBlocks with SE attention → Tanh residual scaling\n",
    "- **Decoder**: Multi-scale conv (stride-2 downsample) → SE attention → Dual pooling → FC head\n",
    "- **Training**: Progressive noise (warmup→ramp→full) + CosineAnnealing LR + BCEWithLogitsLoss\n",
    "- **Dataset**: COCO 2017 (~118K train + 5K val images)\n",
    "- **Message**: 64 bits (8 bytes) embedded per image",
]))


# =====================================================================
# Assemble notebook
# =====================================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
        "colab": {
            "provenance": [],
        },
        "accelerator": "GPU",
        "gpuClass": "standard",
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

output_path = r"d:\Pracice\htlm\staganography\Robust_Steganography_Complete.ipynb"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✅ Notebook regenerated at: {output_path}")
print(f"   Total cells: {len(cells)}")
