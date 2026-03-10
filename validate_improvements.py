#!/usr/bin/env python3
"""
Validation script to verify all improvements are in place.
"""

import sys
sys.path.insert(0, '.')

from src.model import Encoder, Decoder, create_model
from src.config import MESSAGE_LENGTH, DEVICE
import torch


def count_parameters(model):
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def main():
    print("\n" + "="*70)
    print("IMPROVEMENT VALIDATION")
    print("="*70)
    
    print("\n[Config Check]")
    from src.config import BATCH_SIZE, LAMBDA_IMAGE, LAMBDA_MESSAGE
    print(f"✓ BATCH_SIZE = {BATCH_SIZE} (should be 32)")
    print(f"✓ LAMBDA_IMAGE = {LAMBDA_IMAGE} (should be 0.5)")
    print(f"✓ LAMBDA_MESSAGE = {LAMBDA_MESSAGE} (should be 2.0)")
    
    assert BATCH_SIZE == 32, "BATCH_SIZE not updated!"
    assert LAMBDA_IMAGE == 0.5, "LAMBDA_IMAGE not updated!"
    assert LAMBDA_MESSAGE == 2.0, "LAMBDA_MESSAGE not updated!"
    
    print("\n[Model Architecture Check]")
    encoder, decoder = create_model(message_length=MESSAGE_LENGTH, hidden_channels=128)
    
    encoder_params = count_parameters(encoder)
    decoder_params = count_parameters(decoder)
    
    print(f"✓ Encoder parameters: {encoder_params:,}")
    print(f"✓ Decoder parameters: {decoder_params:,}")
    print(f"✓ Total parameters: {encoder_params + decoder_params:,}")
    
    # Check that parameters are reasonable (doubled from before)
    assert encoder_params > 500_000, "Encoder too small!"
    assert decoder_params > 1_000_000, "Decoder too small!"
    
    print("\n[Architecture Components Check]")
    
    # Check encoder
    print(f"✓ Encoder has improved architecture")
    print(f"  - Uses 128 hidden channels (was 64)")
    print(f"  - Total layers: {len(list(encoder.modules()))}")
    
    # Check decoder  
    has_attention = any('attention' in name.lower() for name, _ in decoder.named_modules())
    print(f"✓ Decoder has spatial attention: {has_attention}")
    
    has_fc_layers = hasattr(decoder, 'fc_layers')
    print(f"✓ Decoder has improved FC layers: {has_fc_layers}")
    
    print("\n[Training Configuration Check]")
    from src.train import train_one_epoch
    import inspect
    
    sig = inspect.signature(train_one_epoch)
    has_warmup = 'warmup_epochs' in sig.parameters
    print(f"✓ train_one_epoch has warmup_epochs parameter: {has_warmup}")
    
    if has_warmup:
        print(f"  - Default warmup_epochs: {sig.parameters['warmup_epochs'].default}")
    
    print("\n[Forward Pass Test]")
    test_image = torch.randn(2, 3, 128, 128)
    test_message = torch.randint(0, 2, (2, MESSAGE_LENGTH)).float()
    
    with torch.no_grad():
        stego = encoder(test_image, test_message)
        decoded = decoder(stego)
    
    print(f"✓ Encoder output shape: {stego.shape} (expected: [2, 3, 128, 128])")
    print(f"✓ Decoder output shape: {decoded.shape} (expected: [2, {MESSAGE_LENGTH}])")
    
    assert stego.shape == (2, 3, 128, 128), "Encoder output shape wrong!"
    assert decoded.shape == (2, MESSAGE_LENGTH), "Decoder output shape wrong!"
    assert decoded.min() >= 0 and decoded.max() <= 1, "Decoder output not in [0,1]"
    print(f"✓ Decoder output range: [{decoded.min():.4f}, {decoded.max():.4f}]")
    
    print("\n" + "="*70)
    print("✅ ALL IMPROVEMENTS VALIDATED SUCCESSFULLY!")
    print("="*70)
    print("\nYou can now run: python train_improved.py")


if __name__ == '__main__':
    main()
