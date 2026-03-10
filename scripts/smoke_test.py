"""Smoke test for the steganography project."""
import sys
sys.path.insert(0, '.')
import torch

# Test 1: Model forward pass
print('=== Test 1: Model Forward Pass ===')
from src.model import Encoder, Decoder
enc = Encoder(message_length=64)
dec = Decoder(message_length=64)
x = torch.randn(2, 3, 64, 64)
m = torch.randint(0, 2, (2, 64)).float()
stego = enc(x, m)
logits = dec(stego)
print(f'  Encoder: {x.shape} -> {stego.shape}')
print(f'  Decoder: {stego.shape} -> {logits.shape}')
assert stego.shape == (2, 3, 64, 64), 'Encoder output shape wrong'
assert logits.shape == (2, 64), 'Decoder output shape wrong'
print('  PASS')

# Test 2: Loss function
print()
print('=== Test 2: Loss Function ===')
from src.losses import StegLoss
loss_fn = StegLoss()
loss_dict = loss_fn(x, stego, m, logits)
total = loss_dict["total"]
print(f'  Total loss: {total.item():.4f}')
print(f'  Image loss: {loss_dict["image_total"]:.4f}')
print(f'  Message loss: {loss_dict["message"]:.4f}')
assert not torch.isnan(total), 'Loss is NaN!'
print('  PASS')

# Test 3: Noise layers with noise_strength
print()
print('=== Test 3: Noise Layers ===')
from src.noise_layers import CombinedNoiseLayer, Identity
nl = CombinedNoiseLayer()
nl.train()
noised, name = nl(stego, x, noise_strength=0.5)
print(f'  Noise: {name}, shape: {noised.shape}')
# Test Identity accepts noise_strength
identity = Identity()
result = identity(stego, noise_strength=0.5)
print('  Identity with noise_strength: OK')
print('  PASS')

# Test 4: Backward pass
print()
print('=== Test 4: Backward Pass ===')
total.backward()
enc_grad_ok = all(p.grad is not None for p in enc.parameters() if p.requires_grad)
dec_grad_ok = all(p.grad is not None for p in dec.parameters() if p.requires_grad)
print(f'  Encoder gradients: {"OK" if enc_grad_ok else "MISSING"}')
print(f'  Decoder gradients: {"OK" if dec_grad_ok else "MISSING"}')
assert enc_grad_ok and dec_grad_ok, 'Some gradients are missing'
print('  PASS')

# Test 5: Config values
print()
print('=== Test 5: Config Values ===')
from src.config import MESSAGE_LENGTH, LAMBDA_IMAGE, LAMBDA_MESSAGE, WARMUP_EPOCHS, NOISE_RAMP_EPOCHS
print(f'  MESSAGE_LENGTH: {MESSAGE_LENGTH}')
print(f'  LAMBDA_IMAGE: {LAMBDA_IMAGE}')
print(f'  LAMBDA_MESSAGE: {LAMBDA_MESSAGE}')
print(f'  WARMUP_EPOCHS: {WARMUP_EPOCHS}')
print(f'  NOISE_RAMP_EPOCHS: {NOISE_RAMP_EPOCHS}')
assert MESSAGE_LENGTH == 64
assert LAMBDA_IMAGE == 0.3
assert LAMBDA_MESSAGE == 3.0
print('  PASS')

# Test 6: Progressive noise schedule
print()
print('=== Test 6: Noise Schedule ===')
from src.train import get_noise_strength
assert get_noise_strength(1, 5, 20) == 0.0, 'Epoch 1 should have no noise'
assert get_noise_strength(5, 5, 20) == 0.0, 'Epoch 5 should have no noise'
assert 0.0 < get_noise_strength(10, 5, 20) < 1.0, 'Epoch 10 should have partial noise'
assert get_noise_strength(26, 5, 20) == 1.0, 'Epoch 26 should have full noise'
print('  Schedule: epoch 1=0.0, epoch 5=0.0, epoch 10=partial, epoch 26=1.0')
print('  PASS')

print()
print('=== ALL 6 TESTS PASSED ===')
