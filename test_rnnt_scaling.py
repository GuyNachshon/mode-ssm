"""Test RNN-T loss scaling to ensure it's comparable to CTC loss."""

import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/tzachi/mode-ssm')

from mode_ssm.models.mode_ssm_model import MODESSMModel, MODESSMConfig

# Create small test configuration
config = MODESSMConfig(
    d_model=128,
    num_channels=512,
    encoder_layers=2,
    vocab_size=41,
    blank_idx=0
)

model = MODESSMModel(config)
model.eval()

# Create synthetic batch
batch_size = 2
seq_len = 50
target_len = 10

neural_features = torch.randn(batch_size, seq_len, 512)
sequence_lengths = torch.tensor([50, 45])
phoneme_labels = torch.randint(1, 41, (batch_size, target_len))
label_lengths = torch.tensor([10, 8])

print("=" * 70)
print("Testing RNN-T Loss Scaling")
print("=" * 70)

# Forward pass
with torch.no_grad():
    outputs = model.forward(
        neural_features=neural_features,
        sequence_lengths=sequence_lengths,
        phoneme_labels=phoneme_labels,
        label_lengths=label_lengths,
        training_stage='joint'
    )

# Compute losses
targets = {
    'phoneme_labels': phoneme_labels,
    'sequence_lengths': sequence_lengths,
    'label_lengths': label_lengths
}

loss_weights = {
    'ctc_weight': 0.3,
    'rnnt_weight': 1.0
}

losses = model.compute_loss(outputs, targets, loss_weights, training_stage='joint')

print(f"\n✓ Forward pass successful")
print(f"  Neural features: {list(neural_features.shape)}")
print(f"  Sequence lengths: {sequence_lengths.tolist()}")
print(f"  Target lengths: {label_lengths.tolist()}")

print(f"\n✓ Loss computation successful")
print(f"  CTC loss:        {losses['ctc_loss'].item():.4f}")
print(f"  RNN-T loss:      {losses['rnnt_loss'].item():.4f}")
print(f"  Total loss:      {losses['total_loss'].item():.4f}")

# Check loss ratios
ratio = losses['rnnt_loss'].item() / max(losses['ctc_loss'].item(), 0.01)
print(f"\n✓ RNN-T/CTC ratio: {ratio:.2f}x")

# Test with longer sequences (simulate full dataset)
print(f"\n{'='*70}")
print("Testing with longer sequences (full dataset simulation)")
print(f"{'='*70}")

long_seq_len = 200
long_target_len = 40
neural_features_long = torch.randn(batch_size, long_seq_len, 512)
sequence_lengths_long = torch.tensor([200, 180])
phoneme_labels_long = torch.randint(1, 41, (batch_size, long_target_len))
label_lengths_long = torch.tensor([40, 35])

with torch.no_grad():
    outputs_long = model.forward(
        neural_features=neural_features_long,
        sequence_lengths=sequence_lengths_long,
        phoneme_labels=phoneme_labels_long,
        label_lengths=label_lengths_long,
        training_stage='joint'
    )

losses_long = model.compute_loss(outputs_long, {
    'phoneme_labels': phoneme_labels_long,
    'sequence_lengths': sequence_lengths_long,
    'label_lengths': label_lengths_long
}, loss_weights, training_stage='joint')

print(f"\n✓ Long sequence loss computation")
print(f"  Seq length: {long_seq_len}, Target length: {long_target_len}")
print(f"  CTC loss:        {losses_long['ctc_loss'].item():.4f}")
print(f"  RNN-T loss:      {losses_long['rnnt_loss'].item():.4f}")
print(f"  Total loss:      {losses_long['total_loss'].item():.4f}")

ratio_long = losses_long['rnnt_loss'].item() / max(losses_long['ctc_loss'].item(), 0.01)
print(f"  RNN-T/CTC ratio: {ratio_long:.2f}x")

if ratio > 100:
    print("  ⚠️  WARNING: RNN-T loss still much larger than CTC!")
    print("  This may cause gradient explosion during training.")
elif ratio > 10:
    print("  ⚠️  CAUTION: RNN-T loss ~10x larger, monitor for instability.")
elif ratio < 0.1:
    print("  ⚠️  CAUTION: RNN-T loss much smaller, may not learn properly.")
else:
    print("  ✓ Loss scaling looks reasonable!")

print(f"\n✓ Gradient flow check")
print(f"  CTC has grad:    {losses['ctc_loss'].requires_grad}")
print(f"  RNN-T has grad:  {losses['rnnt_loss'].requires_grad}")
print(f"  Total has grad:  {losses['total_loss'].requires_grad}")

print("\n" + "=" * 70)
print("Test completed successfully!")
print("=" * 70)
