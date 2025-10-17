# Weights & Biases (W&B) Integration

MODE-SSM includes full Weights & Biases integration for experiment tracking, metrics logging, and model visualization.

## Quick Start

### 1. Install and Login

Wandb is already installed. Login to your W&B account:

```bash
wandb login
```

This will prompt you for your API key (get it from https://wandb.ai/authorize).

### 2. Enable W&B in Config

Edit `/home/tzachi/mode-ssm/configs/train.yaml` or pass via command line:

```yaml
wandb:
  enabled: true  # Enable W&B logging
  project: "mode-ssm"
  entity: "your-username"  # Your W&B username or team
  run_name: null  # Auto-generated if null
  tags:
    - "brain-to-text"
    - "ssm"
    - "rnnt"
  notes: "MODE-SSM training with curriculum learning"
```

### 3. Run Training with W&B

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/train.py \
    wandb.enabled=true \
    wandb.entity=your-username \
    wandb.run_name="rnnt-scaled-loss-test" \
    data.data_root=data/raw/t15_copyTask_neuralData/hdf5_data_final \
    data.max_sessions=3 \
    preprocessor.num_channels=512 \
    training.batch_size=1 \
    training.gradient_accumulation_steps=4 \
    training.stages.0.epochs=2 \
    training.stages.1.epochs=2 \
    training.optimizer.lr=0.00005
```

## Logged Metrics

### Training Metrics (per batch)
- `train/loss` - Total training loss
- `train/batch_time` - Time per batch in seconds
- `batch` - Batch index
- `epoch` - Current epoch

### Validation Metrics (per epoch)
- `val/loss` - Total validation loss
- `val/wer` - Word Error Rate (%)
- `val/phoneme_accuracy` - Phoneme-level accuracy (%)
- `val/mode_accuracy` - Mode classification accuracy (%)
- `learning_rate` - Current learning rate

### Training Summary (per epoch)
- `train/wer` - Training Word Error Rate
- `epoch` - Epoch number

## Configuration Tracking

The entire Hydra configuration is automatically logged to W&B, including:
- Model architecture (d_model, vocab_size, encoder layers, etc.)
- Training hyperparameters (lr, batch size, gradient accumulation)
- Data configuration (sessions, paths, augmentation)
- Curriculum stages and loss weights

## Dashboard Examples

### Recommended Custom Charts

1. **Loss Comparison**
   - X-axis: epoch
   - Y-axis: train/loss, val/loss

2. **WER Progress**
   - X-axis: epoch
   - Y-axis: train/wer, val/wer

3. **Learning Rate Schedule**
   - X-axis: epoch
   - Y-axis: learning_rate

4. **Batch Performance**
   - X-axis: batch
   - Y-axis: train/loss, train/batch_time

## Advanced Usage

### Custom Tags

Add tags to organize experiments:

```bash
uv run python scripts/train.py \
    wandb.enabled=true \
    wandb.tags="[rnnt-fix,scaled-loss,3-sessions]"
```

### Custom Run Names

Use descriptive run names:

```bash
uv run python scripts/train.py \
    wandb.enabled=true \
    wandb.run_name="rnnt-scaled-lr5e-5-3sess-batch1"
```

### Multi-Run Comparisons

Train multiple configurations and compare in W&B:

```bash
# Run 1: Conservative learning rate
uv run python scripts/train.py wandb.enabled=true \
    wandb.run_name="lr-1e-5" training.optimizer.lr=0.00001

# Run 2: Higher learning rate
uv run python scripts/train.py wandb.enabled=true \
    wandb.run_name="lr-5e-5" training.optimizer.lr=0.00005

# Run 3: Even higher
uv run python scripts/train.py wandb.enabled=true \
    wandb.run_name="lr-1e-4" training.optimizer.lr=0.0001
```

Then compare all three in the W&B dashboard.

## Offline Mode

If you're running on a machine without internet:

```bash
wandb offline
uv run python scripts/train.py wandb.enabled=true
```

Sync later when you have internet:
```bash
wandb sync outputs/<run_dir>/wandb/
```

## Troubleshooting

### W&B Not Logging

Check that:
1. `wandb.enabled=true` in config or command line
2. You're logged in: `wandb login`
3. The `entity` (username) is correct

### API Key Issues

Reset your API key:
```bash
wandb login --relogin
```

### Slow Logging

W&B uploads asynchronously, so it shouldn't slow training. If it does:
- Check network connection
- Use `wandb offline` mode

## Integration Details

W&B is integrated in `/home/tzachi/mode-ssm/scripts/train.py`:

- **Initialization**: `_setup_logging()` method (line 124-132)
- **Batch logging**: Inside `train_epoch()` (line 496-503)
- **Epoch logging**: Inside `log_metrics()` (line 740-752)
- **Cleanup**: Inside `cleanup()` method (line 784-786)

All logging is only done on the main process in distributed training to avoid duplicate logs.

## Example W&B Dashboard

After training, you'll see:

```
🏃 Run: rnnt-scaled-loss-test
📊 Project: mode-ssm
🏷️ Tags: brain-to-text, ssm, rnnt

Training Metrics:
├── train/loss: 4470.18 → 13.36 (with scaling fix!)
├── train/wer: 30.2% → 1.0%
├── val/loss: 42.3 → 0.089
└── val/wer: 28.5% → 1.0%

Config:
├── training.optimizer.lr: 5e-5
├── training.batch_size: 1
├── preprocessor.num_channels: 512
└── data.max_sessions: 3
```

## Comparison to Previous Runs

Key experiments to track:

1. **RNN-T Loss Unscaled**: Loss 4470 → NaN (gradient explosion)
2. **RNN-T Loss Scaled v1**: Loss 193 → converged slowly
3. **RNN-T Loss Scaled v2**: Loss 8-13 → stable training! ✅

This helps you understand which fixes worked and when.
