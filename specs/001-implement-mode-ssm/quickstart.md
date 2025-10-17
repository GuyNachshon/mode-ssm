# MODE-SSM Quickstart Guide

**Generated**: 2025-10-13
**Feature**: MODE-SSM Brain-to-Text Neural Decoder
**Target Users**: ML researchers, neuroscientists, BCI developers

## 🚀 Quick Start (5 minutes)

### Prerequisites
- **Hardware**: Linux system with CUDA 12.1, 2× A100 40GB (or ≥RTX 4090 24GB)
- **Storage**: 50GB free disk space minimum
- **Python**: 3.10 (required for dependency compatibility)

### 1. Environment Setup
```bash
# Clone repository and setup
git checkout 001-implement-mode-ssm
cd mode-ssm

# Create virtual environment with Python 3.10
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install mamba-ssm==1.2.0 diffusers hydra-core==1.3.2
pip install nemo_toolkit[asr]==1.22.0 h5py jiwer pandas tqdm rich
```

### 2. Data Preparation
```bash
# Download T15 dataset (13.45GB) from competition
# Place files in data/ directory:
# data/t15_train.hdf5, data/t15_val.hdf5, data/t15_test.hdf5

# Verify data format
python scripts/validate_data.py --data_dir data/

# Expected output: "✓ All data files valid, 10,948 trials loaded"
```

### 3. Training (Dual-GPU)
```bash
# Start multi-stage training curriculum
torchrun --nproc_per_node=2 scripts/train.py \
    config=configs/train.yaml \
    --ddp true --amp true

# Expected training time: ~30 hours on 2× A100
# Target: Validation WER ≤ 6% after stage completion
```

### 4. Generate Submission
```bash
# Generate competition submission with TTA
torchrun --nproc_per_node=2 scripts/make_submission.py \
    --checkpoint checkpoints/mode_ssm/best.pt \
    --tta true --output submission.csv

# Validate submission format
python scripts/validate_submission.py submission.csv
# Expected: "✓ Submission valid: 1,234 predictions, chronological order"
```

## 📊 Expected Results

| Stage | Validation WER | Training Time | GPU Memory |
|-------|----------------|---------------|------------|
| CTC Warmup | ~15% | 4 hours | 8GB/GPU |
| Joint Training | ~8% | 12 hours | 10GB/GPU |
| Mode Conditioning | ~6% | 8 hours | 12GB/GPU |
| Flow Bridge | ~5.5% | 6 hours | 14GB/GPU |

**Final Target**: ≤5.0% WER on competition test set

## 🔧 Common Issues & Solutions

### Memory Issues
```bash
# Single GPU fallback (if OOM on dual-GPU)
python scripts/train.py config=configs/train.yaml \
    training.batch_size=16 \
    training.gradient_accumulation_steps=8

# Enable gradient checkpointing for longer sequences
python scripts/train.py config=configs/train.yaml \
    model.encoder.gradient_checkpointing=true
```

### Data Quality Issues
```bash
# Check for corrupted trials (>10% missing channels)
python scripts/analyze_data_quality.py --data_dir data/
# Trials with quality issues are automatically skipped during training
```

### Training Resumption
```bash
# Resume from checkpoint after interruption
torchrun --nproc_per_node=2 scripts/train.py \
    config=configs/train.yaml \
    resume=checkpoints/mode_ssm/latest.pt

# System automatically resumes from last completed stage
```

## 📈 Monitoring Training

### Real-time Metrics
```bash
# Monitor training progress
tensorboard --logdir logs/mode_ssm/

# Key metrics to watch:
# - validation_wer: Target ≤ 6%
# - mode_accuracy: Target ≥ 85%
# - gpu_memory_usage: Monitor for OOM
```

### Checkpoint Management
```bash
# List saved checkpoints
ls -la checkpoints/mode_ssm/
# best.pt (best validation WER)
# latest.pt (latest training state)
# stage_*.pt (stage completion checkpoints)

# Check checkpoint details
python scripts/inspect_checkpoint.py checkpoints/mode_ssm/best.pt
```

## 🧪 Testing & Validation

### Unit Tests
```bash
# Run component tests with synthetic data
pytest tests/unit/ -v
# Expected: All tests pass, synthetic neural data generation works
```

### Integration Tests
```bash
# Test end-to-end pipeline
pytest tests/integration/test_end_to_end.py -v
# Expected: Full pipeline processes synthetic batch successfully
```

### Model Validation
```bash
# Evaluate trained model
python scripts/evaluate_model.py \
    --checkpoint checkpoints/mode_ssm/best.pt \
    --data_path data/t15_val.hdf5

# Expected output: WER metrics, mode classification accuracy, timing stats
```

## 🔄 Test-Time Adaptation

### Session-Level Adaptation
```bash
# Apply TTA during inference (automatic in make_submission.py)
python scripts/evaluate_model.py \
    --checkpoint checkpoints/mode_ssm/best.pt \
    --tta true \
    --adaptation_steps 5

# Expected improvement: 0.3-0.5 pp WER reduction
```

### Manual TTA Analysis
```bash
# Analyze neural drift across sessions
python scripts/analyze_neural_drift.py --data_dir data/
# Shows session-wise statistics and drift patterns
```

## 🎯 Performance Optimization

### Hardware-Specific Tuning
```bash
# A100 optimization (enable torch.compile)
export TORCH_COMPILE=1
python scripts/train.py config=configs/train.yaml

# Mixed GPU setup (A100 + RTX 4090)
export NCCL_P2P_DISABLE=1
torchrun --nproc_per_node=2 scripts/train.py config=configs/train.yaml
```

### Batch Size Optimization
```bash
# Find optimal batch size for your hardware
python scripts/find_max_batch_size.py --model_config configs/model_mode_ssm.yaml
# Suggests optimal batch_size and gradient_accumulation_steps
```

## 📁 Configuration Files

### Main Training Config (`configs/train.yaml`)
```yaml
# Key settings to adjust:
training:
  batch_size: 32          # Reduce if OOM
  learning_rate: 2e-4     # Conservative for stability
  max_epochs: 40          # Total across all stages

model:
  d_model: 512            # Match neural feature dimension
  d_state: 64             # SSM state dimension
  n_layers: 8             # Mamba encoder depth

data:
  missing_channel_threshold: 0.1  # Skip trials with >10% missing channels
  min_sequence_ms: 50             # Minimum sequence length
  max_sequence_ms: 30000          # Maximum sequence length (30s)
```

### Model Architecture Config (`configs/model_mode_ssm.yaml`)
```yaml
encoder:
  type: "mamba_bidirectional"
  d_model: 512
  d_state: 64
  d_conv: 4
  expand: 2
  n_layers: 8

mode_head:
  num_modes: 2            # Silent vs vocalized
  contrastive_learning: true

decoders:
  rnnt:
    vocab_size: 40        # 40 phonemes + blank + silence
    beam_size: 4
  ctc:
    blank_index: 0
```

## 🔍 Debugging & Troubleshooting

### Common Error Messages

**"CUDA out of memory"**
- Reduce batch size: `training.batch_size=16`
- Enable gradient checkpointing: `model.gradient_checkpointing=true`
- Use single GPU: Remove `torchrun --nproc_per_node=2`

**"Data validation failed"**
- Check HDF5 file integrity: `python scripts/validate_data.py`
- Verify T15 dataset format compliance

**"Training diverged (loss > 100)"**
- Reduce learning rate: `training.learning_rate=1e-4`
- Check data preprocessing: Review channel normalization

**"Mode classification accuracy < 50%"**
- Verify block-level mode labels in dataset
- Increase mode loss weight: `training.mode_weight=0.2`

### Debug Mode Training
```bash
# Enable detailed logging and validation
python scripts/train.py config=configs/train.yaml \
    debug=true \
    training.validate_every_n_steps=100 \
    training.log_every_n_steps=10
```

## 📞 Getting Help

### Log Analysis
```bash
# Check training logs for errors
tail -f logs/mode_ssm/train.log

# Analyze convergence patterns
python scripts/plot_training_curves.py logs/mode_ssm/
```

### Performance Profiling
```bash
# Profile training step
python scripts/profile_training.py --config configs/train.yaml
# Identifies bottlenecks in data loading, forward pass, backward pass
```

### Resource Monitoring
```bash
# Monitor GPU utilization
nvidia-smi -l 1

# Monitor system resources
htop
```

---

## ⚡ Success Checklist

After following this quickstart:

- [ ] Environment setup completed without errors
- [ ] T15 dataset validated and loaded successfully
- [ ] Training starts and progresses through curriculum stages
- [ ] Validation WER decreases steadily (target: ≤6% final)
- [ ] Checkpoints saved regularly with metadata
- [ ] Test-time adaptation improves WER by ≥0.3 pp
- [ ] Competition submission generated in correct format
- [ ] All unit and integration tests pass

**Expected total time**: 4-6 hours setup + 30 hours training + 4 hours inference

For advanced usage and customization, see the full documentation in `specs/001-implement-mode-ssm/`.