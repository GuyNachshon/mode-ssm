# Multi-Session Training Guide

## Overview

The MODE-SSM training script now supports two modes:

1. **Single-Session Mode**: Train on data from one HDF5 file (legacy behavior)
2. **Multi-Session Mode**: Train on ALL sessions from a directory (NEW!)

## Dataset Statistics

### Single Session (t15.2023.08.11)
- Training: 288 samples
- Limited to one recording day

### All Sessions (45 sessions)
- **Training: 7,776 samples** (27x more data!)
- **Validation: 1,351 samples**
- Covers multiple recording dates from 2023-2024
- Better generalization across different neural conditions

## Usage

### Multi-Session Mode (Recommended)

Train on ALL 45 sessions:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/train.py \
    data.data_root=data/raw/t15_copyTask_neuralData/hdf5_data_final \
    model.preprocessor.num_channels=512 \
    training.batch_size=1 \
    training.gradient_accumulation_steps=4 \
    training.stages.0.epochs=2 \
    training.stages.1.epochs=3 \
    training.stages.2.epochs=2 \
    training.optimizer.lr=0.00005
```

Train on a subset (e.g., first 10 sessions for faster testing):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/train.py \
    data.data_root=data/raw/t15_copyTask_neuralData/hdf5_data_final \
    data.max_sessions=10 \
    model.preprocessor.num_channels=512 \
    training.batch_size=1
```

### Single-Session Mode (Original)

Train on a single session (for debugging or quick tests):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/train.py \
    data.train_path=data/raw/t15_copyTask_neuralData/hdf5_data_final/t15.2023.08.11/data_train.hdf5 \
    data.val_path=data/raw/t15_copyTask_neuralData/hdf5_data_final/t15.2023.08.13/data_val.hdf5 \
    model.preprocessor.num_channels=512 \
    training.batch_size=1
```

## Configuration

### Key Parameters for Multi-Session Training

- `data.data_root`: Directory containing all session folders (e.g., `hdf5_data_final/`)
- `data.max_sessions`: Optional limit on number of sessions (useful for testing)
- `data.cache_data`: Set to `false` for multi-session to save memory (default behavior)

### Recommended Settings for Full Dataset

With 7,776 training samples, you'll have:
- ~7,776 batches per epoch (at batch_size=1)
- Much longer epochs than single-session mode
- Better model generalization

Recommended adjustments:
- Reduce epochs per stage (already have more data)
- Consider larger batch_size if memory allows
- Use gradient accumulation to simulate larger batches
- Lower learning rate for stability (0.00005 instead of 0.0002)

## Implementation Details

The multi-session loader:
1. Scans `data_root` for all `t15.YYYY.MM.DD` directories
2. Finds `data_train.hdf5` and `data_val.hdf5` in each session
3. Creates individual `Brain2TextDataset` for each session
4. Combines them using PyTorch's `ConcatDataset`
5. No data duplication - memory efficient with `cache_data=false`

## Benefits

- **Better Generalization**: Model sees diverse neural conditions across many days
- **More Data**: 27x increase in training samples
- **Robustness**: Handles session-to-session variability
- **Production-Ready**: More realistic evaluation of model performance
