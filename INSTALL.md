# Installation Guide

## Quick Start (Basic Functionality)

The MODE-SSM system can run with basic functionality using standard PyTorch components:

```bash
# Clone repository
git clone https://github.com/your-org/mode-ssm.git
cd mode-ssm

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Full Installation (All Features)

For complete functionality including advanced SSM components, install additional dependencies:

### 1. Core Dependencies
```bash
# Already included in pyproject.toml
torch>=2.2.0
torchvision>=0.17.0
torchaudio>=2.2.0
hydra-core>=1.3.2
h5py>=3.10.0
numpy>=1.24.0
pandas>=2.0.0
transformers>=4.20.0
```

### 2. Optional Advanced Components

**Mamba SSM (for state space modeling)**
```bash
# Install mamba-ssm for advanced SSM functionality
pip install mamba-ssm==1.2.0

# Note: Requires CUDA and may need compilation
# Fallback: System uses LSTM if mamba-ssm unavailable
```

**NeMo Toolkit (for RNN-T utilities)**
```bash
# Install NeMo for advanced ASR features
pip install nemo_toolkit[asr]==1.22.0

# Note: Large dependency, optional for basic functionality
```

**Performance Libraries**
```bash
# Optional performance optimizations
pip install psutil pynvml  # System monitoring
pip install triton  # GPU optimizations
```

## Hardware Requirements

### Minimum (Basic Training)
- Python 3.8+
- 8GB RAM
- 4GB GPU memory (CUDA optional)

### Recommended (Full Features)
- Python 3.10
- 16GB RAM
- 16GB+ GPU memory (RTX 3090/A100)
- CUDA 11.8+

## Verification

Test your installation:

```bash
# Basic functionality test
python -c "import mode_ssm; print('MODE-SSM installed successfully')"

# Check available features
python -c "
import mode_ssm.models.ssm_encoder as ssm
print(f'Mamba SSM available: {ssm.MAMBA_AVAILABLE}')
"

# Run basic training test
uv run python scripts/train.py --help
```

## Troubleshooting

### Common Issues

**CUDA Not Available**
- System will automatically use CPU fallbacks
- Performance will be reduced but functionality maintained

**Mamba-SSM Installation Fails**
- System uses LSTM fallback automatically
- Install PyTorch with CUDA first: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- Then install mamba-ssm: `pip install mamba-ssm`

**Memory Issues**
- Reduce batch size in config: `data.batch_size=4`
- Enable memory optimization: `hardware.memory_optimization=true`
- Use gradient checkpointing: `hardware.gradient_checkpointing=true`

**Import Errors**
- Verify virtual environment: `which python`
- Check package installation: `pip list | grep mode-ssm`
- Reinstall in development mode: `pip install -e .`

## Development Setup

For development and testing:

```bash
# Install with development dependencies
uv sync --dev

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Run specific test categories
pytest tests/unit/ -v          # Unit tests
pytest tests/integration/ -v   # Integration tests
```

## Docker Setup (Optional)

For containerized deployment:

```bash
# Build container
docker build -t mode-ssm .

# Run training
docker run --gpus all -v $(pwd)/data:/app/data mode-ssm \
    python scripts/train.py
```

## Production Deployment

For production use:

1. **Install with pinned versions**:
   ```bash
   pip install -r requirements.txt --no-deps
   ```

2. **Enable optimizations**:
   ```bash
   export TORCH_COMPILE=1
   export CUDA_VISIBLE_DEVICES=0
   ```

3. **Configure logging**:
   ```bash
   export WANDB_MODE=offline  # Disable wandb if needed
   export PYTHONPATH=/path/to/mode-ssm
   ```

## Next Steps

After installation:

1. **Prepare data**: See `dataset.md` for data format requirements
2. **Configure training**: Edit `configs/model_mode_ssm.yaml`
3. **Start training**: Run `python scripts/train.py`
4. **Monitor progress**: Check logs in `logs/` directory

For detailed usage instructions, see the main `README.md`.