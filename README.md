# MODE-SSM: Multi-Objective Decoder Enhancement with State Space Models

A state-of-the-art neural decoder for brain-to-text applications, featuring advanced architectural components including Mamba SSM encoders, flow bridge denoising, language model fusion, and test-time adaptation.

## 🚀 Features

### Core Architecture
- **Bidirectional Mamba SSM Encoder**: Advanced state space modeling for neural signal processing
- **Multi-Objective Training**: Joint CTC, RNN-T, and mode classification objectives
- **Speaking Mode Classification**: Automatic detection of speaking modes (attempted, overt, etc.)
- **Neural Signal Preprocessing**: Sophisticated signal conditioning and feature extraction

### Advanced Components
- **Flow Bridge Denoising**: Diffusion-based denoising with DDIM sampling for enhanced signal quality
- **Language Model Fusion**: Integration with Hugging Face language models (DialoGPT, GPT-2, etc.)
- **Test-Time Adaptation**: Real-time adaptation to neural drift with entropy minimization
- **Multi-Stage Training Curriculum**: Progressive training from CTC warmup to advanced denoising

### System Features
- **Comprehensive Error Handling**: Robust validation and recovery mechanisms
- **Performance Optimization**: Memory-efficient operations and automatic batch size optimization
- **Advanced Monitoring**: Structured logging, metrics tracking, and resource monitoring
- **Competition Ready**: Built-in submission generation for brain-to-text competitions 

## 📋 Requirements

### System Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ GPU memory (16GB+ recommended for advanced features)

### Key Dependencies
```bash
# Core ML libraries
torch>=2.0.0
torchvision
torchaudio
transformers>=4.20.0
mamba-ssm>=1.2.0

# Data processing
numpy>=1.21.0
scipy>=1.7.0
h5py>=3.7.0
pandas>=1.3.0

# Configuration and utilities
hydra-core>=1.2.0
omegaconf>=2.2.0
wandb>=0.13.0
tensorboard>=2.10.0

# Optional performance libraries
psutil>=5.9.0
pynvml>=11.0.0
```

## 🛠️ Installation

### Quick Setup
```bash
# Clone repository
git clone https://github.com/your-org/mode-ssm.git
cd mode-ssm

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Development Setup
```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify installation
python -m pytest tests/ -v
```

## 🎯 Quick Start

### Basic Training
```bash
# Train basic MODE-SSM model
python scripts/train.py

# Train with specific config
python scripts/train.py --config-name=model_mode_ssm

# Train with advanced features
python scripts/train.py --config-name=advanced_features
```

### Generate Competition Submission
```bash
# Generate submission from best checkpoint
python scripts/generate_submission.py

# Generate with specific checkpoint
python scripts/generate_submission.py checkpoint.path=/path/to/checkpoint.pth
```

### Test-Time Adaptation
```bash
# Evaluate with TTA enabled
python scripts/evaluate.py --config-name=tta_config

# Run TTA analysis
python scripts/analyze_neural_drift.py --dataset-path=/path/to/data
```

## 📁 Project Structure

```
mode-ssm/
├── configs/                    # Hydra configuration files
│   ├── model_mode_ssm.yaml    # Basic model config
│   ├── advanced_features.yaml  # Advanced features config
│   ├── tta_config.yaml        # Test-time adaptation config
│   └── submission.yaml        # Submission generation config
├── mode_ssm/                  # Main package
│   ├── models/                # Model architectures
│   │   ├── mode_ssm_model.py  # Main MODE-SSM model
│   │   ├── preprocessor.py    # Neural signal preprocessing
│   │   ├── ssm_encoder.py     # Mamba SSM encoder
│   │   ├── mode_head.py       # Speaking mode classifier
│   │   ├── denoise_flow.py    # Flow bridge denoiser
│   │   ├── lm_fusion.py       # Language model fusion
│   │   └── tta_loop.py        # Test-time adaptation
│   ├── data/                  # Data processing
│   │   ├── dataset.py         # Neural dataset handling
│   │   ├── collate.py         # Batch collation
│   │   └── transforms.py      # Data augmentation
│   ├── training/              # Training utilities
│   │   ├── trainer.py         # Main training loop
│   │   ├── curriculum.py      # Multi-stage training
│   │   └── losses.py          # Loss functions
│   └── utils/                 # Utilities
│       ├── error_handling.py  # Error handling & validation
│       ├── performance.py     # Performance optimization
│       └── monitoring.py      # Logging & monitoring
├── scripts/                   # Execution scripts
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   ├── generate_submission.py # Submission generation
│   └── analyze_neural_drift.py # TTA analysis
├── tests/                     # Test suites
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── fixtures/             # Test fixtures
└── docs/                     # Documentation
```

## ⚙️ Configuration

### Basic Model Configuration
```yaml
# configs/model_mode_ssm.yaml
model:
  d_model: 512
  num_layers: 6
  num_heads: 8
  dropout: 0.1

  # Mamba SSM settings
  ssm_config:
    d_state: 16
    d_conv: 4
    expand: 2

training:
  stages:
    ctc_warmup:
      enabled: true
      epochs: 8
      ctc_weight: 1.0
    joint:
      enabled: true
      epochs: 12
      ctc_weight: 0.4
      rnnt_weight: 0.6
```

### Advanced Features Configuration
```yaml
# configs/advanced_features.yaml
model:
  use_flow_bridge: true
  use_lm_fusion: false

  flow_bridge:
    num_flow_steps: 20
    noise_schedule: "cosine"
    parameterization: "v"

  lm_fusion:
    lm_model_name: "microsoft/DialoGPT-small"
    fusion_method: "shallow"
    lm_weight: 0.3
```

### Test-Time Adaptation Configuration
```yaml
# configs/tta_config.yaml
tta:
  enabled: true
  session_adaptation_enabled: true
  entropy_minimization_enabled: true
  adaptation_lr: 0.001
  entropy_threshold: 1.8
  confidence_threshold: 0.85
```

## 🔧 Advanced Usage

### Custom Model Components

```python
from mode_ssm.models import MODESSMModel
from omegaconf import DictConfig

# Initialize model with custom config
config = DictConfig({
    'model': {
        'd_model': 512,
        'num_layers': 8,
        'use_flow_bridge': True,
        'flow_bridge': {
            'num_flow_steps': 20,
            'noise_schedule': 'cosine'
        }
    }
})

model = MODESSMModel(config)

# Enable advanced features
model.enable_flow_bridge()
model.enable_lm_fusion("microsoft/DialoGPT-small")
```

### Custom Training Loop

```python
from mode_ssm.training import Trainer
from mode_ssm.utils.monitoring import StructuredLogger

# Setup logging
logger = StructuredLogger(
    name="custom_training",
    log_file="logs/custom_train.log",
    include_metrics=True
)

# Initialize trainer
trainer = Trainer(
    model=model,
    config=config,
    logger=logger
)

# Custom training with callbacks
trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    callbacks=[
        EarlyStoppingCallback(patience=10),
        ModelCheckpointCallback(save_top_k=3),
        TTACallback(adaptation_frequency=100)
    ]
)
```

### Performance Optimization

```python
from mode_ssm.utils.performance import MemoryOptimizer, ModelCompiler

# Optimize model for memory efficiency
model = MemoryOptimizer.optimize_model_memory(model, config)

# Compile model for performance
model = ModelCompiler.compile_model(model, config)

# Find optimal batch size
from mode_ssm.utils.performance import BatchSizeOptimizer

optimal_batch_size = BatchSizeOptimizer.find_max_batch_size(
    model=model,
    sample_input=sample_batch['neural_signals'],
    max_memory_gb=12.0
)
```

## 🧪 Testing

### Run Full Test Suite
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=mode_ssm --cov-report=html

# Run specific test categories
python -m pytest tests/unit/ -v          # Unit tests only
python -m pytest tests/integration/ -v   # Integration tests only
```

### Test Specific Components
```bash
# Test TTA functionality
python -m pytest tests/unit/test_tta_loop.py -v

# Test advanced features
python -m pytest tests/integration/test_advanced_features.py -v

# Test training pipeline
python -m pytest tests/integration/test_training_pipeline.py -v
```

## 📊 Monitoring & Debugging

### Weights & Biases Integration

MODE-SSM includes full W&B integration for experiment tracking:

```bash
# Enable W&B logging
wandb login

# Train with W&B tracking
python scripts/train.py \
    wandb.enabled=true \
    wandb.entity=your-username \
    wandb.project=mode-ssm \
    wandb.run_name="experiment-name"
```

Tracked metrics:
- **Training**: loss, WER, batch time, learning rate
- **Validation**: loss, WER, phoneme accuracy, mode accuracy
- **Config**: Complete Hydra configuration logged automatically

See [WANDB_SETUP.md](WANDB_SETUP.md) for detailed setup and usage.

### Training Monitoring
```python
from mode_ssm.utils.monitoring import TrainingMonitor, SystemMonitor

# Setup monitoring
logger = StructuredLogger("training")
monitor = TrainingMonitor(logger)

# Log training metrics
monitor.log_training_step(
    step=step,
    epoch=epoch,
    loss=loss.item(),
    learning_rate=optimizer.param_groups[0]['lr'],
    batch_size=batch_size
)

# Log validation metrics
monitor.log_validation_step(
    step=step,
    val_loss=val_loss,
    val_metrics={'accuracy': accuracy, 'wer': wer}
)
```

### Performance Profiling
```python
from mode_ssm.utils.performance import profiler

# Profile specific functions
@profiler.profile_function
def forward_pass(model, batch):
    return model(batch)

# Get performance statistics
stats = profiler.get_stats()
profiler.print_stats()
```

## 🏆 Competition Guidelines

### Data Preparation
```bash
# Prepare competition dataset
python scripts/prepare_data.py \
    --input-path=/path/to/raw/data \
    --output-path=/path/to/processed/data \
    --split-ratio=0.8

# Validate data quality
python scripts/validate_data.py --data-path=/path/to/processed/data
```

### Model Training Strategy
1. **Stage 1**: Train basic model with CTC warmup (8 epochs)
2. **Stage 2**: Add joint CTC+RNN-T training (12 epochs)
3. **Stage 3**: Enable mode classification (8 epochs)
4. **Stage 4**: Add flow bridge denoising (6 epochs)
5. **Stage 5**: Fine-tune with TTA (optional)

### Submission Generation
```bash
# Generate competition submission
python scripts/generate_submission.py \
    --checkpoint-path=checkpoints/best_model.pth \
    --test-data-path=data/competition_test.h5 \
    --output-dir=submissions/

# Validate submission format
python scripts/validate_submission.py \
    --submission-path=submissions/submission.csv
```

## 🚨 Troubleshooting

### Common Issues

**Out of Memory Errors**
```bash
# Reduce batch size and enable memory optimization
python scripts/train.py \
    data.batch_size=8 \
    hardware.memory_optimization=true \
    hardware.gradient_checkpointing=true
```

**Training Instability**
```bash
# Use gradient clipping and lower learning rate
python scripts/train.py \
    optimizer.lr=1e-5 \
    advanced.gradient_clipping=1.0 \
    advanced.loss_scaling=dynamic
```

**Slow Training**
```bash
# Enable performance optimizations
python scripts/train.py \
    hardware.mixed_precision=true \
    hardware.compile_model=true \
    data.num_workers=8
```

### Debug Mode
```bash
# Enable debug logging
python scripts/train.py \
    logging.level=DEBUG \
    logging.save_logs=true \
    +debug_mode=true
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Mamba SSM architecture based on [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- Flow bridge implementation inspired by recent advances in diffusion models
- Test-time adaptation techniques adapted from domain adaptation literature
- Competition framework designed for Brain-to-Text challenges

## 📧 Contact

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation for detailed API references

---

**Built with ❤️ for advancing brain-computer interfaces**
