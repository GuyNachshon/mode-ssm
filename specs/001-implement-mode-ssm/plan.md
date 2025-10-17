# Implementation Plan: MODE-SSM Brain-to-Text Neural Decoder

**Branch**: `001-implement-mode-ssm` | **Date**: 2025-10-13 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-implement-mode-ssm/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement a mode-aware state-space neural decoder system for converting intracortical brain signals to text for the Brain-to-Text 2025 challenge. The system combines Mamba-based state-space modeling with multi-mode inference (silent vs vocalized speech) and test-time adaptation to achieve ≤5% Word Error Rate on the T15 dataset containing 256-channel neural recordings spanning 20 months.

## Technical Context

**Language/Version**: Python 3.10
**Primary Dependencies**: PyTorch 2.2, Mamba-SSM 1.2.0, NeMo Toolkit 1.22.0, Hydra-core 1.3.2, h5py, jiwer
**Storage**: HDF5 files for neural data (13.45GB T15 dataset), model checkpoints (≥50GB disk space)
**Testing**: pytest with synthetic neural data generation, integration tests for end-to-end pipeline
**Target Platform**: Linux with CUDA 12.1, dual-GPU A100 40GB (fallback to single GPU)
**Project Type**: Single research/ML project with modular architecture
**Performance Goals**: ≤5.0% WER on competition test set, 30-hour training time on dual A100
**Constraints**: 50ms-30s sequence length limits, >10% missing channels threshold, reproducible training
**Scale/Scope**: 10,948 sentences, 45 sessions, 256 neural channels, 512 features per 20ms bin

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### ✅ Reproducibility-First
- Configuration files mandatory: Hydra YAML configs for all training stages
- Environment specification: Python 3.10, PyTorch 2.2, explicit dependency versions
- Random seeds: Fixed seeds specified for deterministic training
- Documentation: All hyperparameters and preprocessing tracked

### ✅ Modular Architecture
- Independent components: Preprocessor, SSM encoder, mode head, decoders, TTA module
- Single responsibility: Each module handles one aspect (preprocessing, encoding, decoding)
- Clear interfaces: Defined input/output contracts between modules
- Testable components: Each module can be tested independently

### ✅ Test-Driven Development (NON-NEGOTIABLE)
- Unit tests: Required for each model component with synthetic data
- Integration tests: End-to-end pipeline validation before real neural data training
- Test coverage: All components tested before training begins

### ✅ Performance Monitoring
- Comprehensive logging: Loss curves, WER, GPU memory, compute time
- Early stopping: Validation-based stopping mechanisms
- Resource tracking: GPU utilization and memory monitoring
- Continuous monitoring: Real-time training metrics

### ✅ Experimental Rigor
- Baseline comparisons: Competition baseline WER comparisons mandatory
- Statistical validation: Multiple training runs with different seeds
- Hypothesis-driven: Mode conditioning and TTA hypotheses documented
- Failure documentation: Failed experiments logged with lessons learned

### ✅ Data Management Standards
- HDF5 format compliance: T15 dataset structure preserved
- Fixed splits: Predefined train/val/test splits maintained
- Quality handling: >10% missing channels threshold enforced
- Security: Neural data never committed to version control

**GATE STATUS: PASSED** - All constitutional requirements satisfied

## Project Structure

### Documentation (this feature)

```
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```
mode_ssm/
├── models/
│   ├── preprocessor.py        # Neural signal preprocessing and normalization
│   ├── ssm_encoder.py         # Mamba-based state space encoder
│   ├── mode_head.py           # Silent/vocalized mode classification
│   ├── denoise_flow.py        # Optional diffusion/flow bridge for denoising
│   ├── rnnt_ctc_heads.py      # RNNT (primary) and CTC (auxiliary) decoders
│   ├── lm_fusion.py           # Corpus-aware language model fusion
│   └── tta_loop.py            # Test-time adaptation module

datasets/
├── brain2text.py             # T15 dataset loading and preprocessing
├── transforms.py             # Data augmentation and temporal masking
└── distributed_sampler.py    # Multi-GPU distributed sampling

configs/
├── train.yaml                # Main training configuration
├── model_mode_ssm.yaml       # Model architecture parameters
├── lm_fusion.yaml           # Language model fusion settings
└── tta.yaml                 # Test-time adaptation config

tests/
├── unit/
│   ├── test_preprocessor.py  # Unit tests for preprocessing
│   ├── test_ssm_encoder.py   # SSM encoder tests with synthetic data
│   ├── test_mode_head.py     # Mode classification tests
│   └── test_decoders.py      # RNNT/CTC decoder tests
├── integration/
│   ├── test_end_to_end.py    # Full pipeline integration tests
│   └── test_distributed.py   # Multi-GPU training tests
└── fixtures/
    └── synthetic_neural.py   # Synthetic neural data generation

scripts/
├── train.py                  # Main training script with multi-stage curriculum
├── evaluate_model.py         # Model evaluation and WER calculation
├── make_submission.py        # Competition submission generation
└── data_preprocessing.py     # HDF5 data preprocessing utilities
```

**Structure Decision**: Single ML project structure selected. The modular architecture separates neural processing (models/), data handling (datasets/), configuration management (configs/), comprehensive testing (tests/), and execution scripts (scripts/). This supports independent development and testing of each component while maintaining clear separation of concerns for the research workflow.

