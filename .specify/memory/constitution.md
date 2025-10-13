<!--
Sync Impact Report:
Version change: 1.0.0 → 1.0.1
Modified principles: Data Management Standards (enhanced with dataset specifics)
Added sections: None
Removed sections: None
Templates requiring updates: All validated ✅
Follow-up TODOs: None
-->

# MODE-SSM Constitution

## Core Principles

### I. Reproducibility-First
Every experiment, model, and result must be fully reproducible through documented code, configuration files, and environment specifications. All hyperparameters, random seeds, and data preprocessing steps must be explicitly tracked. Configuration files (YAML/JSON) are mandatory for all training runs, and environment setup must be scripted.

**Rationale**: Scientific credibility and debugging efficiency depend on exact reproducibility of results across different systems and time periods.

### II. Modular Architecture
Each component (preprocessor, encoder, decoder, loss functions) must be independently testable and replaceable. Clear interfaces between modules are required. Components should follow single-responsibility principle with explicit input/output contracts.

**Rationale**: Enables systematic ablation studies, component-wise debugging, and incremental improvement of the neural architecture.

### III. Test-Driven Development (NON-NEGOTIABLE)
All model components must have unit tests verifying expected behavior with synthetic data before training on real neural data. Integration tests are required for end-to-end pipeline validation. Tests must pass before any training runs begin.

**Rationale**: Neural models are complex and errors can be subtle. Testing prevents wasted compute resources and ensures model correctness before expensive training.

### IV. Performance Monitoring
All training runs must log comprehensive metrics (loss curves, WER, computational efficiency). Model performance must be continuously monitored during training with early stopping mechanisms. Resource utilization (GPU memory, compute time) must be tracked and optimized.

**Rationale**: Efficient use of computational resources and early detection of training issues are critical for research productivity and competition success.

### V. Experimental Rigor
Every model change or hyperparameter modification must be justified with clear hypotheses and measured with statistical significance. Baseline comparisons are mandatory. All experiments must document methodology, results, and failure cases.

**Rationale**: Scientific validity requires systematic experimentation with proper controls and statistical analysis to draw meaningful conclusions.

## Data Management Standards

All neural data must be handled with strict version control and access patterns. Data preprocessing steps must be deterministic and reproducible. Training, validation, and test splits must be fixed and documented. Data augmentation strategies must be clearly specified and repeatable.

**Dataset-Specific Requirements**: The T15 dataset contains 512 neural features (256 electrodes × 2 features each) binned at 20ms resolution across 10,948 sentences from 45 sessions spanning 20 months. The predefined train/val/test splits MUST be preserved for competition validity. The dataset includes mixed speaking strategies (silent vs vocalized) and multiple sentence corpora that are unlabeled at the block level.

**Temporal Considerations**: Models MUST account for potential neural drift across the 20-month recording period. Test-time adaptation strategies should be implemented and validated. Session and block metadata must be preserved for temporal analysis.

**Format Requirements**: Data loading must use the documented HDF5 structure with proper handling of variable-length sequences. The phoneme mapping (40 classes including CTC blank and silence tokens) must be strictly followed.

**Security Requirements**: Neural data contains sensitive information and must never be committed to version control. Data access must be through secure, documented pathways only. Competition data must be handled according to Kaggle/Dryad usage terms.

## Research Workflow

All feature development follows the scientific method: hypothesis formation, experimental design, implementation, testing, analysis, and documentation. Failed experiments must be documented with lessons learned. Model architectures must be justified based on domain knowledge and prior research.

**Review Process**: All significant model changes require peer review of methodology and implementation before training begins. Results must be validated through multiple runs with different random seeds.

## Governance

This constitution supersedes all other development practices. Model architecture decisions must align with these principles. Any deviations must be explicitly justified and documented.

All experiments must verify compliance with reproducibility and testing requirements. Complexity must be justified through ablation studies and performance improvements. Use quickstart.md and training documentation for runtime development guidance.

**Version**: 1.0.1 | **Ratified**: 2025-10-13 | **Last Amended**: 2025-10-13