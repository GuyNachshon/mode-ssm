# Data Model: MODE-SSM Brain-to-Text System

**Generated**: 2025-10-13
**Context**: Data entities and relationships for MODE-SSM neural decoder

## Core Entities

### Neural Recording
**Purpose**: Time-series neural activity data from 256 intracortical electrodes

**Fields**:
- `session_id: str` - Recording session identifier (e.g., "2023-01-15")
- `block_num: int` - Block number within session
- `trial_num: int` - Trial number within block
- `neural_features: float[T, 512]` - Time-series features (T timesteps × 512 features)
- `n_time_steps: int` - Number of valid timesteps in sequence
- `sampling_rate: float = 50.0` - Hz (20ms bins)
- `electrode_layout: dict` - Mapping of feature indices to electrode arrays
- `quality_mask: bool[512]` - Channel validity flags
- `artifacts: list[tuple]` - Detected artifact time ranges

**Validation Rules**:
- `neural_features.shape[1] == 512` (fixed feature count)
- `50 <= n_time_steps <= 1500` (50ms to 30s duration limits)
- `quality_mask.sum() >= 460` (≥90% channels valid, skip if <90%)
- `session_id` format: "YYYY-MM-DD"

**Relationships**:
- One-to-one with PhonemeSequence (training/validation only)
- Many-to-one with TrainingSession
- One-to-one with SpeakingMode (inferred)

### Phoneme Sequence
**Purpose**: Ground truth phoneme labels for neural recordings

**Fields**:
- `seq_class_ids: int[L]` - Integer phoneme sequence (L labels)
- `seq_len: int` - Number of valid phoneme labels
- `transcription: bytes` - ASCII sentence transcription
- `sentence_label: str` - Raw text sentence
- `phoneme_timing: float[L]` - Phoneme alignment timestamps (optional)
- `corpus_type: str` - Source corpus (Switchboard, OpenWebText, etc.)

**Validation Rules**:
- `0 <= seq_class_ids[i] <= 39` (40-class vocabulary)
- `seq_len <= len(seq_class_ids)`
- `len(sentence_label.strip()) > 0` (non-empty sentence)
- Phoneme mapping follows LOGIT_TO_PHONEME standard

**Relationships**:
- One-to-one with NeuralRecording (training/validation)
- Many-to-one with corpus category

### Speaking Mode
**Purpose**: Classification of speaking strategy (silent vs vocalized)

**Fields**:
- `mode_id: int` - 0=silent, 1=vocalized
- `mode_probability: float[2]` - Softmax probabilities [P(silent), P(vocalized)]
- `confidence: float` - Classification confidence score
- `ground_truth: Optional[int]` - True mode if known
- `block_level: bool` - Whether mode applies to entire block

**Validation Rules**:
- `mode_id in {0, 1}`
- `sum(mode_probability) ≈ 1.0`
- `0.0 <= confidence <= 1.0`

**Relationships**:
- One-to-one with NeuralRecording
- Many-to-one with block-level mode assignments

### Model Checkpoint
**Purpose**: Trained neural network state and metadata

**Fields**:
- `checkpoint_path: str` - File system path to saved model
- `model_state_dict: dict` - PyTorch model parameters
- `optimizer_state_dict: dict` - Optimizer state for resumption
- `epoch: int` - Training epoch number
- `training_stage: str` - Current curriculum stage (ctc_warmup, joint_train, etc.)
- `validation_wer: float` - Word Error Rate on validation set
- `training_config: dict` - Hyperparameters and configuration
- `creation_timestamp: datetime` - When checkpoint was created
- `git_commit: str` - Code version for reproducibility
- `hardware_info: dict` - GPU/system specifications

**Validation Rules**:
- `validation_wer >= 0.0`
- `training_stage in {"ctc_warmup", "joint_train", "mode_train", "denoise_train"}`
- `checkpoint_path` must be absolute path
- Model size estimate ≤ 2GB per checkpoint

**Relationships**:
- Many-to-one with training run
- One-to-many with evaluation results

### Training Session
**Purpose**: Neural recording session with temporal metadata

**Fields**:
- `session_date: date` - Recording date
- `session_id: str` - Unique session identifier
- `participant: str = "T15"` - Participant identifier
- `num_blocks: int` - Number of blocks in session
- `num_trials: int` - Total trials across all blocks
- `recording_quality: float` - Overall signal quality metric
- `electrode_status: dict[int, str]` - Per-electrode health status
- `session_notes: str` - Clinical/experimental notes
- `days_from_implant: int` - Days since electrode implantation

**Validation Rules**:
- `session_date` within T15 dataset range (2021-2023)
- `num_trials > 0`
- `0.0 <= recording_quality <= 1.0`
- `days_from_implant >= 0`

**Relationships**:
- One-to-many with NeuralRecording
- Sequential relationship with other sessions (temporal drift analysis)

## Data Flow Relationships

### Training Pipeline
1. **TrainingSession** → **NeuralRecording** (load raw neural data)
2. **NeuralRecording** → quality validation → **preprocessing**
3. **PhonemeSequence** → alignment → **training targets**
4. **SpeakingMode** → **mode conditioning** → **model training**
5. Training → **ModelCheckpoint** (periodic saves)

### Inference Pipeline
1. **NeuralRecording** → **preprocessing** → **feature extraction**
2. **SpeakingMode** → **mode inference** → **conditional decoding**
3. **ModelCheckpoint** → **model loading** → **prediction**
4. **Test-time adaptation** → **session statistics** → **model update**

### Data Validation Pipeline
1. **NeuralRecording** → **quality assessment** → **artifact detection**
2. **Missing channels** → **quality_mask** → **trial filtering**
3. **Sequence length** → **duration validation** → **batch filtering**
4. **Phoneme alignment** → **label validation** → **training pair creation**

## Storage Format Specifications

### HDF5 Structure (T15 Dataset)
```
trial_{session}_{block}_{trial}/
├── input_features: float32[T, 512]     # Neural features
├── seq_class_ids: int32[L]             # Phoneme labels
├── transcription: |S{max_len}          # Sentence bytes
└── attributes:
    ├── n_time_steps: int
    ├── seq_len: int
    ├── sentence_label: |S{max_len}
    ├── session: |S{date}
    ├── block_num: int
    └── trial_num: int
```

### Checkpoint Format (PyTorch)
```python
{
    'model_state_dict': OrderedDict,    # Model parameters
    'optimizer_state_dict': dict,       # Optimizer state
    'epoch': int,
    'training_stage': str,
    'validation_wer': float,
    'config': OmegaConf,               # Hydra configuration
    'metadata': {
        'timestamp': str,
        'git_commit': str,
        'hardware': dict
    }
}
```

## Data Preprocessing Transformations

### Neural Feature Processing
1. **Channel validation**: Remove trials with >10% missing channels
2. **Z-score normalization**: Per-channel statistics with EMA updates
3. **Temporal augmentation**: Random masking and time warping
4. **Sequence truncation**: Enforce 50ms-30s limits
5. **Batch padding**: Dynamic padding to sequence length

### Phoneme Label Processing
1. **Vocabulary mapping**: Map text → integer IDs via LOGIT_TO_PHONEME
2. **CTC preparation**: Add blank tokens for CTC loss
3. **RNNT preparation**: Add start/end tokens for autoregressive decoding
4. **Label smoothing**: Optional smoothing for better generalization

### Mode Label Processing
1. **Block-level inference**: Aggregate speaking strategy per block
2. **Uncertainty estimation**: Confidence scores for mode classification
3. **Temporal consistency**: Smooth mode transitions within sessions
4. **Missing label handling**: Use model predictions when ground truth unavailable

## Memory Usage Estimates

### Per-Trial Storage
- **NeuralRecording**: 4 bytes × 512 features × 1000 avg timesteps ≈ 2MB
- **PhonemeSequence**: 4 bytes × 50 avg phonemes ≈ 200 bytes
- **Metadata**: ~1KB per trial
- **Total per trial**: ~2.1MB

### Dataset Totals
- **Full T15 dataset**: 10,948 trials × 2.1MB ≈ 23GB (raw)
- **Preprocessed cache**: Additional 50% overhead ≈ 35GB total
- **Model checkpoints**: 50GB minimum for all training stages

### Batch Memory (Training)
- **Batch size 16**: 16 × 2MB × 2 (forward/backward) ≈ 64MB per batch
- **Model parameters**: ~100MB (MODE-SSM architecture)
- **Optimizer states**: ~200MB (AdamW with momentum)
- **Peak GPU memory**: 8-12GB per GPU (dual-GPU setup)