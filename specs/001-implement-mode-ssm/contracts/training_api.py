"""
Training API Contract for MODE-SSM Brain-to-Text System
Defines interfaces for model training, evaluation, and checkpoint management
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import torch
from pathlib import Path


class TrainingStage(Enum):
    """Multi-stage training curriculum stages"""
    CTC_WARMUP = "ctc_warmup"
    JOINT_TRAIN = "joint_train"
    MODE_TRAIN = "mode_train"
    DENOISE_TRAIN = "denoise_train"


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    # Model architecture
    d_model: int = 512
    d_state: int = 64
    n_layers: int = 8

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 2e-4
    weight_decay: float = 0.05
    max_epochs: int = 40

    # Loss weights
    rnnt_weight: float = 1.0
    ctc_weight: float = 0.3
    mode_weight: float = 0.1
    denoise_weight: float = 0.05

    # Hardware
    num_gpus: int = 2
    mixed_precision: bool = True
    gradient_clip_norm: float = 1.0

    # Data
    min_sequence_ms: int = 50
    max_sequence_ms: int = 30000
    missing_channel_threshold: float = 0.1

    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True


@dataclass
class BatchData:
    """Training batch data structure"""
    neural_features: torch.Tensor  # [B, T, 512]
    phoneme_labels: torch.Tensor   # [B, L]
    sequence_lengths: torch.Tensor  # [B]
    label_lengths: torch.Tensor    # [B]
    mode_labels: Optional[torch.Tensor] = None  # [B]
    session_ids: Optional[List[str]] = None
    quality_masks: Optional[torch.Tensor] = None  # [B, 512]


@dataclass
class ModelOutputs:
    """Model prediction outputs"""
    rnnt_logits: torch.Tensor      # [B, T, U, V] - RNNT output logits
    ctc_logits: torch.Tensor       # [B, T, V] - CTC output logits
    mode_logits: torch.Tensor      # [B, 2] - Speaking mode classification
    encoder_outputs: torch.Tensor  # [B, T, D] - Encoder representations
    attention_weights: Optional[torch.Tensor] = None  # [B, T, T] - Optional attention


@dataclass
class TrainingMetrics:
    """Training and validation metrics"""
    epoch: int
    stage: TrainingStage

    # Loss components
    total_loss: float
    rnnt_loss: float
    ctc_loss: float
    mode_loss: float
    denoise_loss: float

    # Performance metrics
    validation_wer: float
    mode_accuracy: float

    # Resource utilization
    gpu_memory_mb: List[float]
    training_time_seconds: float

    # Data quality
    trials_processed: int
    trials_skipped: int
    avg_sequence_length: float


@dataclass
class CheckpointMetadata:
    """Model checkpoint metadata"""
    epoch: int
    stage: TrainingStage
    validation_wer: float
    config: TrainingConfig
    timestamp: str
    git_commit: str
    hardware_info: Dict[str, str]
    file_path: Path
    file_size_mb: float


class DataLoaderInterface(ABC):
    """Abstract interface for neural data loading"""

    @abstractmethod
    def load_batch(self, batch_size: int) -> BatchData:
        """Load a batch of training data"""
        pass

    @abstractmethod
    def get_dataset_stats(self) -> Dict[str, float]:
        """Get dataset statistics for normalization"""
        pass

    @abstractmethod
    def validate_data_quality(self, batch: BatchData) -> Tuple[BatchData, int]:
        """Validate and filter batch data, return (filtered_batch, num_skipped)"""
        pass

    @abstractmethod
    def set_stage(self, stage: TrainingStage) -> None:
        """Configure data loading for specific training stage"""
        pass


class ModelInterface(ABC):
    """Abstract interface for MODE-SSM model"""

    @abstractmethod
    def forward(self, batch: BatchData, stage: TrainingStage) -> ModelOutputs:
        """Forward pass through model"""
        pass

    @abstractmethod
    def compute_losses(self, outputs: ModelOutputs, batch: BatchData,
                      stage: TrainingStage) -> Dict[str, torch.Tensor]:
        """Compute loss components for current training stage"""
        pass

    @abstractmethod
    def get_predictions(self, outputs: ModelOutputs) -> List[str]:
        """Decode model outputs to text predictions"""
        pass

    @abstractmethod
    def apply_test_time_adaptation(self, batch: BatchData) -> None:
        """Apply test-time adaptation for neural drift"""
        pass


class CheckpointManagerInterface(ABC):
    """Abstract interface for checkpoint management"""

    @abstractmethod
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       metrics: TrainingMetrics, config: TrainingConfig) -> Path:
        """Save model checkpoint with metadata"""
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: Path) -> Tuple[Dict, CheckpointMetadata]:
        """Load checkpoint and return state dict and metadata"""
        pass

    @abstractmethod
    def find_best_checkpoint(self, stage: TrainingStage) -> Optional[Path]:
        """Find best checkpoint for given training stage"""
        pass

    @abstractmethod
    def cleanup_old_checkpoints(self, keep_n: int = 5) -> None:
        """Remove old checkpoints to manage disk space"""
        pass

    @abstractmethod
    def get_resume_checkpoint(self, stage: TrainingStage) -> Optional[Path]:
        """Get checkpoint to resume training from after failure"""
        pass


class TrainerInterface(ABC):
    """Abstract interface for model training orchestration"""

    @abstractmethod
    def train_stage(self, stage: TrainingStage, max_epochs: int) -> TrainingMetrics:
        """Train model for specific curriculum stage"""
        pass

    @abstractmethod
    def evaluate_model(self, data_loader: DataLoaderInterface) -> TrainingMetrics:
        """Evaluate model on validation/test data"""
        pass

    @abstractmethod
    def resume_training(self, checkpoint_path: Path) -> TrainingMetrics:
        """Resume training from checkpoint after failure"""
        pass

    @abstractmethod
    def run_full_curriculum(self) -> List[TrainingMetrics]:
        """Execute complete multi-stage training curriculum"""
        pass


class SubmissionGeneratorInterface(ABC):
    """Abstract interface for competition submission generation"""

    @abstractmethod
    def generate_submission(self, test_data: DataLoaderInterface,
                          checkpoint_path: Path, output_path: Path) -> None:
        """Generate CSV submission file from test data"""
        pass

    @abstractmethod
    def validate_submission_format(self, submission_path: Path) -> bool:
        """Validate submission file format meets competition requirements"""
        pass

    @abstractmethod
    def apply_test_time_adaptation(self, model: ModelInterface,
                                 session_data: List[BatchData]) -> None:
        """Apply TTA before processing session data"""
        pass


# Training API Functions

def create_trainer(config: TrainingConfig,
                  data_loader: DataLoaderInterface,
                  model: ModelInterface,
                  checkpoint_manager: CheckpointManagerInterface) -> TrainerInterface:
    """Factory function to create trainer instance"""
    pass


def load_pretrained_model(checkpoint_path: Path) -> ModelInterface:
    """Load pretrained model from checkpoint"""
    pass


def validate_training_config(config: TrainingConfig) -> List[str]:
    """Validate training configuration and return list of issues"""
    pass


def estimate_training_resources(config: TrainingConfig) -> Dict[str, Union[int, float]]:
    """Estimate GPU memory, disk space, and time requirements"""
    pass


def setup_distributed_training(config: TrainingConfig) -> Dict[str, str]:
    """Setup distributed training environment variables"""
    pass


# Competition-specific functions

def calculate_word_error_rate(predictions: List[str], targets: List[str]) -> float:
    """Calculate WER metric for model evaluation"""
    pass


def format_submission_csv(predictions: List[str], trial_ids: List[int],
                         output_path: Path) -> None:
    """Format predictions as competition-compliant CSV"""
    pass


def validate_neural_data_format(data_path: Path) -> bool:
    """Validate HDF5 data follows T15 dataset format"""
    pass


# Error handling and logging

class TrainingError(Exception):
    """Base exception for training errors"""
    pass


class DataValidationError(TrainingError):
    """Error in neural data validation"""
    pass


class CheckpointError(TrainingError):
    """Error in checkpoint save/load operations"""
    pass


class DistributedTrainingError(TrainingError):
    """Error in distributed training setup"""
    pass


class ConfigurationError(TrainingError):
    """Error in training configuration"""
    pass