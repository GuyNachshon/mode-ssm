"""
Multi-stage training curriculum management for MODE-SSM.
Handles CTC warmup → Joint → Mode → Denoise flow training progression.
"""

import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Union
from omegaconf import DictConfig
import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class TrainingStage(Enum):
    """Training stages in curriculum order"""
    CTC_WARMUP = "ctc_warmup"
    JOINT = "joint"
    MODE = "mode"
    DENOISE = "denoise"


@dataclass
class StageConfig:
    """Configuration for a training stage"""
    name: str
    enabled: bool = True
    epochs: int = 10

    # Loss weights
    ctc_weight: float = 1.0
    rnnt_weight: float = 0.0
    mode_weight: float = 0.0
    flow_weight: float = 0.0

    # Component activation
    train_preprocessor: bool = True
    train_encoder: bool = True
    train_mode_head: bool = False
    train_ctc_decoder: bool = True
    train_rnnt_decoder: bool = False
    train_flow_bridge: bool = False

    # Learning rate settings
    learning_rate: Optional[float] = None
    warmup_steps: Optional[int] = None

    # Validation settings
    validate_every_n_epochs: int = 1
    early_stopping_patience: Optional[int] = None
    convergence_threshold: float = 1e-4  # Loss improvement threshold

    # Additional stage-specific config
    additional_config: Dict[str, Any] = field(default_factory=dict)


class StageTransitionManager:
    """Manages transitions between training stages"""

    def __init__(self, stage_configs: Dict[str, StageConfig]):
        self.stage_configs = stage_configs
        self.current_stage = TrainingStage.CTC_WARMUP
        self.stage_epoch = 0
        self.total_epoch = 0
        self.stage_history: List[Dict[str, Any]] = []

        # Validate stage order
        self._validate_stage_configs()

        logger.info(f"Training stages initialized: {list(self.stage_configs.keys())}")

    def _validate_stage_configs(self):
        """Validate stage configurations"""
        required_stages = [TrainingStage.CTC_WARMUP, TrainingStage.JOINT]

        for stage in required_stages:
            if stage.value not in self.stage_configs:
                raise ValueError(f"Required training stage '{stage.value}' not found")

            if not self.stage_configs[stage.value].enabled:
                raise ValueError(f"Required training stage '{stage.value}' is disabled")

    def get_current_stage_config(self) -> StageConfig:
        """Get configuration for current training stage"""
        return self.stage_configs[self.current_stage.value]

    def get_stage_config(self, stage: Union[str, TrainingStage]) -> StageConfig:
        """Get configuration for specific stage"""
        if isinstance(stage, TrainingStage):
            stage = stage.value
        return self.stage_configs[stage]

    def should_transition(self, metrics: Dict[str, float]) -> bool:
        """
        Check if we should transition to next stage.

        Args:
            metrics: Current epoch metrics

        Returns:
            True if should transition to next stage
        """
        config = self.get_current_stage_config()

        # Check if stage epoch limit reached
        if self.stage_epoch >= config.epochs:
            logger.info(f"Stage {self.current_stage.value} epoch limit reached ({config.epochs})")
            return True

        # Check convergence criteria if specified
        if config.convergence_threshold > 0 and len(self.stage_history) >= 2:
            # Get recent losses - try val_loss first, then loss
            recent_losses = []
            for h in self.stage_history[-3:]:
                if 'val_loss' in h:
                    recent_losses.append(h['val_loss'])
                elif 'loss' in h:
                    recent_losses.append(h['loss'])

            if len(recent_losses) >= 2:
                loss_improvement = recent_losses[-2] - recent_losses[-1]
                if loss_improvement < config.convergence_threshold:
                    logger.info(
                        f"Stage {self.current_stage.value} converged "
                        f"(improvement: {loss_improvement:.6f} < {config.convergence_threshold})"
                    )
                    return True

        return False

    def transition_to_next_stage(self) -> Optional[TrainingStage]:
        """
        Transition to the next enabled training stage.

        Returns:
            Next stage or None if no more stages
        """
        # Record current stage completion
        self.stage_history.append({
            'stage': self.current_stage.value,
            'epochs_completed': self.stage_epoch,
            'total_epoch': self.total_epoch
        })

        # Find next enabled stage
        stage_order = [
            TrainingStage.CTC_WARMUP,
            TrainingStage.JOINT,
            TrainingStage.MODE,
            TrainingStage.DENOISE
        ]

        current_idx = stage_order.index(self.current_stage)
        next_stage = None

        for i in range(current_idx + 1, len(stage_order)):
            candidate_stage = stage_order[i]
            if (candidate_stage.value in self.stage_configs and
                self.stage_configs[candidate_stage.value].enabled):
                next_stage = candidate_stage
                break

        if next_stage is None:
            logger.info("All training stages completed")
            return None

        # Transition
        prev_stage = self.current_stage
        self.current_stage = next_stage
        self.stage_epoch = 0

        logger.info(f"Training stage transition: {prev_stage.value} -> {next_stage.value}")
        return next_stage

    def update_epoch(self, metrics: Dict[str, float]):
        """Update epoch counters and history"""
        self.stage_epoch += 1
        self.total_epoch += 1

        # Add to stage history for convergence tracking
        history_entry = {
            'stage': self.current_stage.value,
            'stage_epoch': self.stage_epoch,
            'total_epoch': self.total_epoch,
            **metrics
        }
        self.stage_history.append(history_entry)

        # Keep only recent history to prevent memory growth
        if len(self.stage_history) > 100:
            self.stage_history = self.stage_history[-100:]


class ComponentActivationManager:
    """Manages which model components are active during each stage"""

    def __init__(self, model: nn.Module):
        self.model = model
        self._original_requires_grad = {}

        # Store original parameter states
        self._save_original_states()

    def _save_original_states(self):
        """Save original requires_grad states"""
        for name, param in self.model.named_parameters():
            self._original_requires_grad[name] = param.requires_grad

    def configure_for_stage(self, stage_config: StageConfig):
        """
        Configure model components for training stage.

        Args:
            stage_config: Configuration for current stage
        """
        logger.info(f"Configuring model for stage: {stage_config.name}")

        # Map component flags to parameter patterns
        component_patterns = {
            'train_preprocessor': ['preprocessor'],
            'train_encoder': ['encoder'],
            'train_mode_head': ['mode_head'],
            'train_ctc_decoder': ['ctc_decoder'],
            'train_rnnt_decoder': ['rnnt_decoder'],
            'train_flow_bridge': ['flow_bridge']
        }

        # First, disable all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Enable parameters for active components
        active_components = []
        for component_flag, patterns in component_patterns.items():
            if getattr(stage_config, component_flag, False):
                active_components.extend(patterns)

                for name, param in self.model.named_parameters():
                    if any(pattern in name for pattern in patterns):
                        param.requires_grad = True

        # Log active components
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"  Active components: {active_components}")
        logger.info(f"  Trainable parameters: {trainable_params:,} / {total_params:,} "
                   f"({100.0 * trainable_params / total_params:.1f}%)")

    def restore_original_states(self):
        """Restore original parameter requires_grad states"""
        for name, param in self.model.named_parameters():
            if name in self._original_requires_grad:
                param.requires_grad = self._original_requires_grad[name]


class LossWeightManager:
    """Manages loss weights for different training stages"""

    def __init__(self):
        self.current_weights = {}
        self.stage_weights_history: List[Dict[str, float]] = []

    def update_weights(self, stage_config: StageConfig) -> Dict[str, float]:
        """
        Update loss weights based on stage configuration.

        Args:
            stage_config: Configuration for current stage

        Returns:
            Dictionary of loss weights
        """
        weights = {
            'ctc_weight': stage_config.ctc_weight,
            'rnnt_weight': stage_config.rnnt_weight,
            'mode_weight': stage_config.mode_weight,
            'flow_weight': stage_config.flow_weight
        }

        # Normalize weights to sum to 1 (optional)
        total_weight = sum(w for w in weights.values() if w > 0)
        if total_weight > 0:
            # Keep original weights as they are designed for the curriculum
            pass

        self.current_weights = weights
        self.stage_weights_history.append(weights.copy())

        logger.info(f"Loss weights updated: {weights}")
        return weights

    def get_current_weights(self) -> Dict[str, float]:
        """Get current loss weights"""
        return self.current_weights.copy()


class CurriculumTrainer:
    """Main curriculum training manager"""

    def __init__(
        self,
        model: nn.Module,
        stage_configs: Dict[str, StageConfig],
        optimizer_factory: Callable = None,
        scheduler_factory: Callable = None
    ):
        self.model = model
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory

        # Initialize managers
        self.stage_manager = StageTransitionManager(stage_configs)
        self.component_manager = ComponentActivationManager(model)
        self.loss_manager = LossWeightManager()

        # Current training objects
        self.optimizer = None
        self.scheduler = None

        # Initialize for first stage
        self._initialize_current_stage()

    def _initialize_current_stage(self):
        """Initialize current training stage"""
        stage_config = self.stage_manager.get_current_stage_config()

        # Configure model components
        self.component_manager.configure_for_stage(stage_config)

        # CRITICAL FIX: If transitioning to joint stage, temporarily freeze RNN-T decoder
        # to prevent gradient explosion from randomly initialized weights
        if stage_config.name == 'joint':
            logger.info("🔧 Applying RNN-T warmup: Freezing RNN-T decoder initially")
            for name, param in self.model.named_parameters():
                if 'rnnt_decoder' in name:
                    param.requires_grad = False

            # We'll unfreeze it after a few batches in the training loop
            # Store this info so training script knows to unfreeze later
            self.rnnt_warmup_needed = True
            self.rnnt_warmup_batches = 100  # Unfreeze after 100 batches
        else:
            self.rnnt_warmup_needed = False

        # Update loss weights
        self.loss_manager.update_weights(stage_config)

        # Create optimizer and scheduler for active parameters
        if self.optimizer_factory:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]

            # Use stage-specific learning rate if provided
            optimizer_kwargs = {}
            if stage_config.learning_rate is not None:
                optimizer_kwargs['lr'] = stage_config.learning_rate

            self.optimizer = self.optimizer_factory(trainable_params, **optimizer_kwargs)
            num_trainable_elements = sum(p.numel() for p in trainable_params)
            logger.info(f"Created optimizer for {len(trainable_params)} parameter tensors ({num_trainable_elements:,} elements)")

        if self.scheduler_factory and self.optimizer:
            scheduler_kwargs = {}
            if stage_config.warmup_steps is not None:
                scheduler_kwargs['warmup_steps'] = stage_config.warmup_steps

            self.scheduler = self.scheduler_factory(self.optimizer, **scheduler_kwargs)

    def step_epoch(self, metrics: Dict[str, float]) -> Optional[str]:
        """
        Step to next epoch and check for stage transitions.

        Args:
            metrics: Metrics from current epoch

        Returns:
            New stage name if transitioned, None otherwise
        """
        # Update epoch counters
        self.stage_manager.update_epoch(metrics)

        # Check for stage transition
        if self.stage_manager.should_transition(metrics):
            next_stage = self.stage_manager.transition_to_next_stage()

            if next_stage is not None:
                # Reinitialize for new stage
                self._initialize_current_stage()
                return next_stage.value
            else:
                # Training complete
                return "complete"

        return None

    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss weights"""
        return self.loss_manager.get_current_weights()

    def get_current_stage(self) -> str:
        """Get current training stage name"""
        return self.stage_manager.current_stage.value

    def get_stage_info(self) -> Dict[str, Any]:
        """Get current stage information"""
        config = self.stage_manager.get_current_stage_config()
        return {
            'stage': self.stage_manager.current_stage.value,
            'stage_epoch': self.stage_manager.stage_epoch,
            'total_epoch': self.stage_manager.total_epoch,
            'max_epochs': config.epochs,
            'loss_weights': self.loss_manager.get_current_weights()
        }

    def is_training_complete(self) -> bool:
        """Check if all training stages are complete"""
        # Check if we're past the last stage
        return self.stage_manager.current_stage == TrainingStage.DENOISE and \
               not self.stage_manager.stage_configs.get('denoise', StageConfig('denoise', enabled=False)).enabled

    def maybe_unfreeze_rnnt(self, batch_idx: int) -> bool:
        """
        Check if we should unfreeze RNN-T decoder after warmup period.

        Args:
            batch_idx: Current batch index

        Returns:
            True if RNN-T was just unfrozen
        """
        if hasattr(self, 'rnnt_warmup_needed') and self.rnnt_warmup_needed:
            if batch_idx >= self.rnnt_warmup_batches:
                logger.info(f"🔓 Unfreezing RNN-T decoder after {batch_idx} warmup batches")

                # Unfreeze RNN-T parameters
                stage_config = self.stage_manager.get_current_stage_config()
                for name, param in self.model.named_parameters():
                    if 'rnnt_decoder' in name and stage_config.train_rnnt_decoder:
                        param.requires_grad = True

                # Recreate optimizer with the newly unfrozen parameters
                if self.optimizer_factory:
                    trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                    optimizer_kwargs = {}
                    if stage_config.learning_rate is not None:
                        optimizer_kwargs['lr'] = stage_config.learning_rate

                    self.optimizer = self.optimizer_factory(trainable_params, **optimizer_kwargs)
                    logger.info(f"Recreated optimizer with RNN-T parameters")

                self.rnnt_warmup_needed = False
                return True

        return False


def create_stage_configs(config: DictConfig) -> Dict[str, StageConfig]:
    """
    Create stage configurations from Hydra config.

    Args:
        config: Training configuration

    Returns:
        Dictionary mapping stage names to StageConfig objects
    """
    stages_config = config.get('training', {}).get('stages', [])
    stage_configs = {}

    # Map stage names to canonical names
    stage_name_map = {
        'ctc_warmup': 'ctc_warmup',
        'joint_train': 'joint',
        'mode_train': 'mode',
        'denoise_train': 'denoise'
    }

    # Parse user config (can be list or dict)
    user_stages = {}

    # Check if stages_config is a list-like structure (ListConfig or list)
    try:
        # Try to iterate as list
        for stage_dict in stages_config:
            stage_name = stage_dict.get('name', '')
            canonical_name = stage_name_map.get(stage_name, stage_name)

            # Extract loss weights
            loss_weights = stage_dict.get('loss_weights', {})

            # Map components to training flags
            components = stage_dict.get('components', [])

            user_stages[canonical_name] = {
                'epochs': stage_dict.get('epochs', 10),
                'ctc_weight': loss_weights.get('ctc', 0.0),
                'rnnt_weight': loss_weights.get('rnnt', 0.0),
                'mode_weight': loss_weights.get('mode', 0.0),
                'flow_weight': loss_weights.get('denoise', 0.0),
                'train_preprocessor': 'preprocessor' in components,
                'train_encoder': 'encoder' in components,
                'train_mode_head': 'mode_head' in components,
                'train_ctc_decoder': 'ctc_decoder' in components,
                'train_rnnt_decoder': 'rnnt_decoder' in components,
                'train_flow_bridge': 'flow_bridge' in components
            }
    except (TypeError, AttributeError):
        # If not a list, assume it's a dict-like structure (not currently supported)
        logger.warning("Stages config is not a list, using defaults")
        pass

    # Default configurations for each stage
    defaults = {
        'ctc_warmup': StageConfig(
            name='ctc_warmup',
            enabled=True,
            epochs=10,
            ctc_weight=1.0,
            rnnt_weight=0.0,
            mode_weight=0.0,
            flow_weight=0.0,
            train_preprocessor=True,
            train_encoder=True,
            train_mode_head=False,
            train_ctc_decoder=True,
            train_rnnt_decoder=False,
            train_flow_bridge=False
        ),
        'joint': StageConfig(
            name='joint',
            enabled=True,
            epochs=15,
            ctc_weight=0.5,
            rnnt_weight=0.5,
            mode_weight=0.0,
            flow_weight=0.0,
            train_preprocessor=True,
            train_encoder=True,
            train_mode_head=False,
            train_ctc_decoder=True,
            train_rnnt_decoder=True,
            train_flow_bridge=False
        ),
        'mode': StageConfig(
            name='mode',
            enabled=True,
            epochs=10,
            ctc_weight=0.3,
            rnnt_weight=0.3,
            mode_weight=0.4,
            flow_weight=0.0,
            train_preprocessor=True,
            train_encoder=True,
            train_mode_head=True,
            train_ctc_decoder=True,
            train_rnnt_decoder=True,
            train_flow_bridge=False
        ),
        'denoise': StageConfig(
            name='denoise',
            enabled=False,  # Optional stage
            epochs=5,
            ctc_weight=0.2,
            rnnt_weight=0.2,
            mode_weight=0.2,
            flow_weight=0.4,
            train_preprocessor=True,
            train_encoder=True,
            train_mode_head=True,
            train_ctc_decoder=True,
            train_rnnt_decoder=True,
            train_flow_bridge=False
        )
    }

    # Override with user configuration
    for stage_name, default_config in defaults.items():
        user_config = user_stages.get(stage_name, {})

        # Create new config with merged values
        merged_values = {}
        for key in ['enabled', 'epochs', 'ctc_weight', 'rnnt_weight', 'mode_weight', 'flow_weight',
                    'train_preprocessor', 'train_encoder', 'train_mode_head',
                    'train_ctc_decoder', 'train_rnnt_decoder', 'train_flow_bridge']:
            if key in user_config:
                merged_values[key] = user_config[key]
            else:
                merged_values[key] = getattr(default_config, key)

        stage_config = StageConfig(
            name=stage_name,
            **merged_values
        )

        stage_configs[stage_name] = stage_config

    return stage_configs


def create_curriculum_trainer(
    model: nn.Module,
    config: DictConfig,
    optimizer_factory: Callable = None,
    scheduler_factory: Callable = None
) -> CurriculumTrainer:
    """
    Create curriculum trainer from configuration.

    Args:
        model: Model to train
        config: Training configuration
        optimizer_factory: Function to create optimizer
        scheduler_factory: Function to create scheduler

    Returns:
        Configured CurriculumTrainer
    """
    stage_configs = create_stage_configs(config)

    return CurriculumTrainer(
        model=model,
        stage_configs=stage_configs,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory
    )