"""
Checkpoint management system for MODE-SSM training.
Handles loading, saving, and validation of training checkpoints.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a training checkpoint"""
    epoch: int
    step: int
    loss: float
    val_wer: Optional[float]
    training_stage: str
    timestamp: str
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    mode_accuracy: Optional[float] = None
    best_wer: Optional[float] = None
    convergence_stage: Optional[str] = None


class CheckpointManager:
    """
    Manages training checkpoints with automatic cleanup and validation.

    Features:
    - Automatic best checkpoint tracking
    - Stage-aware checkpoint management
    - Checkpoint validation and recovery
    - Metadata tracking
    - Automatic cleanup of old checkpoints
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        save_best_only: bool = False,
        monitor_metric: str = "val_wer",
        mode: str = "min"  # "min" or "max"
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_only: Whether to save only best checkpoints
            monitor_metric: Metric to monitor for best checkpoint
            mode: Whether to minimize or maximize monitor_metric
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.monitor_metric = monitor_metric
        self.mode = mode

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track checkpoints
        self.checkpoints: List[Path] = []
        self.best_checkpoint_path: Optional[Path] = None
        self.best_metric_value: Optional[float] = None

        # Load existing checkpoints
        self._discover_existing_checkpoints()

        logger.info(f"Checkpoint manager initialized: {self.checkpoint_dir}")
        logger.info(f"  Max checkpoints: {self.max_checkpoints}")
        logger.info(f"  Monitor metric: {self.monitor_metric} ({self.mode})")

    def _discover_existing_checkpoints(self):
        """Discover existing checkpoints in the directory"""
        if not self.checkpoint_dir.exists():
            return

        # Find checkpoint files
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pth"))
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime)

        self.checkpoints = checkpoint_files[-self.max_checkpoints:]

        # Find best checkpoint
        best_path = self.checkpoint_dir / "best_checkpoint.pth"
        if best_path.exists():
            self.best_checkpoint_path = best_path
            # Try to load best metric value from metadata
            try:
                metadata = self.load_checkpoint_metadata(best_path)
                if metadata and hasattr(metadata, self.monitor_metric):
                    self.best_metric_value = getattr(metadata, self.monitor_metric)
            except Exception:
                pass

        logger.info(f"Discovered {len(self.checkpoints)} existing checkpoints")

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        step: int,
        loss: float,
        training_stage: str,
        metrics: Optional[Dict[str, float]] = None,
        model_config: Optional[DictConfig] = None,
        training_config: Optional[DictConfig] = None,
        additional_state: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save training checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            epoch: Current epoch
            step: Current training step
            loss: Current loss value
            training_stage: Current training stage
            metrics: Additional metrics (should include monitor_metric if available)
            model_config: Model configuration
            training_config: Training configuration
            additional_state: Additional state to save

        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint metadata
        metadata = CheckpointMetadata(
            epoch=epoch,
            step=step,
            loss=loss,
            val_wer=metrics.get("val_wer") if metrics else None,
            training_stage=training_stage,
            timestamp=datetime.now().isoformat(),
            model_config=OmegaConf.to_container(model_config) if model_config else {},
            training_config=OmegaConf.to_container(training_config) if training_config else {},
            mode_accuracy=metrics.get("mode_accuracy") if metrics else None,
            best_wer=self.best_metric_value
        )

        # Extract model state dict (handle DDP wrapper)
        if hasattr(model, 'module'):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        # Create checkpoint dictionary
        checkpoint = {
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'training_stage': training_stage,
            'metadata': asdict(metadata)
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if metrics:
            checkpoint['metrics'] = metrics

        if additional_state:
            checkpoint.update(additional_state)

        # Determine if this is the best checkpoint
        is_best = self._is_best_checkpoint(metrics)

        # Save checkpoint
        if self.save_best_only and not is_best:
            logger.debug("Skipping checkpoint save (save_best_only=True and not best)")
            return None

        checkpoint_path = self._generate_checkpoint_path(epoch, step)
        torch.save(checkpoint, checkpoint_path)

        # Update checkpoint tracking
        self.checkpoints.append(checkpoint_path)
        self._cleanup_old_checkpoints()

        # Save best checkpoint
        if is_best:
            self._save_best_checkpoint(checkpoint_path, metrics)

        # Save metadata separately for easier access
        metadata_path = checkpoint_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)

        logger.info(f"Checkpoint saved: {checkpoint_path.name}")
        if is_best:
            logger.info(f"  New best {self.monitor_metric}: {metrics.get(self.monitor_metric)}")

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        load_best: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint from file.

        Args:
            checkpoint_path: Path to checkpoint file (if None, loads latest)
            load_best: Whether to load best checkpoint

        Returns:
            Checkpoint dictionary or None if no checkpoint found
        """
        if load_best and self.best_checkpoint_path:
            checkpoint_path = self.best_checkpoint_path
        elif checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()

        if checkpoint_path is None:
            logger.warning("No checkpoint found to load")
            return None

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return None

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            logger.info(f"Checkpoint loaded: {checkpoint_path.name}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None

    def load_checkpoint_metadata(
        self,
        checkpoint_path: Union[str, Path]
    ) -> Optional[CheckpointMetadata]:
        """
        Load checkpoint metadata.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            CheckpointMetadata or None if not found
        """
        checkpoint_path = Path(checkpoint_path)
        metadata_path = checkpoint_path.with_suffix('.json')

        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
                return CheckpointMetadata(**metadata_dict)
            except Exception as e:
                logger.warning(f"Failed to load metadata from {metadata_path}: {e}")

        # Fallback: try to load from checkpoint file
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'metadata' in checkpoint:
                return CheckpointMetadata(**checkpoint['metadata'])
        except Exception as e:
            logger.warning(f"Failed to load metadata from checkpoint: {e}")

        return None

    def restore_model(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        load_best: bool = False,
        strict: bool = True
    ) -> Optional[CheckpointMetadata]:
        """
        Restore model and optimizer from checkpoint.

        Args:
            model: Model to restore
            optimizer: Optimizer to restore
            scheduler: Scheduler to restore
            checkpoint_path: Path to checkpoint (if None, loads latest)
            load_best: Whether to load best checkpoint
            strict: Whether to strictly match state dict keys

        Returns:
            CheckpointMetadata or None if restoration failed
        """
        checkpoint = self.load_checkpoint(checkpoint_path, load_best)
        if checkpoint is None:
            return None

        try:
            # Load model state
            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint['model_state_dict'], strict=strict
            )

            if missing_keys:
                logger.warning(f"Missing keys when loading model: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading model: {unexpected_keys}")

            # Load optimizer state
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load scheduler state
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Get metadata
            metadata = None
            if 'metadata' in checkpoint:
                metadata = CheckpointMetadata(**checkpoint['metadata'])

            logger.info(f"Model restored from checkpoint")
            return metadata

        except Exception as e:
            logger.error(f"Failed to restore model from checkpoint: {e}")
            return None

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint"""
        if not self.checkpoints:
            return None
        return max(self.checkpoints, key=lambda x: x.stat().st_mtime)

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint"""
        return self.best_checkpoint_path

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all checkpoints with metadata.

        Returns:
            List of checkpoint info dictionaries
        """
        checkpoint_info = []

        for checkpoint_path in sorted(self.checkpoints, key=lambda x: x.stat().st_mtime):
            metadata = self.load_checkpoint_metadata(checkpoint_path)
            info = {
                'path': str(checkpoint_path),
                'name': checkpoint_path.name,
                'size_mb': checkpoint_path.stat().st_size / (1024 * 1024),
                'is_best': checkpoint_path == self.best_checkpoint_path
            }

            if metadata:
                info.update({
                    'epoch': metadata.epoch,
                    'step': metadata.step,
                    'loss': metadata.loss,
                    'val_wer': metadata.val_wer,
                    'training_stage': metadata.training_stage,
                    'timestamp': metadata.timestamp
                })

            checkpoint_info.append(info)

        return checkpoint_info

    def cleanup_all(self):
        """Remove all checkpoints"""
        for checkpoint_path in self.checkpoints:
            self._remove_checkpoint(checkpoint_path)

        if self.best_checkpoint_path and self.best_checkpoint_path.exists():
            self.best_checkpoint_path.unlink()

        self.checkpoints.clear()
        self.best_checkpoint_path = None
        self.best_metric_value = None

        logger.info("All checkpoints removed")

    def _generate_checkpoint_path(self, epoch: int, step: int) -> Path:
        """Generate checkpoint file path"""
        filename = f"checkpoint_epoch_{epoch:04d}_step_{step:06d}.pth"
        return self.checkpoint_dir / filename

    def _is_best_checkpoint(self, metrics: Optional[Dict[str, float]]) -> bool:
        """Check if current metrics represent the best checkpoint"""
        if not metrics or self.monitor_metric not in metrics:
            return False

        current_value = metrics[self.monitor_metric]

        if self.best_metric_value is None:
            return True

        if self.mode == "min":
            return current_value < self.best_metric_value
        else:
            return current_value > self.best_metric_value

    def _save_best_checkpoint(
        self,
        checkpoint_path: Path,
        metrics: Dict[str, float]
    ):
        """Save best checkpoint"""
        best_path = self.checkpoint_dir / "best_checkpoint.pth"
        shutil.copy2(checkpoint_path, best_path)

        # Update best tracking
        self.best_checkpoint_path = best_path
        self.best_metric_value = metrics[self.monitor_metric]

        # Also copy metadata
        metadata_src = checkpoint_path.with_suffix('.json')
        metadata_dst = best_path.with_suffix('.json')
        if metadata_src.exists():
            shutil.copy2(metadata_src, metadata_dst)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints limit"""
        # Filter out non-existent checkpoints first
        self.checkpoints = [cp for cp in self.checkpoints if cp.exists()]

        if len(self.checkpoints) <= self.max_checkpoints:
            return

        # Sort by modification time and keep only the latest
        self.checkpoints.sort(key=lambda x: x.stat().st_mtime)
        checkpoints_to_remove = self.checkpoints[:-self.max_checkpoints]

        for checkpoint_path in checkpoints_to_remove:
            self._remove_checkpoint(checkpoint_path)

        self.checkpoints = self.checkpoints[-self.max_checkpoints:]

    def _remove_checkpoint(self, checkpoint_path: Path):
        """Remove checkpoint file and its metadata"""
        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()

            # Remove metadata file
            metadata_path = checkpoint_path.with_suffix('.json')
            if metadata_path.exists():
                metadata_path.unlink()

            logger.debug(f"Removed checkpoint: {checkpoint_path.name}")
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint {checkpoint_path}: {e}")


def create_checkpoint_manager(config: DictConfig) -> CheckpointManager:
    """
    Create checkpoint manager from configuration.

    Args:
        config: Training configuration with checkpoint settings

    Returns:
        Configured CheckpointManager
    """
    checkpoint_config = config.get('checkpoint', {})

    return CheckpointManager(
        checkpoint_dir=checkpoint_config.get('dir', './checkpoints'),
        max_checkpoints=checkpoint_config.get('max_checkpoints', 5),
        save_best_only=checkpoint_config.get('save_best_only', False),
        monitor_metric=checkpoint_config.get('monitor_metric', 'val_wer'),
        mode=checkpoint_config.get('mode', 'min')
    )