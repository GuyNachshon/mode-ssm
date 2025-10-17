#!/usr/bin/env python3
"""
Multi-stage training script for MODE-SSM with automatic checkpoint resume.
Implements CTC warmup → Joint → Mode → Denoise curriculum with comprehensive logging.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Any
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets.brain2text import Brain2TextDataset, create_dataloaders, collate_batch
from datasets.multi_session_dataset import MultiSessionDataset, collect_session_files
from datasets.transforms import create_augmentation_pipeline
from datasets.distributed_sampler import create_distributed_sampler
from mode_ssm.models.mode_ssm_model import MODESSMModel, MODESSMConfig
from mode_ssm.checkpoint_manager import CheckpointManager, create_checkpoint_manager
from mode_ssm.evaluation_metrics import EvaluationManager, create_evaluation_manager
from mode_ssm.training_stages import CurriculumTrainer, create_curriculum_trainer
from mode_ssm.utils.rich_logging import RichLogger, RichProgressTracker, RichMetricsDisplay
from mode_ssm.utils.monitoring import SystemMonitor
from scripts.distributed_utils import (
    setup_distributed, cleanup_distributed, is_main_process,
    log_distributed_info, save_checkpoint_distributed, reduce_tensor
)


logger = logging.getLogger(__name__)
rich_logger = None  # Will be initialized in main()

# Reduce verbosity of some loggers
logging.getLogger('datasets.brain2text').setLevel(logging.WARNING)
logging.getLogger('mode_ssm.checkpoint_manager').setLevel(logging.WARNING)
logging.getLogger('scripts.distributed_utils').setLevel(logging.WARNING)


class Trainer:
    """Main trainer class for MODE-SSM"""

    def __init__(self, config: DictConfig):
        """
        Initialize trainer.

        Args:
            config: Hydra configuration
        """
        self.config = config
        self.device = None
        self.local_rank = None
        self.distributed = False

        # Training objects
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.curriculum_trainer = None
        self.checkpoint_manager = None
        self.eval_manager = None

        # Data
        self.train_loader = None
        self.val_loader = None
        self.train_sampler = None

        # Metrics tracking
        self.metrics_history = []
        self.best_val_wer = float('inf')
        self.start_epoch = 0

        # Setup
        self._setup_distributed()
        self._setup_logging()
        self._setup_device()

    def _setup_distributed(self):
        """Setup distributed training if available"""
        self.distributed, self.local_rank = setup_distributed()

        if self.distributed:
            log_distributed_info()
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Using device: {self.device}")

    def _setup_logging(self):
        """Setup logging and experiment tracking"""
        if is_main_process():
            # Setup file logging
            log_dir = Path(self.config.get('paths', {}).get('log_dir', 'logs'))
            log_dir.mkdir(parents=True, exist_ok=True)

            log_file = log_dir / f"training_{time.strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)

            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            # Initialize Weights & Biases if configured
            wandb_config = self.config.get('wandb', {})
            if wandb_config.get('enabled', False):
                wandb.init(
                    project=wandb_config.get('project', 'mode-ssm'),
                    name=wandb_config.get('run_name', None),
                    config=OmegaConf.to_container(self.config),
                    tags=wandb_config.get('tags', [])
                )

    def _setup_device(self):
        """Setup device and mixed precision"""
        # Set device (only if CUDA and not already set by distributed)
        if torch.cuda.is_available() and not self.distributed:
            torch.cuda.set_device(0)  # Use first GPU for single-GPU training

        # Mixed precision scaler
        if self.config.training.mixed_precision:
            self.scaler = GradScaler()

        # Reproducibility
        if hasattr(self.config.training, 'seed'):
            torch.manual_seed(self.config.training.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.training.seed)
        else:
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)

    def setup_data(self):
        """Setup data loaders"""
        logger.info("Setting up data loaders...")

        # Create augmentation pipeline
        augmentation = None
        augmentation_config = self.config.get('augmentation', {})
        if augmentation_config.get('enabled', False):
            augmentation = create_augmentation_pipeline(
                OmegaConf.to_container(augmentation_config)
            )

        # Dataset kwargs for both single and multi-session modes
        dataset_kwargs = {
            'min_sequence_ms': self.config.data.min_sequence_ms,
            'max_sequence_ms': self.config.data.max_sequence_ms,
            'missing_channel_threshold': self.config.data.missing_channel_threshold,
            'cache_data': self.config.data.cache_data,
            'filter_quality': True
        }

        # Check if we're using multi-session mode (data_root) or single-session mode (train_path/val_path)
        if hasattr(self.config.data, 'data_root') and self.config.data.data_root:
            # Multi-session mode: load from all sessions in directory
            logger.info(f"Using multi-session mode from: {self.config.data.data_root}")

            max_sessions = self.config.data.get('max_sessions', None)
            train_files = collect_session_files(self.config.data.data_root, "train", max_sessions)
            val_files = collect_session_files(self.config.data.data_root, "val", max_sessions)

            logger.info(f"Found {len(train_files)} train sessions, {len(val_files)} val sessions")

            train_dataset = MultiSessionDataset(
                session_paths=train_files,
                transform=augmentation,
                **dataset_kwargs
            )

            val_dataset = MultiSessionDataset(
                session_paths=val_files,
                transform=None,  # No augmentation for validation
                **dataset_kwargs
            )

            # Log session info
            train_info = train_dataset.get_session_info()
            logger.info(f"Train sessions: {train_info}")

        else:
            # Single-session mode: load from individual file paths
            logger.info("Using single-session mode")

            train_dataset = Brain2TextDataset(
                hdf5_path=self.config.data.train_path,
                transform=augmentation,
                **dataset_kwargs
            )

            val_dataset = Brain2TextDataset(
                hdf5_path=self.config.data.val_path,
                transform=None,  # No augmentation for validation
                **dataset_kwargs
            )

        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Val dataset: {len(val_dataset)} samples")

        # Create distributed sampler
        if self.distributed:
            self.train_sampler = create_distributed_sampler(
                dataset=train_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=True,
                seed=self.config.training.seed,
                drop_last=True,
                use_length_bucketing=self.config.data.use_length_bucketing
            )
        else:
            self.train_sampler = None

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
            collate_fn=collate_batch,
            persistent_workers=True if self.config.data.num_workers > 0 else False,
            drop_last=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.val_batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
            collate_fn=collate_batch,
            persistent_workers=True if self.config.data.num_workers > 0 else False,
            drop_last=False
        )

    def setup_model(self):
        """Setup model and move to device"""
        logger.info("Setting up model...")

        # Get model config - handle both nested and flat structure
        if 'model' in self.config:
            mcfg = self.config.model
        else:
            # Model config is at root level
            mcfg = self.config

        # Create model config
        model_config = MODESSMConfig(
            d_model=mcfg.get('d_model', 512),
            d_state=mcfg.get('d_state', 64),
            d_conv=mcfg.get('d_conv', 4),
            expand=mcfg.get('expand', 2),
            num_channels=mcfg.get('preprocessor', {}).get('num_channels', 512),
            encoder_layers=mcfg.get('encoder', {}).get('n_layers', 8),
            vocab_size=mcfg.get('vocab_size', 40),
            gradient_checkpointing=mcfg.get('gradient_checkpointing', False)
        )

        # Create model
        self.model = MODESSMModel(model_config).to(self.device)

        logger.info(f"Model parameters: {self.model.get_num_parameters():,}")
        logger.info(f"Model size: {self.model.get_model_size():.2f} MB")

        # Wrap with DDP if distributed
        if self.distributed:
            find_unused_params = self.config.training.get('find_unused_parameters', False)
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=find_unused_params
            )

    def setup_training_objects(self):
        """Setup optimizer, scheduler, and training managers"""
        logger.info("Setting up training objects...")

        # Optimizer factory
        def optimizer_factory(parameters, lr=None, **kwargs):
            opt_config = self.config.training.optimizer

            if lr is None:
                lr = opt_config.get('lr', 2e-4)

            opt_name = opt_config.get('name', 'AdamW').lower()
            if opt_name == "adamw":
                return optim.AdamW(
                    parameters,
                    lr=lr,
                    weight_decay=opt_config.get('weight_decay', 0.05),
                    betas=opt_config.get('betas', [0.9, 0.98]),
                    eps=opt_config.get('eps', 1e-8)
                )
            else:
                raise ValueError(f"Unknown optimizer: {opt_name}")

        # Scheduler factory
        def scheduler_factory(optimizer, **kwargs):
            sched_config = self.config.training.scheduler
            sched_name = sched_config.get('name', 'cosine_with_warmup')

            if 'cosine' in sched_name:
                # Cosine with warmup scheduler
                max_steps = sched_config.get('max_steps', 100000)
                return optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max_steps,
                    eta_min=sched_config.get('min_lr_ratio', 0.01) * optimizer.param_groups[0]['lr']
                )
            elif sched_name == "step":
                return optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=sched_config.get('step_size', 10),
                    gamma=sched_config.get('gamma', 0.1)
                )
            else:
                logger.warning(f"Unknown scheduler: {sched_name}, using no scheduler")
                return None

        # Get underlying model for curriculum trainer
        model_for_curriculum = self.model.module if self.distributed else self.model

        # Create curriculum trainer
        self.curriculum_trainer = create_curriculum_trainer(
            model=model_for_curriculum,
            config=self.config,
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory
        )

        self.optimizer = self.curriculum_trainer.optimizer
        self.scheduler = self.curriculum_trainer.scheduler

        # Create checkpoint manager
        self.checkpoint_manager = create_checkpoint_manager(self.config)

        # Create evaluation manager - handle both nested and flat model config
        model_cfg = self.config.get('model', self.config)
        self.eval_manager = create_evaluation_manager(
            OmegaConf.to_container(model_cfg)
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        # Ensure model is in training mode
        self.model.train()

        # Set epoch for distributed sampler
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        epoch_losses = []
        epoch_metrics = []
        batch_times = []

        # Get current stage info
        stage_info = self.curriculum_trainer.get_stage_info()
        loss_weights = stage_info['loss_weights']

        # Log stage info
        if hasattr(self, 'rich_logger') and self.rich_logger:
            stage_str = f"{stage_info['stage']}"
            self.rich_logger.info(f"📚 Epoch {epoch} | Stage: {stage_str} ({stage_info['stage_epoch']}/{stage_info['max_epochs']})")
        else:
            logger.info(f"Epoch {epoch} - Stage: {stage_info['stage']} ({stage_info['stage_epoch']}/{stage_info['max_epochs']})")

        for batch_idx, batch in enumerate(self.train_loader):
            batch_start = time.time()

            # Check if we should unfreeze RNN-T decoder after warmup
            if self.curriculum_trainer.maybe_unfreeze_rnnt(batch_idx):
                # Optimizer was recreated, update our reference
                self.optimizer = self.curriculum_trainer.optimizer

            # Move batch to device
            batch = self._batch_to_device(batch)

            # Forward pass with mixed precision
            with autocast(enabled=self.config.training.mixed_precision):
                outputs = self.model(
                    neural_features=batch['neural_features'],
                    sequence_lengths=batch['sequence_lengths'],
                    phoneme_labels=batch.get('phoneme_labels'),
                    label_lengths=batch.get('label_lengths'),
                    mode_labels=batch.get('mode_labels'),
                    training_stage=stage_info['stage']
                )

                # Compute losses
                losses = self.model.module.compute_loss(
                    outputs=outputs,
                    targets=batch,
                    loss_weights=loss_weights,
                    training_stage=stage_info['stage']
                ) if self.distributed else self.model.compute_loss(
                    outputs=outputs,
                    targets=batch,
                    loss_weights=loss_weights,
                    training_stage=stage_info['stage']
                )

                total_loss = losses['total_loss']

            # Check for NaN loss
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning(f"Skipping batch {batch_idx} due to NaN/Inf loss")
                continue

            # Backward pass
            self.optimizer.zero_grad()

            if self.config.training.mixed_precision:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                max_grad_norm = self.config.training.get('max_grad_norm', 1.0)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_grad_norm
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()

                # Gradient clipping
                max_grad_norm = self.config.training.get('max_grad_norm', 1.0)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_grad_norm
                    )

                self.optimizer.step()

            # Track metrics
            epoch_losses.append(total_loss.item())

            # Evaluate batch for metrics
            with torch.no_grad():
                eval_results = self.eval_manager.evaluate_batch(
                    model_outputs=outputs,
                    batch=batch,
                    training_stage=stage_info['stage']
                )
                epoch_metrics.append(eval_results)

            batch_times.append(time.time() - batch_start)

            # Log progress (only at start and end of epoch, or every 10 batches for small datasets)
            log_interval = max(1, len(self.train_loader) // 4)  # Log 4 times per epoch
            if batch_idx == 0 or batch_idx % log_interval == 0 or batch_idx == len(self.train_loader) - 1:
                avg_loss = sum(epoch_losses[-10:]) / min(10, len(epoch_losses))
                avg_time = sum(batch_times[-10:]) / min(10, len(batch_times))
                progress = (batch_idx + 1) / len(self.train_loader) * 100

                if hasattr(self, 'rich_logger') and self.rich_logger:
                    self.rich_logger.info(
                        f"  └─ Batch {batch_idx+1}/{len(self.train_loader)} ({progress:.0f}%) | "
                        f"Loss: {avg_loss:.4f} | {avg_time*1000:.0f}ms/batch"
                    )
                else:
                    logger.info(
                        f"  Batch {batch_idx+1}/{len(self.train_loader)} | "
                        f"Loss: {avg_loss:.4f} | Time: {avg_time:.2f}s"
                    )

                wandb_config = self.config.get('wandb', {})
                if is_main_process() and wandb_config.get('enabled', False):
                    wandb.log({
                        'train/loss': avg_loss,
                        'train/batch_time': avg_time,
                        'epoch': epoch,
                        'batch': batch_idx
                    })

        # Aggregate metrics
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        aggregated_metrics = self.eval_manager.aggregate_results(epoch_metrics)

        # Log training summary
        if hasattr(self, 'rich_logger') and self.rich_logger:
            self.rich_logger.info(
                f"  ✓ Training complete | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"WER: {aggregated_metrics.wer:.1f}% | "
                f"Phoneme Acc: {aggregated_metrics.phoneme_accuracy:.1f}%"
            )

        return {
            'loss': avg_loss,
            'wer': aggregated_metrics.wer,
            'phoneme_accuracy': aggregated_metrics.phoneme_accuracy,
            'mode_accuracy': aggregated_metrics.mode_accuracy
        }

    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate model.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        val_losses = []
        val_metrics = []

        # Get current stage info
        stage_info = self.curriculum_trainer.get_stage_info()
        loss_weights = stage_info['loss_weights']

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Move batch to device
                batch = self._batch_to_device(batch)

                # Forward pass
                with autocast(enabled=self.config.training.mixed_precision):
                    outputs = self.model(
                        neural_features=batch['neural_features'],
                        sequence_lengths=batch['sequence_lengths'],
                        phoneme_labels=batch.get('phoneme_labels'),
                        label_lengths=batch.get('label_lengths'),
                        mode_labels=batch.get('mode_labels'),
                        training_stage=stage_info['stage']
                    )

                    # Compute losses
                    losses = self.model.module.compute_loss(
                        outputs=outputs,
                        targets=batch,
                        loss_weights=loss_weights,
                        training_stage=stage_info['stage']
                    ) if self.distributed else self.model.compute_loss(
                        outputs=outputs,
                        targets=batch,
                        loss_weights=loss_weights,
                        training_stage=stage_info['stage']
                    )

                val_losses.append(losses['total_loss'].item())

                # Evaluate batch
                eval_results = self.eval_manager.evaluate_batch(
                    model_outputs=outputs,
                    batch=batch,
                    training_stage=stage_info['stage']
                )
                val_metrics.append(eval_results)

        # Aggregate metrics
        avg_loss = sum(val_losses) / len(val_losses)
        aggregated_metrics = self.eval_manager.aggregate_results(val_metrics)

        # Reduce metrics across processes if distributed
        if self.distributed:
            avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
            avg_loss = reduce_tensor(avg_loss_tensor, op='mean').item()

        # Log validation results
        if hasattr(self, 'rich_logger') and self.rich_logger:
            self.rich_logger.info(
                f"  ✓ Validation | "
                f"Loss: {avg_loss:.4f} | "
                f"WER: {aggregated_metrics.wer:.1f}% | "
                f"Phoneme Acc: {aggregated_metrics.phoneme_accuracy:.1f}%"
            )
        else:
            logger.info(f"Validation - Loss: {avg_loss:.4f}, WER: {aggregated_metrics.wer:.3f}")

        return {
            'loss': avg_loss,
            'wer': aggregated_metrics.wer,
            'phoneme_accuracy': aggregated_metrics.phoneme_accuracy,
            'mode_accuracy': aggregated_metrics.mode_accuracy
        }

    def train(self):
        """Main training loop"""
        logger.info("Starting training...")

        # Resume from checkpoint if available
        self.resume_from_checkpoint()

        # Calculate total epochs from curriculum stages
        total_epochs = sum(stage.epochs for stage in self.config.training.stages)
        max_epochs = self.config.training.get('max_epochs', total_epochs)

        # Training loop
        for epoch in range(self.start_epoch, max_epochs):
            epoch_start = time.time()

            # Train epoch
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Update curriculum
            all_metrics = {**train_metrics, **{f'val_{k}': v for k, v in val_metrics.items()}}
            stage_transition = self.curriculum_trainer.step_epoch(all_metrics)

            # Log stage transition
            if stage_transition and stage_transition != "complete":
                if hasattr(self, 'rich_logger') and self.rich_logger:
                    self.rich_logger.info(f"🔄 Stage transition → {stage_transition}")
                else:
                    logger.info(f"Stage transition: {stage_transition}")

            # Save checkpoint
            if is_main_process():
                self.save_checkpoint(epoch, val_metrics)

            # Log metrics
            self.log_metrics(epoch, train_metrics, val_metrics)

            # Check for early stopping
            if val_metrics['wer'] < self.best_val_wer:
                self.best_val_wer = val_metrics['wer']
                if hasattr(self, 'rich_logger') and self.rich_logger:
                    self.rich_logger.info(f"⭐ New best WER: {self.best_val_wer:.1f}%")
                else:
                    logger.info(f"New best validation WER: {self.best_val_wer:.3f}")

            epoch_time = time.time() - epoch_start
            if hasattr(self, 'rich_logger') and self.rich_logger:
                self.rich_logger.info(f"✓ Epoch {epoch} completed in {epoch_time:.1f}s\n")
            else:
                logger.info(f"Epoch {epoch} completed in {epoch_time:.1f}s")

            # Check if training is complete
            if stage_transition == "complete":
                if hasattr(self, 'rich_logger') and self.rich_logger:
                    self.rich_logger.success("🎉 All training stages completed!")
                else:
                    logger.info("All training stages completed!")
                break

        if hasattr(self, 'rich_logger') and self.rich_logger:
            self.rich_logger.success("✨ Training completed successfully!")
        else:
            logger.info("Training completed!")
        self.cleanup()

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save training checkpoint"""
        model_to_save = self.model.module if self.distributed else self.model

        # Get model and training config - handle both nested and flat structure
        model_cfg = self.config.get('model', {})
        training_cfg = self.config.get('training', {})

        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=model_to_save,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            step=epoch * len(self.train_loader),
            loss=metrics.get('loss', 0),
            training_stage=self.curriculum_trainer.get_current_stage(),
            metrics=metrics,
            model_config=model_cfg if model_cfg else None,
            training_config=training_cfg if training_cfg else None
        )

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def resume_from_checkpoint(self):
        """Resume training from checkpoint if available"""
        if not self.config.training.get('resume', False):
            return

        latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
        if latest_checkpoint is None:
            logger.info("No checkpoint found, starting from scratch")
            return

        model_to_load = self.model.module if self.distributed else self.model

        metadata = self.checkpoint_manager.restore_model(
            model=model_to_load,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            checkpoint_path=latest_checkpoint
        )

        if metadata:
            self.start_epoch = metadata.epoch + 1
            self.best_val_wer = metadata.best_wer or float('inf')
            logger.info(f"Resumed from epoch {metadata.epoch}")

    def log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Log metrics to various destinations"""
        if not is_main_process():
            return

        # Console logging is now handled in train() method

        # Log to W&B
        wandb_config = self.config.get('wandb', {})
        if wandb_config.get('enabled', False):
            wandb.log({
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/wer': train_metrics.get('wer', 0),
                'val/loss': val_metrics['loss'],
                'val/wer': val_metrics['wer'],
                'val/phoneme_accuracy': val_metrics.get('phoneme_accuracy', 0),
                'val/mode_accuracy': val_metrics.get('mode_accuracy', 0),
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })

        # Save metrics to file
        metrics_entry = {
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'timestamp': time.time()
        }
        self.metrics_history.append(metrics_entry)

        output_dir = Path(self.config.get('paths', {}).get('output_dir', 'outputs'))
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = output_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def _batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch

    def cleanup(self):
        """Cleanup resources"""
        if self.distributed:
            cleanup_distributed()

        wandb_config = self.config.get('wandb', {})
        if wandb_config.get('enabled', False):
            wandb.finish()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(config: DictConfig):
    """Main training entry point with Rich logging"""
    global rich_logger

    # Allow config modifications
    OmegaConf.set_struct(config, False)

    # Initialize Rich logger
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    rich_logger = RichLogger("MODE-SSM", log_file=log_dir / "train.log")

    # Display system info
    system_info = SystemMonitor.get_system_info()
    rich_logger.log_system_info(system_info)

    # Display config summary
    rich_logger.log_config_summary(config)

    # Display training stages
    if 'training' in config and 'stages' in config.training:
        rich_logger.log_training_stages(config.training.stages)

    rich_logger.info("🚀 Starting MODE-SSM training...")

    # Create trainer
    trainer = Trainer(config)
    trainer.rich_logger = rich_logger  # Pass rich logger to trainer

    try:
        # Setup components
        rich_logger.info("📊 Setting up data loaders...")
        trainer.setup_data()

        rich_logger.info("🏗️  Building model architecture...")
        trainer.setup_model()

        rich_logger.info("⚙️  Configuring training components...")
        trainer.setup_training_objects()

        # Train
        rich_logger.info("🎯 Beginning training curriculum...")
        trainer.train()

        rich_logger.success("✨ Training completed successfully!")

    except Exception as e:
        rich_logger.error(f"❌ Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()