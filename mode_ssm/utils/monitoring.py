"""
Comprehensive logging and monitoring utilities for MODE-SSM.
Provides structured logging, metrics tracking, and system monitoring.
"""

import json
import logging
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np
from omegaconf import DictConfig


class StructuredLogger:
    """Enhanced logging with structured output and metrics tracking."""

    def __init__(
        self,
        name: str,
        log_file: Optional[Union[str, Path]] = None,
        level: int = logging.INFO,
        include_metrics: bool = True
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.metrics_enabled = include_metrics
        self.metrics = {}
        self.start_time = time.time()

        # Clear existing handlers
        self.logger.handlers.clear()

        # Console handler with enhanced formatting
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        # JSON log handler for structured data
        if log_file:
            json_log_file = log_file.parent / f"{log_file.stem}_structured.json"
            self.json_handler = JsonLogHandler(json_log_file)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics with timestamp and optional step."""
        if not self.metrics_enabled:
            return

        timestamp = time.time()
        log_entry = {
            'timestamp': timestamp,
            'elapsed_time': timestamp - self.start_time,
            'metrics': metrics
        }

        if step is not None:
            log_entry['step'] = step

        # Log to console
        metrics_str = ', '.join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                for k, v in metrics.items()])
        self.logger.info(f"METRICS - {metrics_str}")

        # Log structured data if available
        if hasattr(self, 'json_handler'):
            self.json_handler.log_entry(log_entry)

        # Update internal metrics tracking
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append((timestamp, value))

    def log_system_info(self) -> None:
        """Log comprehensive system information."""
        info = SystemMonitor.get_system_info()
        self.logger.info("SYSTEM INFO:")
        for key, value in info.items():
            self.logger.info(f"  {key}: {value}")

    def log_model_info(self, model: torch.nn.Module, config: DictConfig) -> None:
        """Log model architecture and configuration information."""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info("MODEL INFO:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        self.logger.info(f"  Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")

        # Log model architecture summary
        if hasattr(model, 'config'):
            self.logger.info(f"  Model config: {model.config}")

        # Log key config parameters
        if 'model' in config:
            model_config = config.model
            self.logger.info(f"  d_model: {model_config.get('d_model', 'N/A')}")
            self.logger.info(f"  num_layers: {model_config.get('num_layers', 'N/A')}")
            self.logger.info(f"  num_heads: {model_config.get('num_heads', 'N/A')}")

    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all tracked metrics."""
        summary = {}

        for metric_name, values in self.metrics.items():
            if not values:
                continue

            numeric_values = [v for _, v in values if isinstance(v, (int, float))]
            if not numeric_values:
                continue

            summary[metric_name] = {
                'count': len(numeric_values),
                'mean': np.mean(numeric_values),
                'std': np.std(numeric_values),
                'min': np.min(numeric_values),
                'max': np.max(numeric_values),
                'latest': numeric_values[-1]
            }

        return summary


class JsonLogHandler:
    """Handler for structured JSON logging."""

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.lock = threading.Lock()

        # Ensure directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_entry(self, entry: Dict[str, Any]) -> None:
        """Log a structured entry to JSON file."""
        with self.lock:
            with open(self.log_file, 'a') as f:
                json.dump(entry, f, default=str)
                f.write('\n')


class SystemMonitor:
    """Monitor system resources and performance."""

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information."""
        info = {
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }

        # CUDA information
        if torch.cuda.is_available():
            info.update({
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version(),
                'gpu_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
            })

            # GPU details
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info[f'gpu_{i}_name'] = props.name
                info[f'gpu_{i}_memory_gb'] = props.total_memory / 1e9

        # System information
        try:
            import psutil
            info.update({
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / 1e9,
                'disk_free_gb': psutil.disk_usage('/').free / 1e9,
            })
        except ImportError:
            pass

        return info

    @staticmethod
    def get_resource_usage() -> Dict[str, Any]:
        """Get current resource usage."""
        usage = {
            'timestamp': time.time(),
        }

        # GPU usage
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            usage.update({
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated(device) / 1e9,
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved(device) / 1e9,
                'gpu_memory_max_allocated_gb': torch.cuda.max_memory_allocated(device) / 1e9,
            })

            # GPU utilization if nvidia-ml-py available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                usage.update({
                    'gpu_utilization_percent': gpu_util.gpu,
                    'gpu_memory_utilization_percent': gpu_util.memory,
                    'gpu_memory_used_gb': memory_info.used / 1e9,
                    'gpu_memory_total_gb': memory_info.total / 1e9,
                })
            except ImportError:
                pass

        # System usage
        try:
            import psutil
            process = psutil.Process()
            usage.update({
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'process_memory_mb': process.memory_info().rss / 1024 / 1024,
                'process_cpu_percent': process.cpu_percent(),
            })
        except ImportError:
            pass

        return usage


class TrainingMonitor:
    """Monitor training progress and detect issues."""

    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.loss_history = []
        self.lr_history = []
        self.last_improvement_step = 0
        self.best_metric = None
        self.plateau_patience = 50
        self.gradient_accumulation = []

    def log_training_step(
        self,
        step: int,
        epoch: int,
        loss: float,
        learning_rate: float,
        batch_size: int,
        additional_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log training step information."""

        # Update history
        self.loss_history.append((step, loss))
        self.lr_history.append((step, learning_rate))

        # Prepare metrics
        metrics = {
            'step': step,
            'epoch': epoch,
            'loss': loss,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
        }

        if additional_metrics:
            metrics.update(additional_metrics)

        # Add derived metrics
        if len(self.loss_history) > 10:
            recent_losses = [l for _, l in self.loss_history[-10:]]
            metrics['loss_smoothed'] = np.mean(recent_losses)
            metrics['loss_trend'] = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]

        self.logger.log_metrics(metrics, step=step)

        # Check for training issues
        self._check_training_health(step, loss, learning_rate)

    def log_validation_step(
        self,
        step: int,
        val_loss: float,
        val_metrics: Dict[str, Any]
    ) -> None:
        """Log validation step information."""

        metrics = {
            'val_loss': val_loss,
            **{f'val_{k}': v for k, v in val_metrics.items()}
        }

        self.logger.log_metrics(metrics, step=step)

        # Check if this is the best model so far
        current_metric = val_metrics.get('accuracy', val_metrics.get('f1', -val_loss))

        if self.best_metric is None or current_metric > self.best_metric:
            self.best_metric = current_metric
            self.last_improvement_step = step
            self.logger.logger.info(f"New best model at step {step}! Metric: {current_metric:.4f}")

    def log_gradient_stats(self, model: torch.nn.Module, step: int) -> None:
        """Log gradient statistics for monitoring training health."""

        total_norm = 0.0
        param_count = 0
        grad_norms = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                grad_norms.append(param_norm.item())
                param_count += 1

        if param_count > 0:
            total_norm = total_norm ** (1. / 2)

            metrics = {
                'grad_norm_total': total_norm,
                'grad_norm_mean': np.mean(grad_norms),
                'grad_norm_std': np.std(grad_norms),
                'grad_norm_max': np.max(grad_norms),
                'params_with_grad': param_count,
            }

            self.logger.log_metrics(metrics, step=step)

            # Store for trend analysis
            self.gradient_accumulation.append((step, total_norm))

            # Keep only recent gradients
            if len(self.gradient_accumulation) > 100:
                self.gradient_accumulation = self.gradient_accumulation[-100:]

    def _check_training_health(self, step: int, loss: float, lr: float) -> None:
        """Check for common training issues."""

        # Check for NaN/Inf loss
        if not np.isfinite(loss):
            self.logger.logger.error(f"Training instability detected at step {step}: loss is {loss}")

        # Check for exploding loss
        if len(self.loss_history) > 5:
            recent_losses = [l for _, l in self.loss_history[-5:]]
            if loss > 10 * np.mean(recent_losses[:-1]):
                self.logger.logger.warning(f"Potential exploding loss at step {step}: {loss:.4f}")

        # Check for plateau
        if step - self.last_improvement_step > self.plateau_patience:
            self.logger.logger.warning(
                f"No improvement for {step - self.last_improvement_step} steps. "
                f"Consider reducing learning rate or early stopping."
            )

        # Check gradient accumulation trends
        if len(self.gradient_accumulation) > 20:
            recent_grads = [g for _, g in self.gradient_accumulation[-20:]]
            if np.mean(recent_grads) < 1e-8:
                self.logger.logger.warning("Very small gradients detected - possible vanishing gradient problem")
            elif np.mean(recent_grads) > 100:
                self.logger.logger.warning("Large gradients detected - possible exploding gradient problem")

    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        if not self.loss_history:
            return {}

        losses = [l for _, l in self.loss_history]

        summary = {
            'total_steps': len(self.loss_history),
            'final_loss': losses[-1],
            'best_loss': min(losses),
            'loss_reduction': losses[0] - losses[-1] if len(losses) > 1 else 0,
            'best_metric': self.best_metric,
            'last_improvement_step': self.last_improvement_step,
        }

        if self.gradient_accumulation:
            grad_norms = [g for _, g in self.gradient_accumulation]
            summary.update({
                'avg_grad_norm': np.mean(grad_norms),
                'max_grad_norm': max(grad_norms),
                'final_grad_norm': grad_norms[-1],
            })

        return summary


class ResourceAlert:
    """Alert system for resource usage thresholds."""

    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.alerts_sent = set()

    def check_memory_usage(self, threshold_percent: float = 90.0) -> None:
        """Check memory usage and alert if threshold exceeded."""
        if not torch.cuda.is_available():
            return

        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        allocated = torch.cuda.memory_allocated(device)
        usage_percent = (allocated / props.total_memory) * 100

        alert_key = f"memory_usage_{threshold_percent}"

        if usage_percent > threshold_percent and alert_key not in self.alerts_sent:
            self.logger.logger.warning(
                f"High GPU memory usage: {usage_percent:.1f}% "
                f"({allocated / 1e9:.2f}GB / {props.total_memory / 1e9:.2f}GB)"
            )
            self.alerts_sent.add(alert_key)

        # Reset alert if usage drops significantly
        elif usage_percent < threshold_percent - 10 and alert_key in self.alerts_sent:
            self.alerts_sent.remove(alert_key)

    def check_disk_space(self, threshold_gb: float = 5.0) -> None:
        """Check disk space and alert if running low."""
        try:
            import psutil
            free_space_gb = psutil.disk_usage('/').free / 1e9

            alert_key = f"disk_space_{threshold_gb}"

            if free_space_gb < threshold_gb and alert_key not in self.alerts_sent:
                self.logger.logger.error(
                    f"Low disk space: {free_space_gb:.2f}GB remaining"
                )
                self.alerts_sent.add(alert_key)

            elif free_space_gb > threshold_gb + 5 and alert_key in self.alerts_sent:
                self.alerts_sent.remove(alert_key)

        except ImportError:
            pass