"""
Comprehensive error handling and validation utilities for MODE-SSM.
Provides custom exceptions, validation decorators, and error recovery mechanisms.
"""

import functools
import logging
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import torch
import numpy as np
from omegaconf import DictConfig


class MODESSMError(Exception):
    """Base exception for all MODE-SSM errors."""
    pass


class ConfigurationError(MODESSMError):
    """Raised when configuration is invalid or incomplete."""
    pass


class DataError(MODESSMError):
    """Raised when data validation fails."""
    pass


class ModelError(MODESSMError):
    """Raised when model operations fail."""
    pass


class TrainingError(MODESSMError):
    """Raised when training operations fail."""
    pass


class InferenceError(MODESSMError):
    """Raised when inference operations fail."""
    pass


class ResourceError(MODESSMError):
    """Raised when resource allocation fails (memory, GPU, etc.)."""
    pass


class ErrorHandler:
    """Centralized error handling and recovery mechanisms."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts = {}

    def handle_error(
        self,
        error: Exception,
        context: str = "",
        reraise: bool = True,
        max_retries: int = 0
    ) -> None:
        """Handle errors with logging and optional retry logic."""
        error_key = f"{type(error).__name__}:{context}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

        self.logger.error(
            f"Error in {context}: {str(error)}\n"
            f"Error count for this type: {self.error_counts[error_key]}\n"
            f"Traceback: {traceback.format_exc()}"
        )

        if reraise:
            raise error

    def with_error_handling(
        self,
        context: str = "",
        max_retries: int = 0,
        exceptions: tuple = (Exception,)
    ):
        """Decorator for automatic error handling."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                retries = 0
                while retries <= max_retries:
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        if retries == max_retries:
                            self.handle_error(e, context or func.__name__)
                        else:
                            self.logger.warning(
                                f"Retry {retries + 1}/{max_retries} for {context or func.__name__}: {str(e)}"
                            )
                            retries += 1
                return None
            return wrapper
        return decorator


class ConfigValidator:
    """Validates configuration objects and provides detailed error messages."""

    @staticmethod
    def validate_model_config(config: DictConfig) -> None:
        """Validate model configuration."""
        required_fields = ['d_model', 'num_layers', 'num_heads']

        if 'model' not in config:
            raise ConfigurationError("Model configuration is missing")

        model_config = config.model

        for field in required_fields:
            if field not in model_config:
                raise ConfigurationError(f"Required model field '{field}' is missing")

        # Validate dimensions
        if model_config.d_model <= 0:
            raise ConfigurationError("d_model must be positive")

        if model_config.num_layers <= 0:
            raise ConfigurationError("num_layers must be positive")

        if model_config.num_heads <= 0:
            raise ConfigurationError("num_heads must be positive")

        if model_config.d_model % model_config.num_heads != 0:
            raise ConfigurationError("d_model must be divisible by num_heads")

    @staticmethod
    def validate_training_config(config: DictConfig) -> None:
        """Validate training configuration."""
        if 'training' not in config:
            raise ConfigurationError("Training configuration is missing")

        training_config = config.training

        # Validate stages
        if 'stages' not in training_config:
            raise ConfigurationError("Training stages configuration is missing")

        stages = training_config.stages
        total_weight_stages = []

        for stage_name, stage_config in stages.items():
            if not stage_config.get('enabled', False):
                continue

            # Check required fields
            required_stage_fields = ['epochs', 'ctc_weight', 'rnnt_weight', 'mode_weight']
            for field in required_stage_fields:
                if field not in stage_config:
                    raise ConfigurationError(f"Stage '{stage_name}' missing field '{field}'")

            # Validate epochs
            if stage_config.epochs <= 0:
                raise ConfigurationError(f"Stage '{stage_name}' epochs must be positive")

            # Validate weights sum to reasonable value
            weights = [
                stage_config.ctc_weight,
                stage_config.rnnt_weight,
                stage_config.mode_weight,
                stage_config.get('flow_weight', 0.0)
            ]

            if sum(weights) <= 0:
                raise ConfigurationError(f"Stage '{stage_name}' total loss weights must be positive")

            total_weight_stages.append((stage_name, sum(weights)))

    @staticmethod
    def validate_data_config(config: DictConfig) -> None:
        """Validate data configuration."""
        if 'data' not in config:
            raise ConfigurationError("Data configuration is missing")

        data_config = config.data

        # Check paths exist
        if 'train_path' in data_config and data_config.train_path:
            train_path = Path(data_config.train_path)
            if not train_path.exists():
                raise ConfigurationError(f"Training data path does not exist: {train_path}")

        if 'val_path' in data_config and data_config.val_path:
            val_path = Path(data_config.val_path)
            if not val_path.exists():
                raise ConfigurationError(f"Validation data path does not exist: {val_path}")

        # Validate batch size
        if data_config.get('batch_size', 0) <= 0:
            raise ConfigurationError("Batch size must be positive")

        # Validate sequence lengths
        min_seq = data_config.get('min_sequence_ms', 0)
        max_seq = data_config.get('max_sequence_ms', float('inf'))

        if min_seq < 0:
            raise ConfigurationError("min_sequence_ms must be non-negative")

        if max_seq <= min_seq:
            raise ConfigurationError("max_sequence_ms must be greater than min_sequence_ms")


class TensorValidator:
    """Validates tensor shapes, values, and device placement."""

    @staticmethod
    def validate_tensor_shape(
        tensor: torch.Tensor,
        expected_shape: Union[tuple, list],
        name: str = "tensor"
    ) -> None:
        """Validate tensor has expected shape."""
        if tensor.shape != tuple(expected_shape):
            raise DataError(
                f"{name} shape mismatch: expected {expected_shape}, got {tensor.shape}"
            )

    @staticmethod
    def validate_tensor_range(
        tensor: torch.Tensor,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        name: str = "tensor"
    ) -> None:
        """Validate tensor values are within expected range."""
        if min_val is not None and tensor.min().item() < min_val:
            raise DataError(f"{name} has values below minimum {min_val}")

        if max_val is not None and tensor.max().item() > max_val:
            raise DataError(f"{name} has values above maximum {max_val}")

    @staticmethod
    def validate_no_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> None:
        """Validate tensor contains no NaN or Inf values."""
        if torch.isnan(tensor).any():
            raise DataError(f"{name} contains NaN values")

        if torch.isinf(tensor).any():
            raise DataError(f"{name} contains Inf values")

    @staticmethod
    def validate_tensor_device(
        tensor: torch.Tensor,
        expected_device: Union[str, torch.device],
        name: str = "tensor"
    ) -> None:
        """Validate tensor is on expected device."""
        if isinstance(expected_device, str):
            expected_device = torch.device(expected_device)

        if tensor.device != expected_device:
            raise DataError(
                f"{name} device mismatch: expected {expected_device}, got {tensor.device}"
            )


class ResourceMonitor:
    """Monitor system resources and raise errors when limits exceeded."""

    @staticmethod
    def check_gpu_memory(threshold_gb: float = 0.5) -> None:
        """Check if sufficient GPU memory is available."""
        if not torch.cuda.is_available():
            return

        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        allocated_memory = torch.cuda.memory_allocated(device) / 1e9
        available_memory = total_memory - allocated_memory

        if available_memory < threshold_gb:
            raise ResourceError(
                f"Insufficient GPU memory: {available_memory:.2f}GB available, "
                f"{threshold_gb:.2f}GB required"
            )

    @staticmethod
    def check_system_memory(threshold_gb: float = 2.0) -> None:
        """Check if sufficient system memory is available."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / 1e9

            if available_gb < threshold_gb:
                raise ResourceError(
                    f"Insufficient system memory: {available_gb:.2f}GB available, "
                    f"{threshold_gb:.2f}GB required"
                )
        except ImportError:
            # psutil not available, skip check
            pass


def validate_config(config: DictConfig) -> None:
    """Comprehensive configuration validation."""
    try:
        ConfigValidator.validate_model_config(config)
        ConfigValidator.validate_training_config(config)
        ConfigValidator.validate_data_config(config)
    except Exception as e:
        raise ConfigurationError(f"Configuration validation failed: {str(e)}")


def safe_tensor_operation(func: Callable) -> Callable:
    """Decorator for safe tensor operations with validation."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Pre-validation
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    TensorValidator.validate_no_nan_inf(arg)

            result = func(*args, **kwargs)

            # Post-validation
            if isinstance(result, torch.Tensor):
                TensorValidator.validate_no_nan_inf(result)

            return result

        except Exception as e:
            raise ModelError(f"Tensor operation failed in {func.__name__}: {str(e)}")

    return wrapper


def memory_efficient_operation(func: Callable) -> Callable:
    """Decorator for memory-efficient operations with monitoring."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check memory before operation
        ResourceMonitor.check_gpu_memory(threshold_gb=1.0)

        try:
            result = func(*args, **kwargs)

            # Clear cache after operation if needed
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                if allocated > 8.0:  # Clear cache if using > 8GB
                    torch.cuda.empty_cache()

            return result

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                raise ResourceError(f"GPU out of memory in {func.__name__}: {str(e)}")
            raise

    return wrapper