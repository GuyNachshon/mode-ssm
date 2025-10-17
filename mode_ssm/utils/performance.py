"""
Performance optimization utilities for MODE-SSM.
Includes profiling, memory optimization, and performance monitoring.
"""

import functools
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig


class PerformanceProfiler:
    """Profile and monitor performance of operations."""

    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
        self.call_counts = {}

    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function execution time and memory usage."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"

            # Initialize tracking
            if func_name not in self.timings:
                self.timings[func_name] = []
                self.memory_usage[func_name] = []
                self.call_counts[func_name] = 0

            # Memory before
            memory_before = 0
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated()

            # Time execution
            start_time = time.perf_counter()
            result = func(*args, **kwargs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            # Memory after
            memory_after = 0
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated()

            # Record metrics
            execution_time = end_time - start_time
            memory_delta = memory_after - memory_before

            self.timings[func_name].append(execution_time)
            self.memory_usage[func_name].append(memory_delta)
            self.call_counts[func_name] += 1

            return result
        return wrapper

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all profiled functions."""
        stats = {}

        for func_name in self.timings:
            timings = self.timings[func_name]
            memory_usage = self.memory_usage[func_name]

            if not timings:
                continue

            stats[func_name] = {
                'calls': self.call_counts[func_name],
                'total_time': sum(timings),
                'avg_time': np.mean(timings),
                'min_time': min(timings),
                'max_time': max(timings),
                'std_time': np.std(timings),
                'avg_memory_mb': np.mean(memory_usage) / 1024 / 1024 if memory_usage else 0,
                'max_memory_mb': max(memory_usage) / 1024 / 1024 if memory_usage else 0,
            }

        return stats

    def print_stats(self) -> None:
        """Print formatted performance statistics."""
        stats = self.get_stats()

        print("\n" + "="*80)
        print("PERFORMANCE PROFILING RESULTS")
        print("="*80)

        # Sort by total time
        sorted_funcs = sorted(
            stats.items(),
            key=lambda x: x[1]['total_time'],
            reverse=True
        )

        print(f"{'Function':<40} {'Calls':<8} {'Total(s)':<10} {'Avg(ms)':<10} {'Memory(MB)':<12}")
        print("-" * 80)

        for func_name, func_stats in sorted_funcs:
            short_name = func_name.split('.')[-1][:35]
            print(
                f"{short_name:<40} "
                f"{func_stats['calls']:<8} "
                f"{func_stats['total_time']:<10.3f} "
                f"{func_stats['avg_time']*1000:<10.1f} "
                f"{func_stats['avg_memory_mb']:<12.1f}"
            )

    def reset(self) -> None:
        """Reset all profiling data."""
        self.timings.clear()
        self.memory_usage.clear()
        self.call_counts.clear()


class MemoryOptimizer:
    """Memory optimization utilities and strategies."""

    @staticmethod
    def enable_memory_efficient_attention(model: nn.Module) -> None:
        """Enable memory efficient attention if available."""
        try:
            # Try to enable Flash Attention or similar optimizations
            for module in model.modules():
                if hasattr(module, 'enable_flash_attention'):
                    module.enable_flash_attention()
                elif hasattr(module, 'set_memory_efficient'):
                    module.set_memory_efficient(True)
        except Exception:
            pass  # Not all models support this

    @staticmethod
    def optimize_model_memory(model: nn.Module, config: DictConfig) -> nn.Module:
        """Apply comprehensive memory optimizations to model."""

        # Enable gradient checkpointing if specified
        if config.get('gradient_checkpointing', False):
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            else:
                # Manual gradient checkpointing for custom modules
                for module in model.modules():
                    if hasattr(module, 'use_checkpoint'):
                        module.use_checkpoint = True

        # Enable memory efficient attention
        MemoryOptimizer.enable_memory_efficient_attention(model)

        # Convert to half precision if specified
        if config.get('mixed_precision', False):
            # Keep certain layers in fp32 for numerical stability
            for name, module in model.named_modules():
                if any(x in name.lower() for x in ['norm', 'embedding', 'head']):
                    continue  # Keep these in fp32
                if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                    continue  # Keep normalization in fp32

                # Convert to fp16
                if hasattr(module, 'half'):
                    module.half()

        return model

    @staticmethod
    @contextmanager
    def temporary_memory_cleanup():
        """Context manager for temporary memory cleanup."""
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def get_memory_summary() -> Dict[str, Any]:
        """Get comprehensive memory usage summary."""
        summary = {
            'timestamp': time.time(),
            'cuda_available': torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)

            summary.update({
                'device_name': props.name,
                'total_memory_gb': props.total_memory / 1e9,
                'allocated_memory_gb': torch.cuda.memory_allocated(device) / 1e9,
                'reserved_memory_gb': torch.cuda.memory_reserved(device) / 1e9,
                'max_allocated_gb': torch.cuda.max_memory_allocated(device) / 1e9,
                'memory_utilization': torch.cuda.memory_allocated(device) / props.total_memory,
            })

        # System memory if available
        try:
            import psutil
            memory = psutil.virtual_memory()
            summary.update({
                'system_memory_total_gb': memory.total / 1e9,
                'system_memory_available_gb': memory.available / 1e9,
                'system_memory_used_gb': memory.used / 1e9,
                'system_memory_percent': memory.percent,
            })
        except ImportError:
            pass

        return summary

    @staticmethod
    def optimize_dataloader_memory(config: DictConfig) -> Dict[str, Any]:
        """Optimize DataLoader settings for memory efficiency."""
        optimized_config = {}

        # Reduce batch size if memory constrained
        if torch.cuda.is_available():
            available_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

            if available_memory_gb < 8:  # Low memory GPU
                optimized_config['batch_size'] = min(config.get('batch_size', 16), 8)
                optimized_config['num_workers'] = min(config.get('num_workers', 4), 2)
            elif available_memory_gb < 16:  # Medium memory GPU
                optimized_config['batch_size'] = min(config.get('batch_size', 32), 16)
                optimized_config['num_workers'] = min(config.get('num_workers', 8), 4)
            else:  # High memory GPU
                optimized_config['batch_size'] = config.get('batch_size', 32)
                optimized_config['num_workers'] = config.get('num_workers', 8)

        # Always disable caching on memory-constrained systems
        optimized_config['cache_data'] = False
        optimized_config['pin_memory'] = torch.cuda.is_available()

        return optimized_config


class ModelCompiler:
    """Compile models for optimal performance."""

    @staticmethod
    def compile_model(model: nn.Module, config: DictConfig) -> nn.Module:
        """Compile model using available optimization techniques."""

        # Try torch.compile if available and enabled
        if config.get('torch_compile', False):
            try:
                if hasattr(torch, 'compile'):
                    print("Compiling model with torch.compile...")
                    # Use conservative settings for stability
                    model = torch.compile(
                        model,
                        mode='reduce-overhead',  # Balanced performance/compilation time
                        dynamic=True,  # Handle dynamic shapes
                    )
                    print("Model compilation successful!")
                else:
                    print("torch.compile not available, skipping compilation")
            except Exception as e:
                print(f"Model compilation failed, continuing without compilation: {e}")

        return model

    @staticmethod
    def optimize_for_inference(model: nn.Module) -> nn.Module:
        """Optimize model specifically for inference."""
        model.eval()

        # Fuse operations if possible
        try:
            # Fuse conv-bn, linear-relu, etc.
            torch.jit.optimize_for_inference(torch.jit.script(model))
        except Exception:
            pass  # Not all models support this

        # Set inference mode optimizations
        for module in model.modules():
            # Disable dropout
            if isinstance(module, nn.Dropout):
                module.p = 0.0

            # Set batch norm to eval mode permanently
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.track_running_stats = False

        return model


class BatchSizeOptimizer:
    """Automatically find optimal batch size for given memory constraints."""

    @staticmethod
    def find_max_batch_size(
        model: nn.Module,
        sample_input: torch.Tensor,
        max_memory_gb: float = None,
        start_batch_size: int = 1,
        max_batch_size: int = 512
    ) -> int:
        """Find maximum batch size that fits in memory."""

        if not torch.cuda.is_available():
            return start_batch_size

        if max_memory_gb is None:
            # Use 90% of available memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            max_memory_gb = (total_memory * 0.9) / 1e9

        model.eval()
        device = next(model.parameters()).device

        # Binary search for optimal batch size
        min_batch = start_batch_size
        max_batch = max_batch_size
        optimal_batch = min_batch

        while min_batch <= max_batch:
            test_batch = (min_batch + max_batch) // 2

            try:
                # Create test batch
                batch_input = sample_input.repeat(test_batch, *([1] * (sample_input.dim() - 1)))
                batch_input = batch_input.to(device)

                # Clear cache and test forward pass
                torch.cuda.empty_cache()

                with torch.no_grad():
                    _ = model(batch_input)

                # Check memory usage
                memory_used_gb = torch.cuda.memory_allocated(device) / 1e9

                if memory_used_gb <= max_memory_gb:
                    optimal_batch = test_batch
                    min_batch = test_batch + 1
                else:
                    max_batch = test_batch - 1

                # Cleanup
                del batch_input
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    max_batch = test_batch - 1
                    torch.cuda.empty_cache()
                else:
                    raise e

        return optimal_batch


def benchmark_function(func: Callable, *args, num_runs: int = 10, warmup_runs: int = 3) -> Dict[str, float]:
    """Benchmark function performance with multiple runs."""

    # Warmup runs
    for _ in range(warmup_runs):
        try:
            func(*args)
        except Exception:
            pass

    # Benchmark runs
    times = []
    memory_usage = []

    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated()

        start_time = time.perf_counter()
        result = func(*args)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated()
            memory_usage.append((memory_after - memory_before) / 1024 / 1024)  # MB

        times.append(end_time - start_time)

    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'mean_memory_mb': np.mean(memory_usage) if memory_usage else 0,
        'max_memory_mb': np.max(memory_usage) if memory_usage else 0,
    }


# Global profiler instance
profiler = PerformanceProfiler()