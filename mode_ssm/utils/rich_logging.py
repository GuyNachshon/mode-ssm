"""
Rich-enhanced logging and progress tracking for MODE-SSM.
Provides beautiful console output with progress bars, tables, and panels.
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np
from omegaconf import DictConfig

from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    SpinnerColumn
)
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.layout import Layout
from rich.live import Live
from rich.tree import Tree
from rich.status import Status
from rich import box
from rich.align import Align


class RichLogger:
    """Enhanced logger with Rich formatting for beautiful console output."""

    def __init__(self, name: str = "MODE-SSM", log_file: Optional[Path] = None):
        self.name = name
        self.console = Console()
        self.log_file = log_file
        self.start_time = time.time()

        # Print header
        self._print_header()

    def _print_header(self):
        """Print a beautiful header for MODE-SSM."""
        header_text = Text()
        header_text.append("MODE-SSM", style="bold blue")
        header_text.append(" :: Multi-Objective Decoder Enhancement", style="dim")

        header_panel = Panel(
            Align.center(header_text),
            box=box.DOUBLE,
            style="blue",
            title="🧠 Neural Decoder",
            title_align="left"
        )
        self.console.print(header_panel)
        self.console.print()

    def info(self, message: str, **kwargs):
        """Log info message with Rich formatting."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"[dim]{timestamp}[/dim] [blue]ℹ[/blue] {message}", **kwargs)

    def success(self, message: str, **kwargs):
        """Log success message with Rich formatting."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"[dim]{timestamp}[/dim] [green]✓[/green] {message}", **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with Rich formatting."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"[dim]{timestamp}[/dim] [yellow]⚠[/yellow] {message}", **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message with Rich formatting."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"[dim]{timestamp}[/dim] [red]✗[/red] {message}", **kwargs)

    def log_system_info(self, info: Dict[str, Any]):
        """Display system information in a beautiful table."""
        table = Table(title="🖥️  System Information", box=box.ROUNDED)
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Details", style="white")

        # Python and PyTorch info
        table.add_row("Python", info.get('python_version', 'Unknown'))
        table.add_row("PyTorch", info.get('pytorch_version', 'Unknown'))

        # CUDA info
        if info.get('cuda_available', False):
            table.add_row("CUDA", f"✓ Version {info.get('cuda_version', 'Unknown')}")
            table.add_row("GPU Count", str(info.get('gpu_count', 0)))
            for i in range(info.get('gpu_count', 0)):
                gpu_name = info.get(f'gpu_{i}_name', 'Unknown')
                gpu_memory = info.get(f'gpu_{i}_memory_gb', 0)
                table.add_row(f"GPU {i}", f"{gpu_name} ({gpu_memory:.1f}GB)")
        else:
            table.add_row("CUDA", "[red]✗ Not available[/red]")

        # System resources
        if 'cpu_count' in info:
            table.add_row("CPU Cores", str(info['cpu_count']))
        if 'memory_total_gb' in info:
            table.add_row("System RAM", f"{info['memory_total_gb']:.1f}GB")
        if 'disk_free_gb' in info:
            table.add_row("Disk Space", f"{info['disk_free_gb']:.1f}GB free")

        self.console.print(table)
        self.console.print()

    def log_config_summary(self, config: DictConfig):
        """Display configuration summary in panels."""
        columns = []

        # Model configuration
        if 'model' in config:
            model_info = Table(box=None, show_header=False)
            model_info.add_column("Key", style="cyan")
            model_info.add_column("Value", style="white")

            model_config = config.model
            model_info.add_row("d_model", str(model_config.get('d_model', 'N/A')))
            model_info.add_row("layers", str(model_config.get('encoder', {}).get('n_layers', 'N/A')))
            model_info.add_row("channels", str(model_config.get('preprocessor', {}).get('num_channels', 'N/A')))

            columns.append(Panel(model_info, title="🏗️  Model", style="blue"))

        # Training configuration
        if 'training' in config:
            training_info = Table(box=None, show_header=False)
            training_info.add_column("Key", style="cyan")
            training_info.add_column("Value", style="white")

            training_config = config.training
            training_info.add_row("batch_size", str(training_config.get('batch_size', 'N/A')))
            training_info.add_row("learning_rate", str(training_config.get('optimizer', {}).get('lr', 'N/A')))
            training_info.add_row("mixed_precision", str(training_config.get('mixed_precision', 'N/A')))

            columns.append(Panel(training_info, title="🎯  Training", style="green"))

        # Data configuration
        if 'data' in config:
            data_info = Table(box=None, show_header=False)
            data_info.add_column("Key", style="cyan")
            data_info.add_column("Value", style="white")

            data_config = config.data
            train_path = Path(data_config.get('train_path', '')).name or 'N/A'
            val_path = Path(data_config.get('val_path', '')).name or 'N/A'
            data_info.add_row("train_data", train_path)
            data_info.add_row("val_data", val_path)
            data_info.add_row("max_seq_ms", str(data_config.get('max_sequence_ms', 'N/A')))

            columns.append(Panel(data_info, title="📊  Data", style="yellow"))

        if columns:
            self.console.print(Columns(columns))
            self.console.print()

    def log_training_stages(self, stages: List[Dict[str, Any]]):
        """Display training curriculum as a tree."""
        tree = Tree("🎓 Training Curriculum")

        for i, stage in enumerate(stages, 1):
            stage_name = stage.get('name', f'Stage {i}')
            epochs = stage.get('epochs', 'N/A')

            stage_node = tree.add(f"[bold]{stage_name}[/bold] ({epochs} epochs)")

            # Add loss weights
            if 'loss_weights' in stage:
                weights = stage['loss_weights']
                for loss_type, weight in weights.items():
                    if weight > 0:
                        stage_node.add(f"{loss_type}: {weight}")

        self.console.print(tree)
        self.console.print()


class RichProgressTracker:
    """Rich progress tracking for training loops."""

    def __init__(self, total_steps: int, description: str = "Training"):
        self.total_steps = total_steps
        self.description = description
        self.current_step = 0

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=Console(),
        )

        self.task_id = None
        self.start_time = time.time()

    def __enter__(self):
        self.progress.start()
        self.task_id = self.progress.add_task(self.description, total=self.total_steps)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.stop()

    def update(self, step: int, **kwargs):
        """Update progress with current step and optional metrics."""
        self.current_step = step

        # Build description with metrics
        desc_parts = [self.description]
        if kwargs:
            metric_parts = []
            for key, value in kwargs.items():
                if isinstance(value, float):
                    metric_parts.append(f"{key}={value:.4f}")
                else:
                    metric_parts.append(f"{key}={value}")
            if metric_parts:
                desc_parts.append(" • ".join(metric_parts))

        description = " | ".join(desc_parts)
        self.progress.update(self.task_id, completed=step, description=description)

    def advance(self, amount: int = 1, **kwargs):
        """Advance progress by amount."""
        self.update(self.current_step + amount, **kwargs)


class RichMetricsDisplay:
    """Live updating metrics display."""

    def __init__(self):
        self.console = Console()
        self.metrics_history = {}
        self.layout = Layout()

        # Setup layout
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )

    def create_metrics_table(self, metrics: Dict[str, Any]) -> Table:
        """Create a table of current metrics."""
        table = Table(title="📊 Training Metrics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Current", style="white")
        table.add_column("Best", style="green")
        table.add_column("Trend", style="yellow")

        for metric_name, current_value in metrics.items():
            if metric_name in self.metrics_history:
                history = self.metrics_history[metric_name]
                best_value = min(history) if 'loss' in metric_name.lower() else max(history)

                # Simple trend indicator
                if len(history) >= 2:
                    if history[-1] < history[-2]:
                        trend = "↓" if 'loss' in metric_name.lower() else "↑"
                    elif history[-1] > history[-2]:
                        trend = "↑" if 'loss' in metric_name.lower() else "↓"
                    else:
                        trend = "→"
                else:
                    trend = "—"
            else:
                best_value = current_value
                trend = "—"
                self.metrics_history[metric_name] = []

            self.metrics_history[metric_name].append(current_value)

            # Format values
            if isinstance(current_value, float):
                current_str = f"{current_value:.4f}"
                best_str = f"{best_value:.4f}"
            else:
                current_str = str(current_value)
                best_str = str(best_value)

            table.add_row(metric_name, current_str, best_str, trend)

        return table

    def update_display(self,
                      step: int,
                      epoch: int,
                      metrics: Dict[str, Any],
                      elapsed_time: float):
        """Update the live display."""

        # Header
        header_text = Text()
        header_text.append(f"Step {step:,}", style="bold blue")
        header_text.append(f" • Epoch {epoch}", style="dim")
        header_text.append(f" • {elapsed_time:.1f}s", style="dim")

        self.layout["header"].update(Panel(header_text, style="blue"))

        # Body - metrics table
        self.layout["body"].update(self.create_metrics_table(metrics))

        # Footer - system info
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_text = f"GPU Memory: {gpu_memory:.1f}GB"
        else:
            gpu_text = "GPU: Not available"

        footer_text = Text(gpu_text, style="dim")
        self.layout["footer"].update(Panel(footer_text, style="dim"))

        return self.layout


def format_duration(seconds: float) -> str:
    """Format duration in a human-readable way."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_number(num: Union[int, float]) -> str:
    """Format large numbers with appropriate units."""
    if isinstance(num, float):
        if abs(num) < 1:
            return f"{num:.4f}"
        elif abs(num) < 1000:
            return f"{num:.2f}"
        else:
            return f"{num:,.0f}"
    else:
        if abs(num) >= 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif abs(num) >= 1_000:
            return f"{num/1_000:.1f}K"
        else:
            return f"{num:,}"


def create_model_summary_table(model: torch.nn.Module) -> Table:
    """Create a table summarizing model architecture."""
    table = Table(title="🏗️  Model Architecture", box=box.ROUNDED)
    table.add_column("Component", style="cyan")
    table.add_column("Parameters", style="white")
    table.add_column("Shape", style="dim")

    total_params = 0
    trainable_params = 0

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            num_params = sum(p.numel() for p in module.parameters())
            num_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

            if num_params > 0:
                total_params += num_params
                trainable_params += num_trainable

                # Get representative parameter shape
                shapes = [tuple(p.shape) for p in module.parameters() if p.numel() > 0]
                shape_str = str(shapes[0]) if shapes else "—"

                table.add_row(
                    name.split('.')[-1],  # Just the module name
                    format_number(num_params),
                    shape_str
                )

    # Add totals
    table.add_section()
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{format_number(total_params)}[/bold]",
        f"[dim]{trainable_params/total_params*100:.1f}% trainable[/dim]"
    )

    return table