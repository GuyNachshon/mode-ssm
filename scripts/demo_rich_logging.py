#!/usr/bin/env python3
"""
Demo script showcasing Rich logging capabilities for MODE-SSM.
Run this to see beautiful console output in action.
"""

import time
import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from mode_ssm.utils.rich_logging import (
    RichLogger, RichProgressTracker, RichMetricsDisplay, create_model_summary_table
)
from mode_ssm.utils.monitoring import SystemMonitor
from omegaconf import DictConfig


def demo_rich_logging():
    """Demonstrate Rich logging features."""

    # Initialize Rich logger
    logger = RichLogger("MODE-SSM Demo")

    # Demo system info
    system_info = SystemMonitor.get_system_info()
    logger.log_system_info(system_info)

    # Demo config display
    demo_config = DictConfig({
        'model': {
            'd_model': 512,
            'encoder': {'n_layers': 8},
            'preprocessor': {'num_channels': 256}
        },
        'training': {
            'batch_size': 32,
            'optimizer': {'lr': 0.0002},
            'mixed_precision': True
        },
        'data': {
            'train_path': 'data/train.h5',
            'val_path': 'data/val.h5',
            'max_sequence_ms': 3000
        }
    })

    logger.log_config_summary(demo_config)

    # Demo training stages
    demo_stages = [
        {
            'name': 'CTC Warmup',
            'epochs': 8,
            'loss_weights': {'ctc': 1.0, 'rnnt': 0.0, 'mode': 0.0}
        },
        {
            'name': 'Joint Training',
            'epochs': 12,
            'loss_weights': {'ctc': 0.4, 'rnnt': 0.6, 'mode': 0.0}
        },
        {
            'name': 'Mode Classification',
            'epochs': 8,
            'loss_weights': {'ctc': 0.4, 'rnnt': 0.6, 'mode': 0.1}
        },
        {
            'name': 'Denoising Bridge',
            'epochs': 6,
            'loss_weights': {'ctc': 0.3, 'rnnt': 0.6, 'mode': 0.1, 'denoise': 0.05}
        }
    ]

    logger.log_training_stages(demo_stages)

    # Demo different log types
    logger.info("🔍 This is an info message")
    logger.success("✅ This is a success message")
    logger.warning("⚠️  This is a warning message")
    logger.error("❌ This is an error message (demo only!)")

    # Demo progress tracking
    logger.info("📊 Demonstrating progress tracking...")

    total_steps = 50
    with RichProgressTracker(total_steps, "Training Demo") as progress:
        for step in range(total_steps):
            # Simulate training step
            time.sleep(0.05)

            # Update with fake metrics
            metrics = {
                'loss': 2.5 - (step * 0.03),
                'accuracy': min(0.95, step * 0.02),
                'lr': 0.0002 * (0.99 ** step)
            }

            progress.update(step + 1, **metrics)

    # Demo model summary (create a simple model)
    if torch.cuda.is_available() or True:  # Demo even without CUDA
        logger.info("🏗️  Demonstrating model summary...")

        # Create a simple demo model
        class DemoModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(1000, 512)
                self.transformer = torch.nn.TransformerEncoder(
                    torch.nn.TransformerEncoderLayer(512, 8, 2048, dropout=0.1),
                    num_layers=6
                )
                self.classifier = torch.nn.Linear(512, 40)

        demo_model = DemoModel()
        model_table = create_model_summary_table(demo_model)
        logger.console.print(model_table)

    logger.success("🎉 Rich logging demo completed!")


if __name__ == "__main__":
    demo_rich_logging()