#!/usr/bin/env python3
"""
Model evaluation script for MODE-SSM.
Performs comprehensive evaluation on test/validation data with detailed metrics.
"""

import os
import sys
import argparse
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import numpy as np
import pandas as pd

# Replace tqdm with Rich progress
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.console import Console
from rich.table import Table

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets.brain2text import Brain2TextDataset, collate_batch
from datasets.phoneme_vocab import PhonemeVocabulary
from mode_ssm.models.mode_ssm_model import MODESSMModel, MODESSMConfig
from mode_ssm.checkpoint_manager import CheckpointManager
from mode_ssm.evaluation_metrics import EvaluationManager, EvaluationResults
from mode_ssm.models.tta_loop import TTALoop, TTAConfig


logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluator for MODE-SSM"""

    def __init__(self, config: DictConfig):
        """
        Initialize evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model and data
        self.model = None
        self.test_loader = None
        self.checkpoint_manager = None
        self.eval_manager = None
        self.phoneme_vocab = PhonemeVocabulary()

        # Results storage
        self.results = {
            'overall': {},
            'per_session': defaultdict(dict),
            'per_mode': defaultdict(dict),
            'per_length_bin': defaultdict(dict),
            'predictions': []
        }

        # TTA components
        self.tta_loop = None
        self.tta_enabled = getattr(config, 'tta', {}).get('enabled', False)

        logger.info(f"Using device: {self.device}")
        if self.tta_enabled:
            logger.info("Test-time adaptation enabled")

    def setup_model(self):
        """Load model and checkpoint"""
        logger.info("Loading model...")

        # Create model config
        model_config = MODESSMConfig(
            d_model=self.config.model.d_model,
            d_state=self.config.model.d_state,
            d_conv=self.config.model.d_conv,
            expand=self.config.model.expand,
            num_channels=self.config.model.num_channels,
            encoder_layers=self.config.model.encoder_layers,
            vocab_size=self.config.model.vocab_size
        )

        # Create model
        self.model = MODESSMModel(model_config).to(self.device)
        self.model.eval()

        # Load checkpoint
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.checkpoint.dir,
            monitor_metric='val_wer'
        )

        if self.config.checkpoint.use_best:
            checkpoint_path = self.checkpoint_manager.get_best_checkpoint()
        else:
            checkpoint_path = self.checkpoint_manager.get_latest_checkpoint()

        if checkpoint_path is None:
            raise ValueError("No checkpoint found!")

        metadata = self.checkpoint_manager.restore_model(
            model=self.model,
            checkpoint_path=checkpoint_path
        )

        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        if metadata:
            logger.info(f"  Epoch: {metadata.epoch}, Val WER: {metadata.val_wer:.3f}")

        # Setup TTA if enabled
        if self.tta_enabled:
            self._setup_tta()

    def setup_data(self):
        """Setup test data loader"""
        logger.info(f"Loading test data from {self.config.data.test_path}...")

        # Create dataset
        test_dataset = Brain2TextDataset(
            hdf5_path=self.config.data.test_path,
            min_sequence_ms=self.config.data.min_sequence_ms,
            max_sequence_ms=self.config.data.max_sequence_ms,
            missing_channel_threshold=self.config.data.missing_channel_threshold,
            cache_data=self.config.data.cache_data,
            transform=None,  # No augmentation for testing
            filter_quality=True
        )

        logger.info(f"Test dataset: {len(test_dataset)} samples")

        # Get dataset statistics
        stats = test_dataset.get_statistics()
        logger.info(f"Dataset statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value:.2f}")

        # Create data loader
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.evaluation.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
            collate_fn=collate_batch,
            persistent_workers=True if self.config.data.num_workers > 0 else False,
            drop_last=False
        )

        # Create evaluation manager
        self.eval_manager = EvaluationManager(
            vocab_size=self.config.model.vocab_size,
            blank_idx=0,
            silence_idx=39
        )

    def _setup_tta(self):
        """Setup test-time adaptation"""
        logger.info("Setting up test-time adaptation...")

        # Create TTA config from evaluation config
        tta_config = TTAConfig(
            session_adaptation_enabled=self.config.tta.get('session_adaptation_enabled', True),
            entropy_minimization_enabled=self.config.tta.get('entropy_minimization_enabled', True),
            adaptation_lr=self.config.tta.get('adaptation_lr', 0.001),
            adaptation_steps=self.config.tta.get('adaptation_steps', 3),
            statistics_momentum=self.config.tta.get('statistics_momentum', 0.9),
            entropy_threshold=self.config.tta.get('entropy_threshold', 2.0),
            entropy_weight=self.config.tta.get('entropy_weight', 0.5),
            min_samples_for_adaptation=self.config.tta.get('min_samples_for_adaptation', 5),
            adaptation_layers=self.config.tta.get('adaptation_layers', ['preprocessor']),
            confidence_threshold=self.config.tta.get('confidence_threshold', 0.8)
        )

        # Create TTA loop
        self.tta_loop = TTALoop(tta_config, self.model)

        logger.info("TTA setup complete")

    def evaluate(self):
        """Run comprehensive evaluation"""
        logger.info("Starting evaluation...")

        all_metrics = []
        all_predictions = []

        # Progress bar
        progress_bar = tqdm(
            self.test_loader,
            desc="Evaluating",
            disable=not self.config.evaluation.show_progress
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = self._batch_to_device(batch)

                # Apply TTA if enabled
                if self.tta_enabled and self.tta_loop is not None:
                    # Apply test-time adaptation
                    adapted_batch = self.tta_loop.adapt_multi_step(
                        batch,
                        num_steps=self.config.tta.get('adaptation_steps', 3)
                    )

                    # Use adapted features for evaluation
                    batch['neural_features'] = adapted_batch['neural_features']

                    # Store TTA statistics for analysis
                    if 'tta_stats' not in batch:
                        batch['tta_stats'] = []
                    batch['tta_stats'].append(adapted_batch['adaptation_stats'])

                # Get predictions using different decoding modes
                batch_results = self.evaluate_batch(batch)

                # Store results
                all_metrics.append(batch_results['metrics'])
                all_predictions.extend(batch_results['predictions'])

                # Update progress bar
                if batch_idx % 10 == 0:
                    current_wer = np.mean([m.wer for m in all_metrics[-10:] if m.wer > 0])
                    progress_bar.set_postfix({'WER': f'{current_wer:.3f}'})

        # Aggregate results
        self.aggregate_results(all_metrics, all_predictions)

        # Print summary
        self.print_summary()

        # Save results
        self.save_results()

    def evaluate_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Evaluate a single batch.

        Args:
            batch: Input batch

        Returns:
            Dictionary with metrics and predictions
        """
        batch_size = batch['neural_features'].shape[0]

        # Forward pass for all outputs
        with autocast(enabled=self.config.evaluation.use_amp):
            outputs = self.model(
                neural_features=batch['neural_features'],
                sequence_lengths=batch['sequence_lengths'],
                phoneme_labels=batch.get('phoneme_labels'),
                label_lengths=batch.get('label_lengths'),
                mode_labels=batch.get('mode_labels'),
                training_stage='joint',  # Use joint stage for full outputs
                return_all_outputs=True
            )

        # Evaluate with ground truth if available
        metrics = self.eval_manager.evaluate_batch(
            model_outputs=outputs,
            batch=batch,
            training_stage='joint'
        )

        # Get predictions using different strategies
        predictions = []

        for i in range(batch_size):
            # Get sample data
            sample_features = batch['neural_features'][i:i+1]
            sample_length = batch['sequence_lengths'][i:i+1]

            # Inference
            inference_results = self.model.inference(
                neural_features=sample_features,
                sequence_lengths=sample_length,
                decode_mode=self.config.evaluation.decode_mode,
                return_mode_predictions=True
            )

            # Create prediction record
            prediction = {
                'session_id': batch['session_ids'][i] if 'session_ids' in batch else f"session_{i}",
                'block_num': batch['block_nums'][i].item() if 'block_nums' in batch else 0,
                'trial_num': batch['trial_nums'][i].item() if 'trial_nums' in batch else i,
                'sequence_length': sample_length.item(),
                'mode_prediction': inference_results['mode_predictions'][0].item(),
                'mode_confidence': inference_results['mode_confidence'][0].item(),
            }

            # Add decoded phonemes
            if 'decoded_sequences' in inference_results:
                decoded_phonemes = inference_results['decoded_sequences'][0]
                prediction['predicted_phonemes'] = decoded_phonemes
                prediction['predicted_text'] = self.phoneme_vocab.phonemes_to_text(
                    [self.phoneme_vocab.logit_to_phoneme[idx] for idx in decoded_phonemes]
                )

            # Add ground truth if available
            if 'phoneme_labels' in batch and batch['phoneme_labels'] is not None:
                label_len = batch['label_lengths'][i].item()
                true_phonemes = batch['phoneme_labels'][i][:label_len].tolist()
                prediction['true_phonemes'] = true_phonemes
                prediction['true_text'] = self.phoneme_vocab.phonemes_to_text(
                    [self.phoneme_vocab.logit_to_phoneme[idx] for idx in true_phonemes if idx > 0]
                )

            if 'mode_labels' in batch and batch['mode_labels'] is not None:
                prediction['true_mode'] = batch['mode_labels'][i].item()

            predictions.append(prediction)

        return {
            'metrics': metrics,
            'predictions': predictions
        }

    def aggregate_results(
        self,
        all_metrics: List[EvaluationResults],
        all_predictions: List[Dict]
    ):
        """Aggregate evaluation results"""
        # Overall metrics
        aggregated = self.eval_manager.aggregate_results(all_metrics)

        self.results['overall'] = {
            'wer': aggregated.wer,
            'cer': aggregated.cer,
            'phoneme_accuracy': aggregated.phoneme_accuracy,
            'mode_accuracy': aggregated.mode_accuracy,
            'num_samples': aggregated.num_samples,
            'failed_decodings': aggregated.failed_decodings
        }

        # Per-session analysis
        session_predictions = defaultdict(list)
        for pred in all_predictions:
            session_predictions[pred['session_id']].append(pred)

        for session_id, preds in session_predictions.items():
            # Calculate session-specific WER if ground truth available
            if 'true_phonemes' in preds[0]:
                session_wers = []
                for pred in preds:
                    if 'predicted_phonemes' in pred and 'true_phonemes' in pred:
                        # Simple WER calculation
                        wer, _ = self.eval_manager.wer_calculator.calculate_wer(
                            [[self.phoneme_vocab.logit_to_phoneme[idx] for idx in pred['predicted_phonemes']]],
                            [[self.phoneme_vocab.logit_to_phoneme[idx] for idx in pred['true_phonemes']]]
                        )
                        session_wers.append(wer)

                self.results['per_session'][session_id] = {
                    'num_samples': len(preds),
                    'mean_wer': np.mean(session_wers) if session_wers else 0.0,
                    'std_wer': np.std(session_wers) if session_wers else 0.0
                }

        # Per-mode analysis
        mode_predictions = defaultdict(list)
        for pred in all_predictions:
            if 'mode_prediction' in pred:
                mode_predictions[pred['mode_prediction']].append(pred)

        mode_names = ['silent', 'vocalized']
        for mode_idx, preds in mode_predictions.items():
            mode_name = mode_names[mode_idx] if mode_idx < len(mode_names) else f"mode_{mode_idx}"

            # Calculate mode-specific metrics
            mode_correct = 0
            mode_wers = []

            for pred in preds:
                if 'true_mode' in pred:
                    if pred['mode_prediction'] == pred['true_mode']:
                        mode_correct += 1

                if 'predicted_phonemes' in pred and 'true_phonemes' in pred:
                    wer, _ = self.eval_manager.wer_calculator.calculate_wer(
                        [[self.phoneme_vocab.logit_to_phoneme[idx] for idx in pred['predicted_phonemes']]],
                        [[self.phoneme_vocab.logit_to_phoneme[idx] for idx in pred['true_phonemes']]]
                    )
                    mode_wers.append(wer)

            self.results['per_mode'][mode_name] = {
                'num_samples': len(preds),
                'accuracy': mode_correct / len(preds) if preds and 'true_mode' in preds[0] else 0.0,
                'mean_wer': np.mean(mode_wers) if mode_wers else 0.0,
                'mean_confidence': np.mean([p['mode_confidence'] for p in preds])
            }

        # Length-based analysis
        length_bins = [(0, 500), (500, 1000), (1000, 2000), (2000, float('inf'))]
        for bin_min, bin_max in length_bins:
            bin_name = f"{bin_min}-{bin_max if bin_max != float('inf') else 'inf'}"
            bin_preds = [
                p for p in all_predictions
                if bin_min <= p['sequence_length'] < bin_max
            ]

            if bin_preds:
                bin_wers = []
                for pred in bin_preds:
                    if 'predicted_phonemes' in pred and 'true_phonemes' in pred:
                        wer, _ = self.eval_manager.wer_calculator.calculate_wer(
                            [[self.phoneme_vocab.logit_to_phoneme[idx] for idx in pred['predicted_phonemes']]],
                            [[self.phoneme_vocab.logit_to_phoneme[idx] for idx in pred['true_phonemes']]]
                        )
                        bin_wers.append(wer)

                self.results['per_length_bin'][bin_name] = {
                    'num_samples': len(bin_preds),
                    'mean_wer': np.mean(bin_wers) if bin_wers else 0.0,
                    'mean_length': np.mean([p['sequence_length'] for p in bin_preds])
                }

        # Store all predictions
        self.results['predictions'] = all_predictions

    def print_summary(self):
        """Print evaluation summary"""
        logger.info("\n" + "="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)

        # Overall metrics
        logger.info("\nOverall Performance:")
        for key, value in self.results['overall'].items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

        # Per-session summary
        if self.results['per_session']:
            logger.info("\nPer-Session Performance:")
            session_wers = []
            for session_id, metrics in self.results['per_session'].items():
                logger.info(f"  {session_id}: WER={metrics.get('mean_wer', 0):.3f} (n={metrics['num_samples']})")
                if 'mean_wer' in metrics:
                    session_wers.append(metrics['mean_wer'])

            if session_wers:
                logger.info(f"  Average across sessions: {np.mean(session_wers):.3f}")

        # Per-mode summary
        if self.results['per_mode']:
            logger.info("\nPer-Mode Performance:")
            for mode_name, metrics in self.results['per_mode'].items():
                logger.info(
                    f"  {mode_name}: "
                    f"WER={metrics.get('mean_wer', 0):.3f}, "
                    f"Acc={metrics.get('accuracy', 0):.3f}, "
                    f"Conf={metrics['mean_confidence']:.3f} "
                    f"(n={metrics['num_samples']})"
                )

        # Per-length summary
        if self.results['per_length_bin']:
            logger.info("\nPerformance by Sequence Length:")
            for bin_name, metrics in self.results['per_length_bin'].items():
                logger.info(
                    f"  {bin_name}: WER={metrics.get('mean_wer', 0):.3f} "
                    f"(n={metrics['num_samples']}, avg_len={metrics['mean_length']:.0f})"
                )

        logger.info("="*60)

    def save_results(self):
        """Save evaluation results to files"""
        output_dir = Path(self.config.output.dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary JSON
        summary_file = output_dir / f"evaluation_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            # Convert results to JSON-serializable format
            json_results = {
                'overall': self.results['overall'],
                'per_session': dict(self.results['per_session']),
                'per_mode': dict(self.results['per_mode']),
                'per_length_bin': dict(self.results['per_length_bin']),
                'config': OmegaConf.to_container(self.config)
            }
            json.dump(json_results, f, indent=2)

        logger.info(f"Summary saved to {summary_file}")

        # Save predictions to CSV
        if self.results['predictions'] and self.config.output.save_predictions:
            predictions_file = output_dir / f"predictions_{time.strftime('%Y%m%d_%H%M%S')}.csv"

            # Convert predictions to DataFrame
            df_data = []
            for pred in self.results['predictions']:
                row = {
                    'session_id': pred['session_id'],
                    'block_num': pred['block_num'],
                    'trial_num': pred['trial_num'],
                    'sequence_length': pred['sequence_length'],
                    'mode_prediction': pred['mode_prediction'],
                    'mode_confidence': pred['mode_confidence']
                }

                if 'predicted_text' in pred:
                    row['predicted_text'] = pred['predicted_text']
                if 'true_text' in pred:
                    row['true_text'] = pred['true_text']
                if 'true_mode' in pred:
                    row['true_mode'] = pred['true_mode']

                df_data.append(row)

            df = pd.DataFrame(df_data)
            df.to_csv(predictions_file, index=False)

            logger.info(f"Predictions saved to {predictions_file}")

    def _batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch

    def run_inference_speed_test(self):
        """Test inference speed"""
        logger.info("\nRunning inference speed test...")

        # Create dummy input
        batch_sizes = [1, 4, 8, 16]
        seq_len = 1000  # 20 seconds at 50Hz

        for batch_size in batch_sizes:
            dummy_input = torch.randn(batch_size, seq_len, 512).to(self.device)
            sequence_lengths = torch.full((batch_size,), seq_len).to(self.device)

            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = self.model.inference(
                        neural_features=dummy_input,
                        sequence_lengths=sequence_lengths,
                        decode_mode='greedy'
                    )

            # Time inference
            torch.cuda.synchronize()
            start_time = time.time()

            num_iterations = 10
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = self.model.inference(
                        neural_features=dummy_input,
                        sequence_lengths=sequence_lengths,
                        decode_mode='greedy'
                    )

            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time

            avg_time = elapsed_time / num_iterations
            throughput = batch_size / avg_time

            logger.info(
                f"  Batch size {batch_size}: "
                f"{avg_time*1000:.2f}ms/batch, "
                f"{throughput:.2f} samples/sec"
            )


@hydra.main(config_path="../configs", config_name="evaluate", version_base="1.3")
def main(config: DictConfig):
    """Main evaluation entry point"""
    # Create evaluator
    evaluator = ModelEvaluator(config)

    # Setup
    evaluator.setup_model()
    evaluator.setup_data()

    # Run evaluation
    evaluator.evaluate()

    # Optional: Run inference speed test
    if config.evaluation.test_inference_speed:
        evaluator.run_inference_speed_test()


if __name__ == "__main__":
    main()