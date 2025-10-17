#!/usr/bin/env python3
"""
Competition submission generation script for MODE-SSM.
Processes test data with trained model and generates CSV submission for Brain-to-Text 2025.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
import pandas as pd

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from datasets.brain2text import Brain2TextDataset, collate_batch
from datasets.phoneme_vocab import PhonemeVocabulary
from mode_ssm.models.mode_ssm_model import MODESSMModel, MODESSMConfig
from mode_ssm.checkpoint_manager import CheckpointManager
from mode_ssm.submission_formatter import SubmissionFormatter, validate_submission_format
from mode_ssm.models.tta_loop import TTALoop, TTAConfig


logger = logging.getLogger(__name__)


class SubmissionGenerator:
    """Generates competition submissions from trained MODE-SSM model"""

    def __init__(self, config: DictConfig):
        """
        Initialize submission generator.

        Args:
            config: Submission generation configuration
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model and data
        self.model = None
        self.test_loader = None
        self.checkpoint_manager = None
        self.phoneme_vocab = PhonemeVocabulary()

        # Submission formatter
        self.formatter = SubmissionFormatter(
            validate_format=config.submission.validate_format,
            validate_order=config.submission.validate_order
        )

        # TTA components
        self.tta_loop = None
        self.tta_enabled = getattr(config, 'tta', {}).get('enabled', False)

        # Statistics tracking
        self.stats = {
            'total_samples': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'avg_prediction_length': 0.0,
            'avg_inference_time_ms': 0.0,
            'sessions_processed': set()
        }

        logger.info(f"Using device: {self.device}")

    def load_model(self):
        """Load trained model from checkpoint"""
        logger.info("Loading trained model...")

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

        # Determine checkpoint to load
        if self.config.checkpoint.path:
            checkpoint_path = Path(self.config.checkpoint.path)
        elif self.config.checkpoint.use_best:
            checkpoint_path = self.checkpoint_manager.get_best_checkpoint()
        else:
            checkpoint_path = self.checkpoint_manager.get_latest_checkpoint()

        if checkpoint_path is None or not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")

        # Load model weights
        metadata = self.checkpoint_manager.restore_model(
            model=self.model,
            checkpoint_path=checkpoint_path
        )

        logger.info(f"Model loaded from: {checkpoint_path}")
        if metadata:
            logger.info(f"  Training epoch: {metadata.epoch}")
            if metadata.val_wer is not None:
                logger.info(f"  Validation WER: {metadata.val_wer:.3f}")
            else:
                logger.info(f"  Validation WER: Not available")
            logger.info(f"  Training stage: {metadata.training_stage}")

        # Model info
        total_params = self.model.get_num_parameters()
        logger.info(f"Model parameters: {total_params:,}")

        # Setup TTA if enabled
        if self.tta_enabled:
            self._setup_tta()

    def load_test_data(self):
        """Load test dataset"""
        logger.info(f"Loading test data from: {self.config.data.test_path}")

        # Create test dataset
        test_dataset = Brain2TextDataset(
            hdf5_path=self.config.data.test_path,
            min_sequence_ms=self.config.data.min_sequence_ms,
            max_sequence_ms=self.config.data.max_sequence_ms,
            missing_channel_threshold=self.config.data.missing_channel_threshold,
            cache_data=self.config.data.cache_data,
            transform=None,  # No augmentation for inference
            filter_quality=self.config.data.filter_quality
        )

        logger.info(f"Test dataset: {len(test_dataset)} samples")

        # Get dataset statistics
        try:
            stats = test_dataset.get_statistics()
            logger.info("Dataset statistics:")
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.2f}")
                else:
                    logger.info(f"  {key}: {value}")
        except Exception as e:
            logger.warning(f"Could not get dataset statistics: {e}")

        # Create data loader
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.inference.batch_size,
            shuffle=False,  # Must maintain order for submission
            num_workers=self.config.data.num_workers,
            pin_memory=True,
            collate_fn=collate_batch,
            persistent_workers=True if self.config.data.num_workers > 0 else False,
            drop_last=False
        )

        self.stats['total_samples'] = len(test_dataset)

    def _setup_tta(self):
        """Setup test-time adaptation"""
        logger.info("Setting up test-time adaptation...")

        # Create TTA config from submission config
        tta_config = TTAConfig(
            session_adaptation_enabled=self.config.tta.get('session_adaptation_enabled', True),
            entropy_minimization_enabled=self.config.tta.get('entropy_minimization_enabled', True),
            adaptation_lr=self.config.tta.get('adaptation_lr', 0.001),
            adaptation_steps=self.config.tta.get('adaptation_steps', 3),
            statistics_momentum=self.config.tta.get('statistics_momentum', 0.9),
            entropy_threshold=self.config.tta.get('entropy_threshold', 2.0),
            entropy_weight=self.config.tta.get('entropy_weight', 0.5),
            min_samples_for_adaptation=self.config.tta.get('min_samples_for_adaptation', 8),
            adaptation_layers=self.config.tta.get('adaptation_layers', ['preprocessor']),
            confidence_threshold=self.config.tta.get('confidence_threshold', 0.8)
        )

        # Create TTA loop
        self.tta_loop = TTALoop(tta_config, self.model)

        logger.info("TTA setup complete")

    def generate_predictions(self):
        """Generate predictions for all test samples"""
        logger.info("Generating predictions...")

        inference_times = []
        prediction_lengths = []

        # Progress bar
        progress_bar = tqdm(
            self.test_loader,
            desc="Generating predictions",
            disable=not self.config.inference.show_progress
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                batch_start = time.time()

                try:
                    # Move batch to device
                    batch = self._batch_to_device(batch)

                    # Apply TTA if enabled
                    if self.tta_enabled and self.tta_loop is not None:
                        adapted_batch = self.tta_loop.adapt_multi_step(
                            batch,
                            num_steps=self.config.tta.get('adaptation_steps', 3)
                        )
                        # Use adapted features
                        batch['neural_features'] = adapted_batch['neural_features']

                    # Run inference
                    with autocast(enabled=self.config.inference.use_amp):
                        results = self.model.inference(
                            neural_features=batch['neural_features'],
                            sequence_lengths=batch['sequence_lengths'],
                            decode_mode=self.config.inference.decode_mode,
                            return_mode_predictions=True
                        )

                    # Process predictions for each sample in batch
                    batch_size = len(batch['neural_features'])
                    for i in range(batch_size):
                        try:
                            prediction_text = self._process_prediction(results, i)

                            # Add to submission formatter
                            self.formatter.add_prediction(
                                session=batch['session_ids'][i],
                                block=batch['block_nums'][i].item(),
                                trial=batch['trial_nums'][i].item(),
                                prediction=prediction_text
                            )

                            self.stats['successful_predictions'] += 1
                            self.stats['sessions_processed'].add(batch['session_ids'][i])

                            # Track prediction length
                            prediction_lengths.append(len(prediction_text.split()))

                        except Exception as e:
                            logger.error(f"Failed to process sample {i} in batch {batch_idx}: {e}")

                            # Add fallback prediction
                            self.formatter.add_prediction(
                                session=batch['session_ids'][i],
                                block=batch['block_nums'][i].item(),
                                trial=batch['trial_nums'][i].item(),
                                prediction="prediction_failed"
                            )

                            self.stats['failed_predictions'] += 1

                    # Track timing
                    batch_time = (time.time() - batch_start) * 1000  # ms
                    inference_times.append(batch_time / batch_size)  # ms per sample

                    # Update progress bar
                    if batch_idx % 10 == 0:
                        avg_time = sum(inference_times[-10:]) / min(10, len(inference_times))
                        progress_bar.set_postfix({
                            'Time/sample': f'{avg_time:.1f}ms',
                            'Success': f'{self.stats["successful_predictions"]}'
                        })

                except Exception as e:
                    logger.error(f"Failed to process batch {batch_idx}: {e}")
                    self.stats['failed_predictions'] += len(batch['neural_features'])

        # Update final statistics
        if inference_times:
            self.stats['avg_inference_time_ms'] = sum(inference_times) / len(inference_times)

        if prediction_lengths:
            self.stats['avg_prediction_length'] = sum(prediction_lengths) / len(prediction_lengths)

        logger.info(f"Prediction generation complete:")
        logger.info(f"  Successful: {self.stats['successful_predictions']}")
        logger.info(f"  Failed: {self.stats['failed_predictions']}")
        logger.info(f"  Sessions processed: {len(self.stats['sessions_processed'])}")

    def _process_prediction(self, results: Dict[str, torch.Tensor], sample_idx: int) -> str:
        """
        Process model results into prediction text.

        Args:
            results: Model inference results
            sample_idx: Index of sample in batch

        Returns:
            Prediction text string
        """
        if 'decoded_sequences' in results:
            # Use decoded phoneme sequence
            decoded_sequence = results['decoded_sequences'][sample_idx]

            if self.config.submission.convert_phonemes_to_text:
                # Convert phoneme indices to phoneme strings
                phoneme_strings = []
                for idx in decoded_sequence:
                    if 0 <= idx < len(self.phoneme_vocab.logit_to_phoneme):
                        phoneme = self.phoneme_vocab.logit_to_phoneme[idx]
                        if phoneme != 'BLANK':  # Skip blank tokens
                            phoneme_strings.append(phoneme)

                # Convert phonemes to approximate text
                prediction_text = self.phoneme_vocab.phonemes_to_text(phoneme_strings)

                # Clean up text
                prediction_text = prediction_text.strip()
                if not prediction_text:
                    prediction_text = "no_prediction"

            else:
                # Use phoneme sequence directly
                phoneme_strings = [
                    self.phoneme_vocab.logit_to_phoneme[idx]
                    for idx in decoded_sequence
                    if 0 <= idx < len(self.phoneme_vocab.logit_to_phoneme) and
                    self.phoneme_vocab.logit_to_phoneme[idx] != 'BLANK'
                ]
                prediction_text = ' '.join(phoneme_strings)

        else:
            # Fallback: use raw predictions
            if 'predictions' in results:
                raw_predictions = results['predictions'][sample_idx]
                prediction_text = ' '.join([f'token_{idx.item()}' for idx in raw_predictions[:10]])
            else:
                prediction_text = "no_prediction_available"

        # Ensure prediction is not empty
        if not prediction_text or prediction_text.isspace():
            prediction_text = "empty_prediction"

        return prediction_text

    def save_submission(self):
        """Save submission to CSV file"""
        logger.info("Saving submission...")

        # Sort chronologically
        self.formatter.sort_chronologically()

        # Get statistics
        submission_stats = self.formatter.get_statistics()
        logger.info(f"Submission statistics:")
        logger.info(f"  Total records: {submission_stats['total_records']}")
        logger.info(f"  Sessions: {submission_stats['num_sessions']}")

        # Create output directory
        output_dir = Path(self.config.output.dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        if self.config.output.filename:
            submission_file = output_dir / self.config.output.filename
        else:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            submission_file = output_dir / f"submission_{timestamp}.csv"

        # Save submission
        self.formatter.save_to_file(submission_file)

        logger.info(f"Submission saved: {submission_file}")

        # Validate submission if requested
        if self.config.submission.validate_submission:
            logger.info("Validating submission...")
            is_valid, errors = validate_submission_format(submission_file)

            if is_valid:
                logger.info("✅ Submission validation passed!")
            else:
                logger.error("❌ Submission validation failed:")
                for error in errors:
                    logger.error(f"  - {error}")

        # Save metadata
        self._save_metadata(submission_file)

        return submission_file

    def _save_metadata(self, submission_file: Path):
        """Save submission metadata"""
        metadata_file = submission_file.with_suffix('.json')

        metadata = {
            'submission_file': str(submission_file),
            'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_config': OmegaConf.to_container(self.config.model),
            'inference_config': OmegaConf.to_container(self.config.inference),
            'checkpoint_path': str(self.config.checkpoint.get('path', '')),
            'statistics': dict(self.stats),
            'submission_stats': self.formatter.get_statistics()
        }

        # Convert sets to lists for JSON serialization
        if 'sessions_processed' in metadata['statistics']:
            metadata['statistics']['sessions_processed'] = list(
                metadata['statistics']['sessions_processed']
            )

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved: {metadata_file}")

    def _batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch

    def run(self) -> Path:
        """
        Run complete submission generation pipeline.

        Returns:
            Path to generated submission file
        """
        logger.info("Starting submission generation pipeline...")
        start_time = time.time()

        # Load model and data
        self.load_model()
        self.load_test_data()

        # Generate predictions
        self.generate_predictions()

        # Save submission
        submission_file = self.save_submission()

        # Final report
        total_time = time.time() - start_time
        logger.info(f"Submission generation completed in {total_time:.1f}s")
        logger.info(f"Final statistics:")
        logger.info(f"  Total samples: {self.stats['total_samples']}")
        logger.info(f"  Successful predictions: {self.stats['successful_predictions']}")
        logger.info(f"  Success rate: {100.0 * self.stats['successful_predictions'] / self.stats['total_samples']:.1f}%")
        logger.info(f"  Average inference time: {self.stats['avg_inference_time_ms']:.2f}ms/sample")

        return submission_file


@hydra.main(config_path="../configs", config_name="submission", version_base="1.3")
def main(config: DictConfig):
    """Main submission generation entry point"""
    # Create generator
    generator = SubmissionGenerator(config)

    # Generate submission
    submission_file = generator.run()

    logger.info(f"Submission ready: {submission_file}")
    print(f"Submission saved to: {submission_file}")


if __name__ == "__main__":
    main()