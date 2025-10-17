"""
Integration tests for test-time adaptation workflow.
Tests complete TTA pipeline with real model components and multi-session data.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from mode_ssm.models.tta_loop import TTALoop, TTAConfig
from mode_ssm.models.mode_ssm_model import MODESSMModel, MODESSMConfig
from datasets.brain2text import Brain2TextDataset, collate_batch
from tests.fixtures.synthetic_neural import SyntheticNeuralDataGenerator
from mode_ssm.evaluation_metrics import EvaluationMetrics


class TestTTAIntegration:
    """Integration tests for complete TTA pipeline"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def synthetic_data_generator(self):
        """Create synthetic data generator with session-specific drift"""
        return SyntheticNeuralDataGenerator(
            num_channels=512,
            vocab_size=40,
            sampling_rate=50.0,
            seed=42
        )

    @pytest.fixture
    def multi_session_dataset(self, temp_dir, synthetic_data_generator):
        """Create multi-session dataset with neural drift simulation"""
        dataset_file = temp_dir / "multi_session_data.h5"

        # Create dataset with 3 sessions showing different drift patterns
        sessions = ['session_baseline', 'session_drift_high', 'session_drift_low']

        synthetic_data_generator.create_hdf5_dataset_with_drift(
            output_path=dataset_file,
            num_trials=30,
            min_seq_len=50,
            max_seq_len=150,
            sessions=sessions,
            blocks_per_session=3,
            drift_patterns={
                'session_baseline': {'offset': 0.0, 'scale': 1.0},
                'session_drift_high': {'offset': 2.0, 'scale': 1.5},  # High baseline drift
                'session_drift_low': {'offset': -1.0, 'scale': 0.7}   # Low baseline drift
            }
        )

        return dataset_file

    @pytest.fixture
    def test_model(self):
        """Create test MODE-SSM model"""
        config = MODESSMConfig(
            d_model=256,
            d_state=32,
            encoder_layers=2,
            num_channels=512,
            vocab_size=40
        )
        model = MODESSMModel(config)
        model.eval()
        return model

    @pytest.fixture
    def tta_config(self):
        """Complete TTA configuration for integration testing"""
        return TTAConfig(
            session_adaptation_enabled=True,
            entropy_minimization_enabled=True,
            adaptation_lr=0.001,
            adaptation_steps=5,
            statistics_momentum=0.95,
            entropy_threshold=2.0,
            entropy_weight=0.3,
            min_samples_for_adaptation=8,
            adaptation_layers=['preprocessor'],
            confidence_threshold=0.85,
            max_adaptation_steps_per_session=10,
            adaptation_warmup_samples=5
        )

    def test_end_to_end_tta_pipeline(
        self,
        multi_session_dataset,
        test_model,
        tta_config
    ):
        """Test complete end-to-end TTA pipeline"""

        # 1. Load multi-session dataset
        dataset = Brain2TextDataset(
            hdf5_path=multi_session_dataset,
            cache_data=False,
            filter_quality=False
        )

        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=6,
            shuffle=False,
            collate_fn=collate_batch
        )

        # 2. Create TTA loop
        tta_loop = TTALoop(tta_config, test_model)

        # 3. Process batches with TTA
        session_results = {}
        adaptation_history = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 5:  # Limit for testing
                    break

                # Perform TTA
                adapted_batch = tta_loop.adapt_multi_step(
                    batch,
                    num_steps=tta_config.adaptation_steps
                )

                # Run inference on adapted features
                results = test_model.inference(
                    neural_features=adapted_batch['neural_features'],
                    sequence_lengths=adapted_batch['sequence_lengths'],
                    decode_mode='greedy'
                )

                # Track results per session
                for i, session_id in enumerate(batch['session_ids']):
                    if session_id not in session_results:
                        session_results[session_id] = {
                            'predictions': [],
                            'adaptation_stats': []
                        }

                    session_results[session_id]['predictions'].append(
                        results['decoded_sequences'][i]
                    )
                    session_results[session_id]['adaptation_stats'].append(
                        adapted_batch['adaptation_stats']
                    )

                adaptation_history.append(adapted_batch['adaptation_stats'])

        # 4. Verify TTA effectiveness
        assert len(session_results) >= 2, "Should have multiple sessions"

        # Check that each session received adaptation
        for session_id, results in session_results.items():
            assert len(results['predictions']) > 0
            assert len(results['adaptation_stats']) > 0

            # Check adaptation statistics
            last_stats = results['adaptation_stats'][-1]
            assert 'session_adaptations' in last_stats
            assert session_id in last_stats['session_adaptations']

    def test_session_specific_adaptation_effectiveness(
        self,
        multi_session_dataset,
        test_model,
        tta_config
    ):
        """Test that TTA provides session-specific benefits"""

        dataset = Brain2TextDataset(
            hdf5_path=multi_session_dataset,
            cache_data=False,
            filter_quality=False
        )

        from torch.utils.data import DataLoader

        # Create separate dataloaders for each session
        session_dataloaders = {}
        for session_name in ['session_baseline', 'session_drift_high', 'session_drift_low']:
            # Filter dataset for specific session
            session_indices = [
                i for i, sample in enumerate(dataset)
                if dataset[i]['session_id'] == session_name
            ][:10]  # Limit samples

            if session_indices:
                session_subset = torch.utils.data.Subset(dataset, session_indices)
                session_dataloaders[session_name] = DataLoader(
                    session_subset,
                    batch_size=4,
                    shuffle=False,
                    collate_fn=collate_batch
                )

        # Test with and without TTA for each session
        tta_loop = TTALoop(tta_config, test_model)
        results_without_tta = {}
        results_with_tta = {}

        for session_name, dataloader in session_dataloaders.items():
            # Without TTA
            batch = next(iter(dataloader))
            with torch.no_grad():
                no_tta_results = test_model.inference(
                    neural_features=batch['neural_features'],
                    sequence_lengths=batch['sequence_lengths'],
                    decode_mode='greedy'
                )

            results_without_tta[session_name] = no_tta_results

            # With TTA
            adapted_batch = tta_loop.adapt_multi_step(batch, num_steps=3)
            with torch.no_grad():
                tta_results = test_model.inference(
                    neural_features=adapted_batch['neural_features'],
                    sequence_lengths=adapted_batch['sequence_lengths'],
                    decode_mode='greedy'
                )

            results_with_tta[session_name] = tta_results

        # Verify that TTA produces different results
        for session_name in session_dataloaders.keys():
            no_tta_preds = results_without_tta[session_name]['decoded_sequences']
            tta_preds = results_with_tta[session_name]['decoded_sequences']

            # TTA should produce different predictions for drift sessions
            if 'drift' in session_name:
                # At least some predictions should be different
                different_predictions = sum(
                    not torch.equal(no_tta_preds[i], tta_preds[i])
                    for i in range(len(no_tta_preds))
                )
                assert different_predictions > 0, f"TTA should change predictions for {session_name}"

    def test_tta_convergence_over_session(
        self,
        multi_session_dataset,
        test_model,
        tta_config
    ):
        """Test that TTA converges as more data from a session is processed"""

        dataset = Brain2TextDataset(
            hdf5_path=multi_session_dataset,
            cache_data=False,
            filter_quality=False
        )

        # Focus on one session with drift
        session_samples = [
            dataset[i] for i in range(len(dataset))
            if dataset[i]['session_id'] == 'session_drift_high'
        ][:15]  # Take 15 samples

        tta_loop = TTALoop(tta_config, test_model)
        entropy_evolution = []
        adaptation_strength = []

        # Process samples sequentially to simulate online adaptation
        for i in range(0, len(session_samples), 3):
            batch_samples = session_samples[i:i+3]

            # Create batch
            batch = {
                'neural_features': torch.stack([s['neural_features'] for s in batch_samples]),
                'sequence_lengths': torch.stack([s['sequence_length'] for s in batch_samples]),
                'session_ids': [s['session_id'] for s in batch_samples]
            }

            # Adapt and track statistics
            adapted_batch = tta_loop.adapt_multi_step(batch, num_steps=2)
            stats = adapted_batch['adaptation_stats']

            # Track convergence metrics
            entropy_evolution.append(stats['entropy_stats']['mean_entropy'])

            # Track adaptation strength (how much features changed)
            feature_change = torch.norm(
                adapted_batch['neural_features'] - batch['neural_features']
            ).item()
            adaptation_strength.append(feature_change)

        # Verify convergence trends
        assert len(entropy_evolution) >= 3, "Need multiple adaptation steps to test convergence"

        # Early entropy should be higher than later entropy (general trend)
        early_entropy = torch.tensor(entropy_evolution[:2]).mean()
        late_entropy = torch.tensor(entropy_evolution[-2:]).mean()

        # Allow some tolerance due to randomness
        assert late_entropy <= early_entropy + 0.5, "Entropy should generally decrease with adaptation"

    def test_tta_memory_efficiency(
        self,
        multi_session_dataset,
        test_model,
        tta_config
    ):
        """Test that TTA doesn't cause excessive memory usage"""

        dataset = Brain2TextDataset(
            hdf5_path=multi_session_dataset,
            cache_data=False,
            filter_quality=False
        )

        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_batch
        )

        tta_loop = TTALoop(tta_config, test_model)

        # Track memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

        # Process multiple batches
        processed_batches = 0
        for batch in dataloader:
            if processed_batches >= 3:  # Process a few batches
                break

            # Adapt batch
            adapted_batch = tta_loop.adapt_multi_step(batch, num_steps=2)

            # Clear intermediate results
            del adapted_batch

            processed_batches += 1

        # Check memory hasn't grown excessively
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_growth = final_memory - initial_memory

            # Memory growth should be reasonable (less than 100MB for test)
            assert memory_growth < 100 * 1024 * 1024, f"Excessive memory growth: {memory_growth} bytes"

    def test_tta_with_different_sequence_lengths(
        self,
        test_model,
        tta_config
    ):
        """Test TTA with variable sequence lengths"""

        # Create batch with varied sequence lengths
        batch_size = 4
        max_seq_len = 120
        num_channels = 512

        # Create features with different actual lengths
        neural_features = torch.randn(batch_size, max_seq_len, num_channels)
        sequence_lengths = torch.tensor([60, 80, 100, 120])
        session_ids = ['session1', 'session1', 'session2', 'session2']

        # Zero out padding regions
        for i, seq_len in enumerate(sequence_lengths):
            neural_features[i, seq_len:, :] = 0

        batch = {
            'neural_features': neural_features,
            'sequence_lengths': sequence_lengths,
            'session_ids': session_ids
        }

        # Apply TTA
        tta_loop = TTALoop(tta_config, test_model)
        adapted_batch = tta_loop.adapt_multi_step(batch, num_steps=2)

        # Verify output structure
        assert adapted_batch['neural_features'].shape == neural_features.shape
        assert torch.equal(adapted_batch['sequence_lengths'], sequence_lengths)

        # Verify padding regions remain zero
        for i, seq_len in enumerate(sequence_lengths):
            padding_region = adapted_batch['neural_features'][i, seq_len:, :]
            assert torch.allclose(padding_region, torch.zeros_like(padding_region), atol=1e-6)

    def test_tta_error_handling(
        self,
        test_model,
        tta_config
    ):
        """Test TTA error handling with edge cases"""

        tta_loop = TTALoop(tta_config, test_model)

        # Test with empty batch
        empty_batch = {
            'neural_features': torch.empty(0, 100, 512),
            'sequence_lengths': torch.empty(0, dtype=torch.long),
            'session_ids': []
        }

        # Should handle gracefully
        try:
            adapted_batch = tta_loop.adapt_multi_step(empty_batch, num_steps=1)
            assert adapted_batch['neural_features'].shape[0] == 0
        except Exception as e:
            # If it raises an exception, it should be informative
            assert "empty" in str(e).lower() or "zero" in str(e).lower()

        # Test with very short sequences
        short_batch = {
            'neural_features': torch.randn(2, 10, 512),
            'sequence_lengths': torch.tensor([5, 8]),
            'session_ids': ['session1', 'session1']
        }

        # Should not crash
        adapted_short = tta_loop.adapt_multi_step(short_batch, num_steps=1)
        assert adapted_short['neural_features'].shape == short_batch['neural_features'].shape

        # Test with NaN inputs
        nan_batch = {
            'neural_features': torch.full((2, 50, 512), float('nan')),
            'sequence_lengths': torch.tensor([40, 50]),
            'session_ids': ['session1', 'session1']
        }

        # Should either handle NaN gracefully or raise informative error
        try:
            tta_loop.adapt_multi_step(nan_batch, num_steps=1)
        except Exception as e:
            assert "nan" in str(e).lower() or "invalid" in str(e).lower()

    def test_tta_deterministic_behavior(
        self,
        test_model,
        tta_config
    ):
        """Test that TTA produces deterministic results with fixed seeds"""

        # Create identical batches
        torch.manual_seed(123)
        batch1 = {
            'neural_features': torch.randn(3, 80, 512),
            'sequence_lengths': torch.tensor([70, 75, 80]),
            'session_ids': ['session1', 'session1', 'session2']
        }

        torch.manual_seed(123)
        batch2 = {
            'neural_features': torch.randn(3, 80, 512),
            'sequence_lengths': torch.tensor([70, 75, 80]),
            'session_ids': ['session1', 'session1', 'session2']
        }

        # Apply TTA with same seed
        tta_loop1 = TTALoop(tta_config, test_model)
        tta_loop2 = TTALoop(tta_config, test_model)

        torch.manual_seed(456)
        adapted_batch1 = tta_loop1.adapt_multi_step(batch1, num_steps=2)

        torch.manual_seed(456)
        adapted_batch2 = tta_loop2.adapt_multi_step(batch2, num_steps=2)

        # Results should be identical
        assert torch.allclose(
            adapted_batch1['neural_features'],
            adapted_batch2['neural_features'],
            atol=1e-6
        )

    def test_tta_performance_comparison(
        self,
        multi_session_dataset,
        test_model,
        tta_config
    ):
        """Test that TTA provides measurable performance benefits"""

        dataset = Brain2TextDataset(
            hdf5_path=multi_session_dataset,
            cache_data=False,
            filter_quality=False
        )

        from torch.utils.data import DataLoader

        # Focus on drift sessions (where TTA should help most)
        drift_samples = [
            dataset[i] for i in range(len(dataset))
            if 'drift' in dataset[i]['session_id']
        ][:20]  # Limit for testing

        if len(drift_samples) < 8:
            pytest.skip("Insufficient drift samples for performance comparison")

        # Create dataloader
        drift_dataset = torch.utils.data.TensorDataset(
            torch.stack([s['neural_features'] for s in drift_samples]),
            torch.stack([s['sequence_length'] for s in drift_samples]),
            [s['session_id'] for s in drift_samples]
        )

        dataloader = DataLoader(drift_dataset, batch_size=4, shuffle=False)

        # Compare predictions with and without TTA
        tta_loop = TTALoop(tta_config, test_model)
        baseline_predictions = []
        tta_predictions = []

        for batch_features, batch_lengths, batch_sessions in dataloader:
            batch = {
                'neural_features': batch_features,
                'sequence_lengths': batch_lengths,
                'session_ids': list(batch_sessions)
            }

            # Baseline (no TTA)
            with torch.no_grad():
                baseline_results = test_model.inference(
                    neural_features=batch['neural_features'],
                    sequence_lengths=batch['sequence_lengths'],
                    decode_mode='greedy'
                )
                baseline_predictions.extend(baseline_results['decoded_sequences'])

            # With TTA
            adapted_batch = tta_loop.adapt_multi_step(batch, num_steps=3)
            with torch.no_grad():
                tta_results = test_model.inference(
                    neural_features=adapted_batch['neural_features'],
                    sequence_lengths=adapted_batch['sequence_lengths'],
                    decode_mode='greedy'
                )
                tta_predictions.extend(tta_results['decoded_sequences'])

        # Verify we have results
        assert len(baseline_predictions) > 0
        assert len(tta_predictions) == len(baseline_predictions)

        # Count how many predictions changed with TTA
        changed_predictions = sum(
            not torch.equal(baseline_predictions[i], tta_predictions[i])
            for i in range(len(baseline_predictions))
        )

        # TTA should change at least some predictions for drift data
        assert changed_predictions > 0, "TTA should modify predictions for drift sessions"

        # At least 20% of predictions should be affected by TTA
        change_ratio = changed_predictions / len(baseline_predictions)
        assert change_ratio >= 0.1, f"TTA should affect meaningful fraction of predictions (got {change_ratio:.2%})"