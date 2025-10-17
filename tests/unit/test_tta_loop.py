"""
Unit tests for test-time adaptation (TTA) components.
Tests session statistics adaptation and entropy minimization.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Tuple

from mode_ssm.models.tta_loop import (
    SessionStatsAdapter,
    EntropyMinimizer,
    TTAConfig,
    TTALoop,
    AdaptationStatistics
)
from mode_ssm.models.mode_ssm_model import MODESSMModel, MODESSMConfig


class TestSessionStatsAdapter:
    """Unit tests for session-level statistics adaptation"""

    @pytest.fixture
    def tta_config(self):
        """Standard TTA configuration"""
        return TTAConfig(
            session_adaptation_enabled=True,
            adaptation_lr=0.001,
            adaptation_steps=5,
            statistics_momentum=0.9,
            min_samples_for_adaptation=10,
            adaptation_layers=['preprocessor', 'encoder'],
            entropy_threshold=2.0,
            entropy_weight=1.0
        )

    @pytest.fixture
    def stats_adapter(self, tta_config):
        """Create session statistics adapter"""
        return SessionStatsAdapter(tta_config)

    @pytest.fixture
    def sample_neural_features(self):
        """Generate sample neural features for different sessions"""
        batch_size = 8
        seq_len = 100
        num_channels = 512

        # Create features with session-specific statistics
        torch.manual_seed(42)

        # Session 1: Higher baseline activity
        session1_data = torch.randn(batch_size // 2, seq_len, num_channels) + 1.0

        # Session 2: Lower baseline activity
        session2_data = torch.randn(batch_size // 2, seq_len, num_channels) - 0.5

        features = torch.cat([session1_data, session2_data], dim=0)

        # Session labels
        session_ids = ['session1'] * (batch_size // 2) + ['session2'] * (batch_size // 2)

        return features, session_ids

    @pytest.fixture
    def mock_model(self):
        """Create mock MODE-SSM model"""
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

    def test_stats_adapter_initialization(self, tta_config):
        """Test adapter initialization"""
        adapter = SessionStatsAdapter(tta_config)

        assert adapter.config == tta_config
        assert adapter.session_stats == {}
        assert adapter.adaptation_enabled == tta_config.session_adaptation_enabled

    def test_compute_session_statistics(self, stats_adapter, sample_neural_features):
        """Test computation of session-level statistics"""
        features, session_ids = sample_neural_features

        # Compute statistics for each session
        for i, session_id in enumerate(set(session_ids)):
            session_mask = [sid == session_id for sid in session_ids]
            session_features = features[session_mask]

            stats = stats_adapter.compute_session_statistics(session_features, session_id)

            assert 'mean' in stats
            assert 'var' in stats
            assert 'count' in stats
            assert stats['count'] > 0

            # Check dimensions
            assert stats['mean'].shape == (session_features.shape[-1],)
            assert stats['var'].shape == (session_features.shape[-1],)

    def test_update_session_statistics(self, stats_adapter, sample_neural_features):
        """Test updating session statistics with new data"""
        features, session_ids = sample_neural_features
        session_id = session_ids[0]

        # First batch
        first_batch = features[:2]
        stats1 = stats_adapter.update_session_statistics(first_batch, session_id)

        # Second batch
        second_batch = features[2:4]
        stats2 = stats_adapter.update_session_statistics(second_batch, session_id)

        # Statistics should be updated (different from first batch)
        assert not torch.allclose(stats1['mean'], stats2['mean'], atol=1e-6)
        assert stats2['count'] > stats1['count']

    def test_exponential_moving_average_update(self, stats_adapter, sample_neural_features):
        """Test EMA-based statistics update"""
        features, session_ids = sample_neural_features
        session_id = session_ids[0]

        # Initialize with first batch
        batch1 = features[:2]
        stats_adapter.update_session_statistics(batch1, session_id)

        # Get current stats
        current_stats = stats_adapter.session_stats[session_id].copy()

        # Update with second batch
        batch2 = features[2:4]
        new_stats = stats_adapter.update_session_statistics(batch2, session_id)

        # Check that EMA was applied (momentum should prevent drastic changes)
        momentum = stats_adapter.config.statistics_momentum

        # Mean should be between old and new values
        raw_new_mean = batch2.mean(dim=(0, 1))
        expected_mean = momentum * current_stats['mean'] + (1 - momentum) * raw_new_mean

        assert torch.allclose(new_stats['mean'], expected_mean, atol=1e-5)

    def test_adapt_features(self, stats_adapter, sample_neural_features):
        """Test feature adaptation using session statistics"""
        features, session_ids = sample_neural_features

        # Build statistics for each session
        for session_id in set(session_ids):
            session_mask = [sid == session_id for sid in session_ids]
            session_features = features[session_mask]
            stats_adapter.update_session_statistics(session_features, session_id)

        # Adapt features
        adapted_features = stats_adapter.adapt_features(features, session_ids)

        # Check output shape
        assert adapted_features.shape == features.shape

        # Adapted features should be different from original
        assert not torch.allclose(adapted_features, features, atol=1e-5)

        # Check that adaptation reduces session-specific bias
        for session_id in set(session_ids):
            session_mask = [sid == session_id for sid in session_ids]
            original_session_features = features[session_mask]
            adapted_session_features = adapted_features[session_mask]

            # Adapted features should have mean closer to zero
            original_mean = original_session_features.mean(dim=(0, 1))
            adapted_mean = adapted_session_features.mean(dim=(0, 1))

            assert torch.abs(adapted_mean).mean() <= torch.abs(original_mean).mean()

    def test_insufficient_data_handling(self, stats_adapter):
        """Test handling of insufficient data for adaptation"""
        # Very small batch (below minimum threshold)
        small_features = torch.randn(2, 50, 512)  # Only 2 samples
        session_id = "test_session"

        # Should not perform adaptation with insufficient data
        original_features = small_features.clone()
        adapted_features = stats_adapter.adapt_features(small_features, [session_id, session_id])

        # With insufficient data, features should remain unchanged
        if stats_adapter.config.min_samples_for_adaptation > 2:
            assert torch.allclose(adapted_features, original_features, atol=1e-6)

    def test_multiple_sessions_handling(self, stats_adapter):
        """Test adaptation with multiple sessions in same batch"""
        batch_size = 6
        seq_len = 100
        num_channels = 512

        # Create data with different session characteristics
        session1_data = torch.randn(2, seq_len, num_channels) + 2.0  # High baseline
        session2_data = torch.randn(2, seq_len, num_channels) - 1.0  # Low baseline
        session3_data = torch.randn(2, seq_len, num_channels) * 2.0  # High variance

        features = torch.cat([session1_data, session2_data, session3_data], dim=0)
        session_ids = ['session1', 'session1', 'session2', 'session2', 'session3', 'session3']

        # Build statistics
        for session_id in set(session_ids):
            session_mask = [sid == session_id for sid in session_ids]
            session_features = features[session_mask]
            stats_adapter.update_session_statistics(session_features, session_id)

        # Adapt mixed batch
        adapted_features = stats_adapter.adapt_features(features, session_ids)

        # Check that each session was adapted appropriately
        for session_id in set(session_ids):
            session_mask = [sid == session_id for sid in session_ids]
            session_adapted = adapted_features[session_mask]

            # Session-adapted features should have more consistent statistics
            session_mean = session_adapted.mean(dim=(0, 1))
            assert torch.abs(session_mean).mean() < 1.0  # Should be closer to zero


class TestEntropyMinimizer:
    """Unit tests for entropy minimization"""

    @pytest.fixture
    def tta_config(self):
        """TTA configuration for entropy minimization"""
        return TTAConfig(
            entropy_minimization_enabled=True,
            entropy_threshold=2.0,
            entropy_weight=1.0,
            adaptation_lr=0.001,
            adaptation_steps=3,
            confidence_threshold=0.8
        )

    @pytest.fixture
    def entropy_minimizer(self, tta_config):
        """Create entropy minimizer"""
        return EntropyMinimizer(tta_config)

    @pytest.fixture
    def sample_predictions(self):
        """Generate sample model predictions with varying confidence"""
        batch_size = 4
        vocab_size = 40
        seq_len = 20

        # High confidence predictions (low entropy)
        high_conf = torch.zeros(2, seq_len, vocab_size)
        high_conf[:, :, 5] = 10.0  # Strong peak at class 5
        high_conf[:, :, :] += torch.randn(2, seq_len, vocab_size) * 0.1

        # Low confidence predictions (high entropy)
        low_conf = torch.randn(2, seq_len, vocab_size) * 0.5  # More uniform

        predictions = torch.cat([high_conf, low_conf], dim=0)
        return predictions

    def test_entropy_minimizer_initialization(self, tta_config):
        """Test entropy minimizer initialization"""
        minimizer = EntropyMinimizer(tta_config)

        assert minimizer.config == tta_config
        assert minimizer.entropy_threshold == tta_config.entropy_threshold
        assert minimizer.entropy_weight == tta_config.entropy_weight

    def test_compute_prediction_entropy(self, entropy_minimizer, sample_predictions):
        """Test entropy computation"""
        entropy = entropy_minimizer.compute_prediction_entropy(sample_predictions)

        # Check output shape
        assert entropy.shape == (sample_predictions.shape[0],)

        # High confidence samples should have lower entropy
        high_conf_entropy = entropy[:2].mean()
        low_conf_entropy = entropy[2:].mean()

        assert high_conf_entropy < low_conf_entropy

    def test_identify_high_entropy_samples(self, entropy_minimizer, sample_predictions):
        """Test identification of high-entropy samples"""
        high_entropy_mask = entropy_minimizer.identify_high_entropy_samples(sample_predictions)

        # Check output shape
        assert high_entropy_mask.shape == (sample_predictions.shape[0],)
        assert high_entropy_mask.dtype == torch.bool

        # Should identify low-confidence samples (samples 2,3) as high entropy
        assert not high_entropy_mask[0]  # High confidence sample
        assert not high_entropy_mask[1]  # High confidence sample
        assert high_entropy_mask[2]      # Low confidence sample
        assert high_entropy_mask[3]      # Low confidence sample

    def test_compute_entropy_loss(self, entropy_minimizer, sample_predictions):
        """Test entropy loss computation"""
        entropy_loss = entropy_minimizer.compute_entropy_loss(sample_predictions)

        # Loss should be scalar
        assert entropy_loss.dim() == 0
        assert entropy_loss >= 0  # Entropy is non-negative

        # Loss should be higher for high-entropy predictions
        high_conf_loss = entropy_minimizer.compute_entropy_loss(sample_predictions[:2])
        low_conf_loss = entropy_minimizer.compute_entropy_loss(sample_predictions[2:])

        assert low_conf_loss > high_conf_loss

    def test_entropy_gradient_flow(self, entropy_minimizer, sample_predictions):
        """Test that entropy loss produces valid gradients"""
        sample_predictions.requires_grad_(True)

        entropy_loss = entropy_minimizer.compute_entropy_loss(sample_predictions)
        entropy_loss.backward()

        # Check gradients exist and are reasonable
        assert sample_predictions.grad is not None
        assert not torch.isnan(sample_predictions.grad).any()
        assert not torch.isinf(sample_predictions.grad).any()

    def test_confidence_based_weighting(self, entropy_minimizer):
        """Test confidence-based weighting of entropy loss"""
        batch_size = 2
        vocab_size = 40
        seq_len = 10

        # Very high confidence prediction
        high_conf_pred = torch.zeros(1, seq_len, vocab_size)
        high_conf_pred[0, :, 0] = 100.0  # Extremely confident

        # Medium confidence prediction
        med_conf_pred = torch.zeros(1, seq_len, vocab_size)
        med_conf_pred[0, :, 0] = 2.0
        med_conf_pred[0, :, 1] = 1.0

        high_conf_loss = entropy_minimizer.compute_entropy_loss(high_conf_pred)
        med_conf_loss = entropy_minimizer.compute_entropy_loss(med_conf_pred)

        # High confidence predictions should contribute less to loss
        assert high_conf_loss < med_conf_loss

    def test_batch_entropy_statistics(self, entropy_minimizer, sample_predictions):
        """Test computation of batch-level entropy statistics"""
        stats = entropy_minimizer.compute_batch_entropy_statistics(sample_predictions)

        assert 'mean_entropy' in stats
        assert 'std_entropy' in stats
        assert 'high_entropy_fraction' in stats
        assert 'min_entropy' in stats
        assert 'max_entropy' in stats

        # Check that statistics are reasonable
        assert 0 <= stats['high_entropy_fraction'] <= 1
        assert stats['min_entropy'] <= stats['mean_entropy'] <= stats['max_entropy']


class TestTTALoop:
    """Integration tests for complete TTA loop"""

    @pytest.fixture
    def tta_config(self):
        """Complete TTA configuration"""
        return TTAConfig(
            session_adaptation_enabled=True,
            entropy_minimization_enabled=True,
            adaptation_lr=0.001,
            adaptation_steps=3,
            statistics_momentum=0.9,
            entropy_threshold=1.5,
            entropy_weight=0.5,
            min_samples_for_adaptation=5,
            adaptation_layers=['preprocessor'],
            confidence_threshold=0.9
        )

    @pytest.fixture
    def mock_model(self):
        """Mock model for TTA testing"""
        model = Mock()

        # Mock forward pass
        def mock_forward(neural_features, sequence_lengths):
            batch_size = neural_features.shape[0]
            seq_len = 20
            vocab_size = 40

            # Return mock predictions with some randomness
            predictions = torch.randn(batch_size, seq_len, vocab_size)
            return {
                'rnnt_predictions': predictions,
                'ctc_predictions': predictions,
                'mode_predictions': torch.rand(batch_size, 2)
            }

        model.side_effect = mock_forward
        return model

    @pytest.fixture
    def tta_loop(self, tta_config, mock_model):
        """Create TTA loop instance"""
        return TTALoop(tta_config, mock_model)

    @pytest.fixture
    def sample_batch(self):
        """Generate sample batch for TTA"""
        batch_size = 6
        seq_len = 100
        num_channels = 512

        neural_features = torch.randn(batch_size, seq_len, num_channels)
        sequence_lengths = torch.randint(50, seq_len, (batch_size,))
        session_ids = ['session1', 'session1', 'session2', 'session2', 'session3', 'session3']

        return {
            'neural_features': neural_features,
            'sequence_lengths': sequence_lengths,
            'session_ids': session_ids
        }

    def test_tta_loop_initialization(self, tta_config, mock_model):
        """Test TTA loop initialization"""
        tta_loop = TTALoop(tta_config, mock_model)

        assert tta_loop.config == tta_config
        assert tta_loop.model == mock_model
        assert hasattr(tta_loop, 'session_adapter')
        assert hasattr(tta_loop, 'entropy_minimizer')

    def test_single_step_adaptation(self, tta_loop, sample_batch):
        """Test single adaptation step"""
        original_features = sample_batch['neural_features'].clone()

        # Perform one adaptation step
        adapted_batch = tta_loop.adapt_single_step(sample_batch)

        # Check output structure
        assert 'neural_features' in adapted_batch
        assert 'sequence_lengths' in adapted_batch
        assert 'session_ids' in adapted_batch
        assert 'adaptation_stats' in adapted_batch

        # Features should be modified
        assert not torch.allclose(
            adapted_batch['neural_features'],
            original_features,
            atol=1e-5
        )

    def test_multi_step_adaptation(self, tta_loop, sample_batch):
        """Test multi-step adaptation process"""
        original_features = sample_batch['neural_features'].clone()

        # Perform multiple adaptation steps
        final_batch = tta_loop.adapt_multi_step(sample_batch, num_steps=3)

        # Check that features evolved over multiple steps
        assert not torch.allclose(
            final_batch['neural_features'],
            original_features,
            atol=1e-4
        )

        # Check adaptation statistics
        assert 'adaptation_stats' in final_batch
        stats = final_batch['adaptation_stats']
        assert 'total_steps' in stats
        assert stats['total_steps'] == 3

    def test_adaptation_statistics_tracking(self, tta_loop, sample_batch):
        """Test tracking of adaptation statistics"""
        result = tta_loop.adapt_single_step(sample_batch)

        stats = result['adaptation_stats']

        # Check required statistics
        assert 'session_adaptations' in stats
        assert 'entropy_stats' in stats
        assert 'adaptation_step' in stats

        # Check entropy statistics structure
        entropy_stats = stats['entropy_stats']
        assert 'mean_entropy' in entropy_stats
        assert 'high_entropy_fraction' in entropy_stats

    def test_session_specific_adaptation(self, tta_loop):
        """Test that different sessions receive different adaptations"""
        # Create batches with distinct session characteristics
        batch_session1 = {
            'neural_features': torch.randn(2, 100, 512) + 2.0,  # High baseline
            'sequence_lengths': torch.tensor([90, 95]),
            'session_ids': ['session1', 'session1']
        }

        batch_session2 = {
            'neural_features': torch.randn(2, 100, 512) - 1.0,  # Low baseline
            'sequence_lengths': torch.tensor([85, 100]),
            'session_ids': ['session2', 'session2']
        }

        # Adapt each session
        adapted1 = tta_loop.adapt_single_step(batch_session1)
        adapted2 = tta_loop.adapt_single_step(batch_session2)

        # Adaptations should be different for different sessions
        assert not torch.allclose(
            adapted1['neural_features'],
            adapted2['neural_features'],
            atol=0.1
        )

    def test_adaptation_convergence(self, tta_loop, sample_batch):
        """Test that adaptation converges over multiple steps"""
        entropies = []

        current_batch = sample_batch
        for step in range(5):
            adapted_batch = tta_loop.adapt_single_step(current_batch)

            # Track entropy evolution
            entropy_stats = adapted_batch['adaptation_stats']['entropy_stats']
            entropies.append(entropy_stats['mean_entropy'])

            current_batch = adapted_batch

        # Entropy should generally decrease (indicating increased confidence)
        assert entropies[-1] <= entropies[0] + 0.1  # Allow some tolerance

    def test_adaptation_with_disabled_components(self, mock_model):
        """Test TTA with different component configurations"""
        # Test with only session adaptation
        config_session_only = TTAConfig(
            session_adaptation_enabled=True,
            entropy_minimization_enabled=False,
            adaptation_lr=0.001
        )

        tta_session_only = TTALoop(config_session_only, mock_model)

        # Test with only entropy minimization
        config_entropy_only = TTAConfig(
            session_adaptation_enabled=False,
            entropy_minimization_enabled=True,
            adaptation_lr=0.001,
            entropy_weight=1.0
        )

        tta_entropy_only = TTALoop(config_entropy_only, mock_model)

        # Both should work without errors
        sample_batch = {
            'neural_features': torch.randn(4, 100, 512),
            'sequence_lengths': torch.tensor([90, 95, 85, 100]),
            'session_ids': ['session1', 'session1', 'session2', 'session2']
        }

        result_session = tta_session_only.adapt_single_step(sample_batch)
        result_entropy = tta_entropy_only.adapt_single_step(sample_batch)

        assert 'neural_features' in result_session
        assert 'neural_features' in result_entropy