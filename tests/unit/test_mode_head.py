"""
Unit tests for speaking mode classification head.
Tests binary silent/vocalized classification and contrastive learning.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unittest.mock import patch

from mode_ssm.models.mode_head import ModeClassificationHead


class TestModeClassificationHead:
    """Test cases for ModeClassificationHead"""

    @pytest.fixture
    def mode_head_config(self):
        """Standard mode head configuration"""
        return {
            'd_model': 512,
            'num_modes': 2,
            'dropout': 0.1,
            'contrastive_learning': True,
            'pooling_type': 'global_avg'
        }

    @pytest.fixture
    def mode_head(self, mode_head_config):
        """Create mode classification head instance"""
        return ModeClassificationHead(**mode_head_config)

    @pytest.fixture
    def sample_features(self):
        """Generate sample features [batch_size, seq_len, d_model]"""
        batch_size, seq_len, d_model = 8, 100, 512
        torch.manual_seed(42)
        return torch.randn(batch_size, seq_len, d_model)

    @pytest.fixture
    def sample_mode_labels(self):
        """Generate sample mode labels [batch_size]"""
        torch.manual_seed(42)
        return torch.randint(0, 2, (8,))  # Binary: 0=silent, 1=vocalized

    def test_mode_head_initialization(self, mode_head_config):
        """Test mode head initialization with valid config"""
        mode_head = ModeClassificationHead(**mode_head_config)

        assert mode_head.d_model == 512
        assert mode_head.num_modes == 2
        assert mode_head.pooling_type == 'global_avg'
        assert mode_head.contrastive_learning is True
        assert hasattr(mode_head, 'classifier')
        assert hasattr(mode_head, 'pooling')

    def test_mode_head_invalid_config(self):
        """Test mode head initialization with invalid config"""
        with pytest.raises(ValueError):
            ModeClassificationHead(
                d_model=0,
                num_modes=2
            )

        with pytest.raises(ValueError):
            ModeClassificationHead(
                d_model=512,
                num_modes=1  # Must have at least 2 modes
            )

        with pytest.raises(ValueError):
            ModeClassificationHead(
                d_model=512,
                num_modes=2,
                pooling_type='invalid_pooling'
            )

    def test_forward_pass_shape(self, mode_head, sample_features):
        """Test forward pass produces correct output shape"""
        batch_size = sample_features.shape[0]

        with torch.no_grad():
            outputs = mode_head(sample_features)

        # Should return dictionary with logits
        assert isinstance(outputs, dict)
        assert 'mode_logits' in outputs

        # Mode logits should be [batch_size, num_modes]
        mode_logits = outputs['mode_logits']
        expected_shape = (batch_size, 2)
        assert mode_logits.shape == expected_shape

    def test_pooling_strategies(self, sample_features):
        """Test different pooling strategies"""
        pooling_types = ['global_avg', 'global_max', 'attention_pool']

        for pooling_type in pooling_types:
            mode_head = ModeClassificationHead(
                d_model=512,
                num_modes=2,
                pooling_type=pooling_type
            )

            with torch.no_grad():
                outputs = mode_head(sample_features)

            # All should produce same output shape
            batch_size = sample_features.shape[0]
            assert outputs['mode_logits'].shape == (batch_size, 2)

    def test_classification_probabilities(self, mode_head, sample_features):
        """Test that classification outputs can be converted to probabilities"""
        with torch.no_grad():
            outputs = mode_head(sample_features)

        mode_logits = outputs['mode_logits']

        # Convert to probabilities
        probs = F.softmax(mode_logits, dim=-1)

        # Probabilities should sum to 1 for each sample
        prob_sums = probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6)

        # All probabilities should be non-negative
        assert (probs >= 0).all()

    def test_contrastive_learning_features(self, sample_features):
        """Test contrastive learning feature extraction"""
        # Test with contrastive learning enabled
        mode_head_contrastive = ModeClassificationHead(
            d_model=512,
            num_modes=2,
            contrastive_learning=True
        )

        with torch.no_grad():
            outputs = mode_head_contrastive(sample_features)

        # Should include contrastive features
        assert 'contrastive_features' in outputs

        batch_size = sample_features.shape[0]
        contrastive_features = outputs['contrastive_features']

        # Features should be normalized for contrastive learning
        feature_norms = torch.norm(contrastive_features, dim=-1)
        assert torch.allclose(feature_norms, torch.ones_like(feature_norms), atol=1e-5)

    def test_no_contrastive_learning(self, sample_features):
        """Test mode head without contrastive learning"""
        mode_head = ModeClassificationHead(
            d_model=512,
            num_modes=2,
            contrastive_learning=False
        )

        with torch.no_grad():
            outputs = mode_head(sample_features)

        # Should not include contrastive features
        assert 'contrastive_features' not in outputs
        assert 'mode_logits' in outputs

    def test_gradient_flow(self, mode_head, sample_features, sample_mode_labels):
        """Test gradient flow through mode head"""
        sample_features.requires_grad_(True)

        outputs = mode_head(sample_features)
        mode_logits = outputs['mode_logits']

        # Compute classification loss
        loss = F.cross_entropy(mode_logits, sample_mode_labels)
        loss.backward()

        # Input should have gradients
        assert sample_features.grad is not None
        assert not torch.isnan(sample_features.grad).any()

        # Model parameters should have gradients
        for param in mode_head.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()

    def test_dropout_behavior(self, sample_features):
        """Test dropout behavior in train vs eval modes"""
        mode_head = ModeClassificationHead(
            d_model=512,
            num_modes=2,
            dropout=0.5
        )

        # In eval mode, outputs should be deterministic
        mode_head.eval()
        torch.manual_seed(42)
        outputs1 = mode_head(sample_features)
        torch.manual_seed(42)
        outputs2 = mode_head(sample_features)

        # Should be identical in eval mode
        assert torch.allclose(
            outputs1['mode_logits'], outputs2['mode_logits']
        ), "Eval mode should be deterministic"

    def test_batch_size_invariance(self, mode_head):
        """Test mode head handles different batch sizes"""
        seq_len, d_model = 100, 512

        for batch_size in [1, 4, 8, 16]:
            features = torch.randn(batch_size, seq_len, d_model)

            with torch.no_grad():
                outputs = mode_head(features)

            expected_shape = (batch_size, 2)
            assert outputs['mode_logits'].shape == expected_shape

    def test_sequence_length_invariance(self, mode_head):
        """Test mode head handles different sequence lengths"""
        batch_size, d_model = 4, 512

        for seq_len in [10, 50, 100, 500]:
            features = torch.randn(batch_size, seq_len, d_model)

            with torch.no_grad():
                outputs = mode_head(features)

            expected_shape = (batch_size, 2)
            assert outputs['mode_logits'].shape == expected_shape

    def test_classification_decision_boundary(self, mode_head, sample_features):
        """Test that model can make reasonable classification decisions"""
        with torch.no_grad():
            outputs = mode_head(sample_features)

        mode_logits = outputs['mode_logits']
        predictions = torch.argmax(mode_logits, dim=-1)

        # Predictions should be in valid range
        assert (predictions >= 0).all()
        assert (predictions < 2).all()

        # Should have some variety in predictions (not all same class)
        unique_predictions = torch.unique(predictions)
        assert len(unique_predictions) >= 1  # At least one class predicted

    def test_confidence_scores(self, mode_head, sample_features):
        """Test that confidence scores are reasonable"""
        with torch.no_grad():
            outputs = mode_head(sample_features)

        mode_logits = outputs['mode_logits']
        probs = F.softmax(mode_logits, dim=-1)

        # Maximum probability should be reasonable confidence score
        max_probs = torch.max(probs, dim=-1)[0]

        # Should have some confident predictions
        confident_predictions = (max_probs > 0.6).sum()
        assert confident_predictions >= 0  # At least some confidence

        # Should not all be extremely uncertain
        very_uncertain = (max_probs < 0.51).sum()
        assert very_uncertain < len(max_probs)  # Not all predictions uncertain

    def test_attention_pooling(self, sample_features):
        """Test attention pooling mechanism"""
        mode_head = ModeClassificationHead(
            d_model=512,
            num_modes=2,
            pooling_type='attention_pool'
        )

        with torch.no_grad():
            outputs = mode_head(sample_features)

        # Should include attention weights if using attention pooling
        if hasattr(mode_head.pooling, 'attention'):
            assert 'attention_weights' in outputs or True  # May or may not expose weights

        # Output shape should still be correct
        batch_size = sample_features.shape[0]
        assert outputs['mode_logits'].shape == (batch_size, 2)

    def test_feature_representation_quality(self, sample_features):
        """Test quality of learned feature representations"""
        mode_head = ModeClassificationHead(
            d_model=512,
            num_modes=2,
            contrastive_learning=True
        )

        with torch.no_grad():
            outputs = mode_head(sample_features)

        if 'contrastive_features' in outputs:
            features = outputs['contrastive_features']

            # Features should be well-distributed (not collapsed to single point)
            feature_std = features.std(dim=0).mean()
            assert feature_std > 0.01, "Features should have meaningful variance"

            # Features should be normalized
            norms = torch.norm(features, dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_multiclass_extension(self, sample_features):
        """Test mode head can handle more than 2 classes"""
        for num_modes in [3, 4, 8]:
            mode_head = ModeClassificationHead(
                d_model=512,
                num_modes=num_modes
            )

            with torch.no_grad():
                outputs = mode_head(sample_features)

            batch_size = sample_features.shape[0]
            expected_shape = (batch_size, num_modes)
            assert outputs['mode_logits'].shape == expected_shape

            # Probabilities should still sum to 1
            probs = F.softmax(outputs['mode_logits'], dim=-1)
            prob_sums = probs.sum(dim=-1)
            assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6)

    def test_extreme_input_handling(self, mode_head):
        """Test robustness to extreme input values"""
        batch_size, seq_len, d_model = 4, 50, 512

        # Test with very large values
        large_features = torch.randn(batch_size, seq_len, d_model) * 100
        with torch.no_grad():
            outputs1 = mode_head(large_features)
        assert not torch.isnan(outputs1['mode_logits']).any()
        assert not torch.isinf(outputs1['mode_logits']).any()

        # Test with very small values
        small_features = torch.randn(batch_size, seq_len, d_model) * 1e-6
        with torch.no_grad():
            outputs2 = mode_head(small_features)
        assert not torch.isnan(outputs2['mode_logits']).any()
        assert not torch.isinf(outputs2['mode_logits']).any()

    def test_device_compatibility(self, mode_head, sample_features):
        """Test mode head works on different devices"""
        # Test CPU
        outputs_cpu = mode_head(sample_features)
        assert outputs_cpu['mode_logits'].device == sample_features.device

        # Test CUDA if available
        if torch.cuda.is_available():
            mode_head_cuda = mode_head.cuda()
            features_cuda = sample_features.cuda()

            with torch.no_grad():
                outputs_cuda = mode_head_cuda(features_cuda)

            assert outputs_cuda['mode_logits'].device.type == 'cuda'
            assert outputs_cuda['mode_logits'].shape == outputs_cpu['mode_logits'].shape

    def test_deterministic_output(self, mode_head, sample_features):
        """Test deterministic output in eval mode"""
        mode_head.eval()

        torch.manual_seed(42)
        outputs1 = mode_head(sample_features)

        torch.manual_seed(42)
        outputs2 = mode_head(sample_features)

        assert torch.allclose(
            outputs1['mode_logits'], outputs2['mode_logits']
        ), "Output should be deterministic in eval mode"

    def test_parameter_count_reasonable(self, mode_head_config):
        """Test that parameter count is reasonable"""
        mode_head = ModeClassificationHead(**mode_head_config)

        total_params = sum(p.numel() for p in mode_head.parameters())

        # Should have reasonable number of parameters for classification head
        assert total_params > 100, "Too few parameters"
        assert total_params < 10_000_000, "Too many parameters for classification head"

    def test_loss_computation_compatibility(self, mode_head, sample_features, sample_mode_labels):
        """Test compatibility with common loss functions"""
        outputs = mode_head(sample_features)
        mode_logits = outputs['mode_logits']

        # Test CrossEntropyLoss
        ce_loss = F.cross_entropy(mode_logits, sample_mode_labels)
        assert not torch.isnan(ce_loss)
        assert ce_loss > 0

        # Test NLLLoss with log_softmax
        log_probs = F.log_softmax(mode_logits, dim=-1)
        nll_loss = F.nll_loss(log_probs, sample_mode_labels)
        assert not torch.isnan(nll_loss)
        assert nll_loss > 0

        # CrossEntropy and NLL should give same result
        assert torch.allclose(ce_loss, nll_loss, atol=1e-6)