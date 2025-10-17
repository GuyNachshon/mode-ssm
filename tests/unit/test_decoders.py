"""
Unit tests for RNNT and CTC decoders.
Tests joint network, predictor network, and CTC head functionality.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unittest.mock import patch

from mode_ssm.models.rnnt_ctc_heads import RNNTDecoder, CTCDecoder


class TestRNNTDecoder:
    """Test cases for RNNTDecoder"""

    @pytest.fixture
    def rnnt_config(self):
        """Standard RNNT decoder configuration"""
        return {
            'vocab_size': 40,
            'd_model': 512,
            'predictor_layers': 2,
            'predictor_hidden_size': 512,
            'joint_hidden_size': 512,
            'dropout': 0.1,
            'beam_size': 4
        }

    @pytest.fixture
    def rnnt_decoder(self, rnnt_config):
        """Create RNNT decoder instance"""
        return RNNTDecoder(**rnnt_config)

    @pytest.fixture
    def sample_encoder_outputs(self):
        """Sample encoder outputs [batch_size, seq_len, d_model]"""
        batch_size, seq_len, d_model = 4, 100, 512
        torch.manual_seed(42)
        return torch.randn(batch_size, seq_len, d_model)

    @pytest.fixture
    def sample_targets(self):
        """Sample target sequences [batch_size, target_len]"""
        batch_size, target_len = 4, 20
        torch.manual_seed(42)
        return torch.randint(1, 40, (batch_size, target_len))  # Avoid blank (0)

    @pytest.fixture
    def sample_target_lengths(self):
        """Sample target lengths [batch_size]"""
        return torch.tensor([15, 18, 12, 20])

    def test_rnnt_decoder_initialization(self, rnnt_config):
        """Test RNNT decoder initialization"""
        decoder = RNNTDecoder(**rnnt_config)

        assert decoder.vocab_size == 40
        assert decoder.d_model == 512
        assert decoder.predictor_layers == 2
        assert hasattr(decoder, 'predictor')
        assert hasattr(decoder, 'joint_network')

    def test_rnnt_invalid_config(self):
        """Test RNNT decoder with invalid configuration"""
        with pytest.raises(ValueError):
            RNNTDecoder(
                vocab_size=0,
                d_model=512
            )

        with pytest.raises(ValueError):
            RNNTDecoder(
                vocab_size=40,
                d_model=0
            )

    def test_predictor_forward(self, rnnt_decoder, sample_targets):
        """Test predictor network forward pass"""
        batch_size, target_len = sample_targets.shape

        with torch.no_grad():
            predictor_output = rnnt_decoder._forward_predictor(sample_targets)

        # Predictor should output [batch_size, target_len, predictor_hidden_size]
        expected_shape = (batch_size, target_len, rnnt_decoder.predictor_hidden_size)
        assert predictor_output.shape == expected_shape

    def test_joint_network_forward(self, rnnt_decoder, sample_encoder_outputs, sample_targets):
        """Test joint network forward pass"""
        batch_size = sample_encoder_outputs.shape[0]
        seq_len = sample_encoder_outputs.shape[1]
        target_len = sample_targets.shape[1]

        with torch.no_grad():
            # Get predictor outputs
            predictor_output = rnnt_decoder._forward_predictor(sample_targets)

            # Forward through joint network
            joint_output = rnnt_decoder._forward_joint(
                sample_encoder_outputs, predictor_output
            )

        # Joint output should be [batch_size, seq_len, target_len, vocab_size]
        expected_shape = (batch_size, seq_len, target_len, rnnt_decoder.vocab_size)
        assert joint_output.shape == expected_shape

    def test_rnnt_full_forward(self, rnnt_decoder, sample_encoder_outputs, sample_targets):
        """Test full RNNT forward pass"""
        with torch.no_grad():
            outputs = rnnt_decoder(sample_encoder_outputs, sample_targets)

        assert isinstance(outputs, dict)
        assert 'rnnt_logits' in outputs

        batch_size = sample_encoder_outputs.shape[0]
        seq_len = sample_encoder_outputs.shape[1]
        target_len = sample_targets.shape[1]
        expected_shape = (batch_size, seq_len, target_len, 40)

        assert outputs['rnnt_logits'].shape == expected_shape

    def test_rnnt_loss_computation(self, rnnt_decoder, sample_encoder_outputs,
                                   sample_targets, sample_target_lengths):
        """Test RNNT loss computation compatibility"""
        outputs = rnnt_decoder(sample_encoder_outputs, sample_targets)
        rnnt_logits = outputs['rnnt_logits']

        # Should be compatible with RNNT loss (though we don't implement the loss here)
        # Just check that logits have proper shape and values
        assert not torch.isnan(rnnt_logits).any()
        assert not torch.isinf(rnnt_logits).any()

        # Log probabilities should be negative
        log_probs = F.log_softmax(rnnt_logits, dim=-1)
        assert (log_probs <= 0).all()

    def test_beam_search_inference(self, rnnt_decoder, sample_encoder_outputs):
        """Test beam search inference"""
        with torch.no_grad():
            # This would test actual beam search implementation
            # For now, just test that we can call inference methods
            outputs = rnnt_decoder.inference(sample_encoder_outputs)

        assert isinstance(outputs, dict)
        # Implementation would include beam search results

    def test_gradient_flow(self, rnnt_decoder, sample_encoder_outputs, sample_targets):
        """Test gradient flow through RNNT decoder"""
        sample_encoder_outputs.requires_grad_(True)

        outputs = rnnt_decoder(sample_encoder_outputs, sample_targets)
        loss = outputs['rnnt_logits'].sum()
        loss.backward()

        # Encoder outputs should have gradients
        assert sample_encoder_outputs.grad is not None
        assert not torch.isnan(sample_encoder_outputs.grad).any()

        # Model parameters should have gradients
        for param in rnnt_decoder.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()

    def test_different_sequence_lengths(self, rnnt_decoder):
        """Test RNNT decoder with different sequence lengths"""
        batch_size, d_model = 2, 512

        for seq_len in [50, 100, 200]:
            for target_len in [10, 20, 30]:
                encoder_outputs = torch.randn(batch_size, seq_len, d_model)
                targets = torch.randint(1, 40, (batch_size, target_len))

                with torch.no_grad():
                    outputs = rnnt_decoder(encoder_outputs, targets)

                expected_shape = (batch_size, seq_len, target_len, 40)
                assert outputs['rnnt_logits'].shape == expected_shape


class TestCTCDecoder:
    """Test cases for CTCDecoder"""

    @pytest.fixture
    def ctc_config(self):
        """Standard CTC decoder configuration"""
        return {
            'vocab_size': 40,
            'd_model': 512,
            'dropout': 0.1
        }

    @pytest.fixture
    def ctc_decoder(self, ctc_config):
        """Create CTC decoder instance"""
        return CTCDecoder(**ctc_config)

    @pytest.fixture
    def sample_encoder_outputs(self):
        """Sample encoder outputs [batch_size, seq_len, d_model]"""
        batch_size, seq_len, d_model = 4, 100, 512
        torch.manual_seed(42)
        return torch.randn(batch_size, seq_len, d_model)

    @pytest.fixture
    def sample_targets(self):
        """Sample target sequences [batch_size, target_len]"""
        batch_size, target_len = 4, 20
        torch.manual_seed(42)
        return torch.randint(1, 40, (batch_size, target_len))

    def test_ctc_decoder_initialization(self, ctc_config):
        """Test CTC decoder initialization"""
        decoder = CTCDecoder(**ctc_config)

        assert decoder.vocab_size == 40
        assert decoder.d_model == 512
        assert hasattr(decoder, 'classifier')

    def test_ctc_invalid_config(self):
        """Test CTC decoder with invalid configuration"""
        with pytest.raises(ValueError):
            CTCDecoder(
                vocab_size=0,
                d_model=512
            )

        with pytest.raises(ValueError):
            CTCDecoder(
                vocab_size=40,
                d_model=0
            )

    def test_ctc_forward_pass(self, ctc_decoder, sample_encoder_outputs):
        """Test CTC decoder forward pass"""
        batch_size, seq_len, d_model = sample_encoder_outputs.shape

        with torch.no_grad():
            outputs = ctc_decoder(sample_encoder_outputs)

        assert isinstance(outputs, dict)
        assert 'ctc_logits' in outputs

        # CTC logits should be [batch_size, seq_len, vocab_size]
        expected_shape = (batch_size, seq_len, 40)
        assert outputs['ctc_logits'].shape == expected_shape

    def test_ctc_log_probabilities(self, ctc_decoder, sample_encoder_outputs):
        """Test CTC log probability computation"""
        with torch.no_grad():
            outputs = ctc_decoder(sample_encoder_outputs)

        ctc_logits = outputs['ctc_logits']

        # Convert to log probabilities
        log_probs = F.log_softmax(ctc_logits, dim=-1)

        # Log probabilities should sum to 1 in probability space
        probs = torch.exp(log_probs)
        prob_sums = probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6)

        # Log probabilities should be negative
        assert (log_probs <= 0).all()

    def test_ctc_loss_compatibility(self, ctc_decoder, sample_encoder_outputs, sample_targets):
        """Test CTC loss computation compatibility"""
        outputs = ctc_decoder(sample_encoder_outputs)
        ctc_logits = outputs['ctc_logits']

        # Test compatibility with CTC loss
        log_probs = F.log_softmax(ctc_logits, dim=-1)

        # Should be able to transpose for CTC loss (T, N, V)
        log_probs_transposed = log_probs.transpose(0, 1)

        batch_size, seq_len = sample_encoder_outputs.shape[:2]
        target_len = sample_targets.shape[1]

        assert log_probs_transposed.shape == (seq_len, batch_size, 40)
        assert not torch.isnan(log_probs).any()
        assert not torch.isinf(log_probs).any()

    def test_ctc_greedy_decoding(self, ctc_decoder, sample_encoder_outputs):
        """Test CTC greedy decoding"""
        with torch.no_grad():
            outputs = ctc_decoder(sample_encoder_outputs)

        ctc_logits = outputs['ctc_logits']

        # Greedy decoding: take argmax
        predictions = torch.argmax(ctc_logits, dim=-1)

        batch_size, seq_len = sample_encoder_outputs.shape[:2]
        assert predictions.shape == (batch_size, seq_len)

        # Predictions should be valid vocabulary indices
        assert (predictions >= 0).all()
        assert (predictions < 40).all()

    def test_ctc_gradient_flow(self, ctc_decoder, sample_encoder_outputs):
        """Test gradient flow through CTC decoder"""
        sample_encoder_outputs.requires_grad_(True)

        outputs = ctc_decoder(sample_encoder_outputs)
        loss = outputs['ctc_logits'].sum()
        loss.backward()

        # Encoder outputs should have gradients
        assert sample_encoder_outputs.grad is not None
        assert not torch.isnan(sample_encoder_outputs.grad).any()

        # Model parameters should have gradients
        for param in ctc_decoder.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()

    def test_ctc_different_sequence_lengths(self, ctc_decoder):
        """Test CTC decoder with different sequence lengths"""
        batch_size, d_model = 2, 512

        for seq_len in [50, 100, 200, 500]:
            encoder_outputs = torch.randn(batch_size, seq_len, d_model)

            with torch.no_grad():
                outputs = ctc_decoder(encoder_outputs)

            expected_shape = (batch_size, seq_len, 40)
            assert outputs['ctc_logits'].shape == expected_shape

    def test_ctc_batch_size_invariance(self, ctc_decoder):
        """Test CTC decoder with different batch sizes"""
        seq_len, d_model = 100, 512

        for batch_size in [1, 4, 8, 16]:
            encoder_outputs = torch.randn(batch_size, seq_len, d_model)

            with torch.no_grad():
                outputs = ctc_decoder(encoder_outputs)

            expected_shape = (batch_size, seq_len, 40)
            assert outputs['ctc_logits'].shape == expected_shape

    def test_dropout_behavior(self, sample_encoder_outputs):
        """Test dropout behavior in train vs eval modes"""
        ctc_decoder = CTCDecoder(
            vocab_size=40,
            d_model=512,
            dropout=0.5
        )

        # In eval mode, outputs should be deterministic
        ctc_decoder.eval()
        torch.manual_seed(42)
        outputs1 = ctc_decoder(sample_encoder_outputs)
        torch.manual_seed(42)
        outputs2 = ctc_decoder(sample_encoder_outputs)

        assert torch.allclose(
            outputs1['ctc_logits'], outputs2['ctc_logits']
        ), "Eval mode should be deterministic"

    def test_extreme_input_handling(self, ctc_decoder):
        """Test robustness to extreme input values"""
        batch_size, seq_len, d_model = 4, 50, 512

        # Test with very large values
        large_inputs = torch.randn(batch_size, seq_len, d_model) * 100
        with torch.no_grad():
            outputs1 = ctc_decoder(large_inputs)
        assert not torch.isnan(outputs1['ctc_logits']).any()
        assert not torch.isinf(outputs1['ctc_logits']).any()

        # Test with very small values
        small_inputs = torch.randn(batch_size, seq_len, d_model) * 1e-6
        with torch.no_grad():
            outputs2 = ctc_decoder(small_inputs)
        assert not torch.isnan(outputs2['ctc_logits']).any()
        assert not torch.isinf(outputs2['ctc_logits']).any()

    def test_device_compatibility(self, ctc_decoder, sample_encoder_outputs):
        """Test decoder works on different devices"""
        # Test CPU
        outputs_cpu = ctc_decoder(sample_encoder_outputs)
        assert outputs_cpu['ctc_logits'].device == sample_encoder_outputs.device

        # Test CUDA if available
        if torch.cuda.is_available():
            ctc_decoder_cuda = ctc_decoder.cuda()
            inputs_cuda = sample_encoder_outputs.cuda()

            with torch.no_grad():
                outputs_cuda = ctc_decoder_cuda(inputs_cuda)

            assert outputs_cuda['ctc_logits'].device.type == 'cuda'
            assert outputs_cuda['ctc_logits'].shape == outputs_cpu['ctc_logits'].shape

    def test_parameter_count_reasonable(self, ctc_config):
        """Test parameter count is reasonable"""
        ctc_decoder = CTCDecoder(**ctc_config)

        total_params = sum(p.numel() for p in ctc_decoder.parameters())

        # Should have reasonable number of parameters
        # Mainly just the classifier layer: d_model * vocab_size + bias
        expected_params = 512 * 40 + 40  # weights + bias
        assert abs(total_params - expected_params) < 1000  # Allow some flexibility

    def test_output_distribution_properties(self, ctc_decoder, sample_encoder_outputs):
        """Test properties of output distributions"""
        with torch.no_grad():
            outputs = ctc_decoder(sample_encoder_outputs)

        ctc_logits = outputs['ctc_logits']

        # Convert to probabilities
        probs = F.softmax(ctc_logits, dim=-1)

        # Check that distributions have reasonable entropy
        # (not collapsed to single token)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        mean_entropy = entropy.mean()

        # Entropy should be positive (not deterministic)
        assert mean_entropy > 0.1, f"Mean entropy too low: {mean_entropy}"

        # But also not completely uniform (would be log(vocab_size) ≈ 3.69)
        assert mean_entropy < np.log(40), f"Mean entropy too high: {mean_entropy}"